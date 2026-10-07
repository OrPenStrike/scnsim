"""Python-owned ordered objectives, continuation, caches and explicit CMA state.

Numerical adapters own local Newton loops. Every candidate starts continuation
from the committed baseline; no backend owns the optimizer or durable state.
"""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import json
import math
import struct
from importlib.metadata import version
from typing import Callable
from time import perf_counter_ns

import numpy as np

from ..canonical import canonical_json_bytes, float64_from_hex
from .. import errors
from ..errors import CompilerInvariantError, RuntimePreparationError, SCNSimError, SCNSimValidationError, InvalidCandidatePhysicalParameter
from .compiler import compile_model, parameter_key, parameter_values
from .mesh import quantity
from .models import EvaluationJob, EvaluationResult, MeshSpec, NumericalBackend, NumericalFailure
from .prepared import array_from_record, array_record, record_bytes, record_document
from .views import realize_view

CANDIDATE_FAILURES = frozenset(("invalid_candidate_physical_parameter", "eliminated_block_solve_failure",
                                "root_slope_unresolved", "numerical_resolution_unresolved"))


def bits(value: float) -> str:
    """Numeric evidence may include infinity; unlike parameter identity encoding."""
    return struct.pack(">d", float(value)).hex()


def unbits(value: str) -> float:
    return struct.unpack(">d", bytes.fromhex(value))[0]


def complex_record(value: complex) -> dict:
    return {"real_f64": bits(value.real), "imag_f64": bits(value.imag)}


def complex_value(record: dict) -> complex:
    return complex(float64_from_hex(record["real_f64"]), float64_from_hex(record["imag_f64"]))


def leaves(expression: dict) -> list[dict]:
    kind = expression["type"]
    if kind == "quantity_sum":
        return [leaf for term in expression["terms"] for leaf in leaves(term)]
    if kind == "quantity_difference":
        return leaves(expression["left"]) + leaves(expression["right"])
    if kind == "quantity_absolute":
        return leaves(expression["operand"])
    if kind not in ("diagonal_root_projection", "response_element_projection"):
        raise NotImplementedError(f"benchmark quantity {kind!r} is unsupported")
    return [expression]


def expression_value(expression: dict, values: list[float]) -> float:
    iterator = iter(values)

    def evaluate(node: dict) -> float:
        kind = node["type"]
        if kind == "quantity_sum":
            return sum(evaluate(term) for term in node["terms"])
        if kind == "quantity_difference":
            return evaluate(node["left"]) - evaluate(node["right"])
        if kind == "quantity_absolute":
            return abs(evaluate(node["operand"]))
        return next(iterator)

    return evaluate(expression)


class EvaluationFailure(Exception):
    def __init__(self, failure: NumericalFailure, *, result: EvaluationResult | None = None):
        self.failure = failure
        self.result = result
        super().__init__(failure.detail)


def numerical_error(failure: NumericalFailure):
    """Restore the existing public failure type from a numerical handoff."""
    classes = (errors.DirectResponseFormationError, errors.PortRealizabilityError,
               errors.EliminatedBlockSolveFailure, errors.RootSlopeUnresolved,
               errors.NumericalResolutionUnresolved, errors.InvalidCandidatePhysicalParameter,
               errors.CompilerInvariantError, errors.UnsupportedSingularCapacitanceForDiagonalRootV1)
    error_type = {cls.kind: cls for cls in classes}[failure.kind]
    return error_type(failure.detail, stage=failure.stage, evidence=record_document(failure.evidence_bytes))


def checked_results(backend: NumericalBackend, jobs: tuple[EvaluationJob, ...]) -> tuple[EvaluationResult, ...]:
    results = backend.evaluate_batch(jobs)
    if len(results) != len(jobs) or any(job.id != result.id for job, result in zip(jobs, results)):
        raise CompilerInvariantError("numerical batch changed job identity/order", stage="benchmark_protocol")
    return results


def continue_diagonal_root(*, baseline: dict, values: dict, baseline_root: complex,
                           endpoint_view, initial_result: EvaluationResult,
                           candidate_view, make_job, evaluate, identity: str, observe=None):
    """One baseline-anchored dyadic authority for standalone and CMA roots."""
    def advance(left_t, left_root, right_t, depth, result=None):
        if right_t == 1.0:
            point, view = values, endpoint_view
        else:
            point = {}
            for parameter, base in baseline.items():
                candidate = values[parameter]
                if isinstance(base, dict):
                    if canonical_json_bytes(base) != canonical_json_bytes(candidate):
                        raise CompilerInvariantError("root continuation cannot interpolate RLGC", stage="root_continuation")
                    point[parameter] = base
                else:
                    point[parameter] = base + right_t * (candidate - base)
            view = candidate_view(point)
        if result is None:
            result = evaluate((make_job(f"continuation:{identity}:{right_t.hex()}", view, left_root),))[0]
        if observe is not None:
            observe(point, view, result, right_t)
        if result.failure is None:
            return result
        if result.failure.kind != "numerical_resolution_unresolved" or depth >= 32:
            raise EvaluationFailure(result.failure, result=result)
        midpoint = (left_t + right_t) / 2
        middle = advance(left_t, left_root, midpoint, depth + 1)
        return advance(midpoint, middle.root_omega_rad_s, right_t, depth + 1)

    return advance(0.0, baseline_root, 1.0, 0, initial_result)


class Evaluator:
    """One task's baseline anchor and physical-parameter cache authority."""

    def __init__(self, plan: dict, analysis: dict, mesh: MeshSpec, backend: NumericalBackend, *, emit=None):
        self.plan, self.analysis, self.mesh, self.backend = plan, analysis, mesh, backend
        self.spec = analysis["spec"]
        self.emit = emit
        self.preparation_cache = {}
        source = analysis["parameter_source"]
        point = source["parameters"] if source["kind"] == "point" else source["baseline_parameters"] if source["kind"] == "points" else source["base_parameters"]
        self.templates = {parameter_key(binding["parameter"]): binding for binding in point["bindings"]}
        self.base = parameter_values(point)
        self.authorized = {parameter_key(record) for record in self.spec.get("allow_extrapolation", point.get("allow_extrapolation", []))}
        self.anchors: dict[str, complex] = {}
        self.cache: dict[str, dict] = {}

    def cohort_values(self, point: dict) -> dict:
        """Apply declared overrides to the task's effective baseline closure."""
        values = self.base.copy()
        for binding in point["bindings"]:
            key = parameter_key(binding["parameter"])
            if key not in self.templates:
                raise SCNSimValidationError("cohort parameter is outside the captured Plan", stage="benchmark_prepare")
            expected, supplied = self.templates[key]["value"], binding["value"]
            if supplied["type"] != expected["type"] or (supplied["type"] == "quantity_f64" and
                    (supplied["si_unit"], supplied["dimensionality"]) != (expected["si_unit"], expected["dimensionality"])):
                raise SCNSimValidationError("cohort parameter conflicts with its captured type/unit", stage="benchmark_prepare")
            values[key] = supplied if supplied["type"] == "rlgc" else quantity(supplied)
        return values

    def timed_results(self, jobs):
        start = perf_counter_ns()
        try:
            return checked_results(self.backend, jobs)
        finally:
            if self.emit is not None:
                self.emit("timing", {"stage": "numerical_evaluation", "start_tick_ns": start,
                                     "end_tick_ns": perf_counter_ns(), "counts": {"jobs": len(jobs)}})

    def parameter_record(self, values: dict) -> dict:
        bindings = []
        for key, template in self.templates.items():
            value = values[key]
            encoded = value if isinstance(value, dict) else dict(template["value"], si_value_f64=bits(value))
            bindings.append({"parameter": template["parameter"], "value": encoded})
        return {"type": "parameter_set_v2", "bindings": bindings, "allow_extrapolation": []}

    def root_key(self, selector: dict) -> str:
        return canonical_json_bytes({"spec": selector["spec"], "view": selector["view"], "type": "diagonal_root"}).decode()

    def candidate_view(self, values: dict, declaration: dict):
        start = perf_counter_ns()
        try:
            return realize_view(compile_model(self.plan, values, mesh=self.mesh, authorized=self.authorized,
                                             preparation_cache=self.preparation_cache), declaration)
        finally:
            if self.emit is not None:
                self.emit("timing", {"stage": "continuation_lowering_and_view", "start_tick_ns": start,
                                     "end_tick_ns": perf_counter_ns(), "counts": {"points": 1}})

    def root_job(self, identity: str, selector: dict, view, start: complex | None) -> EvaluationJob:
        spec = selector["spec"]
        hint = quantity(spec["root_hint"])
        return EvaluationJob(identity, "diagonal_root", view, coordinate_index=view.terminal_ids.index(spec["coordinate"]),
                             root_hint_hz=hint, omega_start_rad_s=start if start is not None else complex(2 * np.pi * hint))

    def continued_root(self, selector: dict, values: dict, endpoint_view, initial_result: EvaluationResult) -> EvaluationResult:
        """Accepted dyadic repair only for numerical-resolution failure."""
        key = self.root_key(selector)

        try:
            return continue_diagonal_root(
                baseline=self.base, values=values, baseline_root=self.anchors[key],
                endpoint_view=endpoint_view, initial_result=initial_result,
                candidate_view=lambda point: self.candidate_view(point, selector["view"]),
                make_job=lambda identity, view, start: self.root_job(identity, selector, view, start),
                evaluate=self.timed_results, identity=key,
            )
        except EvaluationFailure as error:
            if error.failure.kind != "direct_response_formation":
                raise
            # Only this nonbaseline CMA boundary owns a formation penalty.
            # Conversion follows the failed local solve, without dyadic retry.
            failure = NumericalFailure("numerical_resolution_unresolved", error.failure.stage,
                                       "candidate selected response is numerically unresolved",
                                       error.failure.evidence_bytes)
            raise EvaluationFailure(failure, result=error.result) from error

    def evaluate_many(self, points: list[dict], *, baseline: bool = False) -> list[dict]:
        """Batch compatible leaves, restoring candidate and leaf order afterwards."""
        objectives = self.spec["objectives"]
        states, keys, representatives = [], [], {}
        for values in points:
            record = self.parameter_record(values)
            key = canonical_json_bytes(record).decode()
            keys.append(key)
            if key in self.cache or key in representatives:
                states.append(None)
                continue
            representatives[key] = len(states)
            state = {"parameters": record, "objectives": [], "cost_f64": bits(0), "failure": None,
                     "roots": {}, "views": {}, "values": values}
            start = perf_counter_ns()
            try:
                raw = compile_model(self.plan, values, mesh=self.mesh, authorized=self.authorized, preparation_cache=self.preparation_cache)
                state["discretization"] = json.loads(raw.evidence_bytes)["discretization"]
                for objective in objectives:
                    for leaf in leaves(objective["quantity"]):
                        declaration_key = canonical_json_bytes(leaf["view"]).decode()
                        if declaration_key not in state["views"]:
                            state["views"][declaration_key] = realize_view(raw, leaf["view"])
            except SCNSimError as error:
                if baseline or error.kind not in CANDIDATE_FAILURES:
                    raise
                state["failure"] = {"kind": error.kind, "stage": error.stage, "detail": str(error),
                                    "evidence": json.loads(record_bytes(error.evidence)), "phase": "candidate_compile"}
                state["cost_f64"] = bits(math.inf)
            finally:
                if self.emit is not None:
                    self.emit("timing", {"stage": "lowering_and_view", "start_tick_ns": start,
                                         "end_tick_ns": perf_counter_ns(), "counts": {"candidate": len(states)}})
            states.append(state)
        for objective_index, objective in enumerate(objectives):
            selectors = leaves(objective["quantity"])
            term_values = [[] for _ in states]
            term_records = [[] for _ in states]
            for term_index, selector in enumerate(selectors):
                jobs, active = [], []
                for index, state in enumerate(states):
                    if state is None or state["failure"]:
                        continue
                    view = state["views"][canonical_json_bytes(selector["view"]).decode()]
                    dependency = self.root_key(selector) if selector["type"] == "diagonal_root_projection" else canonical_json_bytes({"spec": selector["spec"], "view": selector["view"], "type": "response_element"}).decode()
                    if dependency in state["roots"]:
                        result = state["roots"][dependency]
                        active.append((index, view, dependency, result))
                        continue
                    identity = f"candidate:{index}:objective:{objective_index}:term:{term_index}"
                    if selector["type"] == "diagonal_root_projection":
                        jobs.append(self.root_job(identity, selector, view, None if baseline else self.anchors[dependency]))
                    else:
                        spec = selector["spec"]
                        jobs.append(EvaluationJob(identity, "response_element", view, frequencies_hz=np.array([quantity(spec["frequency"])]),
                                                  family=spec["family"], input_index=view.terminal_ids.index(spec["input_coordinate"]),
                                                  output_index=view.terminal_ids.index(spec["output_coordinate"])))
                    active.append((index, view, dependency, None))
                new_results = iter(self.timed_results(tuple(jobs))) if jobs else iter(())
                for index, view, dependency, result in active:
                    state = states[index]
                    if result is None:
                        result = next(new_results)
                    try:
                        if selector["type"] == "diagonal_root_projection":
                            if not baseline and result.failure:
                                result = self.continued_root(selector, state["values"], view, result)
                            if result.failure:
                                raise EvaluationFailure(result.failure)
                            omega = result.root_omega_rad_s
                            state["roots"][dependency] = result
                            if baseline:
                                self.anchors[dependency] = omega
                            if omega.imag > 0:
                                raise EvaluationFailure(NumericalFailure("numerical_resolution_unresolved", "newton_certificate", "diagonal root violates passive imaginary-root policy"))
                            value = omega.real / (2 * np.pi) if selector["projection"] == "frequency" else -omega.imag / np.pi
                            actual = complex_record(omega)
                        else:
                            if result.failure:
                                failure = result.failure
                                if not baseline and failure.kind == "direct_response_formation":
                                    failure = NumericalFailure("numerical_resolution_unresolved", failure.stage, failure.detail, failure.evidence_bytes)
                                raise EvaluationFailure(failure)
                            state["roots"][dependency] = result
                            z = result.response_value
                            value = {"real": lambda: z.real, "imag": lambda: z.imag, "magnitude": lambda: abs(z)}[selector["projection"]]()
                            actual = complex_record(z)
                        term_values[index].append(float(value))
                        term_records[index].append({"term_ordinal": term_index, "selector": selector, "status": "success",
                                                    "value_f64": bits(value), "actual_complex": actual,
                                                    "lineage": json.loads(view.lineage_bytes), "evidence": json.loads(result.evidence_bytes)})
                    except (EvaluationFailure, SCNSimError) as error:
                        failure = error.failure if isinstance(error, EvaluationFailure) else NumericalFailure(error.kind, error.stage, str(error), record_bytes(error.evidence))
                        if baseline or failure.kind not in CANDIDATE_FAILURES:
                            if isinstance(error, EvaluationFailure):
                                raise numerical_error(failure) from error
                            raise
                        state["failure"] = dict(asdict(failure), evidence_bytes=json.loads(failure.evidence_bytes),
                                                phase="quantity_evaluation", objective_ordinal=objective_index, term_ordinal=term_index)
                        state["cost_f64"] = bits(math.inf)
                        term_records[index].append({"term_ordinal": term_index, "selector": selector, "status": "failure", "failure": state["failure"]})
            for index, state in enumerate(states):
                if state is None:
                    continue
                if state["failure"]:
                    completed = len(term_records[index])
                    term_records[index].extend({"term_ordinal": i, "selector": selectors[i], "status": "unevaluated"} for i in range(completed, len(selectors)))
                    state["objectives"].append({"id": objective["id"], "status": "failure" if completed else "unevaluated", "terms": term_records[index]})
                    continue
                value = expression_value(objective["quantity"], term_values[index])
                residual = (value - quantity(objective["target"])) / quantity(objective["resolved_scale"])
                if objective["comparison"] == "at_least":
                    residual = max(0, -residual)
                contribution = float64_from_hex(objective["weight_f64"]) * (residual * residual)
                # Preserve the existing objective-specific numerical failure
                # owner; this is not an experiment-wide finiteness screen.
                if not math.isfinite(value) or not math.isfinite(residual) or not math.isfinite(contribution):
                    failure = NumericalFailure("numerical_resolution_unresolved", "objective", "candidate objective is non-finite")
                    if baseline:
                        raise numerical_error(failure)
                    state["failure"] = {"kind": failure.kind, "stage": failure.stage, "detail": failure.detail,
                                        "phase": "objective_aggregation", "objective_ordinal": objective_index}
                    state["cost_f64"] = bits(math.inf)
                    state["objectives"].append({"id": objective["id"], "status": "failure", "terms": term_records[index], "failure": state["failure"]})
                    continue
                cost = unbits(state["cost_f64"]) + contribution
                state["cost_f64"] = bits(cost)
                state["objectives"].append({"id": objective["id"], "status": "success", "value_f64": bits(value),
                                            "cost_f64": bits(contribution), "terms": term_records[index]})
        for key, index in representatives.items():
            state = states[index]
            if not state["failure"] and not math.isfinite(unbits(state["cost_f64"])):
                failure = NumericalFailure("numerical_resolution_unresolved", "objective", "candidate total cost is non-finite")
                if baseline:
                    raise numerical_error(failure)
                state["failure"] = {"kind": failure.kind, "stage": failure.stage, "detail": failure.detail,
                                    "phase": "total_aggregation"}
                state["cost_f64"] = bits(math.inf)
            self.cache[key] = {k: v for k, v in state.items() if k not in ("values", "views", "roots")}
        result, seen = [], set()
        for key in keys:
            result.append(dict(deepcopy(self.cache[key]), candidate_key=key,
                               cache_hit=key not in representatives or key in seen))
            seen.add(key)
        return result


def linquad(x: float, lower: float, upper: float) -> float:
    al = 1 if not math.isfinite(lower) else min((upper - lower) / 2, (1 + abs(lower)) / 20)
    au = 1 if not math.isfinite(upper) else min((upper - lower) / 2, (1 + abs(upper)) / 20)
    if x < lower - 2 * al - (upper - lower) / 2 or x > upper + 2 * au + (upper - lower) / 2:
        period = 2 * (upper - lower + al + au)
        x -= period * math.floor((x - (lower - 2 * al - (upper - lower) / 2)) / period)
    if x > upper + au:
        x -= 2 * (x - upper - au)
    if x < lower - al:
        x += 2 * (lower - al - x)
    if x < lower + al:
        return lower + (x - (lower - al)) ** 2 / (4 * al)
    if x < upper - au:
        return x
    return upper - (x - (upper + au)) ** 2 / (4 * au)


def domain_bounds(variable: dict, base: dict) -> tuple[float, float]:
    domain = variable.get("domain")
    if domain == "NONNEGATIVE":
        return -base[parameter_key(variable["parameter"])] / quantity(variable["scale"]), math.inf
    if domain == "NONPOSITIVE":
        return -math.inf, -base[parameter_key(variable["parameter"])] / quantity(variable["scale"])
    return (-math.inf, math.inf) if domain is not None else (0, 1)


def latent_initial(spec: dict, base: dict) -> np.ndarray:
    result = []
    for variable, encoded in zip(spec["variables"], spec["optimizer"]["baseline_optimizer_coordinates_f64"]):
        y = float64_from_hex(encoded)
        if variable.get("domain") not in ("UNBOUNDED", "POSITIVE", "NEGATIVE"):
            lower, upper = domain_bounds(variable, base)
            al = 1 if not math.isfinite(lower) else min((upper - lower) / 2, (1 + abs(lower)) / 20)
            au = 1 if not math.isfinite(upper) else min((upper - lower) / 2, (1 + abs(upper)) / 20)
            y = lower - al + 2 * math.sqrt(al * (y - lower)) if y < lower + al else y if y < upper - au else upper + au - 2 * math.sqrt(au * (upper - y))
        result.append(y)
    return np.array(result)


def mapped_values(spec: dict, base: dict, latent: np.ndarray) -> dict:
    values = base.copy()
    for variable, x in zip(spec["variables"], latent):
        key = parameter_key(variable["parameter"])
        domain = variable.get("domain")
        if not math.isfinite(float(x)):
            raise InvalidCandidatePhysicalParameter("CMA candidate coordinate is non-finite", stage="unit_map")
        if domain is None:
            z = linquad(float(x), 0, 1)
            low, high = quantity(variable["lower"]), quantity(variable["upper"])
            values[key] = low + z * (high - low) if variable["transform"] == "linear" else low * (high / low) ** z
        elif domain in ("POSITIVE", "NEGATIVE"):
            with np.errstate(over="ignore", under="ignore"):
                values[key] = base[key] * float(np.exp(float(x)))
        else:
            lower, upper = domain_bounds(variable, base)
            coordinate = linquad(float(x), lower, upper) if domain != "UNBOUNDED" else float(x)
            values[key] = base[key] + quantity(variable["scale"]) * coordinate
        value = values[key]
        if (not math.isfinite(value) or
                (domain in ("POSITIVE", "NEGATIVE") and (value == 0 or math.copysign(1, value) != math.copysign(1, base[key]))) or
                (domain == "NONNEGATIVE" and value < 0) or (domain == "NONPOSITIVE" and value > 0)):
            raise InvalidCandidatePhysicalParameter("candidate mapping left its physical domain", stage="unit_map")
    return values


CMA_FIELDS = (
    "_n_dim", "_popsize", "_mu", "_mu_eff", "_cc", "_c1", "_cmu", "_c_sigma", "_d_sigma", "_cm", "_chi_n",
    "_weights", "_p_sigma", "_pc", "_mean", "_C", "_sigma", "_D", "_B", "_bounds", "_n_max_resampling", "_g",
    "_lr_adapt", "_alpha", "_beta_mean", "_beta_Sigma", "_gamma", "_Emean", "_ESigma", "_Vmean", "_VSigma",
    "_eta_mean", "_eta_Sigma", "_tolx", "_tolxup", "_tolfun", "_tolconditioncov", "_funhist_term", "_funhist_values",
)


def cma_snapshot(optimizer) -> dict:
    state = {}
    for name in CMA_FIELDS:
        value = getattr(optimizer, name)
        state[name] = {"array": array_record(value)} if isinstance(value, np.ndarray) else {"float_f64": bits(value)} if isinstance(value, (float, np.floating)) else value
    algorithm, keys, position, has_gaussian, cached = optimizer._rng.get_state()
    return {"schema": "scnsim.python_cma_state", "schema_version": 1, "fields": state,
            "rng": {"algorithm": algorithm, "keys": keys.tolist(), "position": position,
                    "has_gaussian": has_gaussian, "cached_gaussian_f64": bits(cached)}}


def restore_cma(optimizer, record: dict) -> None:
    if set(record["fields"]) != set(CMA_FIELDS) or record["rng"]["algorithm"] != "MT19937":
        raise CompilerInvariantError("checkpoint optimizer state layout differs", stage="benchmark_checkpoint")
    for name, value in record["fields"].items():
        value = array_from_record(value["array"]).copy() if isinstance(value, dict) and "array" in value else unbits(value["float_f64"]) if isinstance(value, dict) else value
        setattr(optimizer, name, value)
    rng = record["rng"]
    optimizer._rng.set_state(("MT19937", np.asarray(rng["keys"], dtype=np.uint32), rng["position"],
                              rng["has_gaussian"], float64_from_hex(rng["cached_gaussian_f64"])))


def optimize(evaluator: Evaluator, emit: Callable, checkpoint: dict | None = None, *,
             checkpoint_policy: str = "generation") -> dict:
    from cmaes import CMA

    if version("cmaes") != "0.13.1":
        raise RuntimePreparationError("benchmark CMA state requires cmaes==0.13.1", stage="benchmark_prepare")

    controls = evaluator.spec["optimizer"]
    optimizer = CMA(mean=latent_initial(evaluator.spec, evaluator.base), sigma=float64_from_hex(controls["initial_sigma_f64"]),
                    population_size=controls["resolved_population_size"], bounds=None, seed=0)
    seed = controls["seed"] & ((1 << 64) - 1)
    optimizer._rng.seed(np.array([seed & 0xffffffff, seed >> 32], dtype=np.uint32))
    # This dependency allocates history with empty(); initialize unused slots so
    # numeric checkpoint bytes never expose unrelated allocator memory.
    optimizer._funhist_values.fill(0)
    if checkpoint:
        restore_cma(optimizer, checkpoint["cma"])
        evaluator.anchors = {key: complex_value(value) for key, value in checkpoint["anchors"].items()}
        evaluator.cache = checkpoint["cache"]
        baseline, best = checkpoint["baseline"], checkpoint["best"]
        next_ordinal = checkpoint["next_ordinal"]
    else:
        baseline = dict(evaluator.evaluate_many([evaluator.base], baseline=True)[0], evaluation_ordinal=0, generation=0, population_column=0)
        best, next_ordinal = baseline, 1

    def resume_state() -> dict | None:
        # Anchors have one durable owner: the acknowledged baseline block.
        return {"cma": cma_snapshot(optimizer)} if checkpoint_policy == "generation" else None

    if checkpoint is None:
        emit("baseline_ready", {
            "baseline": baseline,
            "anchors": {key: complex_record(value) for key, value in evaluator.anchors.items()},
            "resume_state": resume_state(),
        })
    for generation in range(optimizer.generation + 1, controls["complete_generations"] + 1):
        start = perf_counter_ns()
        raw = [optimizer.ask() for _ in range(optimizer.population_size)]
        emit("timing", {"stage": "cma_ask", "start_tick_ns": start, "end_tick_ns": perf_counter_ns(),
                        "counts": {"population": len(raw)}})
        emit("population_observed", {"generation": generation, "latent": array_record(np.stack(raw))})
        records = evaluator.evaluate_many([mapped_values(evaluator.spec, evaluator.base, value) for value in raw])
        costs = []
        for column, record in enumerate(records):
            record.update(evaluation_ordinal=next_ordinal, generation=generation, population_column=column,
                          latent_coordinates=array_record(raw[column]))
            next_ordinal += 1
            cost = unbits(record["cost_f64"])
            costs.append(cost)
            if math.isfinite(cost) and cost < unbits(best["cost_f64"]):
                best = record
            emit("evaluation", record)
        start = perf_counter_ns()
        optimizer.tell([(value.copy(), cost) for value, cost in zip(raw, costs)])
        emit("timing", {"stage": "cma_tell", "start_tick_ns": start, "end_tick_ns": perf_counter_ns(),
                        "counts": {"population": len(raw)}})
        emit("generation_ready", {"generation": optimizer.generation, "next_ordinal": next_ordinal,
                                  "best_ordinal": best["evaluation_ordinal"],
                                  "resume_state": resume_state()})
    return {"type": "optimization", "best_ordinal": best["evaluation_ordinal"],
            "completed_generations": optimizer.generation, "unused_evaluations": controls["unused_evaluations"],
            "algorithm_id": "scnsim.python_cmaes_0.13.1.ask_tell.v1"}
