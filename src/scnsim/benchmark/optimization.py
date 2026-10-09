"""Python-owned ordered objectives, continuation, caches and explicit CMA state.

Numerical adapters own local Newton loops. Every candidate starts continuation
from the committed baseline; no backend owns the optimizer or durable state.
"""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from contextlib import nullcontext
from hashlib import sha256
import json
import math
import struct
from importlib.metadata import version
from threading import Lock
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
from ..execution.quantities import (
    EvaluationFailure,
    QuantityEvaluator,
    expression_leaves as leaves,
    expression_value,
    result_from_record,
)
from .views import realize_view

CANDIDATE_FAILURES = frozenset(("invalid_candidate_physical_parameter", "eliminated_block_solve_failure",
                                "root_slope_unresolved", "numerical_resolution_unresolved"))


def _candidate_actor_error(record: dict):
    """Restore an actual SCNSim actor error without reclassifying its owner."""
    cls = getattr(errors, record.get("error_type", ""), None)
    if not isinstance(cls, type) or not issubclass(cls, SCNSimError):
        raise CompilerInvariantError("candidate actor returned an unknown error type", stage="candidate_protocol")
    return cls(record["detail"], stage=record["stage"], evidence=record.get("evidence", {}))


def bits(value: float) -> str:
    """Numeric evidence may include infinity; unlike parameter identity encoding."""
    return struct.pack(">d", float(value)).hex()


def unbits(value: str) -> float:
    return struct.unpack(">d", bytes.fromhex(value))[0]


def complex_record(value: complex) -> dict:
    return {"real_f64": bits(value.real), "imag_f64": bits(value.imag)}


def complex_value(record: dict) -> complex:
    return complex(float64_from_hex(record["real_f64"]), float64_from_hex(record["imag_f64"]))


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


class Evaluator:
    """One task's baseline anchor and physical-parameter cache authority."""

    def __init__(self, plan: dict, analysis: dict, mesh: MeshSpec, backend: NumericalBackend, *, emit=None,
                 trace=None, operation_resources=None, worker_backend_factory=None, worker_capacity: int = 1,
                 candidate_pool=None):
        self.plan, self.analysis, self.mesh, self.backend = plan, analysis, mesh, backend
        self.spec = analysis["spec"]
        self.emit = emit
        self.trace = trace
        self.operation_resources = operation_resources
        self.worker_backend_factory = worker_backend_factory
        self.worker_capacity = worker_capacity
        self.candidate_pool = candidate_pool
        self.preparation_cache = {}
        source = analysis["parameter_source"]
        point = source["parameters"] if source["kind"] == "point" else source["baseline_parameters"] if source["kind"] == "points" else source["base_parameters"]
        self.templates = {parameter_key(binding["parameter"]): binding for binding in point["bindings"]}
        self.base = parameter_values(point)
        self.authorized = {parameter_key(record) for record in self.spec.get("allow_extrapolation", point.get("allow_extrapolation", []))}
        self.quantity_evaluator = QuantityEvaluator(backend, emit=None)
        self.anchors: dict[str, EvaluationResult] = {}
        self._candidate_anchor_references: dict[str, str] = {}
        self.cache: dict[str, dict] = {}
        self.last_batch_stats = {
            "new_unique_candidates": 0,
            "cache_hit_occurrences": 0,
            "same_generation_duplicate_occurrences": 0,
            "numerical_failure_occurrences": 0,
        }
        self.last_active_worker_count = 0
        self.last_workers_used = 0

    def anchor_references(self) -> dict[str, str]:
        """Return references into the baseline candidate's dependency bodies."""
        from ..execution.quantities import quantity_body_id

        return {key: quantity_body_id(result)[0] for key, result in self.anchors.items()}

    def install_anchor_records(self, references: dict[str, str], bodies: dict[str, dict]) -> None:
        """Install only the parent's already-committed baseline anchor bodies."""
        self.anchors = {key: result_from_record(bodies[body_id])
                        for key, body_id in references.items()}
        if self.candidate_pool is not None:
            from ..execution.candidate_state import encode_anchors
            self.candidate_pool.install_anchors(encode_anchors(references, bodies))

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

    def parameter_record(self, values: dict) -> dict:
        bindings = []
        for key, template in self.templates.items():
            value = values[key]
            encoded = value if isinstance(value, dict) else dict(template["value"], si_value_f64=bits(value))
            bindings.append({"parameter": template["parameter"], "value": encoded})
        return {"type": "parameter_set_v2", "bindings": bindings, "allow_extrapolation": []}

    def candidate_view(self, values: dict, declaration: dict, *, preparation_cache: dict | None = None):
        cache = self.preparation_cache if preparation_cache is None else preparation_cache
        return realize_view(compile_model(self.plan, values, mesh=self.mesh, authorized=self.authorized,
                                          preparation_cache=cache), declaration)

    def _worker_quantity_evaluator(self) -> QuantityEvaluator:
        if self.operation_resources is None:
            return self.quantity_evaluator
        if self.worker_backend_factory is None:
            raise RuntimeError("parallel optimization resources require a worker backend factory")
        return self.operation_resources.worker_state(
            "scnsim.optimization.quantity_evaluator",
            lambda: QuantityEvaluator(self.worker_backend_factory(), emit=None),
        )

    def _evaluate_leaf(self, item, *, objective_index: int, term_index: int, selector: dict,
                       baseline: bool) -> tuple[int, dict | None, NumericalFailure | None, dict]:
        index, state, view = item
        identity = f"candidate:{index}:objective:{objective_index}:term:{term_index}"
        quantity_evaluator = self.quantity_evaluator if baseline else self._worker_quantity_evaluator()
        try:
            evaluated = quantity_evaluator.evaluate(
                selector["spec"], view, view_declaration=selector["view"], identity=identity,
                baseline_values=self.base, values=state["values"], anchors=self.anchors,
                result_cache=state["result_cache"], baseline=baseline,
                candidate_view=lambda point, declaration=selector["view"],
                    cache=state["preparation_cache"]:
                        self.candidate_view(point, declaration, preparation_cache=cache),
                selector=selector, dependency_bodies=state["dependencies"],
                candidate_failure_conversion=True,
            )
            return index, evaluated, None, {}
        except (EvaluationFailure, SCNSimError) as error:
            failure = error.failure if isinstance(error, EvaluationFailure) else NumericalFailure(
                error.kind, error.stage, str(error), record_bytes(error.evidence)
            )
            if baseline or failure.kind not in CANDIDATE_FAILURES:
                if isinstance(error, EvaluationFailure):
                    raise numerical_error(failure) from error
                raise
            dependencies = error.dependencies if isinstance(error, EvaluationFailure) else {}
            return index, None, failure, dependencies

    def evaluate_many(self, points: list[dict], *, baseline: bool = False,
                      generation: int | None = None) -> list[dict]:
        """Evaluate declared quantity leaves through the shared dependency owner."""
        objectives = self.spec["objectives"]
        states, keys, representatives = [], [], {}
        cache_hits = 0
        same_generation_duplicates = 0
        for values in points:
            record = self.parameter_record(values)
            key = canonical_json_bytes(record).decode()
            keys.append(key)
            if key in self.cache:
                cache_hits += 1
                states.append(None)
                continue
            if key in representatives:
                same_generation_duplicates += 1
                states.append(None)
                continue
            representatives[key] = len(states)
            state = {"parameters": record, "objectives": [], "cost_f64": bits(0), "failure": None,
                     "dependencies": {}, "result_cache": {}, "views": {}, "values": values}
            if self.candidate_pool is None:
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
            states.append(state)

        if self.candidate_pool is not None:
            from ..execution.candidate_state import encode_point, PREPARATION_SCHEMA

            point_generation = 0 if baseline else generation
            if point_generation is None:
                raise CompilerInvariantError("parallel candidate wave has no generation", stage="candidate_protocol")
            operation_id = self.candidate_pool.declaration["operation_id"]
            submitted = []
            submitted_indexes = []
            for key, index in representatives.items():
                state = states[index]
                point = encode_point(
                    operation_id=operation_id,
                    generation=point_generation,
                    population_column=index,
                    parameter_set=state["parameters"],
                    baseline=baseline,
                )
                submitted.append(point)
                submitted_indexes.append((key, index))
            prepared_rows = iter(self.candidate_pool.prepare_candidates(
                tuple(submitted), generation=point_generation
            ))
            try:
                for key, index in submitted_indexes:
                    try:
                        encoded = next(prepared_rows)
                    except StopIteration as error:
                        raise CompilerInvariantError(
                            "candidate preparation changed batch length", stage="candidate_protocol"
                        ) from error
                    row = record_document(encoded)
                    state = states[index]
                    expected_route = {
                        "schema": PREPARATION_SCHEMA,
                        "schema_version": 1,
                        "operation_id": operation_id,
                        "generation": point_generation,
                        "population_column": index,
                        "candidate_key": key,
                        "parameter_digest": sha256(canonical_json_bytes(state["parameters"])).hexdigest(),
                    }
                    if any(row.get(name) != value for name, value in expected_route.items()):
                        raise CompilerInvariantError("candidate preparation changed routing identity", stage="candidate_protocol")
                    if row.get("status") == "error":
                        error = _candidate_actor_error(row["failure"])
                        if baseline or error.kind not in CANDIDATE_FAILURES:
                            raise error
                        state["failure"] = {
                            "kind": error.kind,
                            "stage": error.stage,
                            "detail": str(error),
                            "evidence": json.loads(record_bytes(error.evidence)),
                            "phase": "candidate_compile",
                        }
                        state["cost_f64"] = bits(math.inf)
                        continue
                    if row.get("status") != "prepared":
                        raise CompilerInvariantError("candidate actor returned an unknown preparation status", stage="candidate_protocol")
                    state["discretization"] = row["discretization"]
                    state["views"] = row["views"]
                try:
                    next(prepared_rows)
                except StopIteration:
                    pass
                else:
                    raise CompilerInvariantError("candidate preparation changed batch length", stage="candidate_protocol")
            finally:
                close = getattr(prepared_rows, "close", None)
                if close is not None:
                    close()

        # The shared preparation cache is mutated only by this ordered
        # compile-first pass. Each candidate receives an independent mapping
        # for any later continuation compilation; cached values are immutable.
        if self.candidate_pool is None:
            for state in states:
                if state is not None and state["failure"] is None:
                    state["preparation_cache"] = dict(self.preparation_cache)

        active_lock = Lock()
        active_workers = 0
        peak_active_workers = 0
        used_actor_ids = set()

        def tracked_leaf(item, *, objective_index: int, term_index: int,
                         selector: dict) -> tuple[int, dict | None, NumericalFailure | None, dict]:
            nonlocal active_workers, peak_active_workers
            with active_lock:
                active_workers += 1
                peak_active_workers = max(peak_active_workers, active_workers)
            try:
                return self._evaluate_leaf(item, objective_index=objective_index,
                                           term_index=term_index, selector=selector,
                                           baseline=baseline)
            finally:
                with active_lock:
                    active_workers -= 1

        for objective_index, objective in enumerate(objectives):
            selectors = leaves(objective["quantity"])
            term_values = [[] for _ in states]
            term_records = [[] for _ in states]
            for term_index, selector in enumerate(selectors):
                wave = [
                    (index, state, state["views"][canonical_json_bytes(selector["view"]).decode()])
                    for index, state in enumerate(states)
                    if state is not None and not state["failure"]
                ]
                if self.candidate_pool is not None:
                    from ..execution.candidate_state import encode_wave

                    operation_id = self.candidate_pool.declaration["operation_id"]
                    point_generation = 0 if baseline else generation
                    actor_ids = []
                    for index, state, _ in wave:
                        digest = sha256(canonical_json_bytes(state["parameters"])).hexdigest()
                        actor = self.candidate_pool.residency[
                            (operation_id, point_generation, index, digest)
                        ]
                        actor_ids.append(actor.actor_id)
                    commands = tuple(encode_wave(
                        operation_id=operation_id,
                        generation=point_generation,
                        population_column=index,
                        parameter_set=state["parameters"],
                        objective_index=objective_index,
                        term_index=term_index,
                        selector=selector,
                        baseline=baseline,
                    ) for index, state, _ in wave)
                    encoded_outcomes = iter(self.candidate_pool.evaluate_wave(commands))
                    outcomes = []
                    try:
                        for index, state, _ in wave:
                            try:
                                encoded = next(encoded_outcomes)
                            except StopIteration as error:
                                raise CompilerInvariantError(
                                    "candidate wave changed batch length", stage="candidate_protocol"
                                ) from error
                            row = record_document(encoded)
                            expected_route = {
                                "schema": "scnsim.candidate-evaluation.v1",
                                "schema_version": 1,
                                "operation_id": operation_id,
                                "generation": point_generation,
                                "population_column": index,
                                "candidate_key": canonical_json_bytes(state["parameters"]).decode("utf-8"),
                                "parameter_digest": sha256(canonical_json_bytes(state["parameters"])).hexdigest(),
                                "objective_index": objective_index,
                                "term_index": term_index,
                            }
                            if any(row.get(name) != value for name, value in expected_route.items()):
                                raise CompilerInvariantError("candidate evaluation changed routing identity", stage="candidate_protocol")
                            if row.get("dependencies"):
                                state["dependencies"].update(row["dependencies"])
                            if baseline and row.get("anchor_references") is not None:
                                self._candidate_anchor_references = row["anchor_references"]
                            status = row.get("status")
                            if status == "evaluated":
                                body_id = row["body_id"]
                                result_record = row.get("result") or state["dependencies"].get(body_id)
                                if result_record is None:
                                    raise CompilerInvariantError("candidate reply omitted an unarchived quantity body", stage="candidate_protocol")
                                evaluated = {
                                    "result": result_from_record(result_record),
                                    "value": row["value"],
                                    "dependencies": row.get("dependencies", {}),
                                    "body_id": body_id,
                                }
                                outcomes.append((index, evaluated, None, {}))
                            elif status == "numerical_failure":
                                failure_record = row["failure"]
                                failure = NumericalFailure(
                                    failure_record["kind"], failure_record["stage"],
                                    failure_record["detail"], record_bytes(failure_record.get("evidence", {})),
                                )
                                if baseline or failure.kind not in CANDIDATE_FAILURES:
                                    raise numerical_error(failure)
                                outcomes.append((index, None, failure, row.get("dependencies", {})))
                            elif status == "error":
                                error = _candidate_actor_error(row["failure"])
                                if baseline or error.kind not in CANDIDATE_FAILURES:
                                    raise error
                                failure = NumericalFailure(
                                    error.kind, error.stage, str(error), record_bytes(error.evidence)
                                )
                                outcomes.append((index, None, failure, row.get("dependencies", {})))
                            else:
                                raise CompilerInvariantError("candidate actor returned an unknown evaluation status", stage="candidate_protocol")
                        try:
                            next(encoded_outcomes)
                        except StopIteration:
                            pass
                        else:
                            raise CompilerInvariantError("candidate wave changed batch length", stage="candidate_protocol")
                    finally:
                        close = getattr(encoded_outcomes, "close", None)
                        if close is not None:
                            close()
                    used_actor_ids.update(actor_ids)
                elif self.operation_resources is not None and not baseline:
                    outcomes = list(self.operation_resources.map_ordered(
                        lambda item: tracked_leaf(item, objective_index=objective_index,
                                                  term_index=term_index, selector=selector),
                        wave,
                        identity=lambda item: (
                            f"generation:{generation}:objective:{objective_index}:"
                            f"term:{term_index}:candidate:{item[0]}"
                        ),
                    ))
                else:
                    outcomes = [tracked_leaf(item, objective_index=objective_index,
                                             term_index=term_index, selector=selector)
                                for item in wave]

                for index, evaluated, failure, failure_dependencies in outcomes:
                    state = states[index]
                    if evaluated is not None:
                        view = state["views"][canonical_json_bytes(selector["view"]).decode()]
                        result = evaluated["result"]
                        value = evaluated["value"]
                        state["dependencies"].update(evaluated["dependencies"])
                        if result.root_omega_rad_s is not None:
                            actual = complex_record(result.root_omega_rad_s)
                        elif result.response_value is not None:
                            actual = complex_record(result.response_value)
                        else:
                            actual = complex_record(result.coupling_rad_s)
                        term_values[index].append(float(value))
                        term_records[index].append({"term_ordinal": term_index, "selector": selector, "status": "success",
                                                    "value_f64": bits(value), "actual_complex": actual,
                                                    "body_id": evaluated["body_id"],
                                                    "lineage": view["lineage"] if self.candidate_pool is not None else json.loads(view.lineage_bytes),
                                                    "evidence": json.loads(result.evidence_bytes)})
                    else:
                        state["dependencies"].update(failure_dependencies)
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
                                            "cost_f64": bits(contribution), "normalized_residual_f64": bits(residual), "terms": term_records[index]})
        for key, index in representatives.items():
            state = states[index]
            if not state["failure"] and not math.isfinite(unbits(state["cost_f64"])):
                failure = NumericalFailure("numerical_resolution_unresolved", "objective", "candidate total cost is non-finite")
                if baseline:
                    raise numerical_error(failure)
                state["failure"] = {"kind": failure.kind, "stage": failure.stage, "detail": failure.detail,
                                    "phase": "total_aggregation"}
                state["cost_f64"] = bits(math.inf)
            if baseline and self.candidate_pool is not None:
                references = self._candidate_anchor_references
                self.anchors = {
                    anchor_key: result_from_record(state["dependencies"][body_id])
                    for anchor_key, body_id in references.items()
                }
            self.cache[key] = {k: v for k, v in state.items()
                               if k not in ("values", "views", "result_cache", "preparation_cache")}
        result, seen = [], set()
        for key in keys:
            result.append(dict(deepcopy(self.cache[key]), candidate_key=key,
                               cache_hit=key not in representatives or key in seen))
            seen.add(key)
        self.last_batch_stats = {
            "new_unique_candidates": len(representatives),
            "cache_hit_occurrences": cache_hits,
            "same_generation_duplicate_occurrences": same_generation_duplicates,
            "numerical_failure_occurrences": sum(record["failure"] is not None for record in result),
        }
        self.last_active_worker_count = None if self.candidate_pool is not None else peak_active_workers
        self.last_workers_used = len(used_actor_ids)
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
             checkpoint_policy: str = "generation", trace=None) -> dict:
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
        baseline_dependencies = checkpoint["baseline"]["dependencies"]
        evaluator.anchors = {
            key: result_from_record(baseline_dependencies[body_id])
            for key, body_id in checkpoint["anchors"].items()
        }
        if evaluator.candidate_pool is not None:
            evaluator.install_anchor_records(checkpoint["anchors"], baseline_dependencies)
        evaluator.cache = checkpoint["cache"]
        baseline, best = checkpoint["baseline"], checkpoint["best"]
        next_ordinal = checkpoint["next_ordinal"]
    else:
        baseline_context = (evaluator.operation_resources.candidate_compute()
                            if evaluator.operation_resources is not None and evaluator.candidate_pool is None
                            else nullcontext())
        with baseline_context:
            baseline = dict(evaluator.evaluate_many([evaluator.base], baseline=True)[0],
                            evaluation_ordinal=0, generation=0, population_column=0)
        best, next_ordinal = baseline, 1

    def resume_state() -> dict | None:
        # Anchors have one durable owner: the acknowledged baseline block.
        return {"cma": cma_snapshot(optimizer)} if checkpoint_policy == "generation" else None

    if checkpoint is None:
        baseline_ack = emit("baseline_ready", {
            "baseline": baseline,
            "anchors": evaluator.anchor_references(),
            "resume_state": resume_state(),
        })
        if evaluator.candidate_pool is not None:
            if not isinstance(baseline_ack, dict) or baseline_ack.get("committed") is not True:
                raise CompilerInvariantError("candidate actors require an acknowledged baseline", stage="candidate_protocol")
            evaluator.install_anchor_records(evaluator.anchor_references(), baseline["dependencies"])
            evaluator.candidate_pool.release_generation(0)
    for generation in range(optimizer.generation + 1, controls["complete_generations"] + 1):
        with (nullcontext() if trace is None else trace.span("generation", details={"generation": generation})):
            start = perf_counter_ns()
            raw = [optimizer.ask() for _ in range(optimizer.population_size)]
            emit("timing", {"stage": "cma_ask", "start_tick_ns": start, "end_tick_ns": perf_counter_ns(),
                            "counts": {"population": len(raw)}})
            emit("population_observed", {"generation": generation, "latent": array_record(np.stack(raw))})
            points = [mapped_values(evaluator.spec, evaluator.base, value) for value in raw]
            before_counters = (evaluator.operation_resources.counters()
                               if evaluator.operation_resources is not None else None)
            before_pool_statistics = (evaluator.candidate_pool.snapshot_statistics()
                                      if evaluator.candidate_pool is not None else None)
            population_start = perf_counter_ns()
            records = evaluator.evaluate_many(points, generation=generation)
            population_end = perf_counter_ns()
            after_pool_statistics = (evaluator.candidate_pool.snapshot_statistics()
                                     if evaluator.candidate_pool is not None else None)
            if evaluator.operation_resources is not None and trace is not None:
                after_counters = evaluator.operation_resources.counters()
                stats = evaluator.last_batch_stats
                performance = {
                    "generation": generation,
                    "population_size": len(raw),
                    "population_evaluation_wall_ns": population_end - population_start,
                    "average_candidate_wall_ns": (population_end - population_start) // len(raw),
                    "new_unique_candidates": stats["new_unique_candidates"],
                    "cache_hit_occurrences": stats["cache_hit_occurrences"],
                    "same_generation_duplicate_occurrences": stats["same_generation_duplicate_occurrences"],
                    "numerical_failure_occurrences": stats["numerical_failure_occurrences"],
                    "worker_capacity": evaluator.worker_capacity,
                    "active_worker_count": evaluator.last_active_worker_count,
                    "workers_used": evaluator.last_workers_used,
                }
                if before_counters is not None:
                    for name in ("assembly_calls", "new_shape_count", "executable_cache_hits"):
                        performance[name] = after_counters[name] - before_counters[name]
                before_workers = (before_pool_statistics.get("worker_statistics")
                                  if isinstance(before_pool_statistics, dict) else None)
                after_workers = (after_pool_statistics.get("worker_statistics")
                                 if isinstance(after_pool_statistics, dict) else None)
                if isinstance(before_workers, dict) and isinstance(after_workers, dict):
                    for name in ("assembly_calls", "new_shape_count", "executable_cache_hits"):
                        if name in before_workers and name in after_workers:
                            performance[f"worker_{name}"] = after_workers[name] - before_workers[name]
                trace.measure(
                    "population_evaluation", start_tick_ns=population_start,
                    end_tick_ns=population_end,
                    details={"generation_performance": performance},
                )
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
            if evaluator.candidate_pool is not None:
                evaluator.candidate_pool.release_generation(generation)
    return {"type": "optimization", "best_ordinal": best["evaluation_ordinal"],
            "completed_generations": optimizer.generation, "unused_evaluations": controls["unused_evaluations"],
            "algorithm_id": "scnsim.python_cmaes_0.13.1.ask_tell.v1"}
