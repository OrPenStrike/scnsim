"""Shared host authority for Direct quantities and certified dependencies.

This module builds ordered backend jobs, owns baseline-root continuation and
projects typed numerical records. Numerical iteration and certification remain
inside the selected backend; public result construction remains in the decoder.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
import math
from time import perf_counter_ns

import numpy as np

from ..canonical import canonical_json_bytes, float64_from_hex
from ..numerics.models import (
    EvaluationJob,
    EvaluationResult,
    NumericalBackend,
    NumericalFailure,
    RootBranch,
)
from ..numeric_encoding import array_from_record, array_record, record_bytes, record_document


_ARRAY_FIELDS = ("S", "Y", "Z", "operator_values", "null_vector", "branch_roots_rad_s")
_SCALAR_FIELDS = (
    "response_value", "root_omega_rad_s", "root_slope", "numerator_slope", "denominator",
    "residue_a", "residue_b", "coupling_rad_s", "evaluation_omega_rad_s",
)
_ROOT_KINDS = frozenset(("diagonal_root", "operator_element_root", "hybridized_pole", "transfer_zero"))
_ROOT_SELECTORS = frozenset((
    "diagonal_root_projection", "operator_element_root_projection", "hybridized_pole_projection",
    "transfer_zero_projection",
))
_LEAF_SELECTORS = _ROOT_SELECTORS | frozenset(("response_element_projection", "residue_coupling_projection"))


def quantity_record(result: EvaluationResult) -> dict[str, object]:
    """Encode the single numerical result body without widening its dtype."""
    record: dict[str, object] = {
        "id": result.id,
        "status": "failure" if result.failure is not None else "success",
        "evidence": record_document(result.evidence_bytes),
    }
    if result.failure is not None:
        record["failure"] = {
            "kind": result.failure.kind,
            "stage": result.failure.stage,
            "detail": result.failure.detail,
            "evidence": record_document(result.failure.evidence_bytes),
        }
        return record
    for name in _ARRAY_FIELDS:
        value = getattr(result, name)
        if value is not None:
            record[name] = array_record(value)
    for name in _SCALAR_FIELDS:
        value = getattr(result, name)
        if value is not None:
            record[name] = array_record(np.asarray(value))
    return record


def result_from_record(record: Mapping[str, object]) -> EvaluationResult:
    """Hydrate an actual typed body; this performs no backend work."""
    identity = record["id"]
    failure_record = record.get("failure")
    failure = None
    if failure_record is not None:
        failure = NumericalFailure(
            failure_record["kind"], failure_record["stage"], failure_record["detail"],
            record_bytes(failure_record.get("evidence", {})),
        )
    values: dict[str, object] = {
        "id": identity,
        "failure": failure,
        "evidence_bytes": record_bytes(record.get("evidence", {})),
    }
    if failure is None:
        for name in _ARRAY_FIELDS:
            encoded = record.get(name)
            if encoded is not None:
                values[name] = array_from_record(encoded)
        for name in _SCALAR_FIELDS:
            encoded = record.get(name)
            if encoded is not None:
                scalar = array_from_record(encoded)
                values[name] = scalar[()] if scalar.shape == () else scalar
    return EvaluationResult(**values)


def quantity_body_id(result: EvaluationResult) -> tuple[str, dict[str, object]]:
    body = quantity_record(result)
    return sha256(record_bytes(body)).hexdigest(), body


def root_frequency_linewidth(root_omega_rad_s: object) -> tuple[float, float]:
    """Project a certified omega into public Hz fields in declaration units."""
    omega = complex(root_omega_rad_s)
    return omega.real / (2.0 * math.pi), -omega.imag / math.pi


def expression_leaves(expression: dict) -> list[dict]:
    """Flatten scalar expression leaves in their declared evaluation order."""
    kind = expression["type"]
    if kind == "quantity_sum":
        return [leaf for term in expression["terms"] for leaf in expression_leaves(term)]
    if kind == "quantity_difference":
        return expression_leaves(expression["left"]) + expression_leaves(expression["right"])
    if kind == "quantity_absolute":
        return expression_leaves(expression["operand"])
    if kind not in _LEAF_SELECTORS:
        raise NotImplementedError(f"benchmark quantity {kind!r} is unsupported")
    return [expression]


def expression_public_unit(expression: dict) -> str:
    """Return the public unit convention owned by a scalar expression."""
    kind = expression["type"]
    if kind == "quantity_sum":
        return expression_public_unit(expression["terms"][0])
    if kind == "quantity_difference":
        return expression_public_unit(expression["left"])
    if kind == "quantity_absolute":
        return expression_public_unit(expression["operand"])
    if kind in _ROOT_SELECTORS:
        return "hertz"
    if kind == "residue_coupling_projection":
        return "radian / second"
    if kind == "response_element_projection":
        return {"S": "dimensionless", "Y": "siemens", "Z": "ohm"}[
            expression["spec"]["family"]
        ]
    raise NotImplementedError(f"benchmark quantity {kind!r} is unsupported")


def _value_in_public_unit(value: float, source: str, target: str) -> float:
    if source == target:
        return value
    if source == "radian / second" and target == "hertz":
        return value / (2.0 * math.pi)
    if source == "hertz" and target == "radian / second":
        return value * (2.0 * math.pi)
    raise ValueError("quantity expression terms do not share a convertible public unit convention")


def expression_value(expression: dict, values: list[float]) -> float:
    iterator = iter(values)

    def evaluate(node: dict) -> float:
        kind = node["type"]
        if kind == "quantity_sum":
            terms = node["terms"]
            target_unit = expression_public_unit(terms[0])
            total = evaluate(terms[0])
            for term in terms[1:]:
                value = evaluate(term)
                total += _value_in_public_unit(value, expression_public_unit(term), target_unit)
            return total
        if kind == "quantity_difference":
            target_unit = expression_public_unit(node["left"])
            left = evaluate(node["left"])
            right = evaluate(node["right"])
            return left - _value_in_public_unit(
                right, expression_public_unit(node["right"]), target_unit
            )
        if kind == "quantity_absolute":
            return abs(evaluate(node["operand"]))
        return next(iterator)

    return evaluate(expression)


class EvaluationFailure(Exception):
    """A structured backend failure encountered while resolving a dependency."""

    def __init__(self, failure: NumericalFailure, *, result: EvaluationResult | None = None,
                 dependencies: Mapping[str, dict[str, object]] | None = None):
        self.failure = failure
        self.result = result
        self.dependencies = dict(dependencies or {})
        super().__init__(failure.detail)


def continue_root(*, baseline: dict, values: dict, baseline_root: complex,
                  endpoint_view, initial_result: EvaluationResult,
                  candidate_view, make_job, evaluate, identity: str, observe=None,
                  continuation_step_scope=None):
    """The existing baseline-anchored dyadic continuation for every root kind."""
    def evaluate_intermediate_step(left_root, right_t):
        # A candidate actor may need to bind its current mutable state around
        # continuation preparation, evaluation and evidence observation. The
        # typed result is immutable, so it can safely leave that scope before
        # recursion.
        scope = (nullcontext() if continuation_step_scope is None
                 else continuation_step_scope())
        with scope:
            point = {}
            for parameter, base in baseline.items():
                candidate = values[parameter]
                if isinstance(base, dict):
                    if canonical_json_bytes(base) != canonical_json_bytes(candidate):
                        from ..errors import CompilerInvariantError
                        raise CompilerInvariantError("root continuation cannot interpolate RLGC", stage="root_continuation")
                    point[parameter] = base
                else:
                    point[parameter] = base + right_t * (candidate - base)
            view = candidate_view(point)
            result = evaluate((make_job(f"continuation:{identity}:{right_t.hex()}", view, left_root),))[0]
            if observe is not None:
                observe(point, view, result, right_t)
            return result

    def advance(left_t, left_root, right_t, depth, result=None):
        if right_t == 1.0:
            point, view = values, endpoint_view
            if result is None:
                result = evaluate((make_job(f"continuation:{identity}:{right_t.hex()}", view, left_root),))[0]
            if observe is not None:
                observe(point, view, result, right_t)
        else:
            result = evaluate_intermediate_step(left_root, right_t)
        if result.failure is None:
            return result
        if result.failure.kind != "numerical_resolution_unresolved" or depth >= 32:
            raise EvaluationFailure(result.failure, result=result)
        midpoint = (left_t + right_t) / 2
        middle = advance(left_t, left_root, midpoint, depth + 1)
        return advance(midpoint, middle.root_omega_rad_s, right_t, depth + 1)

    return advance(0.0, baseline_root, 1.0, 0, initial_result)


class QuantityEvaluator:
    """Resolve typed quantity dependencies against one operation's anchors."""

    def __init__(self, backend: NumericalBackend, *, emit=None, continuation_step_scope=None):
        self.backend = backend
        self.emit = emit
        self.continuation_step_scope = continuation_step_scope

    @staticmethod
    def dependency_key(spec: Mapping[str, object], view: Mapping[str, object]) -> str:
        return canonical_json_bytes({"spec": spec, "view": view, "type": spec["type"]}).decode()

    def evaluate_jobs(self, jobs: tuple[EvaluationJob, ...]) -> tuple[EvaluationResult, ...]:
        emit = self.emit
        start = perf_counter_ns() if emit is not None else None
        try:
            results = self.backend.evaluate_batch(jobs)
            if len(results) != len(jobs) or any(job.id != result.id for job, result in zip(jobs, results)):
                from ..errors import CompilerInvariantError
                raise CompilerInvariantError("numerical batch changed job identity/order", stage="benchmark_protocol")
            return results
        finally:
            if emit is not None:
                emit("timing", {"stage": "numerical_evaluation", "start_tick_ns": start,
                                "end_tick_ns": perf_counter_ns(), "counts": {"jobs": len(jobs)}})

    @staticmethod
    def _anchor_omega(spec: Mapping[str, object]) -> complex:
        kind = spec["type"]
        anchor = spec["root_hint"] if kind in ("diagonal_root", "operator_element_root") else spec["anchor"]
        if anchor.get("type") == "complex_quantity_f64":
            two_pi = 2.0 * math.pi
            return complex(two_pi * float64_from_hex(anchor["real_si_f64"]),
                           two_pi * float64_from_hex(anchor["imag_si_f64"]))
        return complex(2.0 * math.pi * float64_from_hex(anchor["si_value_f64"]), 0.0)

    def _root_job(self, identity: str, spec: Mapping[str, object], view,
                  start: complex | None) -> EvaluationJob:
        kind = spec["type"]
        omega_hint = self._anchor_omega(spec)
        root_hint_hz = float(omega_hint.real / (2.0 * math.pi))
        if kind == "diagonal_root":
            return EvaluationJob(identity, kind, view,
                                 coordinate_index=view.terminal_ids.index(spec["coordinate"]),
                                 root_hint_hz=root_hint_hz,
                                 omega_start_rad_s=omega_hint if start is None else start)
        if kind == "operator_element_root":
            return EvaluationJob(identity, kind, view,
                                 row_index=view.terminal_ids.index(spec["row"]),
                                 column_index=view.terminal_ids.index(spec["column"]),
                                 root_hint_hz=root_hint_hz,
                                 omega_start_rad_s=omega_hint if start is None else start)
        if kind == "hybridized_pole":
            return EvaluationJob(identity, kind, view, root_hint_hz=root_hint_hz,
                                 omega_start_rad_s=omega_hint if start is None else start)
        if kind == "transfer_zero":
            return EvaluationJob(identity, kind, view,
                                 root_hint_hz=root_hint_hz,
                                 omega_start_rad_s=omega_hint if start is None else start,
                                 family=spec["family"],
                                 input_index=view.terminal_ids.index(spec["input_coordinate"]),
                                 output_index=view.terminal_ids.index(spec["output_coordinate"]))
        raise NotImplementedError(f"root dependency {kind!r} is unsupported")

    @staticmethod
    def _diagonal_policy(result: EvaluationResult) -> EvaluationResult:
        if result.failure is None and result.root_omega_rad_s.imag > 0:
            from ..canonical import float64_hex
            evidence = record_document(result.evidence_bytes)
            omega = complex(result.root_omega_rad_s)
            evidence["root_omega_rad_s"] = {
                "real_f64": float64_hex(omega.real), "imag_f64": float64_hex(omega.imag),
            }
            return replace(result, root_omega_rad_s=None, root_slope=None,
                           failure=NumericalFailure(
                               "numerical_resolution_unresolved", "newton_certificate",
                               "diagonal root violates passive imaginary-root policy", record_bytes(evidence)))
        return result

    def _root_result(self, spec, view, *, view_declaration, identity, baseline_values,
                     values, anchors, result_cache, baseline, candidate_view=None,
                     observe=None, dependency_bodies=None,
                     candidate_failure_conversion=False,
                     defer_baseline_diagonal_policy=False) -> EvaluationResult:
        key = self.dependency_key(spec, view_declaration)
        cached = result_cache.get(key)
        if cached is not None:
            return cached
        if baseline:
            start = None
        else:
            anchor = anchors[key]
            start = anchor.root_omega_rad_s
        job = self._root_job(identity, spec, view, start)
        result = self.evaluate_jobs((job,))[0]
        if baseline:
            if (spec["type"] == "diagonal_root"
                    and not defer_baseline_diagonal_policy):
                result = self._diagonal_policy(result)
            if result.failure is None:
                anchors[key] = result
        elif result.failure is not None:
            try:
                result = continue_root(
                    baseline=baseline_values, values=values, baseline_root=anchors[key].root_omega_rad_s,
                    endpoint_view=view, initial_result=result,
                    candidate_view=candidate_view,
                    make_job=lambda continuation_id, candidate, omega: self._root_job(
                        continuation_id, spec, candidate, omega),
                    evaluate=self.evaluate_jobs, identity=identity, observe=observe,
                    continuation_step_scope=self.continuation_step_scope,
                )
            except EvaluationFailure as error:
                if (not candidate_failure_conversion
                        or error.failure.kind != "direct_response_formation"):
                    raise
                failure = NumericalFailure(
                    "numerical_resolution_unresolved", error.failure.stage,
                    "candidate selected response is numerically unresolved",
                    error.failure.evidence_bytes,
                )
                raise EvaluationFailure(failure, result=error.result,
                                        dependencies=dependency_bodies) from error
        if (spec["type"] == "diagonal_root"
                and not (baseline and defer_baseline_diagonal_policy)):
            result = self._diagonal_policy(result)
        evidence = record_document(result.evidence_bytes)
        evidence.setdefault("quantity_kind", spec["type"])
        result = replace(result, evidence_bytes=record_bytes(evidence))
        result_cache[key] = result
        return result

    @staticmethod
    def _projection(selector: Mapping[str, object], result: EvaluationResult) -> float:
        if result.failure is not None:
            raise EvaluationFailure(result.failure, result=result)
        kind = selector["type"]
        projection = selector.get("projection")
        if kind in _ROOT_SELECTORS:
            frequency, linewidth = root_frequency_linewidth(result.root_omega_rad_s)
            values = {"frequency": frequency, "linewidth": linewidth}
            return float(values[projection])
        if kind == "response_element_projection":
            value = result.response_value
            values = {"real": float(np.real(value)), "imag": float(np.imag(value)),
                      "magnitude": float(abs(value))}
            return values[projection]
        if kind == "residue_coupling_projection":
            value = result.coupling_rad_s
            values = {"real": float(np.real(value)), "imag": float(np.imag(value)),
                      "magnitude": float(abs(value))}
            return values[projection]
        raise NotImplementedError(f"quantity selector {kind!r} is unsupported")

    @staticmethod
    def _branch_binding(spec: Mapping[str, object], view: Mapping[str, object]) -> dict[str, object]:
        return {"kind": spec["type"], "spec": dict(spec), "view": dict(view)}

    def _evaluate_branch(self, role, spec, view, *, view_declaration, identity,
                         baseline_values, values, anchors, result_cache, baseline,
                         candidate_view=None, observe=None,
                         dependency_bodies=None, candidate_failure_conversion=False,
                         defer_baseline_diagonal_policy=False
                         ) -> tuple[EvaluationResult, dict[str, object], dict[str, dict[str, object]]]:
        key = self.dependency_key(spec, view_declaration)
        result = self._root_result(
            spec, view, view_declaration=view_declaration, identity=identity,
            baseline_values=baseline_values, values=values, anchors=anchors,
            result_cache=result_cache, baseline=baseline, candidate_view=candidate_view,
            observe=observe, dependency_bodies=dependency_bodies,
            candidate_failure_conversion=candidate_failure_conversion,
            defer_baseline_diagonal_policy=defer_baseline_diagonal_policy,
        )
        if result.failure is not None:
            raise EvaluationFailure(result.failure, result=result,
                                    dependencies=dependency_bodies)
        body_id, body = quantity_body_id(result)
        role_binding = {"role": role, "binding": self._branch_binding(spec, view_declaration),
                        "body_id": body_id}
        return result, role_binding, {body_id: body}

    def evaluate(self, spec: Mapping[str, object], view, *, view_declaration: Mapping[str, object],
                 identity: str, baseline_values: dict, values: dict, anchors: dict[str, EvaluationResult],
                 result_cache: dict[str, EvaluationResult], baseline: bool,
                 candidate_view=None, selector: Mapping[str, object] | None = None,
                 dependency_bodies: dict[str, dict[str, object]] | None = None,
                 observe=None, candidate_failure_conversion=False,
                 defer_baseline_diagonal_policy=False) -> dict[str, object]:
        """Evaluate one encoded Spec, resolving and retaining shared dependencies."""
        kind = spec["type"]
        key = self.dependency_key(spec, view_declaration)
        body_cache = {} if dependency_bodies is None else dependency_bodies
        cached = result_cache.get(key)
        if cached is not None:
            result = cached
            body_id, body = quantity_body_id(result)
            cached_dependencies = {body_id: body}
            if kind == "residue_normalized_coupling":
                for branch in record_document(result.evidence_bytes).get("branches", []):
                    branch_id = branch["body_id"]
                    if branch_id in body_cache:
                        cached_dependencies[branch_id] = body_cache[branch_id]
            body_cache.update(cached_dependencies)
            return {"result": result, "body_id": body_id, "dependencies": cached_dependencies,
                    "value": None if selector is None else self._projection(selector, result)}

        dependencies: dict[str, dict[str, object]] = {}
        branches: list[dict[str, object]] = []
        if kind in _ROOT_KINDS:
            result = self._root_result(
                spec, view, view_declaration=view_declaration, identity=identity,
                baseline_values=baseline_values, values=values, anchors=anchors,
                result_cache=result_cache, baseline=baseline, candidate_view=candidate_view,
                observe=observe, dependency_bodies=body_cache,
                candidate_failure_conversion=candidate_failure_conversion,
                defer_baseline_diagonal_policy=defer_baseline_diagonal_policy,
            )
        elif kind == "response_element":
            if cached is not None:
                result = cached
            else:
                result = self.evaluate_jobs((EvaluationJob(
                    identity, kind, view, frequencies_hz=np.asarray(
                        [float64_from_hex(spec["frequency"]["si_value_f64"])], dtype=np.float64),
                    family=spec["family"],
                    input_index=view.terminal_ids.index(spec["input_coordinate"]),
                    output_index=view.terminal_ids.index(spec["output_coordinate"]),
                ),))[0]
                result_cache[key] = result
        elif kind == "operator":
            if cached is not None:
                result = cached
            else:
                result = self.evaluate_jobs((EvaluationJob(
                    identity, kind, view,
                    frequencies_hz=np.asarray([float64_from_hex(value["si_value_f64"])
                                               for value in spec["frequencies"]], dtype=np.float64),
                ),))[0]
                result_cache[key] = result
        elif kind == "residue_normalized_coupling":
            branch_rows = []
            branch_results = []
            for role, branch_spec in (("a", spec["branch_a"]), ("b", spec["branch_b"])):
                branch_id = f"{identity}:branch:{role}"
                try:
                    branch_result, binding, bodies = self._evaluate_branch(
                        role, branch_spec, view, view_declaration=view_declaration,
                        identity=branch_id, baseline_values=baseline_values, values=values,
                        anchors=anchors, result_cache=result_cache, baseline=baseline,
                        candidate_view=candidate_view, observe=observe,
                        dependency_bodies=dependencies,
                        candidate_failure_conversion=candidate_failure_conversion,
                        defer_baseline_diagonal_policy=defer_baseline_diagonal_policy,
                    )
                except EvaluationFailure as error:
                    dependencies.update(error.dependencies)
                    raise EvaluationFailure(error.failure, result=error.result,
                                            dependencies=dependencies) from error
                branch_rows.append(binding)
                branch_results.append((branch_spec, branch_result))
                dependencies.update(bodies)
                body_cache.update(bodies)
            root_branches = []
            for branch_spec, branch_result in branch_results:
                omega_hint = self._anchor_omega(branch_spec)
                coordinate_index = (
                    view.terminal_ids.index(branch_spec["coordinate"])
                    if branch_spec["type"] == "diagonal_root" else None
                )
                root_branches.append(RootBranch(
                    branch_spec["type"], coordinate_index,
                    float(omega_hint.real / (2.0 * math.pi)), branch_result,
                ))
            frequency = spec["frequency"]
            if frequency == "complex_root_midpoint":
                frequency_mode = "complex_root_midpoint"
                evaluation_omega = None
            else:
                frequency_mode = "fixed"
                evaluation_omega = complex(2.0 * math.pi * float64_from_hex(frequency["si_value_f64"]), 0.0)
            result = self.evaluate_jobs((EvaluationJob(
                identity, kind, view, branches=tuple(root_branches),
                evaluation_omega_rad_s=evaluation_omega, frequency_mode=frequency_mode,
            ),))[0]
            if result.failure is None:
                evidence = record_document(result.evidence_bytes)
                evidence["branches"] = branch_rows
                result = replace(result, evidence_bytes=record_bytes(evidence))
            result_cache[key] = result
        else:
            raise NotImplementedError(f"Direct quantity {kind!r} is unsupported")

        if result.failure is not None:
            if (kind == "response_element" and candidate_failure_conversion and not baseline
                    and result.failure.kind == "direct_response_formation"):
                failure = NumericalFailure(
                    "numerical_resolution_unresolved", result.failure.stage, result.failure.detail,
                    result.failure.evidence_bytes,
                )
                result = replace(result, failure=failure)
            raise EvaluationFailure(result.failure, result=result, dependencies=dependencies)
        evidence = record_document(result.evidence_bytes)
        evidence.setdefault("quantity_kind", kind)
        if kind != "residue_normalized_coupling" and "branches" in evidence:
            branches = evidence["branches"]
        result = replace(result, evidence_bytes=record_bytes(evidence))
        result_cache[key] = result
        body_id, body = quantity_body_id(result)
        dependencies[body_id] = body
        body_cache.update(dependencies)
        value = None if selector is None else self._projection(selector, result)
        return {"result": result, "body_id": body_id, "dependencies": dependencies,
                "branches": branches, "value": value}
