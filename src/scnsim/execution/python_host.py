"""Same-process numerical host for ordinary prepared operations.

Physical lowering, View realization, baseline continuation and CMA use their
existing single authorities. The caller owns operation/request identities,
durable emit barriers and typed result publication; this module never launches
processes, prepares Julia or promotes an unverified result.
"""

from __future__ import annotations

import itertools
import json
from time import perf_counter_ns

import numpy as np

from ..canonical import canonical_json_bytes
from ..compilation.compiler import compile_model, parameter_key, parameter_values
from ..compilation.mesh import quantity
from ..numerics.models import EvaluationJob, EvaluationResult
from ..compilation.models import MeshSpec
from .optimization import Evaluator, checked_results, optimize
from ..numeric_encoding import bits
from ..numerics.evidence import numerical_error
from ..numeric_encoding import array_record
from .quantities import EvaluationFailure, QuantityEvaluator, quantity_body_id, quantity_record
from ..compilation.views import realize_view


def resolved_points(source: dict) -> list[dict]:
    kind = source["kind"]
    if kind == "point":
        return [source["parameters"]]
    if kind == "points":
        return source["points"]
    if kind == "grid":
        points = []
        for combination in itertools.product(*(axis["values"] for axis in source["axes"])):
            base = {parameter_key(binding["parameter"]): binding for binding in source["base_parameters"]["bindings"]}
            for axis, value in zip(source["axes"], combination):
                base[parameter_key(axis["parameter"])] = {"parameter": axis["parameter"], "value": value}
            points.append(dict(source["base_parameters"], bindings=list(base.values())))
        return points
    raise ValueError(f"unsupported parameter source {kind!r}")


def evaluation_record(result) -> dict:
    return quantity_record(result)


def _point_identity(point):
    return canonical_json_bytes({
        "bindings": sorted(point["bindings"], key=lambda binding: parameter_key(binding["parameter"])),
        "allow_extrapolation": sorted(point["allow_extrapolation"], key=parameter_key),
    })


def execute_quantity_points(plan, analysis, mesh, backend, points, emit):
    """Evaluate standalone/sweep quantities through the shared dependency owner."""
    spec = analysis["spec"]
    preparation_cache = {}
    quantity_evaluator = QuantityEvaluator(backend, emit=emit)
    baseline = {
        "type": "parameter_set_v2", "allow_extrapolation": [],
        "bindings": [{"parameter": {"definitions_id": definition["definitions_id"],
                                   "parameter_id": definition["parameter_id"]},
                      "value": definition["baseline"]}
                     for definition in plan["parameter_closure"]["definitions"]],
    }
    baseline_values = parameter_values(baseline)

    def view_at(values, point, stage):
        start = perf_counter_ns()
        try:
            authorized = {parameter_key(reference) for reference in point["allow_extrapolation"]}
            return realize_view(compile_model(plan, values, mesh=mesh, authorized=authorized,
                                              preparation_cache=preparation_cache), analysis["view"])
        finally:
            emit("timing", {"stage": stage, "start_tick_ns": start, "end_tick_ns": perf_counter_ns(),
                            "counts": {"points": 1}})

    baseline_needed = spec["type"] in {
        "diagonal_root", "operator_element_root", "hybridized_pole", "transfer_zero",
        "residue_normalized_coupling",
    }
    anchors = {}
    baseline_key = _point_identity(baseline)
    cache = {}
    if baseline_needed:
        baseline_view = view_at(baseline_values, baseline, "root_baseline_lowering_and_view")
        baseline_cache = {}
        try:
            baseline_evaluated = quantity_evaluator.evaluate(
                spec, baseline_view, view_declaration=analysis["view"], identity="baseline",
                baseline_values=baseline_values, values=baseline_values, anchors=anchors,
                result_cache=baseline_cache, baseline=True,
                candidate_view=lambda values_at: view_at(values_at, baseline, "continuation_lowering_and_view"),
                dependency_bodies={},
                defer_baseline_diagonal_policy=spec["type"] == "diagonal_root",
            )
        except EvaluationFailure as error:
            failed = error.result if error.result is not None else EvaluationResult("baseline", failure=error.failure)
            baseline_record = quantity_record(failed)
            baseline_record.update(parameters=baseline, origin="baseline", cache_hit=False,
                                   dependencies=error.dependencies,
                                   lineage=json.loads(baseline_view.lineage_bytes),
                                   discretization=json.loads(baseline_view.model.evidence_bytes)["discretization"],
                                   terminal_ids=list(baseline_view.terminal_ids))
            emit("evaluation", baseline_record)
            raise numerical_error(error.failure) from error
        baseline_result = baseline_evaluated["result"]
        baseline_dependencies = dict(baseline_evaluated["dependencies"])
        baseline_body_id = quantity_body_id(baseline_result)[0]
        baseline_dependencies.pop(baseline_body_id, None)
        baseline_record = quantity_record(baseline_result)
        baseline_record.update(parameters=baseline, origin="baseline", cache_hit=False,
                               dependencies=baseline_dependencies,
                               lineage=json.loads(baseline_view.lineage_bytes),
                               discretization=json.loads(baseline_view.model.evidence_bytes)["discretization"],
                               terminal_ids=list(baseline_view.terminal_ids))
        emit("evaluation", baseline_record)
        anchor_references = {key: quantity_body_id(result)[0] for key, result in anchors.items()}
        emit("baseline_ready", {"schema": "scnsim.benchmark_root_anchor", "schema_version": 1,
                                "baseline": baseline_record, "anchors": anchor_references,
                                "resume_state": None})
        requested_baseline_result = (
            quantity_evaluator._diagonal_policy(baseline_result)
            if spec["type"] == "diagonal_root" else baseline_result
        )
        cache[baseline_key] = (baseline_view, requested_baseline_result, baseline_dependencies)

    seen = {baseline_key} if baseline_needed else set()
    for ordinal, point in enumerate(points):
        identity = _point_identity(point)
        if identity in cache:
            continue
        values = parameter_values(point)
        view = view_at(values, point, "lowering_and_view")
        point_cache = {}
        dependencies = {}

        def observe(values_at, realized, attempt, t):
            templates = {parameter_key(binding["parameter"]): binding for binding in point["bindings"]}
            bindings = [{"parameter": template["parameter"], "value": value if isinstance(value, dict)
                         else dict(template["value"], si_value_f64=bits(value))}
                        for parameter, template in templates.items() for value in (values_at[parameter],)]
            actual = dict(point, bindings=bindings)
            row = quantity_record(attempt)
            row.update(parameters=actual, origin="root_continuation", source_index=ordinal,
                       continuation_t_f64=bits(t), lineage=json.loads(realized.lineage_bytes),
                       discretization=json.loads(realized.model.evidence_bytes)["discretization"],
                       terminal_ids=list(realized.terminal_ids))
            emit("evaluation", row)

        try:
            evaluated = quantity_evaluator.evaluate(
                spec, view, view_declaration=analysis["view"], identity=str(ordinal),
                baseline_values=baseline_values, values=values, anchors=anchors,
                result_cache=point_cache, baseline=False,
                candidate_view=lambda values_at: view_at(values_at, point, "continuation_lowering_and_view"),
                dependency_bodies=dependencies, observe=observe,
            )
            result = evaluated["result"]
            dependencies = dict(evaluated["dependencies"])
            dependencies.pop(evaluated["body_id"], None)
        except EvaluationFailure as error:
            result = error.result if error.result is not None else EvaluationResult(str(ordinal), failure=error.failure)
            result = result if result.failure is not None and result.failure == error.failure else EvaluationResult(
                result.id, failure=error.failure, evidence_bytes=result.evidence_bytes)
            dependencies = dict(error.dependencies)
        cache[identity] = (view, result, dependencies)

    for ordinal, point in enumerate(points):
        identity = _point_identity(point)
        view, result, dependencies = cache[identity]
        record = quantity_record(result)
        record.update(source_index=ordinal, origin="requested_point", parameters=point,
                      cache_hit=identity in seen, numerical_source_id=result.id,
                      dependencies=dependencies, lineage=json.loads(view.lineage_bytes),
                      discretization=json.loads(view.model.evidence_bytes)["discretization"],
                      terminal_ids=list(view.terminal_ids))
        if spec["type"] == "response_element":
            record["frequencies_hz"] = array_record(
                np.asarray([quantity(spec["frequency"])], dtype=np.float64)
            )
        seen.add(identity)
        emit("evaluation", record)
    failure = next((cache[_point_identity(point)][1].failure for point in points
                    if cache[_point_identity(point)][1].failure is not None), None)
    if failure is not None and analysis["parameter_source"]["kind"] == "point":
        raise numerical_error(failure)
    terminal = {"type": spec["type"], "point_count": len(points), "backend": backend.identity()}
    if baseline_needed:
        terminal["baseline_anchor_schema"] = "scnsim.benchmark_root_anchor"
    return terminal



def execute_analysis(plan_document, prepared_analysis, *, backend, emit, trace,
                     checkpoint=None, checkpoint_policy="generation",
                     operation_resources=None, worker_backend_factory=None, candidate_pool=None) -> dict:
    """Execute one prepared ordinary request through the caller's durable sink."""
    analysis = prepared_analysis.request()
    plan = plan_document
    mesh = MeshSpec()

    def observe(kind, payload):
        if kind == "timing":
            # Raw task intervals become one trace authority; no duplicate timing
            # event/Measurement serialization through the numerical result sink.
            trace.measure(payload["stage"], start_tick_ns=payload["start_tick_ns"],
                          end_tick_ns=payload["end_tick_ns"], details={"counts": payload["counts"]})
            return None
        return emit(kind, payload)

    with trace.span("numerical_host", details={"operation": analysis["operation"]}):
        if analysis["operation"] == "optimize_direct":
            evaluator = Evaluator(
                plan, analysis, mesh, backend, operation_resources=operation_resources,
                worker_backend_factory=worker_backend_factory,
                worker_capacity=analysis["runtime_semantic"]["resources"]["optimization_workers"],
                candidate_pool=candidate_pool,
            )
            return optimize(evaluator, observe, checkpoint, checkpoint_policy=checkpoint_policy, trace=trace)
        return _execute_points(plan, analysis, mesh, backend, observe)


def _execute_points(plan, analysis, mesh, backend, emit):
    spec = analysis["spec"]
    source = analysis["parameter_source"]
    points = resolved_points(source)
    if spec["type"] != "direct_solve":
        return execute_quantity_points(plan, analysis, mesh, backend, points, emit)
    views, jobs = [], []
    preparation_cache = {}
    for ordinal, point in enumerate(points):
        start = perf_counter_ns()
        authorized = {parameter_key(reference) for reference in point["allow_extrapolation"]}
        try:
            model = compile_model(plan, parameter_values(point), mesh=mesh, authorized=authorized, preparation_cache=preparation_cache)
            view = realize_view(model, analysis["view"])
        finally:
            emit("timing", {"stage": "lowering_and_view", "start_tick_ns": start, "end_tick_ns": perf_counter_ns(),
                            "counts": {"point": ordinal}})
        views.append(view)
        if spec["type"] == "direct_solve":
            job = EvaluationJob(str(ordinal), "direct", view, frequencies_hz=np.array([quantity(value) for value in spec["frequencies"]]))
        else:
            raise NotImplementedError(f"benchmark quantity {spec['type']!r} is unsupported")
        jobs.append(job)
    start = perf_counter_ns()
    try:
        results = checked_results(backend, tuple(jobs))
    finally:
        emit("timing", {"stage": "numerical_evaluation", "start_tick_ns": start, "end_tick_ns": perf_counter_ns(),
                        "counts": {"jobs": len(jobs)}})
    for ordinal, (point, view, result) in enumerate(zip(points, views, results)):
        record = evaluation_record(result)
        record.update(source_index=ordinal, parameters=point, frequencies_hz=array_record(jobs[ordinal].frequencies_hz), lineage=json.loads(view.lineage_bytes),
                      discretization=json.loads(view.model.evidence_bytes)["discretization"], terminal_ids=list(view.terminal_ids))
        emit("evaluation", record)
    first_failure = next((result.failure for result in results if result.failure is not None), None)
    if first_failure is not None and source["kind"] == "point":
        raise numerical_error(first_failure)
    return {"type": spec["type"], "point_count": len(points), "backend": backend.identity()}
