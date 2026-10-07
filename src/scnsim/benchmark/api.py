"""Maintained experimental entrypoint and Julia-free Python task dispatch.

Preparation is shared once; every independent task records its own execution.
Original runtime operations remain separate and are dispatched by task.py.
"""

from __future__ import annotations

from dataclasses import replace
import itertools
import json
from time import perf_counter_ns

import numpy as np

from ..canonical import canonical_json_bytes
from ..specs import DirectSolveSpec, DiagonalRootSpec, ResponseElementSpec, OptimizationSpec
from .compiler import compile_model, parameter_key, parameter_values
from .mesh import quantity
from .models import BenchmarkSpec, EvaluationJob, NumericalFailure
from .optimization import (Evaluator, EvaluationFailure, bits, checked_results, complex_record,
                           continue_diagonal_root, leaves, numerical_error, optimize)
from .prepared import PreparedBenchmark, array_record, mesh_from_record, record_bytes
from .views import realize_view


def benchmark(run, ref, spec, *, workspace, benchmark=None, parameters=None, progress=None, resume_from=None):
    from .task import finish_benchmark, run_benchmark
    from .storage import record_preparation_failure
    from .timing import TimingRecorder

    timing = TimingRecorder()
    prepared = None
    try:
        with timing.span("benchmark_end_to_end"):
            with timing.span("shared_preparation"):
                policy = BenchmarkSpec() if benchmark is None else benchmark
                if not isinstance(policy, BenchmarkSpec):
                    raise TypeError("benchmark must be BenchmarkSpec")
                if policy.device != "cpu":
                    raise NotImplementedError("this benchmark scope supports CPU devices only")
                if policy.task_kind not in ("full", "cohort"):
                    raise ValueError(f"unsupported benchmark task kind {policy.task_kind!r}")
                if resume_from is not None and policy.checkpoint == "off":
                    raise ValueError("checkpoint off does not support resume_from")
                run._require_ref(ref)
                if isinstance(spec, DirectSolveSpec):
                    operation = "solve_direct"
                elif isinstance(spec, (DiagonalRootSpec, ResponseElementSpec)):
                    operation = "evaluate_direct"
                elif isinstance(spec, OptimizationSpec):
                    operation = "optimize_direct"
                else:
                    raise NotImplementedError(f"benchmark does not support {type(spec).__name__}")
                analysis = run._prepare_analysis(operation, ref, spec, parameters)
                if isinstance(spec, OptimizationSpec):
                    for objective in analysis.request()["spec"]["objectives"]:
                        leaves(objective["quantity"])
                prepared = PreparedBenchmark.create(plan_bytes=run._plan_bytes, analysis=analysis, benchmark=policy)
            run_benchmark(run=run, ref=ref, spec=spec, prepared=prepared, workspace=workspace,
                          progress=progress, resume_from=resume_from, timing=timing)
    except BaseException as error:
        try:
            if prepared is None:
                record_preparation_failure(workspace, error=error, timing=timing, plan_sha256=run._plan_sha256)
            else:
                finish_benchmark(workspace, timing)
        except BaseException as observation_error:
            raise error from observation_error
        raise
    return finish_benchmark(workspace, timing)


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
    record = {"id": result.id, "status": "failure" if result.failure else "success", "evidence": json.loads(result.evidence_bytes)}
    if result.failure:
        record["failure"] = {"kind": result.failure.kind, "stage": result.failure.stage,
                             "detail": result.failure.detail, "evidence": json.loads(result.failure.evidence_bytes)}
        return record
    for name in ("S", "Y", "Z"):
        array = getattr(result, name)
        if array is not None:
            record[name] = array_record(array)
    if result.response_value is not None:
        record["response_value"] = complex_record(result.response_value)
    if result.root_omega_rad_s is not None:
        record["root_omega_rad_s"] = complex_record(result.root_omega_rad_s)
        record["frequency_hz_f64"] = bits(result.root_omega_rad_s.real / (2 * np.pi))
        record["linewidth_hz_f64"] = bits(-result.root_omega_rad_s.imag / np.pi)
        if result.root_slope is not None:
            record["root_slope"] = complex_record(result.root_slope)
    return record


def execute_root_points(plan, analysis, mesh, backend, points, emit):
    """Standalone roots share the sealed-baseline continuation authority."""
    spec = analysis["spec"]
    preparation_cache = {}
    baseline = {
        "type": "parameter_set_v2", "allow_extrapolation": [],
        "bindings": [{"parameter": {"definitions_id": definition["definitions_id"],
                                   "parameter_id": definition["parameter_id"]},
                      "value": definition["baseline"]}
                     for definition in plan["parameter_closure"]["definitions"]],
    }
    baseline_values = parameter_values(baseline)

    def key(point):
        return canonical_json_bytes({
            "bindings": sorted(point["bindings"], key=lambda binding: parameter_key(binding["parameter"])),
            "allow_extrapolation": sorted(point["allow_extrapolation"], key=parameter_key),
        })

    def view_at(values, point, stage):
        start = perf_counter_ns()
        try:
            authorized = {parameter_key(reference) for reference in point["allow_extrapolation"]}
            return realize_view(compile_model(plan, values, mesh=mesh, authorized=authorized,
                                              preparation_cache=preparation_cache), analysis["view"])
        finally:
            emit("timing", {"stage": stage, "start_tick_ns": start, "end_tick_ns": perf_counter_ns(),
                            "counts": {"points": 1}})

    def job(identity, view, start):
        return EvaluationJob(identity, "diagonal_root", view,
                             coordinate_index=view.terminal_ids.index(spec["coordinate"]),
                             root_hint_hz=quantity(spec["root_hint"]), omega_start_rad_s=start)

    def evaluate(jobs):
        start = perf_counter_ns()
        try:
            return checked_results(backend, jobs)
        finally:
            emit("timing", {"stage": "numerical_evaluation", "start_tick_ns": start,
                            "end_tick_ns": perf_counter_ns(), "counts": {"jobs": len(jobs)}})

    def diagonal_policy(result):
        if result.failure is None and result.root_omega_rad_s.imag > 0:
            evidence = json.loads(result.evidence_bytes)
            evidence["root_omega_rad_s"] = complex_record(result.root_omega_rad_s)
            return replace(result, root_omega_rad_s=None, root_slope=None,
                           failure=NumericalFailure("numerical_resolution_unresolved", "newton_certificate",
                                                    "diagonal root violates passive imaginary-root policy", record_bytes(evidence)))
        return result

    def observation(point, view, result, **context):
        return dict(evaluation_record(result), parameters=point, lineage=json.loads(view.lineage_bytes),
                    discretization=json.loads(view.model.evidence_bytes)["discretization"],
                    terminal_ids=list(view.terminal_ids), **context)

    baseline_view = view_at(baseline_values, baseline, "root_baseline_lowering_and_view")
    baseline_result = evaluate((job("baseline", baseline_view,
                                    complex(2 * np.pi * quantity(spec["root_hint"]))),))[0]
    baseline_record = observation(baseline, baseline_view, baseline_result, origin="baseline", cache_hit=False,
                                  numerical_source_id=baseline_result.id)
    emit("evaluation", baseline_record)
    if baseline_result.failure is not None:
        raise numerical_error(baseline_result.failure)
    # The child blocks until the parent has sealed this exact baseline record.
    emit("baseline_ready", {"schema": "scnsim.benchmark_root_anchor", "schema_version": 1,
                            "baseline": baseline_record, "resume_state": None})
    cache = {key(baseline): (baseline_view, diagonal_policy(baseline_result))}
    pending = {}
    for ordinal, point in enumerate(points):
        identity = key(point)
        if identity not in cache and identity not in pending:
            values = parameter_values(point)
            pending[identity] = (ordinal, point, values, view_at(values, point, "lowering_and_view"))
    jobs = tuple(job(str(ordinal), view, baseline_result.root_omega_rad_s)
                 for ordinal, point, values, view in pending.values())
    results = evaluate(jobs) if jobs else ()
    for (identity, (ordinal, point, values, view)), result in zip(pending.items(), results):
        templates = {parameter_key(binding["parameter"]): binding for binding in point["bindings"]}

        def observe(values_at, realized, attempt, t):
            bindings = [{"parameter": template["parameter"], "value": value if isinstance(value, dict)
                         else dict(template["value"], si_value_f64=bits(value))}
                        for parameter, template in templates.items() for value in (values_at[parameter],)]
            actual = dict(point, bindings=bindings)
            emit("evaluation", observation(actual, realized, attempt, origin="root_continuation",
                                           source_index=ordinal, continuation_t_f64=bits(t)))

        try:
            result = continue_diagonal_root(
                baseline=baseline_values, values=values, baseline_root=baseline_result.root_omega_rad_s,
                endpoint_view=view, initial_result=result,
                candidate_view=lambda values_at: view_at(values_at, point, "continuation_lowering_and_view"),
                make_job=job, evaluate=evaluate, identity=str(ordinal), observe=observe,
            )
            result = diagonal_policy(result)
        except EvaluationFailure as error:
            failed = error.result if error.result is not None else result
            result = replace(failed, failure=error.failure, root_omega_rad_s=None, root_slope=None)
        cache[identity] = view, result
    seen = {key(baseline)}
    for ordinal, point in enumerate(points):
        identity = key(point)
        view, result = cache[identity]
        record = observation(point, view, result, source_index=ordinal, origin="requested_point",
                             cache_hit=identity in seen, numerical_source_id=result.id)
        seen.add(identity)
        emit("evaluation", record)
    failure = next((cache[key(point)][1].failure for point in points if cache[key(point)][1].failure is not None), None)
    if failure is not None:
        raise numerical_error(failure)
    return {"type": "diagonal_root", "baseline_anchor_schema": "scnsim.benchmark_root_anchor",
            "point_count": len(points), "backend": backend.identity()}


def execute_python_task(prepared: PreparedBenchmark, backend, *, emit, checkpoint=None) -> dict:
    declaration = prepared.declaration()
    plan = json.loads(prepared.plan_bytes)
    analysis = declaration["analysis"]
    policy = declaration["benchmark"]
    mesh = mesh_from_record(policy["mesh"])
    if analysis["operation"] == "optimize_direct":
        evaluator = Evaluator(plan, analysis, mesh, backend, emit=emit)
        if policy["task_kind"] == "cohort":
            baseline = dict(evaluator.evaluate_many([evaluator.base], baseline=True)[0],
                            evaluation_ordinal=0, origin="baseline")
            emit("evaluation", baseline)
            emit("baseline_ready", {"schema": "scnsim.benchmark_cohort_anchor", "schema_version": 1,
                                    "baseline": baseline,
                                    "anchors": {key: complex_record(value) for key, value in evaluator.anchors.items()},
                                    "resume_state": None})
            candidates = evaluator.evaluate_many([evaluator.cohort_values(point) for point in policy["cohort"]])
            for ordinal, candidate in enumerate(candidates, 1):
                candidate.update(evaluation_ordinal=ordinal, origin="cohort")
                emit("evaluation", candidate)
            return {"type": "cohort", "candidate_count": len(candidates),
                    "source_inputs": policy["cohort"]}
        return optimize(evaluator, emit, checkpoint, checkpoint_policy=policy["checkpoint"])
    if policy["task_kind"] != "full":
        raise NotImplementedError("cohort tasks require an Optimization declaration")
    spec = analysis["spec"]
    source = analysis["parameter_source"]
    points = resolved_points(source)
    if spec["type"] == "diagonal_root":
        return execute_root_points(plan, analysis, mesh, backend, points, emit)
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
        elif spec["type"] == "response_element":
            job = EvaluationJob(str(ordinal), "response_element", view, frequencies_hz=np.array([quantity(spec["frequency"])]),
                                family=spec["family"], input_index=view.terminal_ids.index(spec["input_coordinate"]),
                                output_index=view.terminal_ids.index(spec["output_coordinate"]))
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
        record.update(source_index=ordinal, parameters=point, lineage=json.loads(view.lineage_bytes),
                      discretization=json.loads(view.model.evidence_bytes)["discretization"], terminal_ids=list(view.terminal_ids))
        emit("evaluation", record)
    first_failure = next((result.failure for result in results if result.failure is not None), None)
    if first_failure is not None:
        raise numerical_error(first_failure)
    return {"type": spec["type"], "point_count": len(points), "backend": backend.identity()}
