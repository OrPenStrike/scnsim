"""Same-process JAX operation coordinator over the verified task journal.

The numerical host owns lowering, candidate order, continuation and CMA state.
This module owns invocation/task binding, durable evidence before callbacks and
pure result projection. It never creates a Julia request, process or receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4
from contextlib import nullcontext
import sys

from ..benchmark import storage
from ..benchmark.identity import environment_identity, environment_snapshot, operation_task_identifier
from ..benchmark.prepared import record_bytes, record_document
from ..canonical import float64_from_hex
from ..errors import OptimizationProgressCallbackError, ResultUnavailableError, RuntimePreparationError
from ..results.decode_jax import decode_jax_operation
from ..specs import OptimizationProgress
from .config import get_runtime_configuration, runtime_resource_identity


_POSITIVE_INFINITY_CANDIDATE_COST = "7ff0000000000000"


def _candidate_cost_from_hex(value: str) -> float:
    """Decode the optimizer's existing failed-candidate cost sentinel.

    Candidate failure rows use positive infinity as a numeric cost while
    retaining their typed failure record. Other cost tokens keep the canonical
    finite-Float64 decoder's validation contract.
    """
    if value == _POSITIVE_INFINITY_CANDIDATE_COST:
        return float("inf")
    return float64_from_hex(value)


def _decode(success, *, decoder, prepared_analysis, bound_spec):
    return decode_jax_operation(
        decoder, projection=success["projection"], request=prepared_analysis.request(),
        plan_sha256=prepared_analysis.request()["plan_sha256"],
        request_sha256=prepared_analysis.request_sha256,
        attempt_sha256=success["attempt_sha256"], result_sha256=success["result_sha256"],
        bound_spec=bound_spec,
    )


def _acknowledge_spans(trace, ack) -> None:
    """Advance the trace cursor only for rows the storage commit confirms."""
    if not isinstance(ack, Mapping) or ack.get("committed") is not True:
        return
    rows = ack.get("committed_spans", ())
    if rows:
        trace.mark_spans_persisted(tuple(rows))


def resolve_jax_operation(*, binding, prepared_analysis, decoder, bound_spec):
    """Read an exact completed numerical request without initializing JAX."""
    with binding.reader():
        success = storage.find_operation_success(binding, prepared_analysis.request_sha256)
    if success is None:
        raise ResultUnavailableError("No verified JAX result exists for this exact request", stage="resolve",
                                     evidence={"request_sha256": prepared_analysis.request_sha256})
    return _decode(success, decoder=decoder, prepared_analysis=prepared_analysis, bound_spec=bound_spec)


def _notify(on_progress, request, *, phase, generation, best_cost, reused=False, trace=None):
    if on_progress is None:
        return
    controls = request["spec"]["optimizer"]
    total = controls["complete_generations"]
    population = controls["resolved_population_size"]
    progress = OptimizationProgress(
        phase=phase, reused=reused, completed_generations=generation,
        total_generations=total, evaluated_count=1 + generation * population,
        requested_budget=controls["max_evaluations"], achievable_evaluations=1 + total * population,
        best_cost=best_cost,
    )
    try:
        with (nullcontext() if trace is None else trace.span("progress_callback", details={"phase": phase})):
            on_progress(progress)
    except Exception as error:
        raise OptimizationProgressCallbackError(
            "optimization progress callback failed", stage="progress_callback",
            evidence={"error_type": type(error).__name__, "phase": phase},
        ) from error


def execute_jax_operation(*, binding, plan_document, prepared_analysis, decoder, bound_spec,
                          on_progress=None, checkpoint_policy="generation", resume_from=None,
                          commit_every_generations=1, trace):
    """Execute or reuse one ordinary request, with callbacks outside storage locks."""
    request = prepared_analysis.request()
    request_sha = prepared_analysis.request_sha256
    optimization = request["operation"] == "optimize_direct"
    if resume_from is not None and (not optimization or checkpoint_policy != "generation"):
        raise RuntimePreparationError("Only generation-checkpointed JAX Optimization can resume",
                                      stage="resume_prepare")

    def notify_committed_generation(ack):
        if not optimization or ack.get("committed") is not True:
            return
        generation = ack.get("latest_generation")
        if not isinstance(generation, int) or isinstance(generation, bool):
            return
        committed_best_cost = float64_from_hex(ack["best_cost_f64"])
        _notify(on_progress, request, phase="generation", generation=generation,
                best_cost=committed_best_cost, trace=trace)

    # A resume is an explicit request to continue its exact state, never a cache
    # lookup that silently disregards the supplied checkpoint.
    if resume_from is None:
        with trace.span("result_lookup"):
            with binding.reader():
                success = storage.find_operation_success(binding, request_sha)
        if success is not None:
            trace.bind(operation=request["operation"], request_sha256=request_sha,
                       task_id=success["task_id"], environment_sha256=success["environment_sha256"],
                       attempt_id=None, details={"cache_hit": True, "checkpoint_policy": checkpoint_policy,
                                                 "checkpoint_available": success["checkpoint_available"],
                                                 "requested_commit_every_generations": (
                                                     commit_every_generations if optimization else None
                                                 ),
                                                 "commit_every_generations": None})
            trace.add_numerical_ref(success["result_ref"])
            # Cache-hit trace rows are global operation facts; no fictitious
            # numerical attempt is allocated to store these observations.
            with binding.writer():
                ack = storage.record_operation_event(binding, {"event": "cache_hit", "operation_id": trace.operation_id,
                    "checkpoint_policy": checkpoint_policy, "checkpoint_available": success["checkpoint_available"],
                    "result": success["result_ref"], "spans": list(trace.pending_spans())})
            _acknowledge_spans(trace, ack)
            if optimization:
                projection = success["projection"]
                best = next(row for row in (projection["baseline"], *projection["evaluations"])
                            if row["evaluation_ordinal"] == projection["terminal"]["best_ordinal"])
                _notify(on_progress, request, phase="result_reuse", reused=True,
                        generation=projection["terminal"]["completed_generations"],
                        best_cost=float64_from_hex(best["cost_f64"]), trace=trace)
            with trace.span("result_decode"):
                return _decode(success, decoder=decoder, prepared_analysis=prepared_analysis, bound_spec=bound_spec)

    from ..benchmark.backends.jax_backend import get_jax_backend
    from .python_host import execute_analysis

    resources = request["runtime_semantic"]["resources"]
    precision = request["runtime_semantic"]["precision"]
    if runtime_resource_identity() != resources:
        raise RuntimePreparationError("Prepared JAX resources changed before execution", stage="runtime_prepare")
    with trace.span("backend_prepare"):
        backend = get_jax_backend(precision=precision, resources=get_runtime_configuration(), trace=trace)
    writer = None
    attempt_id = None
    completed = False
    try:
        arm = f"{trace.method}/jax/{precision}/{checkpoint_policy}"
        environment = environment_snapshot(arm=arm, device="cpu", cpu_threads=resources["cpu_threads"],
                                           backend=backend.identity())
        environment["runtime_semantic"] = request["runtime_semantic"]
        environment["resources"] = resources
        environment_sha = environment_identity(environment)
        environment["environment_sha256"] = environment_sha
        task_id = operation_task_identifier(
            plan_sha256=request["plan_sha256"], request_sha256=request_sha, method=trace.method,
            backend="jax", precision=precision, resources=resources,
            algorithm_id=request["runtime_semantic"]["algorithm_id"],
            environment_sha256=environment_sha, checkpoint_policy=checkpoint_policy,
            commit_every_generations=commit_every_generations,
        )
        workspace = storage.operation_workspace(binding)
        checkpoint = None
        if resume_from is not None:
            with trace.span("checkpoint_read"):
                with binding.reader():
                    checkpoint = record_document(storage.read_checkpoint(
                        workspace, resume_from, binding=binding, expected_task_id=task_id,
                        expected_request_sha256=request_sha,
                        expected_arm=arm, expected_sample=0, expected_environment_sha256=environment_sha,
                    ))
        with binding.writer():
            request_ref = storage.store_operation_request(
                binding, request_sha256=request_sha, request_bytes=prepared_analysis.request_bytes,
            )
        trace.bind(operation=request["operation"], request_sha256=request_sha, task_id=task_id,
                   environment_sha256=environment_sha, attempt_id=None,
                   details={"cache_hit": False, "checkpoint_policy": checkpoint_policy,
                            "requested_commit_every_generations": (
                                commit_every_generations if optimization else None
                            ),
                            "commit_every_generations": (
                                commit_every_generations if optimization else None
                            ),
                            "request": request_ref})
        with binding.writer():
            storage.ensure_operation_task(binding, {
                "task_id": task_id, "request_sha256": request_sha, "arm": arm, "sample": 0,
                "attempts": [], "events": [], "measurements": [], "environment": environment, "artifacts": [request_ref],
            })
            attempt_id = str(uuid4())
            storage.begin_attempt(workspace, binding=binding, task_id=task_id, attempt_id=attempt_id,
                                  resume_from=resume_from)
        trace.set_attempt(attempt_id)
        with binding.writer():
            writer = storage.begin_task_writer(workspace, binding=binding, task_id=task_id, attempt_id=attempt_id,
                                               diagnostics="boundary", checkpoint_document=checkpoint,
                                               commit_every_generations=commit_every_generations,
                                               phase_scope=lambda kind, details: trace.span(kind, details=details))
            storage.update_attempt(workspace, binding=binding, task_id=task_id, attempt_id=attempt_id,
                                   status="running")
        best_cost = float64_from_hex(checkpoint["best"]["cost_f64"]) if checkpoint else None

        def emit(kind, payload):
            nonlocal best_cost
            value = {**payload, "attempt_id": attempt_id}
            if kind in {"baseline_ready", "generation_ready"}:
                details = {"barrier": kind, "checkpoint_policy": checkpoint_policy,
                           "commit_every_generations": (
                               commit_every_generations if optimization else None
                           )}
                if kind == "generation_ready":
                    details["generation"] = payload["generation"]
                    # Persist only the completed-boundary best value in the
                    # generation ACK; an incomplete later population must not
                    # leak into a callback after an error-tail commit.
                # This span closes after handing off the existing rows, so its
                # own row remains pending until a later boundary/finalization.
                with trace.span("diagnostic_archive", details=details):
                    trace.publish_pending_spans(writer)
                # Submission includes Plan authority acquisition/revalidation;
                # the nested commit includes evidence, diagnostics and HEAD work.
                with trace.span("workspace_submission", details=details):
                    with binding.writer():
                        with trace.span("evidence_checkpoint_commit", details=details):
                            _, ack = storage.commit_barrier(workspace, writer=writer, kind=kind, payload=value)
                _acknowledge_spans(trace, ack)
                if ack["checkpoint"] is not None:
                    trace.add_numerical_ref(ack["checkpoint"])
                if optimization:
                    if kind == "baseline_ready":
                        best_cost = float64_from_hex(payload["baseline"]["cost_f64"])
                        if ack.get("committed") is True:
                            _notify(on_progress, request, phase="initial", generation=0,
                                    best_cost=best_cost, trace=trace)
                    else:
                        notify_committed_generation(ack)
                return ack
            trace.publish_pending_spans(writer)
            if optimization and kind == "evaluation":
                cost = _candidate_cost_from_hex(payload["cost_f64"])
                if best_cost is None or cost < best_cost:
                    best_cost = cost
            if kind in storage._DIAGNOSTIC_EVENTS:
                event = storage.append_event(workspace, task_id=task_id, kind=kind, payload=value, writer=writer)
                _acknowledge_spans(trace, getattr(writer, "last_ack", None))
                return event
            with binding.writer():
                event = storage.append_event(workspace, task_id=task_id, kind=kind, payload=value, writer=writer)
            _acknowledge_spans(trace, getattr(writer, "last_ack", None))
            return event

        if checkpoint is not None:
            trace.add_numerical_ref(resume_from)
            with binding.writer():
                storage.append_event(workspace, task_id=task_id, kind="resumed", writer=writer, force=True,
                                     payload={"attempt_id": attempt_id, "checkpoint": dict(resume_from)})
            _notify(on_progress, request, phase="resume", generation=checkpoint["generation"], best_cost=best_cost, trace=trace)
        terminal = execute_analysis(plan_document, prepared_analysis, backend=backend, emit=emit, trace=trace,
                                    checkpoint=checkpoint, checkpoint_policy=checkpoint_policy)
        if writer is not None:
            trace.publish_pending_spans(writer)
            with binding.writer():
                tail_ack = storage.flush_completed(workspace, writer=writer, reason="terminal")
            _acknowledge_spans(trace, tail_ack)
            if tail_ack.get("checkpoint") is not None:
                trace.add_numerical_ref(tail_ack["checkpoint"])
            notify_committed_generation(tail_ack)
        with trace.span("result_publish"):
            with binding.writer():
                result_ref, result_ack = storage.complete_operation(
                    binding, writer=writer, terminal_bytes=record_bytes(terminal),
                )
                _acknowledge_spans(trace, result_ack)
                completed = True
        trace.add_numerical_ref(result_ref)
        if optimization:
            _notify(on_progress, request, phase="complete", generation=terminal["completed_generations"], best_cost=best_cost, trace=trace)
        with trace.span("result_decode"):
            with binding.reader():
                success = storage.read_operation_success(binding, task_id, attempt_id=attempt_id)
            return _decode(success, decoder=decoder, prepared_analysis=prepared_analysis, bound_spec=bound_spec)
    except BaseException as error:
        if attempt_id is not None and not completed:
            try:
                if writer is not None:
                    trace.publish_pending_spans(writer)
                    with binding.writer():
                        tail_ack = storage.flush_completed(workspace, writer=writer, reason="failure")
                    _acknowledge_spans(trace, tail_ack)
                    if tail_ack.get("checkpoint") is not None:
                        trace.add_numerical_ref(tail_ack["checkpoint"])
                    try:
                        notify_committed_generation(tail_ack)
                    except BaseException as callback_error:
                        error.add_note(
                            f"Generation callback during failure flush also failed: "
                            f"{type(callback_error).__name__}: {callback_error}"
                        )
                    trace.publish_pending_spans(writer)
                failure = storage._error_document(error)
                interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
                with binding.writer():
                    storage.append_event(workspace, task_id=task_id, kind="interrupted" if interrupted else "failed",
                                         payload={"attempt_id": attempt_id,
                                                  "interruption" if interrupted else "failure": failure}, writer=writer, force=True)
                    storage.update_attempt(workspace, binding=binding, task_id=task_id, attempt_id=attempt_id,
                                           status="interrupted" if interrupted else "failure",
                                           interruption=failure if interrupted else None,
                                           failure=None if interrupted else failure)
                if writer is not None:
                    _acknowledge_spans(trace, getattr(writer, "last_ack", None))
            except BaseException as recording_error:
                error.add_note(f"Operation failure recording also failed: {type(recording_error).__name__}: {recording_error}")
        raise
    finally:
        primary = sys.exception()
        try:
            backend.close()
        except BaseException as finalization_error:
            if primary is None:
                raise
            primary.add_note(f"Operation finalization also failed: {type(finalization_error).__name__}: {finalization_error}")
