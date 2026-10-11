"""Same-process JAX operation coordinator over the verified task journal.

The numerical host owns lowering, candidate order, continuation and CMA state.
This module owns invocation/task binding, durable evidence before callbacks and
pure result projection. It never creates a Julia request, process or receipt.
"""

from __future__ import annotations

from uuid import uuid4
from contextlib import nullcontext
import sys

from ..workspace import evidence as storage
from ..workspace.operation_lease import request_execution_lease
from ..numeric_encoding import record_bytes, record_document
from ..canonical import float64_from_hex
from ..errors import EvidenceIntegrityError, OptimizationProgressCallbackError, ResultUnavailableError, RuntimePreparationError
from ..results.decode_jax import decode_jax_operation
from ..specs import OptimizationProgress
from ..diagnostics.identity import (
    environment_identity,
    environment_snapshot,
    operation_task_identifier,
)
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
    request = prepared_analysis.request()
    fixed_reader = None
    if request["operation"] == "optimize_direct":
        fixed_reader = success["fixed_reader"]
    return decode_jax_operation(
        decoder, projection=success["projection"], request=request,
        plan_sha256=request["plan_sha256"],
        request_sha256=prepared_analysis.request_sha256,
        attempt_sha256=success["attempt_sha256"], result_sha256=success["result_sha256"],
        bound_spec=bound_spec, fixed_reader=fixed_reader,
    )


def _candidate_actor_declaration(*, operation_id, plan_document, request, request_sha256,
                                 prepared_analysis, precision, resources):
    return record_bytes({
        "schema": "scnsim.candidate-actor-declaration.v1",
        "schema_version": 1,
        "operation_id": operation_id,
        "plan": {"sha256": request["plan_sha256"], "document": plan_document},
        "request": {"sha256": request_sha256, "document": request},
        "source_units": [value.hex() for value in prepared_analysis.source_unit_bytes],
        "mesh": {"kind": "dynamic", "sections": [], "parameter_key": None,
                 "groups": [], "derivation": {}},
        "precision": precision,
        "resources": resources,
        "algorithm_id": request["runtime_semantic"]["algorithm_id"],
        "population_size": request["spec"]["optimizer"]["resolved_population_size"],
    })


def resolve_jax_operation(*, binding, prepared_analysis, decoder, bound_spec):
    """Read an exact completed numerical request without initializing JAX."""
    with binding.reader():
        result = storage.find_operation_success(
            binding, prepared_analysis.request_sha256,
            projection_consumer=lambda success: _decode(
                success, decoder=decoder, prepared_analysis=prepared_analysis, bound_spec=bound_spec,
            ),
        )
    if result is None:
        raise ResultUnavailableError("No verified JAX result exists for this exact request", stage="resolve",
                                     evidence={"request_sha256": prepared_analysis.request_sha256})
    return result


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
                          on_progress=None, progress_observer=None,
                          checkpoint_policy="generation", resume_from=None,
                          commit_every_generations=1, trace):
    """Execute or reuse one request while serializing its exact result authority."""
    with request_execution_lease(binding, prepared_analysis.request_sha256):
        return _execute_jax_operation(
            binding=binding,
            plan_document=plan_document,
            prepared_analysis=prepared_analysis,
            decoder=decoder,
            bound_spec=bound_spec,
            on_progress=on_progress,
            progress_observer=progress_observer,
            checkpoint_policy=checkpoint_policy,
            resume_from=resume_from,
            commit_every_generations=commit_every_generations,
            trace=trace,
        )


def _execute_jax_operation(*, binding, plan_document, prepared_analysis, decoder, bound_spec,
                           on_progress=None, progress_observer=None,
                           checkpoint_policy="generation", resume_from=None,
                           commit_every_generations=1, trace):
    """Execute or reuse one request with callbacks outside Workspace locks."""
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
                def decode_selected(success):
                    with trace.span("result_decode"):
                        result = _decode(success, decoder=decoder, prepared_analysis=prepared_analysis,
                                         bound_spec=bound_spec)
                    return success, result

                selected = storage.find_operation_success(
                    binding, request_sha, projection_consumer=decode_selected,
                )
        if selected is None:
            success = None
        else:
            success, result = selected
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
            if optimization:
                terminal = success["projection"]["terminal"]
                best_cost = result.best.cost
                if progress_observer is not None:
                    progress_observer.update(
                        phase="result_reuse",
                        completed_generations=terminal["completed_generations"],
                        total_generations=request["spec"]["optimizer"]["complete_generations"],
                        evaluated_count=1 + terminal["completed_generations"] *
                        request["spec"]["optimizer"]["resolved_population_size"],
                        best_cost=best_cost,
                        committed_generations=terminal["completed_generations"],
                        reused=True,
                    )
                _notify(on_progress, request, phase="result_reuse", reused=True,
                        generation=terminal["completed_generations"],
                        best_cost=best_cost, trace=trace)
            if optimization and progress_observer is not None:
                progress_observer.update(
                    phase="complete",
                    completed_generations=terminal["completed_generations"],
                    total_generations=request["spec"]["optimizer"]["complete_generations"],
                    evaluated_count=1 + terminal["completed_generations"] *
                    request["spec"]["optimizer"]["resolved_population_size"],
                    best_cost=best_cost,
                    committed_generations=terminal["completed_generations"],
                    reused=True,
                )
            return result

    from .jax_backend import get_jax_backend
    from .resources import operation_resources

    resources = request["runtime_semantic"]["resources"]
    precision = request["runtime_semantic"]["precision"]
    if runtime_resource_identity() != resources:
        raise RuntimePreparationError("Prepared JAX resources changed before execution", stage="runtime_prepare")
    runtime_configuration = get_runtime_configuration()
    with operation_resources(cpu_threads=runtime_configuration.cpu_threads) as operation_resource:
        if optimization and runtime_configuration.optimization_workers > 1:
            from .candidate_pool import CandidatePool

            declaration_bytes = _candidate_actor_declaration(
                operation_id=trace.operation_id, plan_document=plan_document, request=request,
                request_sha256=request_sha, prepared_analysis=prepared_analysis,
                precision=precision, resources=resources,
            )
            with CandidatePool(
                declaration_bytes, capacity=runtime_configuration.optimization_workers
            ) as candidate_pool:
                readiness = candidate_pool.snapshot_statistics()["actor_readiness"]
                backend_identity = readiness[0]["backend_identity"]
                return _execute_with_backend(
                    binding=binding, plan_document=plan_document, prepared_analysis=prepared_analysis,
                    decoder=decoder, bound_spec=bound_spec, on_progress=on_progress,
                    progress_observer=progress_observer,
                    checkpoint_policy=checkpoint_policy, resume_from=resume_from,
                    commit_every_generations=commit_every_generations, trace=trace,
                    request=request, request_sha=request_sha, optimization=optimization, precision=precision,
                    resources=resources, backend=None, backend_identity=backend_identity,
                    operation_resource=operation_resource, worker_backend_factory=None,
                    candidate_pool=candidate_pool,
                    notify_committed_generation=notify_committed_generation,
                )

        with trace.span("backend_prepare"):
            backend = get_jax_backend(
                precision=precision, resources=runtime_configuration,
                trace=None if optimization else trace,
                operation_resources=operation_resource, diagnostics=not optimization,
            )

        worker_backend_factory = None
        if optimization:
            def worker_backend_factory():
                return operation_resource.worker_state(
                    "scnsim.optimization.jax_backend",
                    lambda: get_jax_backend(
                        precision=precision, resources=runtime_configuration, trace=None,
                        operation_resources=operation_resource, diagnostics=False,
                    ),
                )

        return _execute_with_backend(
            binding=binding, plan_document=plan_document, prepared_analysis=prepared_analysis,
            decoder=decoder, bound_spec=bound_spec, on_progress=on_progress,
            progress_observer=progress_observer,
            checkpoint_policy=checkpoint_policy, resume_from=resume_from,
            commit_every_generations=commit_every_generations, trace=trace,
            request=request, request_sha=request_sha, optimization=optimization, precision=precision,
            resources=resources, backend=backend, backend_identity=backend.identity(),
            operation_resource=operation_resource, worker_backend_factory=worker_backend_factory,
            candidate_pool=None,
            notify_committed_generation=notify_committed_generation,
        )


def _execute_with_backend(*, binding, plan_document, prepared_analysis, decoder, bound_spec,
                          on_progress, progress_observer, checkpoint_policy, resume_from, commit_every_generations,
                          trace, request, request_sha, optimization, precision, resources, backend,
                          backend_identity, operation_resource, worker_backend_factory, candidate_pool,
                          notify_committed_generation):
    from .python_host import execute_analysis

    writer = None
    attempt_id = None
    registration_attempted = False
    registration_committed = False
    completed = False
    spool = None
    total_generations = request["spec"]["optimizer"]["complete_generations"] if optimization else 0
    population_size = request["spec"]["optimizer"]["resolved_population_size"] if optimization else 0
    completed_generations = 0
    committed_generations = 0
    evaluated_count = 0
    best_progress_cost = None

    def update_progress(phase, *, generation=None, evaluated=None, best_cost=None,
                        committed=None, reused=False):
        if not optimization or progress_observer is None:
            return
        progress_observer.update(
            phase=phase,
            completed_generations=completed_generations if generation is None else generation,
            total_generations=total_generations,
            evaluated_count=evaluated_count if evaluated is None else evaluated,
            best_cost=best_progress_cost if best_cost is None else best_cost,
            committed_generations=committed_generations if committed is None else committed,
            reused=reused,
        )
    try:
        arm = f"{trace.method}/jax/{precision}/{checkpoint_policy}"
        environment = environment_snapshot(arm=arm, device="cpu", cpu_threads=resources["cpu_threads"],
                                           backend=backend_identity)
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
        if optimization:
            from .operation_spool import OperationSpool
            spool = OperationSpool(workspace, trace.operation_id, lease_root=binding.root)
        checkpoint = None
        resume_selection = None
        if resume_from is not None:
            with trace.span("checkpoint_read"):
                with binding.reader():
                    checkpoint_value, resume_selection = storage.read_checkpoint(
                        workspace, resume_from, binding=binding, expected_task_id=task_id,
                        expected_request_sha256=request_sha,
                        expected_arm=arm, expected_sample=0, expected_environment_sha256=environment_sha,
                        return_selection=True, spool=spool,
                    )
                    checkpoint = checkpoint_value
                    if not isinstance(checkpoint, dict):
                        checkpoint = record_document(checkpoint)
                    completed_generations = checkpoint["generation"]
                    committed_generations = checkpoint["generation"]
                    evaluated_count = checkpoint["next_ordinal"]
                    best_progress_cost = float64_from_hex(checkpoint["best"]["cost_f64"])
        attempt_id = str(uuid4())
        details = {"cache_hit": False, "checkpoint_policy": checkpoint_policy,
                   "requested_commit_every_generations": (
                       commit_every_generations if optimization else None
                   ),
                   "commit_every_generations": (
                       commit_every_generations if optimization else None
                   )}
        trace.bind(operation=request["operation"], request_sha256=request_sha, task_id=task_id,
                   environment_sha256=environment_sha, attempt_id=None, details=details)
        trace.set_attempt(attempt_id)
        registration_attempted = True
        with binding.writer():
            registration = storage.register_operation_execution(
                binding,
                row=trace.root_row(),
                task={"task_id": task_id, "request_sha256": request_sha, "arm": arm, "sample": 0,
                      "attempts": [], "events": [], "measurements": [], "environment": environment,
                      "artifacts": []},
                request={"request_sha256": request_sha, "request_bytes": prepared_analysis.request_bytes},
                attempt_id=attempt_id,
                resume_from=resume_selection,
            )
            if registration["ack"].get("committed") is not True:
                raise EvidenceIntegrityError(
                    "Operation execution registration was not durably acknowledged.",
                    stage="operation_store",
                    evidence={"operation_id": trace.operation_id},
                )
            registration_committed = True
            request_ref = registration["request_ref"]
            details["request"] = request_ref
            trace.bind(operation=request["operation"], request_sha256=request_sha, task_id=task_id,
                       environment_sha256=environment_sha, attempt_id=attempt_id, details=details)
            writer = storage.begin_task_writer(workspace, binding=binding, operation_id=trace.operation_id,
                                               task_id=task_id, attempt_id=attempt_id,
                                               diagnostics="boundary", checkpoint_document=checkpoint,
                                               commit_every_generations=commit_every_generations,
                                               spool=spool,
                                               phase_scope=lambda kind, details: trace.span(kind, details=details))
        best_cost = float64_from_hex(checkpoint["best"]["cost_f64"]) if checkpoint else None
        if checkpoint is None:
            update_progress("preparing", generation=0, evaluated=0, committed=0)

        def emit(kind, payload):
            nonlocal best_cost, completed_generations, evaluated_count
            nonlocal best_progress_cost, committed_generations
            if kind == "population_evaluation_start":
                update_progress(
                    "population_evaluating", generation=completed_generations,
                    evaluated=evaluated_count, best_cost=best_progress_cost,
                    committed=committed_generations,
                )
                return None
            value = {**payload, "attempt_id": attempt_id}
            if kind == "timing":
                trace.measure(
                    payload["stage"], start_tick_ns=payload["start_tick_ns"],
                    end_tick_ns=payload["end_tick_ns"],
                    details={"stage":payload["stage"], "counts":payload.get("counts", {})},
                )
                return None
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
                if writer.will_commit_barrier(kind):
                    if optimization and kind == "generation_ready":
                        update_progress(
                            "saving", generation=payload["generation"],
                            evaluated=payload["next_ordinal"], committed=committed_generations,
                        )
                    # Submission includes Plan authority acquisition/revalidation;
                    # the nested commit includes evidence, diagnostics and SQL work.
                    with trace.span("workspace_submission", details=details):
                        with binding.writer():
                            with trace.span("evidence_checkpoint_commit", details=details):
                                _, ack = storage.commit_barrier(workspace, writer=writer,
                                                               kind=kind, payload=value)
                else:
                    # Baselines and incomplete commit groups are memory-only.
                    # They neither acquire the Plan writer nor claim commit time.
                    _, ack = storage.commit_barrier(workspace, writer=writer, kind=kind, payload=value)
                if ack["checkpoint"] is not None:
                    trace.add_numerical_ref(ack["checkpoint"])
                if optimization:
                    if kind == "baseline_ready":
                        best_cost = float64_from_hex(payload["baseline"]["cost_f64"])
                        best_progress_cost = best_cost
                        evaluated_count = 1
                        update_progress("baseline", generation=0, evaluated=1, committed=0)
                    else:
                        completed_generations = payload["generation"]
                        evaluated_count = payload["next_ordinal"]
                        if best_cost is not None:
                            best_progress_cost = best_cost
                        if ack.get("committed") is True and isinstance(ack.get("latest_generation"), int):
                            trace.complete_generation(ack["latest_generation"])
                            committed_generations = ack["latest_generation"]
                        update_progress(
                            "running", generation=completed_generations, evaluated=evaluated_count,
                            committed=committed_generations,
                        )
                        notify_committed_generation(ack)
                if ack.get("committed") is True:
                    trace.publish_diagnostics()
                return ack
            if optimization and kind == "evaluation":
                cost = _candidate_cost_from_hex(payload["cost_f64"])
                if best_cost is None or cost < best_cost:
                    best_cost = cost
            if storage.is_diagnostic_event(kind):
                return storage.append_event(workspace, task_id=task_id, kind=kind, payload=value, writer=writer)
            with binding.writer():
                event = storage.append_event(workspace, task_id=task_id, kind=kind, payload=value, writer=writer)
            return event

        if checkpoint is not None:
            trace.add_numerical_ref(resume_from)
            with binding.writer():
                storage.append_event(workspace, task_id=task_id, kind="resumed", writer=writer, force=True,
                                     payload={"attempt_id": attempt_id, "checkpoint": dict(resume_selection)})
            _notify(on_progress, request, phase="resume", generation=checkpoint["generation"], best_cost=best_cost, trace=trace)
            update_progress("resume", generation=checkpoint["generation"],
                            evaluated=checkpoint["next_ordinal"], best_cost=best_cost,
                            committed=checkpoint["generation"])
        with (spool.activate() if spool is not None else nullcontext()):
            terminal = execute_analysis(plan_document, prepared_analysis, backend=backend, emit=emit, trace=trace,
                                        checkpoint=checkpoint, checkpoint_policy=checkpoint_policy,
                                        operation_resources=operation_resource,
                                        worker_backend_factory=worker_backend_factory,
                                        candidate_pool=candidate_pool)
        checkpoint = None
        if writer is not None and writer.completed:
            with binding.writer():
                tail_ack = storage.flush_completed(workspace, writer=writer, reason="terminal")
            if tail_ack.get("checkpoint") is not None:
                trace.add_numerical_ref(tail_ack["checkpoint"])
            if tail_ack.get("committed") is True and isinstance(tail_ack.get("latest_generation"), int):
                trace.complete_generation(tail_ack["latest_generation"])
                committed_generations = tail_ack["latest_generation"]
            if tail_ack.get("committed") is True:
                trace.publish_diagnostics()
            notify_committed_generation(tail_ack)
            update_progress("running", generation=completed_generations,
                            evaluated=evaluated_count, committed=committed_generations)
        update_progress("saving", generation=terminal.get("completed_generations", completed_generations),
                        evaluated=evaluated_count, committed=committed_generations)
        with trace.span("result_publish"):
            with binding.writer():
                result_ref, result_ack = storage.complete_operation(
                    binding, writer=writer, terminal_bytes=record_bytes(terminal),
                )
                completed = True
        trace.add_numerical_ref(result_ref)
        if optimization:
            _notify(on_progress, request, phase="complete", generation=terminal["completed_generations"], best_cost=best_cost, trace=trace)
            update_progress("decoding", generation=terminal["completed_generations"],
                            evaluated=1 + terminal["completed_generations"] * population_size,
                            best_cost=best_cost, committed=terminal["completed_generations"])
        with trace.span("result_decode"):
            with binding.reader():
                result = storage.read_operation_success(
                    binding, task_id, attempt_id=attempt_id,
                    projection_consumer=lambda success: _decode(
                        success, decoder=decoder, prepared_analysis=prepared_analysis, bound_spec=bound_spec,
                    ),
                )
        if optimization:
            completed_generations = terminal["completed_generations"]
            committed_generations = terminal["completed_generations"]
            evaluated_count = 1 + completed_generations * population_size
            update_progress("complete", best_cost=best_cost, committed=committed_generations)
        return result
    except BaseException as error:
        failure_phase = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"

        def update_failure_display():
            if not optimization or progress_observer is None:
                return
            try:
                update_progress(failure_phase)
            except BaseException as observer_error:
                error.add_note(
                    f"Progress observer during failure finalization also failed: "
                    f"{type(observer_error).__name__}: {observer_error}"
                )

        if optimization and progress_observer is not None:
            update_failure_display()
        if attempt_id is not None and not completed:
            try:
                if not registration_committed and registration_attempted:
                    outcome = getattr(error, "operation_registration_outcome", None)
                    if outcome is None:
                        outcome = getattr(error, "operation_transaction_outcome", None)
                    registration_committed = isinstance(outcome, dict) and outcome.get("status") == "committed"
                    if (not registration_committed and isinstance(outcome, dict)
                            and outcome.get("status") == "unknown"):
                        error.add_note(
                            "Execution registration outcome is unknown; no task failure update was retried."
                        )
                    if not registration_committed:
                        # A failed or indeterminate registration has no
                        # durable task/attempt association to expose on the
                        # operation root.
                        trace.bind(operation=request["operation"], request_sha256=request_sha,
                                   task_id=None, environment_sha256=None, attempt_id=None,
                                   details=details)
                    else:
                        committed_registration = getattr(error, "operation_registration_result", None)
                        if isinstance(committed_registration, dict):
                            details["request"] = committed_registration["request_ref"]
                            attempt_binding = committed_registration["attempt_binding"]
                            trace.bind(operation=request["operation"], request_sha256=request_sha,
                                       task_id=attempt_binding["task_id"],
                                       environment_sha256=attempt_binding["environment_sha256"],
                                       attempt_id=attempt_binding["attempt_id"], details=details)
                if registration_committed and writer is None:
                    try:
                        with binding.writer():
                            writer = storage.begin_task_writer(
                                workspace, binding=binding, operation_id=trace.operation_id,
                                task_id=task_id, attempt_id=attempt_id, diagnostics="boundary",
                                checkpoint_document=checkpoint,
                                commit_every_generations=commit_every_generations,
                                spool=spool,
                                phase_scope=lambda kind, details: trace.span(kind, details=details),
                            )
                    except BaseException as writer_setup_error:
                        error.add_note(
                            "Task writer initialization also failed after execution registration: "
                            f"{type(writer_setup_error).__name__}: {writer_setup_error}"
                        )
                if registration_committed and writer is not None and writer.completed:
                    with binding.writer():
                        tail_ack = storage.flush_completed(workspace, writer=writer, reason="failure")
                    committed_tail_generation = tail_ack.get("latest_generation")
                    tail_was_committed = (
                        tail_ack.get("committed") is True
                        and isinstance(committed_tail_generation, int)
                        and not isinstance(committed_tail_generation, bool)
                    )
                    if tail_was_committed:
                        committed_generations = committed_tail_generation
                        update_failure_display()
                    if tail_ack.get("checkpoint") is not None:
                        trace.add_numerical_ref(tail_ack["checkpoint"])
                    if tail_was_committed:
                        trace.complete_generation(committed_tail_generation)
                    try:
                        notify_committed_generation(tail_ack)
                    except BaseException as callback_error:
                        error.add_note(
                            f"Generation callback during failure flush also failed: "
                            f"{type(callback_error).__name__}: {callback_error}"
                        )
                if registration_committed:
                    failure = storage.error_document(error)
                    interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
                    if writer is not None:
                        with binding.writer():
                            storage.append_event(workspace, task_id=task_id,
                                                 kind="interrupted" if interrupted else "failed",
                                                 payload={"attempt_id": attempt_id,
                                                          "interruption" if interrupted else "failure": failure},
                                                 writer=writer, force=True)
                            storage.update_attempt(workspace, binding=binding, operation_id=trace.operation_id,
                                                   task_id=task_id, attempt_id=attempt_id,
                                                   status="interrupted" if interrupted else "failure",
                                                   interruption=failure if interrupted else None,
                                                   failure=None if interrupted else failure)
                    else:
                        # A registration ACK is durable even if the TaskWriter
                        # could not be opened. Preserve the real attempt status
                        # without inventing a task event or passing writer=None.
                        with binding.writer():
                            storage.update_attempt(workspace, binding=binding, operation_id=trace.operation_id,
                                                   task_id=task_id, attempt_id=attempt_id,
                                                   status="interrupted" if interrupted else "failure",
                                                   interruption=failure if interrupted else None,
                                                   failure=None if interrupted else failure)
            except BaseException as recording_error:
                error.add_note(f"Operation failure recording also failed: {type(recording_error).__name__}: {recording_error}")
        raise
    finally:
        primary = sys.exception()
        cleanup_error = None
        try:
            if backend is not None:
                backend.close()
        except BaseException as backend_error:
            if primary is None:
                cleanup_error = backend_error
            else:
                primary.add_note(
                    f"Operation finalization also failed: {type(backend_error).__name__}: {backend_error}"
                )
        try:
            if spool is not None:
                spool.close()
        except BaseException as scratch_error:
            if primary is None:
                if cleanup_error is None:
                    cleanup_error = scratch_error
                else:
                    cleanup_error.add_note(
                        f"Operation scratch cleanup also failed: {type(scratch_error).__name__}: {scratch_error}"
                    )
            else:
                primary.add_note(
                    f"Operation scratch cleanup also failed: {type(scratch_error).__name__}: {scratch_error}"
                )
        if primary is None and cleanup_error is not None:
            raise cleanup_error
