"""Coordinate one request under its caller-held operation lease.

Workspace reader and writer scopes protect short evidence reads and
publications. Native execution, user callbacks, and the verified-success yield
remain outside those scopes; launch and checkpoint acknowledgements stay in
this publication state machine.
"""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from hashlib import sha256
from ..canonical import canonical_json_bytes, float64_from_hex, sha256_hex
from ..workspace.documents import canonical_receipt_document
from ..workspace.artifacts import _error_from_record, _validated_failure_record
from ..workspace import (
    BaselineCheckpoint,
    VerifiedSuccess,
    WorkspaceBinding,
    _IncomingCheckpointEvidenceError,
    verified_generation_links,
)
from ..errors import BackendProtocolError, EvidenceIntegrityError, OptimizationProgressCallbackError
from ..specs import OptimizationProgress
from .preparation import prepare_runtime
from .prepared import PreparedAnalysis
from .process import run_terminal
from .protocol import BootstrapReady
from .receipts import _attempt_document, _failure_record, _receipt, _utc_now
from .staging import (
    _discard_untrusted_outputs,
    _validate_success_staging,
    _validate_terminal_staging_layout,
    _write_logs,
)
from ..workspace.operation_lease import request_execution_lease
from ..workspace.storage import _load_canonical
from ..workspace.validation.requests import _is_current_native_optimization_request


def _bound_plan_document(binding: WorkspaceBinding) -> Mapping[str, object]:
    """Read the exact canonical Plan bound to this Workspace leaf."""

    with binding.reader():
        # reader() verifies the current root/leaf instance and the Plan digest
        # before this detached canonical document is loaded.
        return _load_canonical(binding.leaf / "plan.json")


@contextmanager
def execute_prepared(
    *,
    binding: WorkspaceBinding,
    prepared_analysis: PreparedAnalysis,
    operation_id: str,
    operation_lease,
    native_supervisor,
    on_progress: Callable[[OptimizationProgress], object] | None = None,
    _timing_observer: Callable[[str, int, int, Mapping[str, object]], object] | None = None,
) -> Iterator[VerifiedSuccess]:
    """Serialize an exact request while keeping Workspace writes short."""

    if native_supervisor is None:
        raise RuntimeError("execute_prepared requires its registered operation's native supervisor")
    with request_execution_lease(binding, prepared_analysis.request_sha256) as request_lease:
        with _execute_prepared(
            binding=binding,
            prepared_analysis=prepared_analysis,
            operation_id=operation_id,
            operation_lease=operation_lease,
            request_lease=request_lease,
            native_supervisor=native_supervisor,
            on_progress=on_progress,
            _timing_observer=_timing_observer,
        ) as success:
            yield success


@contextmanager
def _execute_prepared(
    *,
    binding: WorkspaceBinding,
    prepared_analysis: PreparedAnalysis,
    operation_id: str,
    operation_lease,
    request_lease,
    native_supervisor,
    on_progress: Callable[[OptimizationProgress], object] | None = None,
    _timing_observer: Callable[[str, int, int, Mapping[str, object]], object] | None = None,
) -> Iterator[VerifiedSuccess]:
    """Execute under the caller's operation lease and short Workspace writes."""

    request_bytes = prepared_analysis.request_bytes
    request = prepared_analysis.request()
    source_units = prepared_analysis.source_units()
    request_sha = prepared_analysis.request_sha256
    def deliver(frame: Mapping[str, object]) -> None:
        if on_progress is not None:
            on_progress(OptimizationProgress(
                phase=frame["event"], reused=frame["reused"],
                completed_generations=frame["completed_generations"],
                total_generations=frame["total_generations"],
                evaluated_count=frame["evaluated_count"],
                requested_budget=frame["requested_budget"],
                achievable_evaluations=frame["achievable_evaluations"],
                best_cost=float64_from_hex(frame["best_cost_f64"]),
            ))

    def report_reuse(success: VerifiedSuccess) -> None:
        if on_progress is None:
            return
        controls = request["spec"]["optimizer"]
        generations = controls["complete_generations"]
        try:
            deliver({"event": "result_reuse", "reused": True,
                "completed_generations": generations, "total_generations": generations,
                "evaluated_count": 1 + generations * controls["resolved_population_size"],
                "requested_budget": controls["max_evaluations"],
                "achievable_evaluations": 1 + generations * controls["resolved_population_size"],
                "best_cost_f64": success.result["best"]["cost_f64"]})
        except KeyboardInterrupt:
            raise
        except Exception as error:
            raise OptimizationProgressCallbackError("optimization progress callback failed on result reuse",
                stage="progress_callback", evidence={"error_type": type(error).__name__}) from error

    with binding.reader():
        success = binding.find_success(request_sha)
    if success is not None:
        report_reuse(success)
        yield success
        return
    native_supervisor.retain_lease(request_lease.descriptor)
    prepared_runtime = prepare_runtime(
        feature=str(request["operation"]),
        native_supervisor=native_supervisor,
    )
    executable_sha = sha256(prepared_runtime.executable.read_bytes()).hexdigest()
    started = _utc_now()
    with binding.writer():
        success = binding.find_success(request_sha)
        if success is None:
            request_directory = binding.ensure_request(request_sha, request_bytes)
            checkpoint = binding.baseline_checkpoint(request_sha)
            point_checkpoints = (binding.point_checkpoints(request_sha)
                if request["parameter_source"]["kind"] in {"grid", "points"} else ())
            resume_ledger_sha = binding.resume_ledger_sha256(request_sha)
            allocation = binding.allocate_attempt(
                request_sha, operation_id=operation_id, operation_lease=operation_lease,
            )
    if success is not None:
        report_reuse(success)
        yield success
        return
    attempt_sha: str | None = None

    def promote(receipt: Mapping[str, object]) -> None:
        with binding.writer(cleanup_staging=False):
            if (
                receipt.get("outcome") == "success"
                and _is_current_native_optimization_request(request)
            ):
                binding.seal_native_result_index(allocation, receipt)
            if request["parameter_source"]["kind"] in {"grid", "points"}:
                receipt = canonical_receipt_document({**receipt,
                    "point_checkpoint_count": len(binding.point_checkpoints(request_sha))})
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
            try:
                binding.promote_attempt(allocation, receipt)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)

    def seal_protocol_failure(
        error: BackendProtocolError | OptimizationProgressCallbackError,
        *,
        stdout: Sequence[str] = (),
        stderr: Sequence[str] = (),
    ) -> None:
        nonlocal attempt_sha
        if attempt_sha is None:
            with binding.writer(cleanup_staging=False):
                attempt_sha = binding.seal_attempt(
                    allocation,
                    _attempt_document(
                        allocation,
                        started=started,
                        executable_sha=executable_sha,
                        state="allocated",
                        resume_ledger_sha=resume_ledger_sha,
                        optimization=request.get("operation") == "optimize_direct",
                        checkpoint=checkpoint,
                    ),
                )
        _write_logs(allocation.staging_directory, stdout, (*stderr, str(error)))
        _discard_untrusted_outputs(allocation.staging_directory)
        promote(
            _receipt(
                request=request,
                plan_document=_bound_plan_document(binding),
                request_sha=request_sha,
                attempt_sha=attempt_sha,
                outcome="failure",
                artifacts=[],
                source_units=source_units,
                failure=_failure_record(
                    error, request["operation"], request_sha, attempt_sha
                ),
            )
        )

    def seal_interruption(error: KeyboardInterrupt) -> None:
        nonlocal attempt_sha
        if allocation.final_directory.exists():
            return
        if attempt_sha is None:
            with binding.writer(cleanup_staging=False):
                attempt_sha = binding.seal_attempt(
                    allocation,
                    _attempt_document(
                        allocation,
                        started=started,
                        executable_sha=executable_sha,
                        state="allocated",
                        resume_ledger_sha=resume_ledger_sha,
                        optimization=request.get("operation") == "optimize_direct",
                        checkpoint=checkpoint,
                    ),
                )
        _write_logs(allocation.staging_directory, (), ())
        artifacts = verified_generation_links(
            allocation.staging_directory,
            request_path=request_directory / "request.json",
            request_sha256=request_sha,
            attempt_sha256=attempt_sha,
            allow_other_artifacts=True,
        )
        _discard_untrusted_outputs(allocation.staging_directory, keep_ledgers=True)
        promote(
            _receipt(
                request=request,
                plan_document=_bound_plan_document(binding),
                request_sha=request_sha,
                attempt_sha=attempt_sha,
                outcome="interrupted",
                artifacts=artifacts,
                source_units=source_units,
                interruption={
                    "kind": "keyboard_interrupt",
                    "termination": getattr(error, "termination", "terminated"),
                    "interrupted_at_utc": _utc_now(),
                },
            )
        )

    def seal_without_replacing_primary(primary: BaseException, action, label: str) -> None:
        try:
            action()
        except BaseException as secondary:
            primary.add_note(
                f"{label} while recording the primary failure also failed: "
                f"{type(secondary).__name__}: {secondary}"
            )

    def authorize(ready: BootstrapReady) -> str:
        nonlocal attempt_sha
        with binding.writer(cleanup_staging=False):
            attempt_sha = binding.seal_attempt(
                allocation,
                _attempt_document(
                    allocation,
                    started=started,
                    executable_sha=executable_sha,
                    state="launched",
                    ready=ready,
                    resume_ledger_sha=resume_ledger_sha,
                    optimization=request.get("operation") == "optimize_direct",
                    checkpoint=checkpoint,
                ),
            )
        return attempt_sha

    def publish_checkpoint(ready: Mapping[str, object]) -> Mapping[str, object]:
        nonlocal checkpoint
        assert attempt_sha is not None
        with binding.writer(cleanup_staging=False):
            published = binding.publish_baseline_checkpoint(
                request_sha,
                attempt_sha,
                allocation.staging_directory / "baseline-checkpoint.json",
                expected_sha256=str(ready["checkpoint_sha256"]),
                expected_byte_length=int(ready["byte_length"]),
            )
        checkpoint = published
        return {
            "checkpoint_sha256": published.checkpoint_sha256,
            "seal_sha256": published.seal_sha256,
        }

    def publish_point(ready: Mapping[str, object]) -> Mapping[str, object]:
        assert attempt_sha is not None
        try:
            with binding.writer(cleanup_staging=False):
                published = binding.publish_point_checkpoint(request_sha, attempt_sha,
                    allocation.staging_directory, ready)
        except (EvidenceIntegrityError, OSError) as error:
            raise BackendProtocolError("child point checkpoint failed independent validation",
                stage="point_checkpoint") from error
        return {"record_sha256": ready["record_sha256"],
                "seal_sha256": published.seal_sha256}

    try:
        checkpoint_control, publisher_control = _optimization_checkpoint_controls(
            request.get("operation"), checkpoint, publish_checkpoint
        )
        terminal = run_terminal(
            prepared_runtime,
            native_supervisor=native_supervisor,
            _timing_observer=_timing_observer,
            request_path=(request_directory / "request.json").resolve(),
            staging_directory=allocation.staging_directory.resolve(),
            request_sha256=request_sha,
            attempt_ordinal=allocation.ordinal,
            authorize=authorize,
            checkpoint=checkpoint_control,
            publish_checkpoint=publisher_control,
            on_progress=deliver if on_progress is not None else None,
            point_recovery=tuple({"ordinal": index, "seal_sha256": item.seal_sha256}
                for index, item in enumerate(point_checkpoints)) if request["parameter_source"]["kind"] in {"grid", "points"} else None,
            publish_point=publish_point if request["parameter_source"]["kind"] in {"grid", "points"} else None,
        )
    except KeyboardInterrupt as error:
        seal_without_replacing_primary(
            error, lambda: seal_interruption(error), "Interruption evidence finalization"
        )
        raise
    except _IncomingCheckpointEvidenceError as error:
        protocol = BackendProtocolError(
            "child baseline checkpoint failed independent validation",
            stage="optimization_checkpoint",
        )
        seal_without_replacing_primary(
            protocol, lambda: seal_protocol_failure(protocol), "Checkpoint failure finalization"
        )
        raise protocol from error
    except BackendProtocolError as error:
        seal_without_replacing_primary(
            error, lambda: seal_protocol_failure(error), "Protocol failure finalization"
        )
        raise
    except OptimizationProgressCallbackError as error:
        seal_without_replacing_primary(
            error, lambda: seal_protocol_failure(error), "Progress callback failure finalization"
        )
        raise

    assert attempt_sha is not None
    try:
        _write_logs(
            allocation.staging_directory,
            terminal.stdout_log,
            terminal.stderr_log,
        )
        outcome_path = allocation.staging_directory / "outcome.json"
        if (
            allocation.staging_directory.is_symlink()
            or not allocation.staging_directory.is_dir()
            or outcome_path.is_symlink()
            or not outcome_path.is_file()
        ):
            raise BackendProtocolError(
                "outcome.json is not a regular file", stage="outcome"
            )
        outcome_raw = outcome_path.read_bytes()
        outcome = terminal.outcome
        if canonical_json_bytes(outcome) != outcome_raw:
            raise BackendProtocolError(
                "outcome.json is not canonical", stage="outcome"
            )
        if (
            outcome.get("runtime_semantic") != request.get("runtime_semantic")
            or outcome.get("request_sha256") != request_sha
            or outcome.get("attempt_sha256") != attempt_sha
            or outcome.get("status") not in {"success", "failure"}
            or not isinstance(outcome.get("artifacts"), list)
        ):
            raise BackendProtocolError(
                "outcome envelope does not bind this execution",
                stage="outcome",
            )
        expected_fields = {
            "schema",
            "schema_version",
            "request_sha256",
            "attempt_sha256",
            "runtime_semantic",
            "status",
            "artifacts",
            "result_sha256" if outcome["status"] == "success" else "failure",
        }
        if set(outcome) != expected_fields:
            raise BackendProtocolError(
                "outcome envelope has unsupported fields", stage="outcome"
            )
        _validate_terminal_staging_layout(
            allocation.staging_directory,
            success=outcome["status"] == "success",
        )
        artifacts = list(outcome["artifacts"])
        outcome_sha = sha256_hex(outcome_raw)
        bound_plan_document = _bound_plan_document(binding)
        if outcome["status"] == "success":
            _validate_success_staging(
                allocation.staging_directory,
                outcome,
                request,
                bound_plan_document,
                request_path=request_directory / "request.json",
                optimization_checkpoint=checkpoint,
            )
            receipt = _receipt(
                request=request,
                plan_document=bound_plan_document,
                request_sha=request_sha,
                attempt_sha=attempt_sha,
                outcome="success",
                artifacts=artifacts,
                source_units=source_units,
                outcome_sha=outcome_sha,
                result_sha=outcome["result_sha256"],
            )
            failure = None
        else:
            result_path = allocation.staging_directory / "result.json"
            if result_path.exists() or result_path.is_symlink():
                raise BackendProtocolError(
                    "failure outcome must not publish result.json",
                    stage="outcome",
                )
            verified_links = verified_generation_links(
                allocation.staging_directory,
                request_path=request_directory / "request.json",
                request_sha256=request_sha,
                attempt_sha256=attempt_sha,
            )
            if artifacts != verified_links:
                raise BackendProtocolError(
                    "failure outcome does not exactly bind completed generation ledgers",
                    stage="outcome",
                )
            failure = _validated_failure_record(
                outcome.get("failure"), request["operation"],
                request=request, plan=bound_plan_document,
                require_optimization_context=True,
                completed_generations=len(verified_links),
            )
            receipt = _receipt(
                request=request,
                plan_document=bound_plan_document,
                request_sha=request_sha,
                attempt_sha=attempt_sha,
                outcome="failure",
                artifacts=artifacts,
                source_units=source_units,
                outcome_sha=outcome_sha,
                failure=failure,
            )
    except BackendProtocolError as error:
        seal_without_replacing_primary(
            error,
            lambda: seal_protocol_failure(
                error,
                stdout=terminal.stdout_log,
                stderr=terminal.stderr_log,
            ),
            "Protocol failure finalization",
        )
        raise
    except KeyboardInterrupt as error:
        seal_without_replacing_primary(
            error, lambda: seal_interruption(error), "Interruption evidence finalization"
        )
        raise
    except Exception as error:
        protocol = BackendProtocolError(
            "Julia terminal evidence failed closed validation",
            stage="outcome",
            evidence={"error": str(error)},
        )
        seal_without_replacing_primary(
            protocol,
            lambda: seal_protocol_failure(
                protocol,
                stdout=terminal.stdout_log,
                stderr=terminal.stderr_log,
            ),
            "Terminal evidence failure finalization",
        )
        raise protocol from error
    promote(receipt)
    if failure is None:
        with binding.reader():
            success = binding.resolve_success(request_sha)
        yield success
        return
    raise _error_from_record(failure)


def _optimization_checkpoint_controls(
    operation: object,
    checkpoint: BaselineCheckpoint | None,
    publisher: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> tuple[
    Mapping[str, object] | None,
    Callable[[Mapping[str, object]], Mapping[str, object]] | None,
]:
    """Expose checkpoint controls only to the optimization terminal protocol."""

    if operation != "optimize_direct":
        return None, None
    return (
        None
        if checkpoint is None
        else {
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "seal_sha256": checkpoint.seal_sha256,
        },
        publisher,
    )
