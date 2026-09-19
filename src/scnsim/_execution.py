"""Workspace/process coordinator for one already-prepared analysis request."""

from __future__ import annotations

import json
import platform
import shutil
import signal
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from ._analysis import PreparedAnalysis
from ._backend import BootstrapReady, prepare_runtime, run_terminal
from ._canonical import (
    canonical_json_bytes,
    canonical_receipt_document,
    sha256_hex,
    zarr_artifact_manifest,
)
from ._evidence import (
    _direct_request_frequencies,
    _error_from_record,
    _read_zarr,
    _validate_direct_values,
    _validated_failure_record,
    _verify_zarr_catalog_metadata,
)
from ._workspace import (
    AttemptAllocation,
    BaselineCheckpoint,
    VerifiedSuccess,
    WorkspaceBinding,
    _IncomingCheckpointEvidenceError,
    _inside,
    _required_extrapolation_rows,
    _verify_artifact_inventory,
    _verify_generation_artifacts,
    _verify_result_document,
    verified_generation_links,
)
from .errors import (
    BackendProtocolError,
    CompilerInvariantError,
    EvidenceIntegrityError,
    SCNSimError,
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@contextmanager
def execute_prepared(
    *,
    binding: WorkspaceBinding,
    plan_document: Mapping[str, object],
    prepared_analysis: PreparedAnalysis,
) -> Iterator[VerifiedSuccess]:
    """Yield verified success while its workspace ownership lock remains held."""

    request_bytes = prepared_analysis.request_bytes
    request = prepared_analysis.request()
    source_units = prepared_analysis.source_units()
    request_sha = prepared_analysis.request_sha256
    with binding.reader():
        success = binding.find_success(request_sha)
        if success is not None:
            yield success
            return
    prepared_runtime = prepare_runtime()
    executable_sha = sha256(prepared_runtime.executable.read_bytes()).hexdigest()
    started = _utc_now()
    with binding.writer():
        success = binding.find_success(request_sha)
        if success is not None:
            yield success
            return
        request_directory = binding.ensure_request(request_sha, request_bytes)
        checkpoint = binding.baseline_checkpoint(request_sha)
        resume_ledger_sha = binding.resume_ledger_sha256(request_sha)
        allocation = binding.allocate_attempt(request_sha)
        attempt_sha: str | None = None

        def promote(receipt: Mapping[str, object]) -> None:
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
            try:
                binding.promote_attempt(allocation, receipt)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)

        def seal_protocol_failure(
            error: BackendProtocolError,
            *,
            stdout: Sequence[str] = (),
            stderr: Sequence[str] = (),
        ) -> None:
            nonlocal attempt_sha
            if attempt_sha is None:
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
                    plan_document=plan_document,
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
                request_sha256=request_sha,
                attempt_sha256=attempt_sha,
                allow_other_artifacts=True,
            )
            _discard_untrusted_outputs(allocation.staging_directory, keep_ledgers=True)
            promote(
                _receipt(
                    request=request,
                    plan_document=plan_document,
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

        def authorize(ready: BootstrapReady) -> str:
            nonlocal attempt_sha
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

        try:
            checkpoint_control, publisher_control = _optimization_checkpoint_controls(
                request.get("operation"), checkpoint, publish_checkpoint
            )
            terminal = run_terminal(
                prepared_runtime,
                request_path=(request_directory / "request.json").resolve(),
                staging_directory=allocation.staging_directory.resolve(),
                request_sha256=request_sha,
                attempt_ordinal=allocation.ordinal,
                authorize=authorize,
                checkpoint=checkpoint_control,
                publish_checkpoint=publisher_control,
            )
        except KeyboardInterrupt as error:
            seal_interruption(error)
            raise
        except _IncomingCheckpointEvidenceError as error:
            protocol = BackendProtocolError(
                "child baseline checkpoint failed independent validation",
                stage="optimization_checkpoint",
            )
            seal_protocol_failure(protocol)
            raise protocol from error
        except BackendProtocolError as error:
            seal_protocol_failure(error)
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
            if outcome["status"] == "success":
                _validate_success_staging(
                    allocation.staging_directory,
                    outcome,
                    request,
                    plan_document,
                    optimization_checkpoint=checkpoint,
                )
                receipt = _receipt(
                    request=request,
                    plan_document=plan_document,
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
                    request_sha256=request_sha,
                    attempt_sha256=attempt_sha,
                )
                if artifacts != verified_links:
                    raise BackendProtocolError(
                        "failure outcome does not exactly bind completed generation ledgers",
                        stage="outcome",
                    )
                failure = _validated_failure_record(
                    outcome.get("failure"), request["operation"], request=request,
                    require_optimization_context=True,
                    completed_generations=len(verified_links),
                )
                receipt = _receipt(
                    request=request,
                    plan_document=plan_document,
                    request_sha=request_sha,
                    attempt_sha=attempt_sha,
                    outcome="failure",
                    artifacts=artifacts,
                    source_units=source_units,
                    outcome_sha=outcome_sha,
                    failure=failure,
                )
        except BackendProtocolError as error:
            seal_protocol_failure(
                error,
                stdout=terminal.stdout_log,
                stderr=terminal.stderr_log,
            )
            raise
        except KeyboardInterrupt as error:
            seal_interruption(error)
            raise
        except Exception as error:
            protocol = BackendProtocolError(
                "Julia terminal evidence failed closed validation",
                stage="outcome",
                evidence={"error": str(error)},
            )
            seal_protocol_failure(
                protocol,
                stdout=terminal.stdout_log,
                stderr=terminal.stderr_log,
            )
            raise protocol from error
        promote(receipt)
        if failure is None:
            yield binding.resolve_success(request_sha)
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


def _attempt_document(
    allocation: AttemptAllocation,
    *,
    started: str,
    executable_sha: str,
    state: str,
    ready: BootstrapReady | None = None,
    resume_ledger_sha: str | None = None,
    optimization: bool = False,
    checkpoint: BaselineCheckpoint | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": "scnsim.attempt",
        "schema_version": 2 if optimization else 1,
        "request_sha256": allocation.request_sha256,
        "ordinal": allocation.ordinal,
        "ordinal_text": allocation.ordinal_text,
        "directory": allocation.attempt_directory_text,
        "staging_directory": allocation.staging_directory_text,
        "attempt_state": state,
        "started_at_utc": started,
        "julia_executable_sha256": executable_sha,
        "os": platform.system(),
        "architecture": platform.machine() or "unknown",
        "cpu": platform.processor() or "unknown",
    }
    if ready is not None:
        document.update(
            {
                "julia_threads": ready.julia_threads,
                "blas_threads": ready.blas_threads,
                "blas_vendor": ready.blas_vendor,
            }
        )
        fftw_threads = getattr(ready, "fftw_threads", None)
        if fftw_threads is not None:
            document["fftw_threads"] = fftw_threads
    if resume_ledger_sha is not None:
        document["resume_ledger_sha256"] = resume_ledger_sha
    if checkpoint is not None:
        document["baseline_checkpoint_sha256"] = checkpoint.checkpoint_sha256
        document["baseline_checkpoint_seal_sha256"] = checkpoint.seal_sha256
    return document


def _failure_record(
    error: SCNSimError, operation: object, request_sha: str, attempt_sha: str
) -> dict[str, object]:
    return {
        "category": error.category,
        "kind": error.kind,
        "stage": error.stage,
        "message": str(error),
        "evidence": {
            "type": "failure_evidence",
            "operation": operation
            if operation
            in {"solve_direct", "solve_hb", "evaluate_direct", "optimize_direct"}
            else "backend_protocol",
            "context_kind": "protocol",
            "request_sha256": request_sha,
            "attempt_sha256": attempt_sha,
        },
    }


def _receipt(
    *,
    request: Mapping[str, object],
    plan_document: Mapping[str, object],
    request_sha: str,
    attempt_sha: str,
    outcome: str,
    artifacts: Sequence[object],
    source_units: Sequence[Mapping[str, object]],
    outcome_sha: str | None = None,
    result_sha: object | None = None,
    failure: Mapping[str, object] | None = None,
    interruption: Mapping[str, object] | None = None,
) -> dict[str, object]:
    runtime_sha = sha256_hex(request["runtime_semantic"])
    provenance = sha256_hex(
        {"schema": "scnsim.receipt_provenance", "source_units": list(source_units)}
    )
    evidence: dict[str, object] = {
        "runtime_semantic_sha256": runtime_sha,
        "source_units": list(source_units),
        "extrapolation_evidence": _receipt_extrapolation_evidence(
            request, plan_document, require_authorized=outcome == "success"
        ),
        "provenance_sha256": provenance,
    }
    evidence["evidence_sha256"] = sha256_hex(evidence)
    document: dict[str, object] = {
        "request_sha256": request_sha,
        "attempt_sha256": attempt_sha,
        "outcome": outcome,
        "artifacts": list(artifacts),
        "evidence": evidence,
        "sealed_at_utc": _utc_now(),
    }
    if outcome_sha is not None:
        document["outcome_sha256"] = outcome_sha
    if result_sha is not None:
        document["result_sha256"] = result_sha
    if failure is not None:
        document["failure"] = dict(failure)
    if interruption is not None:
        document["interruption"] = dict(interruption)
    return canonical_receipt_document(document)


def _receipt_extrapolation_evidence(
    request: Mapping[str, object],
    plan_document: Mapping[str, object],
    *,
    require_authorized: bool,
) -> list[dict[str, object]]:
    """Project receipt evidence through the workspace's closed fan-out verifier."""
    if request.get("operation") == "optimize_direct":
        return []
    source = request.get("parameter_source")
    if isinstance(source, Mapping) and source.get("kind") in {"grid", "points"}:
        # Point-local authorization is verified against every chunk entry;
        # the request receipt must not pretend an ordered space is one point.
        return []
    parameters = source.get("parameters") if isinstance(source, Mapping) else None
    if not isinstance(parameters, Mapping):
        raise CompilerInvariantError(
            "receipt request has no ParameterSet", stage="receipt"
        )
    return _required_extrapolation_rows(
        plan_document,
        parameters,
        authorization_source="parameter_set",
        require_authorized=require_authorized,
    )


def _require_staging_directory(staging: Path) -> None:
    if staging.parent.is_symlink() or staging.is_symlink() or not staging.is_dir():
        raise EvidenceIntegrityError(
            "attempt staging is not a regular directory",
            stage="workspace",
            evidence={"path": str(staging)},
        )


def _remove_untrusted(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _write_logs(staging: Path, stdout: Sequence[str], stderr: Sequence[str]) -> None:
    _require_staging_directory(staging)
    directory = staging / "logs"
    if directory.exists() or directory.is_symlink():
        _remove_untrusted(directory)
    if not stdout and not stderr:
        return
    directory.mkdir()
    if stdout:
        path = directory / "stdout.log"
        if path.is_symlink():
            path.unlink()
        path.write_text("".join(stdout), encoding="utf-8")
    if stderr:
        path = directory / "stderr.log"
        if path.is_symlink():
            path.unlink()
        path.write_text("".join(stderr), encoding="utf-8")


def _discard_untrusted_outputs(staging: Path, *, keep_ledgers: bool = False) -> None:
    _require_staging_directory(staging)
    outcome = staging / "outcome.json"
    if outcome.exists() or outcome.is_symlink():
        logs = staging / "logs"
        if logs.is_symlink() or logs.exists() and not logs.is_dir():
            _remove_untrusted(logs)
        logs.mkdir(exist_ok=True)
        destination = logs / "untrusted-outcome.json"
        if destination.exists() or destination.is_symlink():
            _remove_untrusted(destination)
        if outcome.is_symlink():
            outcome.unlink()
        elif outcome.is_file():
            shutil.move(outcome, destination)
        else:
            _remove_untrusted(outcome)
    result = staging / "result.json"
    if result.exists() or result.is_symlink():
        _remove_untrusted(result)
    artifacts = staging / "artifacts"
    if artifacts.exists() or artifacts.is_symlink():
        if artifacts.is_symlink():
            artifacts.unlink()
        else:
            generations = artifacts / "generations"
            if keep_ledgers and not generations.is_symlink() and generations.is_dir():
                for child in artifacts.iterdir():
                    if child != generations:
                        _remove_untrusted(child)
                if not any(generations.iterdir()):
                    generations.rmdir()
                    artifacts.rmdir()
            else:
                _remove_untrusted(artifacts)
    allowed = {"attempt.json", "logs"}
    if keep_ledgers and (staging / "artifacts").is_dir():
        allowed.add("artifacts")
    for child in staging.iterdir():
        if child.name not in allowed:
            _remove_untrusted(child)


def _validate_terminal_staging_layout(staging: Path, *, success: bool) -> None:
    _require_staging_directory(staging)
    allowed = {"attempt.json", "logs", "outcome.json", "artifacts"}
    if success:
        allowed.add("result.json")
    unexpected = sorted(
        child.name for child in staging.iterdir() if child.name not in allowed
    )
    if unexpected:
        raise BackendProtocolError(
            "terminal staging contains unsupported entries",
            stage="outcome",
            evidence={"entries": unexpected},
        )


def _validate_success_staging(
    staging: Path,
    outcome: Mapping[str, object],
    request: Mapping[str, object],
    plan: Mapping[str, object],
    *,
    optimization_checkpoint: BaselineCheckpoint | None = None,
) -> None:
    if outcome.get("runtime_semantic") != request.get("runtime_semantic"):
        raise BackendProtocolError(
            "outcome runtime identity does not match the request", stage="outcome"
        )
    result_path = _inside(staging, "result.json")
    if (
        result_path.is_symlink()
        or not result_path.is_file()
        or sha256(result_path.read_bytes()).hexdigest() != outcome.get("result_sha256")
    ):
        raise BackendProtocolError(
            "success outcome does not bind result.json", stage="outcome"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if canonical_json_bytes(result) != result_path.read_bytes():
        raise BackendProtocolError("result.json is not canonical", stage="outcome")
    parameter_source = request.get("parameter_source")
    is_parameter_sweep = isinstance(parameter_source, Mapping) and parameter_source.get(
        "kind"
    ) in {"grid", "points"}
    expected_kind = (
        "parameter_sweep"
        if is_parameter_sweep
        else "direct_response"
        if request.get("operation") == "solve_direct"
        else "hb_batch"
        if request.get("operation") == "solve_hb"
        else "optimization"
        if request.get("operation") == "optimize_direct"
        else request.get("spec", {}).get("type")
        if request.get("operation") == "evaluate_direct"
        and isinstance(request.get("spec"), Mapping)
        else None
    )
    expected_result_fields = (
        {
            "schema",
            "schema_version",
            "result_kind",
            "request_sha256",
            "attempt_sha256",
            "parameter_source_sha256",
            "point_count",
            "chunk_size",
            "manifest",
            "chunks",
        }
        if expected_kind == "parameter_sweep"
        else {
            "schema",
            "schema_version",
            "result_kind",
            "request_sha256",
            "attempt_sha256",
            "parameters",
            "parameters_sha256",
            "ref_lineage",
            "scalar_catalog",
            "array_catalog",
        }
        if expected_kind
        in {
            "direct_response",
            "diagonal_root",
            "operator_element_root",
            "hybridized_pole",
            "transfer_zero",
            "residue_normalized_coupling",
            "response_element",
            "operator",
        }
        else {
            "schema",
            "schema_version",
            "result_kind",
            "request_sha256",
            "attempt_sha256",
            "parameters",
            "parameters_sha256",
            "ref_lineage",
            "baseline",
            "best",
            "completed_generations",
            "unused_evaluations",
            "ledger_artifacts",
        }
        if expected_kind == "optimization"
        else {
            "schema",
            "schema_version",
            "result_kind",
            "request_sha256",
            "attempt_sha256",
            "parameters",
            "parameters_sha256",
            "ref_lineage",
            "lattice",
            "truncation",
            "topology_evidence",
            "cases",
        }
        if expected_kind == "hb_batch"
        else None
    )
    if (
        expected_result_fields is None
        or set(result) != expected_result_fields
        or result.get("schema") != "scnsim.result"
        or result.get("schema_version") != 2
        or result.get("result_kind") != expected_kind
        or result.get("request_sha256") != outcome.get("request_sha256")
        or result.get("attempt_sha256") != outcome.get("attempt_sha256")
    ):
        raise BackendProtocolError(
            "result envelope does not match its request and operation", stage="outcome"
        )
    _verify_result_document(
        result,
        request,
        str(outcome.get("request_sha256")),
        str(outcome.get("attempt_sha256")),
        plan,
        optimization_checkpoint=optimization_checkpoint,
    )
    catalogs: list[Mapping[str, object]] = []
    if expected_kind == "parameter_sweep":
        catalogs.append(result["manifest"])
        expected_links = [dict(result["manifest"])]
    elif expected_kind == "hb_batch":
        expected_links: list[dict[str, object]] = []
        cases = result.get("cases")
        if not isinstance(cases, list):
            raise BackendProtocolError(
                "HB result has no ordered case catalog", stage="outcome"
            )
        for case in cases:
            if not isinstance(case, Mapping):
                raise BackendProtocolError(
                    "HB result case catalog is malformed", stage="outcome"
                )
            if case.get("status") == "failure":
                continue
            artifacts = case.get("artifacts")
            traces = case.get("traces")
            case_id = case.get("case_id")
            if (
                not isinstance(case_id, str)
                or not isinstance(artifacts, Mapping)
                or not isinstance(traces, list)
            ):
                raise BackendProtocolError(
                    "HB success artifact catalog is malformed", stage="outcome"
                )
            ordered = [
                artifacts[name]
                for name in (
                    "s",
                    "y",
                    "z",
                    "backend_native_s",
                    "backend_native_z",
                    "states",
                    "effective_source_vectors",
                )
            ]
            ordered.extend(traces)
            for artifact in ordered:
                if not isinstance(artifact, Mapping):
                    raise BackendProtocolError(
                        "HB success artifact catalog is malformed", stage="outcome"
                    )
                catalogs.append(artifact)
                expected_links.append(
                    {
                        "case_id": case_id,
                        "id": artifact.get("id"),
                        "path": artifact.get("path"),
                        "sha256": artifact.get("sha256"),
                    }
                )
        if outcome.get("artifacts") != expected_links:
            raise BackendProtocolError(
                "HB outcome artifact inventory does not match result.json",
                stage="outcome",
            )
        _verify_artifact_inventory(staging, result, {"artifacts": expected_links})
    elif expected_kind != "optimization":
        catalogs.extend(result["array_catalog"].values())
    else:
        catalogs.extend(result["ledger_artifacts"])
    if expected_kind == "parameter_sweep":
        if outcome.get("artifacts") != expected_links:
            raise BackendProtocolError(
                "batch outcome does not bind its manifest", stage="outcome"
            )
        _verify_artifact_inventory(staging, result, {"artifacts": expected_links})
    elif expected_kind != "hb_batch":
        expected_links = [
            {"id": artifact["id"], "sha256": artifact["sha256"]}
            for artifact in catalogs
        ]
        if outcome.get("artifacts") != expected_links or len(
            {item["id"] for item in expected_links}
        ) != len(expected_links):
            raise BackendProtocolError(
                "outcome artifact inventory does not match result.json", stage="outcome"
            )
        _verify_artifact_inventory(staging, result, {"artifacts": expected_links})
    if expected_kind == "optimization":
        _verify_generation_artifacts(
            staging,
            expected_links,
            request_sha256=str(outcome["request_sha256"]),
            attempt_sha256=str(outcome["attempt_sha256"]),
        )
    for artifact in catalogs:
        path = _inside(staging, str(artifact["path"]))
        if artifact.get("media_type") == "application/vnd+zarr-v2":
            manifest_path = _inside(staging, str(artifact["file_manifest"]))
            rebuilt = zarr_artifact_manifest(
                artifact_directory=path,
                artifact_id=artifact["id"],
                artifact_path=artifact["path"],
            )
            if (
                manifest_path.is_symlink()
                or not manifest_path.is_file()
                or canonical_json_bytes(rebuilt) != manifest_path.read_bytes()
                or sha256(manifest_path.read_bytes()).hexdigest() != artifact["sha256"]
            ):
                raise EvidenceIntegrityError(
                    "Zarr manifest does not match exact artifact bytes",
                    stage="artifact_validation",
                )
            _verify_zarr_catalog_metadata(path, artifact, stage="artifact_validation")
        else:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != artifact["byte_length"]
                or sha256(path.read_bytes()).hexdigest() != artifact["sha256"]
            ):
                raise EvidenceIntegrityError(
                    "file artifact does not match its catalog",
                    stage="artifact_validation",
                )
    if expected_kind == "direct_response":
        arrays = result["array_catalog"]
        _validate_direct_values(
            _read_zarr(staging, arrays["frequencies"], complex_values=False),
            _read_zarr(staging, arrays["s"], complex_values=True),
            _read_zarr(staging, arrays["y"], complex_values=True),
            _read_zarr(staging, arrays["z"], complex_values=True),
            expected_frequency=_direct_request_frequencies(request),
            stage="artifact_validation",
        )
