"""Serial task orchestration and durable evidence for benchmark requests."""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping
from typing import cast

from ..canonical import canonical_json_bytes, sha256_hex
from ..errors import BackendProtocolError, EvidenceIntegrityError, OptimizationProgressCallbackError, SCNSimError
from ..execution.preparation import packaged_julia_resources, prepare_runtime
from ..execution.process import run_terminal
from ..execution.protocol import BootstrapReady
from ..execution.receipts import _attempt_document, _failure_record, _receipt, _utc_now
from ..execution.staging import (
    _decode_direct_point_arrays,
    _discard_untrusted_outputs,
    _validate_success_staging,
    _validate_terminal_staging_layout,
    _write_logs,
)
from ..workspace import bind_workspace, verified_generation_links
from ..workspace.documents import canonical_receipt_document
from ..workspace.artifacts import _error_from_record, _validated_failure_record
from ..workspace.store import _IncomingCheckpointEvidenceError
from ..workspace.records import PointCheckpoint
from .backends.jax_backend import JaxBackend
from .backends.julia_backend import JuliaBackend
from .identity import (
    _NativeThreadSpec,
    _native_thread_spec,
    environment_identity,
    environment_snapshot,
    new_attempt_id,
    task_identifier,
    task_request_identity,
)
from .models import BenchmarkResult, BenchmarkSpec, TaskEvent
from .prepared import PreparedBenchmark, array_record, record_bytes, record_document
from .process import cpu_affinity_profile, run_python_child
from . import storage


def create_python_backend(
    *, arm: str, device: str, cpu_threads: int,
    julia_threads: int | None = None,
    julia_blas_threads: int | None = None,
):
    """Construct exactly the named CPU adapter inside its isolated child."""
    if device != "cpu":
        raise NotImplementedError(f"benchmark device {device!r} is unsupported")
    if arm == "python_julia_reuse":
        return JuliaBackend(
            algorithm="reuse", cpu_threads=cpu_threads,
            julia_threads=julia_threads, julia_blas_threads=julia_blas_threads,
        )
    if arm == "python_julia_lu":
        return JuliaBackend(
            algorithm="lu", cpu_threads=cpu_threads,
            julia_threads=julia_threads, julia_blas_threads=julia_blas_threads,
        )
    if arm == "python_jax":
        return JaxBackend(device="cpu", cpu_threads=cpu_threads)
    raise NotImplementedError(f"arm {arm!r} has no Python numerical adapter")


def _exception_record(error: BaseException) -> dict[str, object]:
    return storage._error_document(error)


def _resource_environment(
    *, arm: str, threads: int, device: str, backend: Mapping[str, object],
    native_threads: _NativeThreadSpec | None = None,
) -> dict[str, object]:
    snapshot = environment_snapshot(
        arm=arm, device=device, cpu_threads=threads, backend=backend,
        native_threads=native_threads,
    )
    profile = cpu_affinity_profile(threads)
    platform_record = snapshot["platform"]
    assert isinstance(platform_record, dict)
    platform_record["benchmark_resource_cap"] = {
        "requested_cpus": profile["cpus"],
        "topology": profile["topology"],
        "selected_count": profile["selected_count"],
    }
    return snapshot


def _with_environment_identity(snapshot: Mapping[str, object]) -> tuple[dict[str, object], str]:
    value = dict(snapshot)
    identity = environment_identity(value)
    value["environment_sha256"] = identity
    return value, identity


def _base_task(
    *, task_id: str, request_sha256: str, arm: str, sample: int,
    environment: Mapping[str, object], input_artifacts: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "request_sha256": request_sha256,
        "arm": arm,
        "sample": sample,
        "attempts": [],
        "events": [],
        "measurements": [],
        "environment": dict(environment),
        "artifacts": [dict(item) for item in input_artifacts],
    }


def _event(
    workspace: Path,
    *, task_id: str,
    kind: str,
    payload: Mapping[str, object],
    progress=None,
    writer=None,
) -> dict[str, object]:
    eligible = kind in {
        "checkpoint_committed", "population_observed", "progress", "timing", "evaluation", "completed",
    }
    saved = storage.append_event(
        workspace, task_id=task_id, kind=kind, payload=payload,
        writer=writer, force=progress is not None and eligible,
    )
    if progress is not None and eligible:
        _notify_progress(workspace, saved, progress)
    return saved


def _notify_progress(workspace: Path, saved: Mapping[str, object], progress) -> None:
    if progress is None:
        return
    task_id = str(saved["task_id"])
    kind = str(saved["kind"])
    payload = saved["payload"]
    try:
        progress(TaskEvent(
            task_id=task_id,
            sequence=int(saved["sequence"]),
            kind=kind,
            payload_bytes=record_bytes(dict(payload)),
        ))
    except BaseException as error:
        try:
            storage.record_callback_failure(
                workspace,
                task_id=task_id,
                sequence=int(saved["sequence"]),
                event_kind=kind,
                error=error,
            )
        except BaseException as record_error:
            raise error from record_error
        raise


def _record_task_failure(
    workspace: Path,
    *,
    task_id: str,
    attempt_id: str,
    error: BaseException,
    interrupted: bool = False,
) -> None:
    task = storage.task_record(workspace, task_id)
    attempt = next((row for row in task["attempts"] if row["attempt_id"] == attempt_id), None)
    if attempt is None:
        raise EvidenceIntegrityError("benchmark attempt disappeared while recording its failure", stage="benchmark_record")
    if attempt["status"] in {"success", "failure", "interrupted"}:
        return
    failure = None if interrupted else _exception_record(error)
    interruption = None if not interrupted else {
        "kind": "keyboard_interrupt",
        "termination": getattr(error, "termination", "terminated"),
    }
    storage.update_attempt(
        workspace, task_id=task_id, attempt_id=attempt_id,
        status="interrupted" if interrupted else "failure",
        failure=failure, interruption=interruption,
    )
    kind = "interrupted" if interrupted else "failed"
    payload: dict[str, object] = {"attempt_id": attempt_id}
    payload["interruption" if interrupted else "failure"] = interruption if interrupted else failure
    already_recorded = any(
        event.get("kind") == kind
        and event.get("payload", {}).get("attempt_id") == attempt_id
        for event in task.get("events", ())
    )
    if not already_recorded:
        _event(workspace, task_id=task_id, kind=kind, payload=payload)


def _add_task_interval(timing, stage: str, task_id: str, start_ns: int, end_ns: int,
                       *, attempt_id: str, details: Mapping[str, object] | None = None) -> None:
    timing.add_interval(
        stage, task_id=task_id, start_tick_ns=start_ns, end_tick_ns=end_ns,
        details={"attempt_id": attempt_id, **dict(details or {})},
    )


def _persist_measurement_delta(workspace: Path, timing) -> None:
    pending = timing.pending_measurements
    if not pending:
        return
    storage.append_measurements(
        workspace, pending, clock_binding=timing.clock_binding,
    )
    timing.mark_persisted(pending)


def _input_artifacts(workspace: Path, prepared: PreparedBenchmark) -> tuple[Mapping[str, object], ...]:
    declaration = prepared.declaration()
    return (
        storage.write_artifact(workspace, f"inputs/plan-{declaration['plan_sha256']}.json",
                               prepared.plan_bytes, role="source_plan"),
        storage.write_artifact(workspace,
                               f"inputs/source-request-{declaration['source_analysis_sha256']}.json",
                               prepared.analysis.request_bytes, role="source_analysis_request"),
        storage.write_artifact(workspace, f"inputs/benchmark-request-{prepared.request_sha256}.json",
                               prepared.declaration_bytes, role="benchmark_declaration"),
    )


def _read_resume(workspace: Path, prepared: PreparedBenchmark, reference: Mapping[str, object]):
    expected_fields = {
        "task_id", "request_sha256", "arm", "sample", "environment_sha256",
        "attempt_id", "checkpoint", "seal", "seal_sha256",
    }
    if set(reference) != expected_fields:
        raise EvidenceIntegrityError("benchmark resume reference has an unsupported field set", stage="benchmark_checkpoint")
    if prepared.analysis.request()["operation"] != "optimize_direct":
        raise EvidenceIntegrityError("only optimization tasks have resumable checkpoints", stage="benchmark_checkpoint")
    task = storage.task_record(workspace, str(reference["task_id"]))
    backend_identity = task["environment"]["backend"]
    if (
        task["request_sha256"] != reference["request_sha256"]
        or task["arm"] != reference["arm"]
        or task["sample"] != reference["sample"]
        or task["environment"].get("environment_sha256") != reference["environment_sha256"]
        or reference["request_sha256"] != task_request_identity(
            benchmark_sha256=prepared.request_sha256,
            arm=str(task["arm"]),
            environment_sha256=str(reference["environment_sha256"]),
            device=str(backend_identity["device"]),
            cpu_threads=int(task["environment"]["cpu_threads_requested"]),
            backend_identity=backend_identity,
            mesh_identity=prepared.declaration()["benchmark"]["mesh"],
        )
    ):
        raise EvidenceIntegrityError("benchmark resume reference does not bind this declaration and task", stage="benchmark_checkpoint")
    source_attempt = next(
        (attempt for attempt in task["attempts"] if attempt["attempt_id"] == reference["attempt_id"]),
        None,
    )
    if source_attempt is None or source_attempt.get("checkpoint") != dict(reference):
        raise EvidenceIntegrityError("benchmark resume reference is not recorded by its source attempt", stage="benchmark_checkpoint")
    raw = storage.read_checkpoint(
        workspace, reference,
        expected_task_id=str(reference["task_id"]),
        expected_request_sha256=str(reference["request_sha256"]),
        expected_arm=str(reference["arm"]),
        expected_sample=int(reference["sample"]),
        expected_environment_sha256=str(reference["environment_sha256"]),
    )
    return task, record_document(raw)


def _python_task(
    *, workspace: Path, prepared: PreparedBenchmark, input_artifacts,
    arm: str, sample: int, cpu_threads: int, progress, timing,
    resume_reference: Mapping[str, object] | None = None,
    resume_document: Mapping[str, object] | None = None,
) -> None:
    attempt_id = new_attempt_id()
    task_started = timing.mark()
    declaration = prepared.declaration()
    diagnostics = declaration["benchmark"]["diagnostics"][arm]
    checkpoint = None
    if resume_reference is not None:
        if resume_document is None:
            raise EvidenceIntegrityError("resume reference lacks its verified checkpoint document", stage="benchmark_checkpoint")
        checkpoint = {
            key: resume_document[key]
            for key in ("generation", "next_ordinal", "cma", "anchors", "baseline", "best", "cache", "generations")
            if key in resume_document
        }
    native_threads = None
    if arm in {"python_julia_reuse", "python_julia_lu"}:
        native_threads = _native_thread_spec(
            cpu_threads,
            julia_threads=declaration["benchmark"].get("julia_threads"),
            julia_blas_threads=declaration["benchmark"].get("julia_blas_threads"),
        )
    request = {
        "schema": "scnsim.benchmark_python_task_request",
        "schema_version": 1,
        "prepared": prepared.wire(),
        "benchmark_sha256": prepared.request_sha256,
        "arm": arm,
        "sample": sample,
        "attempt_id": attempt_id,
        "device": "cpu",
        "cpu_threads": cpu_threads,
        "checkpoint": checkpoint,
        "diagnostics": diagnostics,
    }
    if native_threads is not None:
        request["julia_threads"] = native_threads.julia_threads
        request["julia_blas_threads"] = native_threads.blas_threads
    request_path = workspace / "launches" / f"{attempt_id}.json"
    request_artifact = storage.write_artifact(
        workspace, request_path.relative_to(workspace), record_bytes(request), role="python_child_request",
    )
    state: dict[str, object] = {
        "task_id": None,
        "attempt_started": False,
        "terminal": False,
        "request_artifact": request_artifact,
    }

    def on_event(frame: Mapping[str, object]):
        kind = str(frame["kind"])
        frame_task_id = str(frame["task_id"])
        payload = frame["payload"]
        if kind == "failed" and not frame_task_id:
            failure = payload["failure"]
            error = BackendProtocolError(
                str(failure.get("message", "Python benchmark child failed before environment identity")),
                stage=str(failure.get("stage", "backend_initialization")),
                evidence={"remote_failure": dict(failure)},
            )
            storage.record_task_launch_failure(
                workspace, arm=arm, sample=sample, attempt_id=attempt_id,
                benchmark_sha256=prepared.request_sha256,
                source_analysis_sha256=str(declaration["source_analysis_sha256"]),
                error=error, request_artifact=request_artifact,
            )
            state["terminal"] = True
            return None

        if kind == "ready":
            context = payload["context"]
            environment = dict(payload["environment"])
            environment["environment_sha256"] = context["environment_sha256"]
            task = _base_task(
                task_id=frame_task_id,
                request_sha256=str(context["request_sha256"]),
                arm=arm,
                sample=sample,
                environment=environment,
                input_artifacts=(*input_artifacts, request_artifact),
            )
            storage.ensure_task(workspace, task)
            state["task_id"] = frame_task_id
            storage.begin_attempt(
                workspace, task_id=frame_task_id, attempt_id=attempt_id,
                resume_from=resume_reference,
            )
            state["attempt_started"] = True
            storage.update_attempt(workspace, task_id=frame_task_id, attempt_id=attempt_id, status="launched")
            _event(workspace, task_id=frame_task_id, kind="ready", payload=payload, progress=progress)
            state["writer"] = storage.begin_task_writer(
                workspace,
                task_id=frame_task_id,
                attempt_id=attempt_id,
                diagnostics=diagnostics,
                checkpoint_document=resume_document,
            )
            return {"ready": True}

        if state["task_id"] != frame_task_id:
            raise BackendProtocolError("Python child event changed task binding", stage="benchmark_child_protocol")
        task_id = frame_task_id

        if kind in {"baseline_ready", "generation_ready"}:
            writer = state.get("writer")
            if writer is None:
                raise EvidenceIntegrityError("benchmark task evidence writer is unavailable", stage="benchmark_checkpoint")
            _saved, acknowledgment = storage.commit_barrier(
                workspace,
                writer=writer,
                kind=kind,
                payload={"attempt_id": attempt_id, **dict(payload)},
            )
            return acknowledgment

        if kind == "completed":
            result_bytes = record_bytes(dict(payload["result"]))
            result_ref = storage.write_artifact(
                workspace, f"tasks/{task_id}/attempts/{attempt_id}/result.json",
                result_bytes, role="python_task_result",
            )
            payload = {"attempt_id": attempt_id, "result": result_ref}
        elif kind == "checkpoint_committed":
            payload = {"attempt_id": attempt_id, **dict(payload)}
        else:
            payload = {"attempt_id": attempt_id, **dict(payload)}

        if kind == "timing":
            timing_payload = dict(frame["payload"])
            start_tick = timing_payload.get("start_tick_ns")
            end_tick = timing_payload.get("end_tick_ns")
            if isinstance(start_tick, int) and isinstance(end_tick, int):
                timing.add_interval(
                    str(timing_payload["stage"]), task_id=task_id,
                    start_tick_ns=start_tick, end_tick_ns=end_tick,
                    counts=timing_payload.get("counts"),
                    details={"attempt_id": attempt_id, "source": "python_task"},
                )
        saved = _event(
            workspace,
            task_id=task_id,
            kind=kind,
            payload=payload,
            progress=None if kind == "completed" else progress,
            writer=state.get("writer"),
        )
        if kind in {"completed", "failed", "interrupted"}:
            state["terminal_event"] = saved
        if kind == "completed":
            state["completed_event"] = saved
        return None

    try:
        context, _result_reference = run_python_child(
            request_path,
            arm=arm,
            sample=sample,
            attempt_id=attempt_id,
            benchmark_sha256=prepared.request_sha256,
            device="cpu",
            cpu_threads=cpu_threads,
            expected_julia_threads=(
                None if native_threads is None else native_threads.julia_threads
            ),
            expected_julia_blas_threads=(
                None if native_threads is None else native_threads.blas_threads
            ),
            mesh_identity=declaration["benchmark"]["mesh"],
            on_event=on_event,
            timing=timing,
            expected_task_id=None if resume_reference is None else str(resume_reference["task_id"]),
        )
        task_id = str(context["task_id"])
        _add_task_interval(timing, "task_end_to_end", task_id, task_started, timing.mark(), attempt_id=attempt_id)
        storage.update_attempt(workspace, task_id=task_id, attempt_id=attempt_id, status="success")
        state["terminal"] = True
        completed_event = state.get("completed_event")
        if isinstance(completed_event, Mapping):
            _notify_progress(workspace, completed_event, progress)
    except KeyboardInterrupt as error:
        if state.get("terminal"):
            raise
        task_id = state.get("task_id")
        try:
            if isinstance(task_id, str) and state.get("attempt_started"):
                writer = state.get("writer")
                if writer is not None:
                    writer.flush()
                _record_task_failure(workspace, task_id=task_id, attempt_id=attempt_id, error=error, interrupted=True)
                _add_task_interval(timing, "task_end_to_end", task_id, task_started, timing.mark(), attempt_id=attempt_id)
            elif not state.get("terminal"):
                storage.record_task_launch_failure(
                    workspace, arm=arm, sample=sample, attempt_id=attempt_id,
                    benchmark_sha256=prepared.request_sha256,
                    source_analysis_sha256=str(declaration["source_analysis_sha256"]),
                    error=error, request_artifact=request_artifact,
                )
        except BaseException as record_error:
            raise error from record_error
        state["terminal"] = True
        raise
    except BaseException as error:
        if state.get("terminal"):
            raise
        task_id = state.get("task_id")
        try:
            if isinstance(task_id, str) and state.get("attempt_started"):
                writer = state.get("writer")
                if writer is not None:
                    writer.flush()
                if not state.get("terminal_event"):
                    _event(workspace, task_id=task_id, kind="failed", payload={
                        "attempt_id": attempt_id, "failure": _exception_record(error),
                    })
                _record_task_failure(workspace, task_id=task_id, attempt_id=attempt_id, error=error)
                _add_task_interval(timing, "task_end_to_end", task_id, task_started, timing.mark(), attempt_id=attempt_id)
            elif not state.get("terminal"):
                storage.record_task_launch_failure(
                    workspace, arm=arm, sample=sample, attempt_id=attempt_id,
                    benchmark_sha256=prepared.request_sha256,
                    source_analysis_sha256=str(declaration["source_analysis_sha256"]),
                    error=error, request_artifact=request_artifact,
                )
        except BaseException as record_error:
            raise error from record_error
        state["terminal"] = True
        raise


def _native_task_environment(
    prepared_runtime, wrapper: Path, *, threads: int,
    native_threads: _NativeThreadSpec,
) -> tuple[dict[str, object], str, dict[str, object]]:
    executable_sha = sha256(prepared_runtime.executable.read_bytes()).hexdigest()
    backend = {
        "backend": "original_julia_terminal",
        "device": "cpu",
        "runtime": dict(prepared_runtime.runtime_metadata),
        "julia_version": prepared_runtime.julia_version,
        "julia_executable_sha256": executable_sha,
        "benchmark_entrypoint_sha256": sha256(wrapper.read_bytes()).hexdigest(),
        "requested_julia_threads": native_threads.julia_threads,
        "requested_julia_blas_threads": native_threads.blas_threads,
        "requested_thread_environment": {
            "JULIA_NUM_THREADS": str(native_threads.julia_threads),
            **{
                name: str(native_threads.blas_threads)
                for name in (
                    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    }
    snapshot = _resource_environment(
        arm="original_julia", threads=threads, device="cpu", backend=backend,
        native_threads=native_threads,
    )
    return snapshot, executable_sha, backend


def _path_artifact(root: Path, path: Path, *, role: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceIntegrityError("benchmark artifact is not a regular file", stage="benchmark_artifact",
                                     evidence={"path": str(path)})
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    data = path.read_bytes()
    return {"path": relative, "role": role, "byte_length": len(data), "sha256": sha256(data).hexdigest()}


def _native_checkpoint_document(
    *, prepared: PreparedBenchmark, task_id: str, request_sha256: str,
    arm: str, sample: int, environment_sha256: str,
    native_workspace: str, native_request_sha256: str,
    baseline, resume_ledger_sha256: str | None, source_attempt_sha256: str,
) -> dict[str, object]:
    return {
        "schema": "scnsim.benchmark_original_julia_checkpoint",
        "schema_version": 1,
        "benchmark_sha256": prepared.request_sha256,
        "task_id": task_id,
        "request_sha256": request_sha256,
        "arm": arm,
        "sample": sample,
        "environment_sha256": environment_sha256,
        "plan_sha256": prepared.declaration()["plan_sha256"],
        "source_analysis_sha256": prepared.analysis.request_sha256,
        "mesh": prepared.declaration()["benchmark"]["mesh"],
        "native_workspace": native_workspace,
        "native_request_sha256": native_request_sha256,
        "native_baseline_checkpoint_sha256": baseline.checkpoint_sha256,
        "native_baseline_checkpoint_seal_sha256": baseline.seal_sha256,
        "resume_ledger_sha256": resume_ledger_sha256,
        "source_attempt_sha256": source_attempt_sha256,
    }


def _exact_native_ledger_exists(
    binding,
    request_sha256: str,
    ledger_sha256: str | None,
    *,
    expected_julia_threads: int,
    expected_blas_threads: int,
) -> bool:
    if ledger_sha256 is None:
        return True
    attempts = binding.leaf / "requests" / request_sha256 / "attempts"
    if attempts.is_symlink() or not attempts.is_dir():
        return False
    for final in binding._final_attempt_directories(attempts):
        attempt, receipt, _ = binding._verify_attempt(
            final, request_sha256, final.name, require_final_name=True,
        )
        links = verified_generation_links(
            final,
            request_sha256=request_sha256,
            attempt_sha256=sha256(canonical_json_bytes(attempt)).hexdigest(),
            expected_julia_threads=expected_julia_threads,
            expected_blas_threads=expected_blas_threads,
            allow_other_artifacts=True,
        )
        if receipt["outcome"] != "success" and any(link["sha256"] == ledger_sha256 for link in links):
            return True
        if receipt["outcome"] == "success" and any(link["sha256"] == ledger_sha256 for link in links):
            return True
    return False


def _native_artifact_refs(
    workspace: Path,
    *,
    request_path: Path,
    final_directory: Path,
    observation_path: Path,
    point_checkpoints=(),
) -> list[dict[str, object]]:
    references = [_path_artifact(workspace, request_path, role="original_julia_request")]
    for name, role in (
        ("attempt.json", "original_julia_attempt"),
        ("receipt.json", "original_julia_receipt"),
        ("outcome.json", "original_julia_outcome"),
        ("result.json", "original_julia_result"),
    ):
        path = final_directory / name
        if path.is_file():
            references.append(_path_artifact(workspace, path, role=role))
    generations = final_directory / "artifacts" / "generations"
    if generations.is_dir():
        references.extend(
            _path_artifact(workspace, path, role="original_julia_generation_ledger")
            for path in sorted(generations.iterdir()) if path.is_file()
        )
    if observation_path.is_file():
        references.append(_path_artifact(workspace, observation_path, role="original_julia_timing_sidecar"))
    for checkpoint in point_checkpoints:
        for name, role in (
            ("record.json", "original_julia_point_checkpoint_record"),
            ("source-attempt.json", "original_julia_point_checkpoint_attempt"),
            ("seal.json", "original_julia_point_checkpoint_seal"),
        ):
            references.append(_path_artifact(workspace, checkpoint.directory / name, role=role))
    return references


def _native_numerical_observations(
    *,
    workspace: Path,
    result_document: Mapping[str, object] | None,
    result_artifact: Mapping[str, object] | None,
    artifact_refs: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
    generation_records: list[tuple[int, str, Mapping[str, object]]],
    sweep_records: list[Mapping[str, object]] | None = None,
    point_checkpoints: tuple[
        tuple[PointCheckpoint, Mapping[str, object] | None], ...
    ] | None = None,
    request: Mapping[str, object] | None = None,
    baseline: object = None,
    direct_arrays: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Retain exact values from the result/ledger pass that already verified them."""

    result_kind = (
        result_document.get("result_kind")
        if result_document is not None
        else "optimization"
        if generation_records or baseline is not None
        else "parameter_sweep"
        if point_checkpoints
        else None
    )
    if not isinstance(result_kind, str):
        return None

    result_ref = None if result_artifact is None else dict(result_artifact)
    observations: dict[str, object] = {
        "result_kind": result_kind,
        "result_artifact": result_ref,
        "records": [],
    }
    records: list[dict[str, object]] = []
    ledger_refs = {
        str(reference["sha256"]): dict(reference)
        for reference in artifact_refs
        if reference.get("role") == "original_julia_generation_ledger"
    }
    point_checkpoint_refs = {
        str(reference["sha256"]): dict(reference)
        for reference in artifact_refs
        if reference.get("role") == "original_julia_point_checkpoint_seal"
    }
    point_checkpoint_record_refs = {
        str(reference["path"]): dict(reference)
        for reference in artifact_refs
        if reference.get("role") == "original_julia_point_checkpoint_record"
    }

    if result_document is not None and result_kind == "optimization":
        records.append({
            "role": "baseline",
            "generation": 0,
            "native_ordinal": 0,
            "source_artifact": result_ref,
            "value": result_document["baseline"],
        })
    elif result_document is None and baseline is not None:
        checkpoint = getattr(baseline, "checkpoint", None)
        directory = getattr(baseline, "directory", None)
        if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("baseline"), Mapping) and isinstance(directory, Path):
            records.append({
                "role": "baseline",
                "generation": 0,
                "native_ordinal": 0,
                "source_artifact": _path_artifact(
                    workspace, directory / "checkpoint.json",
                    role="original_julia_baseline_checkpoint",
                ),
                "value": checkpoint["baseline"],
            })

    if result_kind == "optimization":
        for generation, digest, ledger in generation_records:
            source = ledger_refs.get(digest)
            for candidate in ledger["candidates"]:
                records.append({
                    "role": "candidate",
                    "generation": generation,
                    "native_ordinal": candidate["evaluation_ordinal"],
                    "source_artifact": source,
                    "value": candidate,
                })
        if result_document is not None:
            best = result_document.get("best")
            if isinstance(best, Mapping):
                ordinal = best.get("evaluation_ordinal")
                if ordinal == 0:
                    observations["winner"] = {
                        "generation": 0,
                        "native_ordinal": 0,
                        "source_artifact": result_ref,
                        "value": dict(best),
                    }
                else:
                    for generation, digest, ledger in generation_records:
                        winner = next((
                            candidate for candidate in ledger["candidates"]
                            if candidate.get("evaluation_ordinal") == ordinal
                        ), None)
                        if winner is not None:
                            observations["winner"] = {
                                "generation": generation,
                                "native_ordinal": ordinal,
                                "source_artifact": ledger_refs.get(digest),
                                "value": dict(best),
                            }
                            break
    elif result_document is not None and result_kind == "direct_response":
        catalogs = result_document["array_catalog"]
        direct_values = cast(Mapping[str, object], direct_arrays)
        arrays = {
            role: {
                "catalog": catalogs[role],
                "values": array_record(direct_values[role]),
            }
            for role in ("frequencies", "s", "y", "z")
        }
        records.append({
            "role": result_kind,
            "source_artifact": result_ref,
            "value": {
                "parameters": result_document["parameters"],
                "scalar_catalog": result_document["scalar_catalog"],
                "arrays": arrays,
            },
        })
    elif result_document is not None and result_kind == "parameter_sweep":
        if result_ref is None or request is not None and request.get("operation") == "solve_hb":
            return None
        attempt_prefix = str(result_ref["path"]).rsplit("/", 1)[0]

        def point_file_reference(
            row: Mapping[str, object], *, role: str,
        ) -> dict[str, object]:
            return {
                "path": f"{attempt_prefix}/artifacts/parameter_points/{row['path']}",
                "role": role,
                "byte_length": row["byte_length"],
                "sha256": row["sha256"],
            }

        for verified_point in sweep_records or ():
            point = cast(Mapping[str, object], verified_point["point"])
            chunk_file = cast(Mapping[str, object], verified_point["chunk_file"])
            chunk_reference = point_file_reference(
                chunk_file, role="original_julia_parameter_point_chunk",
            )
            payload = verified_point["payload"]
            source = chunk_reference
            point_role = (
                str(payload["result_kind"])
                if isinstance(payload, Mapping)
                else "point"
            )
            value: dict[str, object] = {
                "native_result_kind": (
                    None if not isinstance(payload, Mapping)
                    else payload.get("result_kind")
                ),
                "source_index": point["source_index"],
                "status": point["status"],
                "parameters": point["parameters"],
                "point_artifact": chunk_reference,
            }
            checkpoint_ref = point_checkpoint_refs.get(
                str(point["checkpoint_seal_sha256"])
            )
            if checkpoint_ref is not None:
                value["checkpoint_artifact"] = checkpoint_ref
            if point["status"] == "failure":
                value["failure"] = point["failure"]
            elif isinstance(payload, Mapping):
                payload_file = cast(Mapping[str, object], verified_point["payload_file"])
                source = point_file_reference(
                    payload_file, role="original_julia_parameter_point_payload",
                )
                value["scalar_catalog"] = payload["scalar_catalog"]
                if payload.get("result_kind") == "direct_response":
                    catalogs = cast(Mapping[str, object], payload["array_catalog"])
                    arrays = cast(Mapping[str, object], verified_point["direct_arrays"])
                    value["arrays"] = {
                        role: {
                            "catalog": catalogs[role],
                            "values": array_record(arrays[role]),
                        }
                        for role in ("frequencies", "s", "y", "z")
                    }
            records.append({
                "role": point_role,
                "native_ordinal": point["ordinal"],
                "source_artifact": source,
                "value": value,
            })
    elif result_document is None and result_kind == "parameter_sweep":
        if request is None or request.get("operation") == "solve_hb":
            return None
        workspace_root = workspace.resolve()
        for checkpoint, payload in point_checkpoints or ():
            metadata = cast(Mapping[str, object], checkpoint.record["metadata"])
            ordinal = int(metadata["ordinal"])
            seal_reference = point_checkpoint_refs.get(checkpoint.seal_sha256)
            checkpoint_record_path = (
                checkpoint.directory / "record.json"
            ).resolve().relative_to(workspace_root).as_posix()
            point_reference = point_checkpoint_record_refs.get(
                checkpoint_record_path, seal_reference,
            )
            if point_reference is None:
                continue
            source = point_reference
            point_role = (
                str(payload["result_kind"])
                if isinstance(payload, Mapping)
                else "point"
            )
            value: dict[str, object] = {
                "native_result_kind": (
                    None if not isinstance(payload, Mapping)
                    else payload.get("result_kind")
                ),
                "source_index": metadata["source_index"],
                "status": metadata["status"],
                "parameters": metadata["parameters"],
                "point_artifact": point_reference,
            }
            if seal_reference is not None:
                value["checkpoint_artifact"] = seal_reference
            if metadata["status"] == "failure":
                value["failure"] = metadata["failure"]
            elif isinstance(payload, Mapping):
                files = cast(list[Mapping[str, object]], checkpoint.record["files"])
                payload_file = next(
                    row for row in files if row["path"] == "payload.json"
                )
                payload_path = (
                    checkpoint.directory / "point" / str(payload_file["path"])
                ).resolve().relative_to(workspace_root).as_posix()
                source = {
                    "path": payload_path,
                    "role": "original_julia_parameter_point_checkpoint_payload",
                    "byte_length": payload_file["byte_length"],
                    "sha256": payload_file["sha256"],
                }
                value["scalar_catalog"] = payload["scalar_catalog"]
                if payload.get("result_kind") == "direct_response":
                    arrays = _decode_direct_point_arrays(
                        checkpoint.directory / "point",
                        payload,
                        request,
                        artifact_prefix=(
                            "artifacts/parameter_points/points/"
                            f"{ordinal:06d}/"
                        ),
                    )
                    catalogs = cast(Mapping[str, object], payload["array_catalog"])
                    value["arrays"] = {
                        role: {
                            "catalog": catalogs[role],
                            "values": array_record(arrays[role]),
                        }
                        for role in ("frequencies", "s", "y", "z")
                    }
            records.append({
                "role": point_role,
                "native_ordinal": ordinal,
                "source_artifact": source,
                "value": value,
            })
    elif result_document is not None and result_kind in {
        "diagonal_root", "operator_element_root", "response_element",
    }:
        records.append({
            "role": result_kind,
            "source_artifact": result_ref,
            "value": {
                "parameters": result_document["parameters"],
                "scalar_catalog": result_document["scalar_catalog"],
            },
        })

    if not records:
        return None
    observations["records"] = records
    return observations


def _native_task(
    *, workspace: Path, prepared: PreparedBenchmark, input_artifacts,
    arm: str, sample: int, cpu_threads: int, progress, timing,
    prepared_runtime, wrapper: Path,
    resume_reference: Mapping[str, object] | None = None,
    resume_document: Mapping[str, object] | None = None,
) -> None:
    if arm != "original_julia":
        raise NotImplementedError(f"arm {arm!r} is not an original Julia task")
    declaration = prepared.declaration()
    native_threads = _native_thread_spec(
        cpu_threads,
        julia_threads=declaration["benchmark"].get("julia_threads"),
        julia_blas_threads=declaration["benchmark"].get("julia_blas_threads"),
    )
    if declaration["benchmark"]["task_kind"] != "full":
        raise NotImplementedError("original Julia does not consume Python common-cohort tasks")
    if prepared.analysis.request()["operation"] != "optimize_direct" and resume_reference is not None:
        raise EvidenceIntegrityError("only optimization tasks have resumable checkpoints", stage="benchmark_checkpoint")

    task_started = timing.mark()
    attempt_id = new_attempt_id()
    request = prepared.analysis.request()
    sweep_operation = request["parameter_source"]["kind"] in {"grid", "points"}
    request_sha = prepared.analysis.request_sha256
    snapshot, executable_sha, backend_identity = _native_task_environment(
        prepared_runtime, wrapper, threads=cpu_threads,
        native_threads=native_threads,
    )
    environment, environment_sha = _with_environment_identity(snapshot)
    task_request_sha = task_request_identity(
        benchmark_sha256=prepared.request_sha256,
        arm=arm,
        environment_sha256=environment_sha,
        device=str(backend_identity["device"]),
        cpu_threads=cpu_threads,
        backend_identity=backend_identity,
        mesh_identity=declaration["benchmark"]["mesh"],
    )
    task_id = task_identifier(request_sha256=task_request_sha, sample=sample)
    if resume_reference is not None and resume_reference["task_id"] != task_id:
        raise EvidenceIntegrityError("original Julia resume environment differs from its source task", stage="benchmark_checkpoint")
    task = _base_task(
        task_id=task_id, request_sha256=task_request_sha, arm=arm, sample=sample,
        environment=environment, input_artifacts=input_artifacts,
    )
    storage.ensure_task(workspace, task)
    storage.begin_attempt(workspace, task_id=task_id, attempt_id=attempt_id, resume_from=resume_reference)
    task_state: dict[str, object] = {"terminal": False, "checkpoint": None}
    native_workspace_relative: str
    if resume_reference is None:
        native_workspace_relative = f"tasks/{task_id}/native/{attempt_id}"
    else:
        if not isinstance(resume_document, Mapping) or resume_document.get("schema") != "scnsim.benchmark_original_julia_checkpoint":
            raise EvidenceIntegrityError("original Julia checkpoint payload has an unsupported schema", stage="benchmark_checkpoint")
        if (
            resume_document.get("benchmark_sha256") != prepared.request_sha256
            or resume_document.get("task_id") != task_id
            or resume_document.get("request_sha256") != task_request_sha
            or resume_document.get("arm") != arm
            or resume_document.get("sample") != sample
            or resume_document.get("environment_sha256") != environment_sha
            or resume_document.get("plan_sha256") != declaration["plan_sha256"]
            or resume_document.get("source_analysis_sha256") != request_sha
            or resume_document.get("mesh") != declaration["benchmark"]["mesh"]
            or resume_document.get("native_request_sha256") != request_sha
        ):
            raise EvidenceIntegrityError("original Julia checkpoint does not bind this request and mesh", stage="benchmark_checkpoint")
        native_workspace_relative = str(resume_document["native_workspace"])
        expected_prefix = f"tasks/{task_id}/native/"
        if not native_workspace_relative.startswith(expected_prefix) or ".." in Path(native_workspace_relative).parts:
            raise EvidenceIntegrityError("original Julia checkpoint points outside its task workspace", stage="benchmark_checkpoint")

    native_root = workspace / native_workspace_relative
    declaration_path = workspace / "inputs" / f"benchmark-request-{prepared.request_sha256}.json"
    observation_path = native_root / f"whole-timing-{attempt_id}.json"
    binding = bind_workspace(
        native_root,
        plan_sha256=str(declaration["plan_sha256"]),
        plan_bytes=prepared.plan_bytes,
        versioned=False,
        commit=lambda: None,
        _expected_julia_threads=native_threads.julia_threads,
        _expected_blas_threads=native_threads.blas_threads,
    )
    allocation = None
    baseline = None
    resume_ledger = None
    terminal = None
    standard_attempt_sha: str | None = None
    source_units = prepared.analysis.source_units()
    plan_document = json.loads(prepared.plan_bytes)

    def promote_native_receipt(receipt: Mapping[str, object]) -> None:
        if sweep_operation:
            receipt = canonical_receipt_document({
                **dict(receipt),
                "point_checkpoint_count": len(binding.point_checkpoints(request_sha)),
            })
        binding.promote_attempt(allocation, receipt)

    def native_receipt_failure(
        error: SCNSimError,
        *,
        stdout=(),
        stderr=(),
        interrupted=False,
        record_error: BaseException | None = None,
    ) -> None:
        nonlocal standard_attempt_sha
        generation_records: list[tuple[int, str, Mapping[str, object]]] = []
        assert allocation is not None
        if allocation.final_directory.exists():
            return
        if standard_attempt_sha is None:
            standard_attempt_sha = binding.seal_attempt(
                allocation,
                _attempt_document(
                    allocation, started=_utc_now(), executable_sha=executable_sha,
                    state="allocated", resume_ledger_sha=resume_ledger,
                    optimization=prepared.analysis.request()["operation"] == "optimize_direct",
                    checkpoint=baseline,
                ),
            )
        _write_logs(allocation.staging_directory, stdout, stderr)
        if interrupted:
            native_links, generation_records = cast(tuple[
                list[dict[str, str]],
                list[tuple[int, str, Mapping[str, object]]],
            ], verified_generation_links(
                allocation.staging_directory, request_sha256=request_sha,
                attempt_sha256=standard_attempt_sha, allow_other_artifacts=True,
                expected_julia_threads=native_threads.julia_threads,
                expected_blas_threads=native_threads.blas_threads,
                include_records=True,
            ))
            # The Julia child has exited before this branch. Validate and snapshot
            # only its completed ledger chain; progress frames arrive asynchronously
            # while the child may still be atomically publishing the next ledger.
            save_native_resume_checkpoint(native_links)
            _discard_untrusted_outputs(allocation.staging_directory, keep_ledgers=True)
            receipt = _receipt(
                request=request, plan_document=plan_document, request_sha=request_sha,
                attempt_sha=standard_attempt_sha, outcome="interrupted", artifacts=native_links,
                source_units=source_units,
                interruption={"kind": "keyboard_interrupt", "termination": getattr(error, "termination", "terminated"),
                              "interrupted_at_utc": _utc_now()},
            )
        else:
            _discard_untrusted_outputs(allocation.staging_directory)
            receipt = _receipt(
                request=request, plan_document=plan_document, request_sha=request_sha,
                attempt_sha=standard_attempt_sha, outcome="failure", artifacts=[],
                source_units=source_units,
                failure=_failure_record(error, request["operation"], request_sha, standard_attempt_sha),
            )
        promote_native_receipt(receipt)
        failure_value = None if interrupted else _exception_record(record_error or error)
        interruption_value = None if not interrupted else {
            "kind": "keyboard_interrupt",
            "termination": getattr(error, "termination", "terminated"),
        }
        verified_point_checkpoints = (
            binding._point_checkpoints_with_payloads(request_sha)
            if sweep_operation
            else ()
        )
        point_checkpoint_records = tuple(
            checkpoint for checkpoint, _payload in verified_point_checkpoints
        )
        references = _native_artifact_refs(
            workspace,
            request_path=request_directory / "request.json",
            final_directory=allocation.final_directory,
            observation_path=observation_path,
            point_checkpoints=point_checkpoint_records,
        )
        storage.update_attempt(
            workspace,
            task_id=task_id,
            attempt_id=attempt_id,
            status="interrupted" if interrupted else "failure",
            failure=failure_value,
            interruption=interruption_value,
            artifacts=tuple(references),
        )
        kind = "interrupted" if interrupted else "failed"
        payload: dict[str, object] = {"attempt_id": attempt_id}
        payload["interruption" if interrupted else "failure"] = interruption_value if interrupted else failure_value
        numerical_observations = _native_numerical_observations(
            workspace=workspace,
            result_document=None,
            result_artifact=None,
            artifact_refs=references,
            generation_records=generation_records,
            point_checkpoints=verified_point_checkpoints,
            request=request,
            baseline=baseline,
        )
        if numerical_observations is not None:
            payload["numerical_observations"] = numerical_observations
        _event(workspace, task_id=task_id, kind=kind, payload=payload)

    try:
        with binding.writer():
            request_directory = binding.ensure_request(request_sha, prepared.analysis.request_bytes)
            baseline = binding.baseline_checkpoint(request_sha)
            point_checkpoints = binding.point_checkpoints(request_sha) if sweep_operation else ()
            if resume_reference is not None:
                if baseline is None or (
                    baseline.checkpoint_sha256 != resume_document["native_baseline_checkpoint_sha256"]
                    or baseline.seal_sha256 != resume_document["native_baseline_checkpoint_seal_sha256"]
                ):
                    raise EvidenceIntegrityError("native baseline checkpoint differs from the selected benchmark checkpoint", stage="benchmark_checkpoint")
                resume_ledger = resume_document["resume_ledger_sha256"]
                if not _exact_native_ledger_exists(
                    binding, request_sha, resume_ledger,
                    expected_julia_threads=native_threads.julia_threads,
                    expected_blas_threads=native_threads.blas_threads,
                ):
                    raise EvidenceIntegrityError("selected native generation ledger is absent from its source workspace", stage="benchmark_checkpoint")
            allocation = binding.allocate_attempt(request_sha)
            started = _utc_now()
            benchmark_declaration_path = declaration_path.resolve()
            if benchmark_declaration_path.is_symlink() or not benchmark_declaration_path.is_file():
                raise EvidenceIntegrityError("benchmark declaration input is absent", stage="benchmark_record")

            def publish_checkpoint(ready: Mapping[str, object]) -> Mapping[str, object]:
                nonlocal baseline
                assert allocation is not None and standard_attempt_sha is not None
                baseline = binding.publish_baseline_checkpoint(
                    request_sha, standard_attempt_sha,
                    allocation.staging_directory / "baseline-checkpoint.json",
                    expected_sha256=str(ready["checkpoint_sha256"]),
                    expected_byte_length=int(ready["byte_length"]),
                )
                checkpoint_doc = _native_checkpoint_document(
                    prepared=prepared, task_id=task_id, request_sha256=task_request_sha,
                    arm=arm, sample=sample, environment_sha256=environment_sha,
                    native_workspace=native_workspace_relative, native_request_sha256=request_sha,
                    baseline=baseline, resume_ledger_sha256=resume_ledger,
                    source_attempt_sha256=standard_attempt_sha,
                )
                checkpoint_ref = storage.publish_checkpoint(
                    workspace, task_id=task_id, attempt_id=attempt_id,
                    checkpoint_bytes=record_bytes(checkpoint_doc),
                )
                task_state["checkpoint"] = checkpoint_ref
                return {"checkpoint_sha256": baseline.checkpoint_sha256, "seal_sha256": baseline.seal_sha256}

            def publish_point(ready: Mapping[str, object]) -> Mapping[str, object]:
                assert allocation is not None and standard_attempt_sha is not None
                try:
                    published = binding.publish_point_checkpoint(
                        request_sha, standard_attempt_sha, allocation.staging_directory, ready,
                    )
                except (EvidenceIntegrityError, OSError) as error:
                    raise BackendProtocolError(
                        "benchmark point checkpoint failed independent validation",
                        stage="point_checkpoint",
                    ) from error
                return {
                    "record_sha256": ready["record_sha256"],
                    "seal_sha256": published.seal_sha256,
                }

            def authorize(ready: BootstrapReady) -> str:
                nonlocal standard_attempt_sha
                standard_attempt_sha = binding.seal_attempt(
                    allocation,
                    _attempt_document(
                        allocation, started=started, executable_sha=executable_sha,
                        state="launched", ready=ready,
                        resume_ledger_sha=resume_ledger,
                        optimization=request["operation"] == "optimize_direct",
                        checkpoint=baseline,
                    ),
                )
                storage.update_attempt(workspace, task_id=task_id, attempt_id=attempt_id, status="launched")
                _event(workspace, task_id=task_id, kind="authorized", payload={
                    "attempt_id": attempt_id, "julia_version": ready.julia_version,
                    "julia_threads": ready.julia_threads, "blas_threads": ready.blas_threads,
                    "blas_vendor": ready.blas_vendor,
                })
                return standard_attempt_sha

            def save_native_resume_checkpoint(links: list[dict[str, str]]) -> None:
                if baseline is None or standard_attempt_sha is None:
                    return
                latest = links[-1]["sha256"] if links else resume_ledger
                checkpoint_doc = _native_checkpoint_document(
                    prepared=prepared, task_id=task_id, request_sha256=task_request_sha,
                    arm=arm, sample=sample, environment_sha256=environment_sha,
                    native_workspace=native_workspace_relative, native_request_sha256=request_sha,
                    baseline=baseline, resume_ledger_sha256=latest,
                    source_attempt_sha256=standard_attempt_sha,
                )
                checkpoint_ref = storage.publish_checkpoint(
                    workspace, task_id=task_id, attempt_id=attempt_id,
                    checkpoint_bytes=record_bytes(checkpoint_doc),
                )
                task_state["checkpoint"] = checkpoint_ref

            def on_native_progress(frame: Mapping[str, object]) -> None:
                payload = {"attempt_id": attempt_id, **dict(frame)}
                try:
                    _event(workspace, task_id=task_id, kind="progress", payload=payload, progress=progress)
                except BaseException as error:
                    task_state["callback_error"] = error
                    raise

            def observe_native_timing(stage: str, start_ns: int, end_ns: int, details: Mapping[str, object]) -> None:
                timing.add_interval(
                    stage, task_id=task_id, start_tick_ns=start_ns, end_tick_ns=end_ns,
                    details={"attempt_id": attempt_id, **dict(details)},
                )

            checkpoint_control = None if baseline is None else {
                "checkpoint_sha256": baseline.checkpoint_sha256,
                "seal_sha256": baseline.seal_sha256,
            }
            publisher_control = publish_checkpoint if request["operation"] == "optimize_direct" else None
            with packaged_julia_resources() as (project, _entrypoint, _runtime):
                whole_args = (
                    str(request_directory / "request.json"),
                    str(allocation.staging_directory),
                    str(benchmark_declaration_path),
                    str(observation_path),
                    str(cpu_threads),
                )
                try:
                    terminal = run_terminal(
                        prepared_runtime,
                        request_path=(request_directory / "request.json").resolve(),
                        staging_directory=allocation.staging_directory.resolve(),
                        request_sha256=request_sha,
                        attempt_ordinal=allocation.ordinal,
                        authorize=authorize,
                        checkpoint=checkpoint_control,
                        publish_checkpoint=publisher_control,
                        on_progress=on_native_progress,
                        point_recovery=(
                            tuple({"ordinal": index, "seal_sha256": item.seal_sha256}
                                  for index, item in enumerate(point_checkpoints))
                            if sweep_operation else None
                        ),
                        publish_point=publish_point if sweep_operation else None,
                        _timing_observer=observe_native_timing,
                        _entrypoint=project / "bin" / "scnsim_benchmark.jl",
                        _entrypoint_arguments=("--whole", *whole_args),
                        _julia_threads=native_threads.julia_threads,
                        _julia_blas_threads=native_threads.blas_threads,
                        _cpu_affinity=cpu_affinity_profile(cpu_threads),
                        _preserve_progress_callback_exception=True,
                    )
                except KeyboardInterrupt as error:
                    try:
                        native_receipt_failure(error, interrupted=True)
                    except BaseException as record_error:
                        raise error from record_error
                    raise
                except _IncomingCheckpointEvidenceError as error:
                    protocol = BackendProtocolError(
                        "child baseline checkpoint failed independent validation",
                        stage="optimization_checkpoint",
                    )
                    try:
                        native_receipt_failure(protocol, record_error=error)
                    except BaseException as record_error:
                        raise protocol from record_error
                    raise protocol from error
                except BaseException as error:
                    if isinstance(task_state.get("callback_error"), BaseException):
                        callback_error = task_state["callback_error"]
                        receipt_error = OptimizationProgressCallbackError(
                            "benchmark progress callback failed",
                            stage="progress_callback",
                            evidence={"error_type": type(callback_error).__name__},
                        )
                        try:
                            native_receipt_failure(receipt_error, record_error=callback_error)
                        except BaseException as record_error:
                            raise callback_error from record_error
                        raise callback_error
                    protocol = error if isinstance(error, SCNSimError) else BackendProtocolError(
                        "Julia benchmark task failed before a terminal outcome",
                        stage="benchmark_terminal",
                        evidence={"error_type": type(error).__name__, "message": str(error)},
                    )
                    try:
                        native_receipt_failure(protocol, record_error=error)
                    except BaseException as record_error:
                        raise error from record_error
                    raise

            assert terminal is not None and standard_attempt_sha is not None
            try:
                _write_logs(allocation.staging_directory, terminal.stdout_log, terminal.stderr_log)
                outcome_path = allocation.staging_directory / "outcome.json"
                if outcome_path.is_symlink() or not outcome_path.is_file():
                    raise BackendProtocolError("outcome.json is not a regular file", stage="outcome")
                outcome_raw = outcome_path.read_bytes()
                outcome = terminal.outcome
                if canonical_json_bytes(outcome) != outcome_raw:
                    raise BackendProtocolError("outcome.json is not canonical", stage="outcome")
                if (
                    outcome.get("runtime_semantic") != request.get("runtime_semantic")
                    or outcome.get("request_sha256") != request_sha
                    or outcome.get("attempt_sha256") != standard_attempt_sha
                    or outcome.get("status") not in {"success", "failure"}
                    or not isinstance(outcome.get("artifacts"), list)
                ):
                    raise BackendProtocolError("outcome envelope does not bind this execution", stage="outcome")
                expected_fields = {
                    "schema", "schema_version", "request_sha256", "attempt_sha256",
                    "runtime_semantic", "status", "artifacts",
                    "result_sha256" if outcome["status"] == "success" else "failure",
                }
                if set(outcome) != expected_fields:
                    raise BackendProtocolError("outcome envelope has unsupported fields", stage="outcome")
                _validate_terminal_staging_layout(
                    allocation.staging_directory, success=outcome["status"] == "success",
                )
                artifacts = list(outcome["artifacts"])
                outcome_sha = sha256_hex(outcome_raw)
                generation_records: list[tuple[int, str, Mapping[str, object]]] = []
                success_evidence: Mapping[str, object] | None = None
                if outcome["status"] == "success":
                    success_evidence = _validate_success_staging(
                        allocation.staging_directory, outcome, request, plan_document,
                        optimization_checkpoint=baseline,
                        expected_julia_threads=native_threads.julia_threads,
                        expected_blas_threads=native_threads.blas_threads,
                        collect_sweep_observations=sweep_operation,
                    )
                    generation_records = cast(
                        list[tuple[int, str, Mapping[str, object]]],
                        success_evidence["generation_records"],
                    )
                    if request["operation"] == "optimize_direct":
                        save_native_resume_checkpoint(artifacts)
                    receipt = _receipt(
                        request=request, plan_document=plan_document, request_sha=request_sha,
                        attempt_sha=standard_attempt_sha, outcome="success", artifacts=artifacts,
                        source_units=source_units, outcome_sha=outcome_sha,
                        result_sha=outcome["result_sha256"],
                    )
                    failure = None
                else:
                    if (allocation.staging_directory / "result.json").exists():
                        raise BackendProtocolError("failure outcome must not publish result.json", stage="outcome")
                    native_links, generation_records = cast(tuple[
                        list[dict[str, str]],
                        list[tuple[int, str, Mapping[str, object]]],
                    ], verified_generation_links(
                        allocation.staging_directory,
                        request_sha256=request_sha,
                        attempt_sha256=standard_attempt_sha,
                        expected_julia_threads=native_threads.julia_threads,
                        expected_blas_threads=native_threads.blas_threads,
                        include_records=True,
                    ))
                    if artifacts != native_links:
                        raise BackendProtocolError("failure outcome does not bind completed generation ledgers", stage="outcome")
                    failure = _validated_failure_record(
                        outcome.get("failure"), request["operation"], request=request,
                        plan=plan_document, require_optimization_context=True,
                        completed_generations=len(native_links),
                    )
                    receipt = _receipt(
                        request=request, plan_document=plan_document, request_sha=request_sha,
                        attempt_sha=standard_attempt_sha, outcome="failure", artifacts=artifacts,
                        source_units=source_units, outcome_sha=outcome_sha, failure=failure,
                    )
                    if request["operation"] == "optimize_direct":
                        save_native_resume_checkpoint(native_links)
            except KeyboardInterrupt as error:
                try:
                    native_receipt_failure(error, interrupted=True, stdout=terminal.stdout_log, stderr=terminal.stderr_log)
                except BaseException as record_error:
                    raise error from record_error
                raise
            except BaseException as error:
                protocol = error if isinstance(error, BackendProtocolError) else BackendProtocolError(
                    "Julia terminal evidence failed closed validation",
                    stage="outcome", evidence={"error_type": type(error).__name__, "message": str(error)},
                )
                try:
                    native_receipt_failure(protocol, stdout=terminal.stdout_log, stderr=terminal.stderr_log, record_error=error)
                except BaseException as record_error:
                    raise protocol from record_error
                raise protocol from error

            promote_native_receipt(receipt)
            final_directory = allocation.final_directory
            standard_receipt_path = final_directory / "receipt.json"
            if sweep_operation and failure is not None:
                verified_point_checkpoints = binding._point_checkpoints_with_payloads(request_sha)
                point_checkpoint_records = tuple(
                    checkpoint for checkpoint, _payload in verified_point_checkpoints
                )
            else:
                verified_point_checkpoints = ()
                point_checkpoint_records = (
                    binding.point_checkpoints(request_sha) if sweep_operation else ()
                )
            artifact_refs = _native_artifact_refs(
                workspace,
                request_path=request_directory / "request.json",
                final_directory=final_directory,
                observation_path=observation_path,
                point_checkpoints=point_checkpoint_records,
            )
            result_path = final_directory / "result.json"
            if observation_path.is_file():
                sidecar_ref = next(
                    item for item in artifact_refs
                    if item["role"] == "original_julia_timing_sidecar"
                )
                sidecar = json.loads(observation_path.read_text(encoding="utf-8"))
                if (
                    sidecar.get("schema") != "scnsim.benchmark_whole_timing"
                    or sidecar.get("benchmark_request_sha256") != prepared.request_sha256
                    or sidecar.get("mesh") != declaration["benchmark"]["mesh"]
                    or sidecar.get("cpu_threads_requested") != cpu_threads
                    or sidecar.get("julia_threads_requested") != native_threads.julia_threads
                    or sidecar.get("julia_blas_threads_requested") != native_threads.blas_threads
                    or sidecar.get("julia_threads") != native_threads.julia_threads
                    or sidecar.get("blas_threads") != native_threads.blas_threads
                ):
                    raise BackendProtocolError("whole-Julia timing sidecar does not bind the observed task", stage="benchmark_sidecar")
                _event(workspace, task_id=task_id, kind="timing", payload={
                    "attempt_id": attempt_id, "sidecar": sidecar_ref,
                    "julia_version": sidecar["julia_version"],
                    "julia_threads": sidecar["julia_threads"], "blas_threads": sidecar["blas_threads"],
                    "observations": sidecar["observations"],
                })
            storage.update_attempt(
                workspace, task_id=task_id, attempt_id=attempt_id,
                status="success" if failure is None else "failure",
                failure=None if failure is None else failure,
                artifacts=tuple(artifact_refs),
            )
            if failure is None:
                result_doc = cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], success_evidence)["result"],
                )
                result_ref = next(
                    item for item in artifact_refs if item["role"] == "original_julia_result"
                )
                numerical_observations = _native_numerical_observations(
                    workspace=workspace,
                    result_document=result_doc,
                    result_artifact=result_ref,
                    artifact_refs=artifact_refs,
                    generation_records=generation_records,
                    baseline=baseline,
                    request=request,
                    direct_arrays=cast(
                        Mapping[str, object] | None,
                        success_evidence["direct_arrays"],
                    ),
                    sweep_records=cast(
                        list[Mapping[str, object]],
                        success_evidence["sweep_records"],
                    ),
                )
                completed_payload: dict[str, object] = {
                    "attempt_id": attempt_id,
                    "result_sha256": outcome["result_sha256"],
                    "receipt_sha256": sha256(standard_receipt_path.read_bytes()).hexdigest(),
                    "result": result_ref,
                    "result_kind": result_doc["result_kind"],
                }
                if numerical_observations is not None:
                    completed_payload["numerical_observations"] = numerical_observations
                _event(workspace, task_id=task_id, kind="completed", payload=completed_payload, progress=progress)
                task_state["terminal"] = True
            else:
                failure_payload: dict[str, object] = {
                    "attempt_id": attempt_id, "failure": failure,
                }
                observation_error: BaseException | None = None
                try:
                    numerical_observations = _native_numerical_observations(
                        workspace=workspace,
                        result_document=None,
                        result_artifact=None,
                        artifact_refs=artifact_refs,
                        generation_records=generation_records,
                        point_checkpoints=verified_point_checkpoints,
                        request=request,
                        baseline=baseline,
                    )
                except Exception as error:
                    observation_error = error
                    failure_payload["numerical_observation_error"] = _exception_record(error)
                else:
                    if numerical_observations is not None:
                        failure_payload["numerical_observations"] = numerical_observations
                _event(workspace, task_id=task_id, kind="failed", payload=failure_payload)
                task_state["terminal"] = True
                native_error = _error_from_record(failure)
                if observation_error is not None:
                    raise native_error from observation_error
                raise native_error
    except KeyboardInterrupt as error:
        _record_task_failure(workspace, task_id=task_id, attempt_id=attempt_id, error=error, interrupted=True)
        _add_task_interval(timing, "task_end_to_end", task_id, task_started, timing.mark(), attempt_id=attempt_id)
        task_state["terminal"] = True
        raise
    except BaseException as error:
        if not task_state.get("terminal"):
            _record_task_failure(workspace, task_id=task_id, attempt_id=attempt_id, error=error)
        _add_task_interval(timing, "task_end_to_end", task_id, task_started, timing.mark(), attempt_id=attempt_id)
        task_state["terminal"] = True
        raise
    else:
        _add_task_interval(timing, "task_end_to_end", task_id, task_started, timing.mark(), attempt_id=attempt_id)


def run_benchmark(
    *,
    run,
    ref,
    spec,
    prepared: PreparedBenchmark,
    workspace: str | os.PathLike[str],
    progress=None,
    resume_from: Mapping[str, object] | None = None,
    timing,
) -> None:
    """Run the declared serial arm/profile/sample tasks without result lookup."""
    del run, ref, spec
    root = Path(workspace).expanduser().resolve(strict=False)
    declaration = prepared.declaration()
    policy = declaration["benchmark"]
    storage.initialize_record(root, prepared=prepared, clock_binding=timing.clock_binding)

    def record_setup_error(phase: str, error: BaseException) -> None:
        try:
            storage.record_execution_failure(
                root,
                phase=phase,
                benchmark_sha256=prepared.request_sha256,
                plan_sha256=str(declaration["plan_sha256"]),
                source_analysis_sha256=str(declaration["source_analysis_sha256"]),
                error=error,
            )
        except BaseException as record_error:
            raise error from record_error

    try:
        input_artifacts = _input_artifacts(root, prepared)
    except BaseException as error:
        record_setup_error("benchmark_input_publication", error)
        raise

    try:
        if resume_from is None:
            choices = tuple(
                (str(arm), int(threads), sample, None, None)
                for arm in policy["arms"]
                for threads in policy["cpu_threads"]
                for sample in range(int(policy["repeats"]))
            )
        else:
            resumed_task, checkpoint_document = _read_resume(root, prepared, resume_from)
            arm = str(resumed_task["arm"])
            threads = int(resumed_task["environment"]["cpu_threads_requested"])
            sample = int(resumed_task["sample"])
            if (
                arm not in policy["arms"]
                or threads not in policy["cpu_threads"]
                or sample < 0
                or sample >= int(policy["repeats"])
                or prepared.analysis.request()["operation"] != "optimize_direct"
            ):
                raise EvidenceIntegrityError(
                    "benchmark checkpoint does not select a declared optimization task sample",
                    stage="benchmark_checkpoint",
                )
            choices = ((arm, threads, sample, resume_from, checkpoint_document),)
    except BaseException as error:
        record_setup_error("benchmark_resume_selection", error)
        raise

    runtime = None
    wrapper = None
    with ExitStack() as resources_stack:
        if any(arm == "original_julia" for arm, *_ in choices):
            try:
                with timing.span("shared_julia_runtime_preparation"):
                    runtime = prepare_runtime(feature="Benchmark original_julia")
                    project, _, _ = resources_stack.enter_context(packaged_julia_resources())
                    wrapper = project / "bin" / "scnsim_benchmark.jl"
                    if wrapper.is_symlink() or not wrapper.is_file():
                        raise FileNotFoundError(f"packaged whole-task Julia entrypoint is missing: {wrapper}")
            except BaseException as error:
                record_setup_error("shared_julia_runtime_preparation", error)
                raise

        for arm, threads, sample, selected_reference, selected_checkpoint in choices:
            if arm == "original_julia":
                if runtime is None or wrapper is None:
                    raise RuntimeError("original Julia task lacks its prepared packaged runtime")
                _native_task(
                    workspace=root,
                    prepared=prepared,
                    input_artifacts=input_artifacts,
                    arm=arm,
                    sample=sample,
                    cpu_threads=threads,
                    progress=progress,
                    timing=timing,
                    prepared_runtime=runtime,
                    wrapper=wrapper,
                    resume_reference=selected_reference,
                    resume_document=selected_checkpoint,
                )
            else:
                _python_task(
                    workspace=root,
                    prepared=prepared,
                    input_artifacts=input_artifacts,
                    arm=arm,
                    sample=sample,
                    cpu_threads=threads,
                    progress=progress,
                    timing=timing,
                    resume_reference=selected_reference,
                    resume_document=selected_checkpoint,
                )


def _record_report_failure(
    workspace: Path,
    timing,
    *,
    source_json: Mapping[str, object] | None,
    stage: str,
    error: BaseException,
) -> None:
    started = timing.mark()
    storage.record_report(workspace, {
        "status": "failure",
        "stage": stage,
        "source_json": None if source_json is None else dict(source_json),
        "error": _exception_record(error),
    })
    timing.add_interval(
        "report_record_seal",
        task_id="benchmark",
        start_tick_ns=started,
        end_tick_ns=timing.mark(),
        details={"status": "failure", "stage": stage},
    )
    _persist_measurement_delta(workspace, timing)


def finish_benchmark(
    workspace: str | os.PathLike[str],
    timing,
) -> BenchmarkResult:
    """Persist exact timing/report observations and return the final manifest."""
    root = Path(workspace).expanduser().resolve(strict=False)
    _persist_measurement_delta(root, timing)

    source_json = None
    snapshot_started = timing.mark()
    try:
        source = storage.open_record(root)
        source_bytes = source.manifest_bytes
        source_sha = sha256(source_bytes).hexdigest()
        source_json = storage.write_artifact(
            root,
            f"reports/report-source-{source_sha}.json",
            source_bytes,
            role="benchmark_json_report_source",
        )
    except BaseException as error:
        timing.add_interval(
            "report_json_snapshot",
            task_id="benchmark",
            start_tick_ns=snapshot_started,
            end_tick_ns=timing.mark(),
            details={"status": "failure", "error_type": type(error).__name__},
        )
        try:
            _record_report_failure(root, timing, source_json=None, stage="json_snapshot", error=error)
        except BaseException as record_error:
            raise error from record_error
        raise
    else:
        timing.add_interval(
            "report_json_snapshot",
            task_id="benchmark",
            start_tick_ns=snapshot_started,
            end_tick_ns=timing.mark(),
            details={"status": "success", "artifact_sha256": source_json["sha256"]},
        )

    render_started = timing.mark()
    try:
        html = source.to_html()
    except BaseException as error:
        timing.add_interval(
            "report_render",
            task_id="benchmark",
            start_tick_ns=render_started,
            end_tick_ns=timing.mark(),
            details={"status": "failure", "error_type": type(error).__name__},
        )
        try:
            _record_report_failure(root, timing, source_json=source_json, stage="render", error=error)
        except BaseException as record_error:
            raise error from record_error
        raise
    else:
        timing.add_interval(
            "report_render",
            task_id="benchmark",
            start_tick_ns=render_started,
            end_tick_ns=timing.mark(),
            details={"status": "success"},
        )

    html_ref = None
    publication_started = timing.mark()
    try:
        html_bytes = html.encode("utf-8")
        html_ref = storage.write_artifact(
            root,
            f"reports/report-{sha256(html_bytes).hexdigest()}.html",
            html_bytes,
            role="benchmark_html_report",
        )
    except BaseException as error:
        timing.add_interval(
            "report_html_publication",
            task_id="benchmark",
            start_tick_ns=publication_started,
            end_tick_ns=timing.mark(),
            details={"status": "failure", "error_type": type(error).__name__},
        )
        try:
            _record_report_failure(root, timing, source_json=source_json, stage="html_publication", error=error)
        except BaseException as record_error:
            raise error from record_error
        raise
    else:
        timing.add_interval(
            "report_html_publication",
            task_id="benchmark",
            start_tick_ns=publication_started,
            end_tick_ns=timing.mark(),
            details={"status": "success", "artifact_sha256": html_ref["sha256"]},
        )

    seal_started = timing.mark()
    storage.record_report(root, {
        "status": "success",
        "source_json": dict(source_json),
        "html": dict(html_ref),
    })
    timing.add_interval(
        "report_record_seal",
        task_id="benchmark",
        start_tick_ns=seal_started,
        end_tick_ns=timing.mark(),
        details={"status": "success"},
    )
    _persist_measurement_delta(root, timing)
    return storage.open_record(root)
