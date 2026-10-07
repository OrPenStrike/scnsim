"""Durable benchmark observations and checkpoint artifacts."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..errors import EvidenceIntegrityError
from ..workspace.primitives import _inside
from ..workspace.storage import _atomic_write, _load_canonical
from ..workspace.store import _workspace_lock
from .identity import checkpoint_seal
from .models import BenchmarkResult, Measurement
from .prepared import record_bytes


_RECORD_NAME = "benchmark.json"
_RECORD_SCHEMA = "scnsim.benchmark_record"
_RECORD_VERSION = 1


def _integrity(message: str, **evidence: object) -> EvidenceIntegrityError:
    return EvidenceIntegrityError(message, stage="benchmark_record", evidence=evidence)


def _root_path(workspace: str | os.PathLike[str]) -> Path:
    return Path(workspace).expanduser().resolve(strict=False)


def _record_bytes(value: Mapping[str, object]) -> bytes:
    """Use the shared exact scalar encoding for durable benchmark records."""
    return record_bytes(dict(value))


def _measurement_document(
    measurement: Measurement,
    *,
    clock_binding: Mapping[str, object],
) -> dict[str, object]:
    return {
        "task_id": measurement.task_id,
        "stage": measurement.stage,
        "start_ns": measurement.start_ns,
        "duration_ns": measurement.duration_ns,
        "clock": dict(clock_binding),
        "counts": dict(measurement.counts),
        "memory_bytes": dict(measurement.memory_bytes),
        "details": json.loads(measurement.detail_bytes),
    }


def _read_document(root: Path) -> dict[str, Any]:
    path = _inside(root, _RECORD_NAME)
    document = _load_canonical(path)
    if document.get("schema") != _RECORD_SCHEMA or document.get("schema_version") != _RECORD_VERSION:
        raise _integrity("Unsupported benchmark record envelope.", path=str(path))
    declaration = document.get("declaration")
    benchmark_sha = document.get("benchmark_sha256")
    if declaration is None and benchmark_sha is None:
        if document.get("preparation_failure", {}).get("status") != "failure":
            raise _integrity("Unbound benchmark record lacks its explicit preparation failure.", path=str(path))
        if document.get("tasks") != [] or document.get("reports") != []:
            raise _integrity("Unbound preparation failure cannot claim numerical tasks or reports.", path=str(path))
        return document
    if not isinstance(declaration, dict) or not isinstance(benchmark_sha, str):
        raise _integrity("Benchmark record lacks its immutable declaration identity.", path=str(path))
    declaration_bytes = _record_bytes(declaration)
    if sha256(declaration_bytes).hexdigest() != benchmark_sha:
        raise _integrity("Benchmark declaration does not match its recorded identity.", path=str(path))
    return document


def open_record(workspace: str | os.PathLike[str]) -> BenchmarkResult:
    """Read a complete or partial record without binding or cleaning a workspace."""
    root = _root_path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise _integrity("Benchmark workspace is missing or symlinked.", path=str(root))
    document = _read_document(root)
    return BenchmarkResult.from_document(root, document)


def initialize_record(
    workspace: str | os.PathLike[str],
    *,
    prepared: object,
    clock_binding: Mapping[str, object],
) -> Path:
    """Create the bound root record or reopen it when the declaration matches."""
    root = _root_path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    declaration = prepared.declaration()  # type: ignore[attr-defined]
    benchmark_sha = prepared.request_sha256  # type: ignore[attr-defined]
    plan_sha = declaration["plan_sha256"]
    source_sha = declaration["source_analysis_sha256"]
    record = {
        "schema": _RECORD_SCHEMA,
        "schema_version": _RECORD_VERSION,
        "benchmark_sha256": benchmark_sha,
        "declaration": declaration,
        "plan_sha256": plan_sha,
        "source_analysis_sha256": source_sha,
        "clock": dict(clock_binding),
        "measurements": [],
        "tasks": [],
        "reports": [],
    }
    target = _inside(root, _RECORD_NAME)
    with _workspace_lock(root, exclusive=True):
        if target.is_symlink():
            raise _integrity("Benchmark record path must not be a symlink.", path=str(target))
        if target.exists():
            existing = _read_document(root)
            if existing.get("benchmark_sha256") != benchmark_sha:
                raise _integrity(
                    "Benchmark workspace is already bound to a different declaration.",
                    expected=existing.get("benchmark_sha256"),
                    supplied=benchmark_sha,
                )
            return root
        _atomic_write(target, _record_bytes(record))
    return root


def _mutate(workspace: str | os.PathLike[str], action: Callable[[dict[str, Any]], object]) -> object:
    root = _root_path(workspace)
    with _workspace_lock(root, exclusive=True):
        record = _read_document(root)
        result = action(record)
        _atomic_write(_inside(root, _RECORD_NAME), _record_bytes(record))
    return result


def _task(record: dict[str, Any], task_id: str) -> dict[str, Any]:
    for item in record["tasks"]:
        if item["task_id"] == task_id:
            return item
    raise _integrity("Benchmark task identity is not recorded.", task_id=task_id)


def task_record(workspace: str | os.PathLike[str], task_id: str) -> dict[str, Any]:
    """Return one stored task record without selecting or mutating other tasks."""
    root = _root_path(workspace)
    with _workspace_lock(root, exclusive=False):
        record = _read_document(root)
        return json.loads(_record_bytes(_task(record, task_id)))


def write_artifact(
    workspace: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
    payload: bytes,
    *,
    role: str,
) -> dict[str, object]:
    """Publish immutable bytes below the benchmark workspace."""
    root = _root_path(workspace)
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise _integrity("Benchmark artifact path must be workspace-relative.", path=str(relative))
    target = _inside(root, relative.as_posix())
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise _integrity("Benchmark artifact path traverses a symlink.", path=str(current))
        current.mkdir(exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise _integrity("Benchmark artifact path already holds different bytes.", path=str(target))
    else:
        _atomic_write(target, payload)
    return {
        "path": relative.as_posix(),
        "role": role,
        "byte_length": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def register_task(
    workspace: str | os.PathLike[str],
    task: Mapping[str, object],
) -> None:
    """Append one canonical task declaration without replacing prior evidence."""
    value = json.loads(_record_bytes(task))
    required = {
        "task_id", "request_sha256", "arm", "sample", "attempts", "events",
        "measurements", "environment", "artifacts",
    }
    if set(value) != required:
        raise _integrity("Benchmark task record has an unexpected field set.", fields=sorted(value))

    def apply(record: dict[str, Any]) -> None:
        if any(item["task_id"] == value["task_id"] for item in record["tasks"]):
            prior = next(item for item in record["tasks"] if item["task_id"] == value["task_id"])
            if prior != value:
                raise _integrity("Benchmark task identity was rebound.", task_id=value["task_id"])
            return
        record["tasks"].append(value)

    _mutate(workspace, apply)


def ensure_task(
    workspace: str | os.PathLike[str],
    task: Mapping[str, object],
) -> dict[str, Any]:
    """Register a task once, or return its exact existing environment binding."""
    value = json.loads(_record_bytes(task))
    required = {
        "task_id", "request_sha256", "arm", "sample", "attempts", "events",
        "measurements", "environment", "artifacts",
    }
    if set(value) != required:
        raise _integrity("Benchmark task record has an unexpected field set.", fields=sorted(value))

    def apply(record: dict[str, Any]) -> dict[str, Any]:
        prior = next((item for item in record["tasks"] if item["task_id"] == value["task_id"]), None)
        if prior is None:
            record["tasks"].append(value)
            return value
        stable_prior = {key: prior[key] for key in ("task_id", "request_sha256", "arm", "sample")}
        stable_value = {key: value[key] for key in ("task_id", "request_sha256", "arm", "sample")}
        if stable_prior != stable_value or prior["environment"].get("environment_sha256") != value["environment"].get("environment_sha256"):
            raise _integrity("Benchmark task identity was rebound to another environment.", task_id=value["task_id"])
        return prior

    return _mutate(workspace, apply)  # type: ignore[return-value]


def record_task_launch_failure(
    workspace: str | os.PathLike[str],
    *,
    arm: str,
    sample: int,
    attempt_id: str,
    benchmark_sha256: str,
    source_analysis_sha256: str,
    error: BaseException,
    request_artifact: Mapping[str, object] | None,
) -> None:
    """Record a real launch failure when no child environment identity exists."""
    error_record = _error_document(error)
    failure = {
        "task_id": None,
        "request_sha256": None,
        "environment_sha256": None,
        "arm": arm,
        "sample": sample,
        "attempt_id": attempt_id,
        "benchmark_sha256": benchmark_sha256,
        "source_analysis_sha256": source_analysis_sha256,
        "request_artifact": None if request_artifact is None else json.loads(_record_bytes(request_artifact)),
        "status": "failure",
        "error": error_record,
    }

    def apply(record: dict[str, Any]) -> None:
        record.setdefault("task_launch_failures", []).append(failure)

    _mutate(workspace, apply)


def _error_document(error: BaseException) -> dict[str, object]:
    if isinstance(error, Exception) and hasattr(error, "kind") and hasattr(error, "stage"):
        value: dict[str, object] = {
            "type": type(error).__name__,
            "kind": error.kind,
            "category": error.category,
            "stage": error.stage,
            "message": str(error),
        }
        if hasattr(error, "evidence"):
            value["evidence"] = _error_evidence(dict(error.evidence))
        return value
    return {"type": type(error).__name__, "module": type(error).__module__, "message": str(error)}


def _error_evidence(value: object) -> object:
    """Render path-like evidence explicitly while leaving canonical checks strict."""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _error_evidence(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_error_evidence(item) for item in value]
    return value


def record_execution_failure(
    workspace: str | os.PathLike[str],
    *,
    phase: str,
    benchmark_sha256: str,
    plan_sha256: str,
    source_analysis_sha256: str,
    error: BaseException,
) -> None:
    """Persist a shared execution-preparation failure without a fake task id."""
    failure = {
        "phase": phase,
        "status": "failure",
        "benchmark_sha256": benchmark_sha256,
        "plan_sha256": plan_sha256,
        "source_analysis_sha256": source_analysis_sha256,
        "error": _error_document(error),
    }

    def apply(record: dict[str, Any]) -> None:
        record.setdefault("execution_failures", []).append(failure)

    _mutate(workspace, apply)


def record_callback_failure(
    workspace: str | os.PathLike[str],
    *,
    task_id: str,
    sequence: int,
    event_kind: str,
    error: BaseException,
) -> None:
    """Retain a user callback exception separately from the observed task event."""
    failure = {
        "task_id": task_id,
        "event_sequence": sequence,
        "event_kind": event_kind,
        "error": _error_document(error),
    }

    def apply(record: dict[str, Any]) -> None:
        _task(record, task_id)
        record.setdefault("callback_failures", []).append(failure)

    _mutate(workspace, apply)


def append_event(
    workspace: str | os.PathLike[str],
    *,
    task_id: str,
    kind: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Append the next task-local event and atomically publish the root record."""
    event_payload = json.loads(_record_bytes(payload))

    def apply(record: dict[str, Any]) -> dict[str, object]:
        task = _task(record, task_id)
        event = {
            "task_id": task_id,
            "sequence": len(task["events"]),
            "kind": kind,
            "payload": event_payload,
        }
        task["events"].append(event)
        return event

    return _mutate(workspace, apply)  # type: ignore[return-value]


def append_measurement(
    workspace: str | os.PathLike[str],
    measurement: Measurement,
    *,
    clock_binding: Mapping[str, object],
) -> None:
    """Append one timing observation to its task or benchmark-wide collection."""
    value = _measurement_document(measurement, clock_binding=clock_binding)

    def apply(record: dict[str, Any]) -> None:
        if measurement.task_id == "benchmark":
            record["measurements"].append(value)
        else:
            _task(record, measurement.task_id)["measurements"].append(value)

    _mutate(workspace, apply)


def append_measurements(
    workspace: str | os.PathLike[str],
    measurements: tuple[Measurement, ...],
    *,
    clock_binding: Mapping[str, object],
) -> None:
    """Publish a recorder snapshot under one lock and one atomic root replace."""
    values = tuple(_measurement_document(item, clock_binding=clock_binding) for item in measurements)

    def apply(record: dict[str, Any]) -> None:
        for measurement, value in zip(measurements, values, strict=True):
            if measurement.task_id == "benchmark":
                collection = record["measurements"]
            else:
                collection = _task(record, measurement.task_id)["measurements"]
            if value not in collection:
                collection.append(value)

    _mutate(workspace, apply)


def begin_attempt(
    workspace: str | os.PathLike[str],
    *,
    task_id: str,
    attempt_id: str,
    resume_from: Mapping[str, object] | None = None,
) -> None:
    """Persist an allocated attempt before the child is authorized to advance."""
    resume = json.loads(_record_bytes(resume_from)) if resume_from is not None else None

    def apply(record: dict[str, Any]) -> None:
        task = _task(record, task_id)
        if any(item["attempt_id"] == attempt_id for item in task["attempts"]):
            raise _integrity("Benchmark attempt identifier is already recorded.", attempt_id=attempt_id)
        task["attempts"].append({
            "attempt_id": attempt_id,
            "status": "allocated",
            "resume_from": resume,
            "artifacts": [],
            "failure": None,
            "interruption": None,
        })

    _mutate(workspace, apply)


def update_attempt(
    workspace: str | os.PathLike[str],
    *,
    task_id: str,
    attempt_id: str,
    status: str,
    failure: Mapping[str, object] | None = None,
    interruption: Mapping[str, object] | None = None,
    artifacts: tuple[Mapping[str, object], ...] = (),
    checkpoint: Mapping[str, object] | None = None,
) -> None:
    """Publish terminal attempt state while preserving its earlier evidence."""
    failure_doc = json.loads(_record_bytes(failure)) if failure is not None else None
    interrupt_doc = json.loads(_record_bytes(interruption)) if interruption is not None else None
    artifact_docs = [json.loads(_record_bytes(item)) for item in artifacts]
    checkpoint_doc = json.loads(_record_bytes(checkpoint)) if checkpoint is not None else None

    def apply(record: dict[str, Any]) -> None:
        task = _task(record, task_id)
        attempt = next((item for item in task["attempts"] if item["attempt_id"] == attempt_id), None)
        if attempt is None:
            raise _integrity("Benchmark attempt is not recorded.", task_id=task_id, attempt_id=attempt_id)
        attempt["status"] = status
        attempt["failure"] = failure_doc
        attempt["interruption"] = interrupt_doc
        if checkpoint_doc is not None:
            attempt["checkpoint"] = checkpoint_doc
        for artifact in artifact_docs:
            if artifact not in attempt["artifacts"]:
                attempt["artifacts"].append(artifact)
                if artifact not in task["artifacts"]:
                    task["artifacts"].append(artifact)

    _mutate(workspace, apply)


def publish_checkpoint(
    workspace: str | os.PathLike[str],
    *,
    task_id: str,
    attempt_id: str,
    checkpoint_bytes: bytes,
) -> dict[str, object]:
    """Durably publish and seal canonical checkpoint bytes before returning."""
    try:
        checkpoint = json.loads(checkpoint_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _integrity("Benchmark checkpoint is not JSON.", error=str(error)) from error
    if not isinstance(checkpoint, dict) or _record_bytes(checkpoint) != checkpoint_bytes:
        raise _integrity("Benchmark checkpoint bytes are not canonical JSON.")
    checkpoint_sha = sha256(checkpoint_bytes).hexdigest()
    root = _root_path(workspace)
    result: dict[str, object] = {}
    with _workspace_lock(root, exclusive=True):
        record = _read_document(root)
        task = _task(record, task_id)
        attempt = next((item for item in task["attempts"] if item["attempt_id"] == attempt_id), None)
        if attempt is None:
            raise _integrity("Benchmark checkpoint attempt is not recorded.", task_id=task_id, attempt_id=attempt_id)
        seal_doc = checkpoint_seal(
            task_id=task_id,
            request_sha256=task["request_sha256"],
            arm=task["arm"],
            sample=task["sample"],
            environment_sha256=task["environment"]["environment_sha256"],
            attempt_id=attempt_id,
            checkpoint_sha256=checkpoint_sha,
            byte_length=len(checkpoint_bytes),
        )
        checkpoint_rel = f"tasks/{task_id}/attempts/{attempt_id}/checkpoints/{checkpoint_sha}.json"
        seal_rel = checkpoint_rel[:-5] + ".seal.json"
        checkpoint_path = _inside(root, checkpoint_rel)
        seal_path = _inside(root, seal_rel)
        for path in (checkpoint_path, seal_path):
            current = root
            for part in path.relative_to(root).parts[:-1]:
                current = current / part
                if current.is_symlink():
                    raise _integrity("Benchmark checkpoint path traverses a symlink.", path=str(current))
                current.mkdir(exist_ok=True)
        if checkpoint_path.exists():
            if checkpoint_path.is_symlink() or checkpoint_path.read_bytes() != checkpoint_bytes:
                raise _integrity("Benchmark checkpoint path already holds different bytes.", path=str(checkpoint_path))
        else:
            _atomic_write(checkpoint_path, checkpoint_bytes)
        seal_bytes = _record_bytes(seal_doc)
        if seal_path.exists():
            if seal_path.is_symlink() or seal_path.read_bytes() != seal_bytes:
                raise _integrity("Benchmark checkpoint seal conflicts with existing evidence.", path=str(seal_path))
        else:
            _atomic_write(seal_path, seal_bytes)
        checkpoint_artifact = {
            "path": checkpoint_rel,
            "role": "checkpoint",
            "byte_length": len(checkpoint_bytes),
            "sha256": checkpoint_sha,
        }
        seal_artifact = {
            "path": seal_rel,
            "role": "checkpoint_seal",
            "byte_length": len(seal_bytes),
            "sha256": sha256(seal_bytes).hexdigest(),
        }
        checkpoint_ref = {
            "task_id": task_id,
            "request_sha256": task["request_sha256"],
            "arm": task["arm"],
            "sample": task["sample"],
            "environment_sha256": task["environment"]["environment_sha256"],
            "attempt_id": attempt_id,
            "checkpoint": checkpoint_artifact,
            "seal": seal_artifact,
            "seal_sha256": seal_doc["seal_sha256"],
        }
        attempt["checkpoint"] = checkpoint_ref
        for artifact in (checkpoint_artifact, seal_artifact):
            if artifact not in attempt["artifacts"]:
                attempt["artifacts"].append(artifact)
            if artifact not in task["artifacts"]:
                task["artifacts"].append(artifact)
        _atomic_write(_inside(root, _RECORD_NAME), _record_bytes(record))
        result = checkpoint_ref
    return result


def read_checkpoint(
    workspace: str | os.PathLike[str],
    reference: Mapping[str, object],
    *,
    expected_task_id: str,
    expected_request_sha256: str,
    expected_arm: str,
    expected_sample: int,
    expected_environment_sha256: str,
) -> bytes:
    """Read only the named, sealed checkpoint after exact identity matching."""
    root = _root_path(workspace)
    if (
        reference["task_id"] != expected_task_id
        or reference["request_sha256"] != expected_request_sha256
        or reference["arm"] != expected_arm
        or reference["sample"] != expected_sample
        or reference["environment_sha256"] != expected_environment_sha256
    ):
        raise _integrity("Checkpoint binding does not match the requested task/sample.")
    checkpoint_ref = reference["checkpoint"]
    seal_ref = reference["seal"]
    checkpoint_path = _inside(root, checkpoint_ref["path"])
    seal_path = _inside(root, seal_ref["path"])
    checkpoint_bytes = checkpoint_path.read_bytes()
    seal_bytes = seal_path.read_bytes()
    if (
        len(checkpoint_bytes) != checkpoint_ref["byte_length"]
        or sha256(checkpoint_bytes).hexdigest() != checkpoint_ref["sha256"]
        or len(seal_bytes) != seal_ref["byte_length"]
        or sha256(seal_bytes).hexdigest() != seal_ref["sha256"]
    ):
        raise _integrity("Checkpoint bytes do not match the recorded artifact identities.")
    checkpoint = json.loads(checkpoint_bytes)
    seal = json.loads(seal_bytes)
    expected_seal = checkpoint_seal(
        task_id=expected_task_id,
        request_sha256=expected_request_sha256,
        arm=expected_arm,
        sample=expected_sample,
        environment_sha256=expected_environment_sha256,
        attempt_id=reference["attempt_id"],
        checkpoint_sha256=sha256(checkpoint_bytes).hexdigest(),
        byte_length=len(checkpoint_bytes),
    )
    if _record_bytes(seal) != seal_bytes or seal != expected_seal or reference["seal_sha256"] != seal["seal_sha256"]:
        raise _integrity("Checkpoint seal does not bind the named task and bytes.")
    if not isinstance(checkpoint, dict) or _record_bytes(checkpoint) != checkpoint_bytes:
        raise _integrity("Checkpoint bytes are not canonical JSON.")
    return checkpoint_bytes


def record_report(
    workspace: str | os.PathLike[str],
    report: Mapping[str, object],
) -> None:
    """Append a rendering artifact or rendering error independently of tasks."""
    value = json.loads(_record_bytes(report))

    def apply(record: dict[str, Any]) -> None:
        record["reports"].append(value)

    _mutate(workspace, apply)


def record_preparation_failure(
    workspace: str | os.PathLike[str],
    *,
    error: BaseException,
    timing: object,
    plan_sha256: str | None = None,
) -> None:
    """Preserve pre-task failure timing without inventing a prepared identity."""
    invocation_id = str(uuid.uuid4())
    if isinstance(error, Exception) and hasattr(error, "kind") and hasattr(error, "stage"):
        error_record: dict[str, object] = {
            "type": type(error).__name__,
            "kind": error.kind,
            "category": error.category,
            "stage": error.stage,
            "message": str(error),
        }
    else:
        error_record = {
            "type": type(error).__name__,
            "module": type(error).__module__,
            "message": str(error),
        }
    root = _root_path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    target = _inside(root, _RECORD_NAME)
    with _workspace_lock(root, exclusive=True):
        if target.is_symlink():
            raise _integrity("Benchmark record path must not be a symlink.", path=str(target))
        record = _read_document(root) if target.exists() else None
        clock_binding = timing.clock_binding  # type: ignore[attr-defined]
        measurements = tuple(
            _measurement_document(item, clock_binding=clock_binding)
            for item in timing.measurements  # type: ignore[attr-defined]
        )
        failure = {
            "invocation_id": invocation_id,
            "clock": clock_binding,
            "status": "failure",
            "prepared_sha256": None,
            "declaration": None,
            "request_sha256": None,
            "plan_sha256": plan_sha256,
            "source_analysis_sha256": None,
            "error": error_record,
            "measurements": measurements,
        }
        if target.exists():
            assert record is not None
            if record.get("benchmark_sha256") is None:
                prior = record["preparation_failure"]
                if prior.get("invocation_id") == invocation_id:
                    return
                record.setdefault("preparation_failures", []).append(failure)
            else:
                record.setdefault("preparation_failures", []).append(failure)
            _atomic_write(target, _record_bytes(record))
            return
        record = {
            "schema": _RECORD_SCHEMA,
            "schema_version": _RECORD_VERSION,
            "benchmark_sha256": None,
            "declaration": None,
            "plan_sha256": plan_sha256,
            "source_analysis_sha256": None,
            "clock": clock_binding,
            "measurements": [],
            "tasks": [],
            "reports": [],
            "preparation_failure": failure,
        }
        _atomic_write(target, _record_bytes(record))
