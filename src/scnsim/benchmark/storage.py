"""Operation storage facade and historical readonly file projections.

Current writes belong to sqlite_storage.py. File-journal helpers below exist
only to read immutable historical records; they never migrate or publish them.
Domain expansion is shared so reference storage does not change numerical meaning.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, cast

if sys.platform in {"linux", "darwin"}:
    import fcntl

from ..errors import EvidenceIntegrityError
from ..workspace.primitives import _inside
from ..workspace.storage import _load_canonical
from ..workspace.store import _require_platform
from . import journal
from .identity import checkpoint_seal
from .models import BenchmarkResult, Measurement
from .prepared import record_bytes, record_document


_RECORD_NAME = "benchmark.json"
_RECORD_SCHEMA = "scnsim.benchmark_record"
_V1 = 1
_V2 = 2
_JOURNAL_MARKER = "$scnsim_benchmark_journal"
_OCCURRENCE_FIELDS = frozenset({
    "attempt_id", "candidate_key", "cache_hit", "evaluation_ordinal", "generation",
    "population_column", "latent_coordinates", "source_index", "origin",
    "continuation_t_f64",
})
_DIAGNOSTIC_EVENTS = frozenset({
    "timing", "population_observed", "evaluation", "progress", "operation_span",
})


def _integrity(message: str, **evidence: object) -> EvidenceIntegrityError:
    return EvidenceIntegrityError(message, stage="benchmark_record", evidence=evidence)


def _root_path(workspace: str | os.PathLike[str]) -> Path:
    return Path(workspace).expanduser().resolve(strict=False)


@contextmanager
def _benchmark_reader_lock(root: Path) -> Iterator[None]:
    """Share-lock an existing record without creating or opening it for write."""
    _require_platform()
    if root.is_symlink() or not root.is_dir():
        raise _integrity("Benchmark workspace is missing or symlinked.", path=str(root))
    lock_path = root / ".scnsim.lock"
    if lock_path.is_symlink():
        raise _integrity("Benchmark workspace lock must not be a symlink.", path=str(lock_path))
    descriptor = os.open(
        lock_path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    with os.fdopen(descriptor, "rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)  # type: ignore[name-defined]
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[name-defined]


def _record_bytes(value: Mapping[str, object]) -> bytes:
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
    if document.get("schema") != _RECORD_SCHEMA:
        raise _integrity("Unsupported benchmark record envelope.", path=str(path))
    version = document.get("schema_version")
    if version == _V1:
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
        if sha256(_record_bytes(declaration)).hexdigest() != benchmark_sha:
            raise _integrity("Benchmark declaration does not match its recorded identity.", path=str(path))
        return document
    if version != _V2:
        raise _integrity("Unsupported benchmark record version.", path=str(path), version=version)
    declaration = document.get("declaration")
    benchmark_sha = document.get("benchmark_sha256")
    if declaration is None and benchmark_sha is None:
        if not isinstance(document.get("tasks"), list) or document["tasks"]:
            raise _integrity("Unbound benchmark record cannot claim numerical task identities.", path=str(path))
        return document
    if not isinstance(declaration, dict) or not isinstance(benchmark_sha, str):
        raise _integrity("Benchmark record lacks its immutable declaration identity.", path=str(path))
    if sha256(_record_bytes(declaration)).hexdigest() != benchmark_sha:
        raise _integrity("Benchmark declaration does not match its recorded identity.", path=str(path))
    if not isinstance(document.get("tasks"), list):
        raise _integrity("Benchmark task identity index is malformed.", path=str(path))
    return document


def _require_v2(root: Path) -> dict[str, Any]:
    document = _read_document(root)
    if document.get("schema_version") != _V2:
        raise _integrity("Historical benchmark records are read-only.", path=str(_inside(root, _RECORD_NAME)))
    return document


def _task_binding(document: Mapping[str, object], task_id: str) -> dict[str, Any]:
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise _integrity("Benchmark task identity index is malformed.")
    for item in tasks:
        if isinstance(item, dict) and item.get("task_id") == task_id:
            return item
    raise _integrity("Benchmark task identity is not recorded.", task_id=task_id)


def _task_head(
    root: Path,
    binding: Mapping[str, object],
    *,
    durability_witness: journal._DurabilityWitness | None = None,
) -> dict[str, Any]:
    task_id = str(binding["task_id"])
    relative_path = f"tasks/{task_id}/journal/HEAD.json"
    head = journal.read_head(
        root,
        relative_path,
        schema=journal._TASK_HEAD,
        bind={"task_id": task_id, "request_sha256": binding["request_sha256"]},
    )
    _validate_head_tip(
        root, head, schema=journal._TASK_COMMIT,
        role="benchmark_task_commit",
        bind={"task_id": task_id, "request_sha256": binding["request_sha256"]},
    )
    if durability_witness is not None:
        # These paths have just passed the normal HEAD and tip readers while the
        # caller holds the exclusive lock. Record only their exact directory
        # name edges; nested attempt/value/evidence refs remain unseeded.
        durability_witness.remember_verified_file_path(relative_path)
        reference = head.get("commit")
        if isinstance(reference, Mapping):
            durability_witness.remember_verified_file_path(cast(str, reference["path"]))
    return head


def _global_head(root: Path, benchmark_sha256: str | None) -> dict[str, Any]:
    head = journal.read_head(
        root,
        "journal/global/HEAD.json",
        schema=journal._GLOBAL_HEAD,
        bind={"benchmark_sha256": benchmark_sha256},
    )
    _validate_head_tip(
        root, head, schema=journal._GLOBAL_COMMIT,
        role="benchmark_global_commit", bind={"benchmark_sha256": benchmark_sha256},
    )
    return head


def _validate_head_tip(
    root: Path,
    head: Mapping[str, object],
    *,
    schema: str,
    role: str,
    bind: Mapping[str, object],
) -> None:
    reference = head.get("commit")
    sequence = head.get("commit_sequence")
    if not isinstance(sequence, int) or sequence < 0:
        raise _integrity("Benchmark journal head commit sequence is invalid.")
    if reference is None:
        if sequence != 0:
            raise _integrity("Empty benchmark journal head has a nonzero commit sequence.")
        return
    if not isinstance(reference, Mapping) or sequence == 0:
        raise _integrity("Benchmark journal head predecessor is malformed.")
    commit = _read_domain_document(root, reference, schema=schema, role=role, bind=bind)
    if commit.get("commit_sequence") != sequence:
        raise _integrity("Benchmark journal head does not identify its committed predecessor.")


def _task_chain(root: Path, binding: Mapping[str, object], head: Mapping[str, object]) -> list[dict[str, Any]]:
    return journal.read_chain(
        root,
        head,
        schema=journal._TASK_COMMIT,
        role="benchmark_task_commit",
        bind={"task_id": binding["task_id"], "request_sha256": binding["request_sha256"]},
    )


def _global_chain(root: Path, manifest: Mapping[str, object], head: Mapping[str, object]) -> list[dict[str, Any]]:
    return journal.read_chain(
        root,
        head,
        schema=journal._GLOBAL_COMMIT,
        role="benchmark_global_commit",
        bind={"benchmark_sha256": manifest.get("benchmark_sha256")},
    )


def _read_value(root: Path, task_id: str, reference: Mapping[str, object]) -> dict[str, Any]:
    document = _read_domain_document(
        root,
        reference,
        schema="scnsim.benchmark_value",
        role="benchmark_value",
        bind={"task_id": task_id},
    )
    value = document.get("value")
    if not isinstance(value, dict):
        raise _integrity("Benchmark numerical value block is malformed.", path=reference.get("path"))
    return value


def _materialize_value_marker(root: Path, task_id: str, value: Mapping[str, object]) -> dict[str, object]:
    reference = value.get("reference")
    occurrence = value.get("occurrence", {})
    if not isinstance(reference, Mapping) or not isinstance(occurrence, Mapping):
        raise _integrity("Benchmark journal value reference is malformed.")
    result = _read_value(root, task_id, reference)
    result.update(dict(occurrence))
    return result


def _verify_checkpoint_file(
    root: Path,
    reference: Mapping[str, object],
    *,
    task_binding: Mapping[str, object],
) -> tuple[dict[str, Any], bytes]:
    checkpoint_ref = reference.get("checkpoint")
    seal_ref = reference.get("seal")
    if not isinstance(checkpoint_ref, Mapping) or not isinstance(seal_ref, Mapping):
        raise _integrity("Checkpoint artifact references are malformed.")
    checkpoint_bytes = _read_immutable(root, checkpoint_ref, role="checkpoint")
    seal_bytes = _read_immutable(root, seal_ref, role="checkpoint_seal")
    checkpoint = record_document(checkpoint_bytes)
    seal = record_document(seal_bytes)
    if not isinstance(checkpoint, dict) or not isinstance(seal, dict):
        raise _integrity("Checkpoint and seal artifacts must be JSON objects.")
    expected = checkpoint_seal(
        task_id=str(task_binding["task_id"]),
        request_sha256=str(task_binding["request_sha256"]),
        arm=str(task_binding["arm"]),
        sample=int(task_binding["sample"]),
        environment_sha256=str(task_binding["environment"]["environment_sha256"]),
        attempt_id=str(reference["attempt_id"]),
        checkpoint_sha256=sha256(checkpoint_bytes).hexdigest(),
        byte_length=len(checkpoint_bytes),
    )
    if (
        record_bytes(seal) != seal_bytes
        or record_bytes(checkpoint) != checkpoint_bytes
        or seal != expected
        or reference.get("seal_sha256") != seal.get("seal_sha256")
    ):
        raise _integrity("Checkpoint seal does not bind the named task and bytes.")
    return checkpoint, checkpoint_bytes


def _expand_payload(
    root: Path,
    task_id: str,
    task_binding: Mapping[str, object],
    kind: str,
    compact: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    marker = compact.get(_JOURNAL_MARKER)
    if not isinstance(marker, Mapping):
        result = dict(compact)
        observations = result.get("numerical_observations")
        if isinstance(observations, Mapping):
            result["numerical_observations"] = _expand_observations(root, task_id, observations)
        return result, None
    marker_kind = marker.get("kind")
    if marker_kind == "value":
        result = {key: value for key, value in compact.items() if key != _JOURNAL_MARKER}
        result.update(_materialize_value_marker(root, task_id, marker))
        return result, None
    if marker_kind == "barrier":
        return _expand_barrier(root, task_id, task_binding, kind, compact, marker)
    raise _integrity("Benchmark journal event marker has an unsupported kind.", kind=marker_kind)


def _expand_observations(
    root: Path,
    task_id: str,
    observations: Mapping[str, object],
) -> dict[str, object]:
    result = dict(observations)
    records_value = result.get("records", ())
    if isinstance(records_value, Sequence) and not isinstance(records_value, (str, bytes)):
        records: list[object] = []
        for raw in records_value:
            if not isinstance(raw, Mapping):
                records.append(raw)
                continue
            item = dict(raw)
            marker = item.pop(_JOURNAL_MARKER, None)
            if isinstance(marker, Mapping) and marker.get("kind") == "value":
                item["value"] = _materialize_value_marker(root, task_id, marker)
            records.append(item)
        result["records"] = records
    winner = result.get("winner")
    if isinstance(winner, Mapping):
        winner_doc = dict(winner)
        marker = winner_doc.pop(_JOURNAL_MARKER, None)
        if isinstance(marker, Mapping) and marker.get("kind") == "value":
            winner_doc["value"] = _materialize_value_marker(root, task_id, marker)
        result["winner"] = winner_doc
    return result


def _read_evidence_block(
    root: Path,
    task_id: str,
    reference: Mapping[str, object],
    *,
    schema: str,
    role: str,
) -> dict[str, Any]:
    return _read_domain_document(
        root,
        reference,
        schema=schema,
        role=role,
        bind={"task_id": task_id},
    )


def _expand_barrier(
    root: Path,
    task_id: str,
    task_binding: Mapping[str, object],
    kind: str,
    compact: Mapping[str, object],
    marker: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    evidence_ref = marker.get("evidence")
    if not isinstance(evidence_ref, Mapping):
        raise _integrity("Benchmark barrier lacks its evidence block reference.")
    if kind == "baseline_ready":
        block = _read_evidence_block(
            root, task_id, evidence_ref,
            schema="scnsim.benchmark_baseline_evidence", role="benchmark_baseline_evidence",
        )
        baseline = _materialize_value_marker(root, task_id, {
            "reference": block["baseline"],
            "occurrence": block.get("baseline_occurrence", {}),
        })
        result = dict(block.get("attributes", {}))
        result.update({key: value for key, value in compact.items() if key != _JOURNAL_MARKER})
        result["baseline"] = baseline
        anchors_ref = block.get("anchors")
        if isinstance(anchors_ref, Mapping):
            anchors_block = _read_evidence_block(
                root, task_id, anchors_ref,
                schema="scnsim.benchmark_anchor_values", role="benchmark_anchor_values",
            )
            result["anchors"] = anchors_block["anchors"]
        if marker.get("checkpoint") is None:
            result["resume_state"] = None
        else:
            checkpoint, _ = _verify_checkpoint_file(
                root, marker["checkpoint"], task_binding=task_binding,
            )
            if (
                checkpoint.get("schema") != "scnsim.benchmark_cma_checkpoint"
                or checkpoint.get("baseline_evidence") != dict(evidence_ref)
                or checkpoint.get("generation_evidence") is not None
            ):
                raise _integrity("Baseline barrier checkpoint does not reference its evidence block.")
            result["resume_state"] = {"cma": checkpoint["cma"]}
        return result, dict(block["baseline"])
    if kind == "generation_ready":
        block = _read_evidence_block(
            root, task_id, evidence_ref,
            schema="scnsim.benchmark_generation_evidence", role="benchmark_generation_evidence",
        )
        result = dict(block.get("attributes", {}))
        result.update({key: value for key, value in compact.items() if key != _JOURNAL_MARKER})
        if marker.get("checkpoint") is None:
            result["resume_state"] = None
        else:
            checkpoint, _ = _verify_checkpoint_file(
                root, marker["checkpoint"], task_binding=task_binding,
            )
            if (
                checkpoint.get("schema") != "scnsim.benchmark_cma_checkpoint"
                or checkpoint.get("baseline_evidence") != block.get("baseline")
                or checkpoint.get("generation_evidence") != dict(evidence_ref)
            ):
                raise _integrity("Generation barrier checkpoint does not reference its evidence block.")
            result["resume_state"] = {"cma": checkpoint["cma"]}
        return result, None
    raise _integrity("Benchmark barrier event kind is unsupported.", kind=kind)


def _task_document_from_chain(
    root: Path,
    binding: Mapping[str, object],
    head: Mapping[str, object],
    commits: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> tuple[dict[str, Any], dict[int, dict[str, object]]]:
    task: dict[str, Any] = dict(binding)
    task["attempts"] = []
    task["events"] = []
    task["measurements"] = []
    task["artifacts"] = [dict(item) for item in binding.get("artifacts", ())]
    attempt_by_id: dict[str, dict[str, object]] = {}
    pending_terminal_updates: dict[str, Mapping[str, object]] = {}
    terminal_attempt_ids: set[str] = set()
    values_by_sequence: dict[int, dict[str, object]] = {}
    baseline_values_by_sequence: dict[int, dict[str, object]] = {}
    checkpoint_sequences: dict[tuple[str, str], int] = {}
    expected_event_sequence = 0
    for commit in commits:
        operation = commit.get("operation")
        if not isinstance(operation, Mapping):
            raise _integrity("Benchmark task commit operation is malformed.", task_id=binding["task_id"])
        action = operation.get("kind")
        if action == "attempt_begin":
            attempt = operation.get("attempt")
            if not isinstance(attempt, Mapping):
                raise _integrity("Benchmark attempt commit is malformed.", task_id=binding["task_id"])
            value = dict(attempt)
            attempt_id = str(value["attempt_id"])
            if attempt_id in attempt_by_id:
                raise _integrity("Benchmark attempt identifier was committed more than once.", attempt_id=attempt_id)
            attempt_by_id[attempt_id] = value
            task["attempts"].append(value)
        elif action == "attempt_update":
            attempt_id = str(operation.get("attempt_id"))
            attempt = attempt_by_id.get(attempt_id)
            if attempt is None:
                raise _integrity("Benchmark attempt update has no committed allocation.", attempt_id=attempt_id)
            if operation.get("status") is not None:
                status = operation.get("status")
                if status in {"success", "failure", "interrupted"} and attempt_id not in terminal_attempt_ids:
                    pending_terminal_updates[attempt_id] = operation
                else:
                    attempt["status"] = status
                    attempt["failure"] = operation.get("failure")
                    attempt["interruption"] = operation.get("interruption")
            checkpoint = operation.get("checkpoint")
            if checkpoint is not None:
                attempt["checkpoint"] = checkpoint
            for artifact in operation.get("artifacts", ()):
                if artifact not in attempt["artifacts"]:
                    attempt["artifacts"].append(artifact)
                if artifact not in task["artifacts"]:
                    task["artifacts"].append(artifact)
        elif action == "event":
            event = operation.get("event")
            if not isinstance(event, Mapping):
                raise _integrity("Benchmark task event commit is malformed.", task_id=binding["task_id"])
            sequence = event.get("sequence")
            if sequence != expected_event_sequence:
                raise _integrity("Benchmark task event sequence is not contiguous.", task_id=binding["task_id"])
            expected_event_sequence += 1
            kind = str(event["kind"])
            compact_payload = event.get("payload")
            if not isinstance(compact_payload, Mapping):
                raise _integrity("Benchmark task event payload is malformed.", task_id=binding["task_id"])
            payload, baseline_reference = _expand_payload(
                root, str(binding["task_id"]), binding, kind, compact_payload,
            )
            materialized = {
                "task_id": binding["task_id"],
                "sequence": sequence,
                "kind": kind,
                "payload": payload,
            }
            task["events"].append(materialized)
            if kind == "baseline_ready" and baseline_reference is not None:
                baseline_values_by_sequence[int(sequence)] = baseline_reference
            if kind in {"completed", "failed", "interrupted"}:
                attempt_id = payload.get("attempt_id")
                attempt = attempt_by_id.get(str(attempt_id)) if attempt_id is not None else None
                if attempt is not None:
                    terminal_attempt_ids.add(str(attempt_id))
                    attempt["status"] = {
                        "completed": "success",
                        "failed": "failure",
                        "interrupted": "interrupted",
                    }[kind]
                    attempt["failure"] = payload.get("failure") if kind == "failed" else None
                    attempt["interruption"] = payload.get("interruption") if kind == "interrupted" else None
                    transition = pending_terminal_updates.pop(str(attempt_id), None)
                    if transition is not None:
                        attempt["status"] = transition.get("status")
                        attempt["failure"] = transition.get("failure")
                        attempt["interruption"] = transition.get("interruption")
            checkpoint = operation.get("checkpoint")
            if checkpoint is not None:
                attempt_id = str(payload.get("attempt_id", ""))
                attempt = attempt_by_id.get(attempt_id)
                if attempt is None:
                    raise _integrity("Checkpoint commit has no allocated attempt.", attempt_id=attempt_id)
                attempt["checkpoint"] = checkpoint
                seal_sha256 = checkpoint.get("seal_sha256")
                if isinstance(seal_sha256, str):
                    checkpoint_sequences[(attempt_id, seal_sha256)] = int(sequence)
            for artifact in operation.get("artifacts", ()):
                if artifact not in task["artifacts"]:
                    task["artifacts"].append(artifact)
                attempt_id = str(payload.get("attempt_id", ""))
                attempt = attempt_by_id.get(attempt_id)
                if attempt is not None and artifact not in attempt["artifacts"]:
                    attempt["artifacts"].append(artifact)
            marker = compact_payload.get(_JOURNAL_MARKER)
            if kind == "evaluation" and isinstance(marker, Mapping) and marker.get("kind") == "value":
                reference = marker.get("reference")
                if isinstance(reference, Mapping):
                    values_by_sequence[int(sequence)] = dict(reference)
        elif action == "measurement":
            measurements = operation.get("measurements", ())
            if not isinstance(measurements, Sequence) or isinstance(measurements, (str, bytes)):
                raise _integrity("Benchmark task measurement commit is malformed.", task_id=binding["task_id"])
            task["measurements"].extend(dict(item) for item in measurements)
        else:
            raise _integrity("Benchmark task commit has an unsupported operation.", kind=action)

    expected = head.get("event_sequence")
    if expected != expected_event_sequence:
        raise _integrity("Benchmark task head event sequence does not match its commit chain.", task_id=binding["task_id"])
    for event in task["events"]:
        if event["kind"] in {"completed", "failed", "interrupted"} and "numerical_observations" not in event["payload"]:
            observations = _project_python_observations(
                root, manifest, binding, task, event, values_by_sequence,
                baseline_values_by_sequence, checkpoint_sequences,
            )
            if observations is not None:
                event["payload"]["numerical_observations"] = observations
    return task, values_by_sequence


def _result_kind(manifest: Mapping[str, object]) -> str | None:
    declaration = manifest.get("declaration")
    if not isinstance(declaration, Mapping):
        return None
    benchmark = declaration.get("benchmark")
    analysis = declaration.get("analysis")
    if isinstance(benchmark, Mapping) and benchmark.get("task_kind") == "cohort":
        return "cohort"
    if isinstance(analysis, Mapping):
        if analysis.get("operation") == "optimize_direct":
            return "optimization"
        spec = analysis.get("spec")
        if isinstance(spec, Mapping) and isinstance(spec.get("type"), str):
            return str(spec["type"])
    return None


def _project_python_observations(
    root: Path,
    manifest: Mapping[str, object],
    binding: Mapping[str, object],
    task: Mapping[str, object],
    terminal_event: Mapping[str, object],
    values_by_sequence: Mapping[int, Mapping[str, object]],
    baseline_values_by_sequence: Mapping[int, Mapping[str, object]],
    checkpoint_sequences: Mapping[tuple[str, str], int],
) -> dict[str, object] | None:
    if task.get("arm") == "original_julia":
        return None
    payload = terminal_event.get("payload")
    if not isinstance(payload, Mapping):
        return None
    result_ref = payload.get("result")
    result_document: dict[str, Any] | None = None
    if isinstance(result_ref, Mapping) and result_ref.get("role") in {
        "python_task_result", "operation_result",
    }:
        result_bytes = _read_immutable(root, result_ref, role=str(result_ref["role"]))
        result_document = record_document(result_bytes)
        if record_bytes(result_document) != result_bytes:
            raise _integrity("Python benchmark result artifact is not canonical.", task_id=task["task_id"])
    result_kind = None if result_document is None else result_document.get("type")
    if not isinstance(result_kind, str):
        result_kind = _result_kind(manifest)
    if result_kind is None:
        return None
    terminal_payload = terminal_event.get("payload")
    terminal_sequence = terminal_event.get("sequence")
    terminal_attempt_id = terminal_payload.get("attempt_id") if isinstance(terminal_payload, Mapping) else None
    if not isinstance(terminal_sequence, int) or not isinstance(terminal_attempt_id, str):
        return None
    attempts = task.get("attempts", ())
    attempts_by_id = {
        str(attempt["attempt_id"]): attempt
        for attempt in attempts
        if isinstance(attempt, Mapping) and isinstance(attempt.get("attempt_id"), str)
    }
    event_limits: dict[str, int] = {terminal_attempt_id: terminal_sequence}
    ancestor_id: str | None = terminal_attempt_id
    visited_attempts: set[str] = set()
    while ancestor_id is not None and ancestor_id not in visited_attempts:
        visited_attempts.add(ancestor_id)
        ancestor = attempts_by_id.get(ancestor_id)
        resume = None if ancestor is None else ancestor.get("resume_from")
        next_id = resume.get("attempt_id") if isinstance(resume, Mapping) else None
        seal_sha256 = resume.get("seal_sha256") if isinstance(resume, Mapping) else None
        if not isinstance(next_id, str) or not isinstance(seal_sha256, str):
            break
        checkpoint_sequence = checkpoint_sequences.get((next_id, seal_sha256))
        if checkpoint_sequence is None:
            break
        event_limits[next_id] = checkpoint_sequence
        ancestor_id = next_id
    records: list[dict[str, object]] = []
    rows: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    selected_events = []
    for event in task["events"]:
        if not isinstance(event, Mapping) or not isinstance(event.get("sequence"), int):
            continue
        event_payload = event.get("payload")
        if not isinstance(event_payload, Mapping):
            continue
        attempt_id = event_payload.get("attempt_id")
        if not isinstance(attempt_id, str) or event["sequence"] > event_limits.get(attempt_id, -1):
            continue
        selected_events.append(event)
    for event in selected_events:
        if event.get("kind") != "evaluation":
            continue
        event_payload = event.get("payload")
        if not isinstance(event_payload, Mapping):
            continue
        source_ref = values_by_sequence.get(int(event["sequence"]))
        if source_ref is None:
            continue
        role = str(event_payload.get("origin", ""))
        if role == "baseline" or event_payload.get("evaluation_ordinal") == 0:
            role = "baseline"
        elif role == "cohort":
            role = "candidate"
        elif role == "root_continuation":
            role = "root_continuation"
        elif role == "requested_point":
            role = "requested_point"
        elif result_kind == "optimization":
            role = "candidate"
        else:
            role = result_kind
        ordinal = event_payload.get("evaluation_ordinal", event_payload.get("source_index"))
        row: dict[str, object] = {
            "role": role,
            "source_artifact": dict(source_ref),
            "value": dict(event_payload),
        }
        if event_payload.get("generation") is not None:
            row["generation"] = event_payload["generation"]
        if ordinal is not None:
            row["native_ordinal"] = ordinal
        records.append(row)
        rows.append((row, event_payload))
    if not any(row.get("role") == "baseline" for row in records):
        for event in selected_events:
            sequence = event.get("sequence")
            if event.get("kind") != "baseline_ready" or not isinstance(sequence, int):
                continue
            source_ref = baseline_values_by_sequence.get(sequence)
            event_payload = event.get("payload")
            if source_ref is None or not isinstance(event_payload, Mapping):
                continue
            baseline = event_payload.get("baseline")
            if not isinstance(baseline, Mapping):
                continue
            value = dict(baseline)
            ordinal = value.get("evaluation_ordinal", 0)
            row: dict[str, object] = {
                "role": "baseline",
                "source_artifact": dict(source_ref),
                "value": value,
            }
            generation = value.get("generation")
            if generation is not None:
                row["generation"] = generation
            if ordinal is not None:
                row["native_ordinal"] = ordinal
            records.insert(0, row)
            rows.insert(0, (row, value))
            break
    if result_kind == "diagonal_root":
        indexed = list(enumerate(records))
        indexed.sort(key=lambda pair: (
            0 if pair[1].get("role") == "baseline" else
            1 if pair[1].get("role") == "requested_point" else 2,
            pair[1].get("native_ordinal", 0) if pair[1].get("role") == "requested_point" else pair[0],
            pair[0],
        ))
        records = [row for _, row in indexed]
    observations: dict[str, object] = {
        "result_kind": result_kind,
        "result_artifact": None if result_ref is None else dict(result_ref),
        "records": records,
    }
    if result_document is not None:
        # The terminal artifact intentionally stays compact and hash-stable.
        # This detached read projection exposes only its actual compact summary
        # fields to consumers that do not reopen artifact paths.
        observations["terminal_summary"] = dict(result_document)
    if result_kind == "optimization" and result_document is not None:
        best_ordinal = result_document.get("best_ordinal")
        for row, value in rows:
            if value.get("evaluation_ordinal") == best_ordinal:
                observations["winner"] = {
                    "generation": value.get("generation", 0),
                    "native_ordinal": best_ordinal,
                    "source_artifact": dict(row["source_artifact"]),
                    "value": dict(value),
                }
                break
    return observations


def _materialize_global(
    root: Path,
    manifest: Mapping[str, object],
    head: Mapping[str, object],
    commits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {
        "measurements": [],
        "reports": [],
        "operation_events": [],
        "_task_measurements": {},
    }
    list_fields = {
        "task_launch_failures", "execution_failures", "callback_failures", "preparation_failures",
        "operation_events",
    }
    for key in list_fields:
        result[key] = []
    for commit in commits:
        changes = commit.get("changes")
        if not isinstance(changes, Sequence) or isinstance(changes, (str, bytes)):
            raise _integrity("Benchmark global commit changes are malformed.")
        for change in changes:
            if not isinstance(change, Mapping):
                raise _integrity("Benchmark global commit change is malformed.")
            field = change.get("field")
            value = change.get("value")
            if field == "preparation_failure":
                if "preparation_failure" not in result:
                    result["preparation_failure"] = value
                else:
                    cast_list = result["preparation_failures"]
                    assert isinstance(cast_list, list)
                    cast_list.append(value)
            elif field in list_fields or field in {"reports", "measurements"}:
                cast_list = result[field]
                assert isinstance(cast_list, list)
                if isinstance(value, list):
                    cast_list.extend(value)
                else:
                    cast_list.append(value)
            elif field == "task_measurements":
                task_id = str(change["task_id"])
                collections = result["_task_measurements"]
                assert isinstance(collections, dict)
                collection = collections.setdefault(task_id, [])
                assert isinstance(collection, list)
                if not isinstance(value, list):
                    raise _integrity("Benchmark task measurements are malformed.", task_id=task_id)
                collection.extend(value)
            else:
                raise _integrity("Benchmark global commit names an unsupported collection.", field=field)
    return result


def _materialize_document(root: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    if manifest.get("schema_version") != _V2:
        return dict(manifest)
    with _benchmark_reader_lock(root):
        current = _require_v2(root)
        bindings = [dict(item) for item in current.get("tasks", ())]
        task_heads = {str(item["task_id"]): _task_head(root, item) for item in bindings}
        global_head = _global_head(root, current.get("benchmark_sha256"))
    materialized: dict[str, object] = {
        key: value for key, value in current.items() if key != "tasks"
    }
    tasks: list[dict[str, Any]] = []
    for binding in bindings:
        head = task_heads[str(binding["task_id"])]
        commits = _task_chain(root, binding, head)
        task, _ = _task_document_from_chain(root, binding, head, commits, current)
        tasks.append(task)
    global_commits = _global_chain(root, current, global_head)
    global_document = _materialize_global(root, current, global_head, global_commits)
    if current.get("benchmark_sha256") is None and "preparation_failure" not in global_document:
        raise _integrity("Unbound benchmark record lacks its explicit preparation failure.")
    task_measurements = global_document.pop("_task_measurements")
    assert isinstance(task_measurements, dict)
    for task in tasks:
        task["measurements"].extend(task_measurements.get(str(task["task_id"]), ()))
    materialized["tasks"] = tasks
    materialized.update(global_document)
    return materialized


def _legacy_open_record(workspace: str | os.PathLike[str]) -> BenchmarkResult:
    """Read a complete or partial record without binding or cleaning a workspace."""
    root = _root_path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise _integrity("Benchmark workspace is missing or symlinked.", path=str(root))
    document = _read_document(root)
    if document.get("schema_version") == _V2:
        document = _materialize_document(root, document)
        declaration = document.get("declaration")
        if isinstance(declaration, Mapping) and declaration.get("schema") == "scnsim.operation_trace":
            from .operations import project_operation_record

            return project_operation_record(
                document,
                workspace=root,
                plan_sha256=declaration.get("plan_sha256"),
                workspace_instance_id=declaration.get("workspace_instance_id"),
            )
    return BenchmarkResult.from_document(root, document)


def _legacy_materialize_task(workspace: str | os.PathLike[str], task_id: str) -> dict[str, Any]:
    """Return one verified detached legacy task projection from its journal."""
    root = _root_path(workspace)
    with _benchmark_reader_lock(root):
        manifest = _require_v2(root)
        binding = _task_binding(manifest, task_id)
        head = _task_head(root, binding)
        global_head = _global_head(root, manifest.get("benchmark_sha256"))
    commits = _task_chain(root, binding, head)
    task, _ = _task_document_from_chain(root, binding, head, commits, manifest)
    global_commits = _global_chain(root, manifest, global_head)
    global_document = _materialize_global(root, manifest, global_head, global_commits)
    task["measurements"].extend(global_document["_task_measurements"].get(task_id, ()))
    return task


def operation_workspace(binding: object) -> Path:
    """Return the operation journal path below one bound Plan leaf."""
    leaf = Path(getattr(binding, "leaf"))
    return _inside(leaf, "operations")


def _legacy_operation_task_record(binding: object, task_id: str) -> dict[str, Any]:
    """Materialize one verified operation task without choosing another task."""
    root = operation_workspace(binding)
    task = _legacy_task_record(root, task_id)
    request_sha256 = task.get("request_sha256")
    references = task.get("artifacts", ())
    if not isinstance(request_sha256, str) or not isinstance(references, Sequence):
        raise _integrity("Operation task request binding is malformed.", task_id=task_id)
    request_reference = next((
        item for item in references
        if isinstance(item, Mapping) and item.get("role") == "operation_request"
    ), None)
    if not isinstance(request_reference, Mapping) or request_reference.get("sha256") != request_sha256:
        raise _integrity("Operation task lacks its exact canonical request artifact.", task_id=task_id)
    request_bytes = _read_immutable(root, request_reference, role="operation_request")
    if sha256(request_bytes).hexdigest() != request_sha256:
        raise _integrity("Operation request artifact does not match its task identity.", task_id=task_id)
    return task


def _legacy_open_operation_record(binding: object) -> BenchmarkResult | None:
    """Materialize the verified raw operation journal for internal readers."""
    root = operation_workspace(binding)
    source = _operation_record_source(binding)
    return None if source is None else BenchmarkResult.from_document(root, source)


def _operation_record_source(binding: object) -> dict[str, Any] | None:
    """Return the verified materialized operation journal for one bound leaf."""
    root = operation_workspace(binding)
    manifest = _operation_manifest(binding, root=root)
    if manifest is None:
        return None
    return _materialize_document(root, manifest)


def _operation_manifest(
    binding: object,
    *,
    root: Path | None = None,
) -> dict[str, Any] | None:
    """Read and bind the operation journal header without creating state."""
    directory = operation_workspace(binding) if root is None else root
    path = _inside(directory, _RECORD_NAME)
    if directory.is_symlink() or not directory.is_dir():
        if directory.exists() or directory.is_symlink():
            raise _integrity("Operation journal path is unsafe.", path=str(directory))
        return None
    if path.is_symlink():
        raise _integrity("Operation record path must not be a symlink.", path=str(path))
    if not path.exists():
        return None
    document = _read_document(directory)
    declaration = document.get("declaration")
    if (
        document.get("schema_version") != _V2
        or not isinstance(declaration, Mapping)
        or declaration.get("schema") != "scnsim.operation_trace"
        or declaration.get("schema_version") != 1
        or declaration.get("plan_sha256") != getattr(binding, "plan_sha256")
        or declaration.get("workspace_instance_id") != getattr(binding, "workspace_instance_id")
        or document.get("plan_sha256") != getattr(binding, "plan_sha256")
        or document.get("workspace_instance_id") != getattr(binding, "workspace_instance_id")
    ):
        raise _integrity(
            "Operation journal does not belong to this bound Plan leaf.",
            plan_sha256=getattr(binding, "plan_sha256"),
            workspace_instance_id=getattr(binding, "workspace_instance_id"),
        )
    return document


def _operation_success_from_task(
    task: Mapping[str, object],
    *,
    attempt_id: str | None = None,
) -> dict[str, object] | None:
    """Project one complete JAX success from its already verified task chain."""
    attempts = task.get("attempts", ())
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        raise _integrity("Operation task attempts are malformed.", task_id=task.get("task_id"))
    attempts_by_id = {
        str(value["attempt_id"]): value
        for value in attempts
        if isinstance(value, Mapping) and isinstance(value.get("attempt_id"), str)
    }
    events = task.get("events", ())
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise _integrity("Operation task events are malformed.", task_id=task.get("task_id"))
    successes: list[dict[str, object]] = []
    for event in events:
        if not isinstance(event, Mapping) or event.get("kind") != "completed":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise _integrity("Operation completion event payload is malformed.", task_id=task.get("task_id"))
        selected_attempt_id = payload.get("attempt_id")
        if not isinstance(selected_attempt_id, str):
            raise _integrity("Operation completion event has no attempt identity.", task_id=task.get("task_id"))
        if attempt_id is not None and selected_attempt_id != attempt_id:
            continue
        attempt = attempts_by_id.get(selected_attempt_id)
        if attempt is None or attempt.get("status") != "success":
            continue
        result_ref = payload.get("result")
        if (
            not isinstance(result_ref, Mapping)
            or result_ref.get("role") != "operation_result"
            or not isinstance(result_ref.get("sha256"), str)
        ):
            raise _integrity(
                "Completed JAX operation lacks its exact immutable result artifact.",
                task_id=task.get("task_id"), attempt_id=selected_attempt_id,
            )
        if result_ref not in attempt.get("artifacts", ()):
            raise _integrity(
                "Completed JAX result is not linked from its attempt.",
                task_id=task.get("task_id"), attempt_id=selected_attempt_id,
            )
        observations = payload.get("numerical_observations")
        if not isinstance(observations, Mapping):
            raise _integrity(
                "Completed JAX operation has no verified numerical projection.",
                task_id=task.get("task_id"), attempt_id=selected_attempt_id,
            )
        terminal = observations.get("terminal_summary")
        records = observations.get("records")
        if not isinstance(terminal, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise _integrity(
                "Completed JAX numerical projection is malformed.",
                task_id=task.get("task_id"), attempt_id=selected_attempt_id,
            )
        result_kind = observations.get("result_kind")
        baseline: Mapping[str, object] | None = None
        evaluations: list[dict[str, object]] = []
        for row in records:
            if not isinstance(row, Mapping):
                raise _integrity("JAX numerical projection row is malformed.", task_id=task.get("task_id"))
            value = row.get("value")
            role = row.get("role")
            if not isinstance(value, Mapping):
                raise _integrity("JAX numerical projection value is malformed.", task_id=task.get("task_id"))
            if role == "baseline":
                if baseline is None:
                    baseline = dict(value)
                continue
            if result_kind == "diagonal_root" and role != "requested_point":
                continue
            evaluations.append(dict(value))
        if result_kind == "optimization" and baseline is None:
            raise _integrity(
                "Completed JAX Optimization lacks its committed baseline value.",
                task_id=task.get("task_id"), attempt_id=selected_attempt_id,
            )
        environment = task.get("environment")
        environment_sha256 = environment.get("environment_sha256") if isinstance(environment, Mapping) else None
        if not isinstance(environment_sha256, str):
            raise _integrity("Operation task has no runtime environment identity.", task_id=task.get("task_id"))
        projection: dict[str, object] = {
            "terminal": dict(terminal),
            "evaluations": evaluations,
        }
        if result_kind == "optimization":
            assert baseline is not None
            projection["baseline"] = dict(baseline)
        successes.append({
            "task_id": str(task["task_id"]),
            "environment_sha256": environment_sha256,
            "attempt_sha256": sha256(_record_bytes(dict(attempt))).hexdigest(),
            "result_sha256": str(result_ref["sha256"]),
            "result_ref": dict(result_ref),
            "projection": projection,
            "checkpoint_available": (
                isinstance(attempt.get("checkpoint"), Mapping)
                or isinstance(attempt.get("resume_from"), Mapping)
            ),
            "attempt_id": selected_attempt_id,
        })
    if len(successes) > 1:
        raise _integrity("One JAX operation task has competing successful attempts.", task_id=task.get("task_id"))
    return successes[0] if successes else None


def _legacy_find_operation_success(
    binding: object,
    request_sha256: str,
) -> dict[str, object] | None:
    """Read the selected completed operation for a request, across policies."""
    root = operation_workspace(binding)
    manifest = _operation_manifest(binding, root=root)
    if manifest is None:
        return None
    with _benchmark_reader_lock(root):
        current = _operation_manifest(binding, root=root)
        if current is None:
            return None
        selected = _operation_success_selection(root, current, request_sha256)
    return None if selected is None else _resolve_operation_success_selection(binding, selected)


def _legacy_read_operation_success(
    binding: object,
    task_id: str,
    *,
    attempt_id: str | None = None,
) -> dict[str, object] | None:
    """Read one exact task success without selecting another task or attempt."""
    manifest = _operation_manifest(binding)
    if manifest is None:
        return None
    descriptors = manifest.get("tasks", ())
    descriptor = next((
        item for item in descriptors
        if isinstance(item, Mapping) and item.get("task_id") == task_id
    ), None) if isinstance(descriptors, Sequence) else None
    if descriptor is None:
        return None
    task = _legacy_operation_task_record(binding, task_id)
    return _operation_success_from_task(task, attempt_id=attempt_id)


def _operation_success_selection(
    root: Path,
    manifest: Mapping[str, object],
    request_sha256: str,
) -> dict[str, object] | None:
    head = _global_head(root, manifest.get("benchmark_sha256"))
    commits = _global_chain(root, manifest, head)
    materialized = _materialize_global(root, manifest, head, commits)
    events = materialized.get("operation_events", ())
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise _integrity("Operation selection journal is malformed.")
    for event in events:
        if (
            isinstance(event, Mapping)
            and event.get("event") == "request_success_selected"
            and event.get("request_sha256") == request_sha256
        ):
            return dict(event)
    return None


def _resolve_operation_success_selection(
    binding: object,
    selected: Mapping[str, object],
) -> dict[str, object]:
    task_id = selected.get("task_id")
    attempt_id = selected.get("attempt_id")
    request_sha256 = selected.get("request_sha256")
    if not all(isinstance(value, str) for value in (task_id, attempt_id, request_sha256)):
        raise _integrity("Selected operation result reference is malformed.")
    task = _legacy_operation_task_record(binding, str(task_id))
    if task.get("request_sha256") != request_sha256:
        raise _integrity("Selected operation result belongs to another request.", task_id=task_id)
    success = _operation_success_from_task(task, attempt_id=str(attempt_id))
    if success is None:
        raise _integrity("Selected operation result is not a completed task attempt.", task_id=task_id)
    for key in (
        "environment_sha256", "attempt_sha256", "result_sha256", "result_ref",
    ):
        if selected.get(key) != success.get(key):
            raise _integrity("Selected operation result reference does not match its task evidence.",
                             task_id=task_id, field=key)
    return success


def _new_task_descriptor(task: Mapping[str, object]) -> dict[str, object]:
    value = json.loads(_record_bytes(task))
    required = {
        "task_id", "request_sha256", "arm", "sample", "attempts", "events",
        "measurements", "environment", "artifacts",
    }
    if set(value) != required:
        raise _integrity("Benchmark task record has an unexpected field set.", fields=sorted(value))
    if value["attempts"] or value["events"] or value["measurements"]:
        raise _integrity("New benchmark task identity cannot include prior task history.", task_id=value["task_id"])
    return {
        key: value[key]
        for key in ("task_id", "request_sha256", "arm", "sample", "environment", "artifacts")
    }


def _legacy_task_record(workspace: str | os.PathLike[str], task_id: str) -> dict[str, Any]:
    """Return one stored task record without selecting or mutating other tasks."""
    root = _root_path(workspace)
    manifest = _read_document(root)
    if manifest.get("schema_version") == _V1:
        return json.loads(_record_bytes(_task_binding(manifest, task_id)))
    return _legacy_materialize_task(root, task_id)


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


def _checkpoint_is_committed(
    commits: Sequence[Mapping[str, object]],
    reference: Mapping[str, object],
) -> bool:
    for commit in commits:
        operation = commit.get("operation")
        if not isinstance(operation, Mapping):
            continue
        if operation.get("checkpoint") == dict(reference):
            return True
        if operation.get("kind") == "attempt_update" and operation.get("checkpoint") == dict(reference):
            return True
    return False


def _hydrate_cma_checkpoint(
    root: Path,
    task_id: str,
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    baseline_ref = checkpoint.get("baseline_evidence")
    if not isinstance(baseline_ref, Mapping):
        raise _integrity("CMA checkpoint does not reference its baseline evidence.", task_id=task_id)
    baseline_block = _read_evidence_block(
        root, task_id, baseline_ref,
        schema="scnsim.benchmark_baseline_evidence", role="benchmark_baseline_evidence",
    )
    baseline = _materialize_value_marker(root, task_id, {
        "reference": baseline_block["baseline"],
        "occurrence": baseline_block.get("baseline_occurrence", {}),
    })
    anchors: dict[str, object] = {}
    anchors_ref = baseline_block.get("anchors")
    if isinstance(anchors_ref, Mapping):
        anchors_block = _read_evidence_block(
            root, task_id, anchors_ref,
            schema="scnsim.benchmark_anchor_values", role="benchmark_anchor_values",
        )
        raw_anchors = anchors_block.get("anchors")
        if not isinstance(raw_anchors, dict):
            raise _integrity("CMA checkpoint anchor block is malformed.", task_id=task_id)
        anchors = raw_anchors

    block_ref = checkpoint.get("generation_evidence")
    reversed_blocks: list[tuple[Mapping[str, object], Mapping[str, Any]]] = []
    seen: set[str] = set()
    while block_ref is not None:
        if not isinstance(block_ref, Mapping):
            raise _integrity("CMA checkpoint generation link is malformed.", task_id=task_id)
        digest = str(block_ref.get("sha256"))
        if digest in seen:
            raise _integrity("CMA checkpoint generation links contain a cycle.", task_id=task_id)
        seen.add(digest)
        block = _read_evidence_block(
            root, task_id, block_ref,
            schema="scnsim.benchmark_generation_evidence", role="benchmark_generation_evidence",
        )
        if block.get("baseline") != dict(baseline_ref):
            raise _integrity("CMA checkpoint generation is bound to another baseline.", task_id=task_id)
        reversed_blocks.append((block_ref, block))
        block_ref = block.get("previous")
    generation_blocks = list(reversed(reversed_blocks))
    cache: dict[str, object] = {}
    value_refs: dict[str, Mapping[str, object]] = {}
    records_by_generation: list[list[dict[str, object]]] = []
    best_ordinal = checkpoint.get("best_ordinal")
    best: dict[str, object] | None = None
    baseline_key = baseline.get("candidate_key")
    if isinstance(baseline_key, str):
        baseline_doc = _read_value(root, task_id, baseline_block["baseline"])
        cache[baseline_key] = baseline_doc
        value_refs[baseline_key] = dict(baseline_block["baseline"])
    if baseline.get("evaluation_ordinal") == best_ordinal:
        best = baseline
    for _, block in generation_blocks:
        rows: list[dict[str, object]] = []
        raw_rows = block.get("rows", ())
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise _integrity("CMA generation evidence rows are malformed.", task_id=task_id)
        for row in raw_rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("value"), Mapping):
                raise _integrity("CMA generation evidence row is malformed.", task_id=task_id)
            record = _materialize_value_marker(root, task_id, {
                "reference": row["value"],
                "occurrence": row.get("occurrence", {}),
            })
            key = record.get("candidate_key")
            if isinstance(key, str):
                cache[key] = _read_value(root, task_id, row["value"])
                value_refs.setdefault(key, dict(row["value"]))
            if record.get("evaluation_ordinal") == best_ordinal:
                best = record
            rows.append(record)
        records_by_generation.append(rows)
    if best is None:
        raise _integrity("CMA checkpoint best ordinal has no committed evaluation evidence.", task_id=task_id)
    return {
        "generation": checkpoint["generation"],
        "next_ordinal": checkpoint["next_ordinal"],
        "best_ordinal": best_ordinal,
        "cma": checkpoint["cma"],
        "anchors": anchors,
        "baseline": baseline,
        "best": best,
        "cache": cache,
        "generations": records_by_generation,
        "_journal_links": {
            "baseline_evidence": dict(baseline_ref),
            "generation_evidence": None if checkpoint.get("generation_evidence") is None else dict(checkpoint["generation_evidence"]),
            "value_refs": value_refs,
        },
    }


def _legacy_read_checkpoint(
    workspace: str | os.PathLike[str],
    reference: Mapping[str, object],
    *,
    expected_task_id: str,
    expected_request_sha256: str,
    expected_arm: str,
    expected_sample: int,
    expected_environment_sha256: str,
) -> bytes:
    """Read the exact sealed checkpoint and hydrate its committed evidence."""
    root = _root_path(workspace)
    if (
        reference.get("task_id") != expected_task_id
        or reference.get("request_sha256") != expected_request_sha256
        or reference.get("arm") != expected_arm
        or reference.get("sample") != expected_sample
        or reference.get("environment_sha256") != expected_environment_sha256
    ):
        raise _integrity("Checkpoint binding does not match the requested task/sample.")
    manifest = _read_document(root)
    if manifest.get("schema_version") == _V1:
        binding = _task_binding(manifest, expected_task_id)
        commits: list[dict[str, Any]] = []
    else:
        with _benchmark_reader_lock(root):
            manifest = _read_document(root)
            binding = _task_binding(manifest, expected_task_id)
            if (
                binding["request_sha256"] != expected_request_sha256
                or binding["arm"] != expected_arm
                or binding["sample"] != expected_sample
                or binding["environment"].get("environment_sha256") != expected_environment_sha256
            ):
                raise _integrity("Checkpoint task binding changed before its read.")
            head = _task_head(root, binding)
            commits = _task_chain(root, binding, head)
            if not _checkpoint_is_committed(commits, reference):
                raise _integrity("Checkpoint is not reachable from the current committed task head.")
    if (
        binding["request_sha256"] != expected_request_sha256
        or binding["arm"] != expected_arm
        or binding["sample"] != expected_sample
        or binding["environment"].get("environment_sha256") != expected_environment_sha256
    ):
        raise _integrity("Checkpoint task binding changed before its read.")
    checkpoint, checkpoint_bytes = _verify_checkpoint_file(root, reference, task_binding=binding)
    if manifest.get("schema_version") == _V1 or checkpoint.get("schema") != "scnsim.benchmark_cma_checkpoint":
        return checkpoint_bytes
    if (
        checkpoint.get("task_id") != expected_task_id
        or checkpoint.get("request_sha256") != expected_request_sha256
        or checkpoint.get("arm") != expected_arm
        or checkpoint.get("sample") != expected_sample
        or checkpoint.get("environment_sha256") != expected_environment_sha256
        or checkpoint.get("attempt_id") != reference.get("attempt_id")
    ):
        raise _integrity("CMA checkpoint content differs from its sealed task identity.")
    hydrated = _hydrate_cma_checkpoint(root, expected_task_id, checkpoint)
    return _record_bytes(hydrated)


# Discriminated domain reads share the existing schema/normalization authority.
def _read_immutable(root, reference, *, role):
    if reference.get("storage") == "sqlite":
        from .sqlite_storage import read_object
        return read_object(root, reference, role=role)
    return journal.read_immutable(root, reference, role=role)


def _read_domain_document(root, reference, *, schema, role, bind):
    if reference.get("storage") == "sqlite":
        from .sqlite_storage import read_document
        return read_document(root, reference, schema=schema, role=role, bind=bind)
    return journal.read_document(root, reference, schema=schema, role=role, bind=bind)


def open_legacy_operation_record(binding):
    return _legacy_open_operation_record(binding)


def _merge_sqlite_legacy(root, current):
    from .operations import _count
    previous = _legacy_open_record(root).document()
    document = current.document()
    if previous.get("schema") != "scnsim.operation_benchmark":
        raise _integrity("Historical operation record has an incompatible projection.")
    for field, identity in (("operations", "operation_id"), ("spans", "span_id")):
        rows = [*previous[field], *document[field]]
        if len({row[identity] for row in rows}) != len(rows):
            raise _integrity("Operation identity is duplicated across storage authorities.", field=field)
        document[field] = rows
    document["historical_records"] = previous.get("historical_records", [])
    document["numerical_refs"] = [*previous.get("numerical_refs", []), *document.get("numerical_refs", [])]
    clocks = {clock["id"]: clock for clock in [*previous.get("clock_domains", []), *document.get("clock_domains", [])]}
    document["clock_domains"] = list(clocks.values())
    document["counts"] = {"operations":len(document["operations"]),"spans":len(document["spans"]),
        "operation_status":_count(document["operations"],"status"),
        "span_kind":_count(document["spans"],"kind"),
        "historical_kind":previous.get("counts", {}).get("historical_kind", {})}
    return BenchmarkResult.from_document(root, document)


# One current writer; historical format handlers above are explicitly readonly.
from .sqlite_storage import (
    TaskWriter, append_event, begin_attempt, begin_task_writer, bind_operation,
    bind_operation_attempt, commit_barrier, complete_operation, ensure_operation_task,
    ensure_task, find_operation_success, finish_operation, flush_completed,
    initialize_operation_record, open_record, operation_task_record, query_operation_rows,
    read_checkpoint, read_operation_success, record_operation_event,
    recover_operation_workspace, start_operation, store_operation_request, task_record,
    update_attempt, write_artifact,
)
