"""Durable benchmark observations over immutable task and global journals."""

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
from ..workspace.storage import _atomic_write, _load_canonical
from ..workspace.store import _require_platform, _workspace_lock
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


def _empty_task_head(task_id: str, request_sha256: str) -> dict[str, object]:
    return {
        "schema": journal._TASK_HEAD,
        "schema_version": _V2,
        "task_id": task_id,
        "request_sha256": request_sha256,
        "commit": None,
        "commit_sequence": 0,
        "event_sequence": 0,
        "last_attempt_id": None,
        "attempts": [],
    }


def _empty_global_head(benchmark_sha256: str | None) -> dict[str, object]:
    return {
        "schema": journal._GLOBAL_HEAD,
        "schema_version": _V2,
        "benchmark_sha256": benchmark_sha256,
        "commit": None,
        "commit_sequence": 0,
    }


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
    commit = journal.read_document(root, reference, schema=schema, role=role, bind=bind)
    if commit.get("commit_sequence") != sequence:
        raise _integrity("Benchmark journal head does not identify its committed predecessor.")


def _commit_ref(
    root: Path,
    *,
    scope: str,
    identity: str,
    body: Mapping[str, object],
    durability_witness: journal._DurabilityWitness | None = None,
) -> dict[str, object]:
    payload = _record_bytes(body)
    digest = sha256(payload).hexdigest()
    if scope == "task":
        relative = f"tasks/{identity}/journal/commits/{digest}.json"
        role = "benchmark_task_commit"
    else:
        relative = f"journal/global/commits/{digest}.json"
        role = "benchmark_global_commit"
    return journal.write_immutable(
        root, relative, payload, role=role,
        durability_witness=durability_witness,
    )


def _append_task_commit(
    root: Path,
    binding: Mapping[str, object],
    head: dict[str, Any],
    *,
    operation: Mapping[str, object],
    durability_witness: journal._DurabilityWitness | None = None,
) -> dict[str, object]:
    sequence = int(head["commit_sequence"]) + 1
    body = {
        "schema": journal._TASK_COMMIT,
        "schema_version": _V2,
        "task_id": binding["task_id"],
        "request_sha256": binding["request_sha256"],
        "commit_sequence": sequence,
        "previous": head["commit"],
        "operation": dict(operation),
    }
    reference = _commit_ref(
        root, scope="task", identity=str(binding["task_id"]), body=body,
        durability_witness=durability_witness,
    )
    head["commit"] = reference
    head["commit_sequence"] = sequence
    return reference


def _append_global_commit(
    root: Path,
    manifest: Mapping[str, object],
    head: dict[str, Any],
    *,
    changes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    sequence = int(head["commit_sequence"]) + 1
    body = {
        "schema": journal._GLOBAL_COMMIT,
        "schema_version": _V2,
        "benchmark_sha256": manifest.get("benchmark_sha256"),
        "commit_sequence": sequence,
        "previous": head["commit"],
        "changes": [dict(change) for change in changes],
    }
    reference = _commit_ref(
        root,
        scope="global",
        identity="global",
        body=body,
    )
    head["commit"] = reference
    head["commit_sequence"] = sequence
    return reference


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


def _value_reference(
    root: Path,
    task_id: str,
    value: Mapping[str, object],
    *,
    attempt_id: str,
    event_sequence: int,
    suffix: str = "",
    inherited: Mapping[str, Mapping[str, object]] | None = None,
    durability_witness: journal._DurabilityWitness | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Store one actual numerical body and return its ref and occurrence context."""
    occurrence = {key: value[key] for key in _OCCURRENCE_FIELDS if key in value}
    body = {str(key): item for key, item in value.items() if key not in _OCCURRENCE_FIELDS}
    if isinstance(value.get("candidate_key"), str):
        key_kind = "candidate_key"
        key_value = str(value["candidate_key"])
        inherited_ref = None if inherited is None else inherited.get(key_value)
        if inherited_ref is not None:
            stored = journal.read_document(
                root, inherited_ref,
                schema="scnsim.benchmark_value", role="benchmark_value",
                bind={"task_id": task_id},
            )
            if stored.get("key_kind") != key_kind or stored.get("key") != key_value:
                raise _integrity("Inherited optimizer value does not bind its cache key.", task_id=task_id)
            return dict(inherited_ref), occurrence, dict(stored["value"])
        token = sha256(key_value.encode("utf-8")).hexdigest()
        relative = f"tasks/{task_id}/attempts/{attempt_id}/values/keys/{token}.json"
    elif isinstance(value.get("numerical_source_id"), str):
        key_kind = "numerical_source_id"
        key_value = str(value["numerical_source_id"])
        token = sha256(key_value.encode("utf-8")).hexdigest()
        relative = f"tasks/{task_id}/attempts/{attempt_id}/values/sources/{token}.json"
    else:
        key_kind = "event"
        key_value = f"{event_sequence}:{suffix}"
        token = sha256(key_value.encode("utf-8")).hexdigest()
        relative = f"tasks/{task_id}/attempts/{attempt_id}/values/events/{token}.json"
    document = {
        "schema": "scnsim.benchmark_value",
        "schema_version": _V2,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "key_kind": key_kind,
        "key": key_value,
        "value": body,
    }
    reference = journal.write_document(
        root, relative, document, role="benchmark_value",
        durability_witness=durability_witness,
    )
    return reference, occurrence, body


def _read_value(root: Path, task_id: str, reference: Mapping[str, object]) -> dict[str, Any]:
    document = journal.read_document(
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


def _compact_value_event(
    root: Path,
    task_id: str,
    payload: Mapping[str, object],
    *,
    attempt_id: str,
    event_sequence: int,
    suffix: str = "",
    inherited: Mapping[str, Mapping[str, object]] | None = None,
    durability_witness: journal._DurabilityWitness | None = None,
) -> dict[str, object]:
    reference, occurrence, _ = _value_reference(
        root, task_id, payload, attempt_id=attempt_id,
        event_sequence=event_sequence, suffix=suffix, inherited=inherited,
        durability_witness=durability_witness,
    )
    retained = {key: value for key, value in payload.items() if key in {"attempt_id"} and key not in occurrence}
    retained[_JOURNAL_MARKER] = {
        "kind": "value",
        "reference": reference,
        "occurrence": occurrence,
    }
    return retained


def _compact_observations(
    root: Path,
    task_id: str,
    observations: Mapping[str, object],
    *,
    attempt_id: str,
    event_sequence: int,
    durability_witness: journal._DurabilityWitness | None = None,
) -> dict[str, object]:
    value = dict(observations)
    records_value = value.get("records", ())
    compact_records: list[dict[str, object]] = []
    if isinstance(records_value, Sequence) and not isinstance(records_value, (str, bytes)):
        for index, raw in enumerate(records_value):
            if not isinstance(raw, Mapping) or not isinstance(raw.get("value"), Mapping):
                compact_records.append(dict(raw) if isinstance(raw, Mapping) else {"record": raw})
                continue
            record = dict(raw)
            reference, occurrence, _ = _value_reference(
                root, task_id, record["value"], attempt_id=attempt_id, event_sequence=event_sequence,
                suffix=f"observation-{index}",
                durability_witness=durability_witness,
            )
            record.pop("value")
            record[_JOURNAL_MARKER] = {
                "kind": "value", "reference": reference, "occurrence": occurrence,
            }
            compact_records.append(record)
    value["records"] = compact_records
    winner = value.get("winner")
    if isinstance(winner, Mapping) and isinstance(winner.get("value"), Mapping):
        winner_doc = dict(winner)
        reference, occurrence, _ = _value_reference(
            root, task_id, winner_doc["value"], attempt_id=attempt_id, event_sequence=event_sequence,
            suffix="observation-winner",
            durability_witness=durability_witness,
        )
        winner_doc.pop("value")
        winner_doc[_JOURNAL_MARKER] = {
            "kind": "value", "reference": reference, "occurrence": occurrence,
        }
        value["winner"] = winner_doc
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
    checkpoint_bytes = journal.read_immutable(root, checkpoint_ref, role="checkpoint")
    seal_bytes = journal.read_immutable(root, seal_ref, role="checkpoint_seal")
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
    return journal.read_document(
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
        result_bytes = journal.read_immutable(root, result_ref, role=str(result_ref["role"]))
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


def open_record(workspace: str | os.PathLike[str]) -> BenchmarkResult:
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


def materialize_task(workspace: str | os.PathLike[str], task_id: str) -> dict[str, Any]:
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


def _publish_task_head(root: Path, head: Mapping[str, object]) -> None:
    journal.publish_head(
        root,
        f"tasks/{head['task_id']}/journal/HEAD.json",
        head,
    )


def _ensure_global_head(root: Path, benchmark_sha256: str | None) -> None:
    path = _inside(root, "journal/global/HEAD.json")
    if path.exists():
        journal.read_head(
            root, "journal/global/HEAD.json", schema=journal._GLOBAL_HEAD,
            bind={"benchmark_sha256": benchmark_sha256},
        )
        return
    journal.publish_head(root, "journal/global/HEAD.json", _empty_global_head(benchmark_sha256))


def initialize_record(
    workspace: str | os.PathLike[str],
    *,
    prepared: object,
    clock_binding: Mapping[str, object],
) -> Path:
    """Create or reopen the one writable v2 benchmark journal."""
    root = _root_path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    declaration = prepared.declaration()  # type: ignore[attr-defined]
    benchmark_sha = prepared.request_sha256  # type: ignore[attr-defined]
    record = {
        "schema": _RECORD_SCHEMA,
        "schema_version": _V2,
        "benchmark_sha256": benchmark_sha,
        "declaration": declaration,
        "plan_sha256": declaration["plan_sha256"],
        "source_analysis_sha256": declaration["source_analysis_sha256"],
        "clock": dict(clock_binding),
        "tasks": [],
    }
    target = _inside(root, _RECORD_NAME)
    with _workspace_lock(root, exclusive=True):
        if target.is_symlink():
            raise _integrity("Benchmark record path must not be a symlink.", path=str(target))
        if target.exists():
            existing = _read_document(root)
            if existing.get("schema_version") != _V2:
                raise _integrity("Historical benchmark records are read-only.", path=str(target))
            if existing.get("benchmark_sha256") != benchmark_sha:
                raise _integrity(
                    "Benchmark workspace is already bound to a different declaration.",
                    expected=existing.get("benchmark_sha256"), supplied=benchmark_sha,
                )
            _ensure_global_head(root, benchmark_sha)
            return root
        _ensure_global_head(root, benchmark_sha)
        _atomic_write(target, _record_bytes(record))
    return root


def operation_workspace(binding: object) -> Path:
    """Return the operation journal path below one bound Plan leaf."""
    leaf = Path(getattr(binding, "leaf"))
    return _inside(leaf, "operations")


def initialize_operation_record(
    binding: object,
    *,
    clock_binding: Mapping[str, object],
) -> Path:
    """Create or verify the one Plan-bound ordinary-operation journal.

    This declaration is independent of any numerical request.  Each task in
    the shared journal carries its own exact prepared-request SHA and runtime
    identity; the journal header only binds the collection to its Plan leaf.
    """
    operation_path = operation_workspace(binding)
    operation_path.mkdir(parents=True, exist_ok=True)
    root = _root_path(operation_path)
    plan_sha256 = getattr(binding, "plan_sha256")
    workspace_instance_id = getattr(binding, "workspace_instance_id")
    declaration = {
        "schema": "scnsim.operation_trace",
        "schema_version": 1,
        "plan_sha256": plan_sha256,
        "workspace_instance_id": workspace_instance_id,
    }
    trace_sha256 = sha256(_record_bytes(declaration)).hexdigest()
    target = _inside(root, _RECORD_NAME)
    with _workspace_lock(root, exclusive=True):
        if target.is_symlink():
            raise _integrity("Operation record path must not be a symlink.", path=str(target))
        if target.exists():
            existing = _read_document(root)
            if (
                existing.get("schema_version") != _V2
                or existing.get("benchmark_sha256") != trace_sha256
                or existing.get("declaration") != declaration
            ):
                raise _integrity(
                    "Operation journal belongs to another Plan leaf.",
                    expected_plan_sha256=plan_sha256,
                )
            _ensure_global_head(root, trace_sha256)
            return root
        record = {
            "schema": _RECORD_SCHEMA,
            "schema_version": _V2,
            "benchmark_sha256": trace_sha256,
            "declaration": declaration,
            "plan_sha256": plan_sha256,
            "workspace_instance_id": workspace_instance_id,
            "clock": dict(clock_binding),
            "tasks": [],
        }
        _ensure_global_head(root, trace_sha256)
        _atomic_write(target, _record_bytes(record))
    return root


def start_operation(binding: object, row: Mapping[str, object]) -> None:
    """Durably publish a truthful running root before request preparation."""
    root = operation_workspace(binding)
    if not (root / _RECORD_NAME).is_file():
        raise _integrity("Operation journal was not initialized before recording.")
    _append_global_changes(
        root,
        ({"field": "operation_events", "value": {"event": "started", "row": dict(row)}},),
    )


def bind_operation(
    binding: object,
    *,
    operation_id: str,
    operation: str,
    request_sha256: str,
    task_id: str | None,
    environment_sha256: str | None,
    attempt_id: str | None,
) -> None:
    """Record identities known so far without making an attempt."""
    root = operation_workspace(binding)
    _append_global_changes(
        root,
        ({
            "field": "operation_events",
            "value": {
                "event": "bound",
                "operation_id": operation_id,
                "operation": operation,
                "request_sha256": request_sha256,
                "task_id": task_id,
                "environment_sha256": environment_sha256,
                "attempt_id": attempt_id,
            },
        },),
    )


def bind_operation_attempt(
    binding: object,
    *,
    operation_id: str,
    attempt_id: str,
) -> None:
    """Record the concrete attempt allocated for one operation invocation."""
    root = operation_workspace(binding)
    _append_global_changes(
        root,
        ({
            "field": "operation_events",
            "value": {
                "event": "attempt_bound",
                "operation_id": operation_id,
                "attempt_id": attempt_id,
            },
        },),
    )


def record_operation_event(
    binding: object,
    value: Mapping[str, object],
) -> None:
    """Append one small operation lifecycle fact to the shared global journal."""
    root = operation_workspace(binding)
    _append_global_changes(
        root,
        ({"field": "operation_events", "value": dict(value)},),
    )


def finish_operation(
    binding: object,
    row: Mapping[str, object],
    *,
    failure: Mapping[str, object] | None,
    spans: Sequence[Mapping[str, object]] = (),
) -> None:
    """Append the closed root and original failure classification, if any."""
    root = operation_workspace(binding)
    value: dict[str, object] = {"event": "finished", "row": dict(row)}
    if spans:
        value["spans"] = [dict(item) for item in spans]
    if failure is not None:
        value["failure"] = _error_evidence(dict(failure))  # type: ignore[assignment]
    _append_global_changes(
        root, ({"field": "operation_events", "value": value},),
    )


def ensure_operation_task(
    binding: object,
    task: Mapping[str, object],
) -> dict[str, Any]:
    """Register or reopen the exact request/environment-bound task journal."""
    return ensure_task(operation_workspace(binding), task)


def operation_task_record(binding: object, task_id: str) -> dict[str, Any]:
    """Materialize one verified operation task without choosing another task."""
    root = operation_workspace(binding)
    task = task_record(root, task_id)
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
    request_bytes = journal.read_immutable(root, request_reference, role="operation_request")
    if sha256(request_bytes).hexdigest() != request_sha256:
        raise _integrity("Operation request artifact does not match its task identity.", task_id=task_id)
    return task


def store_operation_request(
    binding: object,
    *,
    request_sha256: str,
    request_bytes: bytes,
) -> dict[str, object]:
    """Archive exact canonical request bytes once for all policy task variants."""
    if sha256(request_bytes).hexdigest() != request_sha256:
        raise _integrity("Prepared operation request bytes do not match their identity.")
    root = operation_workspace(binding)
    with _workspace_lock(root, exclusive=True):
        return journal.write_immutable(
            root,
            f"requests/{request_sha256}/request.json",
            request_bytes,
            role="operation_request",
        )


def open_operation_record(binding: object) -> BenchmarkResult | None:
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


def find_operation_success(
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


def read_operation_success(
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
    task = operation_task_record(binding, task_id)
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
    task = operation_task_record(binding, str(task_id))
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


def select_operation_success(
    binding: object,
    task_id: str,
    attempt_id: str,
) -> dict[str, object]:
    """Commit the first verified success reference for this numerical request."""
    root = operation_workspace(binding)
    task = operation_task_record(binding, task_id)
    success = _operation_success_from_task(task, attempt_id=attempt_id)
    if success is None:
        raise _integrity("Cannot select an operation attempt without a completed result.",
                         task_id=task_id, attempt_id=attempt_id)
    request_sha256 = task.get("request_sha256")
    assert isinstance(request_sha256, str)
    candidate: dict[str, object] = {
        "event": "request_success_selected",
        "request_sha256": request_sha256,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "environment_sha256": success["environment_sha256"],
        "attempt_sha256": success["attempt_sha256"],
        "result_sha256": success["result_sha256"],
        "result_ref": success["result_ref"],
    }
    with _workspace_lock(root, exclusive=True):
        manifest = _operation_manifest(binding, root=root)
        if manifest is None:
            raise _integrity("Operation journal disappeared before success selection.")
        descriptor = _task_binding(manifest, task_id)
        if descriptor.get("request_sha256") != request_sha256:
            raise _integrity("Operation task identity changed before success selection.", task_id=task_id)
        existing = _operation_success_selection(root, manifest, request_sha256)
        if existing is not None:
            return existing
        head = _global_head(root, manifest.get("benchmark_sha256"))
        _append_global_commit(
            root, manifest, head,
            changes=({"field": "operation_events", "value": candidate},),
        )
        journal.publish_head(root, "journal/global/HEAD.json", head)
    return candidate


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
    return journal.write_immutable(root, relative.as_posix(), payload, role=role)


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


def _register_descriptor_locked(root: Path, manifest: dict[str, Any], descriptor: Mapping[str, object]) -> None:
    task_id = str(descriptor["task_id"])
    existing = next((item for item in manifest["tasks"] if item.get("task_id") == task_id), None)
    head_path = _inside(root, f"tasks/{task_id}/journal/HEAD.json")
    if existing is not None:
        if existing != dict(descriptor):
            raise _integrity("Benchmark task identity was rebound.", task_id=task_id)
        _task_head(root, existing)
        return
    if head_path.exists():
        raise _integrity("Unreferenced benchmark task head cannot be promoted.", task_id=task_id)
    head = _empty_task_head(task_id, str(descriptor["request_sha256"]))
    _publish_task_head(root, head)
    manifest["tasks"].append(dict(descriptor))
    _atomic_write(_inside(root, _RECORD_NAME), _record_bytes(manifest))


def register_task(
    workspace: str | os.PathLike[str],
    task: Mapping[str, object],
) -> None:
    """Append one compact task binding; task history lives only in its journal."""
    root = _root_path(workspace)
    descriptor = _new_task_descriptor(task)
    with _workspace_lock(root, exclusive=True):
        manifest = _require_v2(root)
        _register_descriptor_locked(root, manifest, descriptor)


def ensure_task(
    workspace: str | os.PathLike[str],
    task: Mapping[str, object],
) -> dict[str, Any]:
    """Register a task once, or return its verified existing detached record."""
    root = _root_path(workspace)
    descriptor = _new_task_descriptor(task)
    with _workspace_lock(root, exclusive=True):
        manifest = _require_v2(root)
        task_id = str(descriptor["task_id"])
        prior = next((item for item in manifest["tasks"] if item.get("task_id") == task_id), None)
        if prior is None:
            _register_descriptor_locked(root, manifest, descriptor)
        else:
            stable_prior = {key: prior[key] for key in ("task_id", "request_sha256", "arm", "sample")}
            stable_value = {key: descriptor[key] for key in ("task_id", "request_sha256", "arm", "sample")}
            if stable_prior != stable_value or prior["environment"].get("environment_sha256") != descriptor["environment"].get("environment_sha256"):
                raise _integrity("Benchmark task identity was rebound to another environment.", task_id=task_id)
            _task_head(root, prior)
    return materialize_task(root, str(descriptor["task_id"]))


def _update_head_attempt(
    head: dict[str, Any],
    *,
    attempt_id: str,
    status: str | None = None,
    failure: Mapping[str, object] | None = None,
    interruption: Mapping[str, object] | None = None,
    checkpoint: Mapping[str, object] | None = None,
) -> dict[str, object]:
    attempts = head["attempts"]
    attempt = next((item for item in attempts if item.get("attempt_id") == attempt_id), None)
    if attempt is None:
        raise _integrity("Benchmark attempt is not recorded.", task_id=head["task_id"], attempt_id=attempt_id)
    if status is not None:
        attempt["status"] = status
        attempt["failure"] = None if failure is None else json.loads(_record_bytes(failure))
        attempt["interruption"] = None if interruption is None else json.loads(_record_bytes(interruption))
    if checkpoint is not None:
        attempt["checkpoint"] = json.loads(_record_bytes(checkpoint))
    return attempt


def _attempt_change_operation(
    *,
    attempt_id: str,
    status: str | None = None,
    failure: Mapping[str, object] | None = None,
    interruption: Mapping[str, object] | None = None,
    checkpoint: Mapping[str, object] | None = None,
    artifacts: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    return {
        "kind": "attempt_update",
        "attempt_id": attempt_id,
        "status": status,
        "failure": None if failure is None else json.loads(_record_bytes(failure)),
        "interruption": None if interruption is None else json.loads(_record_bytes(interruption)),
        "checkpoint": None if checkpoint is None else json.loads(_record_bytes(checkpoint)),
        "artifacts": [json.loads(_record_bytes(item)) for item in artifacts],
    }


def begin_attempt(
    workspace: str | os.PathLike[str],
    *,
    task_id: str,
    attempt_id: str,
    resume_from: Mapping[str, object] | None = None,
) -> None:
    """Persist an allocated attempt before the child is authorized to advance."""
    root = _root_path(workspace)
    resume = None if resume_from is None else json.loads(_record_bytes(resume_from))
    with _workspace_lock(root, exclusive=True):
        manifest = _require_v2(root)
        binding = _task_binding(manifest, task_id)
        head = _task_head(root, binding)
        if any(item["attempt_id"] == attempt_id for item in head["attempts"]):
            raise _integrity("Benchmark attempt identifier is already recorded.", attempt_id=attempt_id)
        attempt = {
            "attempt_id": attempt_id,
            "status": "allocated",
            "resume_from": resume,
            "artifacts": [],
            "failure": None,
            "interruption": None,
        }
        _append_task_commit(root, binding, head, operation={"kind": "attempt_begin", "attempt": attempt})
        head["attempts"].append({key: value for key, value in attempt.items() if key != "artifacts"})
        head["last_attempt_id"] = attempt_id
        _publish_task_head(root, head)


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
    """Publish an attempt transition and artifact additions to its task head."""
    root = _root_path(workspace)
    operation = _attempt_change_operation(
        attempt_id=attempt_id, status=status, failure=failure, interruption=interruption,
        checkpoint=checkpoint, artifacts=artifacts,
    )
    with _workspace_lock(root, exclusive=True):
        manifest = _require_v2(root)
        binding = _task_binding(manifest, task_id)
        head = _task_head(root, binding)
        _update_head_attempt(
            head, attempt_id=attempt_id, status=status, failure=failure,
            interruption=interruption, checkpoint=checkpoint,
        )
        _append_task_commit(root, binding, head, operation=operation)
        _publish_task_head(root, head)


def _barrier_block_locked(
    root: Path,
    binding: Mapping[str, object],
    head: Mapping[str, object],
    *,
    kind: str,
    payload: Mapping[str, object],
    event_sequence: int,
    evaluation_rows: Sequence[Mapping[str, object]],
    baseline_block: Mapping[str, object] | None,
    generation_block: Mapping[str, object] | None,
    inherited_value_refs: Mapping[str, Mapping[str, object]] | None,
    durability_witness: journal._DurabilityWitness | None = None,
    phase_scope: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> tuple[dict[str, object], dict[str, object] | None, dict[str, object], tuple[dict[str, object], ...]]:
    task_id = str(binding["task_id"])
    attempt_id = str(payload.get("attempt_id", ""))
    if kind == "baseline_ready":
        baseline_value = payload.get("baseline")
        if not isinstance(baseline_value, Mapping):
            raise _integrity("Benchmark baseline barrier lacks its exact baseline record.", task_id=task_id)
        baseline_ref, baseline_occurrence, _ = _value_reference(
            root, task_id, baseline_value, attempt_id=attempt_id,
            event_sequence=event_sequence, suffix="baseline", inherited=inherited_value_refs,
            durability_witness=durability_witness,
        )
        anchors_ref = None
        if isinstance(payload.get("anchors"), Mapping):
            anchors = {
                "schema": "scnsim.benchmark_anchor_values",
                "schema_version": _V2,
                "task_id": task_id,
                "anchors": dict(payload["anchors"]),
            }
            anchor_bytes = _record_bytes(anchors)
            anchor_sha = sha256(anchor_bytes).hexdigest()
            anchors_ref = journal.write_immutable(
                root,
                f"tasks/{task_id}/evidence/anchors/{anchor_sha}.json",
                anchor_bytes,
                role="benchmark_anchor_values",
                durability_witness=durability_witness,
            )
        attributes = {
            key: value for key, value in payload.items()
            if key not in {"attempt_id", "baseline", "anchors", "resume_state"}
        }
        block = {
            "schema": "scnsim.benchmark_baseline_evidence",
            "schema_version": _V2,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "baseline": baseline_ref,
            "baseline_occurrence": baseline_occurrence,
            "anchors": anchors_ref,
            "attributes": attributes,
        }
        raw = _record_bytes(block)
        digest = sha256(raw).hexdigest()
        evidence_ref = journal.write_immutable(
            root,
            f"tasks/{task_id}/evidence/baseline/{digest}.json",
            raw,
            role="benchmark_baseline_evidence",
            durability_witness=durability_witness,
        )
        next_generation_block = None
        checkpoint_generation = int(baseline_value.get("generation", 0))
        next_ordinal = int(baseline_value.get("evaluation_ordinal", 0)) + 1
        best_ordinal = int(baseline_value.get("evaluation_ordinal", 0))
        evidence_for_checkpoint = {"baseline": evidence_ref, "generation": None}
    elif kind == "generation_ready":
        baseline_ref = baseline_block
        if not isinstance(baseline_ref, Mapping):
            raise _integrity("Generation barrier has no committed baseline evidence.", task_id=task_id)
        rows: list[dict[str, object]] = []
        for index, row in enumerate(evaluation_rows):
            value_ref, occurrence, _ = _value_reference(
                root, task_id, row, attempt_id=attempt_id, event_sequence=event_sequence,
                suffix=f"generation-{payload.get('generation')}-{index}",
                inherited=inherited_value_refs,
                durability_witness=durability_witness,
            )
            rows.append({"value": value_ref, "occurrence": occurrence})
        attributes = {
            key: value for key, value in payload.items()
            if key not in {"attempt_id", "resume_state"}
        }
        block = {
            "schema": "scnsim.benchmark_generation_evidence",
            "schema_version": _V2,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "baseline": dict(baseline_ref),
            "previous": None if generation_block is None else dict(generation_block),
            "rows": rows,
            "attributes": attributes,
        }
        raw = _record_bytes(block)
        digest = sha256(raw).hexdigest()
        evidence_ref = journal.write_immutable(
            root,
            f"tasks/{task_id}/evidence/generations/{digest}.json",
            raw,
            role="benchmark_generation_evidence",
            durability_witness=durability_witness,
        )
        next_generation_block = evidence_ref
        checkpoint_generation = int(payload["generation"])
        next_ordinal = int(payload["next_ordinal"])
        best_ordinal = int(payload["best_ordinal"])
        evidence_for_checkpoint = {"baseline": dict(baseline_ref), "generation": evidence_ref}
    else:
        raise _integrity("Unsupported benchmark evidence barrier.", kind=kind)

    resume_state = payload.get("resume_state")
    checkpoint_ref = None
    checkpoint_artifacts: tuple[dict[str, object], ...] = ()
    if isinstance(resume_state, Mapping):
        cma = resume_state.get("cma")
        checkpoint = {
            "schema": "scnsim.benchmark_cma_checkpoint",
            "schema_version": _V2,
            "task_id": task_id,
            "request_sha256": binding["request_sha256"],
            "arm": binding["arm"],
            "sample": binding["sample"],
            "environment_sha256": binding["environment"]["environment_sha256"],
            "attempt_id": attempt_id,
            "generation": checkpoint_generation,
            "next_ordinal": next_ordinal,
            "best_ordinal": best_ordinal,
            "baseline_evidence": evidence_for_checkpoint["baseline"],
            "generation_evidence": evidence_for_checkpoint["generation"],
            "cma": cma,
        }
        if phase_scope is None:
            checkpoint_bytes = _record_bytes(checkpoint)
            checkpoint_ref, checkpoint_artifacts = _save_checkpoint_locked(
                root, binding, attempt_id, checkpoint_bytes,
                durability_witness=durability_witness,
            )
        else:
            details: dict[str, Any] = {
                "generation": checkpoint_generation,
                "checkpoint_policy": str(binding["arm"]).rsplit("/", 1)[-1],
            }
            with phase_scope("checkpoint_state_publish", details):
                checkpoint_bytes = _record_bytes(checkpoint)
                details["checkpoint_bytes"] = len(checkpoint_bytes)
                checkpoint_ref, checkpoint_artifacts = _save_checkpoint_locked(
                    root, binding, attempt_id, checkpoint_bytes,
                    durability_witness=durability_witness,
                )
    event_payload: dict[str, object] = {
        key: value for key, value in payload.items() if key == "attempt_id"
    }
    event_payload[_JOURNAL_MARKER] = {
        "kind": "barrier",
        "evidence": evidence_ref,
        "checkpoint": checkpoint_ref,
    }
    return evidence_ref, checkpoint_ref, event_payload, checkpoint_artifacts


def _commit_task_events(
    root: Path,
    task_id: str,
    events: Sequence[tuple[str, Mapping[str, object]]],
    *,
    barrier: tuple[str, Mapping[str, object], Sequence[Mapping[str, object]], Mapping[str, object] | None, Mapping[str, object] | None] | None = None,
    inherited_value_refs: Mapping[str, Mapping[str, object]] | None = None,
    publication_state: dict[str, object] | None = None,
    phase_scope: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    saved_events: list[dict[str, object]] = []
    acknowledgment: dict[str, object] | None = None
    if publication_state is not None:
        publication_state.clear()
        publication_state.update({"phase": "before_lock", "path": f"tasks/{task_id}/journal/HEAD.json"})
    with _workspace_lock(root, exclusive=True):
        if publication_state is not None:
            publication_state["phase"] = "lock_acquired"
        manifest = _require_v2(root)
        binding = _task_binding(manifest, task_id)
        durability_witness = journal._DurabilityWitness()
        head = _task_head(root, binding, durability_witness=durability_witness)
        prior_head_bytes = _record_bytes(head)
        head_path = f"tasks/{task_id}/journal/HEAD.json"
        if publication_state is not None:
            publication_state.update({
                "phase": "head_snapshotted",
                "path": head_path,
                "prior_head_bytes": prior_head_bytes,
                "prior_sha256": sha256(prior_head_bytes).hexdigest(),
                "new_head_bytes": None,
                "new_sha256": None,
                "publish_returned": False,
            })
        for kind, original_payload in events:
            sequence = int(head["event_sequence"])
            payload = json.loads(_record_bytes(original_payload))
            event_attempt = payload.get("attempt_id")
            if event_attempt is not None and event_attempt != head.get("last_attempt_id"):
                raise _integrity(
                    "Benchmark event does not belong to the current task attempt.",
                    task_id=task_id, attempt_id=event_attempt,
                )
            value_attempt_id = event_attempt if isinstance(event_attempt, str) else head.get("last_attempt_id")
            if kind == "evaluation":
                if not isinstance(value_attempt_id, str):
                    raise _integrity("Benchmark evaluation lacks a bound task attempt.", task_id=task_id)
                compact = _compact_value_event(
                    root, task_id, payload,
                    attempt_id=value_attempt_id,
                    event_sequence=sequence,
                    inherited=inherited_value_refs,
                    durability_witness=durability_witness,
                )
            elif isinstance(payload.get("numerical_observations"), Mapping):
                if not isinstance(value_attempt_id, str):
                    raise _integrity("Benchmark numerical observation lacks a bound task attempt.", task_id=task_id)
                compact = dict(payload)
                compact["numerical_observations"] = _compact_observations(
                    root, task_id, payload["numerical_observations"],
                    attempt_id=value_attempt_id, event_sequence=sequence,
                    durability_witness=durability_witness,
                )
            else:
                compact = payload
            event = {"task_id": task_id, "sequence": sequence, "kind": kind, "payload": compact}
            _append_task_commit(
                root, binding, head, operation={"kind": "event", "event": event},
                durability_witness=durability_witness,
            )
            head["event_sequence"] = sequence + 1
            saved_events.append({"task_id": task_id, "sequence": sequence, "kind": kind, "payload": payload})
        if barrier is not None:
            kind, payload, evaluation_rows, baseline_block, generation_block = barrier
            sequence = int(head["event_sequence"])
            if payload.get("attempt_id") != head.get("last_attempt_id"):
                raise _integrity(
                    "Benchmark barrier does not belong to the current task attempt.",
                    task_id=task_id, attempt_id=payload.get("attempt_id"),
                )
            evidence_ref, checkpoint_ref, compact, checkpoint_artifacts = _barrier_block_locked(
                root, binding, head, kind=kind, payload=payload,
                event_sequence=sequence, evaluation_rows=evaluation_rows,
                baseline_block=baseline_block, generation_block=generation_block,
                inherited_value_refs=inherited_value_refs,
                durability_witness=durability_witness,
                phase_scope=phase_scope,
            )
            event = {"task_id": task_id, "sequence": sequence, "kind": kind, "payload": compact}
            operation = {
                "kind": "event",
                "event": event,
                "evidence": evidence_ref,
                "checkpoint": checkpoint_ref,
                "artifacts": list(checkpoint_artifacts),
            }
            _append_task_commit(
                root, binding, head, operation=operation,
                durability_witness=durability_witness,
            )
            head["event_sequence"] = sequence + 1
            if checkpoint_ref is not None:
                attempt_id = str(payload.get("attempt_id", ""))
                _update_head_attempt(
                    head, attempt_id=attempt_id, checkpoint=checkpoint_ref,
                )
            saved_events.append({"task_id": task_id, "sequence": sequence, "kind": kind, "payload": dict(payload)})
            acknowledgment = {"evidence": evidence_ref, "checkpoint": checkpoint_ref}
        new_head_bytes = _record_bytes(head)
        if publication_state is not None:
            publication_state["phase"] = "head_publish_attempted"
            publication_state["new_head_bytes"] = new_head_bytes
            publication_state["new_sha256"] = sha256(new_head_bytes).hexdigest()
        try:
            journal.publish_head(
                root, head_path, head,
                durability_witness=durability_witness,
            )
            if publication_state is not None:
                publication_state["publish_returned"] = True
                publication_state["phase"] = "head_publish_returned"
        except BaseException as error:
            publication_marker = getattr(error, journal._HEAD_PUBLICATION_RECONCILIATION, None)
            if not isinstance(publication_marker, dict):
                raise
            reconciliation: dict[str, object] = {
                "path": head_path,
                "prior_sha256": sha256(prior_head_bytes).hexdigest(),
                "new_sha256": sha256(new_head_bytes).hexdigest(),
                "visible_state": "unreadable_after_publication_error",
            }
            try:
                observed = journal.read_head(
                    root, head_path,
                    schema=journal._TASK_HEAD,
                    bind={"task_id": task_id, "request_sha256": binding["request_sha256"]},
                )
                observed_bytes = _record_bytes(observed)
                reconciliation["observed_sha256"] = sha256(observed_bytes).hexdigest()
                if observed_bytes == new_head_bytes:
                    reconciliation["visible_state"] = "new_head_visible_unconfirmed"
                elif observed_bytes == prior_head_bytes:
                    reconciliation["visible_state"] = "prior_head_visible_after_error"
                else:
                    reconciliation["visible_state"] = "other_head_visible_after_error"
            except BaseException as reconciliation_error:
                reconciliation["reconciliation_error_type"] = type(reconciliation_error).__name__
                reconciliation["reconciliation_error"] = str(reconciliation_error)
            publication_marker.update(reconciliation)
            if publication_state is not None:
                publication_state["publication_marker"] = publication_marker
            raise
    return saved_events, acknowledgment


class TaskWriter:
    """One task attempt's boundary buffer and evidence links."""

    def __init__(
        self,
        root: Path,
        task_id: str,
        attempt_id: str,
        *,
        diagnostics: str,
        checkpoint_document: Mapping[str, object] | None = None,
        phase_scope: Callable[[str, Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.root = root
        self.task_id = task_id
        self.attempt_id = attempt_id
        self.diagnostics = diagnostics
        self.phase_scope = phase_scope
        self.pending: list[tuple[str, Mapping[str, object]]] = []
        self.generation_rows: list[Mapping[str, object]] = []
        self.publication_uncertain: dict[str, object] | None = None
        self.unacknowledged_diagnostics: tuple[tuple[str, Mapping[str, object]], ...] = ()
        links = checkpoint_document.get("_journal_links", {}) if checkpoint_document else {}
        self.baseline_block = links.get("baseline_evidence") if isinstance(links, Mapping) else None
        self.generation_block = links.get("generation_evidence") if isinstance(links, Mapping) else None
        self.value_refs = links.get("value_refs", {}) if isinstance(links, Mapping) else {}
        if not isinstance(self.value_refs, Mapping):
            self.value_refs = {}
        with _workspace_lock(root, exclusive=False):
            manifest = _require_v2(root)
            binding = _task_binding(manifest, task_id)
            head = _task_head(root, binding)
            self.sequence_hint = int(head["event_sequence"])

    def _ensure_publishable(self) -> None:
        if self.publication_uncertain is not None:
            raise _integrity(
                "Task writer cannot append after an unacknowledged commit outcome.",
                task_id=self.task_id,
                publication=self.publication_uncertain,
            )

    def _commit_events(
        self,
        events: Sequence[tuple[str, Mapping[str, object]]],
        *,
        barrier: tuple[str, Mapping[str, object], Sequence[Mapping[str, object]], Mapping[str, object] | None, Mapping[str, object] | None] | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object] | None]:
        self._ensure_publishable()
        # Keep an explicit local delta until the caller has either received an
        # ACK or reconciled the selected HEAD. A confirmed prior HEAD permits
        # diagnostic-only failure flushing; a selected new or unknown HEAD
        # must never replay the same delta.
        commit_state: dict[str, object] = {}
        commit_events = tuple(events)
        diagnostic_events = tuple(
            (kind, payload) for kind, payload in commit_events
            if kind in _DIAGNOSTIC_EVENTS
        )
        has_barrier = barrier is not None
        try:
            self.pending.clear()
            if has_barrier:
                self.generation_rows.clear()
            return _commit_task_events(
                self.root, self.task_id, commit_events,
                barrier=barrier,
                inherited_value_refs=self.value_refs,
                publication_state=commit_state,
                phase_scope=self.phase_scope,
            )
        except BaseException as error:
            self.pending.clear()
            self.generation_rows.clear()
            outcome = self._reconcile_commit_outcome(commit_state, error)
            if outcome["visible_state"] == "prior_head_still_selected":
                # Only diagnostic event deltas are safe to flush. The barrier
                # itself and its in-memory population never become a checkpoint.
                self.pending.extend(diagnostic_events)
                self.publication_uncertain = None
                self.unacknowledged_diagnostics = ()
            else:
                self.publication_uncertain = outcome
                self.unacknowledged_diagnostics = (
                    diagnostic_events
                    if outcome["visible_state"] == "head_state_unknown"
                    else ()
                )
            raise

    def _reconcile_commit_outcome(
        self,
        state: Mapping[str, object],
        error: BaseException,
    ) -> dict[str, object]:
        prior = state.get("prior_head_bytes")
        new = state.get("new_head_bytes")
        marker = state.get("publication_marker")
        if not isinstance(marker, Mapping):
            attached = getattr(error, journal._HEAD_PUBLICATION_RECONCILIATION, None)
            marker = attached if isinstance(attached, Mapping) else None
        if isinstance(marker, Mapping):
            visible = marker.get("visible_state")
            if visible == "prior_head_visible_after_error":
                return {**dict(marker), "visible_state": "prior_head_still_selected"}
            if visible == "new_head_visible_unconfirmed":
                return {**dict(marker), "visible_state": "new_head_selected_unconfirmed"}

        if not isinstance(prior, bytes):
            return {
                "path": state.get("path"),
                "visible_state": "prior_head_still_selected",
                "reconciliation": "head_publication_was_not_reached",
            }
        try:
            with _workspace_lock(self.root, exclusive=False):
                manifest = _require_v2(self.root)
                binding = _task_binding(manifest, self.task_id)
                observed = _record_bytes(_task_head(self.root, binding))
        except BaseException as reconciliation_error:
            return {
                "path": state.get("path"),
                "visible_state": "head_state_unknown",
                "prior_sha256": state.get("prior_sha256"),
                "new_sha256": state.get("new_sha256"),
                "reconciliation_error_type": type(reconciliation_error).__name__,
                "reconciliation_error": str(reconciliation_error),
            }
        if observed == prior:
            return {
                "path": state.get("path"),
                "visible_state": "prior_head_still_selected",
                "prior_sha256": state.get("prior_sha256"),
            }
        if isinstance(new, bytes) and observed == new:
            return {
                "path": state.get("path"),
                "visible_state": (
                    "new_head_selected_durable"
                    if state.get("publish_returned") is True
                    else "new_head_selected_unconfirmed"
                ),
                "prior_sha256": state.get("prior_sha256"),
                "new_sha256": state.get("new_sha256"),
            }
        return {
            "path": state.get("path"),
            "visible_state": "head_state_unknown",
            "prior_sha256": state.get("prior_sha256"),
            "new_sha256": state.get("new_sha256"),
            "observed_sha256": sha256(observed).hexdigest(),
        }

    def append_event(
        self,
        *,
        kind: str,
        payload: Mapping[str, object],
        force: bool = False,
    ) -> dict[str, object]:
        self._ensure_publishable()
        value = json.loads(_record_bytes(payload))
        if kind == "evaluation" and isinstance(value.get("generation"), int) and value["generation"] > 0:
            self.generation_rows.append(value)
        if self.diagnostics == "boundary" and kind in _DIAGNOSTIC_EVENTS and not force:
            self.pending.append((kind, value))
            event = {
                "task_id": self.task_id,
                "sequence": self.sequence_hint,
                "kind": kind,
                "payload": value,
            }
            self.sequence_hint += 1
            return event
        saved, _ = self._commit_events((*self.pending, (kind, value)))
        self.pending.clear()
        if saved:
            self.sequence_hint = int(saved[-1]["sequence"]) + 1
            return saved[-1]
        raise _integrity("Benchmark task event was not committed.", task_id=self.task_id)

    def commit_barrier(self, kind: str, payload: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        value = json.loads(_record_bytes(payload))
        saved, acknowledgment = self._commit_events(
            tuple(self.pending),
            barrier=(kind, value, tuple(self.generation_rows), self.baseline_block, self.generation_block),
        )
        self.pending.clear()
        if acknowledgment is None or not saved:
            raise _integrity("Benchmark evidence barrier was not committed.", task_id=self.task_id)
        if kind == "baseline_ready":
            self.baseline_block = acknowledgment["evidence"]
            self.generation_block = None
        else:
            self.generation_block = acknowledgment["evidence"]
        self.generation_rows.clear()
        self.sequence_hint = int(saved[-1]["sequence"]) + 1
        return saved[-1], acknowledgment

    def flush(self) -> None:
        if self.publication_uncertain is not None:
            raise _integrity(
                "Task writer cannot flush after an unacknowledged commit outcome.",
                task_id=self.task_id,
                publication=self.publication_uncertain,
                retained_diagnostic_event_count=len(self.unacknowledged_diagnostics),
            )
        if not self.pending:
            return
        saved, _ = self._commit_events(tuple(self.pending))
        self.pending.clear()
        if saved:
            self.sequence_hint = int(saved[-1]["sequence"]) + 1


def begin_task_writer(
    workspace: str | os.PathLike[str],
    *,
    task_id: str,
    attempt_id: str,
    diagnostics: str,
    checkpoint_document: Mapping[str, object] | None = None,
    phase_scope: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> TaskWriter:
    return TaskWriter(
        _root_path(workspace), task_id, attempt_id,
        diagnostics=diagnostics, checkpoint_document=checkpoint_document,
        phase_scope=phase_scope,
    )


def append_event(
    workspace: str | os.PathLike[str],
    *,
    task_id: str,
    kind: str,
    payload: Mapping[str, object],
    writer: TaskWriter | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Append one ordered task event, publishing its immutable body first."""
    if writer is not None:
        return writer.append_event(kind=kind, payload=payload, force=force)
    root = _root_path(workspace)
    saved, _ = _commit_task_events(root, task_id, ((kind, payload),))
    return saved[0]


def commit_barrier(
    workspace: str | os.PathLike[str],
    *,
    writer: TaskWriter,
    kind: str,
    payload: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    return writer.commit_barrier(kind, payload)


def task_record(workspace: str | os.PathLike[str], task_id: str) -> dict[str, Any]:
    """Return one stored task record without selecting or mutating other tasks."""
    root = _root_path(workspace)
    manifest = _read_document(root)
    if manifest.get("schema_version") == _V1:
        return json.loads(_record_bytes(_task_binding(manifest, task_id)))
    return materialize_task(root, task_id)


def _append_global_changes(
    root: Path,
    changes: Sequence[Mapping[str, object]],
) -> None:
    with _workspace_lock(root, exclusive=True):
        manifest = _require_v2(root)
        _ensure_global_head(root, manifest.get("benchmark_sha256"))
        head = _global_head(root, manifest.get("benchmark_sha256"))
        _append_global_commit(root, manifest, head, changes=changes)
        journal.publish_head(root, "journal/global/HEAD.json", head)


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
        "error": _error_document(error),
    }
    _append_global_changes(
        _root_path(workspace), ({"field": "task_launch_failures", "value": failure},),
    )


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
    failure = {
        "phase": phase,
        "status": "failure",
        "benchmark_sha256": benchmark_sha256,
        "plan_sha256": plan_sha256,
        "source_analysis_sha256": source_analysis_sha256,
        "error": _error_document(error),
    }
    _append_global_changes(
        _root_path(workspace), ({"field": "execution_failures", "value": failure},),
    )


def record_callback_failure(
    workspace: str | os.PathLike[str],
    *,
    task_id: str,
    sequence: int,
    event_kind: str,
    error: BaseException,
) -> None:
    _ = task_record(workspace, task_id)
    failure = {
        "task_id": task_id,
        "event_sequence": sequence,
        "event_kind": event_kind,
        "error": _error_document(error),
    }
    _append_global_changes(
        _root_path(workspace), ({"field": "callback_failures", "value": failure},),
    )


def _measurement_changes(
    root: Path,
    values: Sequence[Mapping[str, object]],
) -> None:
    by_task: dict[str, list[dict[str, object]]] = {}
    benchmark_values: list[dict[str, object]] = []
    for value in values:
        task_id = str(value["task_id"])
        if task_id == "benchmark":
            benchmark_values.append(dict(value))
        else:
            by_task.setdefault(task_id, []).append(dict(value))
    changes: list[dict[str, object]] = []
    if benchmark_values:
        changes.append({"field": "measurements", "value": benchmark_values})
    for task_id, measurements in by_task.items():
        changes.append({"field": "task_measurements", "task_id": task_id, "value": measurements})
    if changes:
        _append_global_changes(root, changes)


def append_measurement(
    workspace: str | os.PathLike[str],
    measurement: Measurement,
    *,
    clock_binding: Mapping[str, object],
) -> None:
    value = _measurement_document(measurement, clock_binding=clock_binding)
    _measurement_changes(_root_path(workspace), (value,))


def append_measurements(
    workspace: str | os.PathLike[str],
    measurements: tuple[Measurement, ...],
    *,
    clock_binding: Mapping[str, object],
) -> None:
    values = tuple(_measurement_document(item, clock_binding=clock_binding) for item in measurements)
    if values:
        _measurement_changes(_root_path(workspace), values)


def _save_checkpoint_locked(
    root: Path,
    binding: Mapping[str, object],
    attempt_id: str,
    checkpoint_bytes: bytes,
    *,
    durability_witness: journal._DurabilityWitness | None = None,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    try:
        checkpoint = record_document(checkpoint_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _integrity("Benchmark checkpoint is not JSON.", error=str(error)) from error
    if not isinstance(checkpoint, dict) or _record_bytes(checkpoint) != checkpoint_bytes:
        raise _integrity("Benchmark checkpoint bytes are not canonical JSON.")
    checkpoint_sha = sha256(checkpoint_bytes).hexdigest()
    seal_doc = checkpoint_seal(
        task_id=str(binding["task_id"]), request_sha256=str(binding["request_sha256"]),
        arm=str(binding["arm"]), sample=int(binding["sample"]),
        environment_sha256=str(binding["environment"]["environment_sha256"]),
        attempt_id=attempt_id, checkpoint_sha256=checkpoint_sha, byte_length=len(checkpoint_bytes),
    )
    checkpoint_rel = f"tasks/{binding['task_id']}/attempts/{attempt_id}/checkpoints/{checkpoint_sha}.json"
    seal_rel = checkpoint_rel[:-5] + ".seal.json"
    checkpoint_artifact = journal.write_immutable(
        root, checkpoint_rel, checkpoint_bytes, role="checkpoint",
        durability_witness=durability_witness,
    )
    seal_bytes = _record_bytes(seal_doc)
    seal_artifact = journal.write_immutable(
        root, seal_rel, seal_bytes, role="checkpoint_seal",
        durability_witness=durability_witness,
    )
    reference = {
        "task_id": binding["task_id"], "request_sha256": binding["request_sha256"],
        "arm": binding["arm"], "sample": binding["sample"],
        "environment_sha256": binding["environment"]["environment_sha256"],
        "attempt_id": attempt_id, "checkpoint": checkpoint_artifact,
        "seal": seal_artifact, "seal_sha256": seal_doc["seal_sha256"],
    }
    return reference, (checkpoint_artifact, seal_artifact)


def publish_checkpoint(
    workspace: str | os.PathLike[str],
    *,
    task_id: str,
    attempt_id: str,
    checkpoint_bytes: bytes,
) -> dict[str, object]:
    """Durably publish and seal canonical checkpoint bytes before returning."""
    root = _root_path(workspace)
    with _workspace_lock(root, exclusive=True):
        manifest = _require_v2(root)
        binding = _task_binding(manifest, task_id)
        head = _task_head(root, binding)
        reference, artifacts = _save_checkpoint_locked(root, binding, attempt_id, checkpoint_bytes)
        _update_head_attempt(head, attempt_id=attempt_id, checkpoint=reference)
        operation = _attempt_change_operation(
            attempt_id=attempt_id, checkpoint=reference, artifacts=artifacts,
        )
        _append_task_commit(root, binding, head, operation=operation)
        _publish_task_head(root, head)
    return reference


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


def record_report(
    workspace: str | os.PathLike[str],
    report: Mapping[str, object],
) -> None:
    value = json.loads(_record_bytes(report))
    _append_global_changes(_root_path(workspace), ({"field": "reports", "value": value},))


def record_preparation_failure(
    workspace: str | os.PathLike[str],
    *,
    error: BaseException,
    timing: object,
    plan_sha256: str | None = None,
) -> None:
    """Preserve pre-task failure timing without inventing a prepared identity."""
    invocation_id = str(uuid.uuid4())
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
        "error": _error_document(error),
        "measurements": measurements,
    }
    root = _root_path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    target = _inside(root, _RECORD_NAME)
    with _workspace_lock(root, exclusive=True):
        if target.is_symlink():
            raise _integrity("Benchmark record path must not be a symlink.", path=str(target))
        if target.exists():
            manifest = _read_document(root)
            if manifest.get("schema_version") != _V2:
                raise _integrity("Historical benchmark records are read-only.", path=str(target))
        else:
            manifest = {
                "schema": _RECORD_SCHEMA,
                "schema_version": _V2,
                "benchmark_sha256": None,
                "declaration": None,
                "plan_sha256": plan_sha256,
                "source_analysis_sha256": None,
                "clock": clock_binding,
                "tasks": [],
            }
            _ensure_global_head(root, None)
            _atomic_write(target, _record_bytes(manifest))
        _ensure_global_head(root, manifest.get("benchmark_sha256"))
        head = _global_head(root, manifest.get("benchmark_sha256"))
        initial = manifest.get("benchmark_sha256") is None and not any(
            change.get("field") == "preparation_failure"
            for commit in _global_chain(root, manifest, head)
            for change in commit.get("changes", ())
        )
        field = "preparation_failure" if initial else "preparation_failures"
        _append_global_commit(
            root, manifest, head,
            changes=({"field": field, "value": failure},),
        )
        journal.publish_head(root, "journal/global/HEAD.json", head)
