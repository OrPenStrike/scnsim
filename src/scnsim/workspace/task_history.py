"""Current SQLite task-history and CMA checkpoint projections.

This module consumes only the current SQLite stream and object references. It
contains no file-journal reader or fallback.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from ..errors import EvidenceIntegrityError
from ..numeric_encoding import record_bytes, record_document
from ..diagnostics.identity import checkpoint_seal

_JOURNAL_MARKER = '$scnsim_benchmark_journal'

def _integrity(message: str, **evidence: object) -> EvidenceIntegrityError:
    return EvidenceIntegrityError(message, stage='operation_store', evidence=evidence)

def _record_bytes(value: Mapping[str, object]) -> bytes:
    return record_bytes(dict(value))

def _read_immutable(root, reference, *, role):
    from .sqlite_storage import read_object
    return read_object(root, reference, role=role)

def _read_domain_document(root, reference, *, schema, role, bind):
    from .sqlite_storage import read_document
    return read_document(root, reference, schema=schema, role=role, bind=bind)

def _read_value(root, task_id, reference):
    del task_id
    from .sqlite_storage import read_value
    return dict(read_value(root, reference))

def _materialize_value_marker(root, task_id, value):
    reference=value.get('reference')
    occurrence=value.get('occurrence',{})
    if not isinstance(reference, Mapping) or not isinstance(occurrence, Mapping):
        raise _integrity('Benchmark journal value reference is malformed.')
    result=_read_value(root,task_id,reference)
    result.update(dict(occurrence))
    return result

def _manifest_result_kind(manifest):
    declaration=manifest.get('declaration',{}) if isinstance(manifest,Mapping) else {}
    return declaration.get('result_kind') if isinstance(declaration,Mapping) else None

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
    *, payload_expander=None, project_observations=True, sparse_events=False,
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
            if (sequence < expected_event_sequence if sparse_events else sequence != expected_event_sequence):
                raise _integrity("Benchmark task event sequence is not contiguous.", task_id=binding["task_id"])
            expected_event_sequence = sequence + 1
            kind = str(event["kind"])
            compact_payload = event.get("payload")
            if not isinstance(compact_payload, Mapping):
                raise _integrity("Benchmark task event payload is malformed.", task_id=binding["task_id"])
            payload, baseline_reference = (payload_expander or _expand_payload)(
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
    if not sparse_events and expected != expected_event_sequence:
        raise _integrity("Benchmark task head event sequence does not match its commit chain.", task_id=binding["task_id"])
    for event in task["events"] if project_observations else ():
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


def _hydrate_cma_checkpoint(
    root: Path,
    task_id: str,
    checkpoint: Mapping[str, object],
    *,
    spool=None,
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
    reversed_blocks: list[object] = []
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
        # The live operation resume needs the immutable block links, not a
        # second in-memory copy of every checkpoint block. Re-read one block
        # at a time in chronological order below. Legacy callers retain their
        # original materialized return shape.
        if spool is None:
            reversed_blocks.append((block_ref, block))
        else:
            reversed_blocks.append(dict(block_ref))
        block_ref = block.get("previous")
    generation_blocks = reversed(reversed_blocks)
    cache: dict[str, object] = {}
    value_refs: dict[str, Mapping[str, object]] = {}
    records_by_generation: list[list[dict[str, object]]] = []
    bodies_by_reference: dict[tuple[tuple[str, object], ...], dict[str, object]] = {}

    def value_body(reference: Mapping[str, object]) -> dict[str, object]:
        if spool is not None:
            return _read_value(root, task_id, reference)
        key = tuple(sorted(reference.items()))
        if key not in bodies_by_reference:
            bodies_by_reference[key] = _read_value(root, task_id, reference)
        return bodies_by_reference[key]

    best_ordinal = checkpoint.get("best_ordinal")
    best: dict[str, object] | None = None
    baseline_key = baseline.get("candidate_key")
    if spool is not None:
        # The active evaluator needs only dependencies that back sealed
        # baseline anchors. Keep the full canonical baseline result in the
        # spool cache by reference, and release unrelated decoded branches.
        dependencies = baseline.get("dependencies", {})
        if isinstance(dependencies, Mapping):
            anchor_ids = set(anchors.values())
            baseline["dependencies"] = {
                key: dependencies[key] for key in anchor_ids
            }
    if isinstance(baseline_key, str):
        baseline_body = value_body(baseline_block["baseline"])
        cache[baseline_key] = (spool.put_record(baseline_body)
                               if spool is not None else baseline_body)
        value_refs[baseline_key] = dict(baseline_block["baseline"])
    if baseline.get("evaluation_ordinal") == best_ordinal:
        best = ({"evaluation_ordinal": baseline["evaluation_ordinal"],
                 "cost_f64": baseline["cost_f64"]}
                if spool is not None else baseline)
    for block_item in generation_blocks:
        if spool is None:
            _, block = block_item
        else:
            block = _read_evidence_block(
                root, task_id, block_item,
                schema="scnsim.benchmark_generation_evidence", role="benchmark_generation_evidence",
            )
        rows: list[dict[str, object]] = []
        raw_rows = block.get("rows", ())
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise _integrity("CMA generation evidence rows are malformed.", task_id=task_id)
        for row in raw_rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("value"), Mapping):
                raise _integrity("CMA generation evidence row is malformed.", task_id=task_id)
            body = value_body(row["value"])
            record = dict(body)
            record.update(row.get("occurrence", {}))
            key = record.get("candidate_key")
            if isinstance(key, str):
                cache[key] = (spool.put_record(body) if spool is not None else body)
                value_refs.setdefault(key, dict(row["value"]))
            if record.get("evaluation_ordinal") == best_ordinal:
                best = ({"evaluation_ordinal": record["evaluation_ordinal"],
                         "cost_f64": record["cost_f64"]}
                        if spool is not None else record)
            if spool is None:
                rows.append(record)
        if spool is None:
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
        **({} if spool is not None else {"generations": records_by_generation}),
        "_journal_links": {
            "baseline_evidence": dict(baseline_ref),
            "generation_evidence": None if checkpoint.get("generation_evidence") is None else dict(checkpoint["generation_evidence"]),
            "value_refs": value_refs,
        },
    }


def task_document_from_changes(root, descriptor, changes, manifest):
    """Materialize one task from verified current stream rows."""
    events=[row['event'] for row in changes if row.get('kind')=='event']
    head={'task_id':descriptor['task_id'],'event_sequence':len(events)}
    return _task_document_from_chain(root, descriptor, head,
                                     [{'operation':row} for row in changes], manifest)


def operation_success_from_task(
    task: Mapping[str, object],
    *,
    attempt_id: str | None = None,
    projection_overrides: Mapping[str, Mapping[str, object]] | None = None,
    projection_consumer: Callable[[dict[str, object]], Any] | None = None,
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
        override = None if projection_overrides is None else projection_overrides.get(selected_attempt_id)
        if override is not None:
            terminal = override.get("terminal")
            baseline = override.get("baseline")
            evaluations = override.get("evaluations")
            result_kind = "optimization"
            if (not isinstance(terminal, Mapping) or not isinstance(baseline, Mapping)
                    or not isinstance(evaluations, Iterable) or isinstance(evaluations, (str, bytes, Mapping))):
                raise _integrity(
                    "Completed JAX numerical projection is malformed.",
                    task_id=task.get("task_id"), attempt_id=selected_attempt_id,
                )
            projection = {
                "terminal": terminal,
                "baseline": baseline,
                "evaluations": evaluations,
            }
        else:
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
            baseline = None
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
            projection = {
                "terminal": dict(terminal),
                "evaluations": evaluations,
            }
            if result_kind == "optimization":
                assert baseline is not None
                projection["baseline"] = baseline
        environment = task.get("environment")
        environment_sha256 = environment.get("environment_sha256") if isinstance(environment, Mapping) else None
        if not isinstance(environment_sha256, str):
            raise _integrity("Operation task has no runtime environment identity.", task_id=task.get("task_id"))
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
    success = successes[0] if successes else None
    return projection_consumer(success) if success is not None and projection_consumer is not None else success
