"""Fixed, independently re-openable readers for sealed current results.

The reader keeps only immutable path/binding metadata and a detached selection.
Every operation opens one short SQLite snapshot and verifies the selected
completion, index root, locator, and canonical body before returning it.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from ..errors import _freeze
from ..numeric_encoding import record_bytes, record_document
from .operation_store import OperationStore, _reference


@dataclass(frozen=True, slots=True)
class ResultSelection:
    workspace_instance_id: str
    plan_sha256: str
    request_sha256: str
    task_id: str
    attempt_id: str
    attempt_sha256: str
    terminal_attempt_ref: Mapping[str, Any]
    completion_ref: Mapping[str, Any]
    completion_sha256: str
    result_ref: Mapping[str, Any]
    index_root_ref: Mapping[str, Any]
    index_root_sha256: str
    generation_ancestry_ref: Mapping[str, Any] | None
    generation_count: int
    candidate_count: int
    baseline_locator: Mapping[str, Any]
    best_locator: Mapping[str, Any]

    def __post_init__(self):
        for name in ("terminal_attempt_ref", "completion_ref", "result_ref", "index_root_ref",
                     "generation_ancestry_ref", "baseline_locator", "best_locator"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze(value))


def select_result(snapshot, binding_identity: Mapping[str, Any], completion_ref: Mapping[str, Any],
                  result_ref: Mapping[str, Any], *, attempt_sha256: str) -> ResultSelection:
    """Resolve a selected success to its sealed, detached metadata in one snapshot."""
    from .validation.common import _integrity

    def read(reference, role, schema=None):
        if reference.get("role") != role:
            raise _integrity("Selected completion reference has the wrong role.",
                             expected=role, actual=reference.get("role"))
        raw = snapshot.get_object(reference)
        value = record_document(raw)
        if (not isinstance(value, dict) or record_bytes(value) != raw
                or (schema is not None and value.get("schema") != schema)):
            raise _integrity("Selected completion object is malformed.", reference=dict(reference))
        return value

    completion = read(completion_ref, "workspace_completion", "scnsim.workspace_completion")
    if completion.get("schema_version") != 1:
        raise _integrity("Selected completion version is unsupported.")
    expected_identity = {
        "workspace_instance_id": binding_identity["workspace_instance_id"],
        "plan_sha256": binding_identity["plan_sha256"],
    }
    if any(completion.get(key) != value for key, value in expected_identity.items()):
        raise _integrity("Selected completion differs from its bound Workspace.")
    if (completion.get("result_ref") != dict(result_ref)
            or result_ref.get("role") != "operation_result"):
        raise _integrity("Selected completion differs from its committed result.")
    root_ref = completion.get("index_root_ref")
    root = read(root_ref, "workspace_completion_index_root",
                "scnsim.workspace_completion_index_root")
    for key in ("workspace_instance_id", "plan_sha256", "request_sha256", "task_id", "attempt_id",
                "result_ref", "baseline_ref", "generation_ancestry_ref", "generation_count",
                "candidate_count", "best_ordinal"):
        if root.get(key) != completion.get(key):
            raise _integrity("Selected completion index root differs from completion.", field=key)
    generations = root.get("generations")
    if not isinstance(generations, list) or len(generations) != completion.get("generation_count"):
        raise _integrity("Selected completion generation index count is malformed.")
    sql_generations = snapshot.completion_generation_count(completion_ref["sha256"])
    sql_candidates = snapshot.completion_candidate_count(completion_ref["sha256"])
    if (sql_generations != completion["generation_count"]
            or sql_candidates != completion["candidate_count"]):
        raise _integrity("Selected SQL index counts differ from the sealed completion.")
    best_ordinal = completion.get("best_ordinal")
    if best_ordinal == 0:
        best_locator = {"ordinal": 0, "generation": 0,
                        "block_ref": completion["baseline_ref"]}
    else:
        best_locator = snapshot.read_candidate_index(completion_ref["sha256"], best_ordinal)
        if best_locator is None:
            raise _integrity("Selected best candidate is absent from its sealed SQL index.",
                             best_ordinal=best_ordinal)
        generation_row = next((row for row in generations
                               if row.get("generation") == best_locator.get("generation")), None)
        if generation_row is None:
            raise _integrity("Selected best candidate is outside sealed generation ancestry.",
                             best_ordinal=best_ordinal)
        candidate_index = read(generation_row["candidate_index_ref"], "workspace_candidate_index",
                               "scnsim.workspace_candidate_index")
        sealed_row = next((row for row in candidate_index.get("candidates", ())
                           if row.get("ordinal") == best_ordinal), None)
        if sealed_row != best_locator:
            raise _integrity("Selected best SQL locator differs from its sealed index block.",
                             best_ordinal=best_ordinal)
    terminal_attempt_ref = completion.get("terminal_attempt_ref")
    if (not isinstance(terminal_attempt_ref, Mapping)
            or terminal_attempt_ref.get("role") != "attempt"):
        raise _integrity("Selected completion lacks its terminal attempt reference.",
                         task_id=completion.get("task_id"), attempt_id=completion.get("attempt_id"))
    attempt_pointer = snapshot.read_pointer(
        f"attempt/{completion['task_id']}/{completion['attempt_id']}"
    )
    if (attempt_pointer is None or attempt_pointer.get("reference") != dict(terminal_attempt_ref)):
        raise _integrity("Terminal attempt reference differs from the committed attempt pointer.",
                         task_id=completion.get("task_id"), attempt_id=completion.get("attempt_id"))
    attempt = read(terminal_attempt_ref, "attempt")
    if (terminal_attempt_ref.get("sha256") != attempt_sha256
            or attempt.get("attempt_id") != completion.get("attempt_id")
            or attempt.get("status") != "success"
            or result_ref not in attempt.get("artifacts", ())):
        raise _integrity("Terminal attempt differs from the selected completed result.",
                         task_id=completion.get("task_id"), attempt_id=completion.get("attempt_id"))
    return ResultSelection(
        workspace_instance_id=completion["workspace_instance_id"],
        plan_sha256=completion["plan_sha256"],
        request_sha256=completion["request_sha256"],
        task_id=completion["task_id"], attempt_id=completion["attempt_id"],
        attempt_sha256=attempt_sha256, terminal_attempt_ref=dict(terminal_attempt_ref),
        completion_ref=dict(completion_ref), completion_sha256=completion_ref["sha256"],
        result_ref=dict(result_ref), index_root_ref=dict(root_ref),
        index_root_sha256=root_ref["sha256"],
        generation_ancestry_ref=completion.get("generation_ancestry_ref"),
        generation_count=completion["generation_count"], candidate_count=completion["candidate_count"],
        baseline_locator={"block_ref": completion["baseline_ref"], "ordinal": 0, "generation": 0},
        best_locator=best_locator,
    )


def read_optimization_success(snapshot, binding_identity: Mapping[str, Any], *,
                              descriptor: Mapping[str, Any], attempt_id: str,
                              completion_ref: Mapping[str, Any],
                              result_ref: Mapping[str, Any] | None = None,
                              selection: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the fixed reader for one exact successful v6 Optimization attempt.

    The SQL success selection and attempt-state pointers are read in the same
    snapshot by the caller. This function verifies their canonical objects,
    then retains only detached result-selection metadata and paths.
    """
    from .validation.common import _integrity

    def read(reference, role, schema=None):
        if not isinstance(reference, Mapping) or reference.get("role") != role:
            raise _integrity("Selected Optimization reference has the wrong role.",
                             expected=role, actual=None if not isinstance(reference, Mapping)
                             else reference.get("role"))
        raw = snapshot.get_object(reference)
        value = record_document(raw)
        if (not isinstance(value, dict) or record_bytes(value) != raw
                or (schema is not None and value.get("schema") != schema)):
            raise _integrity("Selected Optimization object is malformed or noncanonical.",
                             reference=dict(reference), expected_schema=schema)
        return raw, value

    task_id = descriptor.get("task_id")
    request_sha256 = descriptor.get("request_sha256")
    if (not isinstance(task_id, str) or not isinstance(request_sha256, str)
            or descriptor.get("environment", {}).get("environment_sha256") is None):
        raise _integrity("Selected Optimization task identity is malformed.")
    _, completion = read(completion_ref, "workspace_completion", "scnsim.workspace_completion")
    if result_ref is None:
        result_ref = completion.get("result_ref")
    if not isinstance(result_ref, Mapping) or result_ref.get("role") != "operation_result":
        raise _integrity("Selected Optimization result has the wrong role.",
                         actual=None if not isinstance(result_ref, Mapping) else result_ref.get("role"))
    _, terminal = read(result_ref, "operation_result")
    if terminal.get("type") != "optimization":
        raise _integrity("Selected Optimization result has the wrong result kind.", task_id=task_id)
    if (completion.get("workspace_instance_id") != binding_identity["workspace_instance_id"]
            or completion.get("plan_sha256") != binding_identity["plan_sha256"]
            or completion.get("request_sha256") != request_sha256
            or completion.get("task_id") != task_id
            or completion.get("attempt_id") != attempt_id
            or completion.get("result_ref") != dict(result_ref)):
        raise _integrity("Selected Optimization completion differs from its bound attempt.",
                         task_id=task_id, attempt_id=attempt_id)

    completion_pointer = snapshot.read_pointer(f"completion/{task_id}/{attempt_id}")
    if (completion_pointer is None
            or completion_pointer.get("reference") != dict(completion_ref)):
        raise _integrity("Selected Optimization completion differs from its committed pointer.",
                         task_id=task_id, attempt_id=attempt_id)

    terminal_attempt_ref = completion.get("terminal_attempt_ref")
    if not isinstance(terminal_attempt_ref, Mapping) or terminal_attempt_ref.get("role") != "attempt":
        raise _integrity("Selected Optimization completion lacks its terminal attempt.",
                         task_id=task_id, attempt_id=attempt_id)
    attempt_raw, attempt = read(terminal_attempt_ref, "attempt")
    attempt_sha256 = sha256(attempt_raw).hexdigest()
    attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
    if (attempt_pointer is None or attempt_pointer.get("reference") != dict(terminal_attempt_ref)
            or attempt.get("attempt_id") != attempt_id or attempt.get("status") != "success"
            or result_ref not in attempt.get("artifacts", ())):
        raise _integrity("Selected Optimization terminal attempt differs from its sealed completion.",
                         task_id=task_id, attempt_id=attempt_id)

    if selection is not None and (
            selection.get("attempt_id") != attempt_id
            or selection.get("task_id") != task_id
            or selection.get("request_sha256") != request_sha256
            or selection.get("result_ref") != dict(result_ref)
            or selection.get("workspace_completion") != dict(completion_ref)):
        raise _integrity("Request success selection differs from its completed Optimization.",
                         task_id=task_id, attempt_id=attempt_id)

    state_pointer = snapshot.read_pointer(f"attempt_state/{task_id}/{attempt_id}")
    if state_pointer is None or state_pointer["reference"].get("role") != "task_change":
        raise _integrity("Selected Optimization has no committed final task state.",
                         task_id=task_id, attempt_id=attempt_id)
    _, state = read(state_pointer["reference"], "task_change")
    if (state.get("kind") != "attempt_update" or state.get("attempt_id") != attempt_id
            or state.get("status") != "success"):
        raise _integrity("Selected Optimization attempt is not durably successful.",
                         task_id=task_id, attempt_id=attempt_id)

    if (attempt.get("attempt_id") != attempt_id
            or attempt.get("resume_from") is not None and not isinstance(attempt.get("resume_from"), Mapping)):
        raise _integrity("Selected Optimization attempt allocation is malformed.",
                         task_id=task_id, attempt_id=attempt_id)

    result_selection = select_result(
        snapshot, binding_identity, completion_ref, result_ref, attempt_sha256=attempt_sha256,
    )
    fixed_reader = FixedResultReader(binding_identity, result_selection)
    return {
        "task_id": task_id,
        "environment_sha256": descriptor["environment"]["environment_sha256"],
        "attempt_sha256": attempt_sha256,
        "result_sha256": result_ref["sha256"],
        "result_ref": dict(result_ref),
        "projection": {"terminal": terminal},
        "fixed_reader": fixed_reader,
        "selection": result_selection,
        "checkpoint_available": (
            snapshot.read_pointer(f"checkpoint/{task_id}/{attempt_id}") is not None
            or isinstance(attempt.get("resume_from"), Mapping)
        ),
        "attempt_id": attempt_id,
    }


class FixedResultReader:
    """Read one already-selected immutable result without retaining a DB handle."""

    def __init__(self, binding_identity: Mapping[str, Any], selection: ResultSelection):
        self._root = Path(binding_identity["root"])
        self._operations = Path(binding_identity["leaf"]) / "operations"
        self._plan_sha256 = binding_identity["plan_sha256"]
        self._workspace_instance_id = binding_identity["workspace_instance_id"]
        self.selection = selection

    def _store(self) -> OperationStore:
        return OperationStore(
            self._operations,
            plan_sha256=self._plan_sha256,
            workspace_instance_id=self._workspace_instance_id,
        )

    def _canonical(self, snapshot, reference, role, schema=None):
        if reference.get("role") != role:
            from .validation.common import _integrity
            raise _integrity("Sealed result reference has the wrong role.", expected=role,
                             actual=reference.get("role"))
        raw = snapshot.get_object(reference)
        value = record_document(raw)
        if (not isinstance(value, dict) or record_bytes(value) != raw
                or (schema is not None and value.get("schema") != schema)):
            from .validation.common import _integrity
            raise _integrity("Sealed result object is not canonical or has the wrong schema.",
                             reference=dict(reference), expected_schema=schema)
        return value

    def _selection_snapshot(self, snapshot):
        selection = self.selection
        completion = self._canonical(snapshot, selection.completion_ref,
                                     "workspace_completion", "scnsim.workspace_completion")
        if (selection.completion_ref.get("sha256") != selection.completion_sha256
                or completion.get("schema_version") != 1
                or completion.get("workspace_instance_id") != selection.workspace_instance_id
                or completion.get("plan_sha256") != selection.plan_sha256
                or completion.get("request_sha256") != selection.request_sha256
                or completion.get("task_id") != selection.task_id
                or completion.get("attempt_id") != selection.attempt_id
                or completion.get("terminal_attempt_ref") != dict(selection.terminal_attempt_ref)
                or completion.get("result_ref") != dict(selection.result_ref)
                or completion.get("index_root_ref") != dict(selection.index_root_ref)
                or completion.get("generation_ancestry_ref") != (
                    None if selection.generation_ancestry_ref is None
                    else dict(selection.generation_ancestry_ref))
                or completion.get("generation_count") != selection.generation_count
                or completion.get("candidate_count") != selection.candidate_count):
            from .validation.common import _integrity
            raise _integrity("Result selection differs from its sealed completion.",
                             task_id=selection.task_id, attempt_id=selection.attempt_id)
        attempt_pointer = snapshot.read_pointer(
            f"attempt/{selection.task_id}/{selection.attempt_id}"
        )
        if (selection.terminal_attempt_ref.get("sha256") != selection.attempt_sha256
                or attempt_pointer is None
                or attempt_pointer.get("reference") != dict(selection.terminal_attempt_ref)):
            from .validation.common import _integrity
            raise _integrity("Selected terminal attempt reference changed.",
                             task_id=selection.task_id, attempt_id=selection.attempt_id)
        index_root = self._canonical(snapshot, selection.index_root_ref,
                                     "workspace_completion_index_root",
                                     "scnsim.workspace_completion_index_root")
        if (selection.index_root_ref.get("sha256") != selection.index_root_sha256
                or index_root.get("schema_version") != 1
                or index_root.get("workspace_instance_id") != selection.workspace_instance_id
                or index_root.get("plan_sha256") != selection.plan_sha256
                or index_root.get("request_sha256") != selection.request_sha256
                or index_root.get("task_id") != selection.task_id
                or index_root.get("attempt_id") != selection.attempt_id
                or index_root.get("result_ref") != dict(selection.result_ref)
                or index_root.get("baseline_ref") != completion.get("baseline_ref")
                or index_root.get("generation_ancestry_ref") != completion.get("generation_ancestry_ref")
                or index_root.get("generation_count") != selection.generation_count
                or index_root.get("candidate_count") != selection.candidate_count
                or index_root.get("best_ordinal") != completion.get("best_ordinal")):
            from .validation.common import _integrity
            raise _integrity("Result index root differs from its sealed completion.",
                             task_id=selection.task_id, attempt_id=selection.attempt_id)
        return completion, index_root

    def read_generation(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self.selection.generation_count
        if index < 0 or index >= self.selection.generation_count:
            raise IndexError(index)
        generation = index + 1
        with self._store().reader() as snapshot:
            if snapshot is None:
                from .validation.common import _integrity
                raise _integrity("Selected Workspace database disappeared.")
            _, index_root = self._selection_snapshot(snapshot)
            sealed = next((row for row in index_root.get("generations", ())
                           if row.get("generation") == generation), None)
            indexed = snapshot.read_generation_index(self.selection.completion_sha256, generation)
            if (sealed is None or indexed is None
                    or any(indexed.get(key) != sealed.get(key) for key in
                           ("generation", "block_ref", "first_ordinal", "row_count", "summary_json"))):
                from .validation.common import _integrity
                raise _integrity("Generation locator differs from its sealed index.", generation=generation)
            candidate_index = self._canonical(
                snapshot, sealed["candidate_index_ref"], "workspace_candidate_index",
                "scnsim.workspace_candidate_index",
            )
            block = self._canonical(snapshot, sealed["block_ref"], "benchmark_generation_evidence",
                                    "scnsim.benchmark_generation_evidence")
            if (candidate_index.get("workspace_instance_id") != self.selection.workspace_instance_id
                    or candidate_index.get("plan_sha256") != self.selection.plan_sha256
                    or candidate_index.get("request_sha256") != self.selection.request_sha256
                    or candidate_index.get("task_id") != self.selection.task_id
                    or candidate_index.get("generation") != generation
                    or candidate_index.get("block_ref") != sealed.get("block_ref")
                    or candidate_index.get("row_count") != sealed.get("row_count")
                    or block.get("task_id") != self.selection.task_id
                    or block.get("attempt_id") != candidate_index.get("attempt_id")
                    or block.get("attributes", {}).get("generation") != generation
                    or len(block.get("rows", ())) != sealed.get("row_count")):
                from .validation.common import _integrity
                raise _integrity("Generation evidence differs from its sealed locator.", generation=generation)
            return block

    def read_candidate(self, evaluation_ordinal: int) -> dict[str, Any]:
        if evaluation_ordinal <= 0 or evaluation_ordinal > self.selection.candidate_count:
            raise IndexError(evaluation_ordinal)
        with self._store().reader() as snapshot:
            if snapshot is None:
                from .validation.common import _integrity
                raise _integrity("Selected Workspace database disappeared.")
            _, index_root = self._selection_snapshot(snapshot)
            indexed = snapshot.read_candidate_index(self.selection.completion_sha256, evaluation_ordinal)
            generation = indexed.get("generation") if indexed is not None else None
            generation_row = next((row for row in index_root.get("generations", ())
                                   if row.get("generation") == generation), None)
            candidate_block = None
            if generation_row is not None:
                candidate_block = self._canonical(snapshot, generation_row["candidate_index_ref"],
                                                  "workspace_candidate_index",
                                                  "scnsim.workspace_candidate_index")
            sealed = None if candidate_block is None else next(
                (row for row in candidate_block.get("candidates", ())
                 if row.get("ordinal") == evaluation_ordinal), None)
            if indexed is None or sealed is None or indexed != sealed:
                from .validation.common import _integrity
                raise _integrity("Candidate locator differs from its sealed index.",
                                 evaluation_ordinal=evaluation_ordinal)
            block = self._canonical(snapshot, sealed["block_ref"], "benchmark_generation_evidence",
                                    "scnsim.benchmark_generation_evidence")
            if (candidate_block.get("workspace_instance_id") != self.selection.workspace_instance_id
                    or candidate_block.get("plan_sha256") != self.selection.plan_sha256
                    or candidate_block.get("request_sha256") != self.selection.request_sha256
                    or candidate_block.get("task_id") != self.selection.task_id
                    or candidate_block.get("generation") != sealed.get("generation")
                    or candidate_block.get("block_ref") != sealed.get("block_ref")
                    or candidate_block.get("attempt_id") != block.get("attempt_id")
                    or block.get("task_id") != self.selection.task_id
                    or block.get("attributes", {}).get("generation") != sealed.get("generation")):
                from .validation.common import _integrity
                raise _integrity("Candidate generation reference differs from its selected task.",
                                 evaluation_ordinal=evaluation_ordinal)
            offset = sealed["row_offset"]
            rows = block.get("rows")
            if not isinstance(offset, int) or not isinstance(rows, list) or offset < 0 or offset >= len(rows):
                from .validation.common import _integrity
                raise _integrity("Candidate row locator is outside its generation block.",
                                 evaluation_ordinal=evaluation_ordinal)
            occurrence = rows[offset]
            if occurrence.get("value") != sealed.get("value_ref"):
                from .validation.common import _integrity
                raise _integrity("Candidate value reference differs from its generation block.",
                                 evaluation_ordinal=evaluation_ordinal)
            raw = snapshot.get_object(sealed["value_ref"])
            value = record_document(raw)
            if not isinstance(value, dict) or record_bytes(value) != raw:
                from .validation.common import _integrity
                raise _integrity("Candidate value body is not canonical.",
                                 evaluation_ordinal=evaluation_ordinal)
            value.update(occurrence.get("occurrence", {}))
            return value

    def read_candidate_discretization(self, flat_population_index: int) -> dict[str, Any]:
        return self.read_candidate(flat_population_index + 1)["discretization"]

    def project(self, kind: str, selector=None):
        if kind not in {"history", "objective", "residual", "parameter", "table", "comparison"}:
            raise ValueError(f"unsupported optimization projection: {kind}")
        with self._store().reader() as snapshot:
            if snapshot is None:
                from .validation.common import _integrity
                raise _integrity("Selected Workspace database disappeared.")
            _, index_root = self._selection_snapshot(snapshot)
            if kind == "comparison":
                baseline_block = self._canonical(snapshot, self.selection.baseline_locator["block_ref"],
                                                 "benchmark_baseline_evidence",
                                                 "scnsim.benchmark_baseline_evidence")
                baseline = self._canonical(snapshot, baseline_block["baseline"], "benchmark_value")
                baseline.update(baseline_block.get("baseline_occurrence", {}))
                best = baseline if self.selection.best_locator.get("generation", 0) == 0 else None
                if best is None:
                    best_ordinal = self.selection.best_locator["ordinal"]
                    indexed = snapshot.read_candidate_index(self.selection.completion_sha256, best_ordinal)
                    generation_row = next((row for row in index_root.get("generations", ())
                                           if row.get("generation") == indexed.get("generation")), None)
                    candidate_block = self._canonical(snapshot, generation_row["candidate_index_ref"],
                                                      "workspace_candidate_index",
                                                      "scnsim.workspace_candidate_index")
                    sealed = next((row for row in candidate_block.get("candidates", ())
                                   if row.get("ordinal") == best_ordinal), None)
                    if sealed is None or sealed != indexed:
                        from .validation.common import _integrity
                        raise _integrity("Best-candidate locator differs from its sealed index.")
                    gen_block = self._canonical(snapshot, sealed["block_ref"],
                                                "benchmark_generation_evidence",
                                                "scnsim.benchmark_generation_evidence")
                    row = gen_block["rows"][sealed["row_offset"]]
                    if row.get("value") != sealed.get("value_ref"):
                        from .validation.common import _integrity
                        raise _integrity("Best-candidate body reference differs from its evidence row.")
                    raw = snapshot.get_object(sealed["value_ref"])
                    best = record_document(raw)
                    if record_bytes(best) != raw:
                        from .validation.common import _integrity
                        raise _integrity("Best-candidate body is not canonical.")
                    best.update(row.get("occurrence", {}))
                request = self._canonical(snapshot, index_root["request_ref"], "operation_request")
                return {"baseline": baseline, "best": best,
                        "declarations": request.get("spec", {})}
            indexed_rows = snapshot.read_candidate_indexes(self.selection.completion_sha256)
            sealed_by_generation = {}
            for generation_row in index_root.get("generations", ()):
                candidate_block = self._canonical(snapshot, generation_row["candidate_index_ref"],
                                                  "workspace_candidate_index",
                                                  "scnsim.workspace_candidate_index")
                if (candidate_block.get("generation") != generation_row.get("generation")
                        or candidate_block.get("block_ref") != generation_row.get("block_ref")
                        or candidate_block.get("row_count") != generation_row.get("row_count")):
                    from .validation.common import _integrity
                    raise _integrity("Candidate index block differs from the sealed generation root.",
                                     generation=generation_row.get("generation"))
                sealed_by_generation[generation_row["generation"]] = candidate_block.get("candidates", ())
            if len(indexed_rows) != self.selection.candidate_count:
                from .validation.common import _integrity
                raise _integrity("SQL candidate index count differs from its completion.")
            rows = []
            for indexed in indexed_rows:
                generation_candidates = sealed_by_generation.get(indexed["generation"], ())
                sealed = next((row for row in generation_candidates
                               if row.get("ordinal") == indexed["ordinal"]), None)
                if sealed != indexed:
                    from .validation.common import _integrity
                    raise _integrity("SQL candidate summary differs from the sealed candidate block.",
                                     evaluation_ordinal=indexed["ordinal"])
                rows.append(indexed["summary_json"])
        # The index was authenticated against its immutable root before these
        # compact rows were detached. Result-layer code owns unit interpretation.
        if kind == "history":
            return rows
        if kind == "parameter":
            return rows
        if kind in {"objective", "residual", "table"}:
            selected = []
            for row in rows:
                components = row.get("components", ())
                if selector is not None:
                    components = tuple(item for item in components
                                       if item.get("objective_id") == selector)
                selected.append({**row, "components": list(components)})
            return selected
        raise AssertionError(kind)
