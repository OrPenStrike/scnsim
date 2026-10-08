"""Durable, process-local operation trace spans.

The numerical request and workspace journal own result identity.  This module
records only operation timing and references to that authority; it never copies
scientific values into the timeline.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter_ns
from typing import Iterator, Mapping, Sequence
from uuid import uuid4

from ..errors import EvidenceIntegrityError
from .prepared import record_bytes


_PROCESS_CLOCK_ID: str | None = None
_PROCESS_CLOCK_ORIGIN_NS: int | None = None


def _clock_binding(start_tick_ns: int | None) -> dict[str, object]:
    """Return the one truthful perf-counter domain shared by this process."""
    global _PROCESS_CLOCK_ID, _PROCESS_CLOCK_ORIGIN_NS
    if _PROCESS_CLOCK_ID is None:
        _PROCESS_CLOCK_ID = str(uuid4())
        _PROCESS_CLOCK_ORIGIN_NS = (
            perf_counter_ns() if start_tick_ns is None else start_tick_ns
        )
    assert _PROCESS_CLOCK_ORIGIN_NS is not None
    return {
        "id": _PROCESS_CLOCK_ID,
        "source": "time.perf_counter_ns",
        "unit": "nanoseconds",
        "monotonic_origin_ns": _PROCESS_CLOCK_ORIGIN_NS,
    }


class OperationRecorder:
    """Collect and persist one public solve/evaluate/optimize invocation.

    Span intervals are inclusive and share a process-local clock.  A supplied
    ``start_tick_ns`` lets the public method start its root before lazy imports,
    backend selection, or request preparation.
    """

    def __init__(
        self,
        binding: object,
        *,
        kind: str,
        backend: str | None,
        precision: str | None,
        start_tick_ns: int | None = None,
    ) -> None:
        self.binding = binding
        self.operation_id = str(uuid4())
        self.root_span_id = str(uuid4())
        self.method = kind
        self.backend = backend
        self.precision = precision
        self.clock = _clock_binding(start_tick_ns)
        self._origin_ns = int(self.clock["monotonic_origin_ns"])
        self._start_tick_ns = perf_counter_ns() if start_tick_ns is None else start_tick_ns
        self._stack: list[str] = [self.root_span_id]
        self._spans: list[dict[str, object]] = []
        self._persisted_spans = 0
        self._handed_spans = 0
        self._entered = False
        self._closed = False
        self._request_sha256: str | None = None
        self._task_id: str | None = None
        self._attempt_id: str | None = None
        self._environment_sha256: str | None = None
        self._operation: str | None = None
        self._numerical_refs: list[dict[str, object]] = []
        self._details: dict[str, object] = {}
        self._end_tick_ns: int | None = None
        self._status = "running"

    def __enter__(self) -> OperationRecorder:
        if self._entered:
            raise RuntimeError("operation recorder cannot be entered twice")
        self._entered = True
        from .storage import initialize_operation_record, start_operation

        with self.binding.writer():  # type: ignore[attr-defined]
            initialize_operation_record(self.binding, clock_binding=self.clock)
            start_operation(self.binding, self.root_row())
        return self

    def __exit__(self, error_type: object, error: BaseException | None, traceback: object) -> bool:
        del error_type, traceback
        if self._closed:
            return False
        self._closed = True
        status = "success" if error is None else (
            "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failure"
        )
        from .storage import finish_operation, record_operation_event

        try:
            finalization_start = perf_counter_ns()
            pending = self.pending_spans()
            with self.binding.writer():  # type: ignore[attr-defined]
                archive_error: BaseException | None = None
                if pending:
                    try:
                        record_operation_event(
                            self.binding,
                            {
                                "event": "spans_archived",
                                "operation_id": self.operation_id,
                                "row": self.root_row(),
                                "spans": [dict(row) for row in pending],
                            },
                        )
                        self.mark_spans_persisted(pending)
                    except BaseException as caught:
                        archive_error = caught
                        if error is not None:
                            add_note = getattr(error, "add_note", None)
                            if callable(add_note):
                                add_note(
                                    "SCNSim operation span archive also failed: "
                                    f"{type(caught).__name__}: {caught}"
                                )
                finalization_end = perf_counter_ns()
                if archive_error is not None and error is None:
                    raise archive_error
                finalization_span = self._span_row(
                    span_id=str(uuid4()),
                    parent_span_id=self.root_span_id,
                    kind="operation_finalization",
                    start_tick_ns=finalization_start,
                    end_tick_ns=finalization_end,
                    status="failure" if archive_error is not None else "success",
                    details={"pending_span_count": len(pending)},
                )
                self._spans.append(finalization_span)
                self._finish_root(finalization_end, status)
                finish_operation(
                    self.binding,
                    self.root_row(),
                    failure=None if error is None else self._error_record(error),
                    spans=(finalization_span,),
                )
        except BaseException as finalization_error:
            if error is None:
                raise
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(
                    "SCNSim operation trace finalization also failed: "
                    f"{type(finalization_error).__name__}: {finalization_error}"
                )
        return False

    def bind(
        self,
        *,
        operation: str,
        request_sha256: str,
        task_id: str | None,
        environment_sha256: str | None,
        attempt_id: str | None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Bind the request and any task/environment identities known so far."""
        if self._closed:
            raise RuntimeError("closed operation recorder cannot be bound")
        self._operation = operation
        self._request_sha256 = request_sha256
        self._task_id = task_id
        self._environment_sha256 = environment_sha256
        self._attempt_id = attempt_id
        self._details = dict(details or {})
        from .storage import bind_operation

        with self.binding.writer():  # type: ignore[attr-defined]
            bind_operation(self.binding, row=self.root_row())

    def set_attempt(self, attempt_id: str | None) -> None:
        """Record the actual attempt allocated after a request cache lookup."""
        self._attempt_id = attempt_id
        if attempt_id is None or self._task_id is None:
            return
        from .storage import bind_operation_attempt

        with self.binding.writer():  # type: ignore[attr-defined]
            bind_operation_attempt(
                self.binding,
                row=self.root_row(),
            )

    def add_numerical_ref(self, reference: Mapping[str, object]) -> None:
        self._numerical_refs.append(dict(reference))

    def publish_pending_spans(self, writer: object) -> tuple[dict[str, object], ...]:
        """Stage unhanded spans for a writer without claiming they are durable."""
        rows = tuple(dict(row) for row in self._spans[self._handed_spans :])
        appended: list[dict[str, object]] = []
        for row in rows:
            writer.append_event(  # type: ignore[attr-defined]
                kind="operation_span",
                payload={"operation_id": self.operation_id, "span": row},
            )
            self._handed_spans += 1
            appended.append(row)
        return tuple(appended)

    def handed_pending_spans(self) -> tuple[dict[str, object], ...]:
        """Return spans staged for a writer that have not received commit ACK."""
        return tuple(
            dict(row)
            for row in self._spans[self._persisted_spans : self._handed_spans]
        )

    @property
    def root_start_tick_ns(self) -> int:
        return self._start_tick_ns

    @property
    def clock_binding(self) -> dict[str, object]:
        return dict(self.clock)

    @contextmanager
    def span(
        self,
        kind: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        if self._closed:
            raise RuntimeError("closed operation recorder cannot add a span")
        span_id = str(uuid4())
        parent_span_id = self._stack[-1] if self._stack else None
        start_tick = perf_counter_ns()
        self._stack.append(span_id)
        error: BaseException | None = None
        try:
            yield span_id
        except BaseException as caught:
            error = caught
            raise
        finally:
            if self._stack and self._stack[-1] == span_id:
                self._stack.pop()
            end_tick = perf_counter_ns()
            self._spans.append(
                self._span_row(
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    kind=kind,
                    start_tick_ns=start_tick,
                    end_tick_ns=end_tick,
                    status=(
                        "success" if error is None else
                        "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failure"
                    ),
                    details=details,
                )
            )

    def measure(
        self,
        kind: str,
        *,
        start_tick_ns: int,
        end_tick_ns: int,
        parent_span_id: str | None = None,
        details: Mapping[str, object] | None = None,
        status: str = "success",
    ) -> None:
        """Record an observed inclusive interval supplied by a numerical owner."""
        self._spans.append(
            self._span_row(
                span_id=str(uuid4()),
                parent_span_id=parent_span_id or (self._stack[-1] if self._stack else None),
                kind=kind,
                start_tick_ns=start_tick_ns,
                end_tick_ns=end_tick_ns,
                status=status,
                details=details,
            )
        )

    def pending_spans(self) -> tuple[dict[str, object], ...]:
        """Return unpersisted rows without changing the local acknowledgement."""
        return tuple(dict(row) for row in self._spans[self._persisted_spans :])

    def mark_spans_persisted(self, rows: tuple[Mapping[str, object], ...]) -> None:
        pending = self._spans[self._persisted_spans :]
        if [record_bytes(dict(row)) for row in pending[: len(rows)]] != [
            record_bytes(dict(row)) for row in rows
        ]:
            raise RuntimeError("operation span persistence cursor no longer matches its committed prefix")
        self._persisted_spans += len(rows)
        self._handed_spans = max(self._handed_spans, self._persisted_spans)

    def root_row(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "method": self.method,
            "operation": self._operation,
            "backend": self.backend,
            "precision": self.precision,
            "request_sha256": self._request_sha256,
            "task_id": self._task_id,
            "attempt_id": self._attempt_id,
            "environment_sha256": self._environment_sha256,
            "status": self._status,
            "span_id": self.root_span_id,
            "parent_span_id": None,
            "kind": "operation",
            "clock": dict(self.clock),
            "start_ns": self._start_tick_ns - self._origin_ns,
            "end_ns": (
                None if self._end_tick_ns is None else self._end_tick_ns - self._origin_ns
            ),
            "details": dict(self._details),
            "numerical_refs": [dict(item) for item in self._numerical_refs],
        }

    def _finish_root(self, end_tick_ns: int, status: str) -> None:
        self._end_tick_ns = end_tick_ns
        self._status = status

    def finished_root_row(self) -> dict[str, object]:
        return self.root_row()

    def _span_row(
        self,
        *,
        span_id: str,
        parent_span_id: str | None,
        kind: str,
        start_tick_ns: int,
        end_tick_ns: int,
        status: str,
        details: Mapping[str, object] | None,
    ) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "kind": kind,
            "clock": dict(self.clock),
            "start_ns": start_tick_ns - self._origin_ns,
            "end_ns": end_tick_ns - self._origin_ns,
            "status": status,
            "details": dict(details or {}),
            "numerical_refs": [],
        }

    @staticmethod
    def _error_record(error: BaseException) -> dict[str, object]:
        record: dict[str, object] = {
            "type": type(error).__name__,
            "module": type(error).__module__,
            "message": str(error),
        }
        for field in ("kind", "category", "stage", "evidence"):
            value = getattr(error, field, None)
            if value is not None:
                record[field] = value
        return record


def _operation_event_identity(event: Mapping[str, object]) -> str | None:
    operation_id = event.get("operation_id")
    if isinstance(operation_id, str):
        return operation_id
    row = event.get("row")
    if isinstance(row, Mapping) and isinstance(row.get("operation_id"), str):
        return str(row["operation_id"])
    return None


def _count(values: Sequence[Mapping[str, object]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        item = value.get(key)
        if isinstance(item, str):
            result[item] = result.get(item, 0) + 1
    return dict(sorted(result.items()))


def _matching_operation(
    row: Mapping[str, object],
    *,
    selected_ids: set[str] | None,
    method: str | None,
    backend: str | None,
    precision: str | None,
    status: str | None,
) -> bool:
    return (
        (selected_ids is None or row.get("operation_id") in selected_ids)
        and (method is None or row.get("method") == method)
        and (backend is None or row.get("backend") == backend)
        and (precision is None or row.get("precision") == precision)
        and (status is None or row.get("status") == status)
    )


def _selected_operation_ids(operations: Sequence[str] | str | None) -> set[str] | None:
    if operations is None:
        return None
    if isinstance(operations, str):
        return {operations}
    return set(operations)


def project_operation_record(
    source: Mapping[str, object] | None,
    *,
    workspace: Path,
    plan_sha256: object,
    workspace_instance_id: object,
    storage_origin: Mapping[str, object] | None = None,
    operations: Sequence[str] | str | None = None,
    method: str | None = None,
    backend: str | None = None,
    precision: str | None = None,
    status: str | None = None,
):
    """Project one verified operation journal without reading or mutating it."""
    if source is None:
        roots: list[dict[str, object]] = []
        raw_spans: list[dict[str, object]] = []
        raw_events: list[dict[str, object]] = []
        tasks: list[dict[str, object]] = []
    else:
        roots_by_id: dict[str, dict[str, object]] = {}
        root_order: list[str] = []
        raw_events = [dict(value) for value in source.get("operation_events", ())
                      if isinstance(value, Mapping)]
        for event in raw_events:
            kind = event.get("event")
            operation_id = _operation_event_identity(event)
            if not isinstance(operation_id, str):
                continue
            if kind == "started":
                row = event.get("row")
                if isinstance(row, Mapping):
                    if operation_id not in roots_by_id:
                        root_order.append(operation_id)
                    roots_by_id[operation_id] = dict(row)
            elif kind == "finished":
                row = event.get("row")
                if isinstance(row, Mapping):
                    if operation_id not in roots_by_id:
                        root_order.append(operation_id)
                    finished_row = dict(row)
                    saved_failure = event.get("failure")
                    if saved_failure is not None:
                        details = (
                            dict(finished_row.get("details", {}))
                            if isinstance(finished_row.get("details"), Mapping)
                            else {}
                        )
                        details["failure"] = saved_failure
                        finished_row["details"] = details
                    roots_by_id[operation_id] = finished_row
            elif kind in {"bound", "attempt_bound", "cache_hit"}:
                row = roots_by_id.get(operation_id)
                if row is None:
                    continue
                for key in (
                    "operation", "request_sha256", "task_id", "environment_sha256", "attempt_id",
                ):
                    if key in event:
                        row[key] = event[key]
                if kind == "cache_hit":
                    details = dict(row.get("details", {})) if isinstance(row.get("details"), Mapping) else {}
                    details.update({
                        key: event[key]
                        for key in ("checkpoint_policy", "checkpoint_available")
                        if key in event
                    })
                    row["details"] = details

        roots = [roots_by_id[item] for item in root_order]
        raw_spans = []
        task_value = source.get("tasks", ())
        tasks = [dict(item) for item in task_value if isinstance(item, Mapping)] if isinstance(task_value, Sequence) else []
        tasks_by_id = {
            str(task["task_id"]): task
            for task in tasks if isinstance(task.get("task_id"), str)
        }
        selected_attempt_by_request = {
            str(event["request_sha256"]): str(event["attempt_id"])
            for event in raw_events
            if event.get("event") == "request_success_selected"
            and isinstance(event.get("request_sha256"), str)
            and isinstance(event.get("attempt_id"), str)
        }
        for row in roots:
            if row.get("operation") != "optimize_direct":
                continue
            task_id = row.get("task_id")
            task = tasks_by_id.get(task_id) if isinstance(task_id, str) else None
            attempts = None if task is None else task.get("attempts")
            if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
                continue
            selected_attempt_id = row.get("attempt_id")
            if selected_attempt_id is None:
                request_sha256 = row.get("request_sha256")
                selected_attempt_id = selected_attempt_by_request.get(str(request_sha256))
            attempt = next((
                candidate for candidate in attempts
                if isinstance(candidate, Mapping)
                and candidate.get("attempt_id") == selected_attempt_id
            ), None)
            if attempt is None:
                continue
            checkpoint = attempt.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                checkpoint = attempt.get("resume_from")
            if not isinstance(checkpoint, Mapping):
                continue
            details = dict(row.get("details", {})) if isinstance(row.get("details"), Mapping) else {}
            details["checkpoint"] = dict(checkpoint)
            details["checkpoint_attempt_id"] = selected_attempt_id
            row["details"] = details
        for event in raw_events:
            operation_id = _operation_event_identity(event)
            span_values = event.get("spans")
            if isinstance(span_values, Sequence) and not isinstance(span_values, (str, bytes)):
                for span in span_values:
                    if isinstance(span, Mapping) and isinstance(operation_id, str):
                        raw_spans.append(dict(span, operation_id=operation_id))
        for task in tasks:
            task_events = task.get("events", ())
            if not isinstance(task_events, Sequence) or isinstance(task_events, (str, bytes)):
                continue
            for event in task_events:
                if not isinstance(event, Mapping) or event.get("kind") != "operation_span":
                    continue
                payload = event.get("payload")
                span = payload.get("span") if isinstance(payload, Mapping) else None
                operation_id = payload.get("operation_id") if isinstance(payload, Mapping) else None
                if isinstance(span, Mapping) and isinstance(operation_id, str):
                    raw_spans.append(dict(span, operation_id=operation_id))

    selected_ids = _selected_operation_ids(operations)
    selected_roots = [
        row for row in roots
        if _matching_operation(
            row,
            selected_ids=selected_ids,
            method=method,
            backend=backend,
            precision=precision,
            status=status,
        )
    ]
    selected_operation_ids = {
        str(row["operation_id"]) for row in selected_roots
        if isinstance(row.get("operation_id"), str)
    }
    selected_spans = [
        row for row in raw_spans
        if row.get("operation_id") in selected_operation_ids
    ]
    if storage_origin is None and source is not None:
        storage_origin = {
            "kind": "file_journal",
            "version": source.get("schema_version"),
            "locator": "benchmark.json",
        }
    if storage_origin is not None:
        origin = dict(storage_origin)
        for row in selected_roots:
            row["storage_origin"] = dict(origin)
        for row in selected_spans:
            row["storage_origin"] = dict(origin)
    selected_history = []
    for index, event in enumerate(raw_events):
        operation_id = _operation_event_identity(event)
        if selected_ids is not None or any(value is not None for value in (method, backend, precision, status)):
            if operation_id not in selected_operation_ids:
                continue
        selected_history.append({
            "event_index": index,
            "kind": str(event.get("event", "operation_event")),
            "operation_id": operation_id,
        })

    numerical_refs = [
        {"operation_id": row.get("operation_id"), "reference": dict(reference)}
        for row in selected_roots
        for reference in row.get("numerical_refs", ())
        if isinstance(reference, Mapping)
    ]
    clocks: dict[str, dict[str, object]] = {}
    for row in [*selected_roots, *selected_spans]:
        clock = row.get("clock")
        if isinstance(clock, Mapping) and isinstance(clock.get("id"), str):
            clocks[str(clock["id"])] = dict(clock)
    document: dict[str, object] = {
        "schema": "scnsim.operation_benchmark",
        "schema_version": 1,
        "plan_sha256": plan_sha256,
        "workspace_instance_id": workspace_instance_id,
        "operations": selected_roots,
        "spans": selected_spans,
        "counts": {
            "operations": len(selected_roots),
            "spans": len(selected_spans),
            "operation_status": _count(selected_roots, "status"),
            "span_kind": _count(selected_spans, "kind"),
            "historical_kind": _count(selected_history, "kind"),
        },
        "clock_domains": [clocks[key] for key in sorted(clocks)],
        "numerical_refs": numerical_refs,
        "historical_records": selected_history,
    }
    from .models import BenchmarkResult

    return BenchmarkResult.from_document(workspace, document)


def project_indexed_operation_rows(
    rows: Mapping[str, object] | None,
    *,
    workspace: Path,
    plan_sha256: object,
    workspace_instance_id: object,
    operations: Sequence[str] | str | None = None,
    method: str | None = None,
    backend: str | None = None,
    precision: str | None = None,
    status: str | None = None,
):
    """Project selected decoded operation and span rows from the SQL index."""
    selected_ids = _selected_operation_ids(operations)
    if rows is None:
        source_operations = source_spans = ()
    else:
        source_operations = rows["operations"]
        source_spans = rows["spans"]
    selected_roots = [
        dict(row)
        for row in source_operations  # type: ignore[union-attr]
        if _matching_operation(
            row,
            selected_ids=selected_ids,
            method=method,
            backend=backend,
            precision=precision,
            status=status,
        )
    ]
    selected_operation_ids = {
        str(row["operation_id"])
        for row in selected_roots
        if isinstance(row.get("operation_id"), str)
    }
    selected_spans = [
        dict(row)
        for row in source_spans  # type: ignore[union-attr]
        if row.get("operation_id") in selected_operation_ids  # type: ignore[union-attr]
    ]
    origin = {"kind": "sqlite", "version": 3, "locator": "operations.sqlite3"}
    for row in selected_roots:
        row["storage_origin"] = dict(origin)
    for row in selected_spans:
        row["storage_origin"] = dict(origin)

    clocks: dict[str, dict[str, object]] = {}
    for row in [*selected_roots, *selected_spans]:
        clock = row.get("clock")
        if isinstance(clock, Mapping) and isinstance(clock.get("id"), str):
            clocks[str(clock["id"])] = dict(clock)
    numerical_refs = [
        {"operation_id": row.get("operation_id"), "reference": dict(reference)}
        for row in selected_roots
        for reference in row.get("numerical_refs", ())
        if isinstance(reference, Mapping)
    ]
    document: dict[str, object] = {
        "schema": "scnsim.operation_benchmark",
        "schema_version": 1,
        "plan_sha256": plan_sha256,
        "workspace_instance_id": workspace_instance_id,
        "operations": selected_roots,
        "spans": selected_spans,
        "counts": {
            "operations": len(selected_roots),
            "spans": len(selected_spans),
            "operation_status": _count(selected_roots, "status"),
            "span_kind": _count(selected_spans, "kind"),
            "historical_kind": {},
        },
        "clock_domains": [clocks[key] for key in sorted(clocks)],
        "numerical_refs": numerical_refs,
        "historical_records": [],
    }
    from .models import BenchmarkResult

    return BenchmarkResult.from_document(workspace, document)


def read_operations(
    binding: object,
    *,
    operations: Sequence[str] | str | None = None,
    method: str | None = None,
    backend: str | None = None,
    precision: str | None = None,
    status: str | None = None,
):
    """Read file-journal history and indexed SQLite operation/span projections."""
    from . import storage

    with binding.reader():  # type: ignore[attr-defined]
        legacy_record = storage.open_legacy_operation_record(binding)
        operation_ids = (
            None if operations is None else (operations,) if isinstance(operations, str) else tuple(operations)
        )
        indexed_rows = storage.query_operation_rows(
            binding,
            operation_ids=operation_ids,
            method=method,
            backend=backend,
            precision=precision,
            status=status,
        )

    workspace = storage.operation_workspace(binding)
    plan_sha256 = getattr(binding, "plan_sha256")
    workspace_instance_id = getattr(binding, "workspace_instance_id")
    legacy_projection = project_operation_record(
        None if legacy_record is None else legacy_record.document(),
        workspace=workspace,
        plan_sha256=plan_sha256,
        workspace_instance_id=workspace_instance_id,
        operations=operations,
        method=method,
        backend=backend,
        precision=precision,
        status=status,
    ).document()
    indexed_projection = project_indexed_operation_rows(
        indexed_rows,
        workspace=workspace,
        plan_sha256=plan_sha256,
        workspace_instance_id=workspace_instance_id,
        operations=operations,
        method=method,
        backend=backend,
        precision=precision,
        status=status,
    ).document()

    legacy_operations = legacy_projection["operations"]
    indexed_operations = indexed_projection["operations"]
    all_operations = [*legacy_operations, *indexed_operations]  # type: ignore[misc]
    seen_operation_ids: dict[str, Mapping[str, object]] = {}
    for row in all_operations:
        if not isinstance(row, Mapping) or not isinstance(row.get("operation_id"), str):
            continue
        operation_id = str(row["operation_id"])
        previous = seen_operation_ids.get(operation_id)
        if previous is not None:
            raise EvidenceIntegrityError(
                "An operation identifier is duplicated across storage projections.",
                stage="benchmark_record",
                evidence={
                    "operation_id": operation_id,
                    "storage_origins": [
                        dict(previous.get("storage_origin", {}))
                        if isinstance(previous.get("storage_origin"), Mapping)
                        else None,
                        dict(row.get("storage_origin", {}))
                        if isinstance(row.get("storage_origin"), Mapping)
                        else None,
                    ],
                },
            )
        seen_operation_ids[operation_id] = row

    all_spans = [
        *legacy_projection["spans"],
        *indexed_projection["spans"],
    ]
    clocks: dict[str, dict[str, object]] = {}
    for row in [*all_operations, *all_spans]:
        if not isinstance(row, Mapping):
            continue
        clock = row.get("clock")
        if isinstance(clock, Mapping) and isinstance(clock.get("id"), str):
            clocks[str(clock["id"])] = dict(clock)
    historical = legacy_projection["historical_records"]
    numerical_refs = [
        {"operation_id": row.get("operation_id"), "reference": dict(reference)}
        for row in all_operations
        if isinstance(row, Mapping)
        for reference in row.get("numerical_refs", ())
        if isinstance(reference, Mapping)
    ]
    document: dict[str, object] = {
        "schema": "scnsim.operation_benchmark",
        "schema_version": 1,
        "plan_sha256": plan_sha256,
        "workspace_instance_id": workspace_instance_id,
        "operations": all_operations,
        "spans": all_spans,
        "counts": {
            "operations": len(all_operations),
            "spans": len(all_spans),
            "operation_status": _count(all_operations, "status"),  # type: ignore[arg-type]
            "span_kind": _count(all_spans, "kind"),  # type: ignore[arg-type]
            "historical_kind": _count(historical, "kind"),
        },
        "clock_domains": [clocks[key] for key in sorted(clocks)],
        "numerical_refs": numerical_refs,
        "historical_records": historical,
    }
    from .models import BenchmarkResult

    return BenchmarkResult.from_document(workspace, document)
