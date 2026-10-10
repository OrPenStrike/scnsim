"""Bounded aggregate and opt-in detailed operation timing diagnostics.

The numerical request and workspace journal own result identity.  This module
records only operation timing and references to that authority; it never copies
scientific values into the timeline.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from threading import Lock, RLock, local
from time import perf_counter_ns
from typing import Iterator, Mapping, Sequence
from uuid import uuid4
import warnings

from ..errors import (
    EvidenceIntegrityError,
    WorkspaceCommitIndeterminateError,
    WorkspaceRecoveryRequiredError,
)
from .prepared import record_bytes
from .models import TimingMode


_PROCESS_CLOCK_ID: str | None = None
_PROCESS_CLOCK_ORIGIN_NS: int | None = None
_OPTIONAL_SQLITE_WRITE_ERROR_CODES = frozenset({
    sqlite3.SQLITE_BUSY,
    sqlite3.SQLITE_LOCKED,
    sqlite3.SQLITE_READONLY,
    sqlite3.SQLITE_IOERR,
    sqlite3.SQLITE_FULL,
    sqlite3.SQLITE_CANTOPEN,
    sqlite3.SQLITE_PROTOCOL,
})


_AGGREGATE_CONTEXT_FIELDS = frozenset({
    "algorithm_id", "arithmetic_precision", "argument_shapes", "backend",
    "batch_size", "cache_condition", "cache_hit", "cpu_threads",
    "columns", "device", "dimension", "dtype", "executable_cache_hit",
    "input_bytes", "input_shape", "kernel", "new_shape", "nodes", "nnz",
    "operation", "output_bytes", "output_shape", "precision", "quantity",
    "requested_cpu_threads", "resource", "resource_profile", "role", "shape",
    "stage", "static_pattern_cache_hit", "static_pattern_sha256",
    "worker_capacity",
})


class _AggregateBucket:
    """Constant-space timing statistics for one stable logical phase group."""

    __slots__ = (
        "group_key_sha256", "kind", "logical_parent", "context", "count",
        "status_counts", "elapsed_ns_sum", "elapsed_ns_min", "elapsed_ns_max",
        "max_context", "max_occurrence",
    )

    def __init__(self, group_key_sha256: str, kind: str, logical_parent: str,
                 context: Mapping[str, object]) -> None:
        self.group_key_sha256 = group_key_sha256
        self.kind = kind
        self.logical_parent = logical_parent
        self.context = dict(context)
        self.count = 0
        self.status_counts: dict[str, int] = {}
        self.elapsed_ns_sum = 0
        self.elapsed_ns_min: int | None = None
        self.elapsed_ns_max: int | None = None
        self.max_context: dict[str, object] | None = None
        self.max_occurrence: int | None = None

    def add(self, *, elapsed_ns: int, status: str, context: Mapping[str, object],
            occurrence: int) -> None:
        self.count += 1
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        self.elapsed_ns_sum += elapsed_ns
        if self.elapsed_ns_min is None or elapsed_ns < self.elapsed_ns_min:
            self.elapsed_ns_min = elapsed_ns
        if (
            self.elapsed_ns_max is None
            or elapsed_ns > self.elapsed_ns_max
            or (elapsed_ns == self.elapsed_ns_max and self.max_occurrence is not None
                and occurrence < self.max_occurrence)
        ):
            self.elapsed_ns_max = elapsed_ns
            self.max_context = dict(context)
            self.max_occurrence = occurrence

    def document(self) -> dict[str, object]:
        return {
            "group_key_sha256": self.group_key_sha256,
            "kind": self.kind,
            "logical_parent": self.logical_parent,
            "context": dict(self.context),
            "count": self.count,
            "status_counts": dict(sorted(self.status_counts.items())),
            "elapsed_ns_sum": self.elapsed_ns_sum,
            "elapsed_ns_min": self.elapsed_ns_min,
            "elapsed_ns_max": self.elapsed_ns_max,
            "max_context": self.max_context,
            "max_occurrence": self.max_occurrence,
        }


class _TimingWindow:
    """One unsaved batch prefix; swapped out while its independent ACK is pending."""

    __slots__ = ("aggregates", "spans", "generations")

    def __init__(self) -> None:
        self.aggregates: dict[str, _AggregateBucket] = {}
        self.spans: list[dict[str, object]] = []
        self.generations: list[dict[str, object]] = []

    def empty(self) -> bool:
        return not (self.aggregates or self.spans or self.generations)


_AGGREGATE_ROOT_SCOPE = "[]"


def _stable_context(details: Mapping[str, object] | None) -> dict[str, object]:
    """Keep only grouping dimensions, never point/frequency/iteration context."""
    if details is None:
        return {}
    return {
        str(key): value
        for key, value in details.items()
        if key in _AGGREGATE_CONTEXT_FIELDS
    }


def _logical_child(parent: str, kind: str) -> str:
    try:
        ancestors = json.loads(parent)
    except (TypeError, json.JSONDecodeError) as error:
        raise EvidenceIntegrityError(
            "Aggregate timing scope token is malformed.",
            stage="operation_timing",
            evidence={"parent": parent},
        ) from error
    if not isinstance(ancestors, list) or any(not isinstance(item, str) for item in ancestors):
        raise EvidenceIntegrityError(
            "Aggregate timing scope token is malformed.",
            stage="operation_timing",
            evidence={"parent": parent},
        )
    return json.dumps([*ancestors, kind], ensure_ascii=False, separators=(",", ":"))


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
        timing: TimingMode = "aggregate",
        start_tick_ns: int | None = None,
    ) -> None:
        if timing not in {"aggregate", "detailed"}:
            raise ValueError("timing must be 'aggregate' or 'detailed'")
        self.binding = binding
        self.operation_id = str(uuid4())
        self.root_span_id = str(uuid4())
        self.method = kind
        self.backend = backend
        self.precision = precision
        self.timing = timing
        self.clock = _clock_binding(start_tick_ns)
        self._origin_ns = int(self.clock["monotonic_origin_ns"])
        self._start_tick_ns = perf_counter_ns() if start_tick_ns is None else start_tick_ns
        self._local = local()
        self._lock = RLock()
        self._publish_lock = Lock()
        self._window = _TimingWindow()
        self._inflight: _TimingWindow | None = None
        self._batch_sequence = 0
        self._occurrence = 0
        self._staged_generations: dict[int, dict[str, object]] = {}
        self._completed_generation = 0
        self._diagnostics_stopped = False
        self._last_diagnostic_error: BaseException | None = None
        self._diagnostic_note_target: BaseException | None = None
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
        diagnostic_error: BaseException | None = None
        try:
            self.publish_diagnostics(primary_error=error)
        except BaseException as caught:
            diagnostic_error = caught
        with self._lock:
            finalize_root = not self._diagnostics_stopped
        if finalize_root:
            try:
                from .storage import finish_operation

                with self.binding.writer():  # type: ignore[attr-defined]
                    # Keep lock acquisition, workspace/Plan revalidation and
                    # writer preparation in the measured root interval. The
                    # final root append itself cannot include its own return.
                    self._finish_root(perf_counter_ns(), status)
                    finish_operation(
                        self.binding,
                        self.root_row(),
                        failure=None if error is None else self._error_record(error),
                    )
            except BaseException as finalization_error:
                if self._end_tick_ns is None:
                    self._finish_root(perf_counter_ns(), status)
                if diagnostic_error is None:
                    diagnostic_error = finalization_error
                else:
                    self._add_note(
                        diagnostic_error,
                        "Operation root finalization also failed: "
                        f"{type(finalization_error).__name__}: {finalization_error}",
                    )
        else:
            self._finish_root(perf_counter_ns(), status)
        if diagnostic_error is not None:
            if error is not None:
                self._add_note(
                    error,
                    "SCNSim operation diagnostics/finalization also failed: "
                    f"{type(diagnostic_error).__name__}: {diagnostic_error}",
                )
                if not self._is_optional_diagnostic_error(diagnostic_error):
                    return False
            elif self._is_optional_diagnostic_error(diagnostic_error):
                self._warn_optional_diagnostic(diagnostic_error)
            else:
                raise diagnostic_error
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

    def _stack(self) -> list[str]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = [self.root_span_id if self.timing == "detailed" else _AGGREGATE_ROOT_SCOPE]
            self._local.stack = stack
        return stack

    @property
    def current_scope(self) -> str:
        """Current actual UUID or stable aggregate logical ancestry token."""
        return self._stack()[-1]

    @property
    def timing_mode(self) -> TimingMode:
        return self.timing

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
        stack = self._stack()
        parent_scope = stack[-1] if stack else None
        scope = (
            str(uuid4()) if self.timing == "detailed"
            else _logical_child(parent_scope or _AGGREGATE_ROOT_SCOPE, kind)
        )
        with self._lock:
            collecting = not self._diagnostics_stopped
        start_tick = perf_counter_ns() if collecting else None
        stack.append(scope)
        error: BaseException | None = None
        try:
            yield scope
        except BaseException as caught:
            error = caught
            raise
        finally:
            if stack and stack[-1] == scope:
                stack.pop()
            if start_tick is not None:
                self._record_interval(
                    scope=scope,
                    parent_scope=parent_scope,
                    kind=kind,
                    start_tick_ns=start_tick,
                    end_tick_ns=perf_counter_ns(),
                    status=self._status_for(error),
                    details=details,
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
        if self._closed:
            raise RuntimeError("closed operation recorder cannot add a measurement")
        with self._lock:
            if self._diagnostics_stopped:
                return
        parent_scope = parent_span_id or self.current_scope
        scope = (
            str(uuid4()) if self.timing == "detailed"
            else _logical_child(parent_scope, kind)
        )
        self._record_interval(
            scope=scope,
            parent_scope=parent_scope,
            kind=kind,
            start_tick_ns=start_tick_ns,
            end_tick_ns=end_tick_ns,
            status=status,
            details=details,
        )

    def _record_interval(
        self,
        *,
        scope: str,
        parent_scope: str | None,
        kind: str,
        start_tick_ns: int,
        end_tick_ns: int,
        status: str,
        details: Mapping[str, object] | None,
    ) -> None:
        elapsed_ns = end_tick_ns - start_tick_ns
        if elapsed_ns < 0:
            raise EvidenceIntegrityError(
                "An operation timing interval has a negative duration.",
                stage="operation_timing",
                evidence={"kind": kind, "start_tick_ns": start_tick_ns, "end_tick_ns": end_tick_ns},
            )
        detail_map = dict(details or {})
        with self._lock:
            if self._diagnostics_stopped:
                return
            generation_summary = detail_map.get("generation_performance")
            if isinstance(generation_summary, Mapping):
                generation = generation_summary.get("generation")
                if isinstance(generation, int) and not isinstance(generation, bool):
                    self._staged_generations[generation] = dict(generation_summary)
                detail_map.pop("generation_performance", None)
            if self.timing == "aggregate":
                stable = _stable_context(detail_map)
                stable["backend"] = self.backend
                stable["precision"] = self.precision
                key_payload = {
                    "operation_id": self.operation_id,
                    "attempt_id": self._attempt_id,
                    "clock_id": self.clock["id"],
                    "kind": kind,
                    "logical_parent": parent_scope or _AGGREGATE_ROOT_SCOPE,
                    "context": stable,
                }
                group_key = sha256(record_bytes(key_payload)).hexdigest()
                bucket = self._window.aggregates.get(group_key)
                if bucket is None:
                    bucket = _AggregateBucket(
                        group_key, kind, parent_scope or _AGGREGATE_ROOT_SCOPE, stable,
                    )
                    self._window.aggregates[group_key] = bucket
                self._occurrence += 1
                bucket.add(
                    elapsed_ns=elapsed_ns,
                    status=status,
                    context=detail_map,
                    occurrence=self._occurrence,
                )
            else:
                self._window.spans.append(self._span_row(
                    span_id=scope,
                    parent_span_id=parent_scope,
                    kind=kind,
                    start_tick_ns=start_tick_ns,
                    end_tick_ns=end_tick_ns,
                    status=status,
                    details=detail_map,
                ))

    def complete_generation(self, generation: int) -> None:
        """Stage generation summaries only through the acknowledged prefix."""
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("generation must be a positive integer")
        with self._lock:
            if self._diagnostics_stopped:
                return
            if generation <= self._completed_generation:
                return
            for number in sorted(key for key in self._staged_generations if key <= generation):
                self._window.generations.append(dict(self._staged_generations.pop(number)))
            self._completed_generation = generation

    def publish_diagnostics(
        self,
        *,
        primary_error: BaseException | None = None,
    ) -> Mapping[str, object] | None:
        """Persist one independent diagnostic batch and acknowledge its exact prefix."""
        with self._publish_lock:
            return self._publish_diagnostics(primary_error=primary_error)

    def _publish_diagnostics(
        self,
        *,
        primary_error: BaseException | None,
    ) -> Mapping[str, object] | None:
        with self._lock:
            if self._diagnostics_stopped:
                stopped_error = self._last_diagnostic_error
                window = None
            elif self._window.empty():
                return None
            else:
                stopped_error = None
                window = self._window
            if window is None:
                pass
            elif self._inflight is not None:
                raise EvidenceIntegrityError(
                    "A timing batch is already awaiting its independent commit ACK.",
                    stage="operation_timing_ack",
                    evidence={"operation_id": self.operation_id},
                )
            else:
                self._inflight = window
                self._window = _TimingWindow()
                self._batch_sequence += 1
                batch_id = f"{self.operation_id}:{self._batch_sequence}"
                batch = {
                    "schema": "scnsim.operation_timing_batch",
                    "schema_version": 1,
                    "batch_id": batch_id,
                    "operation_id": self.operation_id,
                    "attempt_id": self._attempt_id,
                    "clock": dict(self.clock),
                    "timing_mode": self.timing,
                    "aggregates": [
                        window.aggregates[key].document()
                        for key in sorted(window.aggregates)
                    ],
                    "spans": [dict(row) for row in window.spans],
                    "generations": [dict(row) for row in window.generations],
                }
        if window is None:
            if stopped_error is not None:
                self._note_diagnostic_error(primary_error, stopped_error)
            return None
        from .storage import publish_timing_batch

        try:
            acknowledgement = publish_timing_batch(self.binding, batch)
        except (OSError, sqlite3.Error, WorkspaceCommitIndeterminateError,
                WorkspaceRecoveryRequiredError) as error:
            if not self._is_optional_diagnostic_error(error):
                raise
            self._stop_optional_diagnostics(error, primary_error=primary_error)
            return None
        if not isinstance(acknowledgement, Mapping):
            raise EvidenceIntegrityError(
                "Timing publication returned a non-object acknowledgement.",
                stage="operation_timing_ack",
                evidence={"batch_id": batch_id},
            )
        state = acknowledgement.get("state")
        if state != "committed":
            self._stop_optional_diagnostics(RuntimeError(
                "timing batch acknowledgement was not committed "
                f"(batch_id={batch_id!r}, state={state!r})"
            ), primary_error=primary_error)
            return None
        if acknowledgement.get("batch_id") != batch_id:
            raise EvidenceIntegrityError(
                "Timing publication acknowledgement names a different batch.",
                stage="operation_timing_ack",
                evidence={"batch_id": batch_id, "acknowledged_batch_id": acknowledgement.get("batch_id")},
            )
        if not isinstance(acknowledgement.get("transaction_id"), str):
            raise EvidenceIntegrityError(
                "Timing publication acknowledgement has no transaction identity.",
                stage="operation_timing_ack",
                evidence={"batch_id": batch_id},
            )
        if not isinstance(acknowledgement.get("reference"), Mapping):
            raise EvidenceIntegrityError(
                "Timing publication acknowledgement has no canonical object reference.",
                stage="operation_timing_ack",
                evidence={"batch_id": batch_id},
            )
        with self._lock:
            if self._inflight is not window:
                raise EvidenceIntegrityError(
                    "Timing publication ACK does not match the staged prefix.",
                    stage="operation_timing_ack",
                    evidence={"batch_id": batch_id},
                )
            self._inflight = None
        return acknowledgement

    @staticmethod
    def _status_for(error: BaseException | None) -> str:
        if error is None:
            return "success"
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            return "interrupted"
        return "failure"

    @staticmethod
    def _is_optional_diagnostic_error(error: BaseException) -> bool:
        if isinstance(error, sqlite3.Error):
            code = getattr(error, "sqlite_errorcode", None)
            return (
                isinstance(code, int)
                and (code & 0xFF) in _OPTIONAL_SQLITE_WRITE_ERROR_CODES
            )
        return isinstance(
            error,
            (OSError, WorkspaceCommitIndeterminateError, WorkspaceRecoveryRequiredError),
        )

    def _stop_optional_diagnostics(
        self,
        error: BaseException,
        *,
        primary_error: BaseException | None = None,
    ) -> None:
        with self._lock:
            self._diagnostics_stopped = True
            self._last_diagnostic_error = error
            self._window = _TimingWindow()
            self._staged_generations.clear()
        self._note_diagnostic_error(primary_error, error)
        self._warn_optional_diagnostic(error)

    def _note_diagnostic_error(
        self,
        primary_error: BaseException | None,
        diagnostic_error: BaseException,
    ) -> None:
        if primary_error is None:
            return
        with self._lock:
            if self._diagnostic_note_target is primary_error:
                return
            self._diagnostic_note_target = primary_error
        self._add_note(
            primary_error,
            "SCNSim optional operation timing diagnostics also failed: "
            f"{type(diagnostic_error).__name__}: {diagnostic_error}",
        )

    @staticmethod
    def _warn_optional_diagnostic(error: BaseException) -> None:
        message = (
            "SCNSim optional operation timing diagnostics were not persisted: "
            f"{type(error).__name__}: {error}"
        )
        try:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        except Exception:
            # A warnings-as-errors policy must not replace a successful numerical result.
            pass

    @staticmethod
    def _add_note(error: BaseException, message: str) -> None:
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            try:
                add_note(message)
            except Exception:
                pass

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
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "kind": kind,
            "start_ns": start_tick_ns - self._origin_ns,
            "end_ns": end_tick_ns - self._origin_ns,
            "status": status,
            "details": dict(details or {}),
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
) -> bool:
    return (
        (selected_ids is None or row.get("operation_id") in selected_ids)
        and (method is None or row.get("method") == method)
        and (backend is None or row.get("backend") == backend)
        and (precision is None or row.get("precision") == precision)
    )


def _selected_operation_ids(operations: Sequence[str] | str | None) -> set[str] | None:
    if operations is None:
        return None
    if isinstance(operations, str):
        return {operations}
    return set(operations)


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
):
    """Project current SQLite operation authority and independent diagnostics."""
    selected_ids = _selected_operation_ids(operations)
    if rows is None:
        source_operations = source_batches = ()
        source_task_states: Mapping[str, object] = {}
    else:
        source_operations = rows["operations"]
        source_batches = rows["timing_batches"]
        source_task_states = rows["task_states"]
    selected_roots = [
        dict(row)
        for row in source_operations  # type: ignore[union-attr]
        if _matching_operation(
            row,
            selected_ids=selected_ids,
            method=method,
            backend=backend,
            precision=precision,
        )
    ]
    selected_operation_ids = {
        str(row["operation_id"])
        for row in selected_roots
        if isinstance(row.get("operation_id"), str)
    }
    selected_batches = [
        dict(row)
        for row in source_batches  # type: ignore[union-attr]
        if row.get("operation_id") in selected_operation_ids  # type: ignore[union-attr]
    ]
    task_states = {
        operation_id: dict(source_task_states[operation_id])
        for operation_id in sorted(selected_operation_ids)
        if operation_id in source_task_states
        and isinstance(source_task_states[operation_id], Mapping)
    }
    clocks: dict[str, dict[str, object]] = {}
    for row in selected_roots:
        clock = row.get("clock")
        if isinstance(clock, Mapping) and isinstance(clock.get("id"), str):
            clocks[str(clock["id"])] = dict(clock)
    for batch in selected_batches:
        clock = batch.get("clock")
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
        "schema_version": 2,
        "plan_sha256": plan_sha256,
        "workspace_instance_id": workspace_instance_id,
        "operations": selected_roots,
        "timing_batches": selected_batches,
        "task_states": task_states,
        "counts": {
            "operations": len(selected_roots),
            "timing_batches": len(selected_batches),
            "aggregates": sum(
                len(batch.get("aggregates", ()))
                for batch in selected_batches
                if isinstance(batch.get("aggregates", ()), Sequence)
            ),
            "spans": sum(
                len(batch.get("spans", ()))
                for batch in selected_batches
                if isinstance(batch.get("spans", ()), Sequence)
            ),
            "generations": sum(
                len(batch.get("generations", ()))
                for batch in selected_batches
                if isinstance(batch.get("generations", ()), Sequence)
            ),
            "operation_status": _count(selected_roots, "status"),
        },
        "clock_domains": [clocks[key] for key in sorted(clocks)],
        "numerical_refs": numerical_refs,
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
    """Read operation roots, bounded timing batches and current task-state refs."""
    from . import storage

    with binding.reader():  # type: ignore[attr-defined]
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
    return project_indexed_operation_rows(
        indexed_rows,
        workspace=workspace,
        plan_sha256=plan_sha256,
        workspace_instance_id=workspace_instance_id,
        operations=operations,
        method=method,
        backend=backend,
        precision=precision,
    )
