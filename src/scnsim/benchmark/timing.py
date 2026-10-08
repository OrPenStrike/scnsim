"""Monotonic inclusive timing records for actual benchmark work."""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from time import perf_counter_ns
from typing import Iterator, Mapping
from uuid import uuid4

from .models import Measurement
from .prepared import record_bytes


class TimingRecorder:
    """Collect real nested intervals relative to one benchmark-call origin.

    Intervals are intentionally inclusive and may overlap. Consumers must not
    sum parent and child intervals as if they were disjoint work.
    """

    def __init__(self, start_tick_ns: int | None = None) -> None:
        self.clock_id = str(uuid4())
        self.origin_ns = perf_counter_ns() if start_tick_ns is None else start_tick_ns
        self._measurements: list[Measurement] = []
        self._persisted_count = 0
        self._lock = RLock()

    @property
    def measurements(self) -> tuple[Measurement, ...]:
        with self._lock:
            return tuple(self._measurements)

    @property
    def pending_measurements(self) -> tuple[Measurement, ...]:
        """Return only intervals not yet acknowledged by the durable recorder."""
        with self._lock:
            return tuple(self._measurements[self._persisted_count:])

    def mark_persisted(self, measurements: tuple[Measurement, ...]) -> None:
        """Advance the recorder cursor after the corresponding journal commit."""
        with self._lock:
            pending = tuple(self._measurements[self._persisted_count:])
            if pending[:len(measurements)] != measurements:
                raise RuntimeError("timing persistence cursor no longer matches its committed prefix")
            self._persisted_count += len(measurements)

    def mark(self) -> int:
        """Read the same monotonic clock used for all recorder intervals."""
        return perf_counter_ns()

    @property
    def clock_binding(self) -> dict[str, object]:
        """Return the exact clock domain and origin used by this invocation."""
        return {
            "id": self.clock_id,
            "source": "time.perf_counter_ns",
            "unit": "nanoseconds",
            "monotonic_origin_ns": self.origin_ns,
        }

    def add_interval(
        self,
        stage: str,
        *,
        task_id: str,
        start_tick_ns: int,
        end_tick_ns: int,
        counts: Mapping[str, int] | None = None,
        memory_bytes: Mapping[str, int] | None = None,
        details: Mapping[str, object] | None = None,
    ) -> Measurement:
        """Record a span measured by another owner using this monotonic clock."""
        measurement = Measurement(
            task_id=task_id,
            stage=stage,
            start_ns=start_tick_ns - self.origin_ns,
            duration_ns=end_tick_ns - start_tick_ns,
            counts=tuple(sorted((counts or {}).items())),
            memory_bytes=tuple(sorted((memory_bytes or {}).items())),
            detail_bytes=record_bytes(dict(details or {})),
        )
        with self._lock:
            self._measurements.append(measurement)
        return measurement

    @contextmanager
    def span(
        self,
        stage: str,
        *,
        task_id: str = "benchmark",
        counts: Mapping[str, int] | None = None,
        memory_bytes: Mapping[str, int] | None = None,
        details: Mapping[str, object] | None = None,
    ) -> Iterator[None]:
        """Measure an inclusive operation, retaining timing on exceptions."""
        start = self.mark()
        try:
            yield
        finally:
            self.add_interval(
                stage,
                task_id=task_id,
                start_tick_ns=start,
                end_tick_ns=self.mark(),
                counts=counts,
                memory_bytes=memory_bytes,
                details=details,
            )
