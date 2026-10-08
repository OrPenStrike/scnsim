"""Shared benchmark declarations and immutable host/backend handoffs.

The Python task owns candidate order, continuation and durable state. Numerical
adapters consume these descriptors; they do not capture Plans or mutate Views.
Arrays have immutable byte backing, so backend preparation cannot modify them.
BenchmarkSpec owns persistence policy declarations; task/storage own their durable
realization. These policies never change numerical or CMA scheduling semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Arm = Literal["original_julia", "python_julia_reuse", "python_julia_lu", "python_jax"]


def immutable_array(value: object, *, complex_: bool = False) -> FloatArray | ComplexArray:
    """Detach into an immutable binary64 buffer, preserving the input shape."""
    array = np.asarray(value, dtype=np.complex128 if complex_ else np.float64)
    backing = array
    while isinstance(backing, np.ndarray):
        backing = backing.base
    if not array.flags.writeable and isinstance(backing, bytes) and array.flags.c_contiguous:
        return array
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def immutable_numerical_array(value: object) -> np.ndarray:
    """Keep actual numerical precision in immutable result buffers."""
    array = np.asarray(value)
    if array.dtype not in (np.dtype("float32"), np.dtype("float64"), np.dtype("complex64"), np.dtype("complex128")):
        raise TypeError(f"unsupported numerical array dtype {array.dtype}")
    backing = array
    while isinstance(backing, np.ndarray):
        backing = backing.base
    if not array.flags.writeable and isinstance(backing, bytes) and array.flags.c_contiguous:
        return array
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class MeshGroup:
    """One ordered parameter interval with fixed per-line section counts (SI)."""

    lower: float
    upper: float
    lower_inclusive: bool
    upper_inclusive: bool
    sections: tuple[tuple[tuple[str, ...], int], ...]


@dataclass(frozen=True, slots=True)
class MeshSpec:
    """Request-owned mesh override; physical Plan and lengths remain unchanged."""

    kind: Literal["dynamic", "fixed", "grouped"] = "dynamic"
    sections: tuple[tuple[tuple[str, ...], int], ...] = ()
    parameter_key: tuple[str, str] | None = None
    groups: tuple[MeshGroup, ...] = ()
    derivation_bytes: bytes = b"{}"


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Experiment resources; native Julia overrides do not change host quota.

    cpu_threads declares physical affinity profiles and Python/JAX resources.
    Each native Julia count defaults independently to that task's quota.
    checkpoint controls CMA resume snapshots, not required baseline/root anchors.
    Omitted diagnostics resolves to boundary for common Python arms and immediate
    for original Julia. Notified events are always durable first.
    """

    arms: tuple[Arm, ...] = (
        "original_julia", "python_julia_reuse", "python_julia_lu", "python_jax",
    )
    cpu_threads: tuple[int, ...] = (1, 32)
    device: Literal["cpu"] = "cpu"
    repeats: int = 1
    mesh: MeshSpec = field(default_factory=MeshSpec)
    task_kind: Literal["full", "cohort"] = "full"
    cohort_bytes: bytes = b"[]"
    julia_threads: int | None = None
    julia_blas_threads: int | None = None
    checkpoint: Literal["generation", "off"] = "generation"
    diagnostics: Literal["immediate", "boundary"] | None = None

    def __post_init__(self) -> None:
        if self.checkpoint not in ("generation", "off"):
            raise ValueError("checkpoint must be generation or off")
        if self.diagnostics not in (None, "immediate", "boundary"):
            raise ValueError("diagnostics must be immediate, boundary, or None")
        if "original_julia" in self.arms and (
            self.checkpoint != "generation" or self.diagnostics not in (None, "immediate")
        ):
            raise ValueError("original_julia requires generation checkpoints and immediate diagnostics")
        for name in ("julia_threads", "julia_blas_threads"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer or None")
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class SeriesRL:
    """Full-node incidence (n,r), resistance/inductance (r,r), coherent SI."""

    incidence: FloatArray
    resistance: FloatArray
    inductance: FloatArray

    def __post_init__(self) -> None:
        for name in ("incidence", "resistance", "inductance"):
            object.__setattr__(self, name, immutable_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class CompiledModel:
    """One physical point in canonical node/Port order, before View reduction.

    C/K/G are (n,n), B is (n,p), R is (p,p), M is (p,). All are real
    binary64 SI arrays; series-RL impedance remains frequency-dependent.
    Evidence contains resolved bindings, branch rows and actual line grids.
    """

    node_ids: tuple[str, ...]
    port_ids: tuple[str, ...]
    C: FloatArray
    K: FloatArray
    G: FloatArray
    B: FloatArray
    R: FloatArray
    M: FloatArray
    series_rl: tuple[SeriesRL, ...] = ()
    evidence_bytes: bytes = b"{}"

    def __post_init__(self) -> None:
        for name in ("C", "K", "G", "B", "R", "M"):
            object.__setattr__(self, name, immutable_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class RealizedView:
    """Candidate-specific transformed model and selected generalized boundary.

    model retains every physical/internal node and ordered full B/R/M.
    coordinates describes the active public basis, distinct from model.node_ids.
    selected_indices order the root operator; terminal_ids order response axes.
    coordinate_port_map is (n,p) in original_node_ids order;
    selected_map is (q,p), Bk is (n,q) in model.node_ids order,
    Rk/Dk are (q,q), Go is (p,p).
    Boundary arrays are absent only for a non-Port-realizable root-only View.
    """

    model: CompiledModel
    coordinates: tuple[str, ...]
    terminal_ids: tuple[str, ...]
    coordinate_port_map: FloatArray
    selected_indices: tuple[int, ...]
    port_realizable: bool
    selected_map: FloatArray | None = None
    Bk: FloatArray | None = None
    Rk: FloatArray | None = None
    Dk: FloatArray | None = None
    Go: ComplexArray | None = None
    lineage_bytes: bytes = b"{}"
    signature: tuple[object, ...] = ()
    original_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("coordinate_port_map", "selected_map", "Bk", "Rk", "Dk", "Go"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, immutable_array(value, complex_=name == "Go"))


@dataclass(frozen=True, slots=True)
class EvaluationJob:
    """One ordered numerical request; all coordinate indices are zero-based.

    direct uses frequencies_hz and publishes S/Y/Z (f,q,q). response_element
    uses one frequency plus family/output/input in terminal order. diagonal_root
    uses coordinate_index in terminal order and omega_start_rad_s, with the
    original root hint retained as evidence. No candidate coefficients are JIT
    constants. The host owns ids, candidate/leaf context and continuation paths.
    """

    id: str
    kind: Literal["direct", "diagonal_root", "response_element"]
    view: RealizedView
    frequencies_hz: FloatArray | None = None
    coordinate_index: int | None = None
    root_hint_hz: float | None = None
    omega_start_rad_s: complex | None = None
    family: Literal["S", "Y", "Z"] | None = None
    input_index: int | None = None
    output_index: int | None = None

    def __post_init__(self) -> None:
        if self.frequencies_hz is not None:
            object.__setattr__(self, "frequencies_hz", immutable_array(self.frequencies_hz))


@dataclass(frozen=True, slots=True)
class NumericalFailure:
    """Existing numerical failure semantics; infrastructure failures are raised."""

    kind: str
    stage: str
    detail: str
    evidence_bytes: bytes = b"{}"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Synchronized numerical outcome; failure never carries a success value.

    Root omega and slope use radians/second and operator SI conventions.
    response_value is one complex S/Y/Z value; full arrays use frequency-first
    axes. Evidence retains Newton certificates, solve observations and work.
    """

    id: str
    failure: NumericalFailure | None = None
    S: NDArray[np.complex64] | ComplexArray | None = None
    Y: NDArray[np.complex64] | ComplexArray | None = None
    Z: NDArray[np.complex64] | ComplexArray | None = None
    response_value: complex | None = None
    root_omega_rad_s: complex | None = None
    root_slope: complex | None = None
    evidence_bytes: bytes = b"{}"

    def __post_init__(self) -> None:
        for name in ("S", "Y", "Z"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, immutable_numerical_array(value))


class NumericalBackend(Protocol):
    """Adapters own complete local Newton evaluation, not host continuation.

    evaluate_batch returns exactly one result per input job in the same order,
    including structured numerical failures. Process death/protocol corruption
    raises immediately. Return requires actual device/process synchronization.
    close releases adapter resources; neither method mutates descriptors.
    """

    def identity(self) -> dict[str, object]: ...
    def evaluate_batch(self, jobs: tuple[EvaluationJob, ...]) -> tuple[EvaluationResult, ...]: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Exact independent attempt binding, carried by each process frame."""

    request_sha256: str
    arm: Arm
    sample: int
    attempt_id: str
    environment_sha256: str
    device: str
    cpu_threads: int


@dataclass(frozen=True, slots=True)
class TaskEvent:
    """Canonical host/storage event; sequence is monotonic within one task."""

    task_id: str
    sequence: int
    kind: Literal[
        "ready", "authorized", "started", "timing", "baseline_ready",
        "checkpoint_committed", "population_observed",
        "generation_ready", "evaluation", "progress", "completed", "interrupted", "failed",
    ]
    payload_bytes: bytes


@dataclass(frozen=True, slots=True)
class Measurement:
    """One observed monotonic interval; overlapping intervals are not summed."""

    task_id: str
    stage: str
    start_ns: int
    duration_ns: int
    counts: tuple[tuple[str, int], ...] = ()
    memory_bytes: tuple[tuple[str, int], ...] = ()
    detail_bytes: bytes = b"{}"


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Read-only observation manifest; numerical and rendering outcomes differ."""

    workspace: Path
    manifest_bytes: bytes

    def document(self) -> dict[str, object]:
        return json.loads(self.manifest_bytes)

    def to_json(self) -> str:
        return self.manifest_bytes.decode("utf-8")

    def to_html(self) -> str:
        from ..visualization.benchmark import render_benchmark
        return render_benchmark(self)

    def show(self):
        """Display the stored report lazily, without initializing a runtime."""
        from ..results.base import HtmlPresentation
        return HtmlPresentation(self.to_html())

    def write_html(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(self.to_html(), encoding="utf-8")
        return target

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_bytes(self.manifest_bytes)
        return target

    @classmethod
    def open(cls, workspace: str | Path) -> BenchmarkResult:
        from .storage import open_record
        return open_record(workspace)

    @classmethod
    def from_document(cls, workspace: Path, document: dict[str, object]) -> BenchmarkResult:
        from .prepared import record_bytes
        return cls(workspace, record_bytes(document))
