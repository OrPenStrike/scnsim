"""Actual-dtype numerical requests and synchronized detached outcomes.

Consumes immutable compilation records; no Run, Workspace, search scheduling
or result-publication authority is retained by these numerical handoffs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Protocol
import numpy as np
from numpy.typing import NDArray
from ..compilation.models import RealizedView, FloatArray, ComplexArray, immutable_array

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
class EvaluationJob:
    """One ordered numerical request; all coordinate indices are zero-based.

    direct uses frequencies_hz and publishes S/Y/Z (f,q,q). response_element
    uses one frequency plus family/output/input in terminal order. diagonal_root
    uses coordinate_index in terminal order and omega_start_rad_s, with the
    original root hint retained as evidence. No candidate coefficients are JIT
    constants. The host owns ids, candidate/leaf context and continuation paths.
    """

    id: str
    kind: Literal["direct", "diagonal_root", "response_element", "operator_element_root", "hybridized_pole", "transfer_zero", "residue_normalized_coupling", "operator"]
    view: RealizedView
    frequencies_hz: FloatArray | None = None
    coordinate_index: int | None = None
    root_hint_hz: float | None = None
    omega_start_rad_s: complex | None = None
    family: Literal["S", "Y", "Z"] | None = None
    input_index: int | None = None
    output_index: int | None = None
    row_index: int | None = None
    column_index: int | None = None
    branches: tuple[RootBranch, RootBranch] | None = None
    evaluation_omega_rad_s: complex | None = None
    frequency_mode: Literal["fixed", "complex_root_midpoint"] | None = None

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
    operator_values: NDArray | None = None
    null_vector: NDArray | None = None
    numerator_slope: complex | None = None
    denominator: complex | None = None
    residue_a: complex | None = None
    residue_b: complex | None = None
    coupling_rad_s: complex | None = None
    branch_roots_rad_s: NDArray | None = None
    evaluation_omega_rad_s: complex | None = None
    evidence_bytes: bytes = b"{}"

    def __post_init__(self) -> None:
        for name in ("S", "Y", "Z", "operator_values", "null_vector", "branch_roots_rad_s"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, immutable_numerical_array(value))


@dataclass(frozen=True, slots=True)
class RootBranch:
    """Finished host-certified dependency; coupling never locates it again."""

    kind: Literal["diagonal_root", "hybridized_pole"]
    coordinate_index: int | None
    root_hint_hz: float
    result: EvaluationResult


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

