"""Immutable physical compiler-to-numerics handoff in canonical SI order.

Compilation owns binary64 physical coefficients/indices and declared Views.
These records neither initialize numerical runtimes nor own persisted evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

def immutable_array(value: object, *, complex_: bool = False) -> FloatArray | ComplexArray:
    """Detach into an immutable binary64 buffer, preserving the input shape."""
    array = np.asarray(value, dtype=np.complex128 if complex_ else np.float64)
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
class SparseMatrix:
    """Immutable canonical COO; structural zeros survive numeric coalescing.

    Compiler/View own binary64 SI values. Numerical adapters consume indexed
    operands directly; explicit dense conversion belongs only to Julia transport.
    """

    shape: tuple[int, int]
    rows: np.ndarray
    cols: np.ndarray
    values: FloatArray

    def __post_init__(self) -> None:
        for name in ("rows", "cols"):
            array = np.asarray(getattr(self, name), dtype=np.int64)
            object.__setattr__(self, name, np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape))
        object.__setattr__(self, "values", immutable_array(self.values))

    @classmethod
    def from_entries(cls, shape, rows, cols, values) -> SparseMatrix:
        rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if not len(rows):
            return cls(tuple(shape), rows, cols, values)
        order = np.lexsort((cols, rows))
        rows, cols, values = rows[order], cols[order], values[order]
        starts = np.r_[0, np.flatnonzero((rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])) + 1]
        return cls(tuple(shape), rows[starts], cols[starts], np.add.reduceat(values, starts))

    def entry(self, row: int, col: int) -> float:
        return float(np.sum(self.values[(self.rows == row) & (self.cols == col)]))

    def row(self, row: int) -> np.ndarray:
        result = np.zeros(self.shape[1])
        take = self.rows == row
        np.add.at(result, self.cols[take], self.values[take])
        return result

    def column_rows(self, col: int) -> np.ndarray:
        return self.rows[self.cols == col]

    def to_dense(self) -> np.ndarray:
        """Explicit transport-only materialization, never a numerical fallback."""
        result = np.zeros(self.shape)
        np.add.at(result, (self.rows, self.cols), self.values)
        return result

    def right_multiply(self, matrix: np.ndarray) -> SparseMatrix:
        # Dense right operands are small Port maps. Include zero weights so
        # structural identity does not depend on their candidate values.
        q = matrix.shape[1]
        return self.from_entries((self.shape[0], q), np.repeat(self.rows, q),
                                 np.tile(np.arange(q), len(self.rows)),
                                 (self.values[:, None] * matrix[self.cols]).reshape(-1))

    def transform_rows(self, mapping, new_size: int) -> SparseMatrix:
        rows, cols, values = [], [], []
        for row, col, value in zip(self.rows, self.cols, self.values, strict=True):
            for target, weight in mapping[row]:
                rows.append(target); cols.append(col); values.append(weight * value)
        return self.from_entries((new_size, self.shape[1]), rows, cols, values)

    def congruence(self, mapping, new_size: int) -> SparseMatrix:
        rows, cols, values = [], [], []
        for row, col, value in zip(self.rows, self.cols, self.values, strict=True):
            for left, lw in mapping[row]:
                for right, rw in mapping[col]:
                    rows.append(left); cols.append(right); values.append(lw * value * rw)
        return self.from_entries((new_size, new_size), rows, cols, values)


@dataclass(frozen=True, slots=True)
class SeriesRL:
    """Indexed full-node incidence (n,r), local dense R/L (r,r), coherent SI.

    One ordered section owns one complete frequency-dependent contribution;
    coalesce that contribution before its absolute certificate bound.
    """

    incidence: SparseMatrix
    resistance: FloatArray
    inductance: FloatArray

    def __post_init__(self) -> None:
        for name in ("resistance", "inductance"):
            object.__setattr__(self, name, immutable_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class CompiledModel:
    """Candidate-specific sparse physical forms in canonical node/Port order.

    C/K/G are coalesced indexed (n,n), B indexed (n,p), R (p,p), M (p,).
    Binary64 SI values remain separate from structure, including exact zeros.
    """

    node_ids: tuple[str, ...]
    port_ids: tuple[str, ...]
    C: SparseMatrix
    K: SparseMatrix
    G: SparseMatrix
    B: SparseMatrix
    R: FloatArray
    M: FloatArray
    series_rl: tuple[SeriesRL, ...] = ()
    evidence_bytes: bytes = b"{}"

    def __post_init__(self) -> None:
        for name in ("R", "M"):
            object.__setattr__(self, name, immutable_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class RealizedView:
    """Sparse transformed model and ordered selected generalized boundary.

    coordinate_port_map keeps its original-node public evidence axis. Bk is
    indexed (n,q); selected_map/Rk/Dk/Go remain small dense boundary arrays.
    """

    model: CompiledModel
    coordinates: tuple[str, ...]
    terminal_ids: tuple[str, ...]
    coordinate_port_map: FloatArray
    selected_indices: tuple[int, ...]
    port_realizable: bool
    selected_map: FloatArray | None = None
    Bk: SparseMatrix | None = None
    Rk: FloatArray | None = None
    Dk: FloatArray | None = None
    Go: ComplexArray | None = None
    lineage_bytes: bytes = b"{}"
    signature: tuple[object, ...] = ()
    original_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("coordinate_port_map", "selected_map", "Rk", "Dk", "Go"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, immutable_array(value, complex_=name == "Go"))

