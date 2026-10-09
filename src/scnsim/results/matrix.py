from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from pint import Quantity

from ..canonical import float64_from_hex
from ..construction import unavailable
from ..value_storage import quantity_view
from ..visualization.presentation import Theme
from .base import AnalysisResult, HtmlPresentation, MatrixFamilyResult, MatrixView, _fresh_quantity_attribute

if TYPE_CHECKING:
    from plotly.graph_objects import Figure
    from .derived import TraceResult

@dataclass(frozen=True, slots=True)
class ScatteringMatrixResult(MatrixFamilyResult):
    """Selected-view generalized power-wave S matrices and presentation."""

    def __init__(self) -> None:
        unavailable("ScatteringMatrixResult construction")

@dataclass(frozen=True, slots=True)
class ReconciliationEvidence:
    comparable: bool
    reason: str | None
    last_comparable_ancestor: str
    residual: float | None
    evidence_sha256: str

    def __init__(self) -> None:
        unavailable("ReconciliationEvidence construction")

@dataclass(frozen=True, slots=True)
class HBScatteringMatrixResult(ScatteringMatrixResult):
    backend_native: MatrixView | None = None
    reconciliation: ReconciliationEvidence | None = None

    def __init__(self) -> None:
        unavailable("HBScatteringMatrixResult construction")

@dataclass(frozen=True, slots=True)
class DirectSolveResult(AnalysisResult):
    """Complete finite Direct S/Y/Z response from one verified receipt."""

    frequencies: Quantity
    s: ScatteringMatrixResult
    y: MatrixFamilyResult
    z: MatrixFamilyResult
    traces: Mapping[str, TraceResult] = field(default_factory=dict)
    _quantity_fields = frozenset({"frequencies"})

    def __init__(self) -> None:
        unavailable("DirectSolveResult construction")

    def __getattribute__(self, name: str) -> object:
        return _fresh_quantity_attribute(self, name)

    def _family(self, family: Literal["S", "Y", "Z"]) -> MatrixFamilyResult:
        if family == "S":
            return self.s
        if family == "Y":
            return self.y
        if family == "Z":
            return self.z
        raise ValueError("family must be 'S', 'Y', or 'Z'")

    def plot(self, *, family: Literal["S", "Y", "Z"] = "S", **presentation: object) -> Figure:
        return self._family(family).plot(**presentation)

    def show(self, *, family: Literal["S", "Y", "Z"] = "S", **presentation: object) -> None:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(family=family, **presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        family: Literal["S", "Y", "Z"] = "S",
        **presentation: object,
    ) -> Figure:
        return self._family(family).add_to(fig, **presentation)

@dataclass(frozen=True, slots=True)
class DirectQuantityResult(AnalysisResult):
    """One verified scalar Direct quantity and its contract-defined evidence."""

    root: Quantity | None = None
    frequency: Quantity | None = None
    linewidth: Quantity | None = None
    slope: Quantity | None = None
    value: Quantity | None = None
    magnitude: Quantity | None = None
    real: Quantity | None = None
    imag: Quantity | None = None
    zero: Quantity | None = None
    numerator_slope: Quantity | None = None
    denominator: Quantity | None = None
    coupling: Quantity | None = None
    branch_a_residue: Quantity | None = None
    branch_b_residue: Quantity | None = None
    evaluation_omega: Quantity | None = None
    family: Literal["S", "Y", "Z"] | None = None
    evidence: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)
    _presentation: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)
    _quantity_fields = frozenset({
        "root", "frequency", "linewidth", "slope", "value", "magnitude",
        "real", "imag", "zero", "numerator_slope", "denominator", "coupling",
        "branch_a_residue", "branch_b_residue",
        "evaluation_omega",
    })

    def __init__(self) -> None:
        unavailable(f"{type(self).__name__} construction")

    def __getattribute__(self, name: str) -> object:
        if name == "evidence":
            return _fresh_evidence_attribute(object.__getattribute__(self, name))
        return _fresh_quantity_attribute(self, name)

    @property
    def electrical_resolution_envelope_exceeded(self) -> tuple[tuple[str, ...], ...]:
        """Policy lines whose saved complex root exceeds |ω|/(2π) ≤ fmax."""

        root = self.root if self.root is not None else self.zero
        if root is None or self.discretization is None:
            return ()
        frequency = abs(complex(root.to("radian / second").magnitude)) / (2 * math.pi)
        return tuple(
            row.component_path for row in self.discretization
            if row.kind == "electrical_resolution"
            and row.policy is not None
            and frequency > float64_from_hex(row.policy["max_frequency"]["si_value_f64"])
        )

    def plot(self, *, theme: Theme = Theme.AUTO, detailed: bool = False) -> Figure:
        from ..visualization.plots.numerical import scalar_plot

        return scalar_plot(self, theme=theme, detailed=detailed)

    def show(self, *, theme: Theme = Theme.AUTO, detailed: bool = False) -> HtmlPresentation:
        from ..visualization.quantity_html import scalar_show

        return scalar_show(self, theme=theme, detailed=detailed)

    def add_to(self, fig: Figure, *, row: int, col: int) -> Figure:
        from ..visualization.plots.numerical import scalar_add_to

        return scalar_add_to(self, fig, row=row, col=col)


def _fresh_evidence_attribute(value: object) -> object:
    """Detach nested Pint wrappers while retaining immutable result backing."""

    if isinstance(value, Quantity):
        return quantity_view(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _fresh_evidence_attribute(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_fresh_evidence_attribute(item) for item in value)
    return value

@dataclass(frozen=True, slots=True)
class DiagonalRootResult(DirectQuantityResult):
    """Loaded root and local slope evidence from one diagonal-root request."""

    root: Quantity
    frequency: Quantity
    linewidth: Quantity
    slope: Quantity

    def __init__(self) -> None:
        unavailable("DiagonalRootResult construction")

@dataclass(frozen=True, slots=True, kw_only=True)
class OperatorElementRootResult(DirectQuantityResult):
    """Verified locally simple root of one ordered selected-View element."""

    root: Quantity
    frequency: Quantity
    slope: Quantity

    def __init__(self) -> None:
        unavailable("OperatorElementRootResult construction")

@dataclass(frozen=True, slots=True)
class OperatorPointResult:
    """One verified labeled selected-network operator at one frequency."""

    frequency: Quantity
    matrix: Quantity
    coordinates: tuple[str, ...]
    _quantity_fields = frozenset({"frequency", "matrix"})

    def __init__(self) -> None:
        unavailable("OperatorPointResult construction")

    def __getattribute__(self, name: str) -> object:
        return _fresh_quantity_attribute(self, name)

@dataclass(frozen=True, slots=True)
class OperatorResult(AnalysisResult):
    """Verified selected-network operator points in declared frequency order."""

    points: tuple[OperatorPointResult, ...]

    def __init__(self) -> None:
        unavailable("OperatorResult construction")

    def at(self, frequency: Quantity) -> OperatorPointResult:
        """Return the already materialized point at exactly ``frequency``."""

        for point in self.points:
            if point.frequency == frequency:
                return point
        raise KeyError("frequency was not materialized")

    def plot(
        self,
        *,
        frequency: Quantity,
        kind: Literal["table", "heatmap"] = "table",
        component: Literal["magnitude", "phase", "real", "imag"] | None = None,
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        from ..visualization.plots.numerical import operator_plot

        return operator_plot(self, frequency=frequency, kind=kind, component=component, theme=theme)

    def show(self, **presentation: object) -> None:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        row: int,
        col: int,
        frequency: Quantity,
        kind: Literal["table", "heatmap"],
        component: Literal["magnitude", "phase", "real", "imag"] | None = None,
    ) -> Figure:
        from ..visualization.plots.numerical import operator_add_to

        return operator_add_to(
            self, fig, row=row, col=col, frequency=frequency, kind=kind, component=component
        )
