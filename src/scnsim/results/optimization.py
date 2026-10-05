from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ..authoring import ParameterRef, ParameterSet
from ..construction import unavailable
from ..visualization.presentation import Theme
from .base import AnalysisResult, LineDiscretization

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

@dataclass(frozen=True, slots=True)
class OptimizationBest:
    """Lowest finite-cost baseline or population candidate in ledger order."""

    parameters: ParameterSet
    cost: float
    discretization: tuple[LineDiscretization, ...] | None = None

    def __init__(self) -> None:
        unavailable("OptimizationBest construction")

@dataclass(frozen=True, slots=True)
class OptimizationResult(AnalysisResult):
    """Verified CMA winner and immutable completed-generation ledgers."""

    best: OptimizationBest
    ledger: tuple[Mapping[str, object], ...] = ()
    candidate_discretization: tuple[tuple[LineDiscretization, ...] | None, ...] = ()
    _presentation: Mapping[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __init__(self) -> None:
        unavailable("OptimizationResult construction")

    def plot(
        self,
        *,
        kind: Literal["history", "objective", "residual", "parameter", "table", "comparison"] = "history",
        objective: str | None = None,
        parameter: ParameterRef | None = None,
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        from ..visualization.plots.optimization import optimization_plot

        return optimization_plot(
            self, kind=kind, objective=objective, parameter=parameter, theme=theme
        )

    def show(self, **presentation: object) -> None:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        row: int,
        col: int,
        kind: Literal["history", "objective", "residual", "parameter", "table", "comparison"],
        objective: str | None = None,
        parameter: ParameterRef | None = None,
    ) -> Figure:
        from ..visualization.plots.optimization import optimization_add_to

        return optimization_add_to(
            self,
            fig,
            row=row,
            col=col,
            kind=kind,
            objective=objective,
            parameter=parameter,
        )
