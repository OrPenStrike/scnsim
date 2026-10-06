from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, field
from typing import TYPE_CHECKING, Literal

from pint import Quantity

from ..construction import unavailable
from ..errors import HBCaseFailure
from ..value_storage import quantity_view
from ..visualization.presentation import Theme
from .base import AnalysisResult, Result, _freeze

if TYPE_CHECKING:
    from plotly.graph_objects import Figure
    from .base import BiasState, PumpState
    from .derived import TraceResult
    from .matrix import HBScatteringMatrixResult, MatrixFamilyResult

class HBCaseOutcome(Result):
    """One named success or receipt-backed numerical HB failure.

    ``states`` and ``state_node_map`` are operating-point evidence only, never
    View coordinates or current-drive targets.
    """

    __slots__ = (
        "_id", "_failure", "_effective_sources", "_operating_point_closure", "_bias_state", "_pump_state", "_s", "_y", "_z",
        "_traces", "_states", "_state_node_map",
    )

    def __init__(self) -> None:
        unavailable("HBCaseOutcome construction")

    def __setattr__(self, name: str, value: object) -> None:
        raise FrozenInstanceError("cannot assign to field of immutable HBCaseOutcome")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError("cannot delete field of immutable HBCaseOutcome")

    @property
    def id(self) -> str:
        return self._id

    @property
    def succeeded(self) -> bool:
        return self._failure is None

    @property
    def failure(self) -> HBCaseFailure | None:
        return self._failure

    @property
    def effective_sources(self) -> tuple[Mapping[str, object], ...]:
        """Return the exact case drive evidence, including on failed cases."""

        return tuple(_freeze(source) for source in self._effective_sources)

    @property
    def operating_point_closure(self) -> Mapping[str, object]:
        """Return the verified nonlinear closure evidence for a successful case."""

        return self._success(_freeze(self._operating_point_closure))  # type: ignore[return-value]

    def _success(self, value: object) -> object:
        if self._failure is not None:
            raise self._failure
        return value

    @property
    def bias_state(self) -> BiasState:
        return self._success(self._bias_state)  # type: ignore[return-value]

    @property
    def pump_state(self) -> PumpState:
        return self._success(self._pump_state)  # type: ignore[return-value]

    @property
    def s(self) -> HBScatteringMatrixResult:
        return self._success(self._s)  # type: ignore[return-value]

    @property
    def y(self) -> MatrixFamilyResult:
        return self._success(self._y)  # type: ignore[return-value]

    @property
    def z(self) -> MatrixFamilyResult:
        return self._success(self._z)  # type: ignore[return-value]

    @property
    def traces(self) -> Mapping[str, TraceResult]:
        return self._success(self._traces)  # type: ignore[return-value]

    @property
    def states(self) -> Quantity:
        states = self._success(self._states)
        return quantity_view(states)  # type: ignore[arg-type, return-value]

    @property
    def state_node_map(self) -> tuple[Mapping[str, object], ...]:
        return self._success(self._state_node_map)  # type: ignore[return-value]

    def plot(self, **presentation: object) -> Figure:
        from ..visualization.plots.hb import hb_case_plot

        return hb_case_plot(self, **presentation)

    def show(self, **presentation: object) -> None:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(self, fig: Figure, *, row: int, col: int, **presentation: object) -> Figure:
        from ..visualization.plots.hb import hb_case_add_to

        return hb_case_add_to(self, fig, row=row, col=col, **presentation)

@dataclass(frozen=True, slots=True)
class HBBatchResult(AnalysisResult):
    cases: Mapping[str, HBCaseOutcome]
    topology_evidence: Mapping[str, object]
    _presentation: Mapping[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __init__(self) -> None:
        unavailable("HBBatchResult construction")

    def plot(
        self,
        *,
        trace: str | None = None,
        input_channel: str | tuple[str, tuple[int, ...]] | None = None,
        output_channel: str | tuple[str, tuple[int, ...]] | None = None,
        component: Literal["magnitude", "phase", "real", "imag"] | None = None,
        magnitude: Literal["linear", "db"] = "linear",
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        from ..visualization.plots.hb import hb_batch_plot

        return hb_batch_plot(
            self,
            trace=trace,
            input_channel=input_channel,
            output_channel=output_channel,
            component=component,
            magnitude=magnitude,
            theme=theme,
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
        kind: Literal["trace", "status"],
        trace: str | None = None,
        input_channel: str | tuple[str, tuple[int, ...]] | None = None,
        output_channel: str | tuple[str, tuple[int, ...]] | None = None,
        component: Literal["magnitude", "phase", "real", "imag"] | None = None,
        magnitude: Literal["linear", "db"] = "linear",
    ) -> Figure:
        from ..visualization.plots.hb import hb_batch_add_to

        return hb_batch_add_to(
            self,
            fig,
            row=row,
            col=col,
            kind=kind,
            trace=trace,
            input_channel=input_channel,
            output_channel=output_channel,
            component=component,
            magnitude=magnitude,
        )
