from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

from pint import Quantity

from ..authoring import ParameterRef, ParameterSet
from ..canonical import canonical_json_bytes
from ..construction import unavailable
from ..errors import SCNSimError
from ..value_storage import quantity_view
from ..visualization.presentation import Theme
from .base import (
    AnalysisResult, ParameterPointIdentity, Result, ResultIdentity, _VERIFIED_TOKEN,
    _freeze, _is_verified_identity, _is_verified_point_identity, _valid_source_index,
)

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

class ParameterPointOutcome(Result):
    """One success or typed numerical non-success inside a verified batch."""

    __slots__ = ("parameters", "source_index", "identity", "_result", "_failure")

    def __init__(self) -> None:
        unavailable("ParameterPointOutcome construction")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ParameterPointOutcome values are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ParameterPointOutcome values are immutable")

    @property
    def succeeded(self) -> bool:
        return self._failure is None

    @property
    def result(self) -> AnalysisResult:
        if self._failure is not None:
            raise self._failure
        result = self._result() if callable(self._result) else self._result
        from .factory import _is_verified_analysis_result

        if not _is_verified_analysis_result(result) or result.identity is not self.identity:
            raise TypeError("parameter point Result has the wrong derived identity")
        return result

    @property
    def failure(self) -> SCNSimError | None:
        return self._failure

class ParameterPointAccessor(Sequence[ParameterPointOutcome]):
    """Read-only lazy point accessor retaining source-space indexing."""

    __slots__ = (
        "_loader", "_count", "_kind", "_shape", "_axis_parameters", "_ordinals"
    )

    def __init__(self) -> None:
        unavailable("ParameterPointAccessor construction")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ParameterPointAccessor values are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ParameterPointAccessor values are immutable")

    def __len__(self) -> int:
        return len(self._ordinals) if self._ordinals is not None else self._count

    def _ordinal(self, index: object) -> int:
        if self._ordinals is not None:
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("selection point indices must be integers")
            position = index + len(self._ordinals) if index < 0 else index
            if position < 0 or position >= len(self._ordinals):
                raise IndexError("parameter point index is out of range")
            return self._ordinals[position]
        if self._kind == "grid":
            if len(self._shape) == 1 and isinstance(index, int) and not isinstance(index, bool):
                multi = (index,)
            elif isinstance(index, tuple):
                multi = index
            else:
                raise TypeError("grid point indices must match the Cartesian rank")
            if len(multi) != len(self._shape) or any(not isinstance(item, int) or isinstance(item, bool) for item in multi):
                raise TypeError("grid point indices must be integer tuples")
            ordinal = 0
            for item, size in zip(multi, self._shape):
                resolved = item + size if item < 0 else item
                if resolved < 0 or resolved >= size:
                    raise IndexError("parameter grid index is out of range")
                ordinal = ordinal * size + resolved
            return ordinal
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("listed point indices must be integers")
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError("parameter point index is out of range")
        return ordinal

    def __getitem__(self, index: object) -> ParameterPointOutcome:
        return self._loader(self._ordinal(index))

    def __iter__(self) -> Iterator[ParameterPointOutcome]:
        ordinals = range(self._count) if self._ordinals is None else self._ordinals
        for ordinal in ordinals:
            yield self._loader(ordinal)

class ParameterField(Result):
    """Masked scalar samples retaining exact point and parameter identities."""

    __slots__ = ("_samples", "_kind", "_shape", "_axis_parameters", "quantity")

    def __init__(self) -> None:
        unavailable("ParameterField construction")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ParameterField values are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ParameterField values are immutable")

    @property
    def samples(self) -> tuple[Mapping[str, object], ...]:
        # Return fresh Pint wrappers over immutable retained backing; a large
        # field is not copied on every read.
        return tuple(
            MappingProxyType(
                {**sample, "value": _detached_quantity(sample["value"])}
            )
            for sample in self._samples
        )

    def plot(
        self,
        *,
        x: ParameterRef,
        y: ParameterRef | None = None,
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        from ..visualization.plots.parameters import parameter_field_plot

        return parameter_field_plot(self, x=x, y=y, theme=theme)

    def show(self, **presentation: object) -> None:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        row: int,
        col: int,
        x: ParameterRef,
        y: ParameterRef | None = None,
    ) -> Figure:
        from ..visualization.plots.parameters import parameter_field_add_to

        return parameter_field_add_to(self, fig, row=row, col=col, x=x, y=y)

class ParameterSweepSelection(Result):
    __slots__ = ("_parent", "points")

    def __init__(self) -> None:
        unavailable("ParameterSweepSelection construction")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ParameterSweepSelection values are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ParameterSweepSelection values are immutable")

    def collect(self, *, quantity: object) -> ParameterField:
        return self._parent._collect(quantity, self.points)

    def plot(
        self,
        *,
        quantity: object | None = None,
        x: ParameterRef | None = None,
        y: ParameterRef | None = None,
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        from ..visualization.plots.parameters import points_plot

        if quantity is None:
            if x is not None or y is not None:
                raise ValueError("x and y require an explicitly collected quantity")
            return points_plot(self.points, theme=theme, title="Selected parameter sweep outcomes")
        if x is None:
            raise ValueError("x is required when plotting a collected quantity")
        return self.collect(quantity=quantity).plot(x=x, y=y, theme=theme)

    def show(self, **presentation: object) -> None:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        row: int,
        col: int,
        quantity: object | None = None,
        x: ParameterRef | None = None,
        y: ParameterRef | None = None,
    ) -> Figure:
        if quantity is None:
            if x is not None or y is not None:
                raise ValueError("x and y require an explicitly collected quantity")
            from ..visualization.plots.parameters import points_add_to

            return points_add_to(self.points, fig, row=row, col=col)
        if x is None:
            raise ValueError("x is required when adding a collected quantity")
        return self.collect(quantity=quantity).add_to(fig, row=row, col=col, x=x, y=y)

class ParameterSweepResult(AnalysisResult):
    """One receipt-backed ordered parameter batch with lazy point payloads."""

    __slots__ = ("identity", "points", "_selector_encoder", "_allowed_selectors", "_verified_result_token")

    def __init__(self) -> None:
        unavailable("ParameterSweepResult construction")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ParameterSweepResult values are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ParameterSweepResult values are immutable")

    def select(self, *, parameters: ParameterSet) -> ParameterSweepSelection:
        if not isinstance(parameters, ParameterSet):
            raise TypeError("parameters must be a ParameterSet")
        if parameters.values:
            if not len(self.points):
                raise ValueError("selection contains a parameter outside this sweep")
            available = self.points._loader(0).parameters.values
            for parameter in parameters.values:
                match = next((current for current in available if current == parameter), None)
                if match is None or match._definition_record() != parameter._definition_record():
                    raise ValueError("selection contains a parameter outside this sweep")
        ordinals: list[int] = []
        for ordinal, point in enumerate(self.points):
            available = point.parameters.values
            for parameter, wanted in parameters.values.items():
                matches = next((current for current in available if current == parameter), None)
                if matches is None or matches._definition_record() != parameter._definition_record():
                    raise TypeError("verified parameter batch has inconsistent definitions")
                if _parameter_value_bytes(available[matches], matches) != _parameter_value_bytes(wanted, parameter):
                    break
            else:
                ordinals.append(ordinal)
        selection = object.__new__(ParameterSweepSelection)
        object.__setattr__(selection, "_parent", self)
        object.__setattr__(selection, "points", _point_accessor(
            self.points._loader,
            self.points._count,
            self.points._kind,
            self.points._shape,
            self.points._axis_parameters,
            tuple(ordinals),
        ))
        return selection

    def collect(self, *, quantity: object) -> ParameterField:
        return self._collect(quantity, self.points)

    def _collect(self, quantity: object, points: Sequence[ParameterPointOutcome]) -> ParameterField:
        if getattr(quantity, "_view", None) is not None:
            raise ValueError("View-bound selectors are supported only by optimization")
        key = self._selector_encoder(quantity)
        if key not in self._allowed_selectors:
            raise ValueError("quantity was not requested by this sweep")
        projection = getattr(quantity, "projection", None)
        if not isinstance(projection, str):
            raise TypeError("quantity must be a QuantitySelector")
        samples: list[Mapping[str, object]] = []
        for point in points:
            value = None if not point.succeeded else getattr(point.result, projection, None)
            if point.succeeded and not isinstance(value, Quantity):
                raise ValueError("requested quantity is absent from the point Result family")
            samples.append(MappingProxyType({
                "parameters": point.parameters,
                "source_index": point.source_index,
                "identity": point.identity,
                "value": _freeze(value),
                "failure": point.failure,
            }))
        result = object.__new__(ParameterField)
        object.__setattr__(result, "_samples", tuple(samples))
        object.__setattr__(result, "_kind", points._kind)
        object.__setattr__(result, "_shape", points._shape)
        object.__setattr__(result, "_axis_parameters", points._axis_parameters)
        object.__setattr__(result, "quantity", quantity)
        return result

    def plot(
        self,
        *,
        quantity: object | None = None,
        x: ParameterRef | None = None,
        y: ParameterRef | None = None,
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        from ..visualization.plots.parameters import points_plot

        if quantity is None:
            if x is not None or y is not None:
                raise ValueError("x and y require an explicitly collected quantity")
            return points_plot(self.points, theme=theme)
        if x is None:
            raise ValueError("x is required when plotting a collected quantity")
        return self.collect(quantity=quantity).plot(x=x, y=y, theme=theme)

    def show(self, **presentation: object) -> None:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        row: int,
        col: int,
        quantity: object | None = None,
        x: ParameterRef | None = None,
        y: ParameterRef | None = None,
    ) -> Figure:
        if quantity is None:
            if x is not None or y is not None:
                raise ValueError("x and y require an explicitly collected quantity")
            from ..visualization.plots.parameters import points_add_to

            return points_add_to(self.points, fig, row=row, col=col)
        if x is None:
            raise ValueError("x is required when adding a collected quantity")
        return self.collect(quantity=quantity).add_to(fig, row=row, col=col, x=x, y=y)

def _parameter_value_bytes(value: object, parameter: ParameterRef) -> bytes:
    from ..canonical import canonical_json_bytes

    record = ParameterSet({parameter: value})._record()["bindings"][0]["value"]
    return canonical_json_bytes(record)

def _detached_quantity(value: object) -> object:
    """Return a cheap safe wrapper around an immutable stored Quantity."""

    if not isinstance(value, Quantity):
        return value
    return quantity_view(value)

def _point_accessor(
    loader: Callable[[int], ParameterPointOutcome],
    count: int,
    kind: str,
    shape: tuple[int, ...],
    axis_parameters: tuple[ParameterRef, ...],
    ordinals: tuple[int, ...] | None = None,
) -> ParameterPointAccessor:
    result = object.__new__(ParameterPointAccessor)
    object.__setattr__(result, "_loader", loader)
    object.__setattr__(result, "_count", count)
    object.__setattr__(result, "_kind", kind)
    object.__setattr__(result, "_shape", shape)
    object.__setattr__(result, "_axis_parameters", axis_parameters)
    object.__setattr__(result, "_ordinals", ordinals)
    return result

def _point_outcome(
    *,
    parameters: ParameterSet,
    source_index: int | tuple[int, ...],
    identity: ParameterPointIdentity,
    result: AnalysisResult | Callable[[], AnalysisResult] | None,
    failure: SCNSimError | None,
) -> ParameterPointOutcome:
    if not isinstance(parameters, ParameterSet) or not _valid_source_index(source_index) or not _is_verified_point_identity(identity):
        raise TypeError("parameter point evidence is malformed")
    if (result is None) == (failure is None):
        raise TypeError("parameter point must contain exactly one result or failure")
    from .factory import _is_verified_analysis_result

    if result is not None and not callable(result) and (
        not _is_verified_analysis_result(result) or result.identity is not identity
    ):
        raise TypeError("parameter point Result has the wrong derived identity")
    if failure is not None and not isinstance(failure, SCNSimError):
        raise TypeError("parameter point failure must be typed")
    outcome = object.__new__(ParameterPointOutcome)
    object.__setattr__(outcome, "parameters", parameters)
    object.__setattr__(outcome, "source_index", source_index)
    object.__setattr__(outcome, "identity", identity)
    object.__setattr__(outcome, "_result", result)
    object.__setattr__(outcome, "_failure", failure)
    return outcome

def _parameter_sweep_result(
    *,
    identity: ResultIdentity,
    points: ParameterPointAccessor,
    selector_encoder: Callable[[object], bytes],
    allowed_selectors: Sequence[bytes],
) -> ParameterSweepResult:
    if not _is_verified_identity(identity) or not isinstance(points, ParameterPointAccessor):
        raise TypeError("parameter sweep identity or accessor is unverified")
    result = object.__new__(ParameterSweepResult)
    object.__setattr__(result, "identity", identity)
    object.__setattr__(result, "discretization", None)
    object.__setattr__(result, "points", points)
    object.__setattr__(result, "_selector_encoder", selector_encoder)
    object.__setattr__(result, "_allowed_selectors", frozenset(allowed_selectors))
    object.__setattr__(result, "_verified_result_token", _VERIFIED_TOKEN)
    return result
