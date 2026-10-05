from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from html import escape
from typing import TYPE_CHECKING, Literal
from types import MappingProxyType

import numpy as np
from pint import Quantity

from .. import units
from ..construction import unavailable
from ..value_storage import immutable_array, immutable_quantity, quantity_view
from ..visualization.presentation import Theme

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

def _freeze(value: object) -> object:
    """Detach mutable decoder payloads before exposing a Result surface."""

    if isinstance(value, np.ndarray):
        return immutable_array(value)
    if isinstance(value, Quantity):
        if value._REGISTRY is not units.registry:
            raise TypeError("result quantities must use scnsim.units")
        return immutable_quantity(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value

def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value

def _fresh_quantity_attribute(instance: object, name: str) -> object:
    """Expose fresh Pint wrappers while retaining one immutable backing."""

    value = object.__getattribute__(instance, name)
    fields = object.__getattribute__(instance, "_quantity_fields")
    if name in fields and isinstance(value, Quantity):
        return quantity_view(value)
    return value

_VERIFIED_TOKEN = object()

def _is_verified_identity(value: object) -> bool:
    return type(value) is ResultIdentity and getattr(value, "_verified_identity_token", None) is _VERIFIED_TOKEN

def _valid_source_index(value: object) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return value >= 0
    return (
        isinstance(value, tuple)
        and bool(value)
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value)
    )

def _is_verified_point_identity(value: object) -> bool:
    return type(value) is ParameterPointIdentity and getattr(value, "_verified_point_identity_token", None) is _VERIFIED_TOKEN

def _is_verified_result_identity(value: object) -> bool:
    return _is_verified_identity(value) or _is_verified_point_identity(value)

@dataclass(frozen=True, slots=True)
class HtmlPresentation:
    """Small self-contained HTML display object for notebook and headless use."""

    html: str

    def _repr_html_(self) -> str:
        return self.html

    def __str__(self) -> str:
        return self.html

class BiasState(Enum):
    OFF = "off"
    ON = "on"

class PumpState(Enum):
    OFF = "off"
    ON = "on"

@dataclass(frozen=True, slots=True)
class Result:
    """Base role shared by immutable, already-materialized SCNSim values."""

    def __init__(self) -> None:
        unavailable(f"{type(self).__name__} construction")

    def show(self, **presentation: object) -> object:
        return HtmlPresentation(f"<pre>{escape(repr(self))}</pre>")

@dataclass(frozen=True, slots=True)
class ResultIdentity:
    """Immutable Plan/request/attempt/result hashes from one verified receipt."""

    plan_sha256: str
    request_sha256: str
    attempt_sha256: str
    result_sha256: str
    _verified_identity_token: object = field(init=False, repr=False, compare=False)

    def __init__(self) -> None:
        unavailable("ResultIdentity construction")

@dataclass(frozen=True, slots=True)
class ParameterPointIdentity:
    """Identity derived from one verified batch without a fictitious receipt."""

    batch: ResultIdentity
    source_index: int | tuple[int, ...]
    parameters_sha256: str
    _verified_point_identity_token: object = field(init=False, repr=False, compare=False)

    def __init__(self) -> None:
        unavailable("ParameterPointIdentity construction")

@dataclass(frozen=True, slots=True)
class LineDiscretization:
    """One verified physical line grid, independent of selected View."""

    component_path: tuple[str, ...]
    kind: Literal["fixed_count", "electrical_resolution"]
    length: Quantity
    n_sections: int
    dx: Quantity
    modal_velocities: tuple[Quantity, ...] = ()
    hmax: Quantity | None = None
    policy: Mapping[str, object] | None = None
    _quantity_fields = frozenset({"length", "dx", "hmax"})

    def __post_init__(self) -> None:
        for name in ("length", "dx", "hmax"):
            value = object.__getattribute__(self, name)
            if value is not None:
                object.__setattr__(self, name, immutable_quantity(value))
        object.__setattr__(self, "modal_velocities", tuple(
            immutable_quantity(value) for value in object.__getattribute__(self, "modal_velocities")
        ))
        if self.policy is not None:
            object.__setattr__(self, "policy", _freeze(self.policy))

    def __getattribute__(self, name: str) -> object:
        if name == "modal_velocities":
            return tuple(quantity_view(value) for value in object.__getattribute__(self, name))
        return _fresh_quantity_attribute(self, name)

@dataclass(frozen=True, slots=True)
class AnalysisResult(Result):
    """Receipt-backed terminal Result returned by solve, evaluate, or optimize."""

    identity: ResultIdentity | ParameterPointIdentity
    discretization: tuple[LineDiscretization, ...] | None = field(default=None, kw_only=True)
    _verified_result_token: object = field(init=False, repr=False, compare=False)

    def __init__(self) -> None:
        unavailable(f"{type(self).__name__} construction")

@dataclass(frozen=True, slots=True)
class MatrixView:
    """Labeled selected-network matrix data with immutable array payloads."""

    matrix: Quantity
    frequencies: Quantity
    coordinates: tuple[str, ...]
    input_channels: tuple[tuple[str, tuple[int, ...]], ...] = ()
    output_channels: tuple[tuple[str, tuple[int, ...]], ...] = ()
    probe_loads: Mapping[str, Literal["raw", "compensated"]] = field(default_factory=dict)
    _quantity_fields = frozenset({"matrix", "frequencies"})

    def __init__(self) -> None:
        unavailable("MatrixView construction")

    def __getattribute__(self, name: str) -> object:
        return _fresh_quantity_attribute(self, name)

@dataclass(frozen=True, slots=True)
class MatrixFamilyResult(Result):
    """One typed matrix family on an immutable selected-network View."""

    view: MatrixView
    _parent_identity: ResultIdentity | ParameterPointIdentity | None = field(
        default=None, repr=False, compare=False
    )
    _presentation: Mapping[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __init__(self) -> None:
        unavailable(f"{type(self).__name__} construction")

    def plot(
        self,
        *,
        input_channel: str | tuple[str, tuple[int, ...]] | None = None,
        output_channel: str | tuple[str, tuple[int, ...]] | None = None,
        kind: Literal["response", "table", "heatmap"] = "response",
        frequency: Quantity | None = None,
        component: Literal["magnitude", "phase", "real", "imag"] | None = None,
        magnitude: Literal["linear", "db"] = "linear",
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        """Build a detached native Plotly Figure from this stored matrix."""

        from ..visualization.plots.numerical import matrix_plot

        return matrix_plot(
            self,
            input_channel=input_channel,
            output_channel=output_channel,
            kind=kind,
            frequency=frequency,
            component=component,
            magnitude=magnitude,
            theme=theme,
        )

    def show(self, **presentation: object) -> None:
        """Display this stored matrix once."""

        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        row: int,
        col: int,
        kind: Literal["trace", "table", "heatmap"],
        input_channel: str | tuple[str, tuple[int, ...]] | None = None,
        output_channel: str | tuple[str, tuple[int, ...]] | None = None,
        frequency: Quantity | None = None,
        component: Literal["magnitude", "phase", "real", "imag"] | None = None,
        magnitude: Literal["linear", "db"] = "linear",
    ) -> Figure:
        """Insert one explicit matrix presentation into a caller subplot."""

        from ..visualization.plots.numerical import matrix_add_to

        return matrix_add_to(
            self,
            fig,
            row=row,
            col=col,
            kind=kind,
            input_channel=input_channel,
            output_channel=output_channel,
            frequency=frequency,
            component=component,
            magnitude=magnitude,
        )
