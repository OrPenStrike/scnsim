"""Receipt-backed results and downstream presentation values.

Public constructors deliberately fail: a user cannot manufacture a result that
looks like verified workspace evidence.  ``_verified_result`` is the narrow
decoder hook used after workspace receipt/artifact verification.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import MISSING, FrozenInstanceError, dataclass, field, fields
from enum import Enum
from html import escape
import json
from os import O_RDONLY, PathLike, fsync, link, open as os_open
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from pint import Quantity

from . import units
from ._immutable_values import immutable_array, immutable_quantity, quantity_view
from ._scaffold import unavailable
from .authoring import ParameterRef, ParameterSet
from .errors import HBCaseFailure, SCNSimError
from .presentation import Theme

if TYPE_CHECKING:
    import schemdraw
    from plotly.graph_objects import Figure

    from .composition import SchematicCompositionSnapshot


T = TypeVar("T")


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


def _verified_result(cls: type[T], /, **values: object) -> T:
    """Private verified-decoder hook; never call this on unverified evidence.

    The caller supplies exactly the public dataclass fields for ``cls``.  The
    hook validates receipt identity fields and recursively detaches mappings,
    sequences, and NumPy/Pint arrays before the value becomes user-visible.
    """

    if not isinstance(cls, type) or not issubclass(cls, (Result, MatrixView, ResultIdentity, ParameterPointIdentity, ReconciliationEvidence, OptimizationBest, OperatorPointResult)):
        raise TypeError("_verified_result only constructs SCNSim result values")
    if cls is HBCaseOutcome:
        expected = {
            "id", "failure", "effective_sources", "operating_point_closure", "bias_state", "pump_state", "s", "y", "z", "traces", "states", "state_node_map",
        }
        if set(values) != expected:
            raise TypeError("verified HBCaseOutcome fields mismatch")
        failure = values["failure"]
        success = failure is None
        if not isinstance(values["id"], str) or not values["id"]:
            raise ValueError("HB case id must be nonempty")
        if failure is not None and not isinstance(failure, HBCaseFailure):
            raise TypeError("HB failure must be HBCaseFailure")
        required = ("operating_point_closure", "bias_state", "pump_state", "s", "y", "z", "traces", "states", "state_node_map")
        if success != all(values[name] is not None for name in required):
            raise ValueError("HB success must provide every success-only surface")
        if not success and any(values[name] is not None for name in required):
            raise ValueError("HB failure must not retain success-only surfaces")
        if not isinstance(values["effective_sources"], (tuple, list)):
            raise TypeError("HB outcome effective_sources must be an ordered sequence")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_id", values["id"])
        object.__setattr__(instance, "_failure", failure)
        for name in ("effective_sources", "operating_point_closure", "bias_state", "pump_state", "s", "y", "z", "traces", "states", "state_node_map"):
            object.__setattr__(instance, f"_{name}", _freeze(values[name]))
        return instance
    if cls is HBBatchResult:
        if set(values) != {"identity", "cases", "topology_evidence", "_presentation"}:
            raise TypeError("verified HBBatchResult fields mismatch")
        identity, cases = values["identity"], values["cases"]
        if not _is_verified_result_identity(identity) or not isinstance(cases, Mapping) or not cases or not isinstance(values["topology_evidence"], Mapping):
            raise TypeError("verified HBBatchResult requires identity and nonempty cases")
        materialized = dict(cases)
        if any(
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(outcome, HBCaseOutcome)
            or outcome.id != identifier
            for identifier, outcome in materialized.items()
        ):
            raise TypeError("verified HBBatchResult cases are malformed")
        instance = object.__new__(cls)
        object.__setattr__(instance, "identity", identity)
        object.__setattr__(instance, "cases", MappingProxyType(materialized))
        object.__setattr__(instance, "topology_evidence", _freeze(values["topology_evidence"]))
        object.__setattr__(instance, "_presentation", _freeze(values["_presentation"]))
        object.__setattr__(instance, "_verified_result_token", _VERIFIED_TOKEN)
        return instance
    expected = {item.name: item for item in fields(cls) if item.init}
    missing = set(expected) - set(values)
    extra = set(values) - set(expected)
    required_missing = {
        name for name in missing
        if expected[name].default is MISSING and expected[name].default_factory is MISSING
    }
    if required_missing or extra:
        raise TypeError(f"verified {cls.__name__} fields mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for name in missing:
        descriptor = expected[name]
        values[name] = descriptor.default_factory() if descriptor.default_factory is not MISSING else descriptor.default
    if cls is ReportResult:
        html = values.get("html")
        inputs = values.get("inputs")
        presentation_sha256 = values.get("presentation_sha256")
        if not isinstance(html, str) or not html:
            raise TypeError("verified ReportResult requires nonempty HTML")
        if (
            not isinstance(inputs, (tuple, list))
            or not inputs
            or not all(_is_verified_analysis_result(item) for item in inputs)
        ):
            raise TypeError("verified ReportResult requires nonempty verified inputs")
        _sha256(presentation_sha256, name="presentation_sha256")
        if presentation_sha256 not in html:
            raise ValueError("ReportResult HTML must contain its presentation identity")
    if cls is ResultIdentity:
        for name, value in values.items():
            _sha256(value, name=name)
    if cls is ParameterPointIdentity:
        batch = values.get("batch")
        source_index = values.get("source_index")
        if not _is_verified_identity(batch) or not _valid_source_index(source_index):
            raise TypeError("parameter-point identity fields are invalid")
        _sha256(values.get("parameters_sha256"), name="parameters_sha256")
    if issubclass(cls, AnalysisResult) and not _is_verified_result_identity(values.get("identity")):
        raise TypeError("analysis results require a verified batch or point identity")
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, _freeze(value))
    if cls is ResultIdentity:
        object.__setattr__(instance, "_verified_identity_token", _VERIFIED_TOKEN)
    if cls is ParameterPointIdentity:
        object.__setattr__(instance, "_verified_point_identity_token", _VERIFIED_TOKEN)
    if issubclass(cls, AnalysisResult):
        object.__setattr__(instance, "_verified_result_token", _VERIFIED_TOKEN)
    return instance


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


def _is_verified_analysis_result(value: object) -> bool:
    return (
        type(value) in (
            DirectSolveResult,
            DiagonalRootResult,
            DirectQuantityResult,
            OperatorResult,
            OptimizationResult,
            HBBatchResult,
            ParameterSweepResult,
        )
        and getattr(value, "_verified_result_token", None) is _VERIFIED_TOKEN
        and _is_verified_result_identity(getattr(value, "identity", None))
    )


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
class AnalysisResult(Result):
    """Receipt-backed terminal Result returned by solve, evaluate, or optimize."""

    identity: ResultIdentity | ParameterPointIdentity
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

        from ._numeric_presentation import matrix_plot

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

        from ._numeric_presentation import show_figure

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

        from ._numeric_presentation import matrix_add_to

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
        from ._numeric_presentation import show_figure

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
    family: Literal["S", "Y", "Z"] | None = None
    _presentation: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)
    _quantity_fields = frozenset({
        "root", "frequency", "linewidth", "slope", "value", "magnitude",
        "real", "imag", "zero", "numerator_slope", "denominator", "coupling",
        "branch_a_residue", "branch_b_residue",
    })

    def __init__(self) -> None:
        unavailable(f"{type(self).__name__} construction")

    def __getattribute__(self, name: str) -> object:
        return _fresh_quantity_attribute(self, name)

    def plot(self, *, theme: Theme = Theme.AUTO) -> Figure:
        from ._numeric_presentation import scalar_plot

        return scalar_plot(self, theme=theme)

    def show(self, *, theme: Theme = Theme.AUTO) -> None:
        from ._numeric_presentation import show_figure

        return show_figure(self.plot(theme=theme))

    def add_to(self, fig: Figure, *, row: int, col: int) -> Figure:
        from ._numeric_presentation import scalar_add_to

        return scalar_add_to(self, fig, row=row, col=col)


@dataclass(frozen=True, slots=True)
class DiagonalRootResult(DirectQuantityResult):
    """Loaded root and local slope evidence from one diagonal-root request."""

    root: Quantity
    frequency: Quantity
    linewidth: Quantity
    slope: Quantity

    def __init__(self) -> None:
        unavailable("DiagonalRootResult construction")


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
        from ._numeric_presentation import operator_plot

        return operator_plot(self, frequency=frequency, kind=kind, component=component, theme=theme)

    def show(self, **presentation: object) -> None:
        from ._numeric_presentation import show_figure

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
        from ._numeric_presentation import operator_add_to

        return operator_add_to(
            self, fig, row=row, col=col, frequency=frequency, kind=kind, component=component
        )


@dataclass(frozen=True, slots=True)
class OptimizationBest:
    """Lowest finite-cost baseline or population candidate in ledger order."""

    parameters: ParameterSet
    cost: float

    def __init__(self) -> None:
        unavailable("OptimizationBest construction")


@dataclass(frozen=True, slots=True)
class OptimizationResult(AnalysisResult):
    """Verified CMA winner and immutable completed-generation ledgers."""

    best: OptimizationBest
    ledger: tuple[Mapping[str, object], ...] = ()

    def __init__(self) -> None:
        unavailable("OptimizationResult construction")

    def plot(
        self,
        *,
        kind: Literal["history", "objective", "residual", "parameter", "table"] = "history",
        objective: str | None = None,
        parameter: ParameterRef | None = None,
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        from ._numeric_presentation import optimization_plot

        return optimization_plot(
            self, kind=kind, objective=objective, parameter=parameter, theme=theme
        )

    def show(self, **presentation: object) -> None:
        from ._numeric_presentation import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        row: int,
        col: int,
        kind: Literal["history", "objective", "residual", "parameter", "table"],
        objective: str | None = None,
        parameter: ParameterRef | None = None,
    ) -> Figure:
        from ._numeric_presentation import optimization_add_to

        return optimization_add_to(
            self,
            fig,
            row=row,
            col=col,
            kind=kind,
            objective=objective,
            parameter=parameter,
        )


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
        from ._numeric_presentation import hb_case_plot

        return hb_case_plot(self, **presentation)

    def show(self, **presentation: object) -> None:
        from ._numeric_presentation import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(self, fig: Figure, *, row: int, col: int, **presentation: object) -> Figure:
        from ._numeric_presentation import hb_case_add_to

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
        from ._numeric_presentation import hb_batch_plot

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
        from ._numeric_presentation import show_figure

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
        from ._numeric_presentation import hb_batch_add_to

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
        from ._numeric_presentation import parameter_field_plot

        return parameter_field_plot(self, x=x, y=y, theme=theme)

    def show(self, **presentation: object) -> None:
        from ._numeric_presentation import show_figure

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
        from ._numeric_presentation import parameter_field_add_to

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
        from ._numeric_presentation import points_plot

        if quantity is None:
            if x is not None or y is not None:
                raise ValueError("x and y require an explicitly collected quantity")
            return points_plot(self.points, theme=theme, title="Selected parameter sweep outcomes")
        if x is None:
            raise ValueError("x is required when plotting a collected quantity")
        return self.collect(quantity=quantity).plot(x=x, y=y, theme=theme)

    def show(self, **presentation: object) -> None:
        from ._numeric_presentation import show_figure

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
            from ._numeric_presentation import points_add_to

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
        from ._numeric_presentation import points_plot

        if quantity is None:
            if x is not None or y is not None:
                raise ValueError("x and y require an explicitly collected quantity")
            return points_plot(self.points, theme=theme)
        if x is None:
            raise ValueError("x is required when plotting a collected quantity")
        return self.collect(quantity=quantity).plot(x=x, y=y, theme=theme)

    def show(self, **presentation: object) -> None:
        from ._numeric_presentation import show_figure

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
            from ._numeric_presentation import points_add_to

            return points_add_to(self.points, fig, row=row, col=col)
        if x is None:
            raise ValueError("x is required when adding a collected quantity")
        return self.collect(quantity=quantity).add_to(fig, row=row, col=col, x=x, y=y)


def _parameter_value_bytes(value: object, parameter: ParameterRef) -> bytes:
    from ._canonical import canonical_json_bytes

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
    object.__setattr__(result, "points", points)
    object.__setattr__(result, "_selector_encoder", selector_encoder)
    object.__setattr__(result, "_allowed_selectors", frozenset(allowed_selectors))
    object.__setattr__(result, "_verified_result_token", _VERIFIED_TOKEN)
    return result


@dataclass(frozen=True, slots=True)
class TraceResult(Result):
    frequencies: Quantity
    value: Quantity
    _parent_identity: ResultIdentity | ParameterPointIdentity | None = field(
        default=None, repr=False, compare=False
    )
    _presentation: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)
    _quantity_fields = frozenset({"frequencies", "value"})

    def __init__(self) -> None:
        unavailable("TraceResult construction")

    def __getattribute__(self, name: str) -> object:
        return _fresh_quantity_attribute(self, name)

    def plot(
        self,
        *,
        component: Literal["magnitude", "phase", "real", "imag"] | None = None,
        magnitude: Literal["linear", "db"] = "linear",
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        from ._numeric_presentation import trace_plot

        return trace_plot(self, component=component, magnitude=magnitude, theme=theme)

    def show(self, **presentation: object) -> None:
        from ._numeric_presentation import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        row: int,
        col: int,
        component: Literal["magnitude", "phase", "real", "imag"],
        magnitude: Literal["linear", "db"] = "linear",
        name: str | None = None,
    ) -> Figure:
        from ._numeric_presentation import trace_add_to

        return trace_add_to(
            self, fig, row=row, col=col, component=component, magnitude=magnitude, name=name
        )


@dataclass(frozen=True, slots=True)
class ExplanationResult(Result):
    evidence: Mapping[str, object]

    def __init__(self) -> None:
        unavailable("ExplanationResult construction")

    def show(self, **presentation: object) -> HtmlPresentation:
        def table(title: str, headers: tuple[str, ...], rows: object) -> str:
            body = "".join(
                "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
                for row in rows
            )
            head = "".join(f"<th>{escape(header)}</th>" for header in headers)
            return f"<h3>{escape(title)}</h3><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

        evidence = self.evidence
        compiled = evidence.get("compiled", {})
        lineage = evidence.get("ref_lineage", {})
        hierarchy = evidence.get("component_hierarchy", ())
        parameters = evidence.get("parameters", {}).get("bindings", ()) if isinstance(evidence.get("parameters"), Mapping) else ()
        html = table(
            "Identity",
            ("field", "value"),
            ((name, evidence.get(name)) for name in ("plan_sha256", "request_sha256", "runtime_semantic", "spec")),
        )
        html += table(
            "View lineage",
            ("step", "evidence"),
            ((name, lineage.get(name)) for name in ("original", "ptc", "transforms", "retain", "terminal_coordinates", "port_realizable")),
        )
        html += table(
            "Components and parameters",
            ("kind", "identity", "declaration"),
            tuple(("component", item.get("component_path"), item) for item in hierarchy)
            + tuple(("parameter", item.get("parameter"), item.get("value")) for item in parameters),
        )
        if isinstance(compiled, Mapping):
            html += table(
                "Compiler and capability",
                ("field", "value"),
                (
                    ("node_order", compiled.get("node_order")),
                    ("C shape", compiled.get("c_matrix", {}).get("shape")),
                    ("K shape", compiled.get("k_matrix", {}).get("shape")),
                    ("G shape", compiled.get("g_matrix", {}).get("shape")),
                    ("ports", compiled.get("ports")),
                    ("root", compiled.get("root_preflight")),
                    ("optimization", compiled.get("optimization_preflight")),
                    ("Direct / HB", compiled.get("direct_hb_capability")),
                ),
            )
            rows = compiled.get("expanded_branch_rows", ())
            line_rows = tuple(
                row for row in rows
                if isinstance(row, Mapping) and row.get("kind") == "transmission_line_audit"
            )
            if line_rows:
                html += table(
                    "Transmission-line expansion",
                    ("component", "conductors/reference", "sections", "length / dx", "orientation", "stations", "source"),
                    (
                        (
                            row.get("component_path"),
                            (row.get("conductors"), row.get("reference_conductor")),
                            row.get("n_sections"),
                            (row.get("length"), row.get("dx")),
                            row.get("orientation"), row.get("stations"), row.get("rlgc_source"),
                        )
                        for row in line_rows
                    ),
                )
            html += table(
                "Expanded branch rows",
                ("component", "kind", "section", "station/end", "row", "column", "value", "omitted"),
                (
                    (
                        row.get("component_path"), row.get("kind"), row.get("section"),
                        (row.get("station"), row.get("end")), row.get("row_conductor"),
                        row.get("column_conductor"), row.get("value"), row.get("omitted_as_zero"),
                    )
                    for row in rows
                    if isinstance(row, Mapping)
                ),
            )
        return HtmlPresentation(html)


@dataclass(frozen=True, slots=True)
class InventoryResult(Result):
    """Pure read-only evidence inventory; it never selects a result for resolve."""

    requests: tuple[Mapping[str, object], ...]
    maintenance: tuple[Mapping[str, object], ...]

    def __init__(self) -> None:
        unavailable("InventoryResult construction")


@dataclass(frozen=True, slots=True)
class ReportResult(Result):
    html: str
    inputs: tuple[AnalysisResult, ...] = ()
    presentation_sha256: str = ""

    def __init__(self) -> None:
        unavailable("ReportResult construction")

    def show(self, **presentation: object) -> HtmlPresentation:
        return HtmlPresentation(self.html)

    def save(self, path: str | PathLike[str]) -> Path:
        target = Path(path)
        if target.suffix != ".html":
            raise ValueError("report path must end in .html")
        if not target.parent.is_dir():
            raise FileNotFoundError(target.parent)
        if target.exists():
            raise FileExistsError(target)
        with NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as temporary:
            temporary.write(self.html)
            temporary.flush()
            fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            link(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
        directory_fd = os_open(target.parent, O_RDONLY)
        try:
            fsync(directory_fd)
        finally:
            from os import close
            close(directory_fd)
        return target


@dataclass(frozen=True, slots=True)
class CircuitDiagramAudit:
    """Read-only certificate for one frozen, independently checked scene."""

    _data: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        unavailable("CircuitDiagramAudit construction")

    @classmethod
    def _from_data(cls, data: object) -> "CircuitDiagramAudit":
        from ._diagram.audit import DiagramAuditData

        if not isinstance(data, DiagramAuditData):
            raise TypeError("CircuitDiagramAudit requires DiagramAuditData")
        data = DiagramAuditData(
            representation=data.representation,
            plan_id=data.plan_id,
            plan_sha256=data.plan_sha256,
            connectivity_sha256=data.connectivity_sha256,
            semantic_sha256=data.semantic_sha256,
            compiled_graph_sha256=data.compiled_graph_sha256,
            expanded_graph_sha256=data.expanded_graph_sha256,
            presentation_sha256=data.presentation_sha256,
            observed_electrical=_freeze(data.observed_electrical),
            observed_semantic=_freeze(data.observed_semantic),
            observed_rows=_freeze(data.observed_rows),
            verified_rows=_freeze(data.verified_rows),
        )
        result = object.__new__(cls)
        object.__setattr__(result, "_data", data)
        return result

    @property
    def representation(self) -> Literal["authoring", "compiled"]:
        return self._data.representation

    @property
    def plan_id(self) -> str:
        return self._data.plan_id

    @property
    def plan_sha256(self) -> str:
        return self._data.plan_sha256

    @property
    def connectivity_sha256(self) -> str:
        return self._data.connectivity_sha256

    @property
    def semantic_sha256(self) -> str:
        return self._data.semantic_sha256

    @property
    def compiled_graph_sha256(self) -> str | None:
        return self._data.compiled_graph_sha256

    @property
    def expanded_graph_sha256(self) -> str | None:
        return self._data.expanded_graph_sha256

    @property
    def presentation_sha256(self) -> str | None:
        return self._data.presentation_sha256

    def show(self) -> "HtmlPresentation":
        """Present observed scene rows separately from verified snapshot facts."""

        from ._canonical import float64_from_hex

        def text(value: object) -> str:
            """Format certified records without exposing encoder JSON as UI."""

            if value is None:
                return "—"
            if isinstance(value, str):
                if value.startswith("{") and value.endswith("}"):
                    try:
                        decoded = json.loads(value)
                    except json.JSONDecodeError:
                        return value
                    if isinstance(decoded, Mapping):
                        return text(decoded)
                return value
            if isinstance(value, (int, float, bool)):
                return str(value)
            if isinstance(value, Mapping):
                if value.get("type") == "quantity_f64":
                    encoded, unit = value.get("si_value_f64"), value.get("si_unit")
                    if not isinstance(encoded, str) or not isinstance(unit, str):
                        raise ValueError("certificate contains a malformed canonical quantity")
                    try:
                        return f"{float64_from_hex(encoded)!r} {unit}"
                    except ValueError as exc:
                        raise ValueError("certificate contains an invalid canonical quantity") from exc
                return "; ".join(
                    f"{key.replace('_', ' ')}={text(item)}" for key, item in value.items()
                ) or "—"
            if isinstance(value, (tuple, list)):
                return ", ".join(text(item) for item in value) or "—"
            return type(value).__name__

        def table(title: str, rows: object) -> str:
            entries = tuple(row for row in rows if isinstance(row, Mapping)) if isinstance(rows, (tuple, list)) else ()
            columns = tuple(dict.fromkeys(key for row in entries for key in row))
            if not columns:
                return f"<h4>{escape(title)}</h4><p>None</p>"
            body = "".join(
                "<tr>" + "".join(
                    f"<td>{escape(text(row.get(column)))}</td>"
                    for column in columns
                ) + "</tr>"
                for row in entries if isinstance(row, Mapping)
            )
            if not body:
                body = f"<tr><td colspan=\"{len(columns)}\">None</td></tr>"
            headings = "".join(f"<th>{escape(column.replace('_', ' '))}</th>" for column in columns)
            return f"<h4>{escape(title)}</h4><table><tr>{headings}</tr>{body}</table>"

        def values_by_identity(value: object) -> tuple[Mapping[str, object], ...]:
            if not isinstance(value, Mapping):
                return ()
            return tuple({"identity": identity, "value": item} for identity, item in value.items())

        def detail_rows(kind: str) -> tuple[Mapping[str, object], ...]:
            return tuple(
                row for row in self._data.observed_rows
                if isinstance(row, Mapping) and row.get("kind") == kind
            )

        def verified_rows(kind: str) -> tuple[Mapping[str, object], ...]:
            return tuple(
                row
                for row in self._data.verified_rows
                if isinstance(row, Mapping) and row.get("kind") == kind
            )

        def verified_record_rows(kind: str) -> tuple[Mapping[str, object], ...]:
            """Expose complete captured source records without calling them ink."""

            rows: list[Mapping[str, object]] = []
            for row in verified_rows(kind):
                record = row.get("record")
                rows.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"category", "kind"}
                    }
                    if isinstance(record, Mapping)
                    else dict(row)
                )
            return tuple(rows)

        electrical = self._data.observed_electrical
        semantic = self._data.observed_semantic
        identity_row = next(iter(verified_rows("identity")), {})
        identity = dict(identity_row)
        identity["presentation_sha256"] = self.presentation_sha256
        point_values = next(iter(verified_rows("canonical_point_values")), {})
        compiled_expansion = next(iter(verified_rows("compiled_expansion")), {})
        return HtmlPresentation(
            "<section><h3>Observed electrical reconstruction (audit A)</h3>"
            + table("Nets and exact contacts", electrical.get("nets", ()) if isinstance(electrical, Mapping) else ())
            + table("Native physical bodies: visible contacts, references, and ownership", electrical.get("bodies", ()) if isinstance(electrical, Mapping) else ())
            + table("Visible conductive junctions", electrical.get("junctions", ()) if isinstance(electrical, Mapping) else ())
            + table("Ports: role, raw Z0, orientation, and contacts", electrical.get("ports", ()) if isinstance(electrical, Mapping) else ())
            + table("Local physical ground glyphs and returns", electrical.get("grounds", ()) if isinstance(electrical, Mapping) else ())
            + table("Transmission lines: visible CPW endpoints and ordered MTL conductors", electrical.get("transmission_lines", ()) if isinstance(electrical, Mapping) else ())
            + table("Observed mutual couplings", electrical.get("couplings", ()) if isinstance(electrical, Mapping) else ())
            + table("Displayed coupling coefficients and visible polarity", detail_rows("mutual_coupling"))
            + table("Exact-zero omission evidence", electrical.get("omissions", ()) if isinstance(electrical, Mapping) else ())
            + "<h3>Observed visible structure (audit B)</h3>"
            + table("Visible regions, headers, and containment", semantic.get("regions", ()) if isinstance(semantic, Mapping) else ())
            + table("Visible leaf ownership", semantic.get("leaf_ownership", ()) if isinstance(semantic, Mapping) else ())
            + table("Visible Port ownership", semantic.get("port_ownership", ()) if isinstance(semantic, Mapping) else ())
            + table("Visible cross-boundary electrical incidence", semantic.get("boundary_incidence", ()) if isinstance(semantic, Mapping) else ())
            + "<h3>Visible public-analysis-label observations</h3>"
            + table("Exact text, owner scope, and attached electrical net", semantic.get("public_analysis_labels", ()) if isinstance(semantic, Mapping) else ())
            + "<h3>Observed displayed-point evidence</h3>"
            + table("Displayed selected physical values", detail_rows("displayed_parameter_value"))
            + table("Displayed Port reference impedances", detail_rows("port_impedance"))
            + table("Displayed baseline values retained by a compiled ledger", detail_rows("displayed_baseline_value"))
            + table("Full compiled rows", detail_rows("compiled_matrix_row"))
            + table("All observed reconstruction rows", self._data.observed_rows)
            + "<h3>Verified captured point evidence (separate from audit A/B)</h3>"
            + table("Certificate identities, including complete effective parameters", (identity,))
            + table("Verified selected-point physical values", values_by_identity(point_values.get("values") if isinstance(point_values, Mapping) else None))
            + "<h3>Verified captured source records (not inferred from the drawing)</h3>"
            + table("Authored operators", verified_record_rows("authored_operator"))
            + table("Authored buses and taps", verified_record_rows("authored_bus"))
            + table("Authored public exposures", verified_record_rows("authored_exposure"))
            + table("Authored node aliases", verified_record_rows("authored_node_alias"))
            + table("Parameter definitions", verified_record_rows("parameter_definition"))
            + table("Physical parameter field bindings", verified_record_rows("parameter_field_binding"))
            + table("Complete effective parameter point", verified_record_rows("effective_parameter_point"))
            + table("Source-unit records", verified_record_rows("source_unit"))
            + table("Ground-call records", verified_record_rows("ground_call_group"))
            + table("Verified compiled expansion bindings", (compiled_expansion,))
            + "</section>"
        )


@dataclass(frozen=True, slots=True)
class CircuitDiagramResult(Result):
    drawing: schemdraw.Drawing
    audit: CircuitDiagramAudit
    composition: SchematicCompositionSnapshot | None = None

    def __init__(self) -> None:
        unavailable("CircuitDiagramResult construction")

    def show(self) -> schemdraw.Drawing:
        return self.drawing
