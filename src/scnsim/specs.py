"""Immutable public declarations for SCNSim requests.

Specs describe an operation; they neither execute it nor own a result.  The
runtime performs Plan-specific validation when it binds one of these values to
a sealed Plan.  Keeping these values small and immutable makes the canonical
request encoder the single identity authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from html import escape
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np
from pint import Quantity

from . import units
from .canonical import _identifier, float64_hex
from .value_storage import immutable_quantity, quantity_view
from .construction import unavailable
from .authoring import (
    CoordinateRef,
    ElectricalResolution,
    ParameterSpec,
    ParameterSet,
    ElectricNodeRef,
    ParameterRef,
    PortRef,
)
from .errors import InvalidDiagonalRootHint, InvalidOptimizationSpec, SCNSimValidationError
from .presentation import Theme, _require_theme


if TYPE_CHECKING:
    from .results import (
        Result,
        AnalysisResult,
        DirectQuantityResult,
        DirectSolveResult,
        HtmlPresentation,
        OperatorResult,
        OptimizationResult,
        ParameterSweepResult,
        _is_verified_analysis_result,
    )
    from .visualization.diagram.spec import CircuitDiagramSpec
    from .runtime import NetworkViewRef


Coordinate = str | ElectricNodeRef | CoordinateRef


class DiagramSide(str, Enum):
    """Requested perimeter side for a logical Port in an authoring diagram."""

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


def _coordinate_id(value: Coordinate) -> str:
    if isinstance(value, str):
        if not value:
            raise ValueError("coordinate IDs must not be empty")
        return value
    identifier = getattr(value, "id", None)
    if isinstance(identifier, str) and identifier:
        return identifier
    raise TypeError("coordinate must be a nonempty ID or SCNSim coordinate handle")


def _parameter_key(value: ParameterRef) -> tuple[str, str]:
    if not isinstance(value, ParameterRef):
        raise InvalidOptimizationSpec(
            "optimization parameters must be ParameterRef values",
            stage="spec_validation",
        )
    definitions_id = getattr(value, "definitions_id", None)
    parameter_id = getattr(value, "id", None)
    if isinstance(definitions_id, str) and isinstance(parameter_id, str):
        return definitions_id, parameter_id
    raise InvalidOptimizationSpec(
        "optimization parameters must be ParameterRef values",
        stage="spec_validation",
    )


def _require_quantity(value: Quantity, *, name: str) -> Quantity:
    if not isinstance(value, Quantity) or value._REGISTRY is not units.registry:
        raise TypeError(f"{name} must use the scnsim.units registry")
    magnitude = np.asarray(value.magnitude)
    if magnitude.ndim != 0 or not isfinite(float(magnitude)):
        raise ValueError(f"{name} must be a finite scalar Quantity")
    return value


def _quantity_pair_text(value: tuple[Quantity, Quantity] | None) -> str:
    return "—" if value is None else f"{value[0]} to {value[1]}"


def _selector_text(value: object) -> str:
    if isinstance(value, QuantitySum):
        return "(" + " + ".join(_selector_text(term) for term in value.terms) + ")"
    if isinstance(value, QuantityDifference):
        return f"({_selector_text(value.left)} - {_selector_text(value.right)})"
    if isinstance(value, QuantityAbsolute):
        return f"abs({_selector_text(value.operand)})"
    if isinstance(value, QuantitySelector):
        lineage = getattr(value._view, "_lineage", None)
        if isinstance(lineage, Mapping):
            ptc = lineage.get("ptc")
            if isinstance(ptc, Mapping):
                ptc_text = f"PTC[{','.join(str(item) for item in ptc.get('selected_ports', ()))}]"
            else:
                ptc_text = "raw"
            transforms = lineage.get("transforms", ())
            transform_text = ",".join(
                str(item.get("id")) for item in transforms if isinstance(item, Mapping)
            )
            retain = lineage.get("retain")
            retain_text = (
                ",".join(str(item) for item in retain.get("retained_coordinates", ()))
                if isinstance(retain, Mapping) else "all"
            )
            digest = str(lineage.get("lineage_sha256", "unbound"))[:12]
            view = f"{ptc_text}; transform=[{transform_text}]; retain=[{retain_text}]; sha={digest}"
        else:
            view = "unbound"
        spec = value.spec
        if isinstance(spec, DiagonalRootSpec):
            selection = f"coordinate={_coordinate_id(spec.coordinate)}; root_hint={spec.root_hint}"
        elif isinstance(spec, OperatorElementRootSpec):
            selection = f"row={_coordinate_id(spec.row)}; column={_coordinate_id(spec.column)}; root_hint={spec.root_hint}"
        elif isinstance(spec, HybridizedPoleSpec):
            selection = f"coordinates={','.join(_coordinate_id(item) for item in spec.coordinates)}; anchor={spec.anchor}"
        elif isinstance(spec, TransferZeroSpec):
            selection = (
                f"family={spec.family}; input={_coordinate_id(spec.input_coordinate)}; "
                f"output={_coordinate_id(spec.output_coordinate)}; anchor={spec.anchor}"
            )
        elif isinstance(spec, ResponseElementSpec):
            selection = (
                f"family={spec.family}; input={_coordinate_id(spec.input_coordinate)}; "
                f"output={_coordinate_id(spec.output_coordinate)}; frequency={spec.frequency}"
            )
        elif isinstance(spec, ResidueNormalizedCouplingSpec):
            def branch_text(branch: DiagonalRootSpec | HybridizedPoleSpec) -> str:
                if isinstance(branch, DiagonalRootSpec):
                    return f"coordinate={_coordinate_id(branch.coordinate)}; root_hint={branch.root_hint}"
                return f"coordinates={','.join(_coordinate_id(item) for item in branch.coordinates)}; anchor={branch.anchor}"

            selection = (
                f"branch_a=({branch_text(spec.branch_a)}); "
                f"branch_b=({branch_text(spec.branch_b)}); frequency={spec.frequency}"
            )
        else:
            selection = type(spec).__name__
        return f"{value.type}.{value.projection} ({selection}) [View {view}]"
    record = getattr(value, "_canonical_record", None)
    if callable(record):
        result = record()
        if isinstance(result, Mapping):
            return str(result.get("type", "quantity"))
    return type(value).__name__


def _quantity_magnitudes(quantity: Quantity) -> tuple[float, ...]:
    """Return a one-dimensional finite magnitude sequence without coercion."""

    magnitude = np.asarray(quantity.magnitude)
    if magnitude.ndim != 1:
        raise ValueError("frequency grid must be a one-dimensional Quantity")
    values = tuple(float(item) for item in magnitude.tolist())
    if not values or not all(isfinite(item) for item in values):
        raise ValueError("frequency grid must be nonempty and finite")
    return values


def _validate_frequency_grid(quantity: Quantity, *, preserve_occurrences: bool = False) -> None:
    if not isinstance(quantity, Quantity) or quantity._REGISTRY is not units.registry:
        raise TypeError("frequencies must use the scnsim.units registry")
    try:
        coherent = quantity.to("hertz")
    except Exception as exc:  # Pint owns its dimensionality diagnostics.
        raise TypeError("frequencies must be a frequency Quantity") from exc
    values = _quantity_magnitudes(coherent)
    if any(value <= 0.0 for value in values):
        raise ValueError("frequencies must be strictly positive")
    if not preserve_occurrences and any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("frequencies must be strictly increasing without duplicates")


def _validate_frequency_anchor(value: Quantity, *, name: str) -> None:
    """Validate a real or complex finite frequency anchor without discarding its seed."""

    if not isinstance(value, Quantity) or value._REGISTRY is not units.registry:
        raise TypeError(f"{name} must use the scnsim.units registry")
    try:
        coherent = value.to("hertz")
    except Exception as exc:
        raise TypeError(f"{name} must be a frequency Quantity") from exc
    magnitude = np.asarray(coherent.magnitude)
    if magnitude.ndim != 0:
        raise ValueError(f"{name} must be a scalar Quantity")
    scalar = complex(magnitude.item())
    if not isfinite(scalar.real) or not isfinite(scalar.imag) or scalar.real <= 0.0:
        raise ValueError(f"{name} must be finite with a positive real part")


def _detached_quantity(value: Quantity) -> Quantity:
    """Detach authored Pint storage before an immutable Spec retains it."""

    return immutable_quantity(value)


def _mode_tuple(value: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise TypeError(f"{name} must contain integers")
    return value


def _nonnegative_int_tuple(value: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    checked = _mode_tuple(value, name=name)
    if any(item < 0 for item in checked):
        raise ValueError(f"{name} must contain nonnegative integers")
    return checked


def _family(value: str) -> Literal["S", "Y", "Z"]:
    if value not in {"S", "Y", "Z"}:
        raise ValueError("family must be 'S', 'Y', or 'Z'")
    return value  # type: ignore[return-value]


def _selector_unit(value: object) -> str | None:
    if not isinstance(value, QuantitySelector):
        return None
    if value.type in {"diagonal_root_projection", "operator_element_root_projection", "hybridized_pole_projection", "transfer_zero_projection"}:
        return "hertz"
    if value.type == "residue_coupling_projection":
        return "radian / second"
    if value.type == "static_capacitance_element_projection":
        return "farad"
    if value.type == "response_element_projection":
        family = getattr(value.spec, "family", None)
        return {"S": "dimensionless", "Y": "siemens", "Z": "ohm"}.get(family)
    return None


def _validate_selector(value: object) -> str:
    unit = _selector_unit(value)
    if unit is None:
        raise InvalidOptimizationSpec(
            "quantity must be a declared scalar selector",
            stage="spec_validation",
        )
    return unit


def _expression_unit(value: object) -> str:
    if isinstance(value, QuantitySelector):
        return _validate_selector(value)
    if isinstance(value, QuantitySum):
        return _expression_unit(value.terms[0])
    if isinstance(value, QuantityDifference):
        return _expression_unit(value.left)
    if isinstance(value, QuantityAbsolute):
        return _expression_unit(value.operand)
    raise InvalidOptimizationSpec(
        "objective quantity must be a closed scalar expression",
        stage="spec_validation",
    )


def _require_compatible_expressions(*values: object) -> str:
    units_by_value = tuple(_expression_unit(value) for value in values)
    dimensions = tuple(units.registry.Unit(unit).dimensionality for unit in units_by_value)
    if len(set(dimensions)) != 1:
        raise InvalidOptimizationSpec(
            "scalar expression operands must share one dimensionality",
            stage="spec_validation",
        )
    return units_by_value[0]


@dataclass(frozen=True, slots=True)
class DirectSolveSpec:
    """Request a complete Direct S/Y/Z response on one selected view."""

    frequencies: Quantity
    traces: tuple[SParameterTrace, ...] = ()

    def __init__(self, *, frequencies: Quantity, traces: Sequence[SParameterTrace] = ()) -> None:
        _validate_frequency_grid(frequencies, preserve_occurrences=True)
        checked = tuple(traces)
        if any(not isinstance(trace, SParameterTrace) for trace in checked):
            raise TypeError("traces must contain SParameterTrace values")
        if len({trace.id for trace in checked}) != len(checked):
            raise ValueError("trace IDs must be unique")
        object.__setattr__(self, "frequencies", _detached_quantity(frequencies))
        object.__setattr__(self, "traces", checked)

    @property
    def S(self) -> QuantitySelector:
        return QuantitySelector(self, "S", "matrix_projection")

    @property
    def Y(self) -> QuantitySelector:
        return QuantitySelector(self, "Y", "matrix_projection")

    @property
    def Z(self) -> QuantitySelector:
        return QuantitySelector(self, "Z", "matrix_projection")

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "direct_solve", "frequencies": self.frequencies, "traces": tuple(trace._canonical_record() for trace in self.traces)}


@dataclass(frozen=True, slots=True)
class QuantitySelector:
    """A non-executing scalar projection of one typed Direct quantity Spec."""

    spec: object
    projection: str
    type: str
    _view: object | None = None

    def __add__(self, other: object) -> QuantitySum:
        return QuantitySum(self, other)

    def __sub__(self, other: object) -> QuantityDifference:
        return QuantityDifference(self, other)

    def __abs__(self) -> QuantityAbsolute:
        return QuantityAbsolute(self)

    def on(self, view: NetworkViewRef) -> QuantitySelector:
        """Return this selector immutably bound to one existing network View."""

        from .runtime import NetworkViewRef

        if not isinstance(view, NetworkViewRef):
            raise TypeError("QuantitySelector.on() requires NetworkViewRef")
        return QuantitySelector(self.spec, self.projection, self.type, view)

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": self.type, "spec": self.spec._canonical_record(), "projection": self.projection}


@dataclass(frozen=True, slots=True)
class RootSearchSpec:
    """Explicit discovery method and caller region, separate from root Hint.

    Relative bounds bind the original authored Hint, never a tracked root.
    No imaginary restriction means no added imaginary selection cutoff.
    """

    method: Literal["polynomial_transfer_zero", "finite_seeds"]
    _real_frequency_interval: tuple[Quantity, Quantity] | None
    _imag_frequency_interval: tuple[Quantity, Quantity] | None
    hint_frequency_factor: float | None
    _seeds: tuple[Quantity, ...] | None
    tie: Literal["unavailable", "source_order"]

    def __init__(self, *, method: Literal["polynomial_transfer_zero", "finite_seeds"],
                 tie: Literal["unavailable", "source_order"],
                 real_frequency_interval: tuple[Quantity, Quantity] | None = None,
                 hint_frequency_factor: float | None = None,
                 imag_frequency_interval: tuple[Quantity, Quantity] | None = None,
                 seeds: Sequence[Quantity] | None = None) -> None:
        if method not in {"polynomial_transfer_zero", "finite_seeds"}:
            raise ValueError("method must be polynomial_transfer_zero or finite_seeds")
        if (real_frequency_interval is None) == (hint_frequency_factor is None):
            raise ValueError("declare exactly one real_frequency_interval or hint_frequency_factor")
        for name, interval in (("real_frequency_interval", real_frequency_interval),
                               ("imag_frequency_interval", imag_frequency_interval)):
            if interval is None:
                object.__setattr__(self, "_" + name, None)
                continue
            values = tuple(interval)
            if len(values) != 2:
                raise ValueError(f"{name} requires two ordered frequency endpoints")
            magnitudes = tuple(np.asarray(value.to("hertz").magnitude) for value in values)
            if any(value.ndim != 0 or np.iscomplexobj(value) for value in magnitudes):
                raise TypeError(f"{name} endpoints must be real scalar frequency Quantities")
            if magnitudes[0].item() > magnitudes[1].item():
                raise ValueError(f"{name} lower endpoint exceeds upper endpoint")
            object.__setattr__(self, "_" + name, tuple(_detached_quantity(value) for value in values))
        if hint_frequency_factor is not None:
            if isinstance(hint_frequency_factor, bool):
                raise TypeError("hint_frequency_factor must be a positive real number")
            hint_frequency_factor = float(hint_frequency_factor)
            float64_hex(hint_frequency_factor)
            if hint_frequency_factor <= 0:
                raise ValueError("hint_frequency_factor must be positive")
        if method == "polynomial_transfer_zero":
            if seeds is not None:
                raise TypeError("polynomial_transfer_zero does not accept caller seeds")
            checked = None
        else:
            if seeds is None:
                raise TypeError("finite_seeds requires caller-declared seeds")
            checked = tuple(seeds)
            if not checked:
                raise ValueError("finite_seeds requires nonempty ordered seeds")
            for value in checked:
                if np.asarray(value.to("hertz").magnitude).ndim != 0:
                    raise TypeError("finite_seeds require scalar frequency Quantities")
        if tie not in {"unavailable", "source_order"}:
            raise ValueError("tie must be unavailable or source_order")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "hint_frequency_factor", hint_frequency_factor)
        object.__setattr__(self, "_seeds", None if checked is None else tuple(_detached_quantity(value) for value in checked))
        object.__setattr__(self, "tie", tie)

    @property
    def real_frequency_interval(self) -> tuple[Quantity, Quantity] | None:
        return None if self._real_frequency_interval is None else tuple(quantity_view(value) for value in self._real_frequency_interval)

    @property
    def imag_frequency_interval(self) -> tuple[Quantity, Quantity] | None:
        return None if self._imag_frequency_interval is None else tuple(quantity_view(value) for value in self._imag_frequency_interval)

    @property
    def seeds(self) -> tuple[Quantity, ...] | None:
        return None if self._seeds is None else tuple(quantity_view(value) for value in self._seeds)

    def _canonical_record(self) -> Mapping[str, object]:
        region = ({"real_frequency_interval": self.real_frequency_interval}
                  if self.hint_frequency_factor is None else
                  {"hint_frequency_factor_f64": float64_hex(self.hint_frequency_factor)})
        if self.imag_frequency_interval is not None:
            region["imag_frequency_interval"] = self.imag_frequency_interval
        record = {"method": self.method, "region": region,
                  "selection": {"kind": "nearest_hint", "metric": "complex_frequency_euclidean", "tie": self.tie},
                  "tracking": {"kind": "baseline_anchor"}}
        if self.method == "finite_seeds":
            record["seeds"] = self.seeds
        return record


def _root_search_record(search: RootSearchSpec | None) -> dict[str, object]:
    return {} if search is None else {"search": search._canonical_record()}


def _require_root_search(search: RootSearchSpec | None) -> None:
    if search is not None and not isinstance(search, RootSearchSpec):
        raise TypeError("search must be RootSearchSpec or None")


@dataclass(frozen=True, slots=True)
class DiagonalRootSpec:
    """Select one machine-resolved Newton root of a Direct-operator diagonal.

    ``root_hint`` initializes the deterministic baseline Newton basin only. It
    is neither an answer, target, search window, nearest-root request, nor a
    proof of global spectral uniqueness.
    """

    coordinate: Coordinate
    _root_hint: Quantity
    search: RootSearchSpec | None

    def __init__(self, *, coordinate: Coordinate, root_hint: Quantity, search: RootSearchSpec | None = None) -> None:
        _require_root_search(search)
        object.__setattr__(self, "search", search)
        _coordinate_id(coordinate)
        try:
            if isinstance(root_hint, Quantity) and np.iscomplexobj(root_hint.magnitude):
                _validate_frequency_anchor(root_hint, name="root_hint")
            else:
                units.require_positive_quantity(root_hint, "hertz", name="root_hint")
        except Exception as exc:
            raise InvalidDiagonalRootHint(
                "root_hint must be a finite frequency Quantity with a positive real part",
                stage="spec_validation",
            ) from exc
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "_root_hint", _detached_quantity(root_hint))

    @property
    def root_hint(self) -> Quantity:
        return quantity_view(self._root_hint)

    @property
    def frequency(self) -> QuantitySelector:
        return QuantitySelector(self, "frequency", "diagonal_root_projection")

    @property
    def linewidth(self) -> QuantitySelector:
        return QuantitySelector(self, "linewidth", "diagonal_root_projection")

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "diagonal_root", "coordinate": _coordinate_id(self.coordinate), "root_hint": self.root_hint, **_root_search_record(self.search)}


@dataclass(frozen=True, slots=True)
class OperatorElementRootSpec:
    """Select one ordered element root of the complete final View operator."""

    row: Coordinate
    column: Coordinate
    _root_hint: Quantity
    search: RootSearchSpec | None

    def __init__(self, *, row: Coordinate, column: Coordinate, root_hint: Quantity, search: RootSearchSpec | None = None) -> None:
        _require_root_search(search)
        object.__setattr__(self, "search", search)
        _coordinate_id(row)
        _coordinate_id(column)
        try:
            units.require_positive_quantity(root_hint, "hertz", name="root_hint")
        except Exception as exc:
            raise InvalidDiagonalRootHint(
                "root_hint must be a finite positive frequency Quantity",
                stage="spec_validation",
            ) from exc
        object.__setattr__(self, "row", row)
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "_root_hint", _detached_quantity(root_hint))

    @property
    def root_hint(self) -> Quantity:
        return quantity_view(self._root_hint)

    @property
    def frequency(self) -> QuantitySelector:
        return QuantitySelector(self, "frequency", "operator_element_root_projection")

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "type": "operator_element_root",
            "row": _coordinate_id(self.row),
            "column": _coordinate_id(self.column),
            "root_hint": self.root_hint,
            **_root_search_record(self.search),
        }


@dataclass(frozen=True, slots=True)
class HybridizedPoleSpec:
    """Select an anchored complex pole of a retained coupled block.

    It cannot be substituted with a diagonal root or a nearest sampled peak.
    """

    coordinates: tuple[Coordinate, ...]
    _anchor: Quantity
    search: RootSearchSpec | None

    def __init__(self, *, coordinates: Sequence[Coordinate], anchor: Quantity, search: RootSearchSpec | None = None) -> None:
        _require_root_search(search)
        object.__setattr__(self, "search", search)
        checked = tuple(coordinates)
        identifiers = tuple(_coordinate_id(value) for value in checked)
        if len(identifiers) < 2 or len(set(identifiers)) != len(identifiers):
            raise ValueError("HybridizedPoleSpec requires at least two unique coordinates")
        _validate_frequency_anchor(anchor, name="anchor")
        object.__setattr__(self, "coordinates", checked)
        object.__setattr__(self, "_anchor", _detached_quantity(anchor))

    @property
    def anchor(self) -> Quantity:
        return quantity_view(self._anchor)

    @property
    def frequency(self) -> QuantitySelector:
        return QuantitySelector(self, "frequency", "hybridized_pole_projection")

    @property
    def linewidth(self) -> QuantitySelector:
        return QuantitySelector(self, "linewidth", "hybridized_pole_projection")

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "type": "hybridized_pole",
            "coordinates": tuple(_coordinate_id(value) for value in self.coordinates),
            "anchor": self.anchor,
            **_root_search_record(self.search),
        }


@dataclass(frozen=True, slots=True)
class TransferZeroSpec:
    """Select an anchored exact zero of one declared transfer element.

    This is an analytic complex-Newton quantity, not a sampled response minimum.
    """

    _anchor: Quantity
    search: RootSearchSpec | None
    family: Literal["S", "Y", "Z"]
    input_coordinate: Coordinate
    output_coordinate: Coordinate

    def __init__(self, *, anchor: Quantity, family: Literal["S", "Y", "Z"], input_coordinate: Coordinate, output_coordinate: Coordinate, search: RootSearchSpec | None = None) -> None:
        _require_root_search(search)
        object.__setattr__(self, "search", search)
        _validate_frequency_anchor(anchor, name="anchor")
        _family(family)
        _coordinate_id(input_coordinate)
        _coordinate_id(output_coordinate)
        object.__setattr__(self, "_anchor", _detached_quantity(anchor))
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "input_coordinate", input_coordinate)
        object.__setattr__(self, "output_coordinate", output_coordinate)

    @property
    def anchor(self) -> Quantity:
        return quantity_view(self._anchor)

    @property
    def frequency(self) -> QuantitySelector:
        return QuantitySelector(self, "frequency", "transfer_zero_projection")

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "type": "transfer_zero", "anchor": self.anchor, "family": self.family,
            "input_coordinate": _coordinate_id(self.input_coordinate),
            "output_coordinate": _coordinate_id(self.output_coordinate),
            **_root_search_record(self.search),
        }


@dataclass(frozen=True, slots=True)
class ResidueNormalizedCouplingSpec:
    """Evaluate local coupling using explicit pole/root residue evidence.

    This surface never substitutes a fitted splitting for residue evidence.
    """

    branch_a: DiagonalRootSpec | HybridizedPoleSpec
    branch_b: DiagonalRootSpec | HybridizedPoleSpec
    _frequency: Quantity | Literal["complex_root_midpoint"]

    def __init__(self, *, branch_a: DiagonalRootSpec | HybridizedPoleSpec, branch_b: DiagonalRootSpec | HybridizedPoleSpec, frequency: Quantity | Literal["complex_root_midpoint"]) -> None:
        if not isinstance(branch_a, (DiagonalRootSpec, HybridizedPoleSpec)) or not isinstance(branch_b, (DiagonalRootSpec, HybridizedPoleSpec)):
            raise TypeError("branches must be DiagonalRootSpec or HybridizedPoleSpec")
        if isinstance(frequency, str):
            if frequency != "complex_root_midpoint":
                raise ValueError('frequency string must be "complex_root_midpoint"')
            retained_frequency: Quantity | Literal["complex_root_midpoint"] = frequency
        else:
            units.require_positive_quantity(frequency, "hertz", name="frequency")
            retained_frequency = _detached_quantity(frequency)
        object.__setattr__(self, "branch_a", branch_a)
        object.__setattr__(self, "branch_b", branch_b)
        object.__setattr__(self, "_frequency", retained_frequency)

    @property
    def frequency(self) -> Quantity | Literal["complex_root_midpoint"]:
        return self._frequency if isinstance(self._frequency, str) else quantity_view(self._frequency)

    @property
    def real(self) -> QuantitySelector:
        return QuantitySelector(self, "real", "residue_coupling_projection")

    @property
    def imag(self) -> QuantitySelector:
        return QuantitySelector(self, "imag", "residue_coupling_projection")

    @property
    def magnitude(self) -> QuantitySelector:
        return QuantitySelector(self, "magnitude", "residue_coupling_projection")

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "type": "residue_normalized_coupling",
            "branch_a": self.branch_a._canonical_record(),
            "branch_b": self.branch_b._canonical_record(),
            "frequency": self.frequency,
        }


@dataclass(frozen=True, slots=True)
class ResponseElementSpec:
    """Evaluate one exact S/Y/Z element on a selected Direct network.

    This scalar surface never interpolates a sweep.
    """

    family: Literal["S", "Y", "Z"]
    input_coordinate: Coordinate
    output_coordinate: Coordinate
    frequency: Quantity

    def __init__(self, *, family: Literal["S", "Y", "Z"], input_coordinate: Coordinate, output_coordinate: Coordinate, frequency: Quantity) -> None:
        _family(family)
        _coordinate_id(input_coordinate)
        _coordinate_id(output_coordinate)
        units.require_positive_quantity(frequency, "hertz", name="frequency")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "input_coordinate", input_coordinate)
        object.__setattr__(self, "output_coordinate", output_coordinate)
        object.__setattr__(self, "frequency", _detached_quantity(frequency))

    @property
    def magnitude(self) -> QuantitySelector:
        return QuantitySelector(self, "magnitude", "response_element_projection")

    @property
    def real(self) -> QuantitySelector:
        return QuantitySelector(self, "real", "response_element_projection")

    @property
    def imag(self) -> QuantitySelector:
        return QuantitySelector(self, "imag", "response_element_projection")

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "type": "response_element", "family": self.family,
            "input_coordinate": _coordinate_id(self.input_coordinate),
            "output_coordinate": _coordinate_id(self.output_coordinate), "frequency": self.frequency,
        }


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Materialize the full Direct operator on an exact grid in ``dev5``."""

    frequencies: Quantity

    def __init__(self, *, frequencies: Quantity) -> None:
        _validate_frequency_grid(frequencies, preserve_occurrences=True)
        object.__setattr__(self, "frequencies", _detached_quantity(frequencies))

    @property
    def matrix(self) -> QuantitySelector:
        return QuantitySelector(self, "matrix", "operator_projection")

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "operator", "frequencies": self.frequencies}


@dataclass(frozen=True, slots=True)
class StaticCapacitanceSpec:
    """Static C on the View boundary; omitted-coordinate charge is zero.

    The View alone declares ordered retention. Its lowered full finite basis
    owns the eliminated complement, transform, and checked C-only Schur solve.
    No frequency or RF normalization is implied by this declaration.
    """

    @property
    def matrix(self) -> QuantitySelector:
        return QuantitySelector(self, "matrix", "static_capacitance_projection")

    def element(self, *, row: Coordinate, column: Coordinate) -> StaticCapacitanceElementSpec:
        return StaticCapacitanceElementSpec(row=row, column=column)

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "static_capacitance"}


@dataclass(frozen=True, slots=True)
class StaticCapacitanceElementSpec:
    """One ordered entry after the View's static zero-charge reduction."""

    row: Coordinate
    column: Coordinate

    def __init__(self, *, row: Coordinate, column: Coordinate) -> None:
        _coordinate_id(row)
        _coordinate_id(column)
        object.__setattr__(self, "row", row)
        object.__setattr__(self, "column", column)

    @property
    def value(self) -> QuantitySelector:
        return QuantitySelector(self, "value", "static_capacitance_element_projection")

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "static_capacitance_element", "row": _coordinate_id(self.row),
                "column": _coordinate_id(self.column)}


class OptimizationDomain(str, Enum):
    UNBOUNDED = "UNBOUNDED"
    NONNEGATIVE = "NONNEGATIVE"
    NONPOSITIVE = "NONPOSITIVE"
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


@dataclass(frozen=True, slots=True)
class OptimizationProgress:
    """One stage observation with explicit storage provenance.

    Generation counters apply to CMA stages and iteration counters to local
    stages. An unavailable counter is None, never a fabricated generation.
    Workspace durable callbacks follow an acknowledged commit. Memory events
    describe observed/accepted RAM work, with zero durable committed counts.
    """

    phase: Literal["initial", "resume", "generation", "iteration", "stage_transition", "partial", "complete", "result_reuse"]
    reused: bool
    completed_generations: int | None
    total_generations: int | None
    evaluated_count: int
    requested_budget: int
    achievable_evaluations: int | None
    best_cost: float | None
    stage_id: str = "main"
    strategy: Literal["cmaes", "lbfgsb", "projected_lbfgs"] = "cmaes"
    committed_generations: int | None = None
    completed_iterations: int | None = None
    total_iterations: int | None = None
    committed_iterations: int | None = None
    persistence: Literal["workspace", "memory"] = "workspace"


_OMITTED_TRANSFORM = object()


@dataclass(frozen=True, slots=True)
class OptimizationVariable:
    """Bind one public parameter to finite bounds or one open domain."""

    parameter: ParameterRef
    model_default_bounds: tuple[Quantity, Quantity] | None
    consumer_override_bounds: tuple[Quantity, Quantity] | None = None
    transform: Literal["linear", "log"] = "linear"
    domain: OptimizationDomain | None = None
    scale: Quantity | None = None

    def __init__(self, *, parameter: ParameterRef, bounds: tuple[Quantity, Quantity] | None = None,
                 domain: OptimizationDomain | None = None, scale: Quantity | None = None,
                 transform: Literal["linear", "log"] | object = _OMITTED_TRANSFORM) -> None:
        if parameter is None or (bounds is None) == (domain is None):
            raise InvalidOptimizationSpec("invalid optimization variable", stage="spec_validation")
        if domain is not None:
            if not isinstance(domain, OptimizationDomain) or transform is not _OMITTED_TRANSFORM:
                raise InvalidOptimizationSpec("domain fixes its transform", stage="spec_validation")
            linear = domain in {OptimizationDomain.UNBOUNDED, OptimizationDomain.NONNEGATIVE, OptimizationDomain.NONPOSITIVE}
            if linear:
                _require_quantity(scale, name="domain scale")
                if float(scale.magnitude) <= 0.0:
                    raise InvalidOptimizationSpec("linear domain scale must be positive", stage="spec_validation")
            elif scale is not None:
                raise InvalidOptimizationSpec("signed log domains do not accept scale", stage="spec_validation")
            object.__setattr__(self, "parameter", parameter)
            object.__setattr__(self, "model_default_bounds", None)
            object.__setattr__(self, "consumer_override_bounds", None)
            object.__setattr__(self, "transform", "linear" if linear else "log")
            object.__setattr__(self, "domain", domain)
            object.__setattr__(self, "scale", None if scale is None else _detached_quantity(scale))
            return
        transform = "linear" if transform is _OMITTED_TRANSFORM else transform
        if transform not in {"linear", "log"} or scale is not None:
            raise InvalidOptimizationSpec("invalid bounded optimization variable", stage="spec_validation")
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise InvalidOptimizationSpec("bounds must be a pair", stage="spec_validation")
        for name, value in zip(("lower bound", "upper bound"), bounds):
            _require_quantity(value, name=name)
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "model_default_bounds", tuple(_detached_quantity(value) for value in bounds))
        object.__setattr__(self, "consumer_override_bounds", None)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "domain", None)
        object.__setattr__(self, "scale", None)

    @property
    def bounds(self) -> tuple[Quantity, Quantity] | None:
        """Resolved bounds; the runtime performs Plan/baseline validation."""

        return self.consumer_override_bounds or self.model_default_bounds

    def _override(self, bounds: tuple[Quantity, Quantity]) -> OptimizationVariable:
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise InvalidOptimizationSpec("bounds must be a pair", stage="spec_validation")
        if self.domain is not None:
            raise InvalidOptimizationSpec("domain variables cannot receive bounds overrides", stage="spec_validation")
        for name, value in zip(("lower bound", "upper bound"), bounds):
            _require_quantity(value, name=name)
        instance = object.__new__(OptimizationVariable)
        object.__setattr__(instance, "parameter", self.parameter)
        object.__setattr__(instance, "model_default_bounds", self.model_default_bounds)
        object.__setattr__(instance, "consumer_override_bounds", tuple(_detached_quantity(value) for value in bounds))
        object.__setattr__(instance, "transform", self.transform)
        object.__setattr__(instance, "domain", None)
        object.__setattr__(instance, "scale", None)
        return instance

    def _canonical_record(self) -> Mapping[str, object]:
        if self.domain is not None:
            return {"parameter": self.parameter, "domain": self.domain.value,
                    "transform": self.transform, "scale": self.scale}
        return {
            "parameter": self.parameter, "model_default_bounds": self.model_default_bounds,
            "consumer_override_bounds": self.consumer_override_bounds, "lower": self.bounds[0],
            "upper": self.bounds[1], "transform": self.transform,
        }


@dataclass(frozen=True, slots=True)
class QuantitySum:
    """An ordered sum of same-dimensionality scalar expressions."""

    terms: tuple[object, ...]

    def __init__(self, *terms: object) -> None:
        if not terms:
            raise InvalidOptimizationSpec("QuantitySum requires one or more terms", stage="spec_validation")
        _require_compatible_expressions(*terms)
        object.__setattr__(self, "terms", tuple(terms))

    def __add__(self, other: object) -> QuantitySum:
        return QuantitySum(self, other)

    def __sub__(self, other: object) -> QuantityDifference:
        return QuantityDifference(self, other)

    def __abs__(self) -> QuantityAbsolute:
        return QuantityAbsolute(self)

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "quantity_sum", "terms": tuple(_canonical_value(item) for item in self.terms)}


@dataclass(frozen=True, slots=True)
class QuantityDifference:
    """An ordered left-minus-right scalar expression."""

    left: object
    right: object

    def __init__(self, left: object, right: object) -> None:
        _require_compatible_expressions(left, right)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    def __add__(self, other: object) -> QuantitySum:
        return QuantitySum(self, other)

    def __sub__(self, other: object) -> QuantityDifference:
        return QuantityDifference(self, other)

    def __abs__(self) -> QuantityAbsolute:
        return QuantityAbsolute(self)

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "type": "quantity_difference",
            "left": _canonical_value(self.left),
            "right": _canonical_value(self.right),
        }


@dataclass(frozen=True, slots=True)
class QuantityAbsolute:
    """The absolute value of one scalar expression."""

    operand: object

    def __init__(self, operand: object) -> None:
        _expression_unit(operand)
        object.__setattr__(self, "operand", operand)

    def __add__(self, other: object) -> QuantitySum:
        return QuantitySum(self, other)

    def __sub__(self, other: object) -> QuantityDifference:
        return QuantityDifference(self, other)

    def __abs__(self) -> QuantityAbsolute:
        return QuantityAbsolute(self)

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "quantity_absolute", "operand": _canonical_value(self.operand)}


@dataclass(frozen=True, slots=True)
class CostObjective:
    """Compare one scalar quantity with one target inside optimization."""

    id: str
    quantity: object
    target: Quantity
    weight: Quantity
    scale: Quantity | None = None
    comparison: Literal["target", "at_least"] = "target"

    def __init__(
        self,
        *,
        id: str,
        quantity: object,
        target: Quantity,
        weight: Quantity,
        scale: Quantity | None = None,
        comparison: Literal["target", "at_least"] = "target",
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "target", _detached_quantity(target))
        object.__setattr__(self, "weight", _detached_quantity(weight))
        object.__setattr__(self, "scale", None if scale is None else _detached_quantity(scale))
        object.__setattr__(self, "comparison", comparison)
        self.__post_init__()

    def __post_init__(self) -> None:
        try:
            identifier = _identifier(self.id, field="objective id")
        except Exception as exc:
            raise InvalidOptimizationSpec("objective id must be a canonical identifier", stage="spec_validation") from exc
        object.__setattr__(self, "id", identifier)
        quantity_unit = _expression_unit(self.quantity)
        if self.comparison not in {"target", "at_least"}:
            raise InvalidOptimizationSpec("objective comparison must be target or at_least", stage="spec_validation")
        _require_quantity(self.target, name="target")
        try:
            self.target.to(quantity_unit)
        except Exception as exc:
            raise InvalidOptimizationSpec("objective target dimensionality disagrees with selector", stage="spec_validation") from exc
        units.require_positive_quantity(self.weight, "dimensionless", name="weight")
        if self.scale is not None:
            _require_quantity(self.scale, name="scale")
            try:
                self.scale.to(quantity_unit)
            except Exception as exc:
                raise InvalidOptimizationSpec("objective scale dimensionality disagrees with selector", stage="spec_validation") from exc

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "id": self.id,
            "quantity": _canonical_value(self.quantity),
            "comparison": self.comparison,
            "target": self.target,
            "weight": self.weight,
            "scale": self.scale,
        }


@dataclass(frozen=True, slots=True)
class CMAESSpec:
    """Pinned deterministic CMA-ES controls for a Direct optimization request."""

    seed: int = 0
    max_evaluations: int = 200
    population_size: int | None = None
    initial_sigma: float = 0.25

    def __init__(
        self,
        *,
        seed: int = 0,
        max_evaluations: int = 200,
        population_size: int | None = None,
        initial_sigma: float = 0.25,
    ) -> None:
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "max_evaluations", max_evaluations)
        object.__setattr__(self, "population_size", population_size)
        object.__setattr__(self, "initial_sigma", initial_sigma)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or not -(2**63) <= self.seed < 2**63:
            raise InvalidOptimizationSpec("seed must be a signed 64-bit integer", stage="spec_validation")
        if not isinstance(self.max_evaluations, int) or self.max_evaluations < 1:
            raise InvalidOptimizationSpec("max_evaluations must be positive", stage="spec_validation")
        if self.population_size is not None and (not isinstance(self.population_size, int) or self.population_size < 2):
            raise InvalidOptimizationSpec("population_size must be at least two", stage="spec_validation")
        if not isfinite(self.initial_sigma) or self.initial_sigma <= 0.0:
            raise InvalidOptimizationSpec("initial_sigma must be finite and positive", stage="spec_validation")

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "cma_es", "seed": self.seed, "max_evaluations": self.max_evaluations, "population_size": self.population_size, "initial_sigma": self.initial_sigma}


def _sensitivity_physical_record(value: object) -> object:
    """Lossless new declaration encoding; native request codecs stay unchanged."""
    from .canonical import derived_quantity_envelope, float64_hex
    if isinstance(value, Quantity):
        magnitude = np.asarray(value.magnitude)
        if magnitude.ndim == 0:
            return derived_quantity_envelope(value, registry=units.registry)
        return {"type": "quantity_array", "shape": tuple(magnitude.shape), "values": tuple(derived_quantity_envelope(units.registry.Quantity(item, value.units), registry=units.registry) for item in magnitude.flat)}
    if isinstance(value, Mapping):
        return {key: _sensitivity_physical_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_sensitivity_physical_record(item) for item in value)
    if isinstance(value, float):
        return float64_hex(value)
    return value


def _sensitivity_quantity_record(quantity: object) -> Mapping[str, object]:
    if isinstance(quantity, QuantitySelector):
        view = quantity._view
        if view is None:
            raise TypeError("Sensitivity quantities require an explicitly bound View")
        return {"type": quantity.type, "projection": quantity.projection,
                "spec": _sensitivity_physical_record(quantity.spec._canonical_record()),
                "view": _sensitivity_physical_record({"type": "network_view", "plan_sha256": view._run._plan_sha256, "ptc": view._lineage["ptc"], "transforms": view._lineage["transforms"], "retain": view._lineage["retain"]})}
    if isinstance(quantity, QuantitySum):
        return {"type": "quantity_sum", "terms": tuple(_sensitivity_quantity_record(term) for term in quantity.terms)}
    if isinstance(quantity, QuantityDifference):
        return {"type": "quantity_difference", "left": _sensitivity_quantity_record(quantity.left), "right": _sensitivity_quantity_record(quantity.right)}
    if isinstance(quantity, QuantityAbsolute):
        return {"type": "quantity_absolute", "operand": _sensitivity_quantity_record(quantity.operand)}
    raise TypeError("Sensitivity quantity must be a bound physical expression")


def _quantity_mapping(values: Mapping, *, key_type: type, name: str) -> Mapping:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, key_type) for key in values):
        raise TypeError(f"{name} keys must be {key_type.__name__}")
    if any(not isinstance(value, Quantity) or value._REGISTRY is not units.registry for value in values.values()):
        raise TypeError(f"{name} values must use scnsim.units")
    return MappingProxyType({key: immutable_quantity(value) for key, value in values.items()})


def _public_quantities(values: Mapping) -> Mapping:
    return MappingProxyType({key: quantity_view(value) for key, value in values.items()})


@dataclass(frozen=True, slots=True)
class OutputCoordinate:
    """One explicit real coordinate of an ordered physical output."""

    quantity_label: str
    component: Literal["real", "imag"] = "real"
    index: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantity_label", _identifier(self.quantity_label, field="quantity label"))
        if self.component not in ("real", "imag"):
            raise ValueError("component must be real or imag")
        object.__setattr__(self, "index", _nonnegative_int_tuple(tuple(self.index), name="output index"))

    def _canonical_record(self) -> Mapping[str, object]:
        return {"quantity_label": self.quantity_label, "component": self.component, "index": self.index}


@dataclass(frozen=True, slots=True)
class Jacobian:
    """Request the stored full physical real-coordinate Jacobian."""

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "jacobian"}


@dataclass(frozen=True, slots=True, init=False)
class JVP:
    _direction: Mapping[ParameterRef, Quantity]
    _source_direction: Mapping[ParameterRef, Quantity]

    def __init__(self, *, direction: Mapping[ParameterRef, Quantity]) -> None:
        values = _quantity_mapping(direction, key_type=ParameterRef, name="direction")
        object.__setattr__(self, "_source_direction", values)
        object.__setattr__(self, "_direction", MappingProxyType({ref: immutable_quantity(value.to(ref.spec.si_unit)) for ref, value in values.items()}))

    @property
    def direction(self) -> Mapping[ParameterRef, Quantity]:
        return _public_quantities(self._direction)

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "jvp", "direction": tuple({"parameter": key._key_record(), "value": _sensitivity_physical_record(quantity_view(value))} for key, value in self._direction.items())}


@dataclass(frozen=True, slots=True, init=False)
class VJP:
    _weights: Mapping[OutputCoordinate, Quantity]

    def __init__(self, *, weights: Mapping[OutputCoordinate, Quantity]) -> None:
        object.__setattr__(self, "_weights", _quantity_mapping(weights, key_type=OutputCoordinate, name="weights"))

    @property
    def weights(self) -> Mapping[OutputCoordinate, Quantity]:
        return _public_quantities(self._weights)

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "vjp", "weights": tuple({"output": key._canonical_record(), "value": _sensitivity_physical_record(quantity_view(value))} for key, value in self._weights.items())}


@dataclass(frozen=True, slots=True, init=False)
class SensitivityScales:
    _inputs: Mapping[ParameterRef, Quantity]
    _outputs: Mapping[OutputCoordinate, Quantity]

    def __init__(self, *, inputs: Mapping[ParameterRef, Quantity], outputs: Mapping[OutputCoordinate, Quantity]) -> None:
        object.__setattr__(self, "_inputs", _quantity_mapping(inputs, key_type=ParameterRef, name="inputs"))
        object.__setattr__(self, "_outputs", _quantity_mapping(outputs, key_type=OutputCoordinate, name="outputs"))

    @property
    def inputs(self) -> Mapping[ParameterRef, Quantity]:
        return _public_quantities(self._inputs)

    @property
    def outputs(self) -> Mapping[OutputCoordinate, Quantity]:
        return _public_quantities(self._outputs)

    def _canonical_record(self) -> Mapping[str, object]:
        return {"inputs": JVP(direction=self.inputs)._canonical_record()["direction"], "outputs": VJP(weights=self.outputs)._canonical_record()["weights"]}


@dataclass(frozen=True, slots=True, init=False)
class FixedMeshSpec:
    """One planning declaration; resolved section counts are compilation-owned."""

    resolution: ElectricalResolution | None
    sections: Mapping[tuple[str, ...], int] | None

    def __init__(self, *, resolution: ElectricalResolution | None = None, sections: Mapping[tuple[str, ...], int] | None = None) -> None:
        if (resolution is None) == (sections is None):
            raise ValueError("exactly one of resolution and sections is required")
        if resolution is not None and not isinstance(resolution, ElectricalResolution):
            raise TypeError("resolution must be ElectricalResolution")
        counts = None
        if sections is not None:
            counts = {}
            for path, count in sections.items():
                if not isinstance(path, tuple) or not path or not all(isinstance(part, str) and part for part in path):
                    raise TypeError("sections keys must be occurrence path tuples")
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise ValueError("section counts must be positive integers")
                counts[path] = count
            counts = MappingProxyType(counts)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "sections", counts)

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "fixed_mesh", "resolution": None if self.resolution is None else _sensitivity_physical_record(self.resolution._record()), "sections": None if self.sections is None else tuple({"path": path, "count": count} for path, count in sorted(self.sections.items()))}


@dataclass(frozen=True, slots=True, init=False)
class SensitivitySpec:
    quantities: Mapping[str, object]
    wrt: tuple[ParameterRef, ...]
    derivative: Jacobian | JVP | VJP
    discretization: FixedMeshSpec | None
    scales: SensitivityScales | None

    def __init__(self, *, quantities: Mapping[str, object], wrt: Sequence[ParameterRef], derivative: Jacobian | JVP | VJP, discretization: FixedMeshSpec | None = None, scales: SensitivityScales | None = None) -> None:
        if not isinstance(quantities, Mapping) or not quantities:
            raise TypeError("quantities must be a nonempty ordered mapping")
        normalized_quantities = {}
        for label, quantity in quantities.items():
            label = _identifier(label, field="quantity label")
            if label in normalized_quantities:
                raise ValueError("quantity labels collide after normalization")
            normalized_quantities[label] = quantity
            if not isinstance(quantity, (QuantitySelector, QuantitySum, QuantityDifference, QuantityAbsolute)):
                raise TypeError("quantities must contain bound physical quantity selectors")
        parameters = tuple(wrt)
        if not parameters or any(not isinstance(ref, ParameterRef) or not isinstance(ref.spec, ParameterSpec) for ref in parameters):
            raise TypeError("wrt must contain scalar ParameterRefs")
        if len({_parameter_key(ref) for ref in parameters}) != len(parameters):
            raise ValueError("wrt parameters must be unique")
        if not isinstance(derivative, (Jacobian, JVP, VJP)):
            raise TypeError("derivative must be Jacobian, JVP or VJP")
        if discretization is not None and not isinstance(discretization, FixedMeshSpec):
            raise TypeError("discretization must be FixedMeshSpec")
        if scales is not None and not isinstance(scales, SensitivityScales):
            raise TypeError("scales must be SensitivityScales")
        object.__setattr__(self, "quantities", MappingProxyType(normalized_quantities))
        object.__setattr__(self, "wrt", parameters)
        object.__setattr__(self, "derivative", derivative)
        object.__setattr__(self, "discretization", discretization)
        object.__setattr__(self, "scales", scales)

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "sensitivity", "quantities": tuple({"id": label, "quantity": _sensitivity_quantity_record(quantity)} for label, quantity in self.quantities.items()), "wrt": tuple(ref._key_record() for ref in self.wrt), "derivative": self.derivative._canonical_record(), "discretization": None if self.discretization is None else self.discretization._canonical_record(), "scales": None if self.scales is None else self.scales._canonical_record()}


def _validate_local_optimizer_controls(control) -> None:
    for name in ("max_evaluations", "max_iterations", "max_line_search", "commit_every_iterations"):
        value = getattr(control, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InvalidOptimizationSpec(f"{name} must be positive integer", stage="spec_validation")
    for name in ("ftol", "gtol"):
        if getattr(control, name) < 0:
            raise InvalidOptimizationSpec(f"{name} must be nonnegative", stage="spec_validation")
    if control.coordinate_scales is not None:
        object.__setattr__(control, "coordinate_scales", _quantity_mapping(control.coordinate_scales, key_type=ParameterRef, name="coordinate_scales"))


@dataclass(frozen=True, slots=True, kw_only=True)
class LBFGSBSpec:
    """Explicit local-controller controls, distinct from design acceptance."""

    max_evaluations: int
    max_iterations: int
    ftol: float
    gtol: float
    max_line_search: int
    commit_every_iterations: int
    coordinate_scales: Mapping[ParameterRef, Quantity] | None = None

    def __post_init__(self) -> None:
        _validate_local_optimizer_controls(self)

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        return _public_quantities(value) if name == "coordinate_scales" and value is not None else value

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "lbfgsb", "max_evaluations": self.max_evaluations, "max_iterations": self.max_iterations, "ftol": _sensitivity_physical_record(float(self.ftol)), "gtol": _sensitivity_physical_record(float(self.gtol)), "max_line_search": self.max_line_search, "commit_every_iterations": self.commit_every_iterations, "coordinate_scales": None if self.coordinate_scales is None else JVP(direction=self.coordinate_scales)._canonical_record()["direction"]}


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectedLBFGSSpec:
    """Projected L-BFGS controls for value-first rejection of failed trials."""

    max_evaluations: int
    max_iterations: int
    ftol: float
    gtol: float
    max_line_search: int
    commit_every_iterations: int
    coordinate_scales: Mapping[ParameterRef, Quantity] | None = None

    def __post_init__(self) -> None:
        _validate_local_optimizer_controls(self)

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        return _public_quantities(value) if name == "coordinate_scales" and value is not None else value

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "projected_lbfgs", "max_evaluations": self.max_evaluations, "max_iterations": self.max_iterations, "ftol": _sensitivity_physical_record(float(self.ftol)), "gtol": _sensitivity_physical_record(float(self.gtol)), "max_line_search": self.max_line_search, "commit_every_iterations": self.commit_every_iterations, "coordinate_scales": None if self.coordinate_scales is None else JVP(direction=self.coordinate_scales)._canonical_record()["direction"]}


@dataclass(frozen=True, slots=True, kw_only=True)
class OptimizationStage:
    id: str
    optimizer: CMAESSpec | LBFGSBSpec | ProjectedLBFGSSpec
    discretization: FixedMeshSpec | None = None
    seed: ParameterSet | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identifier(self.id, field="optimization stage id"))
        if not isinstance(self.optimizer, (CMAESSpec, LBFGSBSpec, ProjectedLBFGSSpec)):
            raise TypeError("stage optimizer must be CMAESSpec, LBFGSBSpec or ProjectedLBFGSSpec")
        if self.discretization is not None and not isinstance(self.discretization, FixedMeshSpec):
            raise TypeError("stage discretization must be FixedMeshSpec")
        if self.seed is not None and not isinstance(self.seed, ParameterSet):
            raise TypeError("stage seed must be ParameterSet")

    def _canonical_record(self) -> Mapping[str, object]:
        return {"id": self.id, "optimizer": self.optimizer._canonical_record(), "discretization": None if self.discretization is None else self.discretization._canonical_record(), "seed": None if self.seed is None else self.seed._record()}


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridSpec:
    stages: tuple[OptimizationStage, ...]
    comparison_stage_id: str

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages or any(not isinstance(stage, OptimizationStage) for stage in stages):
            raise TypeError("stages must contain OptimizationStage values")
        object.__setattr__(self, "comparison_stage_id", _identifier(self.comparison_stage_id, field="comparison stage id"))
        ids = tuple(stage.id for stage in stages)
        if len(set(ids)) != len(ids) or self.comparison_stage_id not in ids:
            raise ValueError("stage ids must be unique and bind comparison_stage_id")
        object.__setattr__(self, "stages", stages)

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "hybrid", "stages": tuple(stage._canonical_record() for stage in self.stages), "comparison_stage_id": self.comparison_stage_id}


@dataclass(frozen=True, slots=True, kw_only=True)
class AcceptedIterateSelection:
    """Exact committed point for a fresh declared stage sequence, not resume."""

    attempt_id: str
    stage_id: str
    accepted_iterate_id: str

    def __post_init__(self) -> None:
        for name in ("attempt_id", "stage_id", "accepted_iterate_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise TypeError(f"{name} must be a nonempty identity")

    def _canonical_record(self) -> Mapping[str, object]:
        return {"attempt_id": self.attempt_id, "stage_id": self.stage_id, "accepted_iterate_id": self.accepted_iterate_id}


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    """One immutable physical objective request with an explicit optimizer strategy."""

    variables: tuple[OptimizationVariable, ...]
    objectives: tuple[CostObjective, ...]
    optimizer: CMAESSpec | LBFGSBSpec | ProjectedLBFGSSpec | HybridSpec
    allow_extrapolation: tuple[ParameterRef, ...] = ()
    discretization: FixedMeshSpec | None = None

    def __init__(self, *, variables: Sequence[OptimizationVariable], objectives: Sequence[CostObjective], optimizer: CMAESSpec | LBFGSBSpec | ProjectedLBFGSSpec | HybridSpec, discretization: FixedMeshSpec | None = None) -> None:
        self._initialize(variables=variables, objectives=objectives, optimizer=optimizer, allow_extrapolation=(), discretization=discretization)

    def _initialize(
        self,
        *,
        variables: Sequence[OptimizationVariable],
        objectives: Sequence[CostObjective],
        optimizer: CMAESSpec | LBFGSBSpec | ProjectedLBFGSSpec | HybridSpec,
        allow_extrapolation: Sequence[ParameterRef],
        discretization: FixedMeshSpec | None,
    ) -> None:
        checked_variables = tuple(variables)
        checked_objectives = tuple(objectives)
        if not checked_variables or not all(isinstance(item, OptimizationVariable) for item in checked_variables):
            raise InvalidOptimizationSpec("variables must be nonempty OptimizationVariable values", stage="spec_validation")
        if not checked_objectives or not all(isinstance(item, CostObjective) for item in checked_objectives):
            raise InvalidOptimizationSpec("objectives must be nonempty CostObjective values", stage="spec_validation")
        if not isinstance(optimizer, (CMAESSpec, LBFGSBSpec, ProjectedLBFGSSpec, HybridSpec)):
            raise InvalidOptimizationSpec("optimizer must be CMAESSpec, LBFGSBSpec, ProjectedLBFGSSpec or HybridSpec", stage="spec_validation")
        variable_keys = tuple(_parameter_key(item.parameter) for item in checked_variables)
        if len(set(variable_keys)) != len(variable_keys):
            raise InvalidOptimizationSpec("optimization parameters must be unique", stage="spec_validation")
        if len({item.id for item in checked_objectives}) != len(checked_objectives):
            raise InvalidOptimizationSpec("objective IDs must be unique", stage="spec_validation")
        auth = tuple(sorted(tuple(allow_extrapolation), key=_parameter_key))
        active_parameters = {_parameter_key(item.parameter): item.parameter for item in checked_variables}
        if (
            len({_parameter_key(item) for item in auth}) != len(auth)
            or any(_parameter_key(item) not in variable_keys for item in auth)
            or any(active_parameters[_parameter_key(item)] is not item for item in auth)
        ):
            raise InvalidOptimizationSpec("allow_extrapolation must contain unique active parameters", stage="spec_validation")
        if discretization is not None and not isinstance(discretization, FixedMeshSpec):
            raise TypeError("discretization must be FixedMeshSpec")
        object.__setattr__(self, "discretization", discretization)
        object.__setattr__(self, "variables", checked_variables)
        object.__setattr__(self, "objectives", checked_objectives)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "allow_extrapolation", auth)

    def variable(self, parameter: ParameterRef) -> OptimizationVariable:
        """Return the active variable owned by this exact public parameter."""

        key = _parameter_key(parameter)
        for variable in self.variables:
            if _parameter_key(variable.parameter) == key:
                if variable.parameter is parameter:
                    return variable
                raise InvalidOptimizationSpec(
                    "optimization variable ParameterRef is foreign",
                    stage="spec_validation",
                )
        raise KeyError(f"no optimization variable for {'.'.join((*key[0], key[1]))}")

    def with_variable_overrides(self, *, bounds: Mapping[ParameterRef, tuple[Quantity, Quantity]], allow_extrapolation: Sequence[ParameterRef] = ()) -> OptimizationSpec:
        """Return a copy with named bounds replaced and a new authorization set."""

        if not isinstance(bounds, Mapping):
            raise InvalidOptimizationSpec("bounds must map active ParameterRef values to pairs", stage="spec_validation")
        overrides = {_parameter_key(parameter): (parameter, value) for parameter, value in bounds.items()}
        if len(overrides) != len(bounds):
            raise InvalidOptimizationSpec("override parameters must be unique", stage="spec_validation")
        active = {_parameter_key(variable.parameter): variable.parameter for variable in self.variables}
        for key, (parameter, _) in overrides.items():
            if key not in active:
                raise InvalidOptimizationSpec("override parameter is not active", stage="spec_validation")
            if active[key] is not parameter:
                raise InvalidOptimizationSpec(
                    "override ParameterRef is foreign",
                    stage="spec_validation",
                )
        instance = object.__new__(OptimizationSpec)
        instance._initialize(
            variables=tuple(
                variable._override(overrides[_parameter_key(variable.parameter)][1])
                if _parameter_key(variable.parameter) in overrides else variable
                for variable in self.variables
            ),
            objectives=self.objectives,
            optimizer=self.optimizer,
            allow_extrapolation=allow_extrapolation,
            discretization=self.discretization,
        )
        return instance

    def show(self) -> HtmlPresentation:
        """Present model defaults, active overrides, objectives, and CMA controls."""

        from .results import HtmlPresentation

        def mapping_scale(variable: OptimizationVariable) -> str:
            if variable.scale is not None:
                return str(variable.scale)
            if variable.domain in {OptimizationDomain.POSITIVE, OptimizationDomain.NEGATIVE}:
                return f"x = x0 * exp(z); x0 = effective request initial ({variable.parameter.spec.unit})"
            return "—"

        rows = "".join(
            "<tr>"
            f"<td>{escape('.'.join(_parameter_key(variable.parameter)))}</td>"
            f"<td>{escape(_quantity_pair_text(variable.model_default_bounds))}</td>"
            f"<td>{escape(_quantity_pair_text(variable.consumer_override_bounds))}</td>"
            f"<td>{escape(_quantity_pair_text(variable.bounds) if variable.domain is None else variable.domain.value)}</td>"
            f"<td>{escape(variable.transform)}</td>"
            f"<td>{escape(mapping_scale(variable))}</td>"
            "</tr>"
            for variable in self.variables
        )
        objectives = "".join(
            "<li>"
            f"{escape(objective.id)}: {escape(_selector_text(objective.quantity))}; "
            f"comparison={escape(objective.comparison)}; target={escape(str(objective.target))}; "
            f"scale={escape('auto (abs(target); dimensionless zero uses 1)' if objective.scale is None else str(objective.scale))}; "
            f"weight={escape(str(objective.weight))}"
            "</li>"
            for objective in self.objectives
        )
        controls = str(self.optimizer._canonical_record())
        return HtmlPresentation(
            "<table><thead><tr><th>parameter</th><th>model default</th><th>consumer override</th><th>resolved bounds / domain</th><th>transform</th><th>mapping scale</th></tr></thead>"
            f"<tbody>{rows}</tbody></table><h3>objectives</h3><ul>{objectives}</ul><h3>optimizer</h3><p>{escape(controls)}</p>"
        )

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "type": "optimization", "variables": tuple(item._canonical_record() for item in self.variables),
            "objectives": tuple(item._canonical_record() for item in self.objectives),
            "optimizer": self.optimizer._canonical_record(), "allow_extrapolation": self.allow_extrapolation,
            "discretization": None if self.discretization is None else self.discretization._canonical_record(),
        }


@dataclass(frozen=True, slots=True)
class SParameterTrace:
    """Name one selected-matrix S projection for Direct or HB."""

    id: str
    input_port: str
    input_mode: tuple[int, ...]
    output_port: str
    output_mode: tuple[int, ...]

    def __init__(
        self,
        *,
        id: str,
        input_port: str,
        input_mode: tuple[int, ...],
        output_port: str,
        output_mode: tuple[int, ...],
    ) -> None:
        id = _identifier(id, field="trace id")
        input_port = _identifier(input_port, field="trace input Port")
        output_port = _identifier(output_port, field="trace output Port")
        if not isinstance(input_mode, tuple) or not isinstance(output_mode, tuple):
            raise TypeError("trace modes must be tuples")
        if any(not isinstance(mode, int) or isinstance(mode, bool) for mode in (*input_mode, *output_mode)):
            raise TypeError("trace modes must contain integers")
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "input_port", input_port)
        object.__setattr__(self, "input_mode", input_mode)
        object.__setattr__(self, "output_port", output_port)
        object.__setattr__(self, "output_mode", output_mode)

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "id": self.id, "input_port": self.input_port, "input_mode": self.input_mode,
            "output_port": self.output_port, "output_mode": self.output_mode,
        }


ReportChannel = str | tuple[str, tuple[int, ...]]


def _report_channel(value: ReportChannel | None, *, name: str) -> ReportChannel | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and bool(value[0])
        and isinstance(value[1], tuple)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value[1])
    ):
        return value
    raise TypeError(f"{name} must be a nonempty coordinate ID or (coordinate, mode) tuple")


def _exact_report_channel(
    channels: Sequence[tuple[str, tuple[int, ...]]],
    selector: ReportChannel,
    *,
    role: str,
) -> tuple[str, tuple[int, ...]]:
    from .results.selection import _channel_index

    return channels[_channel_index(channels, selector, role=role, default=False)]


@dataclass(frozen=True, slots=True)
class ReportPanel:
    """One closed presentation selection for an exact verified Result."""

    result: Result
    kind: Literal["response", "matrix", "summary", "outcomes", "history"]
    family: Literal["S", "Y", "Z"] | None
    input_channel: ReportChannel | None
    output_channel: ReportChannel | None
    _frequency: Quantity | None = field(repr=False)
    matrix_style: Literal["table", "heatmap"] | None
    component: Literal["magnitude", "phase", "real", "imag"] | None
    magnitude: Literal["linear", "db"] | None
    history_metric: Literal["cost", "objective", "residual", "parameter"] | None
    objective: str | None
    parameter: ParameterRef | None

    def __init__(
        self,
        *,
        result: Result,
        kind: Literal["response", "matrix", "summary", "outcomes", "history"] | None = None,
        family: Literal["S", "Y", "Z"] | None = None,
        input_channel: ReportChannel | None = None,
        output_channel: ReportChannel | None = None,
        frequency: Quantity | None = None,
        matrix_style: Literal["table", "heatmap"] | None = None,
        component: Literal["magnitude", "phase", "real", "imag"] | None = None,
        magnitude: Literal["linear", "db"] | None = None,
        history_metric: Literal["cost", "objective", "residual", "parameter"] | None = None,
        objective: str | None = None,
        parameter: ParameterRef | None = None,
    ) -> None:
        from .results import (
            DirectSolveResult, OperatorResult, OptimizationResult,
            ParameterSweepResult, _is_checked_report_input,
            InMemoryDirectSolveResult, InMemoryAnalysisResult,
            InMemoryParameterSweepResult, InMemoryOptimizationResult,
        )

        if not _is_checked_report_input(result):
            raise TypeError("ReportPanel.result must be a checked report Result")
        is_direct = isinstance(result, (DirectSolveResult, InMemoryDirectSolveResult)) or (
            type(result) is InMemoryAnalysisResult and result.result_kind == "direct_solve" and result.status == "success"
        )
        is_operator = isinstance(result, OperatorResult) or (
            type(result) is InMemoryAnalysisResult and result.result_kind == "operator" and result.status == "success"
        )
        is_optimization = isinstance(result, (OptimizationResult, InMemoryOptimizationResult))
        if family is not None:
            _family(family)
        input_channel = _report_channel(input_channel, name="input_channel")
        output_channel = _report_channel(output_channel, name="output_channel")
        if frequency is not None:
            units.require_positive_quantity(frequency, "hertz", name="frequency")
        if matrix_style not in {None, "table", "heatmap"}:
            raise ValueError("matrix_style must be 'table', 'heatmap', or None")
        if component not in {None, "magnitude", "phase", "real", "imag"}:
            raise ValueError("component is invalid")
        if magnitude not in {None, "linear", "db"}:
            raise ValueError("magnitude must be 'linear', 'db', or None")
        if history_metric not in {None, "cost", "objective", "residual", "parameter"}:
            raise ValueError("history_metric is invalid")
        if objective is not None and (not isinstance(objective, str) or not objective):
            raise TypeError("objective must be a nonempty objective ID or None")
        if parameter is not None and not isinstance(parameter, ParameterRef):
            raise TypeError("parameter must be a ParameterRef or None")

        matrix_requested = frequency is not None or matrix_style is not None
        history_requested = any(value is not None for value in (history_metric, objective, parameter))
        response_requested = any(
            value is not None
            for value in (family, input_channel, output_channel, component, magnitude)
        )
        if kind is None:
            groups = sum((matrix_requested, history_requested, response_requested and not matrix_requested))
            if groups > 1:
                raise ValueError("ReportPanel selector groups are mutually exclusive")
            if matrix_requested:
                kind = "matrix"
            elif history_requested:
                kind = "history"
            elif response_requested:
                kind = "response"
            elif is_direct:
                kind = "response" if len(result.s.view.input_channels) == len(result.s.view.output_channels) == 1 else "summary"
            elif isinstance(result, (ParameterSweepResult, InMemoryParameterSweepResult)):
                kind = "outcomes"
            elif is_optimization:
                kind = "history"
            else:
                kind = "summary"
        if kind not in {"response", "matrix", "summary", "outcomes", "history"}:
            raise ValueError("kind must be 'response', 'matrix', 'summary', 'outcomes', or 'history'")

        if kind == "response":
            if not is_direct:
                raise TypeError("response panels require a Direct Result")
            if matrix_requested or history_requested:
                raise ValueError("response panels do not accept matrix or history selectors")
            if (input_channel is None) != (output_channel is None):
                raise ValueError("input_channel and output_channel must be supplied together")
            family = "S" if family is None else family
            component = "magnitude" if component is None else component
            magnitude = "linear" if magnitude is None else magnitude
            if magnitude == "db" and (family != "S" or component != "magnitude"):
                raise ValueError("dB is supported only for S magnitude")
            if component != "magnitude" and magnitude != "linear":
                raise ValueError("dB magnitude is incompatible with a non-magnitude component")
            if is_direct:
                view = result._family(family).view
                if (
                    input_channel is None
                    and (len(view.input_channels) != 1 or len(view.output_channels) != 1)
                ):
                    raise ValueError("multichannel Direct response panels require exact input/output channels")
                input_channel = _exact_report_channel(
                    view.input_channels,
                    view.input_channels[0] if input_channel is None else input_channel,
                    role="input",
                )
                output_channel = _exact_report_channel(
                    view.output_channels,
                    view.output_channels[0] if output_channel is None else output_channel,
                    role="output",
                )
            history_metric = None
        elif kind == "matrix":
            if not (is_direct or is_operator):
                raise TypeError("matrix panels require a Direct or Operator Result")
            if history_requested or any(value is not None for value in (input_channel, output_channel)):
                raise ValueError("matrix panels accept no response-channel or history selectors")
            if frequency is None:
                raise ValueError("matrix panels require an exact frequency")
            matrix_style = "table" if matrix_style is None else matrix_style
            if is_operator and family is not None:
                raise ValueError("Operator matrix panels do not accept an S/Y/Z family")
            if is_direct:
                family = "S" if family is None else family
                view = result._family(family).view
                from .results.selection import _frequency_index

                frequency = view.frequencies[_frequency_index(view, frequency)]
            else:
                frequency = result.at(frequency).frequency
            if matrix_style == "table":
                if component is not None or magnitude not in {None, "linear"}:
                    raise ValueError("matrix tables retain complex values and accept no component or dB")
                magnitude = "linear"
            else:
                if component is None:
                    raise ValueError("matrix heatmaps require an explicit component")
                magnitude = "linear" if magnitude is None else magnitude
                if is_operator and magnitude != "linear":
                    raise ValueError("Operator heatmaps do not accept dB magnitude")
                if is_direct and magnitude == "db" and (
                    family != "S" or component != "magnitude"
                ):
                    raise ValueError("dB is supported only for S magnitude")
            history_metric = None
        elif kind == "history":
            if not is_optimization:
                raise TypeError("history panels require an OptimizationResult")
            if matrix_requested or response_requested:
                raise ValueError("history panels accept no response or matrix selectors")
            if history_metric is None:
                if objective is not None and parameter is not None:
                    raise ValueError("history panels cannot select objective and parameter together")
                history_metric = "objective" if objective is not None else (
                    "parameter" if parameter is not None else "cost"
                )
            if history_metric in {"objective", "residual"}:
                if objective is None or parameter is not None:
                    raise ValueError("objective/residual history requires only objective")
            elif history_metric == "parameter":
                if parameter is None or objective is not None:
                    raise ValueError("parameter history requires only parameter")
            elif objective is not None or parameter is not None:
                raise ValueError("cost history accepts neither objective nor parameter")
            from .results.selection import _optimization_series

            _optimization_series(
                result,
                kind="history" if history_metric == "cost" else history_metric,
                objective=objective,
                parameter=parameter,
            )
        else:
            if matrix_requested or history_requested or response_requested:
                raise ValueError(f"{kind} panels accept no selection fields")
            if kind == "outcomes" and not isinstance(result, (ParameterSweepResult, InMemoryParameterSweepResult)):
                raise TypeError("outcomes panels require a parameter sweep")
            history_metric = None

        object.__setattr__(self, "result", result)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "input_channel", input_channel)
        object.__setattr__(self, "output_channel", output_channel)
        object.__setattr__(self, "_frequency", None if frequency is None else immutable_quantity(frequency))
        object.__setattr__(self, "matrix_style", matrix_style)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "magnitude", magnitude)
        object.__setattr__(self, "history_metric", history_metric)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "parameter", parameter)

    @property
    def frequency(self) -> Quantity | None:
        return None if self._frequency is None else quantity_view(self._frequency)


@dataclass(frozen=True, slots=True)
class ReportSpec:
    """Choose exact checked saved or detached Results for a pure derived report."""

    inputs: tuple[Result, ...]
    theme: Theme = Theme.AUTO
    panels: tuple[ReportPanel, ...] = ()

    def __init__(
        self,
        *,
        inputs: Sequence[Result],
        theme: Theme = Theme.AUTO,
        panels: Sequence[ReportPanel] = (),
    ) -> None:
        from .results import _is_checked_report_input

        checked = tuple(inputs)
        if not checked or not all(_is_checked_report_input(item) for item in checked):
            raise TypeError("ReportSpec.inputs must be nonempty checked report Result values")
        if any(left is right for index, left in enumerate(checked) for right in checked[index + 1 :]):
            raise ValueError("ReportSpec.inputs must not repeat the same Result object")
        selected = tuple(panels)
        if not all(isinstance(panel, ReportPanel) for panel in selected):
            raise TypeError("ReportSpec.panels must contain ReportPanel values")
        for panel in selected:
            if not any(panel.result is item for item in checked):
                raise ValueError("each ReportPanel.result must be the exact object in ReportSpec.inputs")
        object.__setattr__(self, "inputs", checked)
        object.__setattr__(self, "theme", _require_theme(theme))
        object.__setattr__(self, "panels", selected)


def _canonical_value(value: object) -> object:
    record = getattr(value, "_canonical_record", None)
    return record() if callable(record) else value


def __getattr__(name: str) -> object:
    """Load diagram-only declarations only for explicit diagram consumers."""

    if name != "CircuitDiagramSpec":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .visualization.diagram.spec import CircuitDiagramSpec

    globals()[name] = CircuitDiagramSpec
    return CircuitDiagramSpec
