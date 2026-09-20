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
from ._canonical import _identifier
from ._immutable_values import immutable_quantity, quantity_view
from ._scaffold import unavailable
from .authoring import (
    CoordinateRef,
    ElectricNodeRef,
    ParameterRef,
    PortRef,
)
from .errors import InvalidDiagonalRootHint, InvalidOptimizationSpec, SCNSimValidationError
from .presentation import Theme, _require_theme
from .results import (
    AnalysisResult,
    DirectQuantityResult,
    DirectSolveResult,
    HBBatchResult,
    HtmlPresentation,
    OperatorResult,
    OptimizationResult,
    ParameterSweepResult,
    _is_verified_analysis_result,
)

if TYPE_CHECKING:
    from ._diagram_spec import CircuitDiagramSpec
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


def _validate_frequency_grid(quantity: Quantity) -> None:
    if not isinstance(quantity, Quantity) or quantity._REGISTRY is not units.registry:
        raise TypeError("frequencies must use the scnsim.units registry")
    try:
        coherent = quantity.to("hertz")
    except Exception as exc:  # Pint owns its dimensionality diagnostics.
        raise TypeError("frequencies must be a frequency Quantity") from exc
    values = _quantity_magnitudes(coherent)
    if any(value <= 0.0 for value in values):
        raise ValueError("frequencies must be strictly positive")
    if any(right <= left for left, right in zip(values, values[1:])):
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


def _validate_complex_current(value: Quantity, *, name: str) -> None:
    if not isinstance(value, Quantity) or value._REGISTRY is not units.registry:
        raise TypeError(f"{name} must use the scnsim.units registry")
    try:
        magnitude = np.asarray(value.to("ampere").magnitude)
    except Exception as exc:
        raise TypeError(f"{name} must be a current Quantity") from exc
    if magnitude.ndim != 0:
        raise ValueError(f"{name} must be a finite scalar Quantity")
    scalar = complex(magnitude.item())
    if not isfinite(scalar.real) or not isfinite(scalar.imag):
        raise ValueError(f"{name} must be finite")


def _validate_hb_mode(
    mode: tuple[int, ...],
    *,
    rank: int,
    limits: tuple[int, ...],
    truncation: HBTruncation,
    name: str,
) -> None:
    if len(mode) != rank:
        raise ValueError(f"{name} rank must equal pump-axis rank")
    if any(abs(value) > limit for value, limit in zip(mode, limits)):
        raise ValueError(f"{name} is outside the declared harmonic lattice")
    crop = truncation.max_intermodulation_order
    if crop is not None and sum(abs(value) for value in mode) > crop:
        raise ValueError(f"{name} is outside max_intermodulation_order")


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
        _validate_frequency_grid(frequencies)
        checked = tuple(traces)
        if any(not isinstance(trace, SParameterTrace) for trace in checked):
            raise TypeError("traces must contain SParameterTrace values")
        if len({trace.id for trace in checked}) != len(checked):
            raise ValueError("trace IDs must be unique")
        object.__setattr__(self, "frequencies", _detached_quantity(frequencies))
        object.__setattr__(self, "traces", checked)

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
class DiagonalRootSpec:
    """Select one machine-resolved Newton root of a Direct-operator diagonal.

    ``root_hint`` initializes the deterministic baseline Newton basin only. It
    is neither an answer, target, search window, nearest-root request, nor a
    proof of global spectral uniqueness.
    """

    coordinate: Coordinate
    _root_hint: Quantity

    def __init__(self, *, coordinate: Coordinate, root_hint: Quantity) -> None:
        _coordinate_id(coordinate)
        try:
            units.require_positive_quantity(root_hint, "hertz", name="root_hint")
        except Exception as exc:
            raise InvalidDiagonalRootHint(
                "root_hint must be a finite positive frequency Quantity",
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
        return {"type": "diagonal_root", "coordinate": _coordinate_id(self.coordinate), "root_hint": self.root_hint}


@dataclass(frozen=True, slots=True)
class OperatorElementRootSpec:
    """Select one ordered element root of the complete final View operator."""

    row: Coordinate
    column: Coordinate
    _root_hint: Quantity

    def __init__(self, *, row: Coordinate, column: Coordinate, root_hint: Quantity) -> None:
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
        }


@dataclass(frozen=True, slots=True)
class HybridizedPoleSpec:
    """Select an anchored complex pole of a retained coupled block.

    It cannot be substituted with a diagonal root or a nearest sampled peak.
    """

    coordinates: tuple[Coordinate, ...]
    _anchor: Quantity

    def __init__(self, *, coordinates: Sequence[Coordinate], anchor: Quantity) -> None:
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
        }


@dataclass(frozen=True, slots=True)
class TransferZeroSpec:
    """Select an anchored exact zero of one declared transfer element.

    This is an analytic complex-Newton quantity, not a sampled response minimum.
    """

    _anchor: Quantity
    family: Literal["S", "Y", "Z"]
    input_coordinate: Coordinate
    output_coordinate: Coordinate

    def __init__(self, *, anchor: Quantity, family: Literal["S", "Y", "Z"], input_coordinate: Coordinate, output_coordinate: Coordinate) -> None:
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
        }


@dataclass(frozen=True, slots=True)
class ResidueNormalizedCouplingSpec:
    """Evaluate local coupling using explicit pole/root residue evidence.

    This surface never substitutes a fitted splitting for residue evidence.
    """

    branch_a: DiagonalRootSpec | HybridizedPoleSpec
    branch_b: DiagonalRootSpec | HybridizedPoleSpec
    frequency: Quantity

    def __init__(self, *, branch_a: DiagonalRootSpec | HybridizedPoleSpec, branch_b: DiagonalRootSpec | HybridizedPoleSpec, frequency: Quantity) -> None:
        if not isinstance(branch_a, (DiagonalRootSpec, HybridizedPoleSpec)) or not isinstance(branch_b, (DiagonalRootSpec, HybridizedPoleSpec)):
            raise TypeError("branches must be DiagonalRootSpec or HybridizedPoleSpec")
        units.require_positive_quantity(frequency, "hertz", name="frequency")
        object.__setattr__(self, "branch_a", branch_a)
        object.__setattr__(self, "branch_b", branch_b)
        object.__setattr__(self, "frequency", _detached_quantity(frequency))

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
        _validate_frequency_grid(frequencies)
        object.__setattr__(self, "frequencies", _detached_quantity(frequencies))

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "operator", "frequencies": self.frequencies}


@dataclass(frozen=True, slots=True)
class OptimizationVariable:
    """Bind one public parameter to immutable physical search bounds."""

    parameter: ParameterRef
    model_default_bounds: tuple[Quantity, Quantity]
    consumer_override_bounds: tuple[Quantity, Quantity] | None = None
    transform: Literal["linear", "log"] = "linear"

    def __init__(self, *, parameter: ParameterRef, bounds: tuple[Quantity, Quantity], transform: Literal["linear", "log"] = "linear") -> None:
        if parameter is None or transform not in {"linear", "log"}:
            raise InvalidOptimizationSpec("invalid optimization variable", stage="spec_validation")
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise InvalidOptimizationSpec("bounds must be a pair", stage="spec_validation")
        for name, value in zip(("lower bound", "upper bound"), bounds):
            _require_quantity(value, name=name)
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "model_default_bounds", tuple(_detached_quantity(value) for value in bounds))
        object.__setattr__(self, "consumer_override_bounds", None)
        object.__setattr__(self, "transform", transform)

    @property
    def bounds(self) -> tuple[Quantity, Quantity]:
        """Resolved bounds; the runtime performs Plan/baseline validation."""

        return self.consumer_override_bounds or self.model_default_bounds

    def _override(self, bounds: tuple[Quantity, Quantity]) -> OptimizationVariable:
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise InvalidOptimizationSpec("bounds must be a pair", stage="spec_validation")
        for name, value in zip(("lower bound", "upper bound"), bounds):
            _require_quantity(value, name=name)
        instance = object.__new__(OptimizationVariable)
        object.__setattr__(instance, "parameter", self.parameter)
        object.__setattr__(instance, "model_default_bounds", self.model_default_bounds)
        object.__setattr__(instance, "consumer_override_bounds", tuple(_detached_quantity(value) for value in bounds))
        object.__setattr__(instance, "transform", self.transform)
        return instance

    def _canonical_record(self) -> Mapping[str, object]:
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


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    """One immutable Direct-only multi-variable CMA-ES request declaration."""

    variables: tuple[OptimizationVariable, ...]
    objectives: tuple[CostObjective, ...]
    optimizer: CMAESSpec
    allow_extrapolation: tuple[ParameterRef, ...] = ()

    def __init__(self, *, variables: Sequence[OptimizationVariable], objectives: Sequence[CostObjective], optimizer: CMAESSpec) -> None:
        self._initialize(variables=variables, objectives=objectives, optimizer=optimizer, allow_extrapolation=())

    def _initialize(
        self,
        *,
        variables: Sequence[OptimizationVariable],
        objectives: Sequence[CostObjective],
        optimizer: CMAESSpec,
        allow_extrapolation: Sequence[ParameterRef],
    ) -> None:
        checked_variables = tuple(variables)
        checked_objectives = tuple(objectives)
        if not checked_variables or not all(isinstance(item, OptimizationVariable) for item in checked_variables):
            raise InvalidOptimizationSpec("variables must be nonempty OptimizationVariable values", stage="spec_validation")
        if not checked_objectives or not all(isinstance(item, CostObjective) for item in checked_objectives):
            raise InvalidOptimizationSpec("objectives must be nonempty CostObjective values", stage="spec_validation")
        if not isinstance(optimizer, CMAESSpec):
            raise InvalidOptimizationSpec("optimizer must be CMAESSpec", stage="spec_validation")
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
        )
        return instance

    def show(self) -> HtmlPresentation:
        """Present model defaults, active overrides, objectives, and CMA controls."""

        rows = "".join(
            "<tr>"
            f"<td>{escape('.'.join(_parameter_key(variable.parameter)))}</td>"
            f"<td>{escape(_quantity_pair_text(variable.model_default_bounds))}</td>"
            f"<td>{escape(_quantity_pair_text(variable.consumer_override_bounds))}</td>"
            f"<td>{escape(_quantity_pair_text(variable.bounds))}</td>"
            f"<td>{escape(variable.transform)}</td>"
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
        controls = (
            f"seed={self.optimizer.seed}; max_evaluations={self.optimizer.max_evaluations}; "
            f"population_size={self.optimizer.population_size}; initial_sigma={self.optimizer.initial_sigma}"
        )
        return HtmlPresentation(
            "<table><thead><tr><th>parameter</th><th>model default</th><th>consumer override</th><th>resolved</th><th>transform</th></tr></thead>"
            f"<tbody>{rows}</tbody></table><h3>objectives</h3><ul>{objectives}</ul><h3>optimizer</h3><p>{escape(controls)}</p>"
        )

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "type": "optimization", "variables": tuple(item._canonical_record() for item in self.variables),
            "objectives": tuple(item._canonical_record() for item in self.objectives),
            "optimizer": self.optimizer._canonical_record(), "allow_extrapolation": self.allow_extrapolation,
        }


@dataclass(frozen=True, slots=True)
class PumpAxis:
    """Name one independent positive-frequency HB pump fundamental."""

    id: str
    _frequency: Quantity

    def __init__(self, *, id: str, frequency: Quantity) -> None:
        id = _identifier(id, field="pump-axis id")
        units.require_positive_quantity(frequency, "hertz", name="pump-axis frequency")
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "_frequency", _detached_quantity(frequency))

    @property
    def frequency(self) -> Quantity:
        return quantity_view(self._frequency)

    def _canonical_record(self) -> Mapping[str, object]:
        return {"id": self.id, "frequency": self.frequency}


@dataclass(frozen=True, slots=True)
class CurrentDrive:
    """Declare one logical-Port HB Fourier-current injection channel."""

    id: str
    at: PortRef
    mode: tuple[int, ...]

    def __init__(self, *, id: str, at: PortRef, mode: tuple[int, ...]) -> None:
        id = _identifier(id, field="HB drive id")
        if not isinstance(at, PortRef):
            raise TypeError("CurrentDrive.at must be a PortRef")
        checked = _mode_tuple(mode, name="CurrentDrive.mode")
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "at", at)
        object.__setattr__(self, "mode", checked)

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "id": self.id,
            "port_id": self.at.id,
            "mode": self.mode,
            "orientation": "port_node_to_reference",
        }


@dataclass(frozen=True, slots=True)
class HBCaseSpec:
    """Name one immutable set of declared HB drive coefficients."""

    id: str
    _currents: Mapping[CurrentDrive, Quantity]

    def __init__(self, *, id: str, currents: Mapping[CurrentDrive, Quantity]) -> None:
        id = _identifier(id, field="HB case id")
        if not isinstance(currents, Mapping):
            raise TypeError("HBCaseSpec.currents must be a mapping")
        checked: dict[CurrentDrive, Quantity] = {}
        for drive, current in currents.items():
            if not isinstance(drive, CurrentDrive):
                raise TypeError("HBCaseSpec current keys must be CurrentDrive values")
            _validate_complex_current(current, name=f"current for drive {drive.id!r}")
            checked[drive] = _detached_quantity(current)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "_currents", MappingProxyType(checked))

    @property
    def currents(self) -> Mapping[CurrentDrive, Quantity]:
        return MappingProxyType({drive: quantity_view(current) for drive, current in self._currents.items()})

@dataclass(frozen=True, slots=True)
class HBTruncation:
    """Declare the request-global finite HB operating and response lattices."""

    pump_harmonics: tuple[int, ...]
    modulation_harmonics: tuple[int, ...]
    three_wave_mixing: bool
    four_wave_mixing: bool
    max_intermodulation_order: int | None

    def __init__(
        self,
        *,
        pump_harmonics: tuple[int, ...],
        modulation_harmonics: tuple[int, ...],
        three_wave_mixing: bool,
        four_wave_mixing: bool,
        max_intermodulation_order: int | None = None,
    ) -> None:
        pump = _nonnegative_int_tuple(pump_harmonics, name="pump_harmonics")
        modulation = _nonnegative_int_tuple(modulation_harmonics, name="modulation_harmonics")
        if not isinstance(three_wave_mixing, bool) or not isinstance(four_wave_mixing, bool):
            raise TypeError("HB mixing selections must be bool")
        if (
            max_intermodulation_order is not None
            and (not isinstance(max_intermodulation_order, int) or isinstance(max_intermodulation_order, bool) or max_intermodulation_order < 0)
        ):
            raise ValueError("max_intermodulation_order must be a nonnegative integer or None")
        object.__setattr__(self, "pump_harmonics", pump)
        object.__setattr__(self, "modulation_harmonics", modulation)
        object.__setattr__(self, "three_wave_mixing", three_wave_mixing)
        object.__setattr__(self, "four_wave_mixing", four_wave_mixing)
        object.__setattr__(self, "max_intermodulation_order", max_intermodulation_order)

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "pump_harmonics": self.pump_harmonics,
            "modulation_harmonics": self.modulation_harmonics,
            "max_intermodulation_order": self.max_intermodulation_order,
            "three_wave_mixing": self.three_wave_mixing,
            "four_wave_mixing": self.four_wave_mixing,
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


@dataclass(frozen=True, slots=True)
class HBSolveSpec:
    """Request one immutable shared-basis nonlinear HB batch."""

    pump_axes: tuple[PumpAxis, ...]
    drives: tuple[CurrentDrive, ...]
    _frequencies: Quantity
    cases: tuple[HBCaseSpec, ...]
    truncation: HBTruncation
    traces: tuple[SParameterTrace, ...]
    allow_driven_ptc: bool

    def __init__(
        self,
        *,
        pump_axes: Sequence[PumpAxis],
        drives: Sequence[CurrentDrive],
        frequencies: Quantity,
        cases: Sequence[HBCaseSpec],
        truncation: HBTruncation,
        traces: Sequence[SParameterTrace] = (),
        allow_driven_ptc: bool = False,
    ) -> None:
        axes = tuple(pump_axes)
        declared_drives = tuple(drives)
        declared_cases = tuple(cases)
        declared_traces = tuple(traces)
        if not all(isinstance(axis, PumpAxis) for axis in axes):
            raise TypeError("pump_axes must contain PumpAxis values")
        if len({axis.id for axis in axes}) != len(axes):
            raise ValueError("pump-axis IDs must be unique")
        if not all(isinstance(drive, CurrentDrive) for drive in declared_drives):
            raise TypeError("drives must contain CurrentDrive values")
        if len({drive.id for drive in declared_drives}) != len(declared_drives):
            raise ValueError("HB drive IDs must be unique")
        if not isinstance(truncation, HBTruncation):
            raise TypeError("truncation must be an HBTruncation")
        if len(truncation.pump_harmonics) != len(axes) or len(truncation.modulation_harmonics) != len(axes):
            raise ValueError("HB truncation rank must equal pump-axis rank")
        _validate_frequency_grid(frequencies)
        if not all(isinstance(case, HBCaseSpec) for case in declared_cases) or not declared_cases:
            raise ValueError("cases must be a nonempty sequence of HBCaseSpec values")
        if len({case.id for case in declared_cases}) != len(declared_cases):
            raise ValueError("HB case IDs must be unique")
        if not all(isinstance(trace, SParameterTrace) for trace in declared_traces):
            raise TypeError("traces must contain SParameterTrace values")
        if len({trace.id for trace in declared_traces}) != len(declared_traces):
            raise ValueError("HB trace IDs must be unique")
        if not isinstance(allow_driven_ptc, bool):
            raise TypeError("allow_driven_ptc must be bool")

        rank = len(axes)
        declared_by_mode: set[tuple[str, tuple[int, ...]]] = set()
        for drive in declared_drives:
            _validate_hb_mode(
                drive.mode,
                rank=rank,
                limits=truncation.pump_harmonics,
                truncation=truncation,
                name=f"drive {drive.id!r}",
            )
            key = (drive.at.id, drive.mode)
            inverse = (drive.at.id, tuple(-item for item in drive.mode))
            if drive.mode and any(drive.mode) and inverse in declared_by_mode:
                raise ValueError("HB drives must not independently declare conjugate modes at one Port")
            declared_by_mode.add(key)
        for trace in declared_traces:
            _validate_hb_mode(
                trace.input_mode,
                rank=rank,
                limits=truncation.modulation_harmonics,
                truncation=truncation,
                name=f"trace {trace.id!r} input_mode",
            )
            _validate_hb_mode(
                trace.output_mode,
                rank=rank,
                limits=truncation.modulation_harmonics,
                truncation=truncation,
                name=f"trace {trace.id!r} output_mode",
            )
        declared_set = set(declared_drives)
        for case in declared_cases:
            for drive, current in case._currents.items():
                if drive not in declared_set:
                    raise ValueError("HB case binds a CurrentDrive not declared by this HBSolveSpec")
                if not any(declared is drive for declared in declared_drives):
                    raise ValueError("HB case must bind the exact declared CurrentDrive object")
                if not any(drive.mode) and complex(current.to("ampere").magnitude).imag != 0.0:
                    raise ValueError("a DC HB current coefficient must be real")
        object.__setattr__(self, "pump_axes", axes)
        object.__setattr__(self, "drives", declared_drives)
        object.__setattr__(self, "_frequencies", _detached_quantity(frequencies))
        object.__setattr__(self, "cases", declared_cases)
        object.__setattr__(self, "truncation", truncation)
        object.__setattr__(self, "traces", declared_traces)
        object.__setattr__(self, "allow_driven_ptc", allow_driven_ptc)

    @property
    def frequencies(self) -> Quantity:
        return quantity_view(self._frequencies)

    def _canonical_record(self) -> Mapping[str, object]:
        return {
            "type": "hb_solve",
            "pump_axes": tuple(axis._canonical_record() for axis in self.pump_axes),
            "drives": tuple(drive._canonical_record() for drive in self.drives),
            "frequencies": self.frequencies,
            "cases": tuple(
                {
                    "id": case.id,
                    "currents": tuple(
                        {
                            "drive_id": drive.id,
                            "coefficient": current,
                            "coefficient_convention": "exp_minus_i_m_dot_omega_t_fourier_coefficient",
                        }
                        for drive, current in self._ordered_case_currents(case)
                    ),
                }
                for case in self.cases
            ),
            "truncation": self.truncation._canonical_record(),
            "traces": tuple(trace._canonical_record() for trace in self.traces),
            "allow_driven_ptc": self.allow_driven_ptc,
        }

    def _ordered_case_currents(self, case: HBCaseSpec) -> tuple[tuple[CurrentDrive, Quantity], ...]:
        return tuple((drive, case._currents[drive]) for drive in self.drives if drive in case._currents)


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
    from ._numeric_presentation import _channel_index

    return channels[_channel_index(channels, selector, role=role, default=False)]


def _declared_hb_trace_channels(
    result: HBBatchResult,
) -> Mapping[str, tuple[tuple[str, tuple[int, ...]], tuple[str, tuple[int, ...]]]]:
    presentation = getattr(result, "_presentation", {})
    declarations = (
        presentation.get("declared_traces")
        if isinstance(presentation, Mapping)
        else None
    )
    if not isinstance(declarations, Mapping):
        raise ValueError("HB Result has no verified declared-trace presentation metadata")
    normalized: dict[
        str, tuple[tuple[str, tuple[int, ...]], tuple[str, tuple[int, ...]]]
    ] = {}
    for identifier, declaration in declarations.items():
        if not isinstance(identifier, str) or not isinstance(declaration, Mapping):
            raise ValueError("HB Result declared-trace presentation metadata is malformed")
        channels: list[tuple[str, tuple[int, ...]]] = []
        for name in ("input_channel", "output_channel"):
            channel = declaration.get(name)
            coordinate = channel.get("coordinate") if isinstance(channel, Mapping) else None
            mode = channel.get("mode") if isinstance(channel, Mapping) else None
            if (
                not isinstance(coordinate, str)
                or not coordinate
                or not isinstance(mode, (tuple, list))
                or any(not isinstance(value, int) or isinstance(value, bool) for value in mode)
            ):
                raise ValueError("HB Result declared-trace presentation metadata is malformed")
            channels.append((coordinate, tuple(mode)))
        normalized[identifier] = (channels[0], channels[1])
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ReportPanel:
    """One closed presentation selection for an exact verified Result."""

    result: AnalysisResult
    kind: Literal["response", "matrix", "summary", "outcomes", "history"]
    family: Literal["S", "Y", "Z"] | None
    case: str | None
    trace: str | None
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
        result: AnalysisResult,
        kind: Literal["response", "matrix", "summary", "outcomes", "history"] | None = None,
        family: Literal["S", "Y", "Z"] | None = None,
        case: str | None = None,
        trace: str | None = None,
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
        if not _is_verified_analysis_result(result):
            raise TypeError("ReportPanel.result must be a verified AnalysisResult")
        if family is not None:
            _family(family)
        if case is not None and (not isinstance(case, str) or not case):
            raise TypeError("case must be a nonempty case ID or None")
        if trace is not None and (not isinstance(trace, str) or not trace):
            raise TypeError("trace must be a nonempty trace ID or None")
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
            for value in (family, case, trace, input_channel, output_channel, component, magnitude)
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
            elif isinstance(result, DirectSolveResult):
                kind = "response" if len(result.s.view.input_channels) == len(result.s.view.output_channels) == 1 else "summary"
            elif isinstance(result, (HBBatchResult, ParameterSweepResult)):
                kind = "outcomes"
            elif isinstance(result, OptimizationResult):
                kind = "history"
            else:
                kind = "summary"
        if kind not in {"response", "matrix", "summary", "outcomes", "history"}:
            raise ValueError("kind must be 'response', 'matrix', 'summary', 'outcomes', or 'history'")

        if kind == "response":
            if not isinstance(result, (DirectSolveResult, HBBatchResult)):
                raise TypeError("response panels require a Direct or HB Result")
            if matrix_requested or history_requested:
                raise ValueError("response panels do not accept matrix or history selectors")
            if trace is not None and (input_channel is not None or output_channel is not None):
                raise ValueError("trace and matrix-channel selection are mutually exclusive")
            if (input_channel is None) != (output_channel is None):
                raise ValueError("input_channel and output_channel must be supplied together")
            if isinstance(result, DirectSolveResult) and (case is not None or trace is not None):
                raise ValueError("Direct response panels do not accept HB case or trace selectors")
            if isinstance(result, HBBatchResult) and family not in {None, "S"}:
                raise ValueError("HB response panels support only the S family")
            family = "S" if family is None else family
            component = "magnitude" if component is None else component
            magnitude = "linear" if magnitude is None else magnitude
            if magnitude == "db" and (family != "S" or component != "magnitude"):
                raise ValueError("dB is supported only for S magnitude")
            if component != "magnitude" and magnitude != "linear":
                raise ValueError("dB magnitude is incompatible with a non-magnitude component")
            if isinstance(result, DirectSolveResult):
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
            else:
                if case is not None and case not in result.cases:
                    raise ValueError(f"unknown HB case: {case}")
                selected_cases = (
                    (result.cases[case],)
                    if case is not None
                    else tuple(result.cases.values())
                )
                successful = tuple(
                    outcome for outcome in selected_cases if outcome.succeeded
                )
                declared_traces = _declared_hb_trace_channels(result)
                if trace is not None:
                    if trace not in declared_traces:
                        raise ValueError(f"unknown HB trace: {trace}")
                    if any(trace not in outcome.traces for outcome in successful):
                        raise ValueError(f"unknown HB trace: {trace}")
                if trace is None and input_channel is None and successful:
                    if any(
                        len(outcome.s.view.input_channels) != 1
                        or len(outcome.s.view.output_channels) != 1
                        for outcome in successful
                    ):
                        raise ValueError("multichannel HB response panels require a trace or exact channels")
                if input_channel is not None:
                    if successful:
                        first = successful[0].s.view
                        input_channel = _exact_report_channel(
                            first.input_channels, input_channel, role="HB input"
                        )
                        output_channel = _exact_report_channel(
                            first.output_channels, output_channel, role="HB output"
                        )
                        for outcome in successful[1:]:
                            view = outcome.s.view
                            if (
                                input_channel not in view.input_channels
                                or output_channel not in view.output_channels
                            ):
                                raise ValueError(
                                    "selected HB channels are absent from a successful case"
                                )
                    else:
                        declared_pairs = tuple(declared_traces.values())
                        declared_inputs = tuple(dict.fromkeys(
                            item[0] for item in declared_pairs
                        ))
                        declared_outputs = tuple(dict.fromkeys(
                            item[1] for item in declared_pairs
                        ))
                        input_channel = _exact_report_channel(
                            declared_inputs,
                            input_channel,
                            role="HB declared input",
                        )
                        output_channel = _exact_report_channel(
                            declared_outputs,
                            output_channel,
                            role="HB declared output",
                        )
                        if (input_channel, output_channel) not in declared_pairs:
                            raise ValueError(
                                "failed HB response channels must match one declared trace"
                            )
            history_metric = None
        elif kind == "matrix":
            if not isinstance(result, (DirectSolveResult, OperatorResult)):
                raise TypeError("matrix panels require a Direct or Operator Result")
            if history_requested or any(value is not None for value in (case, trace, input_channel, output_channel)):
                raise ValueError("matrix panels accept no response-channel, case, trace, or history selectors")
            if frequency is None:
                raise ValueError("matrix panels require an exact frequency")
            matrix_style = "table" if matrix_style is None else matrix_style
            if isinstance(result, OperatorResult) and family is not None:
                raise ValueError("Operator matrix panels do not accept an S/Y/Z family")
            if isinstance(result, DirectSolveResult):
                family = "S" if family is None else family
                view = result._family(family).view
                from ._numeric_presentation import _frequency_index

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
                if isinstance(result, OperatorResult) and magnitude != "linear":
                    raise ValueError("Operator heatmaps do not accept dB magnitude")
                if isinstance(result, DirectSolveResult) and magnitude == "db" and (
                    family != "S" or component != "magnitude"
                ):
                    raise ValueError("dB is supported only for S magnitude")
            history_metric = None
        elif kind == "history":
            if not isinstance(result, OptimizationResult):
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
            from ._numeric_presentation import _optimization_series

            _optimization_series(
                result,
                kind="history" if history_metric == "cost" else history_metric,
                objective=objective,
                parameter=parameter,
            )
        else:
            if matrix_requested or history_requested or response_requested:
                raise ValueError(f"{kind} panels accept no selection fields")
            if kind == "outcomes" and not isinstance(result, (HBBatchResult, ParameterSweepResult)):
                raise TypeError("outcomes panels require an HB batch or parameter sweep")
            history_metric = None

        object.__setattr__(self, "result", result)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "case", case)
        object.__setattr__(self, "trace", trace)
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
    """Choose exact existing Analysis Results for a pure derived report."""

    inputs: tuple[AnalysisResult, ...]
    theme: Theme = Theme.AUTO
    panels: tuple[ReportPanel, ...] = ()

    def __init__(
        self,
        *,
        inputs: Sequence[AnalysisResult],
        theme: Theme = Theme.AUTO,
        panels: Sequence[ReportPanel] = (),
    ) -> None:
        checked = tuple(inputs)
        if not checked or not all(_is_verified_analysis_result(item) for item in checked):
            raise TypeError("ReportSpec.inputs must be nonempty AnalysisResult values")
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
    from ._diagram_spec import CircuitDiagramSpec

    globals()[name] = CircuitDiagramSpec
    return CircuitDiagramSpec
