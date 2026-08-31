"""Immutable public declarations for SCNSim requests.

Specs describe an operation; they neither execute it nor own a result.  The
runtime performs Plan-specific validation when it binds one of these values to
a sealed Plan.  Keeping these values small and immutable makes the canonical
request encoder the single identity authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from math import isfinite
from typing import Literal

import numpy as np
from pint import Quantity

from . import units
from ._canonical import _identifier
from ._scaffold import unavailable
from .authoring import CoordinateRef, ElectricNodeRef, ParameterRef, PortRef
from .errors import InvalidDiagonalRootHint, InvalidOptimizationSpec
from .results import AnalysisResult, HtmlPresentation, _is_verified_analysis_result


Coordinate = str | ElectricNodeRef | CoordinateRef


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
    component_id = getattr(value, "component_id", None)
    parameter_id = getattr(value, "id", None)
    if isinstance(component_id, str) and isinstance(parameter_id, str):
        return component_id, parameter_id
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


def _family(value: str) -> Literal["S", "Y", "Z"]:
    if value not in {"S", "Y", "Z"}:
        raise ValueError("family must be 'S', 'Y', or 'Z'")
    return value  # type: ignore[return-value]


def _selector_unit(value: object) -> str | None:
    if not isinstance(value, QuantitySelector):
        return None
    if value.type in {"diagonal_root_projection", "hybridized_pole_projection", "transfer_zero_projection"}:
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
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "traces", checked)

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "direct_solve", "frequencies": self.frequencies, "traces": tuple(trace._canonical_record() for trace in self.traces)}


@dataclass(frozen=True, slots=True)
class QuantitySelector:
    """A non-executing scalar projection of one typed Direct quantity Spec."""

    spec: object
    projection: str
    type: str

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
    root_hint: Quantity

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
        object.__setattr__(self, "root_hint", root_hint)

    @property
    def frequency(self) -> QuantitySelector:
        return QuantitySelector(self, "frequency", "diagonal_root_projection")

    @property
    def linewidth(self) -> QuantitySelector:
        return QuantitySelector(self, "linewidth", "diagonal_root_projection")

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "diagonal_root", "coordinate": _coordinate_id(self.coordinate), "root_hint": self.root_hint}


@dataclass(frozen=True, slots=True)
class HybridizedPoleSpec:
    """Select an anchored complex pole of a retained coupled block.

    It cannot be substituted with a diagonal root or a nearest sampled peak.
    """

    coordinates: tuple[Coordinate, ...]
    anchor: Quantity

    def __init__(self, *, coordinates: Sequence[Coordinate], anchor: Quantity) -> None:
        checked = tuple(coordinates)
        identifiers = tuple(_coordinate_id(value) for value in checked)
        if len(identifiers) < 2 or len(set(identifiers)) != len(identifiers):
            raise ValueError("HybridizedPoleSpec requires at least two unique coordinates")
        _validate_frequency_anchor(anchor, name="anchor")
        object.__setattr__(self, "coordinates", checked)
        object.__setattr__(self, "anchor", anchor)

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

    anchor: Quantity
    family: Literal["S", "Y", "Z"]
    input_coordinate: Coordinate
    output_coordinate: Coordinate

    def __init__(self, *, anchor: Quantity, family: Literal["S", "Y", "Z"], input_coordinate: Coordinate, output_coordinate: Coordinate) -> None:
        _validate_frequency_anchor(anchor, name="anchor")
        _family(family)
        _coordinate_id(input_coordinate)
        _coordinate_id(output_coordinate)
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "input_coordinate", input_coordinate)
        object.__setattr__(self, "output_coordinate", output_coordinate)

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
        object.__setattr__(self, "frequency", frequency)

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
        object.__setattr__(self, "frequency", frequency)

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
        object.__setattr__(self, "frequencies", frequencies)

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
        object.__setattr__(self, "model_default_bounds", bounds)
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
        object.__setattr__(instance, "consumer_override_bounds", bounds)
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
    """The V1 scalar composition: a sum of same-dimensionality selectors."""

    terms: tuple[object, ...]

    def __init__(self, *terms: object) -> None:
        if not terms:
            raise InvalidOptimizationSpec("QuantitySum requires one or more terms", stage="spec_validation")
        dimensions = tuple(units.registry.Unit(_validate_selector(term)).dimensionality for term in terms)
        if len(set(dimensions)) != 1:
            raise InvalidOptimizationSpec("QuantitySum terms must share one dimensionality", stage="spec_validation")
        object.__setattr__(self, "terms", tuple(terms))

    def _canonical_record(self) -> Mapping[str, object]:
        return {"type": "quantity_sum", "terms": tuple(_canonical_value(item) for item in self.terms)}


@dataclass(frozen=True, slots=True)
class CostObjective:
    """Compare one scalar quantity with one target inside optimization."""

    id: str
    quantity: object
    target: Quantity
    weight: Quantity
    scale: Quantity | None = None

    def __init__(
        self,
        *,
        id: str,
        quantity: object,
        target: Quantity,
        weight: Quantity,
        scale: Quantity | None = None,
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "scale", scale)
        self.__post_init__()

    def __post_init__(self) -> None:
        try:
            identifier = _identifier(self.id, field="objective id")
        except Exception as exc:
            raise InvalidOptimizationSpec("objective id must be a canonical identifier", stage="spec_validation") from exc
        object.__setattr__(self, "id", identifier)
        if isinstance(self.quantity, QuantitySum):
            quantity_unit = _selector_unit(self.quantity.terms[0])
        else:
            quantity_unit = _selector_unit(self.quantity)
        if quantity_unit is None:
            raise InvalidOptimizationSpec("objective quantity must be a scalar selector or QuantitySum", stage="spec_validation")
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
        return {"id": self.id, "quantity": _canonical_value(self.quantity), "target": self.target, "weight": self.weight, "scale": self.scale}


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
            f"<li>{escape(objective.id)}: {escape(_selector_text(objective.quantity))}</li>"
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
    """Name one independent HB pump axis; implemented in the HB ``dev6`` slice."""

    def __init__(self, *, id: str, frequency: Quantity) -> None:
        unavailable("PumpAxis construction")


@dataclass(frozen=True, slots=True)
class CurrentDrive:
    """Declare one logical-Port HB drive; unavailable before ``dev6``."""

    def __init__(self, *, id: str, at: PortRef, mode: tuple[int, ...]) -> None:
        unavailable("CurrentDrive construction")


@dataclass(frozen=True, slots=True)
class HBCaseSpec:
    """Name one HB operating condition; unavailable before ``dev6``."""

    def __init__(self, *, id: str, currents: Mapping[CurrentDrive, Quantity]) -> None:
        unavailable("HBCaseSpec construction")


@dataclass(frozen=True, slots=True)
class HBTruncation:
    """Declare a finite HB mode lattice; unavailable before ``dev6``."""

    def __init__(
        self,
        *,
        pump_harmonics: tuple[int, ...],
        modulation_harmonics: tuple[int, ...],
        three_wave_mixing: bool,
        four_wave_mixing: bool,
        max_intermodulation_order: int | None = None,
    ) -> None:
        unavailable("HBTruncation construction")


@dataclass(frozen=True, slots=True)
class SParameterTrace:
    """Name a selected-matrix S projection for Direct; HB use remains dev6-only."""

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
    """Request a shared-basis nonlinear HB case batch; unavailable before ``dev6``."""

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
        unavailable("HBSolveSpec construction")


@dataclass(frozen=True, slots=True)
class ReportSpec:
    """Choose exact existing Analysis Results for a pure derived report."""

    inputs: tuple[AnalysisResult, ...]

    def __init__(self, *, inputs: Sequence[AnalysisResult]) -> None:
        checked = tuple(inputs)
        if not checked or not all(_is_verified_analysis_result(item) for item in checked):
            raise TypeError("ReportSpec.inputs must be nonempty AnalysisResult values")
        object.__setattr__(self, "inputs", checked)


def _canonical_value(value: object) -> object:
    record = getattr(value, "_canonical_record", None)
    return record() if callable(record) else value


@dataclass(frozen=True, slots=True)
class CircuitDiagramSpec:
    representation: Literal["authoring", "compiled"] = "authoring"
    theme: Literal["auto", "light", "dark"] = "auto"
    show_parameter_values: bool = False

    def __init__(
        self,
        *,
        representation: Literal["authoring", "compiled"] = "authoring",
        theme: Literal["auto", "light", "dark"] = "auto",
        show_parameter_values: bool = False,
    ) -> None:
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "theme", theme)
        object.__setattr__(self, "show_parameter_values", show_parameter_values)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.representation not in {"authoring", "compiled"} or self.theme not in {"auto", "light", "dark"}:
            raise ValueError("invalid diagram representation or theme")
