"""Public authoring declarations for primitive, Composite, and RLGC circuits.

The sealed Plan remains the sole physical authority. HB authoring retains its
declared fail-fast boundary for the dev6 slice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import inspect
import struct
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
import unicodedata

import numpy as np

from ._scaffold import unavailable
from .errors import PlanSealedError, SCNSimValidationError
from .units import Quantity, registry, require_positive_quantity, require_quantity

if TYPE_CHECKING:
    from .results import CircuitDiagramResult
    from .specs import CircuitDiagramSpec


def _identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    value = unicodedata.normalize("NFC", value)
    if not value or any(ord(char) < 32 or ord(char) == 127 or char in "/\\" for char in value):
        raise ValueError(f"{field} must be a nonempty portable identifier")
    return value


def _quantity_record(value: Quantity, unit: str) -> dict[str, str]:
    """Encode one physical scalar through the single canonical quantity rule."""

    from ._canonical import quantity_envelope

    return quantity_envelope(value, si_unit=unit, registry=registry)


_UNITS = ("farad", "henry", "ohm", "siemens", "hertz", "meter", "weber", "ampere", "volt", "dimensionless")
_factory_context: ContextVar[tuple[Any, str] | None] = ContextVar("scnsim_library_factory", default=None)


def _unit_name(unit: object) -> str:
    """Resolve one declared Pint unit to SCNSim's closed physical vocabulary."""

    if getattr(unit, "_REGISTRY", None) is not registry:
        raise TypeError("unit must be a Pint Unit from scnsim.units.registry")
    value = registry.Quantity(1, unit)
    for candidate in _UNITS:
        try:
            require_quantity(value, candidate, name="unit")
        except ValueError:
            continue
        return candidate
    raise ValueError("unit is outside SCNSim's supported physical vocabulary")


def _binding(value: object, unit: str, *, name: str, positive: bool = False) -> tuple[Quantity, dict[str, object]]:
    """Close a child physical slot as a constant, identity, or affine binding."""

    if isinstance(value, ParameterRef):
        baseline = require_quantity(value.baseline, unit, name=name)
        if positive and baseline.to(unit).magnitude <= 0:
            raise ValueError(f"{name} baseline must be strictly positive")
        return baseline, {
            "kind": "identity",
            "input": value._canonical_ref(),
            "_input_ref": value,
        }
    if isinstance(value, AffineMap):
        baseline = value._value_at_baseline(unit=unit, name=name)
        if positive and baseline.to(unit).magnitude <= 0:
            raise ValueError(f"{name} baseline must be strictly positive")
        binding = value._canonical_binding()
        binding["_input_ref"] = value.input
        binding["_affine_source"] = MappingProxyType(
            {
                "slope": value.slope,
                "intercept": value.intercept,
                "support": value._support,
            }
        )
        return baseline, binding
    validator = require_positive_quantity if positive else require_quantity
    baseline = validator(value, unit, name=name)
    return baseline, {"kind": "constant", "value": _quantity_record(baseline, unit)}


class PinRef:
    """Stable reference to one named electrical terminal of a component."""

    __slots__ = ("_component", "_name")

    def __init__(self) -> None:
        unavailable("PinRef construction")

    @classmethod
    def _create(cls, component: ComponentInstance, name: str) -> PinRef:
        instance = object.__new__(cls)
        instance._component = component
        instance._name = name
        return instance

    @property
    def name(self) -> str:
        return self._name

    @property
    def component_id(self) -> str:
        """Owning component ID, useful only for authoring inspection."""

        return self._component.id

    def _endpoint(self) -> dict[str, object]:
        return {"component_path": [self._component.id], "pin_id": self.name}


class ElectricNodeRef:
    """Plan-bound handle to one declared equipotential electric node."""

    __slots__ = ("_plan", "_node")

    def __init__(self) -> None:
        unavailable("ElectricNodeRef construction")

    @classmethod
    def _create(cls, plan: CircuitPlan, node: _PlanNode) -> ElectricNodeRef:
        instance = object.__new__(cls)
        instance._plan = plan
        instance._node = node
        return instance

    @property
    def id(self) -> str:
        """Current public or opaque canonical node identity."""

        return self._node.id

    @property
    def is_public(self) -> bool:
        """Whether this node is selectable as a public Runtime coordinate."""

        return self._node.visibility != "internal"


class CoordinateRef:
    """Public handle to one observable coordinate inside a later composite."""

    __slots__ = ("_component", "_name")

    def __init__(self) -> None:
        unavailable("CoordinateRef construction")

    @classmethod
    def _create(cls, component: ComponentInstance, name: str) -> CoordinateRef:
        instance = object.__new__(cls)
        instance._component = component
        instance._name = name
        return instance

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        """Exact public coordinate identity selected by Runtime requests."""

        return f"{self.component_id}.{self.name}"

    @property
    def component_id(self) -> str:
        return self._component.id

    def _canonical_ref(self) -> dict[str, object]:
        return {"component_path": [self.component_id], "coordinate_id": self.id}


class PortRef:
    """Plan-bound handle to one logical Port component."""

    __slots__ = ("_plan", "_id", "_node", "_role", "_reference_impedance")

    def __init__(self) -> None:
        unavailable("PortRef construction")

    @classmethod
    def _create(
        cls,
        plan: CircuitPlan,
        id: str,
        node: ElectricNodeRef,
        role: Literal["terminated", "nonloading_probe"],
        reference_impedance: Quantity,
    ) -> PortRef:
        instance = object.__new__(cls)
        instance._plan = plan
        instance._id = id
        instance._node = node
        instance._role = role
        instance._reference_impedance = reference_impedance
        return instance

    @property
    def id(self) -> str:
        return self._id

    @property
    def role(self) -> Literal["terminated", "nonloading_probe"]:
        return self._role

    @property
    def reference_impedance(self) -> Quantity:
        return self._reference_impedance

    @property
    def node(self) -> ElectricNodeRef:
        """Electric node attached to this logical Port."""

        return self._node


class ParameterRef:
    """Stable public handle to one primitive or sealed Composite parameter."""

    __slots__ = ("_component", "_id", "_baseline", "_unit")

    def __init__(self) -> None:
        unavailable("ParameterRef construction")

    @classmethod
    def _create(
        cls, component: ComponentInstance, id: str, baseline: Quantity, unit: str
    ) -> ParameterRef:
        instance = object.__new__(cls)
        instance._component = component
        instance._id = id
        instance._baseline = baseline
        instance._unit = unit
        return instance

    @property
    def id(self) -> str:
        return self._id

    @property
    def baseline(self) -> Quantity:
        return self._baseline

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def component_id(self) -> str:
        """Owning component ID."""

        return self._component.id

    def _canonical_ref(self) -> dict[str, object]:
        return {"component_path": [self.component_id], "parameter_id": self.id}

    def show(self) -> object:
        """Inspect sealed Composite fan-out without exposing child handles."""

        public_map = next(
            (
                item
                for item in self._component._realization.get("public_parameter_maps", ())
                if item["parameter"] == self._canonical_ref()
            ),
            None,
        )
        consumers = () if public_map is None else tuple(
            sorted(
                public_map["consumers"],
                key=lambda item: (tuple(item["target"]["component_path"]), item["target"]["parameter_id"]),
            )
        )
        fan_out = tuple(_public_parameter_ref(item["target"]) for item in consumers)
        affine_maps = tuple(
            _public_affine_map(item["target"], item["binding"])
            for item in consumers
            if item["binding"]["kind"] == "affine"
        )

        return MappingProxyType(
            {
                "component_id": self.component_id,
                "parameter_id": self.id,
                "baseline": self.baseline,
                "unit": self.unit,
                "fan_out": fan_out,
                "affine_maps": affine_maps,
            }
        )


def _public_parameter_ref(reference: Mapping[str, object]) -> Mapping[str, object]:
    """Return one immutable canonical parameter identity for inspection."""

    return MappingProxyType(
        {
            "component_path": tuple(reference["component_path"]),
            "parameter_id": reference["parameter_id"],
        }
    )


def _public_affine_map(
    target: Mapping[str, object], binding: Mapping[str, object]
) -> Mapping[str, object]:
    """Decode one sealed affine binding without reintroducing child handles."""

    from ._canonical import quantity_from_envelope

    return MappingProxyType(
        {
            "target": _public_parameter_ref(target),
            "input": _public_parameter_ref(binding["input"]),
            "slope": quantity_from_envelope(binding["slope"], registry=registry),
            "intercept": quantity_from_envelope(binding["intercept"], registry=registry),
            "support": tuple(
                quantity_from_envelope(value, registry=registry)
                for value in binding["support"]
            ),
        }
    )


class InductiveBranchRef:
    """Stable oriented handle to an inductive branch exposed for dev4 coupling."""

    __slots__ = ("_component", "_id")

    def __init__(self) -> None:
        unavailable("InductiveBranchRef construction")

    @classmethod
    def _create(cls, component: ComponentInstance, id: str) -> InductiveBranchRef:
        instance = object.__new__(cls)
        instance._component = component
        instance._id = id
        return instance

    @property
    def id(self) -> str:
        return self._id

    @property
    def component_id(self) -> str:
        return self._component.id

    def _canonical_ref(self) -> dict[str, object]:
        return {"component_path": [self.component_id], "branch_id": self.id}


class ParameterSpec:
    """Library declaration of one public physical parameter schema."""

    def __init__(self, *, unit: object) -> None:
        self._unit = _unit_name(unit)
        self.unit = unit

    @property
    def si_unit(self) -> str:
        return self._unit

    def _canonical_record(self) -> dict[str, str]:
        return {"si_unit": self._unit, "dimensionality": _quantity_record(registry.Quantity(1, self._unit), self._unit)["dimensionality"]}


class ParameterSet:
    """Immutable physical values bound to exact ``ParameterRef`` keys."""

    __slots__ = ("_values", "_allow_extrapolation")

    def __init__(
        self,
        values: Mapping[ParameterRef, Quantity],
        *,
        allow_extrapolation: Sequence[ParameterRef] = (),
    ) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("values must map ParameterRef objects to Quantities")
        bound: list[tuple[ParameterRef, Quantity]] = []
        for parameter, value in values.items():
            if not isinstance(parameter, ParameterRef):
                raise TypeError("ParameterSet keys must be ParameterRef objects")
            bound.append(
                (parameter, require_quantity(value, parameter.unit, name=parameter.id))
            )
        bound.sort(key=lambda item: (item[0].component_id, item[0].id))
        if len({parameter for parameter, _ in bound}) != len(bound):
            raise ValueError("ParameterSet contains a duplicate parameter")
        approved = tuple(allow_extrapolation)
        if any(not isinstance(parameter, ParameterRef) for parameter in approved):
            raise TypeError("allow_extrapolation must contain ParameterRef objects")
        approved = tuple(sorted(set(approved), key=lambda item: (item.component_id, item.id)))
        self._values = MappingProxyType(dict(bound))
        self._allow_extrapolation = approved

    @property
    def values(self) -> Mapping[ParameterRef, Quantity]:
        """Ordered read-only mapping used to create an explicitly new request."""

        return self._values

    @property
    def allow_extrapolation(self) -> tuple[ParameterRef, ...]:
        """Exact public parameters authorized for this one request."""

        return self._allow_extrapolation

    def _canonical_record(self) -> dict[str, object]:
        return {
            "type": "parameter_set",
            "bindings": [
                {
                    "parameter": parameter._canonical_ref(),
                    "value": _quantity_record(value, parameter.unit),
                }
                for parameter, value in self._values.items()
            ],
            "allow_extrapolation": [
                parameter._canonical_ref() for parameter in self._allow_extrapolation
            ],
        }


class RLGC:
    """Immutable ordered per-length matrices for one frozen transmission line."""

    __slots__ = (
        "_conductors", "_reference_conductor", "_resistance", "_inductance",
        "_conductance", "_capacitance", "_extraction_frequency", "_source",
        "_source_quantities",
    )

    def __init__(
        self,
        *,
        conductors: Sequence[str],
        reference_conductor: str,
        resistance_per_length: Quantity,
        inductance_per_length: Quantity,
        conductance_per_length: Quantity,
        capacitance_per_length: Quantity,
        extraction_frequency: Quantity | None = None,
    ) -> None:
        if isinstance(conductors, (str, bytes)) or not isinstance(conductors, Sequence):
            raise TypeError("conductors must be a nonempty sequence of identifiers")
        names = tuple(_identifier(name, field="conductor") for name in conductors)
        if not names:
            raise ValueError("RLGC requires at least one conductor")
        if len(set(names)) != len(names):
            raise SCNSimValidationError("RLGC conductor names must be unique", stage="authoring")
        reference = _identifier(reference_conductor, field="reference_conductor")
        if reference in names:
            raise SCNSimValidationError("reference_conductor cannot occur in the RLGC conductor basis", stage="authoring")

        size = len(names)
        resistance = _rlgc_matrix(resistance_per_length, "ohm / meter", size, name="resistance_per_length")
        inductance = _rlgc_matrix(inductance_per_length, "henry / meter", size, name="inductance_per_length")
        conductance = _rlgc_matrix(conductance_per_length, "siemens / meter", size, name="conductance_per_length")
        capacitance = _rlgc_matrix(capacitance_per_length, "farad / meter", size, name="capacitance_per_length")
        _validate_rlgc_matrices(resistance, inductance, conductance, capacitance)

        frequency: float | None
        if extraction_frequency is None:
            frequency = None
        else:
            value = require_positive_quantity(extraction_frequency, "hertz", name="extraction_frequency")
            frequency = float(value.to("hertz").magnitude)
        object.__setattr__(self, "_conductors", names)
        object.__setattr__(self, "_reference_conductor", reference)
        object.__setattr__(self, "_resistance", resistance)
        object.__setattr__(self, "_inductance", inductance)
        object.__setattr__(self, "_conductance", conductance)
        object.__setattr__(self, "_capacitance", capacitance)
        object.__setattr__(self, "_extraction_frequency", frequency)
        object.__setattr__(self, "_source", _freeze_rlgc_source({"source_kind": "manual"}))
        source_quantities = {
            "resistance_per_length": registry.Quantity(1.0, resistance_per_length.units),
            "inductance_per_length": registry.Quantity(1.0, inductance_per_length.units),
            "conductance_per_length": registry.Quantity(1.0, conductance_per_length.units),
            "capacitance_per_length": registry.Quantity(1.0, capacitance_per_length.units),
        }
        if extraction_frequency is not None:
            source_quantities["extraction_frequency"] = registry.Quantity(1.0, extraction_frequency.units)
        object.__setattr__(self, "_source_quantities", MappingProxyType(source_quantities))

    @classmethod
    def _from_source(cls, *, source: Mapping[str, object], **kwargs: object) -> "RLGC":
        value = cls(**kwargs)
        object.__setattr__(value, "_source", _freeze_rlgc_source(source))
        return value

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("RLGC values are immutable")

    @property
    def conductors(self) -> tuple[str, ...]:
        return self._conductors

    @property
    def reference_conductor(self) -> str:
        return self._reference_conductor

    @property
    def resistance_per_length(self) -> Quantity:
        return _rlgc_quantity(self._resistance, "ohm / meter")

    @property
    def inductance_per_length(self) -> Quantity:
        return _rlgc_quantity(self._inductance, "henry / meter")

    @property
    def conductance_per_length(self) -> Quantity:
        return _rlgc_quantity(self._conductance, "siemens / meter")

    @property
    def capacitance_per_length(self) -> Quantity:
        return _rlgc_quantity(self._capacitance, "farad / meter")

    @property
    def extraction_frequency(self) -> Quantity | None:
        return None if self._extraction_frequency is None else registry.Quantity(self._extraction_frequency, "hertz")

    def _canonical_record(self) -> dict[str, object]:
        return {
            "type": "rlgc",
            "conductors": list(self._conductors),
            "reference_conductor": self._reference_conductor,
            "orientation": "extractor_positive_z_is_head_to_tail",
            "resistance_per_length": _rlgc_matrix_record(self._resistance, "ohm / meter", "resistance_per_length"),
            "inductance_per_length": _rlgc_matrix_record(self._inductance, "henry / meter", "inductance_per_length"),
            "conductance_per_length": _rlgc_matrix_record(self._conductance, "siemens / meter", "conductance_per_length"),
            "capacitance_per_length": _rlgc_matrix_record(self._capacitance, "farad / meter", "capacitance_per_length"),
            "extraction_frequency": None if self._extraction_frequency is None else _quantity_record(registry.Quantity(self._extraction_frequency, "hertz"), "hertz"),
            "source": _thaw_rlgc_source(self._source),
        }


def _rlgc_matrix(value: object, unit: str, size: int, *, name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Quantity):
        raise TypeError(f"{name} must be a SCNSim Quantity-valued matrix")
    if value._REGISTRY is not registry:
        raise TypeError(f"{name} must use the scnsim.units registry")
    try:
        matrix = np.asarray(value.to(unit).magnitude, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"{name} must have dimensionality compatible with {unit}") from exc
    if matrix.ndim != 2 or matrix.shape != (size, size):
        raise SCNSimValidationError(f"{name} must be a {size} by {size} matrix", stage="authoring")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} entries must be finite")
    return tuple(tuple(float(entry) for entry in row) for row in matrix)


def _rlgc_quantity(matrix: tuple[tuple[float, ...], ...], unit: str) -> Quantity:
    values = np.asarray(matrix, dtype=np.float64)
    values.setflags(write=False)
    return registry.Quantity(values, unit)


def _rlgc_matrix_record(matrix: tuple[tuple[float, ...], ...], unit: str, dimensionality: str) -> dict[str, object]:
    from ._canonical import float64_hex

    size = len(matrix)
    return {
        "type": "quantity_matrix_f64",
        "shape": [size, size],
        "values_f64": [float64_hex(value) for row in matrix for value in row],
        "si_unit": unit,
        "dimensionality": dimensionality,
    }


def _validate_rlgc_matrices(
    resistance: tuple[tuple[float, ...], ...],
    inductance: tuple[tuple[float, ...], ...],
    conductance: tuple[tuple[float, ...], ...],
    capacitance: tuple[tuple[float, ...], ...],
) -> None:
    matrices = {
        "resistance_per_length": resistance,
        "inductance_per_length": inductance,
        "conductance_per_length": conductance,
        "capacitance_per_length": capacitance,
    }
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in matrices.items()}
    for name, matrix in arrays.items():
        if not np.array_equal(matrix.view(np.uint64), matrix.T.view(np.uint64)):
            raise SCNSimValidationError(f"{name} must be exactly symmetric", stage="authoring")
    for name in ("capacitance_per_length", "conductance_per_length"):
        matrix = arrays[name]
        if np.any(matrix[~np.eye(len(matrix), dtype=bool)] > 0):
            raise SCNSimValidationError(f"{name} off-diagonal entries must be nonpositive", stage="authoring")
        if np.any(matrix.sum(axis=1) < 0):
            raise SCNSimValidationError(f"{name} row sums must be nonnegative", stage="authoring")
    for name in ("capacitance_per_length", "inductance_per_length"):
        try:
            np.linalg.cholesky(arrays[name])
        except np.linalg.LinAlgError as exc:
            raise SCNSimValidationError(f"{name} must be positive definite", stage="authoring") from exc
    for name in ("conductance_per_length", "resistance_per_length"):
        if float(np.linalg.eigvalsh(arrays[name])[0]) < 0:
            raise SCNSimValidationError(f"{name} must be positive semidefinite", stage="authoring")


def _freeze_rlgc_source(value: Mapping[str, object]) -> Mapping[str, object]:
    def freeze(item: object) -> object:
        if isinstance(item, Mapping):
            return MappingProxyType({str(key): freeze(nested) for key, nested in item.items()})
        if isinstance(item, (list, tuple)):
            return tuple(freeze(nested) for nested in item)
        return item

    return MappingProxyType({str(key): freeze(item) for key, item in value.items()})


def _thaw_rlgc_source(value: Mapping[str, object]) -> dict[str, object]:
    def thaw(item: object) -> object:
        if isinstance(item, Mapping):
            return {key: thaw(nested) for key, nested in item.items()}
        if isinstance(item, tuple):
            return [thaw(nested) for nested in item]
        return item

    return {key: thaw(item) for key, item in value.items()}


class AffineMap:
    """Declarative calibrated mapping reserved for Composite expansion."""

    def __init__(
        self,
        *,
        input: ParameterRef,
        slope: object,
        intercept: object,
        support: tuple[object, object],
    ) -> None:
        if not isinstance(input, ParameterRef):
            raise TypeError("input must be a ParameterRef")
        if not isinstance(support, tuple) or len(support) != 2:
            raise TypeError("support must be a two-Quantity tuple")
        self.input = input
        self.slope = slope
        if not isinstance(intercept, Quantity) or intercept._REGISTRY is not registry:
            raise TypeError("intercept must be a Quantity from scnsim.units")
        self._output_unit = _unit_name(intercept.units)
        self.intercept = require_quantity(intercept, self._output_unit, name="intercept")
        from ._canonical import _UNITS

        slope_dimensions = (registry.Unit(self._output_unit) / registry.Unit(input.unit)).dimensionality
        self._slope_unit = next(
            (unit for unit in _UNITS if registry.Unit(unit).dimensionality == slope_dimensions),
            None,
        )
        if self._slope_unit is None:
            raise ValueError("slope dimensionality is outside SCNSim's canonical unit vocabulary")
        try:
            self.slope = require_quantity(self.slope, self._slope_unit, name="slope")
        except Exception as exc:
            raise ValueError("slope dimensionality must map input to intercept") from exc
        self._support = (
            require_quantity(support[0], input.unit, name="support[0]"),
            require_quantity(support[1], input.unit, name="support[1]"),
        )
        if self._support[0].to(input.unit).magnitude >= self._support[1].to(input.unit).magnitude:
            raise ValueError("AffineMap support must be strictly ordered")
        baseline = input.baseline.to(input.unit).magnitude
        if not self._support[0].to(input.unit).magnitude <= baseline <= self._support[1].to(input.unit).magnitude:
            raise ValueError("AffineMap input baseline must lie inside support")

    def _value_at_baseline(self, *, unit: str, name: str) -> Quantity:
        value = self.slope * self.input.baseline + self.intercept
        return require_quantity(value, unit, name=name)

    def _canonical_binding(self) -> dict[str, object]:
        return {
            "kind": "affine",
            "input": self.input._canonical_ref(),
            "slope": _quantity_record(self.slope, self._slope_unit),
            "intercept": _quantity_record(require_quantity(self.intercept, self._output_unit, name="intercept"), self._output_unit),
            "support": [_quantity_record(self._support[0], self.input.unit), _quantity_record(self._support[1], self.input.unit)],
        }


class ComponentInstance:
    """One immutable built-in or sealed Composite component snapshot."""

    __slots__ = ("_factory", "_id", "_pins", "_parameters", "_branches", "_coordinates", "_catalog_id", "_catalog_source", "_realization", "_binding_refs", "_affine_sources", "_ground_groups", "_rlgc_source")

    def __init__(self) -> None:
        unavailable("ComponentInstance construction")

    @classmethod
    def _create(
        cls,
        *,
        id: str,
        factory: str,
        pins: Sequence[str],
        parameters: Mapping[str, tuple[Quantity, str]],
        realization: Mapping[str, object],
        branches: Mapping[str, object] = (),
        coordinates: Mapping[str, object] = (),
        ground_groups: Sequence[Sequence[Mapping[str, object]]] = (),
        rlgc_source: RLGC | None = None,
        catalog_id: str,
        catalog_source: Mapping[str, object],
    ) -> ComponentInstance:
        instance = object.__new__(cls)
        instance._id = _identifier(id, field="component id")
        instance._factory = factory
        instance._catalog_id = catalog_id
        instance._catalog_source = MappingProxyType(dict(catalog_source))
        instance._pins = MappingProxyType({name: PinRef._create(instance, name) for name in pins})
        instance._parameters = MappingProxyType({name: ParameterRef._create(instance, name, baseline, unit) for name, (baseline, unit) in parameters.items()})
        instance._branches = MappingProxyType({name: InductiveBranchRef._create(instance, name) for name in branches})
        instance._coordinates = MappingProxyType({name: CoordinateRef._create(instance, name) for name in coordinates})
        binding_refs: dict[str, ParameterRef] = {}
        affine_sources: dict[str, Mapping[str, object]] = {}
        for name, binding in realization.get("bindings", {}).items():
            reference = binding.pop("_input_ref", None)
            if reference is not None:
                binding_refs[name] = reference
            source = binding.pop("_affine_source", None)
            if source is not None:
                affine_sources[name] = MappingProxyType(
                    {
                        "slope": source["slope"],
                        "intercept": source["intercept"],
                        "support": tuple(source["support"]),
                    }
                )
        instance._realization = MappingProxyType(dict(realization))
        instance._binding_refs = MappingProxyType(binding_refs)
        instance._affine_sources = MappingProxyType(affine_sources)
        instance._ground_groups = tuple(tuple(dict(endpoint) for endpoint in group) for group in ground_groups)
        instance._rlgc_source = None if rlgc_source is None else rlgc_source._source_quantities
        return instance

    @property
    def id(self) -> str:
        """Stable component identity inside its parent Plan."""

        return self._id

    @property
    def factory(self) -> str:
        """Exact catalog factory identity."""

        return self._factory

    @property
    def catalog_id(self) -> str:
        """Stable source catalog identity."""

        return self._catalog_id

    def pin(self, name: str, *, conductor: str | None = None) -> PinRef:
        """Return one declared terminal, optionally qualified by conductor."""

        if conductor is not None:
            if self._factory != "transmission_line":
                unavailable("ComponentInstance.pin(conductor=...)")
            name = f"{_identifier(name, field='pin id')}.{_identifier(conductor, field='conductor')}"
        elif self._factory == "transmission_line":
            raise TypeError("transmission_line pins require conductor=")
        try:
            return self._pins[name]
        except KeyError:
            raise KeyError(f"component {self.id!r} has no pin {name!r}") from None

    def parameter(self, name: str) -> ParameterRef:
        """Return one deliberately public primitive parameter."""

        try:
            return self._parameters[name]
        except KeyError:
            raise KeyError(f"component {self.id!r} has no parameter {name!r}") from None

    def inductive_branch(self, name: str) -> InductiveBranchRef:
        try:
            return self._branches[name]
        except KeyError:
            raise KeyError(f"component {self.id!r} has no inductive branch {name!r}") from None

    def coordinate(self, name: str) -> CoordinateRef:
        try:
            return self._coordinates[name]
        except KeyError:
            raise KeyError(f"component {self.id!r} has no coordinate {name!r}") from None

    def _canonical_snapshot(self, path: Sequence[str] | None = None) -> dict[str, object]:
        path = [self.id] if path is None else list(path)
        bindings = self._realization.get("bindings", {})
        snapshot = {
            "component_path": path,
            "catalog_id": self.catalog_id,
            "factory": self.factory,
            "pin_order": list(self._pins),
            "parameter_bindings": [
                {
                    "id": parameter.id, "binding": dict(bindings.get(parameter.id, {"kind": "constant", "value": _quantity_record(parameter.baseline, parameter.unit)})),
                }
                for parameter in self._parameters.values()
            ],
            "inductive_branches": [
                {"id": name, "positive_endpoint": {"component_path": path, "pin_id": positive}, "negative_endpoint": {"component_path": path, "pin_id": negative}, "inductance": dict(binding)}
                for name, (positive, negative, binding) in self._realization.get("branches", {}).items()
            ],
            "realization": self._snapshot_realization(path),
        }
        return snapshot

    def _snapshot_realization(self, path: Sequence[str]) -> dict[str, object]:
        realization = dict(self._realization)
        for key in ("public_pin_map", "public_coordinate_map"):
            if key in realization:
                realization[key] = [dict(item) for item in realization[key]]
        if "public_inductive_branch_map" in realization:
            realization["public_inductive_branch_map"] = [
                {"public_id": item["public_id"], "target": dict(item["target"])}
                for item in realization["public_inductive_branch_map"]
            ]
        if "public_parameter_maps" in realization:
            realization["public_parameter_maps"] = [
                {"parameter": dict(item["parameter"]), "consumers": [{"target": dict(consumer["target"]), "binding": dict(consumer["binding"])} for consumer in item["consumers"]]} for item in realization["public_parameter_maps"]
            ]
        if "couplings" in realization:
            realization["couplings"] = [
                {**coupling, "branch_a": dict(coupling["branch_a"]), "branch_b": dict(coupling["branch_b"])}
                for coupling in realization["couplings"]
            ]
        realization.pop("bindings", None)
        realization.pop("branches", None)
        if realization.get("kind") != "composite":
            return realization
        children = realization.pop("children")
        node_maps = realization.pop("nodes")
        grounded = realization.pop("grounded")
        realization["children"] = [child._canonical_snapshot([*path, child.id]) for child in children]
        node_ids: dict[str, str] = {}
        for node in node_maps:
            endpoints = [{"component_path": [*path, pin.component_id], "pin_id": pin.name} for pin in node.endpoints]
            node_ids[node.id] = __import__("scnsim._canonical", fromlist=["internal_node_id"]).internal_node_id(endpoints) if node.id.startswith("internal-") else node.id
        realization["private_nodes"] = [
            {"id": node_ids[node.id], "endpoints": [{"component_path": [*path, pin.component_id], "pin_id": pin.name} for pin in node.endpoints]}
            for node in node_maps
        ]
        realization["grounded_endpoints"] = [{"component_path": [*path, pin.component_id], "pin_id": pin.name} for pin in grounded]
        for node_map in realization.get("public_coordinate_map", []):
            node_map["public_id"] = ".".join([*path, node_map["public_id"]])
            node_map["private_node_id"] = node_ids[node_map["private_node_id"]]
        for node_map in realization.get("public_pin_map", []):
            node_map["private_node_id"] = node_ids[node_map["private_node_id"]]
        for parameter_map in realization.get("public_parameter_maps", []):
            for consumer in parameter_map["consumers"]:
                consumer["target"]["component_path"] = [*path, *consumer["target"]["component_path"]]
        for coupling in realization.get("couplings", []):
            for branch in (coupling["branch_a"], coupling["branch_b"]):
                branch["component_path"] = [*path, *branch["component_path"]]
        for branch_map in realization.get("public_inductive_branch_map", []):
            branch_map["target"]["component_path"] = [*path, *branch_map["target"]["component_path"]]
        return realization


class CompositePlan:
    """Private one-factory graph sealed into one immutable ComponentInstance."""

    __slots__ = ("_id", "_library", "_factory", "_sealed", "_built", "_components", "_nodes", "_grounded", "_ground_groups", "_pin_nodes", "_parameters", "_parameter_bindings", "_exposed_pins", "_coordinates", "_branches", "_couplings")

    def __init__(self, *, id: str, library: Library) -> None:
        context = _factory_context.get()
        if not isinstance(library, Library) or context is None or context[0] is not library:
            raise SCNSimValidationError("CompositePlan must be created inside its owning Library factory", stage="authoring")
        self._id, self._library, self._factory = _identifier(id, field="component id"), library, context[1]
        self._sealed = False
        self._built: ComponentInstance | None = None
        self._components: list[ComponentInstance] = []
        self._nodes: list[_PlanNode] = []
        self._grounded: set[PinRef] = set()
        self._ground_groups: list[tuple[PinRef, ...]] = []
        self._pin_nodes: dict[PinRef, _PlanNode | Literal["ground"]] = {}
        self._parameters: dict[str, ParameterRef] = {}
        self._parameter_bindings: dict[str, dict[str, object]] = {}
        self._exposed_pins: dict[str, _PlanNode] = {}
        self._coordinates: dict[str, _PlanNode] = {}
        self._branches: dict[str, InductiveBranchRef] = {}
        self._couplings: list[dict[str, object]] = []

    def _assert_mutable(self) -> None:
        if self._sealed:
            raise PlanSealedError("CompositePlan is permanently sealed", stage="plan_mutation")

    def parameter(self, *, id: str, baseline: object, spec: ParameterSpec) -> ParameterRef:
        self._assert_mutable()
        name = _identifier(id, field="parameter id")
        if not isinstance(spec, ParameterSpec):
            raise TypeError("spec must be a ParameterSpec")
        if name in self._parameters:
            raise SCNSimValidationError("public parameter IDs must be unique", stage="authoring")
        value, binding = _binding(baseline, spec.si_unit, name=name)
        reference = ParameterRef._create(self, name, value, spec.si_unit)
        self._parameters[name] = reference
        self._parameter_bindings[name] = binding
        return reference

    @property
    def id(self) -> str:
        return self._id

    def add(self, component: ComponentInstance) -> ComponentInstance:
        self._assert_mutable()
        if not isinstance(component, ComponentInstance):
            raise TypeError("CompositePlan.add requires a ComponentInstance")
        if any(existing.id == component.id for existing in self._components):
            raise SCNSimValidationError("component IDs must be unique", stage="authoring")
        self._components.append(component)
        return component

    def _validate_pin(self, pin: PinRef) -> None:
        if not isinstance(pin, PinRef) or pin._component not in self._components:
            raise SCNSimValidationError("pin does not belong to this CompositePlan", stage="authoring")
        if pin in self._pin_nodes:
            raise SCNSimValidationError("each pin may belong to one node or ground", stage="authoring")

    def net(self, *pins: PinRef, id: str | None = None) -> ElectricNodeRef:
        self._assert_mutable()
        if not pins or len(set(pins)) != len(pins):
            raise SCNSimValidationError("net requires one or more distinct pins", stage="authoring")
        for pin in pins:
            self._validate_pin(pin)
        node_id = _identifier(id, field="node id") if id is not None else __import__("scnsim._canonical", fromlist=["internal_node_id"]).internal_node_id([pin._endpoint() for pin in pins])
        if any(node.id == node_id for node in self._nodes):
            raise SCNSimValidationError("node IDs must be unique", stage="authoring")
        node = _PlanNode(node_id, "public" if id is not None else "internal", tuple(pins))
        self._nodes.append(node)
        self._pin_nodes.update({pin: node for pin in pins})
        return ElectricNodeRef._create(self, node)

    def ground(self, *pins: PinRef) -> None:
        self._assert_mutable()
        if not pins or len(set(pins)) != len(pins):
            raise SCNSimValidationError("ground requires one or more distinct pins", stage="authoring")
        for pin in pins:
            self._validate_pin(pin)
        self._grounded.update(pins)
        self._ground_groups.append(tuple(pins))
        self._pin_nodes.update({pin: "ground" for pin in pins})

    def _node(self, at: ElectricNodeRef) -> _PlanNode:
        if not isinstance(at, ElectricNodeRef) or at._plan is not self:
            raise SCNSimValidationError("exposure must target an ElectricNodeRef from this CompositePlan", stage="authoring")
        return at._node

    def expose_pin(self, *, id: str, at: ElectricNodeRef) -> PinRef:
        self._assert_mutable()
        name, node = _identifier(id, field="pin id"), self._node(at)
        if name in self._exposed_pins:
            raise SCNSimValidationError("public pin IDs must be unique", stage="authoring")
        if any(existing is node for existing in self._exposed_pins.values()):
            raise SCNSimValidationError("one private node may expose only one public pin", stage="authoring")
        self._exposed_pins[name] = node
        return PinRef._create(self, name)

    def expose_coordinate(self, *, id: str, at: ElectricNodeRef) -> CoordinateRef:
        self._assert_mutable()
        name, node = _identifier(id, field="coordinate id"), self._node(at)
        if name in self._coordinates:
            raise SCNSimValidationError("public coordinate IDs must be unique", stage="authoring")
        if any(existing is node for existing in self._coordinates.values()):
            raise SCNSimValidationError("one private node may expose only one public coordinate", stage="authoring")
        self._coordinates[name] = node
        return CoordinateRef._create(self, name)

    def expose_inductive_branch(self, *, id: str, branch: InductiveBranchRef) -> InductiveBranchRef:
        self._assert_mutable()
        name = _identifier(id, field="inductive branch id")
        if not isinstance(branch, InductiveBranchRef) or branch._component not in self._components:
            raise SCNSimValidationError("inductive branch exposure must target a child branch", stage="authoring")
        if name in self._branches:
            raise SCNSimValidationError("public inductive branch IDs must be unique", stage="authoring")
        self._branches[name] = branch
        return InductiveBranchRef._create(self, name)

    def couple_inductive(self, *, id: str, inductor_a: InductiveBranchRef, inductor_b: InductiveBranchRef, coupling_coefficient: Quantity) -> None:
        self._assert_mutable()
        coupling = _coupling(id, inductor_a, inductor_b, coupling_coefficient, self._components)
        self._couplings.append(coupling)

    def build(self) -> ComponentInstance:
        if self._built is not None:
            return self._built
        self._assert_mutable()
        if not self._components:
            raise SCNSimValidationError("CompositePlan requires at least one child component", stage="plan_seal")
        missing = [pin.component_id + "." + pin.name for component in self._components for pin in component._pins.values() if pin not in self._pin_nodes]
        if missing:
            raise SCNSimValidationError("every composite child pin must be netted or grounded", stage="plan_seal", evidence={"missing_pins": missing})
        if any(len(node.endpoints) == 1 and node not in self._exposed_pins.values() and node not in self._coordinates.values() for node in self._nodes):
            raise SCNSimValidationError("an unexposed composite node must join at least two pins", stage="plan_seal")
        _validate_parameter_bindings(self._parameters, self._components)
        _validate_coupling_graph(self._couplings)
        source = _catalog_source(self._library, self._factory)
        component = ComponentInstance._create(
            id=self.id, factory=self._factory, pins=self._exposed_pins,
            parameters={name: (parameter.baseline, parameter.unit) for name, parameter in self._parameters.items()},
            branches={name: None for name in self._branches}, coordinates=self._coordinates, catalog_id=source["catalog_id"], catalog_source=source,
            ground_groups=tuple(tuple(pin._endpoint() for pin in group) for group in self._ground_groups),
            realization={"kind": "composite", "bindings": self._parameter_bindings, "public_parameters": [{"id": name, "spec": ParameterSpec(unit=registry.Unit(parameter.unit))._canonical_record(), "baseline": _quantity_record(parameter.baseline, parameter.unit)} for name, parameter in self._parameters.items()], "children": tuple(self._components), "nodes": tuple(self._nodes), "grounded": tuple(self._grounded), "couplings": tuple(self._couplings), "public_pin_map": [{"public_id": name, "private_node_id": node.id} for name, node in self._exposed_pins.items()], "public_coordinate_map": [{"public_id": name, "private_node_id": node.id} for name, node in self._coordinates.items()], "public_parameter_maps": _parameter_maps(self._parameters, self._components), "public_inductive_branch_map": [{"public_id": name, "target": target._canonical_ref()} for name, target in self._branches.items()]},
        )
        for parameter in self._parameters.values():
            parameter._component = component
        self._sealed = True
        self._built = component
        return component


def _parameter_maps(parameters: Mapping[str, ParameterRef], children: Sequence[ComponentInstance]) -> list[dict[str, object]]:
    maps: list[dict[str, object]] = []
    for parameter in parameters.values():
        consumers: list[dict[str, object]] = []
        for child in children:
            for target, binding in child._realization.get("bindings", {}).items():
                if binding.get("kind") in {"identity", "affine"} and binding["input"] == parameter._canonical_ref():
                    consumers.append({"target": {"component_path": [child.id], "parameter_id": target}, "binding": dict(binding)})
        if not consumers:
            raise SCNSimValidationError("every Composite parameter must bind one child slot", stage="plan_seal", evidence={"parameter": parameter.id})
        maps.append({"parameter": parameter._canonical_ref(), "consumers": consumers})
    return maps


def _validate_parameter_bindings(
    parameters: Mapping[str, ParameterRef], children: Sequence[ComponentInstance]
) -> None:
    """Close each dynamic child slot over this exact Composite's public refs."""

    declared = tuple(parameters.values())
    for child in children:
        for target, binding in child._realization.get("bindings", {}).items():
            if binding.get("kind") not in {"identity", "affine"}:
                continue
            reference = child._binding_refs.get(target)
            if reference not in declared or binding["input"] != reference._canonical_ref():
                raise SCNSimValidationError(
                    "a Composite child parameter binding must use one declared public parameter",
                    stage="plan_seal",
                    evidence={"child": child.id, "parameter": target},
                )


def _catalog_source(library: Library, factory: str) -> dict[str, object]:
    """Use the canonical provenance authority for every factory-produced snapshot."""

    return __import__("scnsim._canonical", fromlist=["catalog_source_record"]).catalog_source_record(library, factory=factory)


def _coupling(id: str, branch_a: InductiveBranchRef, branch_b: InductiveBranchRef, coefficient: Quantity, components: Sequence[ComponentInstance]) -> dict[str, object]:
    name = _identifier(id, field="coupling id")
    if not isinstance(branch_a, InductiveBranchRef) or not isinstance(branch_b, InductiveBranchRef):
        raise TypeError("inductive coupling requires InductiveBranchRef handles")
    if branch_a is branch_b or branch_a._component not in components or branch_b._component not in components:
        raise SCNSimValidationError("inductive coupling branches must be distinct members of this authoring graph", stage="authoring")
    value = require_quantity(coefficient, "dimensionless", name="coupling_coefficient")
    if not -1 < value.to("dimensionless").magnitude < 1:
        raise ValueError("coupling_coefficient must satisfy abs(k) < 1")
    mutual = value * (_branch_baseline(branch_a) * _branch_baseline(branch_b)) ** 0.5
    return {"id": name, "branch_a": branch_a._canonical_ref(), "branch_b": branch_b._canonical_ref(), "coupling_coefficient": _quantity_record(value, "dimensionless"), "derived_mutual_inductance": _quantity_record(require_quantity(mutual, "henry", name="derived_mutual_inductance"), "henry")}


def _validate_coupling_graph(couplings: Sequence[Mapping[str, object]]) -> None:
    """Apply the accepted strict SPD Gate to one complete coupling graph."""

    if not couplings:
        return
    ids = [coupling["id"] for coupling in couplings]
    if len(set(ids)) != len(ids):
        raise SCNSimValidationError("inductive coupling IDs must be unique", stage="authoring")
    keys: list[tuple[tuple[str, ...], str]] = []
    pairs: set[frozenset[tuple[tuple[str, ...], str]]] = set()
    for coupling in couplings:
        branch_keys = []
        for name in ("branch_a", "branch_b"):
            branch = coupling[name]
            branch_keys.append((tuple(branch["component_path"]), branch["branch_id"]))
        pair = frozenset(branch_keys)
        if pair in pairs:
            raise SCNSimValidationError("one inductive branch pair may have at most one coupling", stage="authoring")
        pairs.add(pair)
        keys.extend(branch_keys)
    ordered = sorted(set(keys))
    index = {key: position for position, key in enumerate(ordered)}
    import numpy as np

    correlation = np.eye(len(ordered), dtype=np.float64)
    for coupling in couplings:
        left = (tuple(coupling["branch_a"]["component_path"]), coupling["branch_a"]["branch_id"])
        right = (tuple(coupling["branch_b"]["component_path"]), coupling["branch_b"]["branch_id"])
        bits = coupling["coupling_coefficient"]["si_value_f64"]
        coefficient = struct.unpack(">d", bytes.fromhex(bits))[0]
        correlation[index[left], index[right]] = coefficient
        correlation[index[right], index[left]] = coefficient
    try:
        np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as exc:
        raise SCNSimValidationError(
            "complete reciprocal inductance matrix must be positive definite",
            stage="authoring",
        ) from exc


def _branch_baseline(reference: InductiveBranchRef) -> Quantity:
    component = reference._component
    if reference.id == "self":
        return component.parameter("inductance").baseline
    if reference.id == "loop" and "loop_inductance" in component._parameters:
        return component.parameter("loop_inductance").baseline
    for branch_map in component._realization.get("public_inductive_branch_map", ()):
        if branch_map["public_id"] == reference.id:
            target = branch_map["target"]
            target_component = component
            for segment in target["component_path"]:
                target_component = next(child for child in target_component._realization["children"] if child.id == segment)
            return _branch_baseline(target_component.inductive_branch(target["branch_id"]))
    raise SCNSimValidationError("inductive branch has no baseline inductance", stage="authoring")


def _builtin_source() -> dict[str, object]:
    context = _factory_context.get()
    return _catalog_source(components, context[1] if context else "primitive")


def _recursive_components(components: Sequence[ComponentInstance]) -> Sequence[ComponentInstance]:
    collected: list[ComponentInstance] = []
    def visit(component: ComponentInstance) -> None:
        collected.append(component)
        for child in component._realization.get("children", ()):
            visit(child)
    for component in components:
        visit(component)
    return collected


class _LibraryMeta(type):
    """Keep every catalog subclass free of mutable instance storage."""

    def __new__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, object], **kwargs: object
    ) -> _LibraryMeta:
        if bases and any(not isinstance(base, _LibraryMeta) for base in bases):
            raise TypeError("Library subclasses cannot use non-Library bases")
        if namespace.get("__slots__", ()) not in ((), []):
            raise TypeError("Library subclasses cannot declare instance state")
        for method_name, method in tuple(namespace.items()):
            if method_name.startswith("_") or not inspect.isfunction(method):
                continue
            @wraps(method)
            def factory_wrapper(self: Library, *args: object, __method: Callable[..., object] = method, __name: str = method_name, **kw: object) -> object:
                token = _factory_context.set((self, __name))
                try:
                    component = __method(self, *args, **kw)
                finally:
                    _factory_context.reset(token)
                expected_source = _catalog_source(type(self), __name)
                if not isinstance(component, ComponentInstance):
                    raise SCNSimValidationError(
                        "a public custom Library factory must return a ComponentInstance",
                        stage="authoring",
                    )
                if (
                    component._factory != __name
                    or component.catalog_id != expected_source["catalog_id"]
                    or dict(component._catalog_source) != expected_source
                ):
                    raise SCNSimValidationError(
                        "a custom Library factory must return a component owned by its invoking catalog",
                        stage="authoring",
                    )
                return component
            namespace[method_name] = factory_wrapper
        namespace["__slots__"] = ()
        return super().__new__(cls, name, bases, namespace, **kwargs)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Library catalog types are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Library catalog types are immutable")


class Library(metaclass=_LibraryMeta):
    """Base class for one immutable, project-owned component catalog."""

    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Library catalogs are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Library catalogs are immutable")


class _BuiltinComponents(Library):
    """Implementation behind the public ``scnsim.components`` catalog."""

    @staticmethod
    def _primitive(
        *, id: str, factory: Literal["resistor", "capacitor", "inductor"], value: Quantity, unit: str
    ) -> ComponentInstance:
        baseline, binding = _binding(value, unit, name=factory, positive=True)
        parameter_id = {"resistor": "resistance", "capacitor": "capacitance", "inductor": "inductance"}[factory]
        return ComponentInstance._create(
            id=id,
            factory=factory,
            pins=("terminal_1", "terminal_2"),
            parameters={parameter_id: (baseline, unit)},
            realization={"kind": factory, parameter_id: binding, "bindings": {parameter_id: binding}, "branches": {"self": ("terminal_1", "terminal_2", binding)} if factory == "inductor" else {}},
            branches={"self": ("terminal_1", "terminal_2", binding)} if factory == "inductor" else {},
            catalog_source=_builtin_source(),
            catalog_id="scnsim.components",
        )

    @staticmethod
    def _leaf(
        *, id: str, factory: str, values: Mapping[str, object], pins: Sequence[str], branches: Mapping[str, tuple[str, str, str]] = {},
    ) -> ComponentInstance:
        units = {name: ("farad" if "capacitance" in name else "ohm" if name == "resistance" else "henry") for name in values}
        bound = {name: _binding(value, units[name], name=name, positive=name != "junction_capacitance") for name, value in values.items()}
        for name, (baseline, _) in bound.items():
            if name == "junction_capacitance" and baseline.to(units[name]).magnitude < 0:
                raise ValueError("junction_capacitance must be nonnegative")
        bindings = {name: binding for name, (_, binding) in bound.items()}
        branch_records = {name: (positive, negative, bindings[parameter]) for name, (positive, negative, parameter) in branches.items()}
        return ComponentInstance._create(
            id=id, factory=factory, pins=pins, parameters={name: (baseline, units[name]) for name, (baseline, _) in bound.items()},
            realization={"kind": factory, **bindings, "bindings": bindings, "branches": branch_records}, branches=branch_records,
            catalog_id="scnsim.components", catalog_source=_builtin_source(),
        )

    @classmethod
    def _josephson_junction(cls, *, id: str, josephson_inductance: object, junction_capacitance: object) -> ComponentInstance:
        return cls._leaf(id=id, factory="josephson_junction", values={"josephson_inductance": josephson_inductance, "junction_capacitance": junction_capacitance}, pins=("terminal_1", "terminal_2"))

    def resistor(self, *, id: str, resistance: Quantity | ParameterRef | AffineMap) -> ComponentInstance:
        """Declare a resistor oriented from ``terminal_1`` to ``terminal_2``."""

        return self._primitive(id=id, factory="resistor", value=resistance, unit="ohm")

    def capacitor(self, *, id: str, capacitance: Quantity | ParameterRef | AffineMap) -> ComponentInstance:
        """Declare a capacitor oriented from ``terminal_1`` to ``terminal_2``."""

        return self._primitive(id=id, factory="capacitor", value=capacitance, unit="farad")

    def inductor(self, *, id: str, inductance: Quantity | ParameterRef | AffineMap) -> ComponentInstance:
        """Declare an inductor oriented from ``terminal_1`` to ``terminal_2``."""

        return self._primitive(id=id, factory="inductor", value=inductance, unit="henry")

    def josephson_junction(
        self,
        *,
        id: str,
        josephson_inductance: object,
        junction_capacitance: object = 0 * registry.farad,
    ) -> ComponentInstance:
        return self._josephson_junction(id=id, josephson_inductance=josephson_inductance, junction_capacitance=junction_capacitance)

    def transmission_line(
        self,
        *,
        id: str,
        length: Quantity | ParameterRef | AffineMap,
        rlgc: RLGC,
        n_sections: int,
    ) -> ComponentInstance:
        if not isinstance(rlgc, RLGC):
            raise TypeError("rlgc must be an RLGC value")
        if isinstance(n_sections, bool) or not isinstance(n_sections, int):
            raise TypeError("n_sections must be a positive Python integer")
        if n_sections < 1:
            raise ValueError("n_sections must be a positive Python integer")
        baseline, binding = _binding(length, "meter", name="length", positive=True)
        pins = tuple(
            f"{end}.{conductor}"
            for end in ("head", "tail")
            for conductor in rlgc.conductors
        )
        return ComponentInstance._create(
            id=id,
            factory="transmission_line",
            pins=pins,
            parameters={"length": (baseline, "meter")},
            realization={
                "kind": "transmission_line",
                "length": binding,
                "rlgc": rlgc._canonical_record(),
                "n_sections": n_sections,
                "pin_conductors": list(rlgc.conductors),
                "bindings": {"length": binding},
                "branches": {},
            },
            rlgc_source=rlgc,
            catalog_source=_builtin_source(),
            catalog_id="scnsim.components",
        )

    def interdigitated_capacitor(
        self,
        *,
        id: str,
        terminal_1_to_reference_capacitance: object,
        terminal_2_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
    ) -> ComponentInstance:
        composite = CompositePlan(id=id, library=self)
        c_1g = composite.parameter(id="terminal_1_to_reference_capacitance", baseline=terminal_1_to_reference_capacitance, spec=ParameterSpec(unit=registry.farad))
        c_2g = composite.parameter(id="terminal_2_to_reference_capacitance", baseline=terminal_2_to_reference_capacitance, spec=ParameterSpec(unit=registry.farad))
        c_12 = composite.parameter(id="terminal_mutual_capacitance", baseline=terminal_mutual_capacitance, spec=ParameterSpec(unit=registry.farad))
        terminal_1_ground = composite.add(self.capacitor(id="terminal_1_to_reference", capacitance=c_1g))
        terminal_2_ground = composite.add(self.capacitor(id="terminal_2_to_reference", capacitance=c_2g))
        mutual = composite.add(self.capacitor(id="terminal_mutual", capacitance=c_12))
        terminal_1 = composite.net(terminal_1_ground.pin("terminal_1"), mutual.pin("terminal_1"), id="terminal_1")
        terminal_2 = composite.net(terminal_2_ground.pin("terminal_1"), mutual.pin("terminal_2"), id="terminal_2")
        composite.ground(terminal_1_ground.pin("terminal_2"), terminal_2_ground.pin("terminal_2"))
        composite.expose_pin(id="terminal_1", at=terminal_1)
        composite.expose_pin(id="terminal_2", at=terminal_2)
        return composite.build()

    def symmetric_squid(
        self,
        *,
        id: str,
        josephson_inductance: object,
        junction_capacitance: object = 0 * registry.farad,
        loop_inductance: object,
    ) -> ComponentInstance:
        composite = CompositePlan(id=id, library=self)
        inductance = composite.parameter(id="josephson_inductance", baseline=josephson_inductance, spec=ParameterSpec(unit=registry.henry))
        capacitance = composite.parameter(id="junction_capacitance", baseline=junction_capacitance, spec=ParameterSpec(unit=registry.farad))
        loop = composite.parameter(id="loop_inductance", baseline=loop_inductance, spec=ParameterSpec(unit=registry.henry))
        junction_1 = composite.add(self.josephson_junction(id="junction_1", josephson_inductance=inductance, junction_capacitance=capacitance))
        loop_inductor = composite.add(self.inductor(id="loop", inductance=loop))
        junction_2 = composite.add(self.josephson_junction(id="junction_2", josephson_inductance=inductance, junction_capacitance=capacitance))
        terminal_1 = composite.net(junction_1.pin("terminal_1"), loop_inductor.pin("terminal_1"), id="terminal_1")
        composite.net(loop_inductor.pin("terminal_2"), junction_2.pin("terminal_1"), id="loop_node")
        terminal_2 = composite.net(junction_1.pin("terminal_2"), junction_2.pin("terminal_2"), id="terminal_2")
        composite.expose_pin(id="terminal_1", at=terminal_1)
        composite.expose_pin(id="terminal_2", at=terminal_2)
        composite.expose_inductive_branch(id="loop", branch=loop_inductor.inductive_branch("self"))
        return composite.build()

    def grounded_parallel_linear_lc_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        inductance: object,
    ) -> ComponentInstance:
        return self._resonator(id=id, grounded=True, branch="linear", values={"capacitance": capacitance, "inductance": inductance})

    def floating_parallel_linear_lc_resonator(
        self,
        *,
        id: str,
        terminal_1_to_reference_capacitance: object,
        terminal_2_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
        inductance: object,
    ) -> ComponentInstance:
        return self._resonator(id=id, grounded=False, branch="linear", values={"terminal_1_to_reference_capacitance": terminal_1_to_reference_capacitance, "terminal_2_to_reference_capacitance": terminal_2_to_reference_capacitance, "terminal_mutual_capacitance": terminal_mutual_capacitance, "inductance": inductance})

    def grounded_parallel_single_junction_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
    ) -> ComponentInstance:
        return self._resonator(id=id, grounded=True, branch="junction", values={"capacitance": capacitance, "josephson_inductance": josephson_inductance, "junction_capacitance": junction_capacitance})

    def floating_parallel_single_junction_resonator(
        self,
        *,
        id: str,
        terminal_1_to_reference_capacitance: object,
        terminal_2_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
    ) -> ComponentInstance:
        return self._resonator(id=id, grounded=False, branch="junction", values={"terminal_1_to_reference_capacitance": terminal_1_to_reference_capacitance, "terminal_2_to_reference_capacitance": terminal_2_to_reference_capacitance, "terminal_mutual_capacitance": terminal_mutual_capacitance, "josephson_inductance": josephson_inductance, "junction_capacitance": junction_capacitance})

    def grounded_parallel_symmetric_squid_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
        loop_inductance: object,
    ) -> ComponentInstance:
        return self._resonator(id=id, grounded=True, branch="squid", values={"capacitance": capacitance, "josephson_inductance": josephson_inductance, "junction_capacitance": junction_capacitance, "loop_inductance": loop_inductance})

    def floating_parallel_symmetric_squid_resonator(
        self,
        *,
        id: str,
        terminal_1_to_reference_capacitance: object,
        terminal_2_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
        loop_inductance: object,
    ) -> ComponentInstance:
        return self._resonator(id=id, grounded=False, branch="squid", values={"terminal_1_to_reference_capacitance": terminal_1_to_reference_capacitance, "terminal_2_to_reference_capacitance": terminal_2_to_reference_capacitance, "terminal_mutual_capacitance": terminal_mutual_capacitance, "josephson_inductance": josephson_inductance, "junction_capacitance": junction_capacitance, "loop_inductance": loop_inductance})

    def _resonator(self, *, id: str, grounded: bool, branch: Literal["linear", "junction", "squid"], values: Mapping[str, object]) -> ComponentInstance:
        composite = CompositePlan(id=id, library=self)
        refs = {name: composite.parameter(id=name, baseline=value, spec=ParameterSpec(unit=registry.Unit("farad" if "capacitance" in name else "henry"))) for name, value in values.items()}
        if grounded:
            capacitor = composite.add(self.capacitor(id="capacitor", capacitance=refs["capacitance"]))
            terminals = [capacitor.pin("terminal_1")]
            returns = [capacitor.pin("terminal_2")]
        else:
            capacitor = composite.add(self.interdigitated_capacitor(id="capacitor", terminal_1_to_reference_capacitance=refs["terminal_1_to_reference_capacitance"], terminal_2_to_reference_capacitance=refs["terminal_2_to_reference_capacitance"], terminal_mutual_capacitance=refs["terminal_mutual_capacitance"]))
            terminals = [capacitor.pin("terminal_1"), capacitor.pin("terminal_2")]
            returns = []
        if branch == "linear":
            element = composite.add(self.inductor(id="inductor", inductance=refs["inductance"]))
        elif branch == "junction":
            element = composite.add(self.josephson_junction(id="junction", josephson_inductance=refs["josephson_inductance"], junction_capacitance=refs["junction_capacitance"]))
        else:
            element = composite.add(self.symmetric_squid(id="squid", josephson_inductance=refs["josephson_inductance"], junction_capacitance=refs["junction_capacitance"], loop_inductance=refs["loop_inductance"]))
            composite.expose_inductive_branch(id="loop", branch=element.inductive_branch("loop"))
        if grounded:
            terminal = composite.net(terminals[0], element.pin("terminal_1"), id="terminal")
            composite.ground(*returns, element.pin("terminal_2"))
            composite.expose_pin(id="terminal", at=terminal)
        else:
            terminal_1 = composite.net(terminals[0], element.pin("terminal_1"), id="terminal_1")
            terminal_2 = composite.net(terminals[-1], element.pin("terminal_2"), id="terminal_2")
            composite.expose_pin(id="terminal_1", at=terminal_1)
            composite.expose_pin(id="terminal_2", at=terminal_2)
        return composite.build()


@dataclass(slots=True)
class _PlanNode:
    id: str
    visibility: Literal["public", "internal", "port_promoted"]
    endpoints: tuple[PinRef, ...]


def _compiled_line_audit(compiled: Mapping[str, object], *, show_values: bool) -> tuple[str, ...]:
    """Render the real recursive compiler preflight as a deterministic audit."""

    from ._canonical import canonical_json_bytes

    original = compiled["ref_lineage"]["original"]
    lines = [
        f"plan_sha256 {compiled['plan_sha256']}",
        f"compiler_sha256 {original['compiled_graph_sha256']}",
        f"expanded_graph_sha256 {compiled['expanded_graph_sha256']}",
    ]
    audit_rows = tuple(
        row for row in compiled["expanded_branch_rows"]
        if row.get("kind") == "transmission_line_audit"
    )
    line_paths = {tuple(row["component_path"]) for row in audit_rows}
    line_nodes = {
        station["compiled_node_id"]
        for row in audit_rows
        for station in row["stations"]
    }
    lines.extend(
        f"node {index}: {identifier}"
        for index, identifier in enumerate(compiled["node_order"])
        if identifier not in line_nodes
    )
    if show_values:
        lines.extend(
            "binding " + canonical_json_bytes(binding).decode("utf-8")
            for binding in compiled["resolved_bindings"]
        )
    for row in compiled["expanded_branch_rows"]:
        if tuple(row.get("component_path", ())) in line_paths:
            continue
        record = dict(row)
        if not show_values:
            for field in (
                "value", "coupling_coefficient", "derived_mutual_inductance",
                "length", "dx",
            ):
                record.pop(field, None)
            if record.get("kind") == "transmission_line_audit":
                record["stations"] = [
                    {
                        key: value for key, value in station.items()
                        if key not in {"compiled_capacitance_total", "compiled_conductance_total"}
                    }
                    for station in record.get("stations", ())
                ]
                source = record.get("rlgc_source")
                if isinstance(source, Mapping):
                    record["rlgc_source"] = {
                        key: value for key, value in source.items()
                        if key != "header_records"
                    }
        lines.append("branch " + canonical_json_bytes(record).decode("utf-8"))
    return tuple(lines)


def _draw_compiled_ladders(
    drawing: object,
    compiled: Mapping[str, object],
    *,
    color: str,
    background: str,
    show_values: bool,
) -> float:
    """Draw each recursively compiled transmission line from preflight rows."""

    import schemdraw.elements as elm
    from ._canonical import float64_from_hex

    rows = tuple(compiled["expanded_branch_rows"])

    def matrix_text(record: Mapping[str, object] | None) -> str:
        if not show_values or not isinstance(record, Mapping):
            return ""
        shape = record.get("shape")
        values = record.get("values_f64")
        if not isinstance(shape, list) or len(shape) != 2 or not isinstance(values, list):
            return ""
        width = shape[1]
        matrix = [values[index:index + width] for index in range(0, len(values), width)]
        rendered = "[" + "; ".join(
            ", ".join(f"{float64_from_hex(value):g}" for value in row)
            for row in matrix
        ) + "]"
        return f"{rendered} {record.get('si_unit', '')}".rstrip()

    def section_matrix(path: tuple[str, ...], kind: str, section: int) -> str:
        selected = [
            row for row in rows
            if tuple(row.get("component_path", ())) == path
            and row.get("kind") == kind and row.get("section") == section
        ]
        if not show_values or not selected:
            return ""
        conductors = audit["conductors"]
        by_pair = {(row["row_conductor"], row["column_conductor"]): row["value"] for row in selected}
        values = [
            [float64_from_hex(by_pair[(left, right)]["si_value_f64"]) for right in conductors]
            for left in conductors
        ]
        unit = selected[0]["value"]["si_unit"]
        return "[" + "; ".join(", ".join(f"{value:g}" for value in row) for row in values) + f"] {unit}"

    def half_shunt(record: Mapping[str, object] | None) -> str:
        if record is None:
            return "none"
        return f"s{record['section']}.{record['end']}"

    cursor_y = 0.0
    for audit in (row for row in rows if row.get("kind") == "transmission_line_audit"):
        path = tuple(audit["component_path"])
        conductors = tuple(audit["conductors"])
        sections = int(audit["n_sections"])
        title = (
            f"line {'.'.join(path)} | conductors={','.join(conductors)} | "
            f"reference={audit['reference_conductor']} | sections={sections} | +z head→tail"
        )
        if show_values:
            length = audit["length"]
            spacing = audit["dx"]
            title += (
                f" | length={float64_from_hex(length['si_value_f64']):g} {length['si_unit']}"
                f" | dx={float64_from_hex(spacing['si_value_f64']):g} {spacing['si_unit']}"
            )
        drawing.add(elm.Line(color=color).endpoints((0.0, cursor_y), (0.2, cursor_y)).label(title, loc="right", color=color))
        cursor_y -= 1.2
        stations = {(item["station"], item["conductor"]): item for item in audit["stations"]}
        step = 4.0
        panel_size = 4
        panel_count = (sections + panel_size - 1) // panel_size
        for panel_index, first_section in enumerate(range(1, sections + 1, panel_size), start=1):
            last_section = min(first_section + panel_size - 1, sections)
            first_station = first_section - 1
            top_y = cursor_y - 1.2
            lane_y = {conductor: top_y - 1.25 * index for index, conductor in enumerate(conductors)}
            bottom_y = min(lane_y.values())
            ground_y = bottom_y - 2.4
            drawing.add(
                elm.Line(color=color).endpoints((0.0, cursor_y), (0.2, cursor_y))
                .label(
                    f"panel {panel_index}/{panel_count} | sections {first_section}–{last_section}",
                    loc="right", color=color,
                )
            )
            if len(conductors) == 1:
                conductor = conductors[0]
                y = lane_y[conductor]
                for section in range(first_section, last_section + 1):
                    left = step * (section - first_section)
                    middle, right = left + step / 2, left + step
                    r_label = "R" + (("\n" + section_matrix(path, "series_resistance", section)) if show_values else "")
                    l_label = "L" + (("\n" + section_matrix(path, "series_inductance", section)) if show_values else "")
                    drawing.add(elm.Resistor(color=color).endpoints((left, y), (middle, y)).label(r_label, color=color))
                    drawing.add(elm.Inductor(color=color).endpoints((middle, y), (right, y)).label(l_label, color=color))
                for station in range(first_station, last_section + 1):
                    x = step * (station - first_station)
                    record = stations[(station, conductor)]
                    provenance = (
                        f"{station}: {record['compiled_node_id']} [{record['attachment']}; "
                        f"L={half_shunt(record['left_half_shunt'])}; R={half_shunt(record['right_half_shunt'])}]"
                    )
                    drawing.add(elm.Dot(open=True, color=color, fill=background).at((x, y)).label(provenance, loc="top", color=color))
                    drawing.add(elm.Line(color=color).endpoints((x - 0.22, y), (x + 0.22, y)))
                    c_text = matrix_text(record.get("compiled_capacitance_total"))
                    g_text = matrix_text(record.get("compiled_conductance_total"))
                    drawing.add(elm.Capacitor(color=color).endpoints((x - 0.22, y), (x - 0.22, ground_y)).label("C" + (("\n" + c_text) if c_text else ""), loc="left", color=color))
                    drawing.add(elm.Resistor(color=color).endpoints((x + 0.22, y), (x + 0.22, ground_y)).label("G" + (("\n" + g_text) if g_text else ""), loc="right", color=color))
                    drawing.add(elm.Line(color=color).endpoints((x - 0.55, ground_y), (x + 0.55, ground_y)))
                    drawing.add(elm.Ground(color=color).at((x, ground_y)))
            else:
                for section in range(first_section, last_section + 1):
                    left = step * (section - first_section)
                    right = left + step
                    for y in lane_y.values():
                        drawing.add(elm.Line(color=color).endpoints((left, y), (right, y)))
                    label = f"section {section}  R·dx / L·dx"
                    if show_values:
                        label += (
                            "\nR=" + section_matrix(path, "series_resistance", section)
                            + "\nL=" + section_matrix(path, "series_inductance", section)
                        )
                    drawing.add(elm.Rect((left + 0.7, bottom_y - 0.35), (right - 0.7, top_y + 0.35), color=color).label(label, color=color))
                for station in range(first_station, last_section + 1):
                    x = step * (station - first_station)
                    for conductor, y in lane_y.items():
                        record = stations[(station, conductor)]
                        provenance = (
                            f"{station}.{conductor}: {record['compiled_node_id']} "
                            f"[{record['attachment']}; L={half_shunt(record['left_half_shunt'])}; "
                            f"R={half_shunt(record['right_half_shunt'])}]"
                        )
                        drawing.add(elm.Dot(open=True, color=color, fill=background).at((x, y)).label(provenance, loc="left", color=color))
                    first = stations[(station, conductors[0])]
                    label = f"station {station}  C/G matrix"
                    if show_values:
                        label += (
                            "\nC=" + matrix_text(first.get("compiled_capacitance_total"))
                            + "\nG=" + matrix_text(first.get("compiled_conductance_total"))
                        )
                    drawing.add(elm.Rect((x - 0.45, ground_y + 0.5), (x + 0.45, bottom_y - 0.35), color=color).label(label, loc="right", color=color))
                    drawing.add(elm.Line(color=color).endpoints((x - 0.6, ground_y), (x + 0.6, ground_y)))
                    drawing.add(elm.Ground(color=color).at((x, ground_y)))
            cursor_y = ground_y - 2.5
    return cursor_y


class CircuitPlan:
    """The single physical authority for one reusable circuit model."""

    __slots__ = (
        "_id", "_sealed", "_components", "_nodes", "_grounded", "_ground_groups", "_ports", "_pin_nodes", "_couplings", "_coordinate_resolution"
    )

    def __init__(self, *, id: str) -> None:
        self._id = _identifier(id, field="plan id")
        if self._id == "ground":
            raise ValueError("plan id cannot be the reserved ground identity")
        self._sealed = False
        self._components: list[ComponentInstance] = []
        self._nodes: list[_PlanNode] = []
        self._grounded: set[PinRef] = set()
        self._ground_groups: list[tuple[PinRef, ...]] = []
        self._ports: list[PortRef] = []
        self._pin_nodes: dict[PinRef, _PlanNode | Literal["ground"]] = {}
        self._couplings: list[dict[str, object]] = []
        self._coordinate_resolution: dict[CoordinateRef, str] | None = None

    @property
    def id(self) -> str:
        """User-supplied stable Plan identity."""

        return self._id

    @property
    def sealed(self) -> bool:
        """Whether the Plan has been permanently consumed by a Run."""

        return self._sealed

    @property
    def components(self) -> tuple[ComponentInstance, ...]:
        return tuple(self._components)

    @property
    def nodes(self) -> tuple[ElectricNodeRef, ...]:
        return tuple(ElectricNodeRef._create(self, node) for node in self._nodes)

    @property
    def ports(self) -> tuple[PortRef, ...]:
        return tuple(self._ports)

    def _coordinate_ids(self) -> set[str]:
        return {coordinate.id for component in self._components for coordinate in component._coordinates.values()}

    def _assert_mutable(self) -> None:
        if self._sealed:
            raise PlanSealedError("CircuitPlan is permanently sealed", stage="plan_mutation")

    def _invalidate_coordinate_resolution(self) -> None:
        self._coordinate_resolution = None

    def _resolve_composite_coordinates(self) -> Mapping[CoordinateRef, str]:
        """Resolve Composite selectors to their sealed outer-Plan coordinate IDs."""

        if self._coordinate_resolution is not None:
            return MappingProxyType(self._coordinate_resolution)
        resolved: dict[CoordinateRef, str] = {}
        reserved = {node.id for node in self._nodes if node.visibility != "internal"}
        for component in self._components:
            realization = component._realization
            pin_nodes = {item["public_id"]: item["private_node_id"] for item in realization.get("public_pin_map", ())}
            for record in realization.get("public_coordinate_map", ()):
                coordinate = component._coordinates[record["public_id"]]
                matched_pins = [pin_id for pin_id, node_id in pin_nodes.items() if node_id == record["private_node_id"]]
                attached = [self._pin_nodes[component.pin(pin_id)] for pin_id in matched_pins]
                if "ground" in attached:
                    raise SCNSimValidationError("a Composite public coordinate cannot resolve to ground", stage="plan_seal")
                nodes = [item for item in attached if isinstance(item, _PlanNode)]
                if len({id(node) for node in nodes}) > 1:
                    raise SCNSimValidationError("one Composite public coordinate cannot resolve to multiple outer nodes", stage="plan_seal")
                if nodes:
                    node = nodes[0]
                    if node.visibility == "internal":
                        promoted = coordinate.id
                        if promoted in reserved or any(other.id == promoted for other in self._nodes if other is not node):
                            raise SCNSimValidationError("promoted Composite coordinate ID collides with an outer Plan node", stage="plan_seal")
                        reserved.discard(node.id)
                        node.id, node.visibility = promoted, "public"
                        reserved.add(promoted)
                    resolved[coordinate] = node.id
                else:
                    if coordinate.id in reserved or coordinate.id in resolved.values():
                        raise SCNSimValidationError("Composite coordinate-only ID collides with an outer Plan coordinate", stage="plan_seal")
                    resolved[coordinate] = coordinate.id
                    reserved.add(coordinate.id)
        self._coordinate_resolution = resolved
        return MappingProxyType(resolved)

    def _resolve_coordinate(self, coordinate: CoordinateRef) -> str:
        """Return the exact sealed Plan coordinate for one Composite selector."""

        if not isinstance(coordinate, CoordinateRef) or coordinate._component not in self._components:
            raise SCNSimValidationError("coordinate does not belong to this Plan", stage="plan_seal")
        try:
            return self._resolve_composite_coordinates()[coordinate]
        except KeyError:
            raise SCNSimValidationError("coordinate is not exposed by this Composite", stage="plan_seal") from None

    def _validate_pin(self, pin: PinRef) -> None:
        if not isinstance(pin, PinRef) or pin._component not in self._components:
            raise SCNSimValidationError("pin does not belong to this Plan", stage="authoring")
        if pin in self._pin_nodes:
            raise SCNSimValidationError("each pin may belong to one node or ground", stage="authoring")

    def add(self, component: ComponentInstance) -> ComponentInstance:
        """Add one immutable built-in or custom component snapshot."""

        self._assert_mutable()
        if not isinstance(component, ComponentInstance):
            raise TypeError("CircuitPlan.add requires a ComponentInstance")
        if any(binding.get("kind") != "constant" for binding in component._realization.get("bindings", {}).values()):
            raise SCNSimValidationError("CircuitPlan cannot own cross-component parameter bindings", stage="authoring")
        if any(existing.id == component.id for existing in self._components):
            raise SCNSimValidationError("component IDs must be unique", stage="authoring")
        coordinates = {coordinate.id for coordinate in component._coordinates.values()}
        if coordinates & (self._coordinate_ids() | {node.id for node in self._nodes} | {port.id for port in self._ports}):
            raise SCNSimValidationError("public coordinate IDs must be unique in a CircuitPlan", stage="authoring")
        self._components.append(component)
        self._invalidate_coordinate_resolution()
        return component

    def net(self, *pins: PinRef, id: str | None = None) -> ElectricNodeRef:
        """Create one complete equipotential node from exactly these pins."""

        self._assert_mutable()
        if not pins:
            raise SCNSimValidationError("net requires at least one pin", stage="authoring")
        if len(set(pins)) != len(pins):
            raise SCNSimValidationError("net cannot repeat a pin", stage="authoring")
        for pin in pins:
            self._validate_pin(pin)
        endpoints = [pin._endpoint() for pin in pins]
        if id is None:
            from ._canonical import internal_node_id

            node_id = internal_node_id(endpoints)
            visibility: Literal["public", "internal", "port_promoted"] = "internal"
        else:
            node_id = _identifier(id, field="node id")
            if node_id == "ground" or node_id.startswith("internal-"):
                raise SCNSimValidationError("node ID is reserved", stage="authoring")
            visibility = "public"
        if node_id in self._coordinate_ids():
            raise SCNSimValidationError("node ID collides with a composite public coordinate", stage="authoring")
        if any(node.id == node_id for node in self._nodes):
            raise SCNSimValidationError("node IDs must be unique", stage="authoring")
        node = _PlanNode(node_id, visibility, tuple(pins))
        self._nodes.append(node)
        self._pin_nodes.update({pin: node for pin in pins})
        self._invalidate_coordinate_resolution()
        return ElectricNodeRef._create(self, node)

    def ground(self, *pins: PinRef) -> None:
        """Attach one nonempty local pin group to the canonical Plan ground."""

        self._assert_mutable()
        if not pins:
            raise SCNSimValidationError("ground requires at least one pin", stage="authoring")
        if len(set(pins)) != len(pins):
            raise SCNSimValidationError("ground cannot repeat a pin", stage="authoring")
        for pin in pins:
            self._validate_pin(pin)
        self._grounded.update(pins)
        self._ground_groups.append(tuple(pins))
        self._pin_nodes.update({pin: "ground" for pin in pins})
        self._invalidate_coordinate_resolution()

    def add_port(
        self,
        *,
        id: str,
        at: ElectricNodeRef,
        role: Literal["terminated", "nonloading_probe"],
        reference_impedance: Quantity,
    ) -> PortRef:
        """Attach one logical Port to an existing Plan node."""

        self._assert_mutable()
        port_id = _identifier(id, field="port id")
        if port_id in self._coordinate_ids():
            raise SCNSimValidationError("port ID collides with a composite public coordinate", stage="authoring")
        if not isinstance(at, ElectricNodeRef) or at._plan is not self:
            raise SCNSimValidationError("Port must attach to an ElectricNodeRef from this Plan", stage="authoring")
        if role not in ("terminated", "nonloading_probe"):
            raise ValueError("role must be 'terminated' or 'nonloading_probe'")
        if any(port.id == port_id for port in self._ports):
            raise SCNSimValidationError("port IDs must be unique", stage="authoring")
        if any(port.node._node is at._node for port in self._ports):
            raise SCNSimValidationError("one electric node may own at most one Port", stage="authoring")
        if at._node.visibility == "internal":
            at._node.id = port_id
            at._node.visibility = "port_promoted"
        reference = require_positive_quantity(reference_impedance, "ohm", name="reference_impedance")
        port = PortRef._create(self, port_id, at, role, reference)
        self._ports.append(port)
        self._invalidate_coordinate_resolution()
        return port

    def _validate_complete(self, *, check_anonymous_nodes: bool = True) -> None:
        if not self._components:
            raise SCNSimValidationError("CircuitPlan requires at least one component", stage="plan_seal")
        missing = [
            pin.component_id + "." + pin.name
            for component in self._components
            for pin in component._pins.values()
            if pin not in self._pin_nodes
        ]
        if missing:
            raise SCNSimValidationError("every component pin must be netted or grounded", stage="plan_seal", evidence={"missing_pins": missing})
        if check_anonymous_nodes and any(node.visibility == "internal" and len(node.endpoints) < 2 for node in self._nodes):
            raise SCNSimValidationError(
                "an anonymous node without a Port must join at least two pins",
                stage="plan_seal",
            )
        _validate_coupling_graph(self._couplings)

    def _seal(self) -> CircuitPlan:
        """Validate and permanently seal the Plan for ``CircuitRun``."""

        if not self._sealed:
            self._validate_complete(check_anonymous_nodes=False)
            self._resolve_composite_coordinates()
            self._validate_complete()
            self._sealed = True
        return self

    def _canonical_snapshot(self) -> dict[str, object]:
        """Return the closed Plan payload consumed by canonical encoding."""

        self._validate_complete(check_anonymous_nodes=False)
        self._resolve_composite_coordinates()
        self._validate_complete()
        catalog_sources: dict[str, dict[str, object]] = {}
        for component in _recursive_components(self._components):
            source = dict(component._catalog_source)
            existing = catalog_sources.setdefault(component.catalog_id, source)
            if existing != source:
                raise SCNSimValidationError(
                    "components from one catalog must share one captured source identity",
                    stage="plan_seal",
                )
        return {
            "plan_id": self.id,
            "catalog_sources": list(catalog_sources.values()),
            "components": [self._component_snapshot(component) for component in self._components],
            "nodes": [
                {
                    "node_id": node.id,
                    "visibility": node.visibility,
                    "endpoints": [pin._endpoint() for pin in node.endpoints],
                }
                for node in self._nodes
            ],
            "grounded_endpoints": [pin._endpoint() for pin in self._grounded],
            "ports": [
                {
                    "port_id": port.id,
                    "node_id": port.node.id,
                    "role": port.role,
                    "reference_impedance": _quantity_record(port.reference_impedance, "ohm"),
                    "orientation": "node_to_reference",
                }
                for port in self._ports
            ],
            "couplings": list(self._couplings),
        }

    def _component_snapshot(self, component: ComponentInstance) -> dict[str, object]:
        """Project Plan-owned Composite-coordinate resolution without changing its snapshot."""

        snapshot = component._canonical_snapshot()
        realization = snapshot["realization"]
        if realization.get("kind") == "composite":
            resolved = self._resolve_composite_coordinates()
            for record in realization["public_coordinate_map"]:
                for coordinate in component._coordinates.values():
                    if record["public_id"] == coordinate.id:
                        record["public_id"] = resolved[coordinate]
                        break
        return snapshot

    _canonical_record = _canonical_snapshot

    def render_schematic(self, spec: CircuitDiagramSpec | None = None) -> CircuitDiagramResult:
        """Materialize a read-only authoring or declared-line compiled schematic."""

        self._validate_complete()
        from .results import CircuitDiagramResult, _verified_result
        from .specs import CircuitDiagramSpec
        from .presentation import _palette, _themed_drawing

        spec = CircuitDiagramSpec() if spec is None else spec
        if not isinstance(spec, CircuitDiagramSpec):
            raise TypeError("render_schematic() requires CircuitDiagramSpec")
        try:
            import schemdraw.elements as elm
        except ImportError as exc:
            raise RuntimeError("Schemdraw is required for authoring schematics") from exc
        palette = _palette(spec.theme)
        color = palette.foreground
        background = palette.background
        if spec.representation == "compiled":
            from .runtime import _compiled_schematic_evidence

            compiled = _compiled_schematic_evidence(self)
            drawing = _themed_drawing(spec.theme)
            drawing.config(unit=1.0, color=color, bgcolor=background, lw=1.2, fontsize=8)
            legend_y = _draw_compiled_ladders(
                drawing,
                compiled,
                color=color,
                background=background,
                show_values=spec.show_parameter_values,
            )
            for index, line in enumerate(
                _compiled_line_audit(compiled, show_values=spec.show_parameter_values),
                start=1,
            ):
                y = legend_y - float(index)
                drawing.add(
                    elm.Line(color=color)
                    .endpoints((0.0, y), (0.2, y))
                    .label(line, loc="right", color=color)
                )
            return _verified_result(
                CircuitDiagramResult, drawing=drawing, representation="compiled"
            )
        drawing = _themed_drawing(spec.theme)
        drawing.config(unit=2.5, color=color, bgcolor=background, lw=1.8, fontsize=11)
        node_y = {id(node): 3.0 * (index + 1) for index, node in enumerate(self._nodes)}
        rail_end = 3.0 * (len(self._components) + 1)
        for node in self._nodes:
            y = node_y[id(node)]
            drawing.add(elm.Line(color=color).endpoints((-1.0, y), (rail_end, y)))
            drawing.add(elm.Dot(open=True, color=color, fill=background).at((-1.0, y)).label(node.id, loc="left", color=color))
        ground_y = 0.0
        drawing.add(elm.Line(color=color).endpoints((-1.0, ground_y), (rail_end, ground_y)))
        for index, component in enumerate(self._components, start=1):
            x = 3.0 * index
            pins = tuple(component._pins.values())
            parameter = next(iter(component._parameters.values()), None)
            label = component.id if parameter is None or not spec.show_parameter_values else f"{component.id}\n{parameter.baseline:~P}"
            if component._ground_groups:
                label += "\nGND"
            if tuple(component._pins) != ("terminal_1", "terminal_2"):
                ys = [ground_y if self._pin_nodes[pin] == "ground" else node_y[id(self._pin_nodes[pin])] for pin in pins]
                drawing.add(elm.Rect((x - 0.55, min(ys) - 0.3), (x + 0.55, max(ys) + 0.3), color=color).label(label, loc="right", color=color))
                continue
            terminal_1, terminal_2 = pins
            target_1 = self._pin_nodes[terminal_1]
            target_2 = self._pin_nodes[terminal_2]
            y1 = ground_y if target_1 == "ground" else node_y[id(target_1)]
            y2 = ground_y if target_2 == "ground" else node_y[id(target_2)]
            direction = "down" if y1 >= y2 else "up"
            element_class = {
                "resistor": elm.Resistor,
                "capacitor": elm.Capacitor,
                "inductor": elm.Inductor,
                "josephson_junction": elm.Josephson,
                "interdigitated_capacitor": elm.Capacitor,
                "symmetric_squid": elm.Inductor2,
            }.get(component.factory, elm.RBox if "resonator" in component.factory else elm.Resistor)
            element = getattr(element_class(color=color).at((x, y1)), direction)().length(abs(y1 - y2) or 0.2)
            drawing.add(element.label(label, loc="right", color=color))
        for group in self._ground_groups:
            first = group[0]
            component_index = self._components.index(first._component) + 1
            drawing.add(elm.Ground(color=color).at((3.0 * component_index, ground_y)))
        for port in self._ports:
            y = node_y[id(port.node._node)]
            drawing.add(
                elm.Dot(open=True, color=color, fill=background)
                .at((rail_end, y))
                .label(f"{port.id} ({port.reference_impedance:~P})", loc="right", color=color)
            )
        return _verified_result(
            CircuitDiagramResult, drawing=drawing, representation="authoring"
        )

    def couple_inductive(
        self,
        *,
        id: str,
        inductor_a: InductiveBranchRef,
        inductor_b: InductiveBranchRef,
        coupling_coefficient: Quantity,
    ) -> None:
        self._assert_mutable()
        coupling = _coupling(id, inductor_a, inductor_b, coupling_coefficient, self._components)
        self._couplings.append(coupling)


components = _BuiltinComponents()
"""The immutable built-in SCNSim component catalog."""
