"""Pure prepared declarations between a sealed Run and canonical execution.

The records in this module deliberately carry no ``CircuitRun``, workspace,
process callback, or candidate realization. Runtime ownership checks and
parameter resolution happen before construction; the canonical request
encoder consumes only these finalized bytes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from . import units
from ._canonical import (
    canonical_json_bytes,
    canonical_request_document,
    complex_quantity_envelope,
    float64_hex,
    quantity_envelope,
)
from ._physical_values import RLGCParameterSpec
from .authoring import CoordinateRef, ElectricNodeRef, ParameterSet
from .errors import CompilerInvariantError, InvalidOptimizationSpec
from .specs import (
    DiagonalRootSpec,
    OperatorElementRootSpec,
    DirectSolveSpec,
    HBSolveSpec,
    HybridizedPoleSpec,
    OperatorSpec,
    OptimizationSpec,
    QuantityAbsolute,
    QuantityDifference,
    QuantitySelector,
    QuantitySum,
    ResidueNormalizedCouplingSpec,
    ResponseElementSpec,
    SParameterTrace,
    TransferZeroSpec,
    _expression_unit,
    _selector_unit,
)


@dataclass(frozen=True, slots=True)
class BoundOptimizationLeaf:
    """One authored objective leaf and its separately shareable dependency."""

    objective_id: str
    objective_ordinal: int
    term_ordinal: int
    projection: str
    declaration_bytes: bytes
    dependency_bytes: bytes

    @classmethod
    def create(
        cls,
        *,
        objective_id: str,
        objective_ordinal: int,
        term_ordinal: int,
        declaration: Mapping[str, object],
    ) -> BoundOptimizationLeaf:
        if (
            not isinstance(objective_id, str)
            or not objective_id
            or not isinstance(objective_ordinal, int)
            or isinstance(objective_ordinal, bool)
            or objective_ordinal < 0
            or not isinstance(term_ordinal, int)
            or isinstance(term_ordinal, bool)
            or term_ordinal < 0
        ):
            raise CompilerInvariantError(
                "bound optimization leaf identity is malformed",
                stage="request_encode",
            )
        projection = declaration.get("projection")
        selector_type = declaration.get("type")
        spec = declaration.get("spec")
        view = declaration.get("view")
        if (
            not isinstance(projection, str)
            or not projection
            or not isinstance(selector_type, str)
            or not selector_type
            or not isinstance(spec, Mapping)
            or not isinstance(view, Mapping)
        ):
            raise CompilerInvariantError(
                "bound optimization leaf declaration is malformed",
                stage="request_encode",
            )
        return cls(
            objective_id=objective_id,
            objective_ordinal=objective_ordinal,
            term_ordinal=term_ordinal,
            projection=projection,
            declaration_bytes=canonical_json_bytes(declaration),
            dependency_bytes=canonical_json_bytes(
                {"type": selector_type, "spec": spec, "view": view}
            ),
        )

    def declaration(self) -> dict[str, object]:
        return json.loads(self.declaration_bytes)

    @property
    def dependency_sha256(self) -> str:
        return sha256(self.dependency_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class BoundOptimization:
    """One completely normalized optimization declaration and ordered leaves."""

    spec_bytes: bytes
    leaves: tuple[BoundOptimizationLeaf, ...]

    @classmethod
    def create(
        cls,
        *,
        spec: Mapping[str, object],
        leaves: Sequence[BoundOptimizationLeaf],
    ) -> BoundOptimization:
        checked = tuple(leaves)
        if not checked or not all(
            isinstance(leaf, BoundOptimizationLeaf) for leaf in checked
        ):
            raise CompilerInvariantError(
                "bound optimization requires ordered leaves",
                stage="request_encode",
            )
        expected = sorted(
            checked,
            key=lambda leaf: (leaf.objective_ordinal, leaf.term_ordinal),
        )
        if list(checked) != expected or len(
            {(leaf.objective_ordinal, leaf.term_ordinal) for leaf in checked}
        ) != len(checked):
            raise CompilerInvariantError(
                "bound optimization leaf order is not canonical",
                stage="request_encode",
            )
        return cls(canonical_json_bytes(spec), checked)

    def spec(self) -> dict[str, object]:
        return json.loads(self.spec_bytes)


@dataclass(frozen=True, slots=True)
class PreparedAnalysis:
    """Final immutable request declaration with non-executable source evidence."""

    request_bytes: bytes
    source_unit_bytes: tuple[bytes, ...]
    bound_optimization: BoundOptimization | None = None

    @classmethod
    def create(
        cls,
        *,
        plan_sha256: str,
        operation: str,
        view: Mapping[str, object],
        spec: Mapping[str, object],
        parameter_source: Mapping[str, object],
        runtime_semantic: Mapping[str, object],
        source_units: Sequence[Mapping[str, object]],
        bound_optimization: BoundOptimization | None = None,
    ) -> PreparedAnalysis:
        request = canonical_request_document(
            plan_sha256=plan_sha256,
            operation=operation,
            view=view,
            spec=spec,
            parameter_source=parameter_source,
            runtime_semantic=runtime_semantic,
        )
        return cls(
            request_bytes=canonical_json_bytes(request),
            source_unit_bytes=tuple(
                canonical_json_bytes(record) for record in source_units
            ),
            bound_optimization=bound_optimization,
        )

    def request(self) -> dict[str, object]:
        return json.loads(self.request_bytes)

    def source_units(self) -> tuple[dict[str, object], ...]:
        return tuple(json.loads(record) for record in self.source_unit_bytes)

    @property
    def request_sha256(self) -> str:
        return sha256(self.request_bytes).hexdigest()


def _coordinate_binding_key(
    value: str | ElectricNodeRef | CoordinateRef,
) -> tuple[str, str]:
    if isinstance(value, str):
        if not value:
            raise ValueError("coordinate IDs must not be empty")
        return "string", value
    if isinstance(value, ElectricNodeRef):
        return "electric_node", value.id
    if isinstance(value, CoordinateRef):
        return (
            "coordinate",
            canonical_json_bytes(
                {"scope": list(value.scope.path()), "id": value.id}
            ).decode("utf-8"),
        )
    raise TypeError("coordinate must be a public SCNSim coordinate handle or ID")


def _quantity_coordinates(
    spec: object,
) -> tuple[str | ElectricNodeRef | CoordinateRef, ...]:
    if isinstance(spec, DiagonalRootSpec):
        return (spec.coordinate,)
    if isinstance(spec, OperatorElementRootSpec):
        return spec.row, spec.column
    if isinstance(spec, HybridizedPoleSpec):
        return tuple(spec.coordinates)
    if isinstance(spec, TransferZeroSpec):
        return spec.input_coordinate, spec.output_coordinate
    if isinstance(spec, ResidueNormalizedCouplingSpec):
        return (
            *_quantity_coordinates(spec.branch_a),
            *_quantity_coordinates(spec.branch_b),
        )
    if isinstance(spec, ResponseElementSpec):
        return spec.input_coordinate, spec.output_coordinate
    if isinstance(spec, OperatorSpec):
        return ()
    raise TypeError("unsupported Direct quantity Spec")


def _bound_coordinate_id(
    value: str | ElectricNodeRef | CoordinateRef,
    coordinate_bindings: Mapping[tuple[str, str], str],
) -> str:
    try:
        return coordinate_bindings[_coordinate_binding_key(value)]
    except KeyError as error:
        raise CompilerInvariantError(
            "prepared coordinate binding is incomplete",
            stage="request_encode",
        ) from error


def _frequency_grid(value: object) -> list[dict[str, str]]:
    converted = value.to("hertz")
    magnitudes = np.asarray(converted.magnitude, dtype=np.float64)
    return [
        quantity_envelope(
            units.registry.Quantity(float(item), "hertz"),
            si_unit="hertz",
            registry=units.registry,
        )
        for item in magnitudes
    ]


def _frequency_anchor_envelope(value: object) -> dict[str, str]:
    """Preserve an authored complex seed, including an explicit ``x + 0j``."""

    magnitude = getattr(value, "magnitude", None)
    authored_complex = (
        isinstance(magnitude, complex)
        or getattr(getattr(magnitude, "dtype", None), "kind", None) == "c"
    )
    encoder = complex_quantity_envelope if authored_complex else quantity_envelope
    return encoder(value, si_unit="hertz", registry=units.registry)


def _encode_root(
    spec: DiagonalRootSpec,
    *,
    coordinate_bindings: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    return {
        "type": "diagonal_root",
        "coordinate": _bound_coordinate_id(spec.coordinate, coordinate_bindings),
        "root_hint": quantity_envelope(
            spec.root_hint, si_unit="hertz", registry=units.registry
        ),
    }


def _encode_element_root(
    spec: OperatorElementRootSpec,
    *,
    coordinate_bindings: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    return {
        "type": "operator_element_root",
        "row": _bound_coordinate_id(spec.row, coordinate_bindings),
        "column": _bound_coordinate_id(spec.column, coordinate_bindings),
        "root_hint": quantity_envelope(spec.root_hint, si_unit="hertz", registry=units.registry),
    }


def _encode_scalar_expression(
    value: object,
    *,
    coordinate_bindings: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    if isinstance(value, QuantitySelector):
        return {
            "type": value.type,
            "spec": _encode_direct_quantity(
                value.spec, coordinate_bindings=coordinate_bindings
            ),
            "projection": value.projection,
        }
    if isinstance(value, QuantitySum):
        return {
            "type": "quantity_sum",
            "terms": [
                _encode_scalar_expression(
                    term,
                    coordinate_bindings=coordinate_bindings,
                )
                for term in value.terms
            ],
        }
    if isinstance(value, QuantityDifference):
        return {
            "type": "quantity_difference",
            "left": _encode_scalar_expression(
                value.left, coordinate_bindings=coordinate_bindings
            ),
            "right": _encode_scalar_expression(
                value.right, coordinate_bindings=coordinate_bindings
            ),
        }
    if isinstance(value, QuantityAbsolute):
        return {
            "type": "quantity_absolute",
            "operand": _encode_scalar_expression(
                value.operand, coordinate_bindings=coordinate_bindings
            ),
        }
    raise InvalidOptimizationSpec(
        "objective quantity must be a supported scalar expression",
        stage="spec_validation",
    )


def _encode_direct_quantity(
    spec: DiagonalRootSpec
    | OperatorElementRootSpec
    | HybridizedPoleSpec
    | TransferZeroSpec
    | ResidueNormalizedCouplingSpec
    | ResponseElementSpec
    | OperatorSpec,
    *,
    coordinate_bindings: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    if isinstance(spec, DiagonalRootSpec):
        return _encode_root(spec, coordinate_bindings=coordinate_bindings)
    if isinstance(spec, OperatorElementRootSpec):
        return _encode_element_root(spec, coordinate_bindings=coordinate_bindings)
    if isinstance(spec, HybridizedPoleSpec):
        return {
            "type": "hybridized_pole",
            "coordinates": [
                _bound_coordinate_id(value, coordinate_bindings)
                for value in spec.coordinates
            ],
            "anchor": _frequency_anchor_envelope(spec.anchor),
        }
    if isinstance(spec, TransferZeroSpec):
        return {
            "type": "transfer_zero",
            "anchor": _frequency_anchor_envelope(spec.anchor),
            "family": spec.family,
            "input_coordinate": _bound_coordinate_id(
                spec.input_coordinate, coordinate_bindings
            ),
            "output_coordinate": _bound_coordinate_id(
                spec.output_coordinate, coordinate_bindings
            ),
        }
    if isinstance(spec, ResidueNormalizedCouplingSpec):
        return {
            "type": "residue_normalized_coupling",
            "branch_a": _encode_direct_quantity(
                spec.branch_a, coordinate_bindings=coordinate_bindings
            ),
            "branch_b": _encode_direct_quantity(
                spec.branch_b, coordinate_bindings=coordinate_bindings
            ),
            "frequency": (
                spec.frequency
                if isinstance(spec.frequency, str)
                else quantity_envelope(
                    spec.frequency, si_unit="hertz", registry=units.registry
                )
            ),
        }
    if isinstance(spec, ResponseElementSpec):
        return {
            "type": "response_element",
            "family": spec.family,
            "input_coordinate": _bound_coordinate_id(
                spec.input_coordinate, coordinate_bindings
            ),
            "output_coordinate": _bound_coordinate_id(
                spec.output_coordinate, coordinate_bindings
            ),
            "frequency": quantity_envelope(
                spec.frequency, si_unit="hertz", registry=units.registry
            ),
        }
    if isinstance(spec, OperatorSpec):
        return {"type": "operator", "frequencies": _frequency_grid(spec.frequencies)}
    raise TypeError("unsupported Direct quantity Spec")


def _encode_spec(
    spec: DirectSolveSpec
    | HBSolveSpec
    | DiagonalRootSpec
    | OperatorElementRootSpec
    | HybridizedPoleSpec
    | TransferZeroSpec
    | ResidueNormalizedCouplingSpec
    | ResponseElementSpec
    | OperatorSpec
    | OptimizationSpec,
    parameters: ParameterSet,
    *,
    coordinate_bindings: Mapping[tuple[str, str], str],
    trace_channels: Mapping[str, str],
    optimization_quantities: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    def trace_record(trace: SParameterTrace) -> dict[str, object]:
        record = dict(trace._canonical_record())
        try:
            record["input_port"] = trace_channels[trace.input_port]
            record["output_port"] = trace_channels[trace.output_port]
        except KeyError as error:
            raise CompilerInvariantError(
                "prepared trace-channel binding is incomplete",
                stage="request_encode",
            ) from error
        return record

    if isinstance(spec, DirectSolveSpec):
        return {
            "type": "direct_solve",
            "frequencies": _frequency_grid(spec.frequencies),
            "traces": [trace_record(trace) for trace in spec.traces],
        }
    if isinstance(spec, HBSolveSpec):
        return {
            "type": "hb_solve",
            "pump_axes": [
                {
                    "id": axis.id,
                    "frequency": quantity_envelope(
                        axis.frequency, si_unit="hertz", registry=units.registry
                    ),
                }
                for axis in spec.pump_axes
            ],
            "drives": [
                {
                    "id": drive.id,
                    "port_id": drive.at.id,
                    "mode": list(drive.mode),
                    "orientation": "port_node_to_reference",
                }
                for drive in spec.drives
            ],
            "frequencies": _frequency_grid(spec.frequencies),
            "cases": [
                {
                    "id": case.id,
                    "currents": [
                        {
                            "drive_id": drive.id,
                            "coefficient": complex_quantity_envelope(
                                case.currents[drive],
                                si_unit="ampere",
                                registry=units.registry,
                            ),
                            "coefficient_convention": "exp_minus_i_m_dot_omega_t_fourier_coefficient",
                        }
                        for drive in spec.drives
                        if drive in case.currents
                    ],
                }
                for case in spec.cases
            ],
            "truncation": {
                "pump_harmonics": list(spec.truncation.pump_harmonics),
                "modulation_harmonics": list(spec.truncation.modulation_harmonics),
                "max_intermodulation_order": spec.truncation.max_intermodulation_order,
                "three_wave_mixing": spec.truncation.three_wave_mixing,
                "four_wave_mixing": spec.truncation.four_wave_mixing,
            },
            "traces": [trace_record(trace) for trace in spec.traces],
            "allow_driven_ptc": spec.allow_driven_ptc,
        }
    if isinstance(
        spec,
        (
            DiagonalRootSpec,
            OperatorElementRootSpec,
            HybridizedPoleSpec,
            TransferZeroSpec,
            ResidueNormalizedCouplingSpec,
            ResponseElementSpec,
            OperatorSpec,
        ),
    ):
        return _encode_direct_quantity(spec, coordinate_bindings=coordinate_bindings)
    if optimization_quantities is None or len(optimization_quantities) != len(
        spec.objectives
    ):
        raise CompilerInvariantError(
            "optimization quantity normalization is incomplete",
            stage="request_encode",
        )
    variables: list[dict[str, object]] = []
    baseline_optimizer_coordinates: list[str] = []
    for variable in spec.variables:
        parameter = variable.parameter
        if isinstance(parameter.spec, RLGCParameterSpec):
            raise InvalidOptimizationSpec(
                "RLGC parameters cannot be continuous optimization variables",
                stage="spec_validation",
            )
        parameter_unit = parameter.spec.si_unit
        lower, upper = variable.bounds
        low = quantity_envelope(lower, si_unit=parameter_unit, registry=units.registry)
        high = quantity_envelope(upper, si_unit=parameter_unit, registry=units.registry)
        low_value = float(lower.to(parameter_unit).magnitude)
        high_value = float(upper.to(parameter_unit).magnitude)
        baseline_value = float(
            parameters.values[parameter].to(parameter_unit).magnitude
        )
        if low_value >= high_value:
            raise InvalidOptimizationSpec(
                "optimization lower bound must be below upper bound",
                stage="spec_validation",
            )
        if not low_value <= baseline_value <= high_value:
            raise InvalidOptimizationSpec(
                "sealed baseline must lie within resolved variable bounds",
                stage="spec_validation",
            )
        if variable.transform == "log" and low_value <= 0.0:
            raise InvalidOptimizationSpec(
                "log optimization bounds must be strictly positive",
                stage="spec_validation",
            )
        coordinate = (
            (baseline_value - low_value) / (high_value - low_value)
            if variable.transform == "linear"
            else math.log(baseline_value / low_value)
            / math.log(high_value / low_value)
        )
        if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
            raise InvalidOptimizationSpec(
                "sealed baseline has no finite unit-box coordinate",
                stage="spec_validation",
            )
        baseline_optimizer_coordinates.append(float64_hex(coordinate))
        default = [
            quantity_envelope(item, si_unit=parameter_unit, registry=units.registry)
            for item in variable.model_default_bounds
        ]
        override = (
            None
            if variable.consumer_override_bounds is None
            else [
                quantity_envelope(item, si_unit=parameter_unit, registry=units.registry)
                for item in variable.consumer_override_bounds
            ]
        )
        variables.append(
            {
                "parameter": parameter._key_record(),
                "model_default_bounds": default,
                "consumer_override_bounds": override,
                "lower": low,
                "upper": high,
                "transform": variable.transform,
            }
        )
    n = len(variables)
    population = spec.optimizer.population_size or (4 + math.floor(3 * math.log(n)))
    generations = (spec.optimizer.max_evaluations - 1) // population
    if generations < 1:
        raise ValueError(
            "CMA-ES budget must fit the baseline and one complete generation"
        )
    unused = spec.optimizer.max_evaluations - (1 + generations * population)
    objectives: list[dict[str, object]] = []
    for objective, quantity in zip(spec.objectives, optimization_quantities):
        objective_unit = _expression_unit(objective.quantity)
        target = quantity_envelope(
            objective.target, si_unit=objective_unit, registry=units.registry
        )
        target_value = abs(float(objective.target.to(objective_unit).magnitude))
        if objective.scale is None:
            if target_value == 0.0:
                if objective.target.dimensionless:
                    scale_value = units.registry.Quantity(1.0, "dimensionless")
                    scale_source = "dimensionless_unity"
                else:
                    raise ValueError(
                        "a dimensional zero target requires an explicit objective scale"
                    )
            else:
                scale_value = units.registry.Quantity(target_value, objective_unit)
                scale_source = "relative_target"
        else:
            scale_value = objective.scale
            scale_source = "explicit"
        scale_magnitude = float(scale_value.to(objective_unit).magnitude)
        if not math.isfinite(scale_magnitude) or scale_magnitude <= 0.0:
            raise InvalidOptimizationSpec(
                "objective scale must be finite and strictly positive",
                stage="spec_validation",
            )
        weight = float(objective.weight.to("dimensionless").magnitude)
        if not math.isfinite(weight) or weight <= 0.0:
            raise InvalidOptimizationSpec(
                "objective weight must be finite and strictly positive",
                stage="spec_validation",
            )
        objectives.append(
            {
                "id": objective.id,
                "quantity": dict(quantity),
                "comparison": objective.comparison,
                "target": target,
                "weight_f64": float64_hex(weight),
                "resolved_scale": quantity_envelope(
                    scale_value, si_unit=objective_unit, registry=units.registry
                ),
                "scale_source": scale_source,
            }
        )
    return {
        "type": "optimization",
        "variables": variables,
        "objectives": objectives,
        "optimizer": {
            "type": "cma_es",
            "seed": spec.optimizer.seed,
            "max_evaluations": spec.optimizer.max_evaluations,
            "population_size": spec.optimizer.population_size,
            "resolved_population_size": population,
            "initial_sigma_f64": float64_hex(spec.optimizer.initial_sigma),
            "baseline_optimizer_coordinates_f64": baseline_optimizer_coordinates,
            "box_transform_id": "cmaes-jl-0.2.6-linquad-unit-box.v1",
            "complete_generations": generations,
            "unused_evaluations": unused,
            "hidden_stops": "disabled",
        },
        "allow_extrapolation": [
            parameter._key_record() for parameter in spec.allow_extrapolation
        ],
    }
