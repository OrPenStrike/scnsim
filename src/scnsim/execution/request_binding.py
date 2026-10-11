"""Request provenance and View encoding from explicit immutable declarations.

No Run, backend, Workspace, mutable operation or result authority is retained.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np
from .. import units
from ..authoring import ElectricNodeRef, CoordinateRef, ParameterRef, ParameterSet, ParameterSpace
from ..authoring.physical_values import RLGC, RLGCParameterSpec
from ..canonical import canonical_json_bytes, complex_quantity_envelope, quantity_envelope
from ..errors import CompilerInvariantError, InvalidOptimizationSpec
from .run_binding import _parameter_key, compatible_parameter
from ..specs import (
    DiagonalRootSpec,
    OperatorElementRootSpec,
    DirectSolveSpec,
    HBSolveSpec,
    HybridizedPoleSpec,
    OperatorSpec,
    OptimizationSpec,
    OptimizationProgress,
    QuantityAbsolute,
    QuantityDifference,
    QuantitySelector,
    QuantitySum,
    ReportSpec,
    ResidueNormalizedCouplingSpec,
    ResponseElementSpec,
    TransferZeroSpec,
    _selector_unit,
)

def _view_declaration(lineage: Mapping[str, object]) -> dict[str, object]:
    """Project a lazy Python View to the request's declarative-only record."""

    ptc = lineage.get("ptc")
    transforms = lineage.get("transforms")
    retain = lineage.get("retain")
    if transforms is None or not isinstance(transforms, Sequence) or isinstance(transforms, (str, bytes)):
        raise CompilerInvariantError("View transform declaration is malformed", stage="request_encode")
    return {
        "type": "network_view",
        "ptc": None if ptc is None else {"selected_ports": list(ptc["selected_ports"])},
        "transforms": [
            {
                "id": item["id"],
                "input_coordinates": list(item["input_coordinates"]),
                "output_coordinates": list(item["output_coordinates"]),
            }
            for item in transforms
        ],
        "retain": None if retain is None else {
            "retained_coordinates": list(retain["retained_coordinates"]),
        },
    }

def _source_unit_identity(
    *,
    scope: str,
    component_path: Sequence[str] = (),
    parameter_id: str,
    field: str,
) -> str:
    """Encode one source-unit authority without flattening path segments."""

    if (
        not isinstance(scope, str)
        or not scope
        or not isinstance(parameter_id, str)
        or not parameter_id
        or not isinstance(field, str)
        or not field
        or any(not isinstance(segment, str) or not segment for segment in component_path)
    ):
        raise CompilerInvariantError("source-unit provenance identity is malformed", stage="request_encode")
    return canonical_json_bytes(
        {
            "scope": scope,
            "component_path": list(component_path),
            "parameter_id": parameter_id,
            "field": field,
        }
    ).decode("utf-8")

def _quantity_selectors(value: object) -> tuple[QuantitySelector, ...]:
    if isinstance(value, QuantitySelector):
        return (value,)
    if isinstance(value, QuantitySum):
        selectors = tuple(
            selector
            for term in value.terms
            for selector in _quantity_selectors(term)
        )
        if selectors:
            return selectors
    if isinstance(value, QuantityDifference):
        return _quantity_selectors(value.left) + _quantity_selectors(value.right)
    if isinstance(value, QuantityAbsolute):
        return _quantity_selectors(value.operand)
    raise InvalidOptimizationSpec(
        "objectives require a closed Direct scalar expression",
        stage="spec_validation",
    )

def source_units(
        source_provenance,
        parameter_lookup,
        spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | OperatorElementRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        parameters: ParameterSet,
        *,
        parameter_space: ParameterSet | ParameterSpace | None,
    ) -> list[dict[str, object]]:
        captured = source_provenance.get("source_units")
        if not isinstance(captured, Sequence) or isinstance(captured, (str, bytes)):
            raise CompilerInvariantError(
                "snapshot source-unit provenance is missing",
                stage="request_encode",
            )
        evidence: list[dict[str, object]] = []
        identities: set[str] = set()
        for raw in captured:
            if not isinstance(raw, Mapping) or set(raw) != {
                "identity", "source_unit", "canonical_si_unit", "canonical_dimensionality",
            }:
                raise CompilerInvariantError(
                    "snapshot source-unit provenance is malformed",
                    stage="request_encode",
                )
            row = dict(raw)
            identity = row.get("identity")
            if not isinstance(identity, str) or not identity or identity in identities:
                raise CompilerInvariantError(
                    "snapshot source-unit provenance identities are invalid",
                    stage="request_encode",
                )
            identities.add(identity)
            evidence.append(row)

        def add(identity: str, value: object, si_unit: str) -> None:
            if identity in identities:
                raise CompilerInvariantError(
                    "source-unit provenance has duplicate parameter authority",
                    stage="request_encode",
                    evidence={"identity": identity},
                )
            identities.add(identity)
            magnitude = np.asarray(value.magnitude)
            probe = (
                value
                if magnitude.ndim == 0
                else units.registry.Quantity(float(magnitude.flat[0]), value.units)
            )
            source_magnitude = getattr(probe, "magnitude", None)
            encoded = (
                complex_quantity_envelope(probe, si_unit=si_unit, registry=units.registry)
                if isinstance(source_magnitude, complex) or getattr(getattr(source_magnitude, "dtype", None), "kind", None) == "c"
                else quantity_envelope(probe, si_unit=si_unit, registry=units.registry)
            )
            evidence.append(
                {
                    "identity": identity,
                    "source_unit": str(value.units),
                    "canonical_si_unit": encoded["si_unit"],
                    "canonical_dimensionality": encoded["dimensionality"],
                }
            )

        for parameter, value in parameters.values.items():
            definitions_id, identifier = _parameter_key(parameter)
            if isinstance(parameter.spec, RLGCParameterSpec):
                if not isinstance(value, RLGC):
                    raise CompilerInvariantError("resolved RLGC parameter is malformed", stage="request_encode")
                units_by_field = {
                    "resistance_per_length": "ohm / meter",
                    "inductance_per_length": "henry / meter",
                    "conductance_per_length": "siemens / meter",
                    "capacitance_per_length": "farad / meter",
                    "extraction_frequency": "hertz",
                }
                for field, quantity in value._source_quantities.items():
                    add(
                        _source_unit_identity(
                            scope="request_parameter_rlgc",
                            component_path=(definitions_id,),
                            parameter_id=identifier,
                            field=field,
                        ),
                        quantity,
                        units_by_field[field],
                    )
            else:
                source_unit = parameters._source_units.get(parameter)
                source_value = value if source_unit is None else value.to(source_unit)
                add(
                    _source_unit_identity(
                        scope="request_parameter",
                        component_path=(definitions_id,),
                        parameter_id=identifier,
                        field="value",
                    ),
                    source_value,
                    parameter.spec.si_unit,
                )
        if isinstance(parameter_space, ParameterSpace) and parameter_space.kind == "grid":
            for axis_index, ((parameter, values), source_units) in enumerate(
                zip(parameter_space.axes, parameter_space._axis_source_units)
            ):
                current = compatible_parameter(parameter_lookup, parameter)
                if isinstance(current.spec, RLGCParameterSpec):
                    units_by_field = {
                        "resistance_per_length": "ohm / meter",
                        "inductance_per_length": "henry / meter",
                        "conductance_per_length": "siemens / meter",
                        "capacitance_per_length": "farad / meter",
                        "extraction_frequency": "hertz",
                    }
                    for value_index, value in enumerate(values):
                        if not isinstance(value, RLGC):
                            raise CompilerInvariantError(
                                "grid RLGC parameter is malformed",
                                stage="request_encode",
                            )
                        for field, quantity in value._source_quantities.items():
                            add(
                                _source_unit_identity(
                                    scope="request_grid_axis_rlgc",
                                    component_path=(current.definitions_id,),
                                    parameter_id=current.id,
                                    field=f"{axis_index}:{value_index}:{field}",
                                ),
                                quantity,
                                units_by_field[field],
                            )
                    continue
                for value_index, (value, source_unit) in enumerate(zip(values, source_units)):
                    source_value = value if source_unit is None else value.to(source_unit)
                    add(
                        _source_unit_identity(
                            scope="request_grid_axis",
                            component_path=(current.definitions_id,),
                            parameter_id=current.id,
                            field=f"{axis_index}:{value_index}",
                        ),
                        source_value,
                        current.spec.si_unit,
                    )
        elif isinstance(parameter_space, ParameterSpace) and parameter_space.kind == "points":
            for point_index, point in enumerate(parameter_space._points):
                for parameter, value in point.values.items():
                    current = compatible_parameter(parameter_lookup, parameter)
                    if isinstance(current.spec, RLGCParameterSpec):
                        if not isinstance(value, RLGC):
                            raise CompilerInvariantError(
                                "listed RLGC parameter is malformed",
                                stage="request_encode",
                            )
                        units_by_field = {
                            "resistance_per_length": "ohm / meter",
                            "inductance_per_length": "henry / meter",
                            "conductance_per_length": "siemens / meter",
                            "capacitance_per_length": "farad / meter",
                            "extraction_frequency": "hertz",
                        }
                        for field, quantity in value._source_quantities.items():
                            add(
                                _source_unit_identity(
                                    scope="request_listed_point_rlgc",
                                    component_path=(current.definitions_id,),
                                    parameter_id=current.id,
                                    field=f"{point_index}:{field}",
                                ),
                                quantity,
                                units_by_field[field],
                            )
                        continue
                    source_unit = point._source_units.get(parameter)
                    if source_unit is None:
                        continue
                    add(
                        _source_unit_identity(
                            scope="request_listed_point",
                            component_path=(current.definitions_id,),
                            parameter_id=current.id,
                            field=f"{point_index}",
                        ),
                        value.to(source_unit),
                        current.spec.si_unit,
                    )
        if isinstance(spec, DirectSolveSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="frequencies", field="value"), spec.frequencies, "hertz")
        elif isinstance(spec, HBSolveSpec):
            add(_source_unit_identity(scope="request_hb", parameter_id="frequencies", field="value"), spec.frequencies, "hertz")
            for axis in spec.pump_axes:
                add(_source_unit_identity(scope="request_hb", parameter_id=axis.id, field="pump_frequency"), axis.frequency, "hertz")
            for case in spec.cases:
                for drive in spec.drives:
                    if drive in case.currents:
                        add(
                            _source_unit_identity(
                                scope="request_hb",
                                parameter_id=case.id,
                                field=f"drive:{drive.id}:coefficient",
                            ),
                            case.currents[drive],
                            "ampere",
                        )
        elif isinstance(spec, DiagonalRootSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="root_hint", field="value"), spec.root_hint, "hertz")
        elif isinstance(spec, OperatorElementRootSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="root_hint", field="value"), spec.root_hint, "hertz")
        elif isinstance(spec, HybridizedPoleSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="hybridized_pole", field="anchor"), spec.anchor, "hertz")
        elif isinstance(spec, TransferZeroSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="transfer_zero", field="anchor"), spec.anchor, "hertz")
        elif isinstance(spec, ResponseElementSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="response_element", field="frequency"), spec.frequency, "hertz")
        elif isinstance(spec, OperatorSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="operator", field="frequencies"), spec.frequencies, "hertz")
        elif isinstance(spec, ResidueNormalizedCouplingSpec):
            if not isinstance(spec.frequency, str):
                add(_source_unit_identity(scope="request_spec", parameter_id="residue_normalized_coupling", field="frequency"), spec.frequency, "hertz")
            for branch_name, branch in (("branch_a", spec.branch_a), ("branch_b", spec.branch_b)):
                if isinstance(branch, DiagonalRootSpec):
                    add(_source_unit_identity(scope="request_spec", parameter_id="residue_normalized_coupling", field=f"{branch_name}:root_hint"), branch.root_hint, "hertz")
                else:
                    add(_source_unit_identity(scope="request_spec", parameter_id="residue_normalized_coupling", field=f"{branch_name}:anchor"), branch.anchor, "hertz")
        else:
            for index, variable in enumerate(spec.variables):
                parameter = variable.parameter
                definitions_id, identifier = _parameter_key(parameter)
                for role, bounds in (
                    ("model_default", variable.model_default_bounds),
                    ("consumer_override", variable.consumer_override_bounds),
                ):
                    if bounds is None:
                        continue
                    add(
                        _source_unit_identity(
                            scope="request_optimization_variable",
                            component_path=(definitions_id,),
                            parameter_id=identifier,
                            field=f"{index}:{role}:lower",
                        ),
                        bounds[0],
                        parameter.spec.si_unit,
                    )
                    add(
                        _source_unit_identity(
                            scope="request_optimization_variable",
                            component_path=(definitions_id,),
                            parameter_id=identifier,
                            field=f"{index}:{role}:upper",
                        ),
                        bounds[1],
                        parameter.spec.si_unit,
                    )
                if variable.scale is not None:
                    add(_source_unit_identity(scope="request_optimization_variable",
                        component_path=(definitions_id,), parameter_id=identifier,
                        field=f"{index}:domain:scale"), variable.scale, parameter.spec.si_unit)
            for index, objective in enumerate(spec.objectives):
                parameter_id = f"objective:{index}"
                selectors = _quantity_selectors(objective.quantity)
                selector = selectors[0]
                objective_unit = _selector_unit(selector)
                if objective_unit is None:
                    raise InvalidOptimizationSpec(
                        "optimization objective has no scalar quantity unit",
                        stage="spec_validation",
                    )
                add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field="target"), objective.target, objective_unit)
                add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field="weight"), objective.weight, "dimensionless")
                if objective.scale is not None:
                    add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field="scale"), objective.scale, objective_unit)
                for term_index, term in enumerate(selectors):
                    selected_spec = term.spec
                    prefix = f"selector:{term_index}"
                    if isinstance(selected_spec, (DiagonalRootSpec, OperatorElementRootSpec)):
                        add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:root_hint"), selected_spec.root_hint, "hertz")
                    elif isinstance(selected_spec, HybridizedPoleSpec):
                        add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:anchor"), selected_spec.anchor, "hertz")
                    elif isinstance(selected_spec, TransferZeroSpec):
                        add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:anchor"), selected_spec.anchor, "hertz")
                    elif isinstance(selected_spec, ResponseElementSpec):
                        add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:frequency"), selected_spec.frequency, "hertz")
                    elif isinstance(selected_spec, ResidueNormalizedCouplingSpec):
                        if not isinstance(selected_spec.frequency, str):
                            add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:frequency"), selected_spec.frequency, "hertz")
                        for branch_name, branch in (("branch_a", selected_spec.branch_a), ("branch_b", selected_spec.branch_b)):
                            field = "root_hint" if isinstance(branch, DiagonalRootSpec) else "anchor"
                            add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:{branch_name}:{field}"), getattr(branch, field), "hertz")
        return sorted(evidence, key=lambda item: str(item["identity"]))


def prepare_request(*, plan_sha256, operation, view, encoded_spec,
                    parameter_source, runtime_base, source_units,
                    backend, precision):
    """Encode validated request data once, without retaining facade state."""
    from .identity import _jax_runtime_identity
    from .prepared import PreparedAnalysis
    semantic = dict(runtime_base)
    if operation == "solve_direct":
        semantic["algorithm_id"] = "scnsim.direct_response.v1"
    elif operation == "solve_hb":
        semantic["algorithm_id"] = "scnsim.hb_response.josephsoncircuits.v1"
    elif operation == "evaluate_direct":
        semantic["algorithm_id"] = {
            "diagonal_root": "scnsim.diagonal_root.newton32.v2",
            "operator_element_root": "scnsim.operator_element_root.newton32.v1",
            "hybridized_pole": "scnsim.hybridized_pole.newton32.v1",
            "transfer_zero": "scnsim.transfer_zero.newton32.v4",
            "residue_normalized_coupling": "scnsim.residue_normalized_coupling.v2",
            "response_element": "scnsim.response_element.v1",
            "operator": "scnsim.direct_operator.v1",
        }[encoded_spec["type"]]
    elif operation == "optimize_direct":
        semantic["algorithm_id"] = (
            "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v9"
        )
    else:
        raise CompilerInvariantError(
            "operation is outside the runtime", stage="request_encode"
        )
    if backend == "jax":
        from .request import _JAX_ALGORITHMS
        semantic = _jax_runtime_identity(runtime_base, precision=precision)
        semantic["algorithm_id"] = _JAX_ALGORITHMS[
            encoded_spec["type"] if operation == "evaluate_direct" else operation
        ]
    return PreparedAnalysis.create(
        plan_sha256=plan_sha256,
        operation=operation,
        view=view,
        spec=encoded_spec,
        parameter_source=parameter_source,
        runtime_semantic=semantic,
        source_units=source_units,
    )

