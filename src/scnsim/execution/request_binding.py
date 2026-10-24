"""Request provenance and View encoding from explicit immutable declarations.

No Run, backend, Workspace, mutable operation or result authority is retained.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np
from .. import units
from ..authoring import ElectricNodeRef, CoordinateRef, ParameterRef, ParameterSet, ParameterSpace
from ..authoring.physical_values import RLGC, RLGCParameterSpec
from ..canonical import canonical_json_bytes, complex_quantity_envelope, quantity_envelope, derived_quantity_envelope
from ..errors import CompilerInvariantError, InvalidOptimizationSpec
from .run_binding import _parameter_key, compatible_parameter
from ..specs import (
    DiagonalRootSpec,
    OperatorElementRootSpec,
    DirectSolveSpec,
    HybridizedPoleSpec,
    OperatorSpec,
    StaticCapacitanceSpec,
    StaticCapacitanceElementSpec,
    OptimizationSpec,
    SensitivitySpec,
    JVP,
    VJP,
    LBFGSBSpec,
    ProjectedLBFGSSpec,
    HybridSpec,
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
        spec: DirectSolveSpec | DiagonalRootSpec | OperatorElementRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | StaticCapacitanceSpec | StaticCapacitanceElementSpec | OptimizationSpec,
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
            for axis_index, ((parameter, values), axis_source_units) in enumerate(
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
                for value_index, (value, source_unit) in enumerate(zip(values, axis_source_units)):
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
        def add_search(declaration, *, scope, parameter_id, prefix=""):
            search = getattr(declaration, "search", None)
            if search is None:
                return
            for axis in ("real_frequency_interval", "imag_frequency_interval"):
                for ordinal, value in enumerate(getattr(search, axis) or ()):
                    add(_source_unit_identity(scope=scope, parameter_id=parameter_id,
                                              field=f"{prefix}search:{axis}:{ordinal}"), value, "hertz")
            for ordinal, value in enumerate(search.seeds or ()):
                add(_source_unit_identity(scope=scope, parameter_id=parameter_id,
                                          field=f"{prefix}search:seed:{ordinal}"), value, "hertz")

        add_search(spec, scope="request_root_search", parameter_id="root")
        if isinstance(spec, ResidueNormalizedCouplingSpec):
            for name, branch in (("branch_a", spec.branch_a), ("branch_b", spec.branch_b)):
                add_search(branch, scope="request_root_search", parameter_id=name)
        if isinstance(spec, SensitivitySpec):
            # Each leaf has its own View/spec declaration and source-unit domain.
            inherited = set(identities)
            for label, expression in spec.quantities.items():
                for leaf_ordinal, leaf in enumerate(_quantity_selectors(expression)):
                    for row in source_units(source_provenance, parameter_lookup, leaf.spec,
                                            parameters, parameter_space=parameter_space):
                        if row["identity"] in inherited:
                            continue
                        identity = canonical_json_bytes({"scope": "request_sensitivity_leaf",
                                                         "quantity_label": label,
                                                         "leaf_ordinal": leaf_ordinal,
                                                         "source_identity": row["identity"]}).decode("utf-8")
                        identities.add(identity)
                        evidence.append({**row, "identity": identity})

            def derived_rows(role, values, key_record):
                for key, value in values.items():
                    encoded = derived_quantity_envelope(value, registry=units.registry)
                    identity = canonical_json_bytes({"scope": "request_sensitivity", "role": role,
                                                     "coordinate": key_record(key)}).decode("utf-8")
                    identities.add(identity)
                    evidence.append({"identity": identity, "source_unit": str(value.units),
                                     "canonical_si_unit": encoded["si_unit"],
                                     "canonical_dimensionality": encoded["dimensionality"]})

            if isinstance(spec.derivative, JVP):
                derived_rows("direction", spec.derivative._source_direction, lambda key: key._key_record())
            elif isinstance(spec.derivative, VJP):
                derived_rows("weights", spec.derivative.weights, lambda key: key._canonical_record())
            if spec.scales is not None:
                derived_rows("input_scales", spec.scales.inputs, lambda key: key._key_record())
                derived_rows("output_scales", spec.scales.outputs, lambda key: key._canonical_record())
            return sorted(evidence, key=lambda item: str(item["identity"]))
        if isinstance(spec, DirectSolveSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="frequencies", field="value"), spec.frequencies, "hertz")
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
        elif isinstance(spec, (StaticCapacitanceSpec, StaticCapacitanceElementSpec)):
            pass  # Coordinate-only declaration introduces no frequency/source unit.
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
                    add_search(selected_spec, scope="request_optimization_objective", parameter_id=parameter_id, prefix=f"{prefix}:")
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
                            add_search(branch, scope="request_optimization_objective", parameter_id=parameter_id, prefix=f"{prefix}:{branch_name}:")
                            field = "root_hint" if isinstance(branch, DiagonalRootSpec) else "anchor"
                            add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:{branch_name}:{field}"), getattr(branch, field), "hertz")
        if isinstance(spec, OptimizationSpec):
            stages = spec.optimizer.stages if isinstance(spec.optimizer, HybridSpec) else ()
            controls = tuple((stage.id, stage.optimizer) for stage in stages) or (("main", spec.optimizer),)
            for stage_id, control in controls:
                if isinstance(control, (LBFGSBSpec, ProjectedLBFGSSpec)) and control.coordinate_scales is not None:
                    for parameter, value in control.coordinate_scales.items():
                        add(_source_unit_identity(scope="request_local_optimizer", component_path=(stage_id, parameter.definitions_id),
                                                  parameter_id=parameter.id, field="coordinate_scale"), value, parameter.spec.si_unit)
            meshes = [(("common",), spec.discretization)] + [(("stage", stage.id), stage.discretization) for stage in stages]
            for mesh_path, mesh in meshes:
                if mesh is not None and mesh.resolution is not None:
                    add(_source_unit_identity(scope="request_mesh", component_path=mesh_path,
                                              parameter_id="resolution", field="max_frequency"), mesh.resolution.max_frequency, "hertz")
            for stage in stages:
                if stage.seed is not None:
                    for parameter, value in stage.seed.values.items():
                        source_unit = stage.seed._source_units.get(parameter)
                        if isinstance(parameter.spec, RLGCParameterSpec):
                            field_units = {"resistance_per_length": "ohm / meter", "inductance_per_length": "henry / meter", "conductance_per_length": "siemens / meter", "capacitance_per_length": "farad / meter", "extraction_frequency": "hertz"}
                            for field, quantity in value._source_quantities.items():
                                add(_source_unit_identity(scope="request_stage_seed", component_path=(stage.id, parameter.definitions_id), parameter_id=parameter.id, field=field), quantity, field_units[field])
                        else:
                            add(_source_unit_identity(scope="request_stage_seed", component_path=(stage.id, parameter.definitions_id), parameter_id=parameter.id, field="value"), value if source_unit is None else value.to(source_unit), parameter.spec.si_unit)
        return sorted(evidence, key=lambda item: str(item["identity"]))


def prepare_request(*, plan_sha256, operation, view, encoded_spec,
                    parameter_source, runtime_base, source_units,
                    precision, parameter_authorizations=()):
    """Encode validated request data once, without retaining facade state."""
    from .identity import _jax_runtime_identity
    from .prepared import PreparedAnalysis
    from .request import _JAX_ALGORITHMS
    if operation not in {"solve_direct", "evaluate_direct", "optimize_direct"}:
        raise CompilerInvariantError(
            "operation is outside the runtime", stage="request_encode"
        )
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
        parameter_authorizations=parameter_authorizations,
    )

