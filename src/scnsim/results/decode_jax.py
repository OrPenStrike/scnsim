"""Pure typed projection of independently verified JAX operation evidence.

The operation store owns hashes, commit/value verification and ordered records.
This decoder neither opens artifacts nor executes numerical work: actual arrays,
scalars, candidate ordinals and winner references feed the existing Result types.
Native Julia receipt decoding remains a separate unchanged verification boundary.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .. import units
from ..authoring.identity import canonical_parameters_sha256
from ..numeric_encoding import array_from_record, complex_value, decode_record, record_bytes
from ..numerics.evidence import numerical_error
from ..numerics.models import NumericalFailure
from ..canonical import (
    canonical_json_bytes,
    complex_quantity_envelope,
    float64_from_hex,
    quantity_from_envelope,
)
from ..execution.prepared import _coordinate_binding_key, _encode_scalar_expression, _quantity_coordinates
from ..errors import EvidenceIntegrityError
from ..specs import QuantitySelector
from .base import LineDiscretization, MatrixFamilyResult, MatrixView, ParameterPointIdentity, ResultIdentity
from .derived import TraceResult
from .factory import _verified_result
from .matrix import (
    DiagonalRootResult,
    DirectQuantityResult,
    DirectSolveResult,
    OperatorElementRootResult,
    OperatorPointResult,
    OperatorResult,
    StaticCapacitanceResult,
    StaticCapacitanceElementResult,
    ScatteringMatrixResult,
)
from .lazy_direct import DirectTraceMapping
from .optimization_projection import decode_staged_optimization
from .sweep import _parameter_sweep_result, _point_accessor, _point_outcome


def _discretization(rows):
    if rows is None:
        return None
    # Compiler evidence uses the shared numeric record codec, rather than a
    # fabricated native quantity catalog. Convert its actual SI values once.
    return tuple(LineDiscretization(
        component_path=tuple(row["component_path"]), kind=row["kind"],
        length=units.registry.Quantity(row["length_m"], "meter"),
        n_sections=row["n_sections"], dx=units.registry.Quantity(row["dx_m"], "meter"),
        modal_velocities=tuple(units.registry.Quantity(value, "meter / second")
                               for value in row.get("modal_velocities_m_s", ())),
        hmax=None if "hmax_m" not in row else units.registry.Quantity(row["hmax_m"], "meter"),
        policy=row.get("policy"),
    ) for row in decode_record(rows))


def _presentation(record, request):
    return {"view": request["view"], "ref_lineage": record.get("lineage"), "spec": request["spec"]}


def _failure(record):
    failure = record["failure"]
    return numerical_error(NumericalFailure(failure["kind"], failure["stage"], failure["detail"],
                                           record_bytes(failure.get("evidence", {}))))


def _numeric_array(record, name):
    """Read the precision-preserving quantity codec or a retained legacy scalar."""

    value = record[name]
    if isinstance(value, Mapping) and "dtype" in value:
        return array_from_record(value)
    return np.asarray(complex_value(value))


def _numeric_quantity(record, name, unit):
    return units.registry.Quantity(_numeric_array(record, name), unit)


def _complex_angular_frequency(record, name):
    return complex_quantity_envelope(
        _numeric_quantity(record, name, "radian / second"),
        si_unit="radian / second",
        dimensionality="inverse_time",
        registry=units.registry,
    )


def _residue_coupling_evidence(term, dependencies):
    """Project canonical coupling and branch bodies into the public term schema."""
    coupling = dependencies[term["body_id"]]
    branches = {
        branch["role"]: dependencies[branch["body_id"]]
        for branch in coupling["evidence"]["branches"]
    }
    return {
        "branch_a_root": _complex_angular_frequency(branches["a"], "root_omega_rad_s"),
        "branch_b_root": _complex_angular_frequency(branches["b"], "root_omega_rad_s"),
        "evaluation_omega": _complex_angular_frequency(coupling, "evaluation_omega_rad_s"),
        "coupling": _complex_angular_frequency(coupling, "coupling_rad_s"),
    }


def _root_frequency_linewidth(record, root):
    if "frequency_hz_f64" in record and "linewidth_hz_f64" in record:
        return (
            float64_from_hex(record["frequency_hz_f64"]),
            float64_from_hex(record["linewidth_hz_f64"]),
        )
    # Host and readonly decoding share one public Hz projection; the canonical
    # root scalar above remains at the backend's recorded arithmetic precision.
    from ..execution.quantities import root_frequency_linewidth

    return root_frequency_linewidth(root)


def _quantity_evidence(record, request):
    evidence = dict(record["evidence"])
    spec = request["spec"]
    if spec["type"] == "operator_element_root":
        evidence["row"] = spec["row"]
        evidence["column"] = spec["column"]
    elif spec["type"] == "transfer_zero":
        evidence["family"] = spec["family"]
        evidence["input"] = spec["input_coordinate"]
        evidence["output"] = spec["output_coordinate"]
    return evidence


def _affine_evidence(record):
    rows = record["evidence"].get("affine_evidence")
    return {"affine_evidence": decode_record(rows)} if rows else {}


def _coupling_branch_evidence(record, dependencies):
    branch_evidence = []
    for branch in record["evidence"]["branches"]:
        body = dependencies[branch["body_id"]]
        detail = {
            "role": branch["role"],
            "binding": branch["binding"],
            "root": _numeric_quantity(body, "root_omega_rad_s", "radian / second"),
            "slope": _numeric_quantity(body, "root_slope", "siemens"),
            "certificates": body["evidence"],
        }
        if "null_vector" in body:
            detail["null_vector"] = _numeric_quantity(body, "null_vector", "dimensionless")
        residue_name = "residue_a" if branch["role"] == "a" else "residue_b"
        detail["residue"] = _numeric_quantity(record, residue_name, "ohm")
        branch_evidence.append(detail)
    return tuple(branch_evidence)


def _point_result(identity, record, request):
    if record.get("status") == "failure":
        raise _failure(record)
    kind = request["spec"]["type"]
    grid = _discretization(record.get("discretization"))
    presentation = _presentation(record, request)
    if kind in {"static_capacitance", "static_capacitance_element"}:
        evidence = _quantity_evidence(record, request)
        axes = {name: tuple(evidence[name]) for name in ("row_coordinates", "column_coordinates", "zero_charge")}
        value = _numeric_quantity(record, "capacitance_values", "farad")
        if kind == "static_capacitance":
            if np.asarray(value.magnitude).shape != (len(axes["row_coordinates"]), len(axes["column_coordinates"])):
                raise EvidenceIntegrityError("Static capacitance shape differs from its compiled coordinate axes", stage="result_decode")
            return _verified_result(StaticCapacitanceResult, identity=identity, discretization=grid,
                                    matrix=value, **axes, evidence=evidence)
        if np.asarray(value.magnitude).shape != ():
            raise EvidenceIntegrityError("Static capacitance element is not scalar", stage="result_decode")
        return _verified_result(StaticCapacitanceElementResult, identity=identity, discretization=grid,
                                value=value, row=request["spec"]["row"], column=request["spec"]["column"],
                                zero_charge=axes["zero_charge"], evidence=evidence)
    if kind == "diagonal_root":
        root = _numeric_quantity(record, "root_omega_rad_s", "radian / second")
        frequency, linewidth = _root_frequency_linewidth(record, root.magnitude)
        return _verified_result(
            DiagonalRootResult, identity=identity, discretization=grid,
            root=root,
            frequency=units.registry.Quantity(frequency, "hertz"),
            linewidth=units.registry.Quantity(linewidth, "hertz"),
            slope=_numeric_quantity(record, "root_slope", "siemens"),
            evidence=_affine_evidence(record),
            _presentation=presentation,
        )
    if kind == "response_element":
        family = request["spec"]["family"]
        unit = {"S": "dimensionless", "Y": "siemens", "Z": "ohm"}[family]
        value = _numeric_array(record, "response_value")
        return _verified_result(
            DirectQuantityResult, identity=identity, discretization=grid, family=family,
            value=units.registry.Quantity(value, unit), magnitude=units.registry.Quantity(np.abs(value), unit),
            real=units.registry.Quantity(value.real, unit), imag=units.registry.Quantity(value.imag, unit),
            evidence=_affine_evidence(record),
            _presentation=presentation,
        )
    if kind == "operator_element_root":
        root = _numeric_quantity(record, "root_omega_rad_s", "radian / second")
        frequency, _ = _root_frequency_linewidth(record, root.magnitude)
        return _verified_result(
            OperatorElementRootResult,
            identity=identity,
            discretization=grid,
            root=root,
            frequency=units.registry.Quantity(frequency, "hertz"),
            slope=_numeric_quantity(record, "root_slope", "siemens"),
            evidence=_quantity_evidence(record, request),
            _presentation=presentation,
        )
    if kind == "hybridized_pole":
        root = _numeric_quantity(record, "root_omega_rad_s", "radian / second")
        frequency, linewidth = _root_frequency_linewidth(record, root.magnitude)
        evidence = _quantity_evidence(record, request)
        evidence["null_vector"] = _numeric_quantity(record, "null_vector", "dimensionless")
        return _verified_result(
            DirectQuantityResult,
            identity=identity,
            discretization=grid,
            root=root,
            frequency=units.registry.Quantity(frequency, "hertz"),
            linewidth=units.registry.Quantity(linewidth, "hertz"),
            slope=_numeric_quantity(record, "root_slope", "siemens"),
            evidence=evidence,
            _presentation=presentation,
        )
    if kind == "transfer_zero":
        zero = _numeric_quantity(record, "root_omega_rad_s", "radian / second")
        frequency, _ = _root_frequency_linewidth(record, zero.magnitude)
        return _verified_result(
            DirectQuantityResult,
            identity=identity,
            discretization=grid,
            zero=zero,
            frequency=units.registry.Quantity(frequency, "hertz"),
            numerator_slope=_numeric_quantity(record, "numerator_slope", "dimensionless"),
            denominator=_numeric_quantity(record, "denominator", "dimensionless"),
            family=request["spec"]["family"],
            evidence=_quantity_evidence(record, request),
            _presentation=presentation,
        )
    if kind == "residue_normalized_coupling":
        coupling = _numeric_array(record, "coupling_rad_s")
        evidence = _quantity_evidence(record, request)
        evidence["branches"] = _coupling_branch_evidence(record, record["dependencies"])
        return _verified_result(
            DirectQuantityResult,
            identity=identity,
            discretization=grid,
            coupling=units.registry.Quantity(coupling, "radian / second"),
            value=units.registry.Quantity(coupling, "radian / second"),
            magnitude=units.registry.Quantity(np.abs(coupling), "radian / second"),
            real=units.registry.Quantity(coupling.real, "radian / second"),
            imag=units.registry.Quantity(coupling.imag, "radian / second"),
            branch_a_residue=_numeric_quantity(record, "residue_a", "ohm"),
            branch_b_residue=_numeric_quantity(record, "residue_b", "ohm"),
            evaluation_omega=_numeric_quantity(record, "evaluation_omega_rad_s", "radian / second"),
            evidence=evidence,
            _presentation=presentation,
        )
    if kind == "operator":
        values = array_from_record(record["operator_values"])
        coordinates = tuple(record["evidence"]["terminal_ids"])
        frequencies = tuple(
            quantity_from_envelope(value, registry=units.registry)
            for value in request["spec"]["frequencies"]
        )
        points = tuple(
            _verified_result(
                OperatorPointResult,
                frequency=frequency,
                matrix=units.registry.Quantity(matrix, "siemens / second"),
                coordinates=coordinates,
            )
            for frequency, matrix in zip(frequencies, values, strict=True)
        )
        return _verified_result(
            OperatorResult, identity=identity, discretization=grid, points=points,
            evidence=_affine_evidence(record)
        )
    if kind != "direct_solve":
        raise EvidenceIntegrityError("JAX result has an unsupported quantity family", stage="result_decode")
    frequencies = units.registry.Quantity(array_from_record(record["frequencies_hz"]), "hertz")
    coordinates = tuple(record["terminal_ids"])
    channels = tuple((coordinate, ()) for coordinate in coordinates)
    loads = record["evidence"]["axes"]["probe_loads"]
    matrices = {name: array_from_record(record[name]) for name in ("S", "Y", "Z")}

    def family_result(name, unit, cls):
        view = _verified_result(MatrixView, matrix=units.registry.Quantity(matrices[name], unit),
                                frequencies=frequencies, coordinates=coordinates,
                                input_channels=channels, output_channels=channels, probe_loads=loads)
        return _verified_result(cls, view=view, _parent_identity=identity,
                                _presentation=dict(presentation, family=name))

    traces = {}
    for declaration in request["spec"]["traces"]:
        output = coordinates.index(declaration["output_port"])
        input_ = coordinates.index(declaration["input_port"])
        traces[declaration["id"]] = _verified_result(
            TraceResult, frequencies=frequencies,
            value=units.registry.Quantity(matrices["S"][:, output, input_], "dimensionless"),
            _parent_identity=identity,
            _presentation=dict(presentation, id=declaration["id"], family="S",
                               input_channel={"coordinate": coordinates[input_], "mode": []},
                               output_channel={"coordinate": coordinates[output], "mode": []}),
        )
    return _verified_result(
        DirectSolveResult, identity=identity, discretization=grid, frequencies=frequencies,
        s=family_result("S", "dimensionless", ScatteringMatrixResult),
        y=family_result("Y", "siemens", MatrixFamilyResult),
        z=family_result("Z", "ohm", MatrixFamilyResult), traces=traces,
        evidence=_affine_evidence(record),
    )


def _direct_solve(identity, request, fixed_reader):
    """Construct a Direct Result from compact axes and a fixed body reader."""
    if fixed_reader is None:
        raise EvidenceIntegrityError(
            "Direct Solve decoding requires its exact fixed result reader",
            stage="result_decode",
        )
    selection = getattr(fixed_reader, "selection", None)
    metadata = getattr(fixed_reader, "metadata", None)
    if (
        selection is None
        or not isinstance(metadata, Mapping)
        or getattr(selection, "plan_sha256", None) != identity.plan_sha256
        or getattr(selection, "request_sha256", None) != identity.request_sha256
        or getattr(selection, "outcome", None) != "success"
        or metadata.get("operation") != "solve_direct"
    ):
        raise EvidenceIntegrityError(
            "Direct Solve reader is not bound to this exact successful request",
            stage="result_decode",
        )
    if (
        request.get("operation") != "solve_direct"
        or request.get("parameter_source", {}).get("kind") != "point"
    ):
        raise EvidenceIntegrityError(
            "Direct Solve request does not have its declared point source",
            stage="result_decode",
        )
    frequencies = np.asarray(metadata["frequencies_hz"], dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size != metadata["frequency_count"]:
        raise EvidenceIntegrityError(
            "Direct Solve frequency axis differs from its fixed metadata",
            stage="result_decode",
        )
    coordinates = tuple(metadata["terminal_ids"])
    channels = tuple(metadata["input_channels"])
    output_channels = tuple(metadata["output_channels"])
    units_by_family = metadata["family_units"]
    if not isinstance(units_by_family, Mapping):
        raise EvidenceIntegrityError(
            "Direct Solve family units are missing from fixed metadata",
            stage="result_decode",
        )
    identity_metadata = (
        metadata.get("workspace_instance_id"),
        metadata.get("plan_sha256"),
        metadata.get("request_sha256"),
        metadata.get("task_id"),
        metadata.get("attempt_id"),
    )
    selected_identity = (
        getattr(selection, "workspace_instance_id", None),
        getattr(selection, "plan_sha256", None),
        getattr(selection, "request_sha256", None),
        getattr(selection, "task_id", None),
        getattr(selection, "attempt_id", None),
    )
    if identity_metadata != selected_identity:
        raise EvidenceIntegrityError(
            "Direct Solve fixed metadata differs from its selected completion",
            stage="result_decode",
        )
    if len(channels) != len(coordinates) or len(output_channels) != len(coordinates):
        raise EvidenceIntegrityError(
            "Direct Solve channels differ from its fixed terminal axis",
            stage="result_decode",
        )
    presentation = {
        "view": metadata["view"],
        "ref_lineage": metadata["lineage"],
        "spec": request["spec"],
    }
    frequency_quantity = units.registry.Quantity(frequencies, "hertz")

    def family_result(name, result_type):
        unit = units_by_family[name]
        view = _verified_result(
            MatrixView,
            matrix=None,
            frequencies=frequency_quantity,
            coordinates=coordinates,
            input_channels=channels,
            output_channels=output_channels,
            probe_loads=metadata["probe_loads"],
            _fixed_reader=fixed_reader,
            _matrix_family=name,
            _matrix_unit=unit,
        )
        return _verified_result(
            result_type,
            view=view,
            _parent_identity=identity,
            _presentation=dict(presentation, family=name),
        )

    traces = DirectTraceMapping(
        reader=fixed_reader,
        channels=channels,
        declarations=tuple(request["spec"]["traces"]),
        frequencies=frequencies,
        identity=identity,
        presentation=presentation,
    )
    return _verified_result(
        DirectSolveResult,
        identity=identity,
        discretization=_discretization(metadata["discretization"]),
        frequencies=frequency_quantity,
        s=family_result("S", ScatteringMatrixResult),
        y=family_result("Y", MatrixFamilyResult),
        z=family_result("Z", MatrixFamilyResult),
        traces=traces,
        evidence=({"affine_evidence": decode_record(metadata["affine_evidence"])}
                  if metadata.get("affine_evidence") else {}),
    )


def _term_unit(selector):
    selector_type = selector["type"]
    if selector_type in {
        "diagonal_root_projection",
        "operator_element_root_projection",
        "hybridized_pole_projection",
        "transfer_zero_projection",
    }:
        return "hertz", "inverse_time"
    if selector_type == "residue_coupling_projection":
        return "radian / second", "inverse_time"
    return {"S": ("dimensionless", "dimensionless"), "Y": ("siemens", "conductance"),
            "Z": ("ohm", "resistance")}[selector["spec"]["family"]]


def _candidate_dependencies(record):
    """Keep exact quantity bodies addressable from each public ledger row.

    Successful objective terms point at canonical quantity bodies by content
    id. Residue-coupling bodies in turn retain their ordered branch references
    to the same candidate-level map. Public term evidence is reconstructed
    from these authoritative bodies instead of being duplicated in the ledger.
    """
    dependencies = dict(record.get("dependencies", {}))
    for objective in record["objectives"]:
        for term in objective["terms"]:
            if term.get("status") != "success":
                continue
            body_id = term.get("body_id")
            if not isinstance(body_id, str) or body_id not in dependencies:
                raise EvidenceIntegrityError(
                    "Successful optimization term references an absent quantity body",
                    stage="result_decode",
                    evidence={"body_id": body_id},
                )
            body = dependencies[body_id]
            if not isinstance(body, Mapping) or not isinstance(body.get("evidence"), Mapping):
                raise EvidenceIntegrityError(
                    "Successful optimization term quantity body has no canonical evidence",
                    stage="result_decode",
                    evidence={"body_id": body_id},
                )
            for branch in body.get("evidence", {}).get("branches", ()):
                dependencies[branch["body_id"]]
    return dependencies


def _candidate(record, spec):
    """Retain the actual record and adapt names to existing ledger presentation."""
    dependencies = _candidate_dependencies(record)
    components = []
    for objective, definition in zip(record["objectives"], spec["objectives"], strict=True):
        component = dict(objective, objective_id=objective["id"], quantity=definition["quantity"])
        if "value_f64" in objective:
            component["value"] = dict(definition["target"], si_value_f64=objective["value_f64"])
        if "cost_f64" in objective:
            component["weighted_cost_f64"] = objective["cost_f64"]
        terms = []
        for term in objective["terms"]:
            adapted = dict(term)
            if term.get("status") == "success" and "body_id" in term:
                adapted["evidence"] = dependencies[term["body_id"]]["evidence"]
            if "value_f64" in term:
                unit, dimensionality = _term_unit(term["selector"])
                adapted["value"] = {"type": "quantity_f64", "si_unit": unit,
                                    "dimensionality": dimensionality, "si_value_f64": term["value_f64"]}
            if "lineage" in term:
                adapted["ref_lineage"] = term["lineage"]
            if "failure" in term:
                adapted["failure"] = dict(term["failure"], message=term["failure"]["detail"])
            if (term.get("status") == "success" and "body_id" in term
                    and term["selector"]["type"] == "residue_coupling_projection"):
                adapted["residue_coupling_evidence"] = _residue_coupling_evidence(term, dependencies)
            terms.append(adapted)
        component["terms"] = terms
        components.append(component)
    outcome = {"status": "success" if record["failure"] is None else "failure",
               "objective_components": components}
    if record["failure"] is None:
        outcome["cost_f64"] = record["cost_f64"]
    else:
        outcome["failure"] = record["failure"]
    # Approval belongs to the request/check, not the candidate a caller reuses.
    # Keep the verified stored body intact for exact cache/resume ownership.
    parameters = dict(record["parameters"], allow_extrapolation=[])
    return dict(record, parameters=parameters, dependencies=dependencies, outcome=outcome)


def _optimization(decoder, identity, request, fixed_reader):
    if fixed_reader is None:
        raise EvidenceIntegrityError(
            "JAX Optimization decoding requires its fixed result reader",
            stage="result_decode",
        )
    return decode_staged_optimization(
        decoder,
        identity,
        fixed_reader,
        decode_discretization=_discretization,
        adapt_candidate=lambda record: _candidate(record, request["spec"]),
    )


def _sweep(decoder, identity, records, request):
    source = request["parameter_source"]
    kind = source["kind"]
    shape = tuple(source["shape"]) if kind == "grid" else ()
    axes = tuple(decoder._decode_parameter_ref(axis["parameter"]) for axis in source["axes"]) if kind == "grid" else ()
    outcomes = []
    for row in records:
        ordinal = row["source_index"]
        index = tuple(int(i) for i in np.unravel_index(ordinal, shape)) if kind == "grid" else ordinal
        parameters = decoder._decode_parameter_set(row["parameters"])
        point_identity = _verified_result(ParameterPointIdentity, batch=identity, source_index=index,
                                          parameters_sha256=canonical_parameters_sha256(row["parameters"]))
        failed = row.get("status") == "failure"
        outcomes.append(_point_outcome(parameters=parameters, source_index=index, identity=point_identity,
                                       result=None if failed else _point_result(point_identity, row, request),
                                       failure=_failure(row) if failed else None))
    spec = request["spec"]
    selector_kind = {
        "diagonal_root": "diagonal_root_projection",
        "operator_element_root": "operator_element_root_projection",
        "hybridized_pole": "hybridized_pole_projection",
        "transfer_zero": "transfer_zero_projection",
        "residue_normalized_coupling": "residue_coupling_projection",
        "response_element": "response_element_projection",
    }.get(spec["type"])
    projections = {
        "diagonal_root": ("frequency", "linewidth"),
        "operator_element_root": ("frequency",),
        "hybridized_pole": ("frequency", "linewidth"),
        "transfer_zero": ("frequency",),
        "residue_normalized_coupling": ("real", "imag", "magnitude"),
        "response_element": ("magnitude", "real", "imag"),
    }.get(spec["type"], ())
    allowed = tuple(canonical_json_bytes({"type": selector_kind, "spec": spec, "projection": p}) for p in projections)
    derived = {coordinate for transform in request["view"].get("transforms", ())
               for coordinate in transform.get("output_coordinates", ())}

    def selector_encoder(value):
        if not isinstance(value, QuantitySelector):
            raise TypeError("quantity must be a QuantitySelector")
        return canonical_json_bytes(_encode_scalar_expression(
            value, coordinate_bindings={_coordinate_binding_key(coordinate):
                coordinate if isinstance(coordinate, str) and coordinate in derived else decoder._coordinate_id(coordinate)
                for coordinate in _quantity_coordinates(value.spec)}))

    return _parameter_sweep_result(identity=identity,
        points=_point_accessor(outcomes.__getitem__, len(outcomes), kind, shape, axes),
        selector_encoder=selector_encoder, allowed_selectors=allowed)


def _merge_source_parameters(base, overlay):
    """Apply one sparse listed-point overlay in the declared binding order."""
    def key(reference):
        return reference["definitions_id"], reference["parameter_id"]

    bindings = {key(row["parameter"]): dict(row) for row in base["bindings"]}
    order = [key(row["parameter"]) for row in base["bindings"]]
    for row in overlay["bindings"]:
        parameter_key = key(row["parameter"])
        if parameter_key not in bindings:
            raise EvidenceIntegrityError(
                "Listed sweep point references a parameter outside its baseline",
                stage="result_decode",
            )
        bindings[parameter_key] = dict(row)
    authorizations = {
        key(reference): dict(reference)
        for reference in (*base["allow_extrapolation"], *overlay["allow_extrapolation"])
    }
    return {
        "type": "parameter_set_v2",
        "bindings": [bindings[parameter_key] for parameter_key in order],
        "allow_extrapolation": [authorizations[parameter_key] for parameter_key in sorted(authorizations)],
    }


def _declared_sweep_point(source, ordinal):
    """Project one exact source position without consulting a stored point body."""
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise IndexError(ordinal)
    if source["kind"] == "points":
        if ordinal >= len(source["points"]):
            raise IndexError(ordinal)
        return ordinal, _merge_source_parameters(
            source["baseline_parameters"], source["points"][ordinal]
        )
    shape = tuple(source["shape"])
    count = 1
    for size in shape:
        count *= size
    if ordinal >= count:
        raise IndexError(ordinal)
    remainder = ordinal
    indices = [0] * len(shape)
    for axis_index in range(len(shape) - 1, -1, -1):
        size = shape[axis_index]
        indices[axis_index] = remainder % size
        remainder //= size
    overlay = {
        "type": "parameter_set_v2",
        "bindings": [
            {"parameter": axis["parameter"], "value": axis["values"][indices[index]]}
            for index, axis in enumerate(source["axes"])
        ],
        "allow_extrapolation": [],
    }
    return tuple(indices), _merge_source_parameters(source["base_parameters"], overlay)


def _fixed_sweep(decoder, identity, request, projection, fixed_reader):
    """Construct a lazy batch over one exact sealed Sweep selection."""
    if fixed_reader is None:
        raise EvidenceIntegrityError(
            "JAX Sweep decoding requires its exact fixed result reader",
            stage="result_decode",
        )
    selection = getattr(fixed_reader, "selection", None)
    metadata = getattr(fixed_reader, "metadata", None)
    source = request.get("parameter_source")
    operation = request.get("operation")
    spec = request.get("spec")
    direct_points = (
        operation == "solve_direct"
        and isinstance(source, Mapping)
        and source.get("kind") in {"grid", "points"}
        and isinstance(spec, Mapping)
        and spec.get("type") == "direct_solve"
    )
    evaluation_sweep = (
        operation == "evaluate_direct"
        and isinstance(source, Mapping)
        and source.get("kind") in {"grid", "points"}
    )
    if (
        selection is None
        or not isinstance(metadata, Mapping)
        or not isinstance(source, Mapping)
        or source.get("kind") not in {"grid", "points"}
        or not (direct_points or evaluation_sweep)
        or getattr(selection, "outcome", None) != "success"
        or metadata.get("operation") != operation
        or metadata.get("source_kind") != source.get("kind")
        or getattr(selection, "plan_sha256", None) != identity.plan_sha256
        or getattr(selection, "request_sha256", None) != identity.request_sha256
        or metadata.get("plan_sha256") != identity.plan_sha256
        or metadata.get("request_sha256") != identity.request_sha256
        or metadata.get("task_id") != getattr(selection, "task_id", None)
        or metadata.get("attempt_id") != getattr(selection, "attempt_id", None)
    ):
        raise EvidenceIntegrityError(
            "Sweep reader is not bound to this exact successful request",
            stage="result_decode",
        )
    terminal = projection.get("terminal")
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("point_count") != selection.point_count
        or metadata.get("point_count") != selection.point_count
    ):
        raise EvidenceIntegrityError(
            "Sweep terminal point count differs from its fixed selection",
            stage="result_decode",
        )

    kind = source["kind"]
    shape = tuple(source["shape"]) if kind == "grid" else ()
    point_request = {"spec": request["spec"], "view": request["view"]}
    axes = (
        tuple(decoder._decode_parameter_ref(axis["parameter"]) for axis in source["axes"])
        if kind == "grid"
        else ()
    )

    def source_point(ordinal):
        source_index, parameter_record = _declared_sweep_point(source, ordinal)
        return decoder._decode_parameter_set(parameter_record), source_index

    def plot_rows(ordinals):
        # The status table is a compact index projection. In particular, do
        # not hydrate each occurrence body merely to render its status.
        summary = fixed_reader.project_axis()
        occurrences = summary.get("occurrences") if isinstance(summary, Mapping) else None
        if (
            not isinstance(occurrences, (list, tuple))
            or len(occurrences) != selection.point_count
            or summary.get("point_count") != selection.point_count
        ):
            raise EvidenceIntegrityError(
                "Sweep status projection differs from its fixed selection",
                stage="result_decode",
            )
        selected = range(selection.point_count) if ordinals is None else ordinals
        for ordinal in selected:
            occurrence = occurrences[ordinal]
            if (
                not isinstance(occurrence, Mapping)
                or occurrence.get("ordinal") != ordinal
                or occurrence.get("status") not in {"success", "failure"}
                or not isinstance(occurrence.get("parameters_sha256"), str)
            ):
                raise EvidenceIntegrityError(
                    "Sweep status projection contains a malformed occurrence",
                    stage="result_decode",
                    evidence={"ordinal": ordinal},
                )
            if occurrence["status"] == "failure":
                failure = occurrence.get("failure")
                if (
                    not isinstance(failure, Mapping)
                    or not all(
                        isinstance(failure.get(name), str) and failure[name]
                        for name in ("kind", "stage", "message")
                    )
                ):
                    raise EvidenceIntegrityError(
                        "Sweep failure summary is malformed",
                        stage="result_decode",
                        evidence={"ordinal": ordinal},
                    )
            elif "failure" in occurrence:
                raise EvidenceIntegrityError(
                    "Successful Sweep summary unexpectedly contains a failure",
                    stage="result_decode",
                    evidence={"ordinal": ordinal},
                )
            _, parameter_record = _declared_sweep_point(source, ordinal)
            binding_records = sorted(
                parameter_record["bindings"],
                key=lambda binding: (
                    binding["parameter"]["definitions_id"],
                    binding["parameter"]["parameter_id"],
                ),
            )
            yield {
                "source_index": occurrence["source_index"],
                "status": occurrence["status"],
                "parameters_sha256": occurrence["parameters_sha256"],
                "parameter_values": tuple(
                    (binding["parameter"], binding["value"])
                    for binding in binding_records
                ),
                "failure": occurrence.get("failure"),
            }

    def load_point(ordinal):
        row = fixed_reader.read_point(ordinal)
        raw_source_index = row["source_index"]
        source_index = (
            tuple(raw_source_index) if isinstance(raw_source_index, (list, tuple))
            else raw_source_index
        )
        parameters_record = row["parameters"]
        parameters_sha256 = row["parameters_sha256"]
        parameters = decoder._decode_parameter_set(parameters_record)
        point_identity = _verified_result(
            ParameterPointIdentity,
            batch=identity,
            source_index=source_index,
            parameters_sha256=parameters_sha256,
        )
        if row.get("status") == "failure":
            return _point_outcome(
                parameters=parameters,
                source_index=source_index,
                identity=point_identity,
                result=None,
                failure=_failure(row),
            )
        if row.get("status") != "success":
            raise EvidenceIntegrityError(
                "Sweep point status is malformed",
                stage="result_decode",
                evidence={"ordinal": ordinal},
            )
        return _point_outcome(
            parameters=parameters,
            source_index=source_index,
            identity=point_identity,
            result=lambda: _point_result(point_identity, row, point_request),
            failure=None,
        )

    request_spec = request.get("spec")
    if not isinstance(request_spec, Mapping):
        raise EvidenceIntegrityError("Sweep Spec is malformed", stage="result_decode")
    selector_kind = {
        "diagonal_root": "diagonal_root_projection",
        "operator_element_root": "operator_element_root_projection",
        "hybridized_pole": "hybridized_pole_projection",
        "transfer_zero": "transfer_zero_projection",
        "residue_normalized_coupling": "residue_coupling_projection",
        "response_element": "response_element_projection",
    }.get(request_spec.get("type"))
    projections = {
        "diagonal_root": ("frequency", "linewidth"),
        "operator_element_root": ("frequency",),
        "hybridized_pole": ("frequency", "linewidth"),
        "transfer_zero": ("frequency",),
        "residue_normalized_coupling": ("real", "imag", "magnitude"),
        "response_element": ("magnitude", "real", "imag"),
    }.get(request_spec.get("type"), ())
    allowed = tuple(
        canonical_json_bytes({"type": selector_kind, "spec": request_spec, "projection": name})
        for name in projections
    )
    derived = {
        coordinate
        for transform in point_request["view"].get("transforms", ())
        if isinstance(transform, Mapping)
        for coordinate in transform.get("output_coordinates", ())
        if isinstance(coordinate, str)
    }

    def selector_encoder(value):
        if not isinstance(value, QuantitySelector):
            raise TypeError("quantity must be a QuantitySelector")
        return canonical_json_bytes(_encode_scalar_expression(
            value,
            coordinate_bindings={
                _coordinate_binding_key(coordinate): (
                    coordinate
                    if isinstance(coordinate, str) and coordinate in derived
                    else decoder._coordinate_id(coordinate)
                )
                for coordinate in _quantity_coordinates(value.spec)
            },
        ))

    points = _point_accessor(
        load_point,
        selection.point_count,
        kind,
        shape,
        axes,
        source_loader=source_point,
        plot_loader=plot_rows,
    )
    return _parameter_sweep_result(
        identity=identity,
        points=points,
        selector_encoder=selector_encoder,
        allowed_selectors=allowed,
    )


def decode_jax_operation(decoder, *, projection, request, plan_sha256, request_sha256,
                         attempt_sha256, result_sha256, bound_spec, fixed_reader=None):
    """Decode verified immutable references without a synthetic native receipt."""
    del bound_spec  # Exact encoded request owns selector and output identities.
    identity = _verified_result(ResultIdentity, plan_sha256=plan_sha256, request_sha256=request_sha256,
                                attempt_sha256=attempt_sha256, result_sha256=result_sha256)
    if request["operation"] == "sensitivity":
        if fixed_reader is None:
            raise EvidenceIntegrityError(
                "Sensitivity decoding requires its exact fixed terminal reader",
                stage="result_decode",
            )
        return decoder._decode_sensitivity(
            identity,
            request,
            projection,
            fixed_reader=fixed_reader,
        )
    if request["operation"] == "optimize_direct":
        return _optimization(decoder, identity, request, fixed_reader)
    if (request["operation"] == "solve_direct"
            and request["parameter_source"]["kind"] in {"grid", "points"}):
        return _fixed_sweep(decoder, identity, request, projection, fixed_reader)
    if request["operation"] == "solve_direct":
        return _direct_solve(identity, request, fixed_reader)
    if (request["operation"] == "evaluate_direct"
            and request["parameter_source"]["kind"] in {"grid", "points"}):
        return _fixed_sweep(decoder, identity, request, projection, fixed_reader)
    records = projection["evaluations"]
    if request["spec"]["type"] in {
        "diagonal_root",
        "operator_element_root",
        "hybridized_pole",
        "transfer_zero",
        "residue_normalized_coupling",
    }:
        records = [row for row in records if row.get("origin") == "requested_point"]
    if len(records) != projection["terminal"]["point_count"]:
        raise EvidenceIntegrityError("JAX terminal point count differs from its requested evaluations", stage="result_decode")
    if request["parameter_source"]["kind"] != "point":
        return _sweep(decoder, identity, records, request)
    return _point_result(identity, records[0], request)
