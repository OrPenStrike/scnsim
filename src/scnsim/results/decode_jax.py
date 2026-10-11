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
from .lazy_optimization import LazyCandidateDiscretizationSequence, LazyGenerationSequence
from .matrix import (
    DiagonalRootResult,
    DirectQuantityResult,
    DirectSolveResult,
    OperatorElementRootResult,
    OperatorPointResult,
    OperatorResult,
    ScatteringMatrixResult,
)
from .optimization import OptimizationBest, OptimizationResult
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


def _coupling_branch_evidence(record, dependencies):
    branch_evidence = []
    for branch in record["evidence"]["branches"]:
        body = dependencies[branch["body_id"]]
        detail = {
            "role": branch["role"],
            "binding": branch["binding"],
            "root": _numeric_quantity(body, "root_omega_rad_s", "radian / second"),
            "slope": _numeric_quantity(body, "root_slope", "siemens"),
            "certificates": body["evidence"]["certificates"],
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
    if kind == "diagonal_root":
        root = _numeric_quantity(record, "root_omega_rad_s", "radian / second")
        frequency, linewidth = _root_frequency_linewidth(record, root.magnitude)
        return _verified_result(
            DiagonalRootResult, identity=identity, discretization=grid,
            root=root,
            frequency=units.registry.Quantity(frequency, "hertz"),
            linewidth=units.registry.Quantity(linewidth, "hertz"),
            slope=_numeric_quantity(record, "root_slope", "siemens"),
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
            OperatorResult, identity=identity, discretization=grid, points=points
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
    to the same candidate-level map, so the ledger does not duplicate bodies or
    turn those references into synthetic result identities.
    """
    dependencies = dict(record.get("dependencies", {}))
    for objective in record["objectives"]:
        for term in objective["terms"]:
            if term.get("status") != "success" or "body_id" not in term:
                continue
            body = dependencies[term["body_id"]]
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
    return dict(record, dependencies=dependencies, outcome=outcome)


def _optimization(decoder, identity, request, fixed_reader):
    if fixed_reader is None:
        raise EvidenceIntegrityError(
            "JAX Optimization decoding requires its fixed result reader",
            stage="result_decode",
        )
    selected = fixed_reader.selection
    comparison = fixed_reader.project("comparison")
    baseline_record = dict(comparison["baseline"])
    baseline_record.setdefault("evaluation_ordinal", 0)
    baseline_record.setdefault("generation", 0)
    initial_candidate = _candidate(baseline_record, request["spec"])
    best_ordinal = selected.best_locator["ordinal"]
    best_record = baseline_record if best_ordinal == 0 else dict(comparison["best"])
    best_candidate = initial_candidate if best_ordinal == 0 else _candidate(best_record, request["spec"])
    generation_count = selected.generation_count
    candidate_count = selected.candidate_count
    ledger = LazyGenerationSequence(
        fixed_reader,
        generation_count,
        lambda record: _candidate(record, request["spec"]),
    )
    candidate_discretization = LazyCandidateDiscretizationSequence(
        fixed_reader,
        candidate_count,
        lambda record: _discretization(record),
    )
    return _verified_result(
        OptimizationResult, identity=identity, discretization=_discretization(best_candidate.get("discretization")),
        best=_verified_result(OptimizationBest, parameters=decoder._decode_parameter_set(best_candidate["parameters"]),
                              cost=float64_from_hex(best_candidate["cost_f64"]),
                              discretization=_discretization(best_candidate.get("discretization"))),
        _fixed_reader=fixed_reader,
        ledger=ledger, candidate_discretization=candidate_discretization,
        _presentation={"initial_parameters": decoder._decode_parameter_set(baseline_record["parameters"]),
                       "initial_candidate": initial_candidate,
                       "best_candidate": best_candidate,
                       "objectives": tuple(request["spec"]["objectives"]),
                       "variables": tuple(request["spec"]["variables"]),
                       "best_evaluation_ordinal": best_ordinal},
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


def decode_jax_operation(decoder, *, projection, request, plan_sha256, request_sha256,
                         attempt_sha256, result_sha256, bound_spec, fixed_reader=None):
    """Decode verified immutable references without a synthetic native receipt."""
    del bound_spec  # Exact encoded request owns selector and output identities.
    identity = _verified_result(ResultIdentity, plan_sha256=plan_sha256, request_sha256=request_sha256,
                                attempt_sha256=attempt_sha256, result_sha256=result_sha256)
    if request["operation"] == "optimize_direct":
        return _optimization(decoder, identity, request, fixed_reader)
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
