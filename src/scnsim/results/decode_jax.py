"""Pure typed projection of independently verified JAX operation evidence.

The operation store owns hashes, commit/value verification and ordered records.
This decoder neither opens artifacts nor executes numerical work: actual arrays,
scalars, candidate ordinals and winner references feed the existing Result types.
Native Julia receipt decoding remains a separate unchanged verification boundary.
"""
from __future__ import annotations

import numpy as np

from .. import units
from ..authoring.identity import canonical_parameters_sha256
from ..benchmark.optimization import complex_value, numerical_error
from ..benchmark.models import NumericalFailure
from ..benchmark.prepared import array_from_record, decode_record, record_bytes
from ..canonical import canonical_json_bytes, float64_from_hex
from ..execution.prepared import _coordinate_binding_key, _encode_scalar_expression, _quantity_coordinates
from ..errors import EvidenceIntegrityError
from ..specs import QuantitySelector
from .base import LineDiscretization, MatrixFamilyResult, MatrixView, ParameterPointIdentity, ResultIdentity
from .derived import TraceResult
from .factory import _verified_result
from .matrix import DiagonalRootResult, DirectQuantityResult, DirectSolveResult, ScatteringMatrixResult
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


def _point_result(identity, record, request):
    if record.get("status") == "failure":
        raise _failure(record)
    kind = request["spec"]["type"]
    grid = _discretization(record.get("discretization"))
    presentation = _presentation(record, request)
    if kind == "diagonal_root":
        return _verified_result(
            DiagonalRootResult, identity=identity, discretization=grid,
            root=units.registry.Quantity(complex_value(record["root_omega_rad_s"]), "radian / second"),
            frequency=units.registry.Quantity(float64_from_hex(record["frequency_hz_f64"]), "hertz"),
            linewidth=units.registry.Quantity(float64_from_hex(record["linewidth_hz_f64"]), "hertz"),
            slope=units.registry.Quantity(complex_value(record["root_slope"]), "siemens"),
            _presentation=presentation,
        )
    if kind == "response_element":
        family = request["spec"]["family"]
        unit = {"S": "dimensionless", "Y": "siemens", "Z": "ohm"}[family]
        value = complex_value(record["response_value"])
        return _verified_result(
            DirectQuantityResult, identity=identity, discretization=grid, family=family,
            value=units.registry.Quantity(value, unit), magnitude=units.registry.Quantity(abs(value), unit),
            real=units.registry.Quantity(value.real, unit), imag=units.registry.Quantity(value.imag, unit),
            _presentation=presentation,
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
    if selector["type"] == "diagonal_root_projection":
        return "hertz", "inverse_time"
    return {"S": ("dimensionless", "dimensionless"), "Y": ("siemens", "conductance"),
            "Z": ("ohm", "resistance")}[selector["spec"]["family"]]


def _candidate(record, spec):
    """Retain the actual record and adapt names to existing ledger presentation."""
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
            terms.append(adapted)
        component["terms"] = terms
        components.append(component)
    outcome = {"status": "success" if record["failure"] is None else "failure",
               "objective_components": components}
    if record["failure"] is None:
        outcome["cost_f64"] = record["cost_f64"]
    else:
        outcome["failure"] = record["failure"]
    return dict(record, outcome=outcome)


def _optimization(decoder, identity, projection, request):
    baseline = projection["baseline"]
    records = projection["evaluations"]
    best_ordinal = projection["terminal"]["best_ordinal"]
    best = next(row for row in (baseline, *records) if row["evaluation_ordinal"] == best_ordinal)
    generations = {}
    for row in records:
        generations.setdefault(row["generation"], []).append(_candidate(row, request["spec"]))
    ledger = tuple({"generation": generation, "candidates": candidates}
                   for generation, candidates in generations.items())
    return _verified_result(
        OptimizationResult, identity=identity, discretization=_discretization(best.get("discretization")),
        best=_verified_result(OptimizationBest, parameters=decoder._decode_parameter_set(best["parameters"]),
                              cost=float64_from_hex(best["cost_f64"]),
                              discretization=_discretization(best.get("discretization"))),
        ledger=ledger, candidate_discretization=tuple(_discretization(row.get("discretization")) for row in records),
        _presentation={"initial_parameters": decoder._decode_parameter_set(baseline["parameters"]),
                       "initial_candidate": _candidate(baseline, request["spec"]),
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
    selector_kind = {"diagonal_root": "diagonal_root_projection", "response_element": "response_element_projection"}.get(spec["type"])
    projections = {"diagonal_root": ("frequency", "linewidth"), "response_element": ("magnitude", "real", "imag")}.get(spec["type"], ())
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
                         attempt_sha256, result_sha256, bound_spec):
    """Decode verified immutable references without a synthetic native receipt."""
    del bound_spec  # Exact encoded request owns selector and output identities.
    identity = _verified_result(ResultIdentity, plan_sha256=plan_sha256, request_sha256=request_sha256,
                                attempt_sha256=attempt_sha256, result_sha256=result_sha256)
    if request["operation"] == "optimize_direct":
        return _optimization(decoder, identity, projection, request)
    records = projection["evaluations"]
    if request["spec"]["type"] == "diagonal_root":
        records = [row for row in records if row.get("origin") == "requested_point"]
    if len(records) != projection["terminal"]["point_count"]:
        raise EvidenceIntegrityError("JAX terminal point count differs from its requested evaluations", stage="result_decode")
    if request["parameter_source"]["kind"] != "point":
        return _sweep(decoder, identity, records, request)
    return _point_result(identity, records[0], request)
