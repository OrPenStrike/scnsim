"""Request, operation-specification, and lineage verification."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from itertools import product

from ...canonical import canonical_json_bytes as _canonical_bytes, sha256_hex as _sha256
from ...errors import UnsupportedEvidenceVersionError
from .common import (
    _IDENTIFIER,
    _SHA256,
    _f64_value,
    _finite_f64,
    _identifiers,
    _integrity,
    _parameter_key_integrity,
    _valid_sha,
    _verify_bounds,
    _verify_branch_refs,
    _verify_parameter_set_document,
    _verify_parameter_value,
    _verify_quantity_compatible,
    _verify_quantity_role,
)

_NATIVE_OPTIMIZATION_ALGORITHM_ID = (
    "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v9"
)
_RUNTIME_SEMANTIC_FIELDS = frozenset({
    "algorithm_id",
    "python_source_sha256",
    "julia_source_sha256",
    "julia_version",
    "project_sha256",
    "manifest_sha256",
})


def _is_current_native_optimization_request(
    request: Mapping[str, object],
) -> bool:
    """Recognize the exact current Julia Optimization request identity."""
    spec = request.get("spec")
    runtime = request.get("runtime_semantic")
    return (
        request.get("operation") == "optimize_direct"
        and isinstance(spec, Mapping)
        and spec.get("type") == "optimization"
        and isinstance(runtime, Mapping)
        and set(runtime) == _RUNTIME_SEMANTIC_FIELDS
        and runtime.get("algorithm_id") == _NATIVE_OPTIMIZATION_ALGORITHM_ID
    )


def _verify_request_document(
    request: Mapping[str, object],
    plan_sha256: str,
    plan: Mapping[str, object],
) -> None:
    operation = request.get("operation")
    spec = request.get("spec")
    runtime = request.get("runtime_semantic")
    algorithms = {
        "solve_direct": {"direct_solve": "scnsim.direct_response.v1"},
        "solve_hb": {"hb_solve": "scnsim.hb_response.josephsoncircuits.v1"},
        "evaluate_direct": {
            "diagonal_root": "scnsim.diagonal_root.newton32.v2",
            "operator_element_root": "scnsim.operator_element_root.newton32.v1",
            "hybridized_pole": "scnsim.hybridized_pole.newton32.v1",
            "transfer_zero": "scnsim.transfer_zero.newton32.v4",
            "residue_normalized_coupling": "scnsim.residue_normalized_coupling.v2",
            "response_element": "scnsim.response_element.v1",
            "operator": "scnsim.direct_operator.v1",
        },
        "optimize_direct": {"optimization": _NATIVE_OPTIMIZATION_ALGORITHM_ID},
    }
    expected_algorithm = algorithms.get(operation, {}).get(spec.get("type") if isinstance(spec, dict) else None)
    if (
        set(request) != {"schema", "schema_version", "plan_sha256", "operation", "view", "spec", "parameter_source", "runtime_semantic"}
        or request.get("schema") != "scnsim.request"
        or request.get("schema_version") != 2
        or request.get("plan_sha256") != plan_sha256
        or expected_algorithm is None
        or any(not isinstance(request.get(field), dict) for field in ("view", "spec", "parameter_source", "runtime_semantic"))
    ):
        raise _integrity("Stored request envelope is open or inconsistent.")
    if set(runtime) != _RUNTIME_SEMANTIC_FIELDS or not isinstance(runtime.get("julia_version"), str) or not runtime["julia_version"]:
        raise _integrity("Stored runtime semantic identity is open or malformed.")
    for field in ("python_source_sha256", "julia_source_sha256", "project_sha256", "manifest_sha256"):
        _valid_sha(runtime.get(field))
    algorithm_id = runtime.get("algorithm_id")
    historical_optimization_algorithms = {
        "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v3",
        "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v4",
        "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v5",
        "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v6",
        "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v7",
        "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v8",
    }
    if operation == "optimize_direct" and algorithm_id in historical_optimization_algorithms:
        raise UnsupportedEvidenceVersionError(
            "Stored optimization evidence requires its original source-bound runtime.",
            stage="workspace",
            evidence={
                "operation": "optimize_direct",
                "expected_algorithm_id": expected_algorithm,
                "observed_algorithm_id": algorithm_id,
            },
        )
    if operation == "evaluate_direct" and isinstance(spec, dict) and spec.get("type") == "diagonal_root" and algorithm_id == "scnsim.diagonal_root.newton32.v1":
        raise UnsupportedEvidenceVersionError(
            "Stored diagonal-root evidence requires its original source-bound runtime.",
            stage="workspace",
            evidence={
                "operation": "evaluate_direct",
                "expected_algorithm_id": expected_algorithm,
                "observed_algorithm_id": algorithm_id,
            },
        )
    if algorithm_id != expected_algorithm:
        raise _integrity("Stored request runtime identity is inconsistent.")
    _verify_parameter_source(request["parameter_source"], plan)
    terminal, port_realizable = _verify_view_declaration(request["view"], plan)
    if operation == "solve_direct":
        _verify_v1_direct_spec(spec, terminal, port_realizable)
    elif operation == "solve_hb":
        logical_ports = [port["id"] for port in plan["connectivity"]["ports"]]
        _verify_v1_hb_spec(spec, terminal, port_realizable, logical_ports)
    elif operation == "evaluate_direct":
        _verify_v1_evaluation_spec(spec, terminal, port_realizable)
    else:
        _verify_v1_optimization_spec(spec, plan, request["parameter_source"].get("parameters"))
        leaves = _optimization_selector_leaves(spec)
        if not leaves or request["view"] != leaves[0].get("view"):
            raise _integrity("Optimization primary View is not its first normalized selector View.")
        parameters = request["parameter_source"].get("parameters")
        if not isinstance(parameters, Mapping):
            raise _integrity("Optimization request requires one complete parameter point.")
        active = {_parameter_key_integrity(item["parameter"]) for item in spec["variables"]}
        declared = {_parameter_key_integrity(item) for item in spec["allow_extrapolation"]}
        if declared != ({_parameter_key_integrity(item) for item in parameters["allow_extrapolation"]} & active):
            raise _integrity("Optimization request has inconsistent active extrapolation authorities.")

def _verify_parameter_source(source: object, plan: Mapping[str, object]) -> None:
    if not isinstance(source, dict):
        raise _integrity("Parameter source is malformed.")
    try:
        from ...execution.request import canonical_parameter_source

        if canonical_parameter_source(source) != source:
            raise ValueError("noncanonical parameter source")
    except Exception as error:
        raise _integrity("Parameter source is open or noncanonical.") from error
    definitions = plan.get("parameter_closure", {}).get("definitions")
    if not isinstance(definitions, list):
        raise _integrity("Plan parameter closure is malformed.")
    expected = {
        _parameter_key_integrity({
            "definitions_id": item.get("definitions_id"),
            "parameter_id": item.get("parameter_id"),
        })
        for item in definitions
        if isinstance(item, Mapping)
    }
    if len(expected) != len(definitions):
        raise _integrity("Plan parameter definitions are malformed.")

    def parameter_set(value: object, *, complete: bool) -> None:
        _verify_parameter_set_document(value)
        keys = {_parameter_key_integrity(item["parameter"]) for item in value["bindings"]}
        if not keys <= expected or (complete and keys != expected):
            raise _integrity("Parameter source does not bind the Plan's consumed definitions.")

    kind = source["kind"]
    if kind == "point":
        parameter_set(source["parameters"], complete=True)
    elif kind == "grid":
        parameter_set(source["base_parameters"], complete=True)
        axis_keys = []
        for axis in source["axes"]:
            key = _parameter_key_integrity(axis["parameter"])
            if key not in expected:
                raise _integrity("Grid axis does not bind a consumed Plan parameter.")
            axis_keys.append(key)
            for value in axis["values"]:
                _verify_parameter_value(value)
        if len(set(axis_keys)) != len(axis_keys):
            raise _integrity("Grid axes repeat a parameter.")
    elif kind == "points":
        parameter_set(source["baseline_parameters"], complete=True)
        for point in source["points"]:
            parameter_set(point, complete=False)

def _verify_view_declaration(
    view: object,
    plan: Mapping[str, object],
) -> tuple[list[str], bool]:
    if not isinstance(view, dict) or set(view) != {"type", "ptc", "transforms", "retain"} or view.get("type") != "network_view":
        raise _integrity("Declarative View is malformed.")
    _, public = _plan_coordinates(plan)
    available = list(sorted(public))
    connectivity = plan.get("connectivity")
    ports = connectivity.get("ports") if isinstance(connectivity, Mapping) else None
    if not isinstance(ports, list):
        raise _integrity("Plan Port inventory is malformed.")
    port_ids = [port.get("id") for port in ports]
    if any(not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in port_ids) or len(set(port_ids)) != len(port_ids):
        raise _integrity("Plan Port IDs are malformed.")
    net_to_compiler = {
        node["final_net"]: node["compiler_node_id"]
        for node in connectivity["node_coordinates"]
    }
    port_coordinates = {
        net_to_compiler[port["net"]]
        for port in ports
        if port.get("net") in net_to_compiler
    }
    ptc = view.get("ptc")
    if ptc is not None:
        if not isinstance(ptc, dict) or set(ptc) != {"selected_ports"}:
            raise _integrity("View PTC declaration is malformed.")
        selected = ptc["selected_ports"]
        if not isinstance(selected, list) or not selected or selected != [item for item in port_ids if item in selected] or len(set(selected)) != len(selected):
            raise _integrity("View PTC Port order is malformed.")
        by_id = {port["id"]: port for port in ports}
        if any(by_id[item].get("role") != "nonloading_probe" for item in selected):
            raise _integrity("View PTC selects a loading Port.")
    transforms = view.get("transforms")
    if not isinstance(transforms, list):
        raise _integrity("View transforms are malformed.")
    transform_ids: set[str] = set()
    for transform in transforms:
        if not isinstance(transform, dict) or set(transform) != {"id", "input_coordinates", "output_coordinates"}:
            raise _integrity("View transform is malformed.")
        identifier = transform.get("id")
        inputs = transform.get("input_coordinates")
        outputs = transform.get("output_coordinates")
        expected_outputs = [f"{identifier}.common", f"{identifier}.differential"]
        if (
            not isinstance(identifier, str)
            or _IDENTIFIER.fullmatch(identifier) is None
            or identifier in transform_ids
            or not isinstance(inputs, list)
            or len(inputs) != 2
            or len(set(inputs)) != 2
            or any(item not in available for item in inputs)
            or outputs != expected_outputs
            or any(item in available for item in expected_outputs)
        ):
            raise _integrity("View transform basis transition is malformed.")
        transform_ids.add(identifier)
        index = min(available.index(item) for item in inputs)
        both_ports = all(item in port_coordinates for item in inputs)
        available = [item for item in available if item not in inputs]
        available[index:index] = expected_outputs
        port_coordinates.difference_update(inputs)
        if both_ports:
            port_coordinates.update(expected_outputs)
    retain = view.get("retain")
    if retain is None:
        return list(port_ids), bool(port_ids)
    if not isinstance(retain, dict) or set(retain) != {"retained_coordinates"}:
        raise _integrity("View retain declaration is malformed.")
    retained = retain["retained_coordinates"]
    if not isinstance(retained, list) or not retained or len(set(retained)) != len(retained) or any(item not in available for item in retained):
        raise _integrity("View retained basis is malformed.")
    return list(retained), all(item in port_coordinates for item in retained)

def _verify_v1_lineage(lineage: object, plan: Mapping[str, object] | None) -> tuple[list[str], bool]:
    """Close the full dev5 View grammar without reimplementing compilation.

    Matrix bytes are compiler-owned evidence; the workspace binds their hashes,
    ordering and applicability rather than manufacturing a second compiler in
    Python.
    """

    if not isinstance(lineage, dict) or set(lineage) != {
        "type", "original", "ptc", "transforms", "retain", "terminal_coordinates", "port_realizable", "lineage_sha256",
    } or lineage.get("type") != "network_view_lineage":
        raise _integrity("View lineage envelope is open or malformed.")
    if lineage.get("lineage_sha256") != _sha256(_canonical_bytes({key: value for key, value in lineage.items() if key != "lineage_sha256"})):
        raise _integrity("View lineage hash does not bind its contents.")
    original = lineage.get("original")
    if not isinstance(original, dict) or set(original) != {"type", "compiled_graph_sha256", "coordinate_order", "port_order", "port_realizable"} or original.get("type") != "original":
        raise _integrity("Original View lineage is malformed.")
    _valid_sha(original.get("compiled_graph_sha256"))
    original_coordinates = _identifiers(original.get("coordinate_order"), field="Original coordinate order")
    if plan is not None and original_coordinates != _plan_coordinates(plan)[0]:
        raise _integrity("Original View coordinate order disagrees with the sealed Plan.")
    connectivity = plan.get("connectivity") if plan is not None else None
    plan_ports = connectivity.get("ports") if isinstance(connectivity, Mapping) else None
    if plan is not None and not isinstance(plan_ports, list):
        raise _integrity("Sealed Plan ports are malformed.")
    expected_ports = [port.get("id") for port in plan_ports if isinstance(port, dict)] if isinstance(plan_ports, list) else None
    port_roles = {
        port.get("id"): port.get("role")
        for port in plan_ports or ()
        if isinstance(port, dict)
    }
    port_order = _identifiers(original.get("port_order"), field="Original Port order", nonempty=False)
    if (expected_ports is not None and port_order != expected_ports) or not isinstance(original.get("port_realizable"), bool):
        raise _integrity("Original View Port identity disagrees with the sealed Plan.")
    coordinates = list(original_coordinates)
    ptc = lineage.get("ptc")
    if ptc is not None:
        if not isinstance(ptc, dict) or set(ptc) != {"type", "selected_ports", "load_mask_sha256", "loads", "reconstruction_residual_f64", "output_coordinate_order", "evidence_sha256"} or ptc.get("type") != "ptc":
            raise _integrity("PTC lineage step is malformed.")
        selected = _identifiers(ptc.get("selected_ports"), field="PTC selected Ports")
        if any(port not in port_order for port in selected) or ptc.get("output_coordinate_order") != coordinates:
            raise _integrity("PTC lineage does not preserve the original coordinate basis.")
        if any(port_roles.get(port) != "nonloading_probe" for port in selected):
            raise _integrity("PTC selects a Port that is not a nonloading probe.")
        if not isinstance(ptc.get("loads"), list) or [item.get("port_id") if isinstance(item, dict) else None for item in ptc["loads"]] != selected:
            raise _integrity("PTC load evidence does not match its selected Port order.")
        for item in ptc["loads"]:
            if not isinstance(item, dict) or set(item) != {"port_id", "reference_impedance", "before", "after"} or item.get("before") != "raw" or item.get("after") != "compensated":
                raise _integrity("PTC load evidence is malformed.")
            _verify_quantity_role(item.get("reference_impedance"), complex_value=False, unit="ohm", dimensionality="resistance")
        _valid_sha(ptc.get("load_mask_sha256")); _valid_sha(ptc.get("evidence_sha256")); _f64_value(ptc.get("reconstruction_residual_f64"))
    transforms = lineage.get("transforms")
    if not isinstance(transforms, list):
        raise _integrity("Transform lineage must be an array.")
    for step in transforms:
        fields = {"type", "input_coordinates", "weights_f64", "differential_id", "common_id", "included_external_cut_branches", "excluded_direct_mutual_branches", "reference_matrix", "principal_root", "reconstruction_residual_f64", "output_coordinate_order", "evidence_sha256"}
        if not isinstance(step, dict) or set(step) != fields or step.get("type") != "transform_pair":
            raise _integrity("Transform lineage step is malformed.")
        pair = _identifiers(step.get("input_coordinates"), field="Transform input coordinates")
        if len(pair) != 2 or any(item not in coordinates for item in pair):
            raise _integrity("Transform input coordinates are not an ordered current pair.")
        weights = step.get("weights_f64")
        if not isinstance(weights, list) or len(weights) != 2 or any(not _finite_f64(item) for item in weights):
            raise _integrity("Transform weights are malformed.")
        common, differential = step.get("common_id"), step.get("differential_id")
        if any(not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in (common, differential)) or differential == common or differential in coordinates or common in coordinates:
            raise _integrity("Transform output coordinate identity is malformed.")
        expected = [item for item in coordinates if item not in pair] + [common, differential]
        if step.get("output_coordinate_order") != expected:
            raise _integrity("Transform output ordering is not canonical.")
        for matrix in ("reference_matrix", "principal_root"):
            evidence = step.get(matrix)
            if not isinstance(evidence, dict) or set(evidence) != {"rows", "columns", "sha256"} or any(not isinstance(evidence.get(field), int) or evidence[field] < 0 for field in ("rows", "columns")):
                raise _integrity("Transform matrix evidence is malformed.")
            _valid_sha(evidence.get("sha256"))
        _verify_branch_refs(step.get("included_external_cut_branches"), field="Transform included cut branches", nonempty=True)
        _verify_branch_refs(step.get("excluded_direct_mutual_branches"), field="Transform excluded direct-mutual branches", nonempty=False)
        _valid_sha(step.get("evidence_sha256")); _f64_value(step.get("reconstruction_residual_f64"))
        coordinates = expected
    retain = lineage.get("retain")
    if retain is not None:
        fields = {"type", "retained_coordinates", "eliminated_coordinates", "output_coordinate_order", "a_matrix", "b_matrix", "r_matrix", "d_matrix", "q_matrix", "selected_projector", "omitted_projector", "omitted_matched_loads", "source_boundary_sha256", "deembedding_evidence_sha256"}
        if not isinstance(retain, dict) or set(retain) != fields or retain.get("type") != "retain":
            raise _integrity("Retain lineage step is malformed.")
        retained = _identifiers(retain.get("retained_coordinates"), field="Retained coordinates")
        eliminated = _identifiers(retain.get("eliminated_coordinates"), field="Eliminated coordinates", nonempty=False)
        if set(retained) | set(eliminated) != set(coordinates) or set(retained) & set(eliminated) or retain.get("output_coordinate_order") != retained:
            raise _integrity("Retain lineage is not an exact partition of its input basis.")
        for matrix in ("a_matrix", "b_matrix", "r_matrix", "d_matrix", "q_matrix", "selected_projector", "omitted_projector", "omitted_matched_loads"):
            evidence = retain.get(matrix)
            if not isinstance(evidence, dict) or set(evidence) != {"rows", "columns", "sha256"}:
                raise _integrity("Retain matrix evidence is malformed.")
            _valid_sha(evidence.get("sha256"))
        _valid_sha(retain.get("source_boundary_sha256")); _valid_sha(retain.get("deembedding_evidence_sha256"))
        coordinates = retained
    terminal = _identifiers(lineage.get("terminal_coordinates"), field="Terminal channel order")
    port_realizable = lineage.get("port_realizable")
    # The compiler's original basis is physical-node ordered, whereas the raw
    # public Direct boundary is the declared logical-Port order.  Transforms
    # alter quantity coordinates but do not themselves create a wave boundary;
    # only terminal retain() selects transformed channel IDs.
    expected_terminal = coordinates if retain is not None else port_order
    if terminal != expected_terminal or not isinstance(port_realizable, bool):
        raise _integrity("Terminal View capability disagrees with its lineage.")
    return terminal, port_realizable

def _verify_v1_direct_spec(spec: object, terminal: list[str], port_realizable: bool) -> None:
    if not port_realizable:
        raise _integrity("Direct S/Y/Z request is not Port-realizable.")
    if not isinstance(spec, dict) or set(spec) != {"type", "frequencies", "traces"} or spec.get("type") != "direct_solve":
        raise _integrity("Direct solve Spec is malformed.")
    frequencies = spec.get("frequencies")
    if not isinstance(frequencies, list) or not frequencies:
        raise _integrity("Direct solve frequency grid is malformed.")
    previous = 0.0
    for frequency in frequencies:
        _verify_quantity_role(frequency, complex_value=False, unit="hertz", dimensionality="inverse_time")
        value = _f64_value(frequency["si_value_f64"])
        if value <= previous:
            raise _integrity("Direct solve frequency grid is not strictly positive and increasing.")
        previous = value
    traces = spec.get("traces")
    if not isinstance(traces, list):
        raise _integrity("Direct trace declarations are malformed.")
    trace_ids: set[str] = set()
    for trace in traces:
        if (
            not isinstance(trace, dict)
            or set(trace) != {"id", "input_port", "input_mode", "output_port", "output_mode"}
            or not isinstance(trace.get("id"), str)
            or _IDENTIFIER.fullmatch(trace["id"]) is None
            or trace["id"] in trace_ids
            or trace.get("input_port") not in terminal
            or trace.get("output_port") not in terminal
            or trace.get("input_mode") != []
            or trace.get("output_mode") != []
        ):
            raise _integrity("Direct trace declaration is malformed.")
        trace_ids.add(trace["id"])

def _verify_v1_hb_spec(
    spec: object,
    terminal: list[str],
    port_realizable: bool,
    logical_ports: list[str],
) -> None:
    """Close the declared HB request before workspace evidence is allocated.

    Lattice realization remains Julia-owned, but a sealed request must already
    bind a complete, unique authoring declaration to a port-realizable View.
    """

    if not port_realizable:
        raise _integrity("HB S/Y/Z request is not Port-realizable.")
    expected = {
        "type", "pump_axes", "drives", "frequencies", "cases", "truncation",
        "traces", "allow_driven_ptc",
    }
    if not isinstance(spec, dict) or set(spec) != expected or spec.get("type") != "hb_solve":
        raise _integrity("HB solve Spec is malformed.")
    axes = spec.get("pump_axes")
    drives = spec.get("drives")
    cases = spec.get("cases")
    traces = spec.get("traces")
    frequencies = spec.get("frequencies")
    truncation = spec.get("truncation")
    if not isinstance(axes, list) or not isinstance(drives, list) or not isinstance(cases, list) or not cases or not isinstance(traces, list) or not isinstance(frequencies, list) or not frequencies or not isinstance(truncation, dict) or not isinstance(spec.get("allow_driven_ptc"), bool):
        raise _integrity("HB solve Spec has incomplete declarations.")
    axis_ids: set[str] = set()
    for axis in axes:
        if not isinstance(axis, dict) or set(axis) != {"id", "frequency"} or not isinstance(axis.get("id"), str) or _IDENTIFIER.fullmatch(axis["id"]) is None or axis["id"] in axis_ids:
            raise _integrity("HB pump-axis declaration is malformed.")
        _verify_quantity_role(axis.get("frequency"), complex_value=False, unit="hertz", dimensionality="inverse_time")
        if _f64_value(axis["frequency"]["si_value_f64"]) <= 0.0:
            raise _integrity("HB pump-axis frequency is not strictly positive.")
        axis_ids.add(axis["id"])
    previous = 0.0
    for frequency in frequencies:
        _verify_quantity_role(frequency, complex_value=False, unit="hertz", dimensionality="inverse_time")
        value = _f64_value(frequency["si_value_f64"])
        if value <= previous:
            raise _integrity("HB response frequency grid is not strictly positive and increasing.")
        previous = value
    drive_ids: set[str] = set()
    for drive in drives:
        if (
            not isinstance(drive, dict)
            or set(drive) != {"id", "port_id", "mode", "orientation"}
            or not isinstance(drive.get("id"), str)
            or _IDENTIFIER.fullmatch(drive["id"]) is None
            or drive["id"] in drive_ids
            or drive.get("port_id") not in logical_ports
            or drive.get("orientation") != "port_node_to_reference"
            or not _valid_mode_tuple(drive.get("mode"), len(axis_ids))
        ):
            raise _integrity("HB current-drive declaration is malformed.")
        drive_ids.add(drive["id"])
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "currents"} or not isinstance(case.get("id"), str) or _IDENTIFIER.fullmatch(case["id"]) is None or case["id"] in case_ids or not isinstance(case.get("currents"), list):
            raise _integrity("HB case declaration is malformed.")
        current_ids: set[str] = set()
        for current in case["currents"]:
            if (
                not isinstance(current, dict)
                or set(current) != {"drive_id", "coefficient", "coefficient_convention"}
                or current.get("drive_id") not in drive_ids
                or current["drive_id"] in current_ids
                or current.get("coefficient_convention") != "exp_minus_i_m_dot_omega_t_fourier_coefficient"
            ):
                raise _integrity("HB case current binding is malformed.")
            _verify_quantity_role(current.get("coefficient"), complex_value=True, unit="ampere", dimensionality="current")
            current_ids.add(current["drive_id"])
        case_ids.add(case["id"])
    expected_truncation = {"pump_harmonics", "modulation_harmonics", "max_intermodulation_order", "three_wave_mixing", "four_wave_mixing"}
    if set(truncation) != expected_truncation or not isinstance(truncation.get("pump_harmonics"), list) or not isinstance(truncation.get("modulation_harmonics"), list) or len(truncation["pump_harmonics"]) != len(axis_ids) or len(truncation["modulation_harmonics"]) != len(axis_ids) or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in [*truncation["pump_harmonics"], *truncation["modulation_harmonics"]]) or (truncation.get("max_intermodulation_order") is not None and (not isinstance(truncation["max_intermodulation_order"], int) or isinstance(truncation["max_intermodulation_order"], bool) or truncation["max_intermodulation_order"] < 0)) or not isinstance(truncation.get("three_wave_mixing"), bool) or not isinstance(truncation.get("four_wave_mixing"), bool):
        raise _integrity("HB truncation declaration is malformed.")
    trace_ids: set[str] = set()
    for trace in traces:
        if (
            not isinstance(trace, dict)
            or set(trace) != {"id", "input_port", "input_mode", "output_port", "output_mode"}
            or not isinstance(trace.get("id"), str)
            or _IDENTIFIER.fullmatch(trace["id"]) is None
            or trace["id"] in trace_ids
            or trace.get("input_port") not in terminal
            or trace.get("output_port") not in terminal
            or not _valid_mode_tuple(trace.get("input_mode"), len(axis_ids))
            or not _valid_mode_tuple(trace.get("output_mode"), len(axis_ids))
        ):
            raise _integrity("HB trace declaration is malformed.")
        trace_ids.add(trace["id"])

def _valid_mode_tuple(value: object, rank: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == rank
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )

def _hb_operating_lattice_is_vacuous(spec: Mapping[str, object]) -> bool:
    """Reproduce the pinned JC empty operating-basis condition from the request.

    This delegates to the full pinned ordering reconstruction below, so its
    vacuity rule cannot drift from the actual RFFT/parity/crop construction.
    """

    return not _hb_declared_modes_from_spec(spec, response=False)

def _hb_declared_modes_from_spec(spec: Mapping[str, object], *, response: bool) -> list[list[int]]:
    """Mirror the pinned JC 0.5.4 Fourier construction and its ordering.

    The receipt does not merely attest that a returned set fits the requested
    bounds.  `calcfreqsrdft`/`calcfreqsdft`, `truncfreqs`, and (for the
    operating basis) `removeconjfreqs` determine the ordered public channel
    basis.  Reconstructing it here closes result reuse against a backend that
    has silently permuted an otherwise valid lattice.
    """

    axes = spec.get("pump_axes")
    truncation = spec.get("truncation")
    drives = spec.get("drives")
    if not isinstance(axes, list) or not isinstance(truncation, Mapping) or not isinstance(drives, list):
        raise _integrity("HB request cannot reproduce its pinned JC lattice.")
    rank = len(axes)
    limits_key = "modulation_harmonics" if response else "pump_harmonics"
    limits = truncation.get(limits_key)
    if (
        not isinstance(limits, list)
        or len(limits) != rank
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in limits)
        or not isinstance(truncation.get("three_wave_mixing"), bool)
        or not isinstance(truncation.get("four_wave_mixing"), bool)
    ):
        raise _integrity("HB request has an invalid JC lattice truncation.")
    crop = truncation.get("max_intermodulation_order")
    if crop is not None and (not isinstance(crop, int) or isinstance(crop, bool) or crop < 0):
        raise _integrity("HB request has an invalid JC intermodulation crop.")
    declared_dc = any(
        isinstance(drive, Mapping)
        and isinstance(drive.get("mode"), list)
        and len(drive["mode"]) == rank
        and all(value == 0 for value in drive["mode"])
        for drive in drives
    )
    if rank == 0:
        # This is SCNSim's documented private JC adapter: the backend uses an
        # inert `(0,)`, while the public rank-zero basis is `()`.
        return [[]] if response or declared_dc else []

    # Julia CartesianIndices is column-major: the first pump axis advances
    # first.  Iterate reversed Python products to preserve that exact order.
    dimensions = [2 * limit + 1 for limit in limits] if response else [limits[0] + 1, *[2 * limit + 1 for limit in limits[1:]]]
    modes: list[tuple[int, ...]] = []
    for reversed_indices in product(*(range(1, dimension + 1) for dimension in reversed(dimensions))):
        indices = tuple(reversed(reversed_indices))
        mode = tuple(
            index - 1 if index <= limit + 1 else -dimension + index - 1
            for index, limit, dimension in zip(indices, limits, dimensions)
        )
        absolute_order = sum(abs(value) for value in mode)
        criterion = (
            (response and all(value == 0 for value in mode))
            or ((truncation["four_wave_mixing"] if response else truncation["three_wave_mixing"]) and absolute_order > 0 and absolute_order % 2 == 0)
            or ((truncation["three_wave_mixing"] if response else truncation["four_wave_mixing"]) and absolute_order % 2 == 1)
            or (not response and declared_dc and all(value == 0 for value in mode))
        )
        if criterion and (sum(value != 0 for value in mode) == 1 or crop is None or absolute_order <= crop):
            modes.append(mode)
    if response:
        return [list(mode) for mode in modes]

    # `removeconjfreqs` removes the lexicographically greater coordinate of
    # every JC-conjugate pair and retains original Cartesian order.
    nw = tuple(dimensions)
    nt = (2 * nw[0] - 1, *nw[1:])
    removed: set[tuple[int, ...]] = set()
    for reversed_indices in product(*(range(1, dimension + 1) for dimension in reversed(nw))):
        coordinate = tuple(reversed(reversed_indices))
        target = tuple((length - (index - 1)) % length + 1 for index, length in zip(coordinate, nt))
        if coordinate != target and all(index <= dimension for index, dimension in zip(target, nw)):
            removed.add(max(coordinate, target))
    retained: list[list[int]] = []
    for reversed_indices in product(*(range(1, dimension + 1) for dimension in reversed(nw))):
        coordinate = tuple(reversed(reversed_indices))
        if coordinate in removed:
            continue
        mode = tuple(
            index - 1 if index <= limit + 1 else -dimension + index - 1
            for index, limit, dimension in zip(coordinate, limits, dimensions)
        )
        if mode in modes:
            retained.append(list(mode))
    return retained

def _verify_v1_evaluation_spec(
    spec: object,
    terminal: list[str],
    port_realizable: bool,
    *,
    residue_branch: bool = False,
) -> None:
    if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
        raise _integrity("Direct evaluation Spec is malformed.")
    kind = spec["type"]
    if kind == "diagonal_root":
        coordinate = spec.get("coordinate")
        invalid = coordinate not in terminal or (residue_branch and len(terminal) < 2)
        if set(spec) != {"type", "coordinate", "root_hint"} or invalid:
            raise _integrity("Diagonal-root Spec is incompatible with its final View.")
        _verify_quantity_role(spec.get("root_hint"), complex_value=False, unit="hertz", dimensionality="inverse_time")
        if _f64_value(spec["root_hint"]["si_value_f64"]) <= 0.0:
            raise _integrity("Diagonal-root hint must be positive.")
    elif kind == "operator_element_root":
        if set(spec) != {"type", "row", "column", "root_hint"} or spec.get("row") not in terminal or spec.get("column") not in terminal:
            raise _integrity("Operator-element-root Spec is incompatible with its final View.")
        _verify_quantity_role(spec.get("root_hint"), complex_value=False, unit="hertz", dimensionality="inverse_time")
        if _f64_value(spec["root_hint"]["si_value_f64"]) <= 0.0:
            raise _integrity("Operator-element-root hint must be positive.")
    elif kind == "hybridized_pole":
        if set(spec) != {"type", "coordinates", "anchor"} or len(terminal) < 2 or _identifiers(spec.get("coordinates"), field="Hybridized-pole coordinates") != terminal:
            raise _integrity("Hybridized-pole coordinates must equal the complete retained View.")
        _verify_frequency_anchor(spec.get("anchor"))
    elif kind == "transfer_zero":
        if set(spec) != {"type", "anchor", "family", "input_coordinate", "output_coordinate"} or spec.get("family") not in {"S", "Y", "Z"} or spec.get("input_coordinate") not in terminal or spec.get("output_coordinate") not in terminal:
            raise _integrity("Transfer-zero Spec is malformed.")
        if spec.get("family") == "S" and not port_realizable:
            raise _integrity("S-family transfer-zero evaluation is not Port-realizable.")
        _verify_frequency_anchor(spec.get("anchor"))
    elif kind == "residue_normalized_coupling":
        branches = (spec.get("branch_a"), spec.get("branch_b"))
        if (
            set(spec) != {"type", "branch_a", "branch_b", "frequency"}
            or any(not isinstance(branch, dict) or branch.get("type") not in {"diagonal_root", "hybridized_pole"} for branch in branches)
        ):
            raise _integrity("Residue-normalized coupling Spec is malformed.")
        _verify_v1_evaluation_spec(branches[0], terminal, port_realizable, residue_branch=True)
        _verify_v1_evaluation_spec(branches[1], terminal, port_realizable, residue_branch=True)
        frequency = spec.get("frequency")
        if frequency == "complex_root_midpoint":
            pass
        else:
            _verify_quantity_role(frequency, complex_value=False, unit="hertz", dimensionality="inverse_time")
            if _f64_value(frequency["si_value_f64"]) <= 0.0:
                raise _integrity("Residue coupling frequency must be positive.")
    elif kind == "response_element":
        if set(spec) != {"type", "family", "input_coordinate", "output_coordinate", "frequency"} or spec.get("family") not in {"S", "Y", "Z"} or spec.get("input_coordinate") not in terminal or spec.get("output_coordinate") not in terminal:
            raise _integrity("Response-element Spec is malformed.")
        if spec.get("family") == "S" and not port_realizable:
            raise _integrity("S-family response evaluation is not Port-realizable.")
        _verify_quantity_role(spec.get("frequency"), complex_value=False, unit="hertz", dimensionality="inverse_time")
    elif kind == "operator":
        if set(spec) != {"type", "frequencies"}:
            raise _integrity("Operator Spec is malformed.")
        _verify_v1_direct_spec({"type": "direct_solve", "frequencies": spec.get("frequencies"), "traces": []}, terminal, True)
    else:
        raise _integrity("Direct evaluation Spec is outside dev5.")

def _verify_frequency_anchor(value: object) -> None:
    if isinstance(value, dict) and value.get("type") == "quantity_f64":
        _verify_quantity_role(value, complex_value=False, unit="hertz", dimensionality="inverse_time")
    else:
        _verify_quantity_role(value, complex_value=True, unit="hertz", dimensionality="inverse_time")

def _verify_v1_optimization_spec(spec: object, plan: Mapping[str, object], initial_parameters: object) -> None:
    if not isinstance(spec, dict) or set(spec) != {"type", "variables", "objectives", "optimizer", "allow_extrapolation"} or spec.get("type") != "optimization":
        raise _integrity("Optimization Spec is malformed.")
    variables, objectives, optimizer, authorizations = spec.get("variables"), spec.get("objectives"), spec.get("optimizer"), spec.get("allow_extrapolation")
    if not isinstance(variables, list) or not variables or not isinstance(objectives, list) or not objectives or not isinstance(optimizer, dict) or not isinstance(authorizations, list):
        raise _integrity("Optimization Spec has malformed collections.")
    variable_keys: list[tuple[tuple[str, ...], str]] = []
    initial = {_parameter_key_integrity(binding["parameter"]): binding["value"]
        for binding in initial_parameters.get("bindings", [])} if isinstance(initial_parameters, Mapping) else {}
    for variable in variables:
        if not isinstance(variable, dict):
            raise _integrity("Optimization variable is malformed.")
        key = _parameter_key_integrity(variable.get("parameter")); variable_keys.append(key)
        if "domain" in variable:
            domain = variable.get("domain")
            linear = domain in {"UNBOUNDED", "NONNEGATIVE", "NONPOSITIVE"}
            signed = domain in {"POSITIVE", "NEGATIVE"}
            if set(variable) != {"parameter", "domain", "transform", "scale"} or not (linear or signed) or variable.get("transform") != ("linear" if linear else "log"):
                raise _integrity("Optimization domain variable is malformed.")
            scale = variable.get("scale")
            value = initial.get(key)
            if not isinstance(value, Mapping):
                raise _integrity("Optimization domain initial parameter is absent.")
            x0 = _f64_value(value.get("si_value_f64"))
            if (domain == "POSITIVE" and x0 <= 0.0 or domain == "NEGATIVE" and x0 >= 0.0 or
                domain == "NONNEGATIVE" and x0 < 0.0 or domain == "NONPOSITIVE" and x0 > 0.0):
                raise _integrity("Optimization domain initial value is outside its domain.")
            if linear:
                _verify_quantity_compatible(scale, value)
                magnitude = _f64_value(scale["si_value_f64"])
                if magnitude <= 0.0 or (domain != "UNBOUNDED" and not math.isfinite(x0 / magnitude)):
                    raise _integrity("Optimization domain scale must be positive.")
            elif scale is not None:
                raise _integrity("Signed log domain cannot have physical scale.")
            continue
        if set(variable) != {"parameter", "model_default_bounds", "consumer_override_bounds", "lower", "upper", "transform"} or variable.get("transform") not in {"linear", "log"}:
            raise _integrity("Optimization bounded variable is malformed.")
        for name in ("model_default_bounds", "consumer_override_bounds"):
            bounds = variable.get(name)
            if bounds is None and name == "consumer_override_bounds":
                continue
            _verify_bounds(bounds)
        _verify_quantity_compatible(variable.get("lower"), variable.get("upper"))
        if variable.get("consumer_override_bounds") is None:
            if variable.get("model_default_bounds") != [variable.get("lower"), variable.get("upper")]:
                raise _integrity("Optimization resolved bounds do not preserve model defaults.")
        elif variable.get("consumer_override_bounds") != [variable.get("lower"), variable.get("upper")]:
            raise _integrity("Optimization resolved bounds do not match consumer override.")
    if len(set(variable_keys)) != len(variable_keys):
        raise _integrity("Optimization variables are not unique.")
    authorization_keys = [_parameter_key_integrity(item) for item in authorizations]
    if authorization_keys != sorted(set(authorization_keys)) or any(key not in variable_keys for key in authorization_keys):
        raise _integrity("Optimization extrapolation authorization is not sorted active variables.")
    objective_ids: set[str] = set()
    for objective in objectives:
        if not isinstance(objective, dict) or set(objective) != {"id", "quantity", "comparison", "target", "weight_f64", "resolved_scale", "scale_source"} or not isinstance(objective.get("id"), str) or _IDENTIFIER.fullmatch(objective["id"]) is None or objective["id"] in objective_ids:
            raise _integrity("Optimization objective is malformed.")
        objective_ids.add(objective["id"])
        role = _verify_selector(objective.get("quantity"), plan)
        _verify_quantity_role(objective.get("target"), complex_value=False, unit=role[0], dimensionality=role[1])
        _verify_quantity_role(objective.get("resolved_scale"), complex_value=False, unit=role[0], dimensionality=role[1])
        resolved_scale = objective["resolved_scale"]
        weight = objective.get("weight_f64")
        if (
            not _finite_f64(weight)
            or _f64_value(weight) <= 0.0
            or _f64_value(resolved_scale["si_value_f64"]) <= 0.0
            or objective.get("comparison") not in {"target", "at_least"}
            or objective.get("scale_source") not in {"relative_target", "dimensionless_unity", "explicit"}
        ):
            raise _integrity("Optimization objective scale is malformed.")
    required_optimizer = {"type", "seed", "max_evaluations", "population_size", "resolved_population_size", "initial_sigma_f64", "baseline_optimizer_coordinates_f64", "box_transform_id", "complete_generations", "unused_evaluations", "hidden_stops"}
    expected_map = "cmaes-jl-0.2.6-native-domains.v1" if any("domain" in v for v in variables) else "cmaes-jl-0.2.6-linquad-unit-box.v1"
    if set(optimizer) != required_optimizer or optimizer.get("type") != "cma_es" or optimizer.get("box_transform_id") != expected_map or optimizer.get("hidden_stops") != "disabled":
        raise _integrity("Optimization controls are malformed.")
    baseline_coordinates = optimizer.get("baseline_optimizer_coordinates_f64")
    if (
        not isinstance(baseline_coordinates, list)
        or len(baseline_coordinates) != len(variables)
        or any(not _finite_f64(value) or (
            _f64_value(value) != 0.0 if "domain" in variable
            else not 0.0 <= _f64_value(value) <= 1.0
        ) for value, variable in zip(baseline_coordinates, variables))
    ):
        raise _integrity("Optimization baseline coordinates are malformed.")

def _optimization_selector_leaves(spec: object) -> list[Mapping[str, object]]:
    if not isinstance(spec, Mapping) or not isinstance(spec.get("objectives"), list):
        return []
    leaves: list[Mapping[str, object]] = []
    for objective in spec["objectives"]:
        quantity = objective.get("quantity") if isinstance(objective, Mapping) else None
        if isinstance(quantity, Mapping):
            leaves.extend(_selector_terms(quantity))
    return leaves

def _optimization_leaf_catalog(objectives: object) -> list[tuple[dict[str, object], Mapping[str, object]]]:
    if not isinstance(objectives, list):
        raise _integrity("Optimization objectives are malformed.")
    leaves: list[tuple[dict[str, object], Mapping[str, object]]] = []
    for objective in objectives:
        if not isinstance(objective, Mapping) or not isinstance(objective.get("id"), str):
            raise _integrity("Optimization objective identity is malformed.")
        for ordinal, selector in enumerate(_selector_terms(objective.get("quantity")), 1):
            leaves.append(({"objective_id": objective["id"], "term_ordinal": ordinal}, selector))
    return leaves

def _optimization_dependency(selector: Mapping[str, object], *, kind: str = "quantity") -> dict[str, object]:
    view = selector.get("view")
    if not isinstance(view, Mapping):
        raise _integrity("Optimization selector View is malformed.")
    view_sha = _sha256(_canonical_bytes(view))
    selector_type = selector.get("type")
    if selector_type in {"diagonal_root_projection", "residue_diagonal_root_projection", "operator_element_root_projection"}:
        spec = selector.get("spec")
        if not isinstance(spec, Mapping):
            raise _integrity("Optimization root selector is malformed.")
        row = spec.get("row") if selector_type == "operator_element_root_projection" else spec.get("coordinate")
        column = spec.get("column") if selector_type == "operator_element_root_projection" else spec.get("coordinate")
        record = {"type": "selected_element_root", "view": view, "row": row, "column": column, "root_hint": spec.get("root_hint")}
    else:
        record = {"type": selector_type, "spec": selector.get("spec"), "view": view}
    dependency_sha = view_sha if kind == "view" else _sha256(_canonical_bytes(record))
    return {"kind": kind, "view_sha256": view_sha, "dependency_sha256": dependency_sha}

def _verify_optimization_context_shape(value: object) -> Mapping[str, object]:
    required = {"schema", "schema_version", "phase", "candidate", "owner", "affected_leaves"}
    if not isinstance(value, Mapping) or not required.issubset(value) or not set(value).issubset(required | {"dependency", "aggregation_witness"}):
        raise _integrity("Optimization failure context is open or malformed.")
    if value.get("schema") != "scnsim.optimization_failure_context" or value.get("schema_version") != 1 or value.get("phase") not in {
        "candidate_prepare", "candidate_compile", "view_realization", "baseline_root_anchor",
        "quantity_evaluation", "objective_aggregation", "total_aggregation",
    }:
        raise _integrity("Optimization failure phase is malformed.")
    candidate = value.get("candidate")
    if not isinstance(candidate, Mapping) or set(candidate) != {"evaluation_ordinal", "origin", "generation", "population_column"}:
        raise _integrity("Optimization failure candidate position is malformed.")
    ordinal, origin, generation, column = (candidate.get(key) for key in ("evaluation_ordinal", "origin", "generation", "population_column"))
    integer = lambda item: isinstance(item, int) and not isinstance(item, bool)
    if not integer(ordinal) or not integer(generation) or ordinal < 0 or generation < 0:
        raise _integrity("Optimization failure candidate ordinals are malformed.")
    if origin == "baseline":
        if (ordinal, generation, column) != (0, 0, None):
            raise _integrity("Optimization baseline failure position is inconsistent.")
    elif origin == "population":
        if ordinal < 1 or generation < 1 or not integer(column) or column < 1:
            raise _integrity("Optimization population failure position is inconsistent.")
    else:
        raise _integrity("Optimization failure candidate origin is unknown.")
    owner = value.get("owner")
    valid_owner = (
        isinstance(owner, Mapping)
        and (
            set(owner) == {"kind"} and owner.get("kind") in {"candidate", "dependency"}
            or set(owner) == {"kind", "leaf"} and owner.get("kind") == "leaf"
            or set(owner) == {"kind", "objective_id"} and owner.get("kind") == "objective" and isinstance(owner.get("objective_id"), str)
        )
    )
    if not valid_owner:
        raise _integrity("Optimization failure owner is malformed.")
    affected = value.get("affected_leaves")
    locators = [owner.get("leaf")] if owner.get("kind") == "leaf" else []
    if not isinstance(affected, list) or any(not isinstance(item, Mapping) for item in [*affected, *locators]):
        raise _integrity("Optimization failure leaf locators are malformed.")
    for locator in [*affected, *locators]:
        if set(locator) != {"objective_id", "term_ordinal"} or not isinstance(locator.get("objective_id"), str) or not integer(locator.get("term_ordinal")) or locator["term_ordinal"] < 1:
            raise _integrity("Optimization failure leaf locator is malformed.")
    if len({_canonical_bytes(item) for item in affected}) != len(affected):
        raise _integrity("Optimization affected leaves repeat a locator.")
    dependency = value.get("dependency")
    if dependency is not None and (
        not isinstance(dependency, Mapping)
        or set(dependency) != {"kind", "view_sha256", "dependency_sha256"}
        or dependency.get("kind") not in {"view", "quantity"}
        or _SHA256.fullmatch(str(dependency.get("view_sha256", ""))) is None
        or _SHA256.fullmatch(str(dependency.get("dependency_sha256", ""))) is None
    ):
        raise _integrity("Optimization failure dependency is malformed.")
    if owner.get("kind") == "dependency" and dependency is None:
        raise _integrity("Optimization dependency failure has no dependency identity.")
    witness = value.get("aggregation_witness")
    needs_witness = origin == "baseline" and value.get("phase") in {
        "objective_aggregation",
        "total_aggregation",
    }
    if needs_witness and witness is None:
        raise _integrity("Optimization baseline aggregation failure lacks its witness.")
    if not needs_witness and witness is not None:
        raise _integrity("Optimization failure carries an inapplicable aggregation witness.")
    if witness is not None:
        if not isinstance(witness, Mapping):
            raise _integrity("Optimization aggregation witness is malformed.")
        if value.get("phase") == "objective_aggregation":
            if (
                set(witness) != {"kind", "objective_id", "terms"}
                or witness.get("kind") != "objective"
                or not isinstance(witness.get("objective_id"), str)
                or not isinstance(witness.get("terms"), list)
            ):
                raise _integrity("Optimization objective aggregation witness is malformed.")
        elif value.get("phase") == "total_aggregation":
            if (
                set(witness) != {"kind", "objective_components"}
                or witness.get("kind") != "total"
                or not isinstance(witness.get("objective_components"), list)
            ):
                raise _integrity("Optimization total aggregation witness is malformed.")
        else:
            raise _integrity("Optimization non-aggregation failure carries an aggregation witness.")
    return value

def _is_projection_only_optimization_failure(value: object) -> bool:
    """Classify the closed selector failures that own no shared dependency."""

    if not isinstance(value, Mapping):
        return False
    evidence = value.get("evidence")
    return (
        value.get("kind") == "invalid_optimization_spec"
        and value.get("stage") in {"selector", "quantity_sum"}
        and isinstance(evidence, Mapping)
        and evidence.get("operation") == "optimize_direct"
        and evidence.get("context_kind") == "optimization_candidate"
    )

def _is_shared_element_root_failure(value: object, selector: Mapping[str, object]) -> bool:
    """Distinguish a shared numerical root from a leaf-local passive policy."""

    if selector.get("type") not in {"diagonal_root_projection", "operator_element_root_projection"} or not isinstance(value, Mapping):
        return False
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("operation") != "optimize_direct":
        return False
    return (
        evidence.get("context_kind") in {"direct_quantity", "direct_response"}
        and value.get("kind") in {
            "eliminated_block_solve_failure", "root_slope_unresolved",
            "numerical_resolution_unresolved", "direct_response_formation",
        }
        or evidence.get("context_kind") == "optimization_candidate"
        and value.get("kind") == "numerical_resolution_unresolved"
        and value.get("stage") == "series_rl"
    )

def _is_leaf_local_passive_root_failure(value: object, selector: Mapping[str, object]) -> bool:
    if not isinstance(value, Mapping) or value.get("kind") != "numerical_resolution_unresolved" or value.get("stage") != "newton_certificate":
        return False
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("operation") != "optimize_direct":
        return False
    return (
        selector.get("type") == "diagonal_root_projection" and evidence.get("context_kind") == "optimization_candidate"
        or selector.get("type") == "residue_coupling_projection" and evidence.get("context_kind") == "direct_quantity"
    )

def _optimization_failure_requires_context(value: object, operation: object) -> bool:
    """Return whether a sealed optimization failure is execution-owned."""

    if operation != "optimize_direct" or not isinstance(value, Mapping):
        return False
    evidence = value.get("evidence")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("operation") != "optimize_direct"
        or evidence.get("context_kind")
        not in {"optimization_candidate", "direct_quantity", "direct_response"}
    ):
        return False
    kind = value.get("kind")
    return (
        kind
        in {
            "direct_response_formation",
            "invalid_candidate_physical_parameter",
            "eliminated_block_solve_failure",
            "root_slope_unresolved",
            "numerical_resolution_unresolved",
            "unsupported_singular_capacitance_for_diagonal_root_v1",
            "port_realizability",
        }
        or kind == "compiler_invariant"
        and value.get("stage") not in {"optimization", "optimization_replay"}
        or _is_projection_only_optimization_failure(value)
    )

def _verify_failure_document(value: object, operation: object) -> None:
    if not isinstance(value, dict) or set(value) != {"category", "kind", "stage", "message", "evidence"}:
        raise _integrity("Failure envelope is open or malformed.")
    evidence = value.get("evidence")
    allowed = {
        "type", "operation", "context_kind", "plan_sha256", "request_sha256",
        "attempt_sha256", "workspace_instance_id", "component_path", "parameter",
        "coordinate_id", "port_id", "case_id", "candidate_ordinal", "artifact_id",
        "artifact_path", "expected_sha256", "actual_sha256", "backend_exit_code",
        "evidence_sha256", "optimization_context",
    }
    categories = {
        "plan_sealed": "state",
        "workspace_plan_replaced": "state",
        "workspace_commit_indeterminate": "state",
        "workspace_versioning_downgrade_forbidden": "state",
        "unsupported_runtime_platform": "capability",
        "unsupported_singular_capacitance_for_diagonal_root_v1": "capability",
        "scaffold_unavailable": "capability",
        "port_realizability": "validation",
        "invalid_diagonal_root_hint": "validation",
        "invalid_optimization_spec": "validation",
        "direct_response_formation": "execution",
        "invalid_candidate_physical_parameter": "execution",
        "compiler_invariant": "execution",
        "eliminated_block_solve_failure": "execution",
        "root_slope_unresolved": "execution",
        "numerical_resolution_unresolved": "execution",
        "runtime_preparation": "execution",
        "backend_protocol": "execution",
        "optimization_progress_callback": "execution",
        "result_unavailable": "evidence",
        "evidence_integrity": "evidence",
    }
    kind = value.get("kind")
    contexts = {
        "authoring", "workspace", "runtime", "compile", "direct_response",
        "direct_quantity", "optimization_candidate", "hb_case", "protocol",
        "artifact", "resolution", "scaffold",
    }
    if (
        not isinstance(evidence, dict)
        or not {"type", "operation", "context_kind"}.issubset(evidence)
        or not set(evidence).issubset(allowed)
        or evidence.get("type") != "failure_evidence"
        or evidence.get("operation") not in {operation, "backend_protocol"}
        or evidence.get("context_kind") not in contexts
        or kind not in categories
        or value.get("category") != categories.get(kind)
        or not isinstance(value.get("stage"), str)
        or not value["stage"]
        or not isinstance(value.get("message"), str)
        or not value["message"]
    ):
        raise _integrity("Failure discriminator or evidence is malformed.")
    context = evidence.get("optimization_context")
    if _optimization_failure_requires_context(value, operation) and context is None:
        raise _integrity("Optimization execution failure lacks its phase context.")
    if context is not None:
        if operation != "optimize_direct" or evidence.get("operation") != "optimize_direct":
            raise _integrity("Non-optimization failure carries optimization context.")
        _verify_optimization_context_shape(context)

def _verify_selector(value: object, plan: Mapping[str, object]) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise _integrity("Optimization selector is malformed.")
    kind = value.get("type")
    if kind == "quantity_sum":
        if set(value) != {"type", "terms"} or not isinstance(value.get("terms"), list) or not value["terms"]:
            raise _integrity("QuantitySum is malformed.")
        roles = [_verify_selector(item, plan) for item in value["terms"]]
        if any(role[1] != roles[0][1] for role in roles[1:]):
            raise _integrity("QuantitySum terms have incompatible physical roles.")
        return roles[0]
    if kind == "quantity_difference":
        if set(value) != {"type", "left", "right"}:
            raise _integrity("QuantityDifference is malformed.")
        roles = [_verify_selector(value[side], plan) for side in ("left", "right")]
        if roles[0][1] != roles[1][1]:
            raise _integrity("QuantityDifference operands have incompatible physical roles.")
        return roles[0]
    if kind == "quantity_absolute":
        if set(value) != {"type", "operand"}:
            raise _integrity("QuantityAbsolute is malformed.")
        return _verify_selector(value["operand"], plan)
    fields = {"type", "spec", "projection", "view"}
    kind = value.get("type")
    expected = {
        "diagonal_root_projection": ("diagonal_root", {"frequency", "linewidth"}, ("hertz", "inverse_time")),
        "operator_element_root_projection": ("operator_element_root", {"frequency"}, ("hertz", "inverse_time")),
        "hybridized_pole_projection": ("hybridized_pole", {"frequency", "linewidth"}, ("hertz", "inverse_time")),
        "transfer_zero_projection": ("transfer_zero", {"frequency"}, ("hertz", "inverse_time")),
        "residue_coupling_projection": ("residue_normalized_coupling", {"real", "imag", "magnitude"}, ("radian / second", "inverse_time")),
        "response_element_projection": ("response_element", {"magnitude", "real", "imag"}, None),
    }.get(kind)
    if set(value) != fields or expected is None or value.get("projection") not in expected[1] or not isinstance(value.get("spec"), dict) or value["spec"].get("type") != expected[0]:
        raise _integrity("Optimization selector is outside the Direct catalog.")
    terminal, port_realizable = _verify_view_declaration(value.get("view"), plan)
    _verify_v1_evaluation_spec(value["spec"], terminal, port_realizable)
    if expected[2] is not None:
        return expected[2]
    family = value["spec"].get("family")
    return {"S": ("dimensionless", "dimensionless"), "Y": ("siemens", "conductance"), "Z": ("ohm", "resistance")}[family]

def _plan_coordinates(plan: Mapping[str, object]) -> tuple[list[str], set[str]]:
    """Return the snapshot-owned compiler basis and public subset."""

    connectivity = plan.get("connectivity")
    nodes = connectivity.get("node_coordinates") if isinstance(connectivity, Mapping) else None
    if not isinstance(nodes, list) or not nodes:
        raise _integrity("Sealed Plan coordinate inventory is malformed.")
    order: list[str] = []
    public: set[str] = set()
    for node in nodes:
        if (
            not isinstance(node, Mapping)
            or set(node) != {"final_net", "compiler_node_id", "visibility", "public_aliases"}
            or not isinstance(node.get("compiler_node_id"), str)
            or not node["compiler_node_id"]
            or node.get("compiler_node_id") != node.get("final_net")
            or node.get("visibility") not in {"public", "internal"}
            or not isinstance(node.get("public_aliases"), list)
        ):
            raise _integrity("Sealed Plan node inventory is malformed.")
        compiler_id = node["compiler_node_id"]
        if compiler_id in order:
            raise _integrity("Sealed Plan compiler coordinates are not unique.")
        order.append(compiler_id)
        if node["visibility"] == "public":
            if not node["public_aliases"]:
                raise _integrity("Public compiler coordinate has no public alias.")
            public.add(compiler_id)
        elif node["public_aliases"]:
            raise _integrity("Internal compiler coordinate exposes a public alias.")
    return order, public

def _lineage_matrix(label: str, values: list[list[float]], applicability: str) -> dict[str, object]:
    rows = len(values)
    columns = len(values[0]) if rows else 0
    bits = [struct.pack(">d", value).hex() for row in values for value in row]
    digest = _sha256(_canonical_bytes({
        "schema": "scnsim.lineage_matrix",
        "schema_version": 1,
        "label": label,
        "applicability": applicability,
        "shape": [rows, columns],
        "row_major_f64": bits,
    }))
    return {"rows": rows, "columns": columns, "sha256": digest}

def _selector_terms(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Mapping):
        raise _integrity("Optimization objective quantity is malformed.")
    if value.get("type") == "quantity_sum":
        terms = value.get("terms")
        if not isinstance(terms, list) or not terms or any(not isinstance(term, Mapping) for term in terms):
            raise _integrity("Optimization QuantitySum terms are malformed.")
        return [leaf for term in terms for leaf in _selector_terms(term)]
    if value.get("type") == "quantity_difference":
        if set(value) != {"type", "left", "right"}:
            raise _integrity("Optimization QuantityDifference is malformed.")
        return _selector_terms(value["left"]) + _selector_terms(value["right"])
    if value.get("type") == "quantity_absolute":
        if set(value) != {"type", "operand"}:
            raise _integrity("Optimization QuantityAbsolute is malformed.")
        return _selector_terms(value["operand"])
    return [value]

def _verify_selector_lineage(
    selector: Mapping[str, object],
    lineage: object,
    plan: Mapping[str, object],
) -> None:
    declaration = selector.get("view")
    declared_terminal, declared_port_realizable = _verify_view_declaration(declaration, plan)
    terminal, port_realizable = _verify_v1_lineage(lineage, plan)
    if terminal != declared_terminal or port_realizable != declared_port_realizable:
        raise _integrity("Optimization term lineage disagrees with its declared View.")
    if not isinstance(declaration, Mapping) or not isinstance(lineage, Mapping):
        raise _integrity("Optimization term View evidence is malformed.")
    declared_ptc = declaration.get("ptc")
    actual_ptc = lineage.get("ptc")
    if (None if actual_ptc is None else {"selected_ports": actual_ptc.get("selected_ports")}) != declared_ptc:
        raise _integrity("Optimization term PTC lineage disagrees with its declaration.")
    declared_transforms = declaration.get("transforms")
    actual_transforms = lineage.get("transforms")
    if not isinstance(declared_transforms, list) or not isinstance(actual_transforms, list) or len(actual_transforms) != len(declared_transforms):
        raise _integrity("Optimization term transform lineage count is inconsistent.")
    for declared, actual in zip(declared_transforms, actual_transforms):
        if (
            not isinstance(declared, Mapping)
            or not isinstance(actual, Mapping)
            or actual.get("input_coordinates") != declared.get("input_coordinates")
            or [actual.get("common_id"), actual.get("differential_id")] != declared.get("output_coordinates")
        ):
            raise _integrity("Optimization term transform lineage disagrees with its declaration.")
    declared_retain = declaration.get("retain")
    actual_retain = lineage.get("retain")
    if (None if actual_retain is None else {"retained_coordinates": actual_retain.get("retained_coordinates")}) != declared_retain:
        raise _integrity("Optimization term retain lineage disagrees with its declaration.")

def _optimization_failure_context(failure: object) -> Mapping[str, object]:
    if not isinstance(failure, Mapping) or not isinstance(failure.get("evidence"), Mapping):
        raise _integrity("Optimization failure lacks evidence.")
    context = failure["evidence"].get("optimization_context")
    return _verify_optimization_context_shape(context)
