"""Harmonic-balance result payload verification."""

from __future__ import annotations

import re
from collections.abc import Mapping

from ...canonical import canonical_json_bytes as _canonical_bytes, sha256_hex as _sha256
from .common import (
    _IDENTIFIER,
    _f64_value,
    _finite_f64,
    _identifiers,
    _integrity,
    _valid_sha,
    _verify_quantity_role,
)
from .requests import (
    _hb_declared_modes_from_spec,
    _hb_operating_lattice_is_vacuous,
    _lineage_matrix,
    _valid_mode_tuple,
    _verify_v1_lineage,
)
from .result_artifacts import _expected_probe_load_state, _verify_zarr_datasets

def _hb_compiled_node_order(
    plan: Mapping[str, object], original_coordinates: list[str], discretization: object,
) -> list[str]:
    """Independently reproduce the raw compiler basis for exact HB Port hashes."""

    connectivity = plan.get("connectivity")
    declared = connectivity.get("node_coordinates") if isinstance(connectivity, Mapping) else None
    if not isinstance(declared, list) or [row.get("compiler_node_id") for row in declared if isinstance(row, Mapping)] != original_coordinates or len(declared) != len(original_coordinates):
        raise _integrity("HB public coordinate order disagrees with the sealed Plan.")
    nodes = list(original_coordinates)
    seen = set(nodes)
    leaves = plan.get("physical_leaves")
    if not isinstance(leaves, list):
        raise _integrity("HB sealed Plan has no physical leaf inventory.")
    line_rows = iter(discretization) if isinstance(discretization, list) else None
    for leaf in leaves:
        if not isinstance(leaf, Mapping) or leaf.get("model") != "transmission_line":
            continue
        metadata = leaf.get("model_metadata")
        pins = leaf.get("pin_order")
        path = leaf.get("path")
        if not isinstance(metadata, Mapping) or not isinstance(pins, list) or not isinstance(path, list) or len(pins) % 2:
            raise _integrity("HB transmission-line declaration is malformed.")
        conductors = [pin.removeprefix("head.") for pin in pins[: len(pins) // 2] if isinstance(pin, str) and pin.startswith("head.")]
        if not conductors or len(conductors) * 2 != len(pins) or pins != [*(f"head.{name}" for name in conductors), *(f"tail.{name}" for name in conductors)]:
            raise _integrity("HB line conductor order disagrees with its sealed Pins.")
        row = next(line_rows) if line_rows is not None else None
        sections = row.get("n_sections") if isinstance(row, Mapping) else metadata.get("n_sections")
        if not isinstance(sections, int) or isinstance(sections, bool) or sections < 1:
            raise _integrity("HB line grid has no valid section count.")
        for station in range(1, sections):
            for conductor in conductors:
                identity: dict[str, object] = {
                    "schema": "scnsim.line_station", "schema_version": 1,
                    "component_path": path, "station": station, "conductor": conductor,
                }
                if "discretization" in metadata:
                    identity["n_sections"] = sections
                node = "internal-" + _sha256(_canonical_bytes(identity))
                if node in seen:
                    raise _integrity("HB compiled line station identity collides.")
                seen.add(node)
                nodes.append(node)
    return nodes

def _verify_hb_batch_result(
    result: Mapping[str, object],
    request: Mapping[str, object],
    plan: Mapping[str, object],
    *,
    discretization: object,
) -> None:
    """Verify the case-local HB Result catalog before receipt promotion.

    HB artifacts are semantic catalog records on successful cases.  Their
    manifest hashes attest only to the byte trees; receipt links retain the
    case-local semantic key so repeated role names in separate cases cannot
    collapse into a global artifact namespace.
    """

    common = {"schema", "schema_version", "result_kind", "request_sha256", "attempt_sha256"}
    expected = common | {"lattice", "truncation", "topology_evidence", "cases"}
    if set(result) != expected:
        raise _integrity("HB batch Result envelope is open or incomplete.")
    spec = request.get("spec")
    if not isinstance(spec, dict):
        raise _integrity("HB batch Result has no solve Spec.")
    if result.get("truncation") != spec.get("truncation"):
        raise _integrity("HB Result truncation disagrees with its request.")
    lattice = result.get("lattice")
    lattice_fields = {
        "pump_axes", "operating_point_modes", "input_modes", "output_modes",
        "matrix_order", "tuple_frequency_collision_check_sha256",
    }
    if not isinstance(lattice, dict) or set(lattice) != lattice_fields or lattice.get("pump_axes") != spec.get("pump_axes") or lattice.get("matrix_order") != "port_major_mode_minor":
        raise _integrity("HB lattice evidence is malformed or disagrees with its request.")
    from ...canonical import float64_hex

    pump_rank = len(spec.get("pump_axes", []))
    pump_axes = spec.get("pump_axes")
    frequencies = spec.get("frequencies")
    if not isinstance(pump_axes, list) or not isinstance(frequencies, list):
        raise _integrity("HB request cannot reproduce its lattice frequencies.")
    pump_frequencies = [_f64_value(axis["frequency"]["si_value_f64"]) for axis in pump_axes]
    response_frequencies = [_f64_value(frequency["si_value_f64"]) for frequency in frequencies]
    vacuous_operating_lattice = _hb_operating_lattice_is_vacuous(spec)
    for field, is_response_lattice in (("operating_point_modes", False), ("input_modes", True), ("output_modes", True)):
        modes = lattice.get(field)
        if not isinstance(modes, list):
            raise _integrity("HB lattice is missing an ordered mode basis.", field=field)
        expected_modes = _hb_declared_modes_from_spec(spec, response=is_response_lattice)
        if field == "operating_point_modes" and vacuous_operating_lattice:
            if modes:
                raise _integrity("HB operating lattice is nonempty for a vacuous pinned JC basis.")
            continue
        if not modes:
            raise _integrity("HB lattice is missing an ordered mode basis.", field=field)
        seen: set[tuple[int, ...]] = set()
        for order, item in enumerate(modes):
            keys = {"mode", "signed_frequency", "order"} if not is_response_lattice else {"mode", "signed_frequency_grid", "order"}
            if not isinstance(item, dict) or set(item) != keys or not _valid_mode_tuple(item.get("mode"), pump_rank) or item.get("order") != order:
                raise _integrity("HB lattice mode row is malformed.", field=field)
            mode = tuple(item["mode"])
            if mode in seen:
                raise _integrity("HB lattice repeats a mode tuple.", field=field)
            seen.add(mode)
        if [item["mode"] for item in modes] != expected_modes:
            raise _integrity("HB lattice mode order disagrees with pinned JosephsonCircuits construction.", field=field)
        _verify_hb_lattice_injectivity(modes, response=is_response_lattice, field=field)
        for item in modes:
            mode = tuple(item["mode"])
            values = item.get("signed_frequency_grid") if is_response_lattice else [item.get("signed_frequency")]
            expected_values = (
                [frequency + sum((float(coefficient) * pump for coefficient, pump in zip(mode, pump_frequencies)), 0.0) for frequency in response_frequencies]
                if is_response_lattice else [sum((float(coefficient) * pump for coefficient, pump in zip(mode, pump_frequencies)), 0.0)]
            )
            if not isinstance(values, list) or len(values) != len(expected_values):
                raise _integrity("HB lattice frequency evidence is malformed.", field=field)
            for value, expected_value in zip(values, expected_values):
                _verify_quantity_role(value, complex_value=False, unit="hertz", dimensionality="inverse_time")
                if value["si_value_f64"] != float64_hex(expected_value):
                    raise _integrity("HB lattice signed frequency disagrees with its sealed axes.", field=field)
                if is_response_lattice and expected_value == 0.0:
                    raise _integrity("HB response lattice contains a zero-frequency sideband.", field=field)
    input_modes = lattice["input_modes"]
    if lattice.get("output_modes") != input_modes:
        raise _integrity("HB input and output response lattices disagree.")
    collision_entries = [
        {"mode": row["mode"], "frequency": value["si_value_f64"]}
        for row in input_modes
        for value in row["signed_frequency_grid"]
    ]
    expected_collision = _sha256(_canonical_bytes({
        "schema": "scnsim.hb_tuple_frequency_collision", "schema_version": 1,
        "entries": collision_entries,
    }))
    if lattice.get("tuple_frequency_collision_check_sha256") != expected_collision:
        raise _integrity("HB tuple-frequency collision evidence disagrees with the sealed lattice.")
    cases = result.get("cases")
    declared_cases = spec.get("cases")
    if not isinstance(cases, list) or not isinstance(declared_cases, list) or len(cases) != len(declared_cases):
        raise _integrity("HB Result case inventory disagrees with its declaration.")
    lineage = request.get("ref_lineage")
    if not isinstance(lineage, Mapping):
        raise _integrity("HB batch Result has no realized View lineage.")
    terminal, _ = _verify_v1_lineage(lineage, plan)
    _verify_hb_topology_evidence(result.get("topology_evidence"), spec, lineage)
    original = lineage.get("original") if isinstance(lineage, Mapping) else None
    native_ports = _identifiers(original.get("port_order"), field="HB original Port order", nonempty=False) if isinstance(original, Mapping) else []
    original_coordinates = _identifiers(
        original.get("coordinate_order"), field="HB original coordinate order"
    ) if isinstance(original, Mapping) else []
    compiled_nodes = _hb_compiled_node_order(plan, original_coordinates, discretization)
    connectivity = plan.get("connectivity")
    plan_ports = connectivity.get("ports") if isinstance(connectivity, Mapping) else None
    if not isinstance(plan_ports, list):
        raise _integrity("HB sealed Plan has no Port inventory.")
    expected_injection_sha256: dict[str, str] = {}
    for port in plan_ports:
        if (
            not isinstance(port, Mapping)
            or not isinstance(port.get("id"), str)
            or not isinstance(port.get("net"), str)
            or port["id"] in expected_injection_sha256
            or port["net"] not in original_coordinates
        ):
            raise _integrity("HB sealed Port cannot reproduce its compiler injection map.")
        incidence = [0.0] * len(compiled_nodes)
        incidence[compiled_nodes.index(port["net"])] = 1.0
        expected_injection_sha256[port["id"]] = _sha256(
            _canonical_bytes(
                {
                    "schema": "scnsim.hb_injection_map",
                    "schema_version": 1,
                    "port_id": port["id"],
                    "incidence_f64": [float64_hex(item) for item in incidence],
                }
            )
        )
    expected_probe = _expected_probe_load_state(lineage)
    native_probe = [{"port_id": port, "state": "raw"} for port in native_ports]
    operating_modes = [item["mode"] for item in lattice["operating_point_modes"]]
    for ordinal, (outcome, declared) in enumerate(zip(cases, declared_cases), 1):
        if not isinstance(outcome, dict) or not isinstance(declared, dict) or outcome.get("case_ordinal") != ordinal or outcome.get("case_id") != declared.get("id"):
            raise _integrity("HB Result cases are not declaration ordered.")
        _verify_hb_effective_sources(
            outcome.get("effective_sources"),
            pump_rank,
            spec=spec,
            declared_case=declared,
            expected_injection_sha256=expected_injection_sha256,
            operating_modes=operating_modes,
        )
        status = outcome.get("status")
        if status == "failure":
            if set(outcome) != {"case_ordinal", "case_id", "status", "effective_sources", "failure"}:
                raise _integrity("Failed HB outcome leaks success-only evidence.")
            failure = outcome.get("failure")
            if not isinstance(failure, dict) or set(failure) != {"kind", "stage", "message", "evidence_sha256"} or failure.get("kind") != "hb_case_failure" or failure.get("stage") not in {"operating_point", "linearization", "response_formation"} or not isinstance(failure.get("message"), str) or not failure["message"]:
                raise _integrity("HB case failure is malformed.")
            expected_failure_evidence = _sha256(
                _canonical_bytes(
                    {
                        "schema": "scnsim.hb_case_failure",
                        "schema_version": 1,
                        "case_ordinal": ordinal,
                        "case_id": declared["id"],
                        "stage": failure["stage"],
                        "message": failure["message"],
                        "effective_sources": outcome["effective_sources"],
                    }
                )
            )
            if failure.get("evidence_sha256") != expected_failure_evidence:
                raise _integrity("HB case failure evidence disagrees with its sealed outcome.")
            continue
        if status != "success":
            raise _integrity("HB case has an unknown terminal status.")
        success_fields = {
            "case_ordinal", "case_id", "status", "bias_state", "pump_state",
            "effective_sources", "operating_point_closure", "artifacts", "traces", "reconciliation",
            "backend_normalization_evidence_sha256", "state_node_map",
        }
        if set(outcome) != success_fields or outcome.get("bias_state") not in {"off", "on"} or outcome.get("pump_state") not in {"off", "on"}:
            raise _integrity("Successful HB outcome is open or malformed.")
        expected_normalization_evidence = _sha256(
            _canonical_bytes(
                {"normalization": "backend_photon_flux_to_scnsim_power_wave"}
            )
        )
        if outcome.get("backend_normalization_evidence_sha256") != expected_normalization_evidence:
            raise _integrity("HB backend normalization evidence is not reproducible.")
        _verify_hb_operating_point_closure(outcome.get("operating_point_closure"), operating_modes)
        _verify_hb_reconciliation(outcome.get("reconciliation"), lineage)
        _verify_hb_state_node_map(outcome.get("state_node_map"))
        _verify_hb_case_catalog(
            outcome.get("artifacts"), outcome.get("traces"), ordinal,
            spec, lattice, outcome["state_node_map"], terminal,
            expected_probe, native_ports, native_probe,
        )

def _verify_hb_effective_sources(
    value: object,
    pump_rank: int,
    *,
    spec: Mapping[str, object],
    declared_case: Mapping[str, object],
    expected_injection_sha256: Mapping[str, str],
    operating_modes: list[list[int]],
) -> None:
    drives = spec.get("drives")
    bindings = declared_case.get("currents")
    if not isinstance(value, list) or not isinstance(drives, list) or not isinstance(bindings, list) or len(value) != len(drives):
        raise _integrity("HB outcome has no effective-source evidence.")
    by_drive: dict[str, Mapping[str, object]] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping) or not isinstance(binding.get("drive_id"), str) or binding["drive_id"] in by_drive:
            raise _integrity("HB case current declaration is malformed.")
        by_drive[binding["drive_id"]] = binding
    from ...canonical import float64_hex

    for source, drive in zip(value, drives):
        if not isinstance(source, dict) or set(source) != {"drive_id", "mode", "coefficient", "generated_conjugate", "backend_binding", "injection_map_sha256"} or not isinstance(source.get("drive_id"), str) or _IDENTIFIER.fullmatch(source["drive_id"]) is None or not _valid_mode_tuple(source.get("mode"), pump_rank):
            raise _integrity("HB effective-source row is malformed.")
        if not isinstance(drive, Mapping) or source.get("drive_id") != drive.get("id") or source.get("mode") != drive.get("mode"):
            raise _integrity("HB effective sources are not in drive declaration order.")
        _verify_quantity_role(source.get("coefficient"), complex_value=True, unit="ampere", dimensionality="current")
        binding = by_drive.pop(source["drive_id"], None)
        coefficient = source["coefficient"]
        if binding is None:
            if _f64_value(coefficient["real_si_f64"]) != 0.0 or _f64_value(coefficient["imag_si_f64"]) != 0.0:
                raise _integrity("An omitted HB current did not materialize as exact zero.")
        elif coefficient != binding.get("coefficient"):
            raise _integrity("HB effective-source coefficient disagrees with its case declaration.")
        source_mode = list(source["mode"])
        inverse_mode = [-value for value in source_mode]
        is_dc = all(value == 0 for value in source_mode)
        expected_generated = (
            {"mode": source_mode, "coefficient": coefficient}
            if is_dc else {
                "mode": inverse_mode,
                "coefficient": {
                    "type": "complex_quantity_f64",
                    "real_si_f64": coefficient["real_si_f64"],
                    "imag_si_f64": float64_hex(-_f64_value(coefficient["imag_si_f64"])),
                    "si_unit": "ampere",
                    "dimensionality": "current",
                },
            }
        )
        if source.get("generated_conjugate") != expected_generated:
            raise _integrity("HB generated conjugate disagrees with its declared coefficient.")
        if is_dc:
            expected_representative = [0] if pump_rank == 0 else source_mode
            backend_coefficient = coefficient
        elif source_mode in operating_modes:
            expected_representative = source_mode
            backend_coefficient = {
                "type": "complex_quantity_f64",
                "real_si_f64": coefficient["real_si_f64"],
                "imag_si_f64": float64_hex(-_f64_value(coefficient["imag_si_f64"])),
                "si_unit": "ampere",
                "dimensionality": "current",
            }
        elif inverse_mode in operating_modes:
            expected_representative = inverse_mode
            backend_coefficient = coefficient
        else:
            raise _integrity("HB source mode and its generated conjugate are absent from the operating lattice.")
        try:
            representative_index = operating_modes.index(source_mode if pump_rank == 0 else expected_representative)
        except ValueError as error:
            raise _integrity("HB source representative is absent from the operating lattice.") from error
        expected_backend = {
            "representative_mode": expected_representative,
            "representative_index": representative_index,
            "coefficient": backend_coefficient,
            "coefficient_convention": "exp_plus_i_m_dot_omega_t_josephsoncircuits_source",
        }
        if source.get("backend_binding") != expected_backend:
            raise _integrity("HB backend source binding disagrees with the sealed case and lattice.")
        expected_injection = expected_injection_sha256.get(str(drive.get("port_id")))
        if expected_injection is None or source.get("injection_map_sha256") != expected_injection:
            raise _integrity("HB effective-source injection map disagrees with the sealed compiler basis.")
    if by_drive:
        raise _integrity("HB case current names a drive absent from effective sources.")

def _verify_hb_lattice_injectivity(
    modes: list[object], *, response: bool, field: str,
) -> None:
    """Reject a non-injective tuple/frequency channel basis by exact bits."""

    if response:
        grid_length: int | None = None
        for item in modes:
            grid = item.get("signed_frequency_grid") if isinstance(item, Mapping) else None
            if not isinstance(grid, list):
                raise _integrity("HB response lattice frequency grid is malformed.", field=field)
            if grid_length is None:
                grid_length = len(grid)
            elif len(grid) != grid_length:
                raise _integrity("HB response lattice frequency grids disagree in length.", field=field)
        if grid_length is None:
            raise _integrity("HB response lattice has no mode rows.", field=field)
        for frequency_ordinal in range(grid_length):
            seen_frequencies: set[str] = set()
            for item in modes:
                grid = item["signed_frequency_grid"]
                frequency = grid[frequency_ordinal]
                if not isinstance(frequency, Mapping) or not isinstance(frequency.get("si_value_f64"), str):
                    raise _integrity("HB response lattice frequency evidence is malformed.", field=field)
                bits = frequency["si_value_f64"]
                if bits in seen_frequencies:
                    raise _integrity("HB response lattice has a duplicate signed frequency at one declared grid ordinal.", field=field)
                seen_frequencies.add(bits)
        return
    seen_frequencies: set[str] = set()
    for item in modes:
        frequency = item.get("signed_frequency") if isinstance(item, Mapping) else None
        if not isinstance(frequency, Mapping) or not isinstance(frequency.get("si_value_f64"), str):
            raise _integrity("HB operating lattice frequency evidence is malformed.", field=field)
        bits = frequency["si_value_f64"]
        if bits in seen_frequencies:
            raise _integrity("HB operating lattice has a duplicate signed frequency.", field=field)
        seen_frequencies.add(bits)

def _verify_hb_topology_evidence(
    value: object,
    spec: Mapping[str, object],
    lineage: Mapping[str, object],
) -> None:
    """Bind HB's loaded nonlinear and selected response topologies to one View."""

    original = lineage.get("original")
    if not isinstance(original, Mapping):
        raise _integrity("HB topology evidence has no original compiler lineage.")
    intrinsic = _valid_sha(original.get("compiled_graph_sha256"))
    full_lineage = _valid_sha(lineage.get("lineage_sha256"))
    balance_lineage = _hb_lineage_prefix_sha(lineage, "load_or_ptc")
    expected = {
        "allow_driven_ptc": spec.get("allow_driven_ptc"),
        "intrinsic_compiled_graph_sha256": intrinsic,
        "nonlinear_balance": {
            "load_state": "loaded",
            "lineage_sha256": balance_lineage,
        },
        "response_linearization": {
            "load_state": "compensated" if lineage.get("ptc") is not None else "raw",
            "lineage_sha256": full_lineage,
        },
    }
    if value != expected:
        raise _integrity("HB topology evidence disagrees with its sealed View and driven-PTC authorization.")

def _verify_hb_operating_point_closure(value: object, operating_modes: list[list[int]]) -> None:
    """Verify the fixed HB residual disjunction or the exact vacuous exception."""

    if not operating_modes:
        if value != {"status": "not_applicable", "reason": "no_operating_point_lattice"}:
            raise _integrity("Vacuous HB operating lattice has the wrong closure evidence.")
        return
    if not isinstance(value, Mapping) or set(value) != {
        "status", "absolute_residual_f64", "relative_residual", "successful_disjunct",
    } or value.get("status") != "satisfied":
        raise _integrity("HB operating-point closure is malformed.")
    absolute = value.get("absolute_residual_f64")
    if not _finite_f64(absolute) or _f64_value(absolute) < 0.0:
        raise _integrity("HB operating-point absolute residual is malformed.")
    absolute_passes = _f64_value(absolute) <= 1.0e-8
    relative = value.get("relative_residual")
    relative_passes = False
    if isinstance(relative, Mapping) and set(relative) == {"status", "value_f64"} and relative.get("status") == "value":
        relative_value = relative.get("value_f64")
        if not _finite_f64(relative_value) or _f64_value(relative_value) < 0.0:
            raise _integrity("HB operating-point relative residual is malformed.")
        relative_passes = _f64_value(relative_value) < 1.0e-8
    elif not (
        isinstance(relative, Mapping)
        and relative == {"status": "not_applicable", "reason": "zero_state_norm"}
    ):
        raise _integrity("HB operating-point relative residual is malformed.")
    disjunct = value.get("successful_disjunct")
    expected_disjunct = (
        "both" if absolute_passes and relative_passes
        else "absolute" if absolute_passes
        else "relative" if relative_passes
        else None
    )
    if disjunct != expected_disjunct:
        raise _integrity("HB operating-point closure does not satisfy the fixed residual disjunction.")

def _verify_hb_reconciliation(value: object, lineage: Mapping[str, object]) -> None:
    fields = {"comparable", "reason", "last_comparable_ancestor", "normalization", "evidence_sha256"}
    if not isinstance(value, dict) or not fields.issubset(value) or not set(value).issubset(fields | {"residual_f64", "coordinate_projection"}) or not isinstance(value.get("comparable"), bool) or value.get("normalization") != "backend_photon_flux_to_scnsim_power_wave":
        raise _integrity("HB reconciliation evidence is malformed.")
    _valid_sha(value.get("last_comparable_ancestor")); _valid_sha(value.get("evidence_sha256"))
    comparable = value["comparable"]
    expected_reason = _hb_reconciliation_reason(lineage)
    expected_ancestor = _hb_lineage_prefix_sha(lineage, expected_reason)
    if value.get("last_comparable_ancestor") != expected_ancestor:
        raise _integrity("HB reconciliation ancestor does not bind the actual lineage prefix.")
    if comparable:
        residual = value.get("residual_f64")
        projection = _verify_hb_coordinate_projection(value.get("coordinate_projection"), lineage)
        if expected_reason is not None or value.get("reason") is not None or not _finite_f64(residual) or _f64_value(residual) < 0.0:
            raise _integrity("Comparable HB reconciliation lacks its normalized residual.")
        expected_evidence = _sha256(_canonical_bytes({
            "coordinate_producer_sha256": _hb_coordinate_producer_sha(lineage),
            "coordinate_projection": projection,
            "residual_f64": residual,
        }))
        if value.get("evidence_sha256") != expected_evidence:
            raise _integrity("HB comparable reconciliation evidence does not bind its coordinate producer and residual.")
    else:
        if "residual_f64" in value or "coordinate_projection" in value or value.get("reason") != expected_reason:
            raise _integrity("Incomparable HB reconciliation is malformed.")
        expected_evidence = _sha256(_canonical_bytes({
            "reason": expected_reason,
            "last_comparable_ancestor": expected_ancestor,
        }))
        if value.get("evidence_sha256") != expected_evidence:
            raise _integrity("HB incomparable reconciliation evidence does not bind its reason and lineage prefix.")

def _hb_reconciliation_reason(lineage: Mapping[str, object]) -> str | None:
    original = lineage.get("original")
    retain = lineage.get("retain")
    ptc = lineage.get("ptc")
    transforms = lineage.get("transforms")
    if not isinstance(original, Mapping) or not isinstance(transforms, list):
        raise _integrity("HB reconciliation cannot reconstruct its lineage.")
    ports = _identifiers(original.get("port_order"), field="HB reconciliation original Ports", nonempty=False)
    plain_port_subset = (
        isinstance(retain, Mapping)
        and ptc is None
        and not transforms
        and isinstance(retain.get("retained_coordinates"), list)
        and all(value in ports for value in retain["retained_coordinates"])
    )
    if ptc is not None:
        return "load_or_ptc"
    if transforms:
        return "reference_plane"
    if retain is not None and not plain_port_subset:
        return "channel_basis"
    return None

def _hb_lineage_prefix_sha(lineage: Mapping[str, object], reason: str | None) -> str:
    """Rebuild Julia's longest-comparable canonical lineage prefix exactly."""

    original = lineage.get("original")
    if not isinstance(original, Mapping):
        raise _integrity("HB reconciliation lineage has no original step.")
    if reason is None or reason in {"reference_matrix", "normalization", "signed_frequency_grid"}:
        return _valid_sha(lineage.get("lineage_sha256"))
    terminal = _identifiers(original.get("port_order"), field="HB reconciliation original Ports", nonempty=False)
    prefix: dict[str, object] = {
        "type": "network_view_lineage",
        "original": dict(original),
        "ptc": lineage.get("ptc") if reason in {"reference_plane", "channel_basis"} else None,
        "transforms": list(lineage.get("transforms", [])) if reason == "channel_basis" else [],
        "retain": None,
        "terminal_coordinates": terminal,
        "port_realizable": original.get("port_realizable"),
    }
    prefix["lineage_sha256"] = _sha256(_canonical_bytes(prefix))
    return str(prefix["lineage_sha256"])

def _hb_coordinate_producer_sha(lineage: Mapping[str, object]) -> str:
    """Return the sealed source identity of a comparable selected Port map."""

    retain = lineage.get("retain")
    if isinstance(retain, Mapping):
        q_matrix = retain.get("q_matrix")
        if not isinstance(q_matrix, Mapping):
            raise _integrity("HB comparable retain lineage has no Q-matrix evidence.")
        return _valid_sha(q_matrix.get("sha256"))
    original = lineage.get("original")
    if not isinstance(original, Mapping):
        raise _integrity("HB comparable lineage has no original mapping identity.")
    return _valid_sha(original.get("compiled_graph_sha256"))

def _verify_hb_coordinate_projection(value: object, lineage: Mapping[str, object]) -> dict[str, object]:
    """Bind the response-side Q row map to its selected and native bases."""

    if not isinstance(value, Mapping) or set(value) != {"shape", "values_f64"}:
        raise _integrity("HB comparable reconciliation has no closed coordinate projection.")
    shape = value.get("shape")
    bits = value.get("values_f64")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in shape)
        or not isinstance(bits, list)
        or len(bits) != shape[0] * shape[1]
        or any(not _finite_f64(item) for item in bits)
    ):
        raise _integrity("HB comparable coordinate projection shape or values are malformed.")
    from ...canonical import float64_hex

    if any(float64_hex(_f64_value(item)) != item for item in bits):
        raise _integrity("HB comparable coordinate projection has noncanonical Float64 values.")
    original = lineage.get("original")
    if not isinstance(original, Mapping):
        raise _integrity("HB comparable reconciliation has no original View basis.")
    ports = _identifiers(original.get("port_order"), field="HB comparable original Ports", nonempty=False)
    retain = lineage.get("retain")
    terminal = _identifiers(lineage.get("terminal_coordinates"), field="HB comparable terminal coordinates")
    expected_shape = [len(terminal), len(ports)]
    if shape != expected_shape:
        raise _integrity("HB comparable coordinate projection does not span its selected and original Port bases.")
    normalized = {"shape": list(shape), "values_f64": list(bits)}
    if retain is None:
        expected_identity = [float64_hex(1.0 if row == column else 0.0) for row in range(len(ports)) for column in range(len(ports))]
        if terminal != ports or normalized != {"shape": [len(ports), len(ports)], "values_f64": expected_identity}:
            raise _integrity("HB comparable original Port projection is not the canonical identity.")
        return normalized
    if not isinstance(retain, Mapping):
        raise _integrity("HB comparable retain projection has malformed lineage.")
    q_matrix = retain.get("q_matrix")
    if not isinstance(q_matrix, Mapping) or q_matrix.get("rows") != shape[0] or q_matrix.get("columns") != shape[1]:
        raise _integrity("HB comparable projection disagrees with retain Q-matrix dimensions.")
    values = [
        [_f64_value(bits[row * shape[1] + column]) for column in range(shape[1])]
        for row in range(shape[0])
    ]
    expected_q = _lineage_matrix("q", values, "port_realizable")
    if q_matrix.get("sha256") != expected_q["sha256"]:
        raise _integrity("HB comparable projection does not reproduce retain Q-matrix evidence.")
    return normalized

def _verify_hb_state_node_map(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise _integrity("HB success lacks its state-node map.")
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != {"state_index", "compiler_node_id", "source"} or row.get("state_index") != index or not isinstance(row.get("compiler_node_id"), str) or _IDENTIFIER.fullmatch(row["compiler_node_id"]) is None or not isinstance(row.get("source"), dict):
            raise _integrity("HB state-node map is malformed.")
        source = row["source"]
        kind = source.get("kind")
        valid = (
            kind == "plan_node"
            and set(source) == {"kind", "plan_node_id", "visibility"}
            and isinstance(source.get("plan_node_id"), str)
            and _IDENTIFIER.fullmatch(source["plan_node_id"]) is not None
            and source.get("visibility") in {"public", "port_promoted"}
        ) or (
            kind == "component_private"
            and set(source) == {"kind", "component_path", "private_node_id"}
            and isinstance(source.get("component_path"), list)
            and bool(source["component_path"])
            and all(isinstance(segment, str) and _IDENTIFIER.fullmatch(segment) is not None for segment in source["component_path"])
            and isinstance(source.get("private_node_id"), str)
            and _IDENTIFIER.fullmatch(source["private_node_id"]) is not None
        ) or (
            kind == "anonymous_internal"
            and set(source) == {"kind", "internal_node_id"}
            and isinstance(source.get("internal_node_id"), str)
            and re.fullmatch(r"internal-[0-9a-f]{64}", source["internal_node_id"]) is not None
        )
        if not valid:
            raise _integrity("HB state-node source mapping is malformed.")

def _verify_hb_case_catalog(
    artifacts: object,
    traces: object,
    ordinal: int,
    spec: Mapping[str, object],
    lattice: Mapping[str, object],
    state_node_map: list[dict[str, object]],
    terminal: list[str],
    expected_probe: list[dict[str, str]],
    native_ports: list[str],
    native_probe: list[dict[str, str]],
) -> None:
    roles = ("s", "y", "z", "backend_native_s", "backend_native_z", "states", "effective_source_vectors")
    if not isinstance(artifacts, dict) or set(artifacts) != set(roles) or not isinstance(traces, list):
        raise _integrity("HB case artifact catalog is incomplete.")
    input_modes = [item["mode"] for item in lattice["input_modes"]]
    output_modes = [item["mode"] for item in lattice["output_modes"]]
    operating_modes = [item["mode"] for item in lattice["operating_point_modes"]]
    compiler_nodes = [item["compiler_node_id"] for item in state_node_map]
    for role in roles:
        native = role in {"backend_native_s", "backend_native_z"}
        _verify_hb_catalog_artifact(
            artifacts[role], ordinal, role, spec,
            native_ports if native else terminal,
            native_probe if native else expected_probe,
            input_modes=input_modes,
            output_modes=output_modes,
            operating_modes=operating_modes,
            compiler_nodes=compiler_nodes,
        )
    declared_traces = spec.get("traces")
    if not isinstance(declared_traces, list) or len(traces) != len(declared_traces):
        raise _integrity("HB trace catalog disagrees with declaration.")
    for artifact, declaration in zip(traces, declared_traces):
        if not isinstance(declaration, dict) or not isinstance(artifact, dict) or artifact.get("id") != declaration.get("id"):
            raise _integrity("HB trace catalog is not declaration ordered.")
        _verify_hb_catalog_artifact(
            artifact, ordinal, str(declaration["id"]), spec, terminal,
            expected_probe, trace=True, input_modes=input_modes,
            output_modes=output_modes, operating_modes=operating_modes,
            compiler_nodes=compiler_nodes,
        )

def _verify_hb_catalog_artifact(
    artifact: object,
    ordinal: int,
    role: str,
    spec: Mapping[str, object],
    terminal: list[str],
    expected_probe: list[dict[str, str]],
    *,
    trace: bool = False,
    input_modes: list[list[int]],
    output_modes: list[list[int]],
    operating_modes: list[list[int]],
    compiler_nodes: list[str],
) -> None:
    if not isinstance(artifact, dict):
        raise _integrity("HB artifact catalog entry is malformed.", artifact_id=role)
    base = {"id", "path", "sha256", "media_type", "file_manifest", "dtype", "shape", "chunks", "complex_storage", "group_metadata", "datasets", "axes", "unit", "dimensionality", "chunk_policy"}
    matrix = not trace and role in {"s", "y", "z", "backend_native_s", "backend_native_z"}
    expected_fields = base | ({"coordinate_ids", "probe_load_state", "output_channels", "input_channels"} if matrix else set())
    if set(artifact) != expected_fields or artifact.get("id") != role or artifact.get("media_type") != "application/vnd+zarr-v2" or artifact.get("group_metadata") != {"zarr_format": 2}:
        raise _integrity("HB artifact catalog entry has the wrong semantic role.", artifact_id=role)
    prefix = f"artifacts/cases/{ordinal:06d}/"
    expected_path = f"{prefix}traces/{role}.zarr" if trace else f"{prefix}{role}.zarr"
    expected_manifest = expected_path.removesuffix(".zarr") + ".manifest.json"
    if artifact.get("path") != expected_path or artifact.get("file_manifest") != expected_manifest:
        raise _integrity("HB artifact path does not match its case ordinal.", artifact_id=role)
    _valid_sha(artifact.get("sha256"))
    shape = artifact.get("shape")
    chunks = artifact.get("chunks")
    allow_empty_leading = not trace and role in {"states", "effective_source_vectors"}
    if (
        not isinstance(shape, list)
        or not isinstance(chunks, list)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in chunks)
        or any(value == 0 and (not allow_empty_leading or index != 0) for index, value in enumerate(shape))
    ):
        raise _integrity("HB artifact has invalid shape or chunks.", artifact_id=role)
    if matrix:
        if len(shape) != 3 or len(chunks) != 3 or chunks != [min(shape[0], 1024), shape[1], shape[2]] or artifact.get("dtype") != "complex128" or artifact.get("complex_storage") != "paired_float64_real_imag" or artifact.get("chunk_policy") != "frequency_slab_full_matrix_v1":
            raise _integrity("HB matrix artifact storage is malformed.", artifact_id=role)
        units = {"s": ("dimensionless", "dimensionless"), "y": ("siemens", "conductance"), "z": ("ohm", "resistance"), "backend_native_s": ("dimensionless", "dimensionless"), "backend_native_z": ("ohm", "resistance")}
        if (artifact.get("unit"), artifact.get("dimensionality")) != units[role] or artifact.get("coordinate_ids") != terminal or artifact.get("probe_load_state") != expected_probe:
            raise _integrity("HB matrix artifact semantic metadata disagrees with the View.", artifact_id=role)
        channels = artifact.get("output_channels"), artifact.get("input_channels")
        expected_output_channels = [
            {"coordinate": coordinate, "mode": mode}
            for coordinate in terminal for mode in output_modes
        ]
        expected_input_channels = [
            {"coordinate": coordinate, "mode": mode}
            for coordinate in terminal for mode in input_modes
        ]
        if (
            not all(isinstance(channels_value, list) for channels_value in channels)
            or channels[0] != expected_output_channels
            or channels[1] != expected_input_channels
            or len(channels[0]) != shape[1]
            or len(channels[1]) != shape[2]
        ):
            raise _integrity("HB matrix channel catalog disagrees with its shape.", artifact_id=role)
        rank = len(spec.get("pump_axes", []))
        for channel_list in channels:
            seen: set[tuple[str, tuple[int, ...]]] = set()
            for channel in channel_list:
                if not isinstance(channel, dict) or set(channel) != {"coordinate", "mode"} or channel.get("coordinate") not in terminal or not _valid_mode_tuple(channel.get("mode"), rank):
                    raise _integrity("HB matrix channel label is malformed.", artifact_id=role)
                key = (str(channel["coordinate"]), tuple(channel["mode"]))
                if key in seen:
                    raise _integrity("HB matrix channel labels repeat.", artifact_id=role)
                seen.add(key)
        expected_axes = [
            {"id": "frequency", "kind": "frequency", "request_field": "spec.frequencies"},
            {"id": "output_channel", "kind": "output_channel", "values": channels[0]},
            {"id": "input_channel", "kind": "input_channel", "values": channels[1]},
        ]
        if artifact.get("axes") != expected_axes:
            raise _integrity("HB matrix axes disagree with its channel catalog.", artifact_id=role)
    else:
        is_state = role == "states"
        if trace:
            if len(shape) != 1 or len(chunks) != 1 or chunks != [min(shape[0], 1024)] or artifact.get("unit") != "dimensionless" or artifact.get("dimensionality") != "dimensionless" or artifact.get("chunk_policy") != "frequency_capped_1024_v1":
                raise _integrity("HB trace artifact storage is malformed.", artifact_id=role)
            expected_axes = [{"id": "frequency", "kind": "frequency", "request_field": "spec.frequencies"}]
        else:
            expected_chunks = [max(1, shape[0]), shape[1]] if len(shape) == 2 else None
            if len(shape) != 2 or len(chunks) != 2 or shape[1] < 1 or chunks != expected_chunks or artifact.get("unit") != ("weber" if is_state else "ampere") or artifact.get("dimensionality") != ("magnetic_flux" if is_state else "current") or artifact.get("chunk_policy") != "single_complete_array_v1":
                raise _integrity("HB state/source artifact storage is malformed.", artifact_id=role)
            axes = artifact.get("axes")
            if not isinstance(axes, list) or len(axes) != 2 or not all(isinstance(axis, dict) for axis in axes) or axes[0].get("kind") != "pump_mode" or axes[1].get("kind") != "node_coordinate":
                raise _integrity("HB state/source axes are malformed.", artifact_id=role)
            pump_modes, nodes = axes[0].get("values"), axes[1].get("values")
            if (
                not isinstance(pump_modes, list)
                or not isinstance(nodes, list)
                or len(pump_modes) != shape[0]
                or len(nodes) != shape[1]
                or any(not _valid_mode_tuple(mode, len(spec.get("pump_axes", []))) for mode in pump_modes)
                or any(not isinstance(node, str) or _IDENTIFIER.fullmatch(node) is None for node in nodes)
                or len({tuple(mode) for mode in pump_modes}) != len(pump_modes)
                or len(set(nodes)) != len(nodes)
                or pump_modes != operating_modes
                or nodes != compiler_nodes
            ):
                raise _integrity("HB state/source axis values disagree with its shape.", artifact_id=role)
            expected_axes = axes
        if artifact.get("dtype") != "complex128" or artifact.get("complex_storage") != "paired_float64_real_imag" or artifact.get("axes") != expected_axes:
            raise _integrity("HB non-matrix artifact has invalid representation.", artifact_id=role)
    _verify_zarr_datasets(artifact.get("datasets"), shape=shape, chunks=chunks, names=["real", "imag"])
