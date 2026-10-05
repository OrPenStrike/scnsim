"""Canonical identities for captured Plans and resolved parameter points.

These builders consume immutable normalized records; they never mutate live
authoring state or prepare a numerical runtime."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from ..canonical import (
    _identifier,
    _sha256,
    _validation,
    canonical_json_bytes,
    canonical_value,
    quantity_envelope,
    sha256_hex,
)


def _sort_id_maps(values: Iterable[Mapping[str, object]], *, key: str = "id") -> list[dict[str, object]]:
    copied = [dict(value) for value in values]
    copied.sort(key=lambda value: _identifier(value.get(key), field=key))
    if len({_identifier(value.get(key), field=key) for value in copied}) != len(copied):
        raise _validation("canonical identifiers must be unique", field=key)
    return copied


def canonical_plan_document(snapshot: Mapping[str, object]) -> dict[str, object]:
    """Close the one normalized structured-authoring record as a V2 Plan.

    ``AuthoringSnapshot.source_provenance`` is intentionally not accepted
    here.  Recursive traversal and compiler lookup are projections of these
    normalized tables, never additional records hashed beside them.
    """

    document = dict(snapshot)
    required = {
        "schema", "schema_version", "plan_id", "scope_hierarchy",
        "occurrences", "physical_leaves", "connectivity", "parameter_closure",
    }
    if set(document) != required or document.get("schema") != "scnsim.authoring_snapshot" or document.get("schema_version") != 2:
        raise _validation("authoring snapshot does not match the structured V2 handoff", fields=sorted(document))
    document["schema"] = "scnsim.plan"
    document["plan_id"] = _identifier(document["plan_id"], field="plan_id")

    def path_of(value: Mapping[str, object], *, field: str = "path") -> tuple[str, ...]:
        path = value.get(field)
        if not isinstance(path, Sequence) or isinstance(path, (str, bytes)) or not path:
            raise _validation("structured record requires a nonempty occurrence path", field=field)
        return tuple(_identifier(item, field=field) for item in path)

    occurrences = [dict(item) for item in _iter_mappings(document["occurrences"], "occurrences")]
    occurrences.sort(key=path_of)
    if len({path_of(item) for item in occurrences}) != len(occurrences):
        raise _validation("occurrence paths must be unique")
    document["occurrences"] = occurrences

    leaves = [dict(item) for item in _iter_mappings(document["physical_leaves"], "physical_leaves")]
    leaves.sort(key=path_of)
    if len({path_of(item) for item in leaves}) != len(leaves):
        raise _validation("physical-leaf paths must be unique")
    occurrence_paths = {path_of(item) for item in occurrences}
    if any(path_of(item) not in occurrence_paths for item in leaves):
        raise _validation("physical leaf has no owning occurrence")
    document["physical_leaves"] = leaves

    connectivity = dict(_mapping(document["connectivity"], "connectivity"))
    expected_connectivity = {
        "physical_endpoints", "endpoint_nets", "node_coordinates", "canonical_ground", "ports", "couplings",
    }
    if set(connectivity) != expected_connectivity or connectivity.get("canonical_ground") != "ground":
        raise _validation("structured connectivity fields are invalid")
    endpoints = [dict(item) for item in _iter_mappings(connectivity["physical_endpoints"], "physical_endpoints")]
    endpoints.sort(key=lambda item: (path_of(item), _identifier(item.get("pin"), field="pin")))
    endpoint_keys = [(path_of(item), _identifier(item.get("pin"), field="pin")) for item in endpoints]
    if len(set(endpoint_keys)) != len(endpoint_keys):
        raise _validation("physical endpoints must be unique")
    connectivity["physical_endpoints"] = endpoints
    endpoint_nets = [dict(item) for item in _iter_mappings(connectivity["endpoint_nets"], "endpoint_nets")]
    endpoint_nets.sort(key=lambda item: canonical_json_bytes(item))
    if len({canonical_json_bytes(item.get("endpoint")) for item in endpoint_nets}) != len(endpoint_nets):
        raise _validation("structural endpoints must map to one final net")
    connectivity["endpoint_nets"] = endpoint_nets
    node_coordinates = [
        dict(item) for item in _iter_mappings(connectivity["node_coordinates"], "node_coordinates")
    ]
    final_nets: list[str] = []
    compiler_nodes: list[str] = []
    for node in node_coordinates:
        if set(node) != {"final_net", "compiler_node_id", "visibility", "public_aliases"}:
            raise _validation("node-coordinate fields are invalid")
        final_nets.append(_identifier(node["final_net"], field="final_net"))
        compiler_nodes.append(_identifier(node["compiler_node_id"], field="compiler_node_id"))
        if node["visibility"] not in {"public", "internal"}:
            raise _validation("node-coordinate visibility is invalid")
        aliases = node["public_aliases"]
        if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
            raise _validation("node-coordinate aliases must be an ordered array")
    if len(set(final_nets)) != len(final_nets) or len(set(compiler_nodes)) != len(compiler_nodes):
        raise _validation("node-coordinate identities must be unique")
    if any(item.get("net") not in set(final_nets) | {"ground"} for item in endpoints):
        raise _validation("physical endpoint targets an unknown final net")
    connectivity["node_coordinates"] = node_coordinates  # Compiler order is explicit.
    ports = [dict(item) for item in _iter_mappings(connectivity["ports"], "ports")]
    port_ids = [_identifier(item.get("id"), field="port.id") for item in ports]
    if len(set(port_ids)) != len(port_ids):
        raise _validation("Port IDs must be unique")
    connectivity["ports"] = ports  # Declaration order is semantic.
    connectivity["couplings"] = _sort_id_maps(
        _iter_mappings(connectivity["couplings"], "couplings")
    )
    document["connectivity"] = connectivity

    closure = dict(_mapping(document["parameter_closure"], "parameter_closure"))
    if set(closure) != {"definitions", "field_bindings"}:
        raise _validation("parameter closure fields are invalid")
    definitions = [dict(item) for item in _iter_mappings(closure["definitions"], "definitions")]
    definitions.sort(key=lambda item: _parameter_ref_key(item))
    definition_keys = [_parameter_ref_key(item) for item in definitions]
    if len(set(definition_keys)) != len(definition_keys):
        raise _validation("consumed parameter definitions must be unique")
    bindings = [dict(item) for item in _iter_mappings(closure["field_bindings"], "field_bindings")]
    bindings.sort(key=lambda item: (path_of(item), _identifier(item.get("field"), field="field")))
    binding_keys = [(path_of(item), _identifier(item.get("field"), field="field")) for item in bindings]
    if len(set(binding_keys)) != len(binding_keys):
        raise _validation("physical fields cannot have competing parameter bindings")
    for binding in bindings:
        if _parameter_ref_key(binding.get("parameter")) not in set(definition_keys):
            raise _validation("physical field binding names an unconsumed parameter")
    closure["definitions"] = definitions
    closure["field_bindings"] = bindings
    document["parameter_closure"] = closure
    return canonical_value(document)  # type: ignore[return-value]


def canonical_plan_snapshot(snapshot: object) -> dict[str, object]:
    """Canonicalize an ``AuthoringSnapshot`` without importing model types."""

    semantic_record = getattr(snapshot, "semantic_record", None)
    if not isinstance(semantic_record, Mapping):
        raise TypeError("snapshot must expose an immutable semantic_record mapping")
    return canonical_plan_document(semantic_record)


def canonical_parameters_sha256(parameter_record: Mapping[str, object]) -> str:
    """Identify one complete effective point, independently of source spelling."""

    return sha256_hex({
        "schema": "scnsim.parameter_point_identity",
        "schema_version": 2,
        "parameters": canonical_parameter_set(parameter_record),
    })


def canonical_resolved_plan_point(
    point: object, *, plan_sha256: str
) -> dict[str, object]:
    """Encode the immutable point handed to the standalone compiler audit."""

    snapshot = getattr(point, "snapshot", None)
    parameter_record = getattr(point, "parameter_record", None)
    resolved_fields = getattr(point, "resolved_fields", None)
    if snapshot is None or not isinstance(parameter_record, Mapping) or not isinstance(resolved_fields, Mapping):
        raise TypeError("point must be a ResolvedPlanPoint")
    plan = canonical_plan_snapshot(snapshot)
    expected_plan_sha = sha256_hex(plan)
    if _sha256(plan_sha256, field="plan_sha256") != expected_plan_sha:
        raise _validation("resolved point is paired with a different Plan")
    units_by_field: dict[tuple[tuple[str, ...], str], str] = {}
    for leaf in _iter_mappings(plan["physical_leaves"], "physical_leaves"):
        path = _component_path(leaf.get("path"))
        for field in _iter_mappings(leaf.get("fields"), "physical leaf fields"):
            identifier = _identifier(field.get("id"), field="field")
            unit = _as_str(field.get("unit"), "field.unit")
            units_by_field[(path, identifier)] = unit
    if set(resolved_fields) != set(units_by_field):
        raise _validation("resolved physical fields do not exactly cover the Plan")

    from ..units import registry

    rows: list[dict[str, object]] = []
    for key in sorted(units_by_field):
        value = resolved_fields[key]
        unit = units_by_field[key]
        record = getattr(value, "_record", None)
        if unit == "rlgc":
            if not callable(record):
                raise _validation("resolved RLGC field has no structured record")
            encoded = record()
            if not isinstance(encoded, Mapping) or encoded.get("type") != "rlgc":
                raise _validation("resolved RLGC field is malformed")
        else:
            encoded = quantity_envelope(value, si_unit=unit, registry=registry)
        rows.append({"path": list(key[0]), "field": key[1], "value": encoded})
    parameters = canonical_parameter_set(parameter_record)
    return canonical_value({
        "schema": "scnsim.resolved_plan_point",
        "schema_version": 2,
        "plan_sha256": expected_plan_sha,
        "parameters": parameters,
        "parameters_sha256": canonical_parameters_sha256(parameters),
        "resolved_fields": rows,
    })  # type: ignore[return-value]


def canonical_diagram_digests(
    plan: object, *, representation: str
) -> dict[str, str]:
    """Return the representation-tagged diagram facts owned by canonicalization."""

    if representation not in {"authoring", "compiled"}:
        raise ValueError("representation must be 'authoring' or 'compiled'")
    if isinstance(plan, Mapping):
        raw = dict(plan)
        if raw.get("schema") == "scnsim.plan":
            raw["schema"] = "scnsim.authoring_snapshot"
        document = canonical_plan_document(raw)
    else:
        document = canonical_plan_snapshot(plan)
    plan_sha256 = sha256_hex(document)
    common = {
        "schema_version": 2,
        "representation": representation,
        "plan_sha256": plan_sha256,
    }
    connectivity = sha256_hex({
        **common,
        "schema": "scnsim.diagram_connectivity_identity",
        "connectivity": document["connectivity"],
    })
    semantic = sha256_hex({
        **common,
        "schema": "scnsim.diagram_semantic_identity",
        "scope_hierarchy": document["scope_hierarchy"],
        "occurrences": document["occurrences"],
        "physical_leaves": document["physical_leaves"],
        "parameter_closure": document["parameter_closure"],
    })
    return {
        "plan_sha256": plan_sha256,
        "connectivity_sha256": connectivity,
        "semantic_sha256": semantic,
    }


def canonical_expanded_graph_sha256(
    *,
    plan_sha256: str,
    node_order: Sequence[str],
    resolved_bindings: Sequence[Mapping[str, object]],
    expanded_branch_rows: Sequence[Mapping[str, object]],
) -> str:
    """Bind the exact compiler expansion fields promised to diagram evidence."""

    return sha256_hex({
        "schema": "scnsim.expanded_graph_identity",
        "schema_version": 2,
        "plan_sha256": _sha256(plan_sha256, field="plan_sha256"),
        "node_order": list(node_order),
        "resolved_bindings": list(resolved_bindings),
        "expanded_branch_rows": list(expanded_branch_rows),
    })


def _component_path(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise _validation("component path must be nonempty segments")
    return tuple(_identifier(segment, field="component_path") for segment in value)


def _as_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise _validation("expected string", field=field)
    return value


def _iter_mappings(value: object, field: str) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _validation("expected array", field=field)
    output: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise _validation("array item must be an object", field=field)
        output.append(item)
    return output


def canonical_parameter_set(parameters: Mapping[str, object]) -> dict[str, object]:
    """Close one V2 ParameterSet by logical definition/local identity."""

    document = dict(parameters)
    if set(document) != {"type", "bindings", "allow_extrapolation"} or document.get("type") != "parameter_set_v2":
        raise _validation("invalid parameter-set envelope")
    bindings = _iter_mappings(document["bindings"], "parameter bindings")
    if any(set(item) != {"parameter", "value"} for item in bindings):
        raise _validation("parameter binding fields are invalid")
    for item in bindings:
        reference = _mapping(item["parameter"], "parameter_ref")
        if set(reference) != {"definitions_id", "parameter_id"}:
            raise _validation("parameter ref fields are invalid")
    bindings.sort(key=lambda item: _parameter_ref_key(_mapping(item, "parameter binding").get("parameter")))
    if len({_parameter_ref_key(item.get("parameter")) for item in bindings}) != len(bindings):
        raise _validation("parameter bindings must be unique")
    document["bindings"] = [dict(item) for item in bindings]
    authorizations = _iter_mappings(document["allow_extrapolation"], "allow_extrapolation")
    if any(set(item) != {"definitions_id", "parameter_id"} for item in authorizations):
        raise _validation("parameter authorization ref fields are invalid")
    authorizations.sort(key=lambda item: _parameter_ref_key(item))
    if len({_parameter_ref_key(item) for item in authorizations}) != len(authorizations):
        raise _validation("allow_extrapolation values must be unique")
    document["allow_extrapolation"] = [dict(item) for item in authorizations]
    return canonical_value(document)  # type: ignore[return-value]


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _validation("expected object", field=field)
    return value


def _parameter_ref_key(value: object) -> tuple[str, str]:
    ref = _mapping(value, "parameter_ref")
    if not {"definitions_id", "parameter_id"}.issubset(ref):
        raise _validation("parameter ref fields are invalid")
    return (
        _identifier(ref["definitions_id"], field="definitions_id"),
        _identifier(ref["parameter_id"], field="parameter_id"),
    )
