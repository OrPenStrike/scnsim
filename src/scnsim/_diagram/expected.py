"""Independent expected manifests for certified circuit diagrams.

This module deliberately consumes only the immutable canonical Plan capture,
and, for a compiled projection, the compiler's sealed expanded-graph evidence.
It does not import the authoring semantic IR or any placement data.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from .. import units
from .._canonical import (
    canonical_json_bytes,
    quantity_envelope,
    sha256_hex,
)
from ..errors import SCNSimValidationError
from .._authoring_snapshot import ResolvedPlanPoint

_Representation = Literal["authoring", "compiled"]
_Path = tuple[str, ...]

def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_audit", evidence=evidence)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail("canonical diagram input needs an object", field=field)
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail("canonical diagram input needs an array", field=field)
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail("canonical diagram input needs a nonempty string", field=field)
    return value


def _token(kind: str, **fields: object) -> str:
    return canonical_json_bytes({"kind": kind, **fields}).decode("utf-8")


@dataclass(frozen=True, slots=True)
class _PointExpectedManifest:
    representation: _Representation
    electrical: Mapping[str, object]
    structural: Mapping[str, object]
    expected_values: Mapping[str, object]
    verified: Mapping[str, object]
    compiled_graph_sha256: str | None = None
    expanded_graph_sha256: str | None = None


def _point_semantic(point: ResolvedPlanPoint) -> Mapping[str, object]:
    if not isinstance(point, ResolvedPlanPoint):
        raise TypeError("structured witness requires ResolvedPlanPoint")
    semantic = _mapping(point.snapshot.semantic_record, "semantic_record")
    if semantic.get("schema") != "scnsim.authoring_snapshot" or semantic.get("schema_version") != 2:
        raise _fail("structured witness requires the normalized V2 authoring snapshot")
    return semantic


def _v2_path(value: object, field: str) -> _Path:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail("structured witness path must be an array", field=field)
    result = tuple(_string(item, field) for item in value)
    return result


def _v2_quantity(point: ResolvedPlanPoint, path: _Path, field: str, unit: str) -> Mapping[str, object]:
    value = point.resolved_fields.get((path, field))
    if value is None:
        raise _fail("resolved point does not exactly cover a displayed physical field", path=list(path), field=field)
    if unit == "rlgc":
        record = getattr(value, "_record", None)
        if not callable(record):
            raise _fail("resolved RLGC field has no structured record", path=list(path), field=field)
        encoded = record()
        if not isinstance(encoded, Mapping):
            raise _fail("resolved RLGC record is malformed", path=list(path), field=field)
        return encoded
    return cast(
        Mapping[str, object],
        quantity_envelope(value, si_unit=unit, registry=units.registry),
    )


def _v2_contact(*, path: _Path | None = None, pin: str | None = None, port: str | None = None) -> str:
    if port is not None:
        return _token("visible_port_contact", port_id=port)
    assert path is not None and pin is not None
    return _token("visible_physical_terminal", component_path=list(path), pin_id=pin)


def _v2_net_map(semantic: Mapping[str, object]) -> tuple[dict[str, str], tuple[dict[str, object], ...]]:
    connectivity = _mapping(semantic.get("connectivity"), "connectivity")
    contacts: dict[str, set[str]] = defaultdict(set)
    for raw in _sequence(connectivity.get("physical_endpoints"), "physical_endpoints"):
        row = _mapping(raw, "physical_endpoint")
        path = _v2_path(row.get("path"), "physical_endpoint.path")
        pin = _string(row.get("pin"), "physical_endpoint.pin")
        net = _string(row.get("net"), "physical_endpoint.net")
        contacts[net].add(_v2_contact(path=path, pin=pin))
    for raw in _sequence(connectivity.get("ports"), "ports"):
        row = _mapping(raw, "port")
        contacts[_string(row.get("net"), "port.net")].add(
            _v2_contact(port=_string(row.get("id"), "port.id"))
        )
    result: dict[str, str] = {"ground": "ground"}
    rows: list[dict[str, object]] = []
    ordered_sources = sorted(
        (source for source in contacts if source != "ground"),
        key=lambda source: canonical_json_bytes(sorted(contacts[source])),
    )
    ordinals = {
        source: f"netv-{ordinal}" for ordinal, source in enumerate(ordered_sources)
    }
    for source, values in sorted(contacts.items()):
        if not values:
            raise _fail("captured electrical net has no visible physical contact", net=source)
        visible = "ground" if source == "ground" else ordinals[source]
        result[source] = visible
        rows.append({"net": visible, "contacts": sorted(values)})
    return result, tuple(sorted(rows, key=canonical_json_bytes))


def _v2_rebased_path(value: object, field: str, *, basis: _Path) -> _Path:
    path = _v2_path(value, field)
    return (*basis, *path) if basis else path


def _v2_endpoint(
    value: object,
    *,
    basis: _Path = (),
    region_paths: frozenset[_Path] = frozenset(),
) -> dict[str, object]:
    endpoint = dict(_mapping(value, "structured endpoint"))
    kind = endpoint.get("kind")
    if kind == "ground":
        if set(endpoint) != {"kind"}:
            raise _fail("ground endpoint has extra fields")
        return endpoint
    if kind not in {"bus", "tap", "pin"}:
        raise _fail("structured endpoint kind is unsupported", kind=kind)
    scope = _v2_rebased_path(endpoint.get("scope"), "endpoint.scope", basis=basis)
    identifier = _string(endpoint.get("id"), "endpoint.id")
    if kind == "tap":
        _string(endpoint.get("bus"), "endpoint.bus")
    if kind == "pin":
        component = _string(endpoint.get("component"), "endpoint.component")
        if not isinstance(endpoint.get("public"), bool):
            raise _fail("pin endpoint public flag is malformed")
        boundary_path = scope if endpoint["public"] is True else (*scope, component)
        if endpoint["public"] is True or boundary_path in region_paths:
            return {"kind": "boundary_pin", "scope": list(boundary_path), "id": identifier}
        return {
            "kind": "physical_pin",
            "path": [*scope, component],
            "pin_id": identifier,
        }
    endpoint["scope"] = list(scope)
    return cast(dict[str, object], _plain(endpoint))


def _v2_element(value: object, *, basis: _Path = ()) -> dict[str, object]:
    row = _mapping(value, "structured element")
    return {
        "path": list(_v2_rebased_path(row.get("path"), "element.path", basis=basis)),
        "pin_1": _string(row.get("pin_1"), "element.pin_1"),
        "pin_2": _string(row.get("pin_2"), "element.pin_2"),
    }


def _v2_scopes(root: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []

    def visit(scope: Mapping[str, object], parent: _Path | None) -> None:
        path = _v2_path(scope.get("path"), "scope.path")
        _string(scope.get("kind"), "scope.kind")
        if path:
            records.append(
                {
                    "path": list(path),
                    "parent": [] if parent is None else list(parent),
                    "local_id": path[-1],
                }
            )
        for child in _sequence(scope.get("children"), "scope.children"):
            visit(_mapping(child, "child scope"), path)
        for body in _sequence(scope.get("component_bodies"), "scope.component_bodies"):
            visit(_mapping(_mapping(body, "component body").get("body"), "component body scope"), path)

    visit(root, None)
    return tuple(sorted(records, key=lambda row: tuple(cast(Sequence[str], row["path"]))))


def _v2_visible_contacts(
    semantic: Mapping[str, object],
) -> tuple[dict[str, tuple[str, ...]], dict[str, _Path]]:
    """Project only physical/Port contacts that can be recovered from ink."""

    connectivity = _mapping(semantic.get("connectivity"), "connectivity")
    by_net: dict[str, list[str]] = defaultdict(list)
    owners: dict[str, _Path] = {}
    for raw in _sequence(connectivity.get("physical_endpoints"), "physical_endpoints"):
        row = _mapping(raw, "physical_endpoint")
        path = _v2_path(row.get("path"), "physical_endpoint.path")
        pin = _string(row.get("pin"), "physical_endpoint.pin")
        contact = _v2_contact(path=path, pin=pin)
        by_net[_string(row.get("net"), "physical_endpoint.net")].append(contact)
        owners[contact] = path[:-1]
    for raw in _sequence(connectivity.get("ports"), "ports"):
        row = _mapping(raw, "port")
        contact = _v2_contact(port=_string(row.get("id"), "port.id"))
        by_net[_string(row.get("net"), "port.net")].append(contact)
        owners[contact] = ()
    return (
        {net: tuple(sorted(set(contacts))) for net, contacts in by_net.items()},
        owners,
    )


def _v2_parent_pin_contacts(point: ResolvedPlanPoint, root: Mapping[str, object]) -> set[tuple[_Path, str]]:
    """Read actual parent pin uses from declarations, not global net equality."""
    contacts: set[tuple[_Path, str]] = set()

    def endpoint(raw: object, owner: _Path) -> None:
        if not isinstance(raw, Mapping) or raw.get("kind") != "pin":
            return
        scope = _v2_path(raw.get("scope"), "pin.scope")
        path = scope if raw.get("public") else (*scope, _string(raw.get("component"), "pin.component"))
        if path[:-1] == owner:
            contacts.add((path, _string(raw.get("id"), "pin.id")))

    def visit(scope: Mapping[str, object]) -> None:
        owner = _v2_path(scope.get("path"), "scope.path")
        for raw in _sequence(scope.get("structures"), "scope.structures"):
            row = _mapping(raw, "structure")
            for key in ("start", "at", "end"):
                endpoint(row.get(key), owner)
            for raw_endpoint in row.get("endpoints", ()):
                endpoint(raw_endpoint, owner)
            for branch in row.get("branches", (row,)):
                for member in branch.get("elements", ()):
                    path = _v2_path(member.get("path"), "member.path")
                    if path[:-1] == owner:
                        for key in ("pin_1", "pin_2"):
                            contacts.add((path, _string(member.get(key), key)))
        exposure = _mapping(scope.get("exposures"), "scope.exposures")
        for raw in _sequence(exposure.get("pins"), "scope.exposures.pins"):
            endpoint(_mapping(raw, "public pin").get("intrinsic_endpoint"), owner)
        for child in scope.get("children", ()):
            visit(child)
        for body in scope.get("component_bodies", ()):
            visit(body["body"])

    visit(root)
    for group in point.snapshot.source_provenance.get("ground_pins_call_groups", ()):
        for pin in group:
            scope = _v2_path(pin.get("scope"), "ground pin.scope")
            endpoint(pin, scope[:-1] if pin.get("public") else scope)
    return contacts


def _v2_structural(
    point: ResolvedPlanPoint,
    semantic: Mapping[str, object],
    net_map: Mapping[str, str],
) -> Mapping[str, object]:
    """Build the source-side expectation for visible ownership evidence only."""

    root = _mapping(semantic.get("scope_hierarchy"), "scope_hierarchy")
    parent_contacts = _v2_parent_pin_contacts(point, root)
    scope_rows = _v2_scopes(root)
    contacts_by_net, contact_owners = _v2_visible_contacts(semantic)
    boundary_rows: list[dict[str, object]] = []

    def visit(scope: Mapping[str, object]) -> None:
        path = _v2_path(scope.get("path"), "scope.path")
        exposure = _mapping(scope.get("exposures"), "scope.exposures")
        pins = tuple(
            _mapping(raw, "scope public pin")
            for raw in _sequence(exposure.get("pins"), "scope.exposures.pins")
        )
        if not path and pins:
            raise _fail("root scope cannot expose a child-boundary pin")
        for pin in pins:
            source_net = _string(pin.get("final_net"), "exposure.final_net")
            try:
                visible_net = net_map[source_net]
            except KeyError as error:
                raise _fail(
                    "public boundary net has no visible physical contact",
                    scope=list(path),
                    net=source_net,
                ) from error
            contacts = contacts_by_net.get(source_net, ())
            inside = sorted(
                contact
                for contact in contacts
                if contact_owners[contact][: len(path)] == path
            )
            outside = sorted(contact for contact in contacts if contact not in inside)
            boundary = (
                {"mode": "only"}
                if len(pins) == 1
                else {
                    "mode": "named",
                    "id": _string(pin.get("id"), "exposure.id"),
                }
            )
            boundary_rows.append(
                {
                    "region": list(path),
                    "boundary": boundary,
                    "parent_incidence": "connected" if (path, pin["id"]) in parent_contacts else "open",
                    "net": visible_net,
                    "inside_contacts": inside,
                    "outside_contacts": outside,
                }
            )
        for child in _sequence(scope.get("children"), "scope.children"):
            visit(_mapping(child, "child scope"))
        for body in _sequence(scope.get("component_bodies"), "scope.component_bodies"):
            visit(_mapping(_mapping(body, "component body").get("body"), "component body scope"))

    visit(root)
    physical = [
        {
            "path": list(_v2_path(row.get("path"), "physical_leaf.path")),
            "owner": list(_v2_path(row.get("path"), "physical_leaf.path")[:-1]),
            "model": _string(row.get("model"), "physical_leaf.model"),
        }
        for row in (_mapping(item, "physical leaf") for item in _sequence(semantic.get("physical_leaves"), "physical_leaves"))
    ]
    ports = [
        {"port_id": _string(row.get("id"), "port.id"), "owner": []}
        for row in (_mapping(item, "port") for item in _sequence(_mapping(semantic.get("connectivity"), "connectivity").get("ports"), "ports"))
    ]
    return cast(
        Mapping[str, object],
        _freeze(
            {
                "schema": "scnsim.diagram_structural_manifest",
                "schema_version": 2,
                "representation": "authoring",
                "regions": list(scope_rows),
                "leaf_ownership": sorted(physical, key=lambda row: cast(list[str], row["path"])),
                "port_ownership": sorted(ports, key=lambda row: cast(str, row["port_id"])),
                "boundary_incidence": sorted(boundary_rows, key=canonical_json_bytes),
            }
        ),
    )


def _v2_verified_source_rows(
    point: ResolvedPlanPoint,
    semantic: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    """Return complete captured authoring facts that are not image claims."""

    rows: list[dict[str, object]] = []

    def add(kind: str, **fields: object) -> None:
        rows.append({"category": "verified_source", "kind": kind, **fields})

    def visit(scope: Mapping[str, object]) -> None:
        path = list(_v2_path(scope.get("path"), "scope.path"))
        for raw in _sequence(scope.get("structures"), "scope.structures"):
            add("authored_operator", scope=path, record=_plain(_mapping(raw, "structure")))
        for raw in _sequence(scope.get("buses"), "scope.buses"):
            add("authored_bus", scope=path, record=_plain(_mapping(raw, "bus")))
        exposures = _mapping(scope.get("exposures"), "scope.exposures")
        for collection, exposure_kind in (
            ("pins", "pin"),
            ("coordinates", "coordinate"),
            ("branches", "branch"),
            ("parameters", "parameter"),
        ):
            for raw in _sequence(exposures.get(collection), f"scope.exposures.{collection}"):
                add(
                    "authored_exposure",
                    scope=path,
                    exposure_kind=exposure_kind,
                    record=_plain(_mapping(raw, "exposure")),
                )
        for child in _sequence(scope.get("children"), "scope.children"):
            visit(_mapping(child, "child scope"))
        for raw in _sequence(scope.get("component_bodies"), "scope.component_bodies"):
            body = _mapping(raw, "component body")
            visit(_mapping(body.get("body"), "component body scope"))

    visit(_mapping(semantic.get("scope_hierarchy"), "scope_hierarchy"))
    connectivity = _mapping(semantic.get("connectivity"), "connectivity")
    for raw in _sequence(semantic.get("physical_leaves"), "physical_leaves"):
        add("oriented_physical_leaf", record=_plain(_mapping(raw, "physical leaf")))
    for raw in _sequence(connectivity.get("node_coordinates"), "node_coordinates"):
        add("authored_node_alias", record=_plain(_mapping(raw, "node coordinate")))
    closure = _mapping(semantic.get("parameter_closure"), "parameter_closure")
    for raw in _sequence(closure.get("definitions"), "parameter_closure.definitions"):
        add("parameter_definition", record=_plain(_mapping(raw, "parameter definition")))
    for raw in _sequence(closure.get("field_bindings"), "parameter_closure.field_bindings"):
        add("parameter_field_binding", record=_plain(_mapping(raw, "parameter binding")))
    add("effective_parameter_point", record=_plain(point.parameter_record))
    provenance = _mapping(point.snapshot.source_provenance, "source_provenance")
    for raw in _sequence(provenance.get("source_units"), "source_provenance.source_units"):
        add("source_unit", record=_plain(_mapping(raw, "source unit")))
    for raw in _sequence(
        provenance.get("ground_pins_call_groups"),
        "source_provenance.ground_pins_call_groups",
    ):
        add("ground_call_group", record=_plain(_sequence(raw, "ground call group")))
    return tuple(
        cast(Mapping[str, object], _freeze(row))
        for row in sorted(rows, key=canonical_json_bytes)
    )


def _v2_authoring(point: ResolvedPlanPoint) -> _PointExpectedManifest:
    from .._canonical import canonical_diagram_digests, canonical_parameters_sha256

    semantic = _point_semantic(point)
    net_map, nets = _v2_net_map(semantic)
    connectivity = _mapping(semantic.get("connectivity"), "connectivity")
    endpoint_net = {
        (_v2_path(row.get("path"), "physical_endpoint.path"), _string(row.get("pin"), "physical_endpoint.pin")):
        net_map[_string(row.get("net"), "physical_endpoint.net")]
        for row in (_mapping(item, "physical endpoint") for item in _sequence(connectivity.get("physical_endpoints"), "physical_endpoints"))
    }
    values: dict[str, object] = {}
    bodies: list[dict[str, object]] = []
    for raw in _sequence(semantic.get("physical_leaves"), "physical_leaves"):
        leaf = _mapping(raw, "physical leaf")
        path = _v2_path(leaf.get("path"), "physical_leaf.path")
        pins = tuple(_string(item, "physical_leaf.pin") for item in _sequence(leaf.get("pin_order"), "physical_leaf.pin_order"))
        fields: list[dict[str, object]] = []
        for raw_field in _sequence(leaf.get("fields"), "physical_leaf.fields"):
            field = _mapping(raw_field, "physical field")
            field_id = _string(field.get("id"), "physical_field.id")
            unit = _string(field.get("unit"), "physical_field.unit")
            value = _v2_quantity(point, path, field_id, unit)
            fields.append({"id": field_id, "unit": unit})
            if unit != "rlgc":
                values[_token("physical_field", component_path=list(path), field=field_id)] = _plain(value)
        record: dict[str, object] = {
            "path": list(path),
            "model": _string(leaf.get("model"), "physical_leaf.model"),
            "pin_order": list(pins),
            "terminals": [{"pin_id": pin, "net": endpoint_net[(path, pin)]} for pin in pins],
            "fields": fields,
            "oriented_branches": _plain(leaf.get("oriented_branches")),
        }
        if record["model"] == "transmission_line":
            rlgc = point.resolved_fields.get((path, "rlgc"))
            conductors = getattr(rlgc, "conductors", None)
            reference = getattr(rlgc, "reference_conductor", None)
            if not isinstance(conductors, tuple) or not conductors or not isinstance(reference, str):
                raise _fail("resolved transmission line lacks ordered RLGC evidence", path=list(path))
            record["conductors"] = list(conductors)
            record["reference_conductor"] = reference
            record["line_kind"] = "CPW" if len(conductors) == 1 else "MTL"
            if len(conductors) == 1:
                record["conductors"] = []
                record["reference_conductor"] = None
                record["n_sections"] = _mapping(leaf.get("model_metadata"), "physical_leaf.model_metadata")["n_sections"]
        bodies.append(record)
    ports = []
    for raw in _sequence(connectivity.get("ports"), "ports"):
        port = _mapping(raw, "port")
        port_id = _string(port.get("id"), "port.id")
        impedance = _mapping(port.get("reference_impedance"), "port.reference_impedance")
        ports.append(
            {
                "port_id": port_id,
                "role": _string(port.get("role"), "port.role"),
                "node_net": net_map[_string(port.get("net"), "port.net")],
                "reference_net": "ground",
                "reference_impedance": _plain(impedance),
                "orientation": "node_to_reference",
                "load_kind": "raw_reference_impedance",
            }
        )
        values[_token("port_impedance", port_id=port_id)] = _plain(impedance)
    couplings = []
    for row in (
        _mapping(item, "coupling")
        for item in _sequence(connectivity.get("couplings"), "couplings")
    ):
        coupling_id = _string(row.get("id"), "coupling.id")
        branches = []
        for field in ("inductor_a", "inductor_b"):
            branch = _mapping(row.get(field), f"coupling.{field}")
            branches.append(
                {
                    "path": list(_v2_path(branch.get("path"), f"coupling.{field}.path")),
                    "branch_id": _string(
                        branch.get("branch_id"), f"coupling.{field}.branch_id"
                    ),
                }
            )
        coefficient = _mapping(
            row.get("coupling_coefficient"), "coupling.coupling_coefficient"
        )
        couplings.append(
            {
                "coupling_id": coupling_id,
                "branches": sorted(branches, key=canonical_json_bytes),
            }
        )
        values[_token("coupling_coefficient", coupling_id=coupling_id)] = _plain(
            coefficient
        )
    electrical = cast(
        Mapping[str, object],
        _freeze(
            {
                "schema": "scnsim.diagram_electrical_manifest",
                "schema_version": 2,
                "representation": "authoring",
                "nets": list(nets),
                "bodies": sorted(bodies, key=lambda row: cast(list[str], row["path"])),
                "ports": sorted(ports, key=lambda row: cast(str, row["port_id"])),
                "couplings": sorted(couplings, key=lambda row: cast(str, row["coupling_id"])),
            }
        ),
    )
    digests = canonical_diagram_digests(point.snapshot, representation="authoring")
    verified = cast(
        Mapping[str, object],
        _freeze(
            {
                "plan_id": semantic["plan_id"],
                **digests,
                "parameters_sha256": canonical_parameters_sha256(point.parameter_record),
                "source_provenance": _plain(point.snapshot.source_provenance),
                "canonical_values": values,
                "source_rows": _v2_verified_source_rows(point, semantic),
            }
        ),
    )
    from .equivalence import normalize_authoring
    from .values import display_projection

    # Keep the frozen verified source quantities above exact. Only the visible
    # authoring correspondence uses the deterministic six-digit projection.
    values = {identity: dict(display_projection(value)) for identity, value in values.items()}
    electrical = _plain(electrical)
    for port in electrical["ports"]:
        port["reference_impedance"] = dict(display_projection(port["reference_impedance"]))
    electrical, structural, values = normalize_authoring(
        electrical, _v2_structural(point, semantic, net_map), values
    )
    return _PointExpectedManifest(
        "authoring",
        cast(Mapping[str, object], _freeze(electrical)),
        cast(Mapping[str, object], _freeze(structural)),
        cast(Mapping[str, object], _freeze(values)),
        verified,
    )


def _compiled_row_projection(row: Mapping[str, object]) -> dict[str, object]:
    """Project one compiler row onto the complete visible matrix grammar."""

    kind = _string(row.get("kind"), "expanded row kind")
    if kind == "transmission_line_audit":
        return {
            "kind": kind,
            "component_path": _plain(row.get("component_path")),
            "conductors": _plain(row.get("conductors")),
            "reference_conductor": row.get("reference_conductor"),
            "n_sections": row.get("n_sections"),
            "length": _plain(row.get("length")),
            "dx": _plain(row.get("dx")),
            "orientation": row.get("orientation"),
            "stations": _plain(row.get("stations")),
            "rlgc_source": _plain(row.get("rlgc_source")),
        }
    if kind == "mutual_inductance":
        return {
            "kind": kind,
            "coupling_id": row.get("coupling_id"),
            "branch_a": _plain(row.get("branch_a")),
            "branch_b": _plain(row.get("branch_b")),
            "coupling_coefficient": _plain(row.get("coupling_coefficient")),
            "derived_mutual_inductance": _plain(row.get("derived_mutual_inductance")),
            "omitted_as_zero": row.get("omitted_as_zero"),
        }
    return {
        "kind": kind,
        "component_path": _plain(row.get("component_path")),
        "section": row.get("section"),
        "station": row.get("station"),
        "end": row.get("end"),
        "row_conductor": row.get("row_conductor"),
        "column_conductor": row.get("column_conductor"),
        "branch_id": row.get("branch_id"),
        "value": _plain(row.get("value")),
        "omitted_as_zero": row.get("omitted_as_zero"),
        "terminal_1_to_terminal_2": row.get("terminal_1_to_terminal_2"),
        "incidence_f64": _plain(row.get("incidence_f64")),
        "physical_positive_incidence_f64": _plain(
            row.get("physical_positive_incidence_f64")
        ),
        "physical_negative_incidence_f64": _plain(
            row.get("physical_negative_incidence_f64")
        ),
    }



def _v2_compiled(point: ResolvedPlanPoint, compiled: Mapping[str, object]) -> _PointExpectedManifest:
    from .._canonical import (
        canonical_diagram_digests,
        canonical_expanded_graph_sha256,
        canonical_parameters_sha256,
        canonical_plan_snapshot,
        sha256_hex as canonical_sha256_hex,
    )

    semantic = _point_semantic(point)
    plan_sha = canonical_sha256_hex(canonical_plan_snapshot(point.snapshot))
    if compiled.get("plan_sha256") != plan_sha:
        raise _fail("compiled evidence belongs to a different Plan point")
    parameters_sha = canonical_parameters_sha256(point.parameter_record)
    if compiled.get("parameters_sha256") != parameters_sha:
        raise _fail("compiled evidence belongs to a different effective point")
    node_order = tuple(_string(item, "compiled.node_order") for item in _sequence(compiled.get("node_order"), "compiled.node_order"))
    bindings = tuple(_mapping(item, "compiled.resolved_binding") for item in _sequence(compiled.get("resolved_bindings"), "compiled.resolved_bindings"))
    source_rows = tuple(_mapping(item, "compiled.expanded_branch_row") for item in _sequence(compiled.get("expanded_branch_rows"), "compiled.expanded_branch_rows"))
    expanded = canonical_expanded_graph_sha256(
        plan_sha256=plan_sha,
        node_order=node_order,
        resolved_bindings=bindings,
        expanded_branch_rows=source_rows,
    )
    if compiled.get("expanded_graph_sha256") != expanded:
        raise _fail("compiled expanded-graph identity is invalid")
    compiler = _string(compiled.get("compiled_graph_sha256"), "compiled.compiled_graph_sha256")
    rows = tuple(_compiled_row_projection(row) for row in source_rows)
    connectivity = _mapping(semantic.get("connectivity"), "connectivity")
    ports = []
    for raw in _sequence(connectivity.get("ports"), "ports"):
        port = _mapping(raw, "port")
        node = _string(port.get("net"), "port.net")
        if node not in node_order:
            raise _fail("compiled Port node is absent from compiler node order", port=port.get("id"))
        ports.append(
            {
                "port_id": _string(port.get("id"), "port.id"),
                "node_id": node,
                "role": _string(port.get("role"), "port.role"),
                "orientation": "node_to_reference",
                "reference_impedance": _plain(port.get("reference_impedance")),
                "reference_node": "ground",
                "load_kind": "raw_reference_impedance",
            }
        )
    active = [row for row in rows if row.get("kind") != "transmission_line_audit" and row.get("omitted_as_zero") is not True]
    audits = [row for row in rows if row.get("kind") == "transmission_line_audit"]
    omissions = [row for row in rows if row.get("kind") != "transmission_line_audit" and row.get("omitted_as_zero") is True]
    mutual = [row for row in active if row.get("kind") == "mutual_inductance"]
    physical = [row for row in active if row.get("kind") != "mutual_inductance"]
    electrical = cast(Mapping[str, object], _freeze({
        "schema": "scnsim.diagram_connectivity_manifest", "schema_version": 1,
        "representation": "compiled", "node_order": list(node_order),
        "expanded_terms": physical, "transmission_lines": audits,
        "couplings": mutual, "omissions": omissions,
        "ports": sorted(ports, key=lambda row: cast(str, row["port_id"])),
    }))
    structural = cast(Mapping[str, object], _freeze({
        "schema": "scnsim.diagram_semantic_manifest", "schema_version": 1,
        "representation": "compiled",
        "compiler_hierarchy": [{"component_path": row["component_path"], "kind": "pi_ladder", "n_sections": row["n_sections"], "conductors": row["conductors"]} for row in audits],
        "expanded_membership": [{"component_path": row.get("component_path"), "compiled_kind": row["kind"], "section": row.get("section"), "station": row.get("station"), "row_conductor": row.get("row_conductor"), "column_conductor": row.get("column_conductor"), "omitted_as_zero": row.get("omitted_as_zero")} for row in rows if row.get("kind") != "transmission_line_audit"],
        "matrix_evidence": audits, "couplings": mutual, "omissions": omissions,
    }))
    digests = canonical_diagram_digests(point.snapshot, representation="compiled")
    verified = cast(Mapping[str, object], _freeze({
        "plan_id": semantic["plan_id"], **digests,
        "parameters_sha256": parameters_sha,
        "compiled_graph_sha256": compiler,
        "expanded_graph_sha256": expanded,
        "resolved_bindings": _plain(bindings),
        "source_provenance": _plain(point.snapshot.source_provenance),
        "canonical_values": {},
        "source_rows": _v2_verified_source_rows(point, semantic),
    }))
    return _PointExpectedManifest("compiled", electrical, structural, MappingProxyType({}), verified, compiler, expanded)


def build_point_expected(
    point: ResolvedPlanPoint,
    *,
    representation: _Representation = "authoring",
    compiled_evidence: Mapping[str, object] | None = None,
) -> _PointExpectedManifest:
    """Build V2 expectations only from one immutable point/compiler handoff."""

    if representation == "authoring":
        if compiled_evidence is not None:
            raise TypeError("authoring witness does not accept compiled evidence")
        return _v2_authoring(point)
    if representation != "compiled":
        raise ValueError("representation must be 'authoring' or 'compiled'")
    if compiled_evidence is None:
        raise TypeError("compiled witness requires exact same-point compiler evidence")
    return _v2_compiled(point, compiled_evidence)
