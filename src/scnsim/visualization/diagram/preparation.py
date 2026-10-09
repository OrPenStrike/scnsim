"""Capture immutable authoring inventory for explicit schematic layouts.

This module binds diagram targets to one resolved Plan point.  It measures
native fragments only; a caller supplies every pose, boundary, caption anchor,
junction, and route later through :class:`SchematicLayout`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path

from ...authoring.identity import canonical_diagram_digests, canonical_parameters_sha256
from ...authoring.snapshot import AuthoringSnapshot, ResolvedPlanPoint
from ...canonical import canonical_json_bytes, float64_hex, sha256_hex
from ...errors import SCNSimValidationError
from ..schematic import (
    DiagramRef,
    MeasuredFragment,
    ScopeInventory,
    SchematicPreparation,
    _scope_layout_sha256,
)
from .metrics import DEFAULT_METRICS, _font_faces, shape_text
from .native import fragment_text, measure_native
from .scene import Point


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_layout", evidence=evidence)


def _plain_path(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or any(not isinstance(part, str) or not part for part in value):
        raise _fail("captured diagram path is malformed", field=field)
    return tuple(value)


def _rows(value: object, *, field: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (tuple, list)) or any(not isinstance(row, Mapping) for row in value):
        raise _fail("captured diagram rows are malformed", field=field)
    return tuple(value)


def _key(value: Mapping[str, object]) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _ref(
    preparation_sha256: str,
    kind: str,
    path: tuple[str, ...],
    *,
    scope_path: tuple[str, ...] | None = None,
    **identity: object,
) -> DiagramRef:
    document = {"kind": kind, "path": list(path), **identity}
    return DiagramRef._create(
        preparation_sha256,
        _key(document),
        kind,
        path if scope_path is None else scope_path,
    )


def _scope_rows(root: Mapping[str, object]):
    """Return source-order scopes, with child-body ownership kept explicit."""
    output = []
    seen = set()

    def visit(scope: Mapping[str, object], parent: tuple[str, ...] | None) -> None:
        path = _plain_path(scope.get("path"), field="scope.path")
        if path in seen:
            raise _fail("captured Plan repeats a scope path", path=path)
        seen.add(path)
        output.append((path, parent, scope))
        for child in _rows(scope.get("children", ()), field="scope.children"):
            visit(child, path)
        for body_row in _rows(scope.get("component_bodies", ()), field="scope.component_bodies"):
            body = body_row.get("body")
            if not isinstance(body, Mapping):
                raise _fail("captured component body has no scope record", path=path)
            visit(body, path)

    visit(root, None)
    return tuple(output)


def _metric_record() -> Mapping[str, str]:
    names = (
        "unit_length", "native_span", "label_clearance", "terminal_stub",
        "obstacle_clearance", "routing_lane_pitch", "port_lead",
        "port_load_span", "port_circle_radius", "polarity_mark_span",
        "ground_stem", "ground_bar_step", "ground_half_width",
        "grounded_branch_depth", "panel_gap", "junction_stagger",
        "jump_gap", "jump_height", "primary_text_size", "port_id_text_size",
        "secondary_text_size", "tertiary_text_size", "figure_inches_per_unit",
        "symbol_linewidth", "conductive_linewidth", "annotation_linewidth",
    )
    return {name: float64_hex(float(getattr(DEFAULT_METRICS, name))) for name in names}


_SOURCE_IDENTITY_PATHS = (
    "visualization/schematic.py",
    "visualization/quantity_presentation.py",
    "visualization/diagram/spec.py",
    "visualization/diagram/preparation.py",
    "visualization/diagram/structured.py",
    "visualization/diagram/composition_geometry.py",
    "visualization/diagram/composition_obstacles.py",
    "visualization/diagram/native.py",
    "visualization/diagram/metrics.py",
    "visualization/diagram/scene.py",
    "visualization/diagram/audit.py",
    "visualization/diagram/expected.py",
    "visualization/diagram/reconstruction.py",
    "visualization/diagram/witness.py",
    "visualization/diagram/drawing.py",
    "visualization/diagram/pipeline.py",
    "visualization/diagram/routing.py",
    "visualization/diagram/values.py",
    "authoring/assembly.py",
    "authoring/capture.py",
    "authoring/identity.py",
    "authoring/snapshot.py",
    "authoring/physical_values.py",
)


def _diagram_source_identity() -> tuple[Mapping[str, str], ...]:
    package_root = Path(__file__).resolve().parents[2]
    return tuple(
        {
            "path": f"scnsim/{relative_path}",
            "sha256": sha256((package_root / relative_path).read_bytes()).hexdigest(),
        }
        for relative_path in _SOURCE_IDENTITY_PATHS
    )


def _font_identity() -> tuple[Mapping[str, str], ...]:
    return tuple(
        {"family": face.family, "sha256": face.sha256}
        for face in _font_faces()
    )


def _source_kind_for_native(target: DiagramRef, text_path: tuple[object, ...]) -> str:
    if target.kind == "port":
        return "port_label"
    field_name = text_path[-1] if text_path else ""
    if field_name in {"visible_name", "title"}:
        return "occurrence_name"
    if field_name == "value" or field_name in {"length_label", "section_label"}:
        return "occurrence_value"
    return "structure_caption"


def _native_source(target: DiagramRef) -> Mapping[str, object]:
    value = json.loads(target.key)
    return value


def _caption_ref(
    preparation_sha256: str,
    owner_path: tuple[str, ...],
    source_kind: str,
    source_id: object,
) -> DiagramRef:
    return _ref(preparation_sha256, "caption", owner_path,
                source_kind=source_kind, id=source_id)


def _public_analysis_labels(semantic: Mapping[str, object]):
    from .expected import _v2_net_map, _v2_public_analysis_labels

    net_map, _ = _v2_net_map(semantic)
    return tuple(_v2_public_analysis_labels(semantic["scope_hierarchy"], net_map)), net_map


def _detached_point(point: ResolvedPlanPoint) -> ResolvedPlanPoint:
    """Drop capture-only mutable authoring handles from a retained preparation."""
    snapshot = AuthoringSnapshot.create(
        semantic_record=point.snapshot.semantic_record,
        source_provenance=point.snapshot.source_provenance,
    )
    return ResolvedPlanPoint.create(
        snapshot=snapshot,
        effective_parameters=point.effective_parameters,
        parameter_record=point.parameter_record,
        resolved_fields=point.resolved_fields,
    )


def capture_schematic(plan: object, spec: object, parameters: object | None) -> SchematicPreparation:
    """Capture one exact resolved point and its native target inventory."""
    from .spec import CircuitDiagramSpec

    if not isinstance(spec, CircuitDiagramSpec):
        raise TypeError("schematic preparation requires CircuitDiagramSpec")
    if spec.representation != "authoring" or spec.layout is not None:
        raise _fail("schematic preparation requires authoring presentation options without a layout")

    snapshot = plan._capture_authoring_snapshot()
    point = plan._resolve_parameter_point(parameters, snapshot=snapshot)
    if not isinstance(point, ResolvedPlanPoint):
        raise TypeError("Plan resolution did not return ResolvedPlanPoint")
    semantic = point.snapshot.semantic_record
    if semantic.get("schema") != "scnsim.authoring_snapshot" or semantic.get("schema_version") != 2:
        raise _fail("schematic preparation requires normalized authoring snapshot version 2")

    digests = canonical_diagram_digests(point.snapshot, representation="authoring")
    parameters_sha = canonical_parameters_sha256(point.parameter_record)
    source_sha = sha256_hex({
        "schema": "scnsim.schematic_source_identity",
        "schema_version": 1,
        "source_provenance": point.snapshot.source_provenance,
        "diagram_sources": _diagram_source_identity(),
    })
    metrics = _metric_record()
    fonts = _font_identity()
    metrics_sha = sha256_hex({
        "schema": "scnsim.diagram_metrics",
        "schema_version": 1,
        "values": metrics,
        "fonts": fonts,
    })
    options = {
        "representation": spec.representation,
        "theme": spec.theme.value,
        "show_parameter_values": spec.show_parameter_values,
        "show_provenance": spec.show_provenance,
    }
    options_sha = sha256_hex({"schema": "scnsim.schematic_presentation_options", "schema_version": 1, **options})
    identity_core = {
        "schema": "scnsim.schematic_preparation",
        "schema_version": 1,
        "plan_sha256": digests["plan_sha256"],
        "parameters_sha256": parameters_sha,
        "source_sha256": source_sha,
        "metrics_sha256": metrics_sha,
        "presentation_options_sha256": options_sha,
    }
    preparation_sha = sha256_hex(identity_core)
    identity = {**identity_core, "preparation_sha256": preparation_sha}

    scope_records = _scope_rows(semantic["scope_hierarchy"])
    scope_refs = {
        path: _ref(preparation_sha, "scope", path)
        for path, _, _ in scope_records
    }

    leaf_rows = _rows(semantic.get("physical_leaves"), field="physical_leaves")
    leaves_by_path = {
        _plain_path(leaf.get("path"), field="physical_leaf.path"): leaf
        for leaf in leaf_rows
    }
    leaves_by_scope: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    leaf_refs: dict[tuple[str, ...], DiagramRef] = {}
    for leaf in leaf_rows:
        leaf_path = _plain_path(leaf.get("path"), field="physical_leaf.path")
        owner_path = leaf_path[:-1]
        if owner_path not in scope_refs:
            raise _fail("physical leaf has no captured owning scope", path=leaf_path)
        leaves_by_scope[owner_path].append(leaf)
        leaf_refs[leaf_path] = _ref(preparation_sha, "occurrence", leaf_path)

    connectivity = semantic.get("connectivity")
    if not isinstance(connectivity, Mapping):
        raise _fail("captured authoring point has no connectivity record")
    physical_rows = _rows(connectivity.get("physical_endpoints"), field="connectivity.physical_endpoints")
    final_by_pin = {
        (_plain_path(row.get("path"), field="physical_endpoint.path"), row.get("pin")): row.get("net")
        for row in physical_rows
    }
    final_by_endpoint = {
        canonical_json_bytes(row.get("endpoint")).decode("utf-8"): row.get("final_net")
        for row in _rows(connectivity.get("endpoint_nets", ()), field="connectivity.endpoint_nets")
        if isinstance(row.get("endpoint"), Mapping)
    }
    final_by_exposure: dict[tuple[tuple[str, ...], str], str] = {}
    final_by_bus: dict[tuple[tuple[str, ...], str], str] = {}
    final_by_tap: dict[tuple[tuple[str, ...], str, str], str] = {}
    final_by_coordinate: dict[tuple[tuple[str, ...], str], str] = {}
    for path, _, scope in scope_records:
        for bus in _rows(scope.get("buses", ()), field="scope.buses"):
            bus_id, final_net = bus.get("id"), bus.get("final_net")
            if isinstance(bus_id, str) and isinstance(final_net, str):
                final_by_bus[path, bus_id] = final_net
            for tap in _rows(bus.get("taps", ()), field="scope.bus.taps"):
                tap_id, tap_net = tap.get("id"), tap.get("final_net")
                if isinstance(bus_id, str) and isinstance(tap_id, str) and isinstance(tap_net, str):
                    final_by_tap[path, bus_id, tap_id] = tap_net
        exposures = scope.get("exposures", {})
        if isinstance(exposures, Mapping):
            for exposure in _rows(exposures.get("pins", ()), field="scope.exposures.pins"):
                if isinstance(exposure.get("id"), str) and isinstance(exposure.get("final_net"), str):
                    final_by_exposure[path, exposure["id"]] = exposure["final_net"]
            for coordinate in _rows(exposures.get("coordinates", ()), field="scope.exposures.coordinates"):
                if isinstance(coordinate.get("id"), str) and isinstance(coordinate.get("final_net"), str):
                    final_by_coordinate[path, coordinate["id"]] = coordinate["final_net"]
    pin_refs: dict[tuple[tuple[str, ...], str], DiagramRef] = {}
    for leaf in leaf_rows:
        path = _plain_path(leaf.get("path"), field="physical_leaf.path")
        for pin in leaf.get("pin_order", ()):
            if not isinstance(pin, str):
                raise _fail("physical pin identifier is malformed", path=path)
            pin_refs[path, pin] = _ref(preparation_sha, "contact", path,
                contact_kind="pin", id=pin)

    ports_by_scope: dict[tuple[str, ...], list[tuple[DiagramRef, Mapping[str, object]]]] = defaultdict(list)
    port_rows = _rows(connectivity.get("ports", ()), field="connectivity.ports")
    for port in port_rows:
        port_id = port.get("id")
        if not isinstance(port_id, str) or not port_id:
            raise _fail("captured Port identifier is malformed")
        ref = _ref(preparation_sha, "port", (), id=port_id)
        ports_by_scope[()].append((ref, port))

    boundary_rows: dict[tuple[str, ...], list[tuple[DiagramRef, Mapping[str, object]]]] = defaultdict(list)
    for path, _, scope in scope_records:
        exposures = scope.get("exposures", {})
        if not isinstance(exposures, Mapping):
            raise _fail("captured scope exposure record is malformed", scope=path)
        for exposure in _rows(exposures.get("pins", ()), field="scope.exposures.pins"):
            exposure_id = exposure.get("id")
            if not isinstance(exposure_id, str) or not exposure_id:
                raise _fail("captured boundary pin identifier is malformed", scope=path)
            ref = _ref(preparation_sha, "contact", path,
                contact_kind="boundary", id=exposure_id)
            boundary_rows[path].append((ref, exposure))

    ground_rows: dict[tuple[str, ...], list[tuple[DiagramRef, Mapping[str, object]]]] = defaultdict(list)
    ground_structure_sources: dict[
        tuple[tuple[str, ...], str], list[Mapping[str, object]]
    ] = defaultdict(list)
    for owner_path, _, scope in scope_records:
        for structure in _rows(scope.get("structures", ()), field="scope.structures"):
            kind = structure.get("kind")
            if kind not in {"series", "parallel", "branch"}:
                continue
            endpoint_fields = ("start", "end") if kind in {"series", "parallel"} else ("at", "end")
            if kind == "parallel":
                branches = _rows(structure.get("branches", ()), field="structure.branches")
                member_groups = tuple(
                    _rows(branch.get("elements", ()), field="structure.branch.elements")
                    for branch in branches
                )
            else:
                member_groups = (_rows(structure.get("elements", ()), field="structure.elements"),)
            for endpoint_field in endpoint_fields:
                endpoint = structure.get(endpoint_field)
                if not isinstance(endpoint, Mapping) or endpoint.get("kind") != "ground":
                    continue
                member_index = 0 if endpoint_field in {"start", "at"} else -1
                pin_field = "pin_1" if member_index == 0 else "pin_2"
                for branch_index, members in enumerate(member_groups):
                    member = members[member_index]
                    member_path = _plain_path(member.get("path"), field="structure.member.path")
                    pin = member.get(pin_field)
                    ground_structure_sources[member_path, pin].append({
                        "scope": list(owner_path),
                        "structure": structure,
                        "endpoint_field": endpoint_field,
                        "branch_index": branch_index,
                        "member": member,
                        "pin": pin,
                    })
    ground_groups = point.snapshot.source_provenance.get("ground_pins_call_groups", ())
    if not isinstance(ground_groups, (tuple, list)):
        raise _fail("captured ground groups are malformed")
    for group_index, group in enumerate(ground_groups):
        for endpoint in _rows(group, field="ground_pins_call_groups"):
            endpoint_kind = endpoint.get("kind")
            endpoint_scope = _plain_path(endpoint.get("scope", ()), field="ground.endpoint.scope")
            if endpoint_kind == "pin":
                pin = endpoint.get("id")
                if not isinstance(pin, str) or not pin:
                    raise _fail("captured ground pin identifier is malformed", endpoint=endpoint)
                component = endpoint.get("component")
                if isinstance(component, str) and component:
                    owner_path = endpoint_scope
                    key_id: object = pin
                    component_path = (*endpoint_scope, component)
                    if component_path in scope_refs:
                        source_kind = "exposed_pin"
                        key_path = component_path
                        final_net = final_by_endpoint.get(canonical_json_bytes(endpoint).decode("utf-8"))
                    elif component_path in leaves_by_path:
                        source_kind = "physical_pin"
                        key_path = component_path
                        final_net = final_by_pin.get((key_path, pin))
                    else:
                        raise _fail("captured ground pin has no physical or exposed owner", endpoint=endpoint)
                elif endpoint.get("public") is True:
                    source_kind = "exposed_pin"
                    key_path = endpoint_scope
                    owner_path = endpoint_scope[:-1]
                    key_id = pin
                    final_net = final_by_exposure.get((endpoint_scope, pin))
                else:
                    raise _fail("captured ground endpoint has no exact pin owner", endpoint=endpoint)
            elif endpoint_kind == "bus":
                source_kind = "bus"
                owner_path = endpoint_scope
                key_path = endpoint_scope
                key_id = endpoint.get("id")
                final_net = final_by_bus.get((endpoint_scope, key_id)) if isinstance(key_id, str) else None
            elif endpoint_kind == "tap":
                source_kind = "tap"
                owner_path = endpoint_scope
                key_path = endpoint_scope
                key_id = {"bus": endpoint.get("bus"), "id": endpoint.get("id")}
                bus_id, tap_id = endpoint.get("bus"), endpoint.get("id")
                final_net = final_by_tap.get((endpoint_scope, bus_id, tap_id)) if isinstance(bus_id, str) and isinstance(tap_id, str) else None
            elif endpoint_kind == "coordinate":
                source_kind = "coordinate"
                owner_path = endpoint_scope
                key_path = endpoint_scope
                key_id = endpoint.get("id")
                final_net = final_by_coordinate.get((endpoint_scope, key_id)) if isinstance(key_id, str) else None
            else:
                raise _fail("captured ground endpoint has an unsupported source kind", endpoint=endpoint)
            if not owner_path in scope_refs:
                raise _fail("captured ground glyph has no owning scope", endpoint=endpoint, scope=owner_path)
            key_fields = {"id": key_id}
            if source_kind != "physical_pin":
                key_fields["endpoint_kind"] = source_kind
            target = _ref(
                preparation_sha,
                "ground",
                key_path,
                scope_path=owner_path,
                **key_fields,
            )
            if not isinstance(final_net, str):
                raise _fail("captured ground endpoint has no final connectivity record", endpoint=endpoint)
            ground_rows[owner_path].append((target, {
                "kind": "ground_endpoint", "group_index": group_index,
                "endpoint": endpoint, "endpoint_kind": source_kind,
                "final_net": final_net,
            }))

    # Structural declarations can attach a chain or branch directly to the
    # Plan ground without a separate ground_pins() call.  The resolved
    # physical endpoint table is the source authority for those returns; add
    # one drawable native ground target per actual grounded terminal, while
    # reusing targets already captured from explicit ground_pins() groups.
    captured_ground_refs = {
        ref for rows in ground_rows.values() for ref, _ in rows
    }
    for endpoint_row in physical_rows:
        if endpoint_row.get("net") != "ground":
            continue
        endpoint_path = _plain_path(endpoint_row.get("path"), field="physical_endpoint.path")
        pin = endpoint_row.get("pin")
        owner_path = endpoint_path[:-1]
        if owner_path not in scope_refs:
            raise _fail("captured physical ground has no owning scope", endpoint=endpoint_row)
        target = _ref(
            preparation_sha,
            "ground",
            endpoint_path,
            scope_path=owner_path,
            id=pin,
        )
        if target in captured_ground_refs:
            continue
        ground_rows[owner_path].append((target, {
            "kind": "ground_endpoint",
            "endpoint_kind": "physical_pin",
            "endpoint": endpoint_row,
            "structure_sources": tuple(
                ground_structure_sources.get((endpoint_path, pin), ())
            ),
            "final_net": "ground",
        }))
        captured_ground_refs.add(target)

    # Native fragments are measured before a layout exists; every cardinal
    # variant is independently produced from the same resolved point.
    native_targets: list[tuple[DiagramRef, tuple[str, ...], Mapping[str, object]]] = []
    for path, rows in leaves_by_scope.items():
        for leaf in rows:
            target = leaf_refs[_plain_path(leaf.get("path"), field="physical_leaf.path")]
            native_targets.append((target, path, leaf))
    for path, rows in ports_by_scope.items():
        native_targets.extend((ref, path, row) for ref, row in rows)
    for path, rows in ground_rows.items():
        native_targets.extend((ref, path, row) for ref, row in rows if ref.kind == "ground")

    native: dict[tuple[DiagramRef, int], MeasuredFragment] = {}
    for target, _, _ in native_targets:
        for orientation in (0, 90, 180, 270):
            native[target, orientation] = measure_native(
                point, target, orientation=orientation,
                show_values=spec.show_parameter_values,
                metrics=DEFAULT_METRICS,
            )

    # A global lookup from final nets to captured contact refs is used only to
    # expose source-owned analysis aliases and ordered required-contact sets.
    net_refs: dict[str, list[DiagramRef]] = defaultdict(list)
    for (path, pin), ref in pin_refs.items():
        net = final_by_pin.get((path, pin))
        if isinstance(net, str):
            net_refs[net].append(ref)
    for ref, port in ports_by_scope.get((), ()):
        net = port.get("net")
        if isinstance(net, str):
            net_refs[net].append(ref)
    for path, rows in boundary_rows.items():
        for ref, row in rows:
            net = row.get("final_net")
            if isinstance(net, str):
                net_refs[net].append(ref)
    for path, rows in ground_rows.items():
        for ref, source in rows:
            net = source.get("final_net")
            if isinstance(net, str):
                net_refs[net].append(ref)

    def scoped_contacts(scope_path: tuple[str, ...], net: str) -> tuple[DiagramRef, ...]:
        """Return same-net contacts that the owning scope can actually route to."""
        refs: list[DiagramRef] = []
        for leaf in leaves_by_scope.get(scope_path, ()):
            leaf_path = _plain_path(leaf.get("path"), field="physical_leaf.path")
            for pin in leaf.get("pin_order", ()):
                ref = pin_refs[leaf_path, pin]
                if final_by_pin.get((leaf_path, pin)) == net:
                    refs.append(ref)
        for ref, exposure in boundary_rows.get(scope_path, ()):
            if exposure.get("final_net") == net:
                refs.append(ref)
        for child_path, _, _ in scope_records:
            if len(child_path) == len(scope_path) + 1 and child_path[:-1] == scope_path:
                for ref, exposure in boundary_rows.get(child_path, ()):
                    if exposure.get("final_net") == net:
                        refs.append(ref)
        for ref, port in ports_by_scope.get(scope_path, ()):
            if port.get("net") == net:
                refs.append(ref)
        for ref, source in ground_rows.get(scope_path, ()):
            if source.get("final_net") == net:
                refs.append(ref)
        return tuple(dict.fromkeys(refs))

    captions_by_scope: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for target, owner_path, native_source_record in native_targets:
        base = native[target, 0]
        source_doc = _native_source(target)
        for text_path, run in fragment_text(base.scene).items():
            source_kind = _source_kind_for_native(target, tuple(text_path))
            ref = _caption_ref(preparation_sha, owner_path, source_kind,
                {"source": source_doc, "text_path": list(text_path)})
            row = {
                "ref": ref,
                "target": target,
                "text_path": tuple(text_path),
                "source": {"kind": "native_field", "target": source_doc,
                           "record": native_source_record},
                "text": run.text,
                "role": run.role,
                "size": run.size,
                "run": run,
            }
            captions_by_scope[owner_path].append(row)

    for path, _, scope in scope_records:
        # A child scope is an occurrence-local view of its template. Its
        # visible ownership identity comes from the rebased path, while the
        # captured scope record retains the original template identity.
        title = scope.get("id") if not path else path[-1]
        if not isinstance(title, str) or not title:
            raise _fail("captured scope has no source title", scope=path)
        run = shape_text(
            title, at=Point(0.0, 0.0), size=1.25 * DEFAULT_METRICS.primary_text_size,
            role="region-id",
        )
        ref = _caption_ref(preparation_sha, path, "scope_title", {"scope": list(path), "id": title})
        captions_by_scope[path].append({
            "ref": ref, "target": scope_refs[path], "text_path": None,
            "source": {"kind": "scope_title", "scope": scope},
            "text": title, "role": run.role, "size": run.size, "run": run,
        })
        for boundary_ref, exposure in (
            boundary_rows.get(path, ())
            if len(boundary_rows.get(path, ())) > 1 else ()
        ):
            pin_id = exposure["id"]
            run = shape_text(
                pin_id, at=Point(0.0, 0.0), size=DEFAULT_METRICS.tertiary_text_size,
                role="public-pin",
            )
            ref = _caption_ref(preparation_sha, path, "boundary", {"scope": list(path), "id": pin_id})
            captions_by_scope[path].append({
                "ref": ref, "target": boundary_ref, "text_path": None,
                "source": {"kind": "scope_exposure", "scope": list(path), "record": exposure},
                "text": pin_id, "role": run.role, "size": run.size, "run": run,
            })

    analysis_by_scope: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    analysis_labels, visible_net_map = _public_analysis_labels(semantic)
    source_for_visible = {visible: source for source, visible in visible_net_map.items()}
    for label in analysis_labels:
        scope_path = _plain_path(label.get("scope"), field="analysis_label.scope")
        net = label.get("net")
        if not isinstance(net, str):
            raise _fail("captured analysis label has no final net", scope=scope_path)
        run = shape_text(
            label["text"], at=Point(0.0, 0.0), size=DEFAULT_METRICS.tertiary_text_size,
            role="analysis-label",
        )
        ref = _caption_ref(preparation_sha, scope_path, "analysis_label", label)
        source_net = source_for_visible.get(net)
        analysis_by_scope[scope_path].append({
            "ref": ref,
            "contacts": () if source_net is None else scoped_contacts(scope_path, source_net),
            "text": label["text"],
            "role": run.role,
            "size": run.size,
            "run": run,
            "source": {"kind": "analysis_label", "record": label},
        })

    structures_by_scope: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for path, _, scope in scope_records:
        for structure in _rows(scope.get("structures", ()), field="scope.structures"):
            if structure.get("kind") != "coupling":
                structures_by_scope[path].append({"kind": structure.get("kind"), "source": structure})
                continue
            branch_rows = []
            positive_contacts = []
            for field_name in ("inductor_a", "inductor_b"):
                branch = structure.get(field_name)
                if not isinstance(branch, Mapping):
                    raise _fail("captured coupling branch is malformed", scope=path, coupling=structure.get("id"))
                branch_path = _plain_path(branch.get("path"), field=f"coupling.{field_name}.path")
                branch_id = branch.get("branch_id")
                if not isinstance(branch_id, str) or not branch_id:
                    raise _fail("captured coupling branch identifier is malformed", scope=path)
                leaf = leaves_by_path.get(branch_path)
                if leaf is None:
                    raise _fail("captured coupling branch has no physical occurrence", branch=branch)
                oriented = next(
                    (row for row in leaf.get("oriented_branches", ()) if row.get("id") == branch_id),
                    None,
                )
                if not isinstance(oriented, Mapping):
                    raise _fail("captured coupling branch has no resolved orientation", branch=branch)
                positive_pin = oriented.get("positive_pin")
                contact_ref = pin_refs.get((branch_path, positive_pin))
                if contact_ref is None:
                    raise _fail("captured coupling positive terminal has no contact ref", branch=branch)
                branch_rows.append({"path": list(branch_path), "branch_id": branch_id})
                positive_contacts.append(contact_ref)
            coupling_id = structure.get("id")
            if not isinstance(coupling_id, str) or not coupling_id:
                raise _fail("captured coupling has no source id", scope=path)
            target = _ref(
                preparation_sha,
                "structure",
                path,
                structure_kind="coupling",
                id=coupling_id,
            )
            label = "COUPLING:" + coupling_id
            coefficient = structure.get("coupling_coefficient")
            if not isinstance(coefficient, Mapping):
                raise _fail("captured coupling coefficient is malformed", coupling=coupling_id)
            if spec.show_parameter_values:
                from .values import format_envelope
                label += "; k=" + format_envelope(coefficient)
            else:
                value = coefficient.get("si_value_f64")
                from ...canonical import float64_from_hex
                magnitude = float64_from_hex(value)
                label += "; k" + ("<0" if magnitude < 0 else ">0" if magnitude > 0 else "=0")
            run = shape_text(
                label, at=Point(0.0, 0.0), size=DEFAULT_METRICS.tertiary_text_size,
                role="structure-id",
            )
            structures_by_scope[path].append({
                "ref": target,
                "kind": "coupling",
                "branches": tuple(branch_rows),
                "positive_contacts": tuple(positive_contacts),
                "text": label,
                "role": run.role,
                "size": run.size,
                "run": run,
                "source": structure,
            })

    inventories = []
    for path, parent, scope in scope_records:
        contacts = []
        for leaf in leaves_by_scope.get(path, ()):
            leaf_path = _plain_path(leaf["path"], field="physical_leaf.path")
            target = leaf_refs[leaf_path]
            measured = native[target, 0]
            for pin in leaf.get("pin_order", ()):
                ref = pin_refs[leaf_path, pin]
                point_record = measured.contacts.get(ref)
                if not isinstance(point_record, Mapping):
                    raise _fail("native measurement omitted a captured physical contact", path=leaf_path, pin=pin)
                contacts.append({
                    "ref": ref, "side": point_record["side"],
                    "source": {"kind": "physical_terminal", "path": list(leaf_path), "pin": pin},
                    "boundary": False,
                })
        for ref, exposure in boundary_rows.get(path, ()):
            contacts.append({
                "ref": ref, "side": None,
                "source": {"kind": "scope_exposure", "scope": list(path), "record": exposure},
                "boundary": True,
            })
        direct_children = []
        for child_path, _, _ in scope_records:
            if len(child_path) == len(path) + 1 and child_path[:-1] == path:
                direct_children.append(scope_refs[child_path])
        # Component-body scopes are immediate children and their exposed pin
        # refs are the parent's actual component contacts.
        child_boundary_refs = []
        for child_path in [tuple(json.loads(ref.key)["path"]) for ref in direct_children]:
            for boundary_ref, exposure in boundary_rows.get(child_path, ()):
                contacts.append({
                    "ref": boundary_ref, "side": None,
                    "source": {"kind": "child_scope_exposure", "scope": list(child_path), "record": exposure},
                    "boundary": False,
                })
                child_boundary_refs.append(boundary_ref)
        for port_ref, port_row in ports_by_scope.get(path, ()):
            measured = native[port_ref, 0]
            row = measured.contacts.get(port_ref)
            if not isinstance(row, Mapping):
                raise _fail("native Port measurement omitted its external contact", port=port_ref.key)
            contacts.append({
                "ref": port_ref, "side": row["side"],
                "source": {"kind": "port", "record": port_row}, "boundary": False,
            })
        local_ground_refs = []
        for ground_ref, ground_source in ground_rows.get(path, ()):
            measured = native[ground_ref, 0]
            row = measured.contacts.get(ground_ref)
            if not isinstance(row, Mapping):
                raise _fail("native ground measurement omitted its terminal", ground=ground_ref.key)
            contacts.append({
                "ref": ground_ref, "side": row["side"],
                "source": ground_source, "boundary": False,
            })
            local_ground_refs.append(ground_ref)

        contact_refs = {row["ref"] for row in contacts}
        required = []
        for net, refs in sorted(net_refs.items()):
            local = tuple(ref for ref in refs if ref in contact_refs)
            if local:
                required.append({"net": net, "contacts": local})
        structures = tuple(structures_by_scope.get(path, ()))
        inventory = ScopeInventory(
            ref=scope_refs[path],
            parent=None if parent is None else scope_refs[parent],
            physical_occurrences=tuple(leaf_refs[_plain_path(row["path"], field="physical_leaf.path")] for row in leaves_by_scope.get(path, ())),
            child_scopes=tuple(direct_children),
            contacts=tuple(contacts),
            ports=tuple(ref for ref, _ in ports_by_scope.get(path, ())),
            grounds=tuple(local_ground_refs),
            captions=tuple(captions_by_scope.get(path, ())),
            analysis_labels=tuple(analysis_by_scope.get(path, ())),
            structures=structures,
            required_contact_sets=tuple(required),
        )
        inventories.append(inventory)

    return SchematicPreparation._from_capture(
        identity=identity,
        root=scope_refs[()],
        scopes=tuple(inventories),
        native=native,
        point=_detached_point(point),
        spec=spec,
    )


def measure_scope(preparation: SchematicPreparation, scope_ref, scope_layout):
    """Realize one explicitly declared scope bottom-up with exact memoization."""
    from ..schematic import ScopeMeasurement, SchematicScopeLayout

    if not isinstance(scope_ref, DiagramRef):
        raise TypeError("measure_scope requires a captured DiagramRef")
    if scope_ref.preparation_sha256 != preparation.identity["preparation_sha256"]:
        raise _fail("scope ref belongs to a different preparation", ref=scope_ref.key)
    if not isinstance(scope_layout, SchematicScopeLayout) or scope_layout.scope != scope_ref:
        raise TypeError("measure_scope requires the matching captured scope layout")
    layout_sha = _scope_layout_sha256(scope_layout)
    memo_key = (preparation.identity["preparation_sha256"], scope_ref.key, layout_sha)
    with preparation._measurement_lock:
        cached = preparation._measurements.get(memo_key)
        if cached is not None:
            return cached
        scope_inventory = next((row for row in preparation.scopes if row.ref == scope_ref), None)
        if scope_inventory is None:
            raise _fail("layout names a scope absent from captured inventory", scope=scope_ref.key)
        expected_children = set(scope_inventory.child_scopes)
        if set(scope_layout.children) != expected_children:
            raise _fail("scope layout must declare each captured child exactly once",
                        scope=scope_ref.key,
                        missing=tuple(ref.key for ref in expected_children - set(scope_layout.children)),
                        extra=tuple(ref.key for ref in set(scope_layout.children) - expected_children))
        children = {
            child_ref: measure_scope(preparation, child_ref, child_layout)
            for child_ref, child_layout in scope_layout.children.items()
        }
        native_refs = set(scope_inventory.physical_occurrences) | set(scope_inventory.ports) | set(scope_inventory.grounds)
        native_measurements = {
            key: value for key, value in preparation.native.items()
            if key[0] in native_refs
        }
        from .composition_geometry import realize_explicit_scope
        realization = realize_explicit_scope(
            preparation, scope_layout,
            native_measurements=native_measurements,
            children=children,
        )
        if not isinstance(realization, ScopeMeasurement):
            raise TypeError("explicit scope realization must return ScopeMeasurement")
        result = realization
        # Assignment is the memo publication point after complete realization.
        preparation._measurements[memo_key] = result
        return result


__all__ = ["capture_schematic", "measure_scope"]
