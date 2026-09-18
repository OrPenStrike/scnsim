"""Immutable presentation inventory and recipes over one authoring snapshot.

This module records owner-local Block contacts, not electrical graph edits.
Series midpoints, Parallel rails, and built-Composite implementation ownership
remain with the authoring projection. No mutable authoring handles enter a
captured recipe or the geometry lowerer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import TypeAlias

from .._authoring_snapshot import AuthoringSnapshot, freeze
from .._canonical import canonical_plan_snapshot, sha256_hex
from ..errors import SCNSimValidationError

Key: TypeAlias = tuple[object, ...]
GroupKey: TypeAlias = tuple[tuple[str, ...], str]


def endpoint_key(endpoint: Mapping[str, object]) -> Key:
    """Use the same typed key shape as public layout capture."""
    kind = endpoint["kind"]
    scope = tuple(endpoint.get("scope", ()))
    if kind == "pin":
        return (kind, scope, endpoint["component"], endpoint["id"], endpoint.get("public", False))
    if kind == "tap":
        return (kind, scope, endpoint["bus"], endpoint["id"])
    if kind == "port":
        return (kind, endpoint["id"])
    if kind == "ground":
        return (kind,)
    return (kind, scope, endpoint["id"])


@dataclass(frozen=True, slots=True)
class Block:
    key: Key
    kind: str
    scope: tuple[str, ...]
    addressable: bool
    default_axis: str
    order_members: tuple[Key, ...] | None = None


@dataclass(frozen=True, slots=True)
class Attachment:
    key: Key
    scope: tuple[str, ...]
    net: str
    kind: str
    endpoint: Mapping[str, object]
    boundary: str | None
    block: Key | None
    preferred_side: str | None


@dataclass(frozen=True, slots=True)
class WiringGroup:
    key: GroupKey
    scope: tuple[str, ...]
    net: str
    attachments: tuple[Attachment, ...]
    aliases: tuple[Key, ...]
    addressable: bool


@dataclass(frozen=True, slots=True)
class GroundTarget:
    key: Key
    scope: tuple[str, ...]
    endpoint: Mapping[str, object]
    boundary: str | None
    block: Key | None
    addressable: bool


@dataclass(frozen=True, slots=True)
class Junction:
    id: str
    kind: str
    sides: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Connection:
    a: Key
    b: Key


@dataclass(frozen=True, slots=True)
class WiringRecipe:
    group: WiringGroup
    junctions: tuple[Junction, ...]
    connections: tuple[Connection, ...]
    automatic: bool


@dataclass(frozen=True, slots=True)
class CompositionInventory:
    snapshot: AuthoringSnapshot
    plan_sha256: str
    blocks: Mapping[Key, Block]
    groups: Mapping[GroupKey, WiringGroup]
    local_groups: Mapping[GroupKey, WiringGroup]
    endpoint_aliases: Mapping[Key, tuple[tuple[GroupKey, Key], ...]]
    group_aliases: Mapping[Key, GroupKey]
    ground_targets: Mapping[Key, GroundTarget]
    terminal_keys: frozenset[Key]
    port_keys: frozenset[Key]
    taps: Mapping[Key, tuple[Key, ...]]
    scope_boundaries: Mapping[Key, Attachment]


@dataclass(frozen=True, slots=True)
class CapturedComposition:
    inventory: CompositionInventory
    axes: Mapping[Key, str]
    order: Mapping[Key, tuple[Key, ...]]
    terminal_sides: Mapping[Key, str]
    port_sides: Mapping[Key, str]
    port_load_sides: Mapping[Key, str]
    tap_order: Mapping[Key, tuple[Key, ...]]
    ground_sides: Mapping[Key, str]
    wiring: Mapping[GroupKey, WiringRecipe]


@dataclass(frozen=True, slots=True)
class CompositionIntent:
    """Captured explicit declarations awaiting one fixed default completion.

    Mapping membership records an explicit constraint. Omitted declarations
    receive stable defaults, never geometry-ranked alternatives.
    """

    inventory: CompositionInventory
    axes: Mapping[Key, str]
    branch_axes: Mapping[Key, str]
    order: Mapping[Key, tuple[Key, ...]]
    terminal_sides: Mapping[Key, str]
    port_sides: Mapping[Key, str]
    port_load_sides: Mapping[Key, str]
    tap_order: Mapping[Key, tuple[Key, ...]]
    ground_sides: Mapping[Key, str]
    fixed_wiring: Mapping[GroupKey, WiringRecipe]

    def materialize(self) -> CapturedComposition:
        """Complete fixed defaults once through the public operations."""
        from ..composition import materialize_composition

        return materialize_composition(self)


def inventory(snapshot: AuthoringSnapshot) -> CompositionInventory:
    """Enumerate actual local incidences across every authored relation.

    A boundary pin appears internally and at actual outward incidences. A Bus is only
    a group alias. An authored Tap names existing incident contacts and never
    increases junction degree. Parallel contributes one shared boundary at
    each end, not one externally rewritable contact per branch.
    """
    semantic = snapshot.semantic_record
    connectivity = semantic["connectivity"]
    endpoint_nets = {
        endpoint_key(row["endpoint"]): row["final_net"]
        for row in connectivity["endpoint_nets"]
    }
    endpoint_nets[("ground",)] = "ground"

    def net_for(ep):
        key = endpoint_key(ep)
        if key not in endpoint_nets:
            raise SCNSimValidationError(
                "captured structure endpoint is absent from Plan connectivity",
                stage="schematic_layout",
                evidence={"endpoint": key},
            )
        return endpoint_nets[key]

    scopes: dict[tuple[str, ...], tuple[Mapping[str, object], bool]] = {}

    def visit(scope: Mapping[str, object], addressable: bool) -> None:
        path = tuple(scope["path"])
        scopes[path] = (scope, addressable)
        for child in scope["children"]:
            visit(child, addressable)
        for body in scope["component_bodies"]:
            visit(body["body"], False)

    visit(semantic["scope_hierarchy"], True)
    occurrences = {tuple(row["path"]): row for row in semantic["occurrences"]}
    # Canonical reference equivalence is too broad for presentation. Retain
    # each local return network using only explicitly authored zero-wire
    # relations, never by unioning through a raw GroundRef.
    ground_classes: dict[tuple[str, ...], dict[Key, Key]] = {}
    wire_classes: dict[tuple[str, ...], dict[Key, Key]] = {}
    outward_incidence: dict[tuple[str, ...], set[Key]] = {}
    for path, (scope, _) in scopes.items():
        parents: dict[Key, Key] = {}

        def find(key, parents=parents):
            parents.setdefault(key, key)
            while parents[key] != key:
                key = parents[key]
            return key

        def join(keys, parents=parents, find=find):
            if not keys:
                return
            first = find(keys[0])
            for key in keys[1:]:
                parents[find(key)] = first

        for bus in scope["buses"]:
            join([("bus", path, bus["id"]), *(("tap", path, bus["id"], tap["id"]) for tap in bus["taps"])])
        for exposure in scope["exposures"]["pins"]:
            join([("pin", path, exposure["id"], exposure["id"], True), endpoint_key(exposure["intrinsic_endpoint"])])
        for structure in scope["structures"]:
            if structure["kind"] == "link":
                join([endpoint_key(ep) for ep in structure["endpoints"]])
        def classes(parents=parents, find=find):
            members = defaultdict(list)
            for key in parents:
                members[find(key)].append(key)
            return {
                key: tuple(sorted(members[find(key)], key=repr)) for key in parents
            }

        # Keep the existing local-ground authority separate from signal wires.
        ground_classes[path] = classes()
        incident = {
            endpoint_key(ep)
            for row in scope["structures"] if row["kind"] == "link"
            for ep in row["endpoints"]
        }

        def member_pin(member, field):
            member_path = tuple(member["path"])
            name = member[field]
            if member_path in occurrences:
                return ("pin", member_path[:-1], member_path[-1], name, False)
            return ("pin", member_path, name, name, True)

        def wire(keys, incident=incident, join=join):
            # A raw reference is not a page-wide conductive wire class.
            keys = [key for key in keys if key != ("ground",)]
            incident.update(keys)
            join(keys)

        for structure in scope["structures"]:
            if structure["kind"] not in {"series", "parallel", "branch"}:
                continue
            start = structure["at" if structure["kind"] == "branch" else "start"]
            for branch in structure.get("branches", (structure,)):
                members = branch["elements"]
                wire([endpoint_key(start), member_pin(members[0], "pin_1")])
                for first, second in pairwise(members):
                    wire([member_pin(first, "pin_2"), member_pin(second, "pin_1")])
                wire([member_pin(members[-1], "pin_2"), endpoint_key(structure["end"])])
        if not path:
            for port in connectivity["ports"]:
                buses = [bus for bus in scope["buses"] if bus["final_net"] == port["net"]]
                if len(buses) != 1:
                    raise SCNSimValidationError(
                        "Port requires one unique root Bus local wiring class",
                        stage="schematic_layout",
                        evidence={"port": port["id"], "buses": tuple(bus["id"] for bus in buses)},
                    )
                wire([("port", port["id"]), ("bus", (), buses[0]["id"])])
        wire_classes[path] = classes()
        outward_incidence[path] = incident

    def group_key(path, ep):
        net = net_for(ep)
        if net != "ground":
            members = wire_classes[path].get(endpoint_key(ep), (endpoint_key(ep),))
            return (path, "wire:" + sha256_hex(members))
        members = ground_classes[path].get(endpoint_key(ep), (endpoint_key(ep),))
        return (path, "ground:" + sha256_hex(members))

    blocks: dict[Key, Block] = {}
    contacts: dict[GroupKey, dict[Key, Attachment]] = defaultdict(dict)
    aliases: dict[Key, list[tuple[GroupKey, Key]]] = defaultdict(list)
    group_aliases: dict[Key, GroupKey] = {}
    ground_targets: dict[Key, GroundTarget] = {}
    terminals: set[Key] = set()
    port_keys: set[Key] = set()
    taps: dict[Key, tuple[Key, ...]] = {}

    def add(path, key, ep, *, kind, block=None, boundary=None, side=None, alias=()):
        net = net_for(ep)
        if ep["kind"] == "ground":
            return
        group = group_key(path, ep)
        contacts[group][key] = Attachment(key, path, net, kind, freeze(ep), boundary, block, side)
        for name in alias:
            entry = (group, key)
            if entry not in aliases[name]:
                aliases[name].append(entry)

    def pin(path, ep, *, block, kind="pin", side=None):
        key = ("contact", path, endpoint_key(ep))
        add(path, key, ep, kind=kind, block=block, side=side, alias=(endpoint_key(ep),))
        terminals.add(endpoint_key(ep))
        return key

    for path, (scope, addressable) in scopes.items():
        used = {
            tuple(member["path"])
            for structure in scope["structures"]
            for branch in structure.get("branches", (structure,))
            for member in branch.get("elements", ())
        }
        peers = {
            ("component", path, occurrence[-1])
            for occurrence, row in occurrences.items()
            if occurrence[:-1] == path and row["body_kind"] != "ordinary" and occurrence not in used
        } | {("scope", tuple(child["path"])) for child in scope["children"]}
        ordered_peers = []
        for declaration in scope["declaration_order"]:
            for peer in sorted(peers, key=repr):
                expected_kind = "subsystem" if peer[0] == "scope" else "component"
                if declaration["kind"] == expected_kind and (peer[1][-1] if peer[0] == "scope" else peer[2]) == declaration["id"] and peer not in ordered_peers:
                    ordered_peers.append(peer)
        ordered_peers.extend(sorted(peers - set(ordered_peers), key=repr))
        blocks[("scope", path)] = Block(("scope", path), "scope", path, addressable, "horizontal", tuple(ordered_peers))
        for peer in peers:
            if peer[0] == "component":
                blocks[peer] = Block(peer, "component", path, addressable, "horizontal")
        for structure in scope["structures"]:
            kind = structure["kind"]
            key = (kind, path, structure["id"])
            if kind == "link":
                group_aliases[key] = group_key(path, structure["endpoints"][0])
                continue
            if kind not in {"series", "parallel", "branch"}:
                continue
            grounded = structure["end"]["kind"] == "ground"
            axis = "vertical" if kind == "parallel" or (kind == "branch" and grounded) else "horizontal"
            branches = structure.get("branches", (structure,))
            order_members = tuple(("series", path, row["id"]) for row in branches) if kind == "parallel" else None
            blocks[key] = Block(key, kind, path, addressable, axis, order_members)
            for boundary in (("at", "end") if kind == "branch" else ("start", "end")):
                ep = structure[boundary]
                contact_key = ("boundary", key, boundary)
                if ep["kind"] == "ground":
                    # Only an authored GroundRef creates this intrinsic glyph.
                    # Parent-grounded ordinary returns remain local wire groups.
                    ground_targets[key] = GroundTarget(key, path, freeze(ep), boundary, key, addressable)
                    continue
                side = ("left" if boundary != "end" else "right") if axis == "horizontal" else ("top" if boundary != "end" else "bottom")
                add(path, contact_key, ep, kind="structure", block=key, boundary=boundary, side=side, alias=((*key, boundary),))
                if ep["kind"] == "tap":
                    aliases[endpoint_key(ep)].append((group_key(path, ep), contact_key))
        for bus in scope["buses"]:
            key = ("bus", path, bus["id"])
            group_aliases[key] = group_key(path, {"kind": "bus", "scope": path, "id": bus["id"]})
            taps[key] = tuple(("tap", path, bus["id"], tap["id"]) for tap in bus["taps"])
            terminals.update(taps[key])
        # The inner side of an exposed public boundary belongs to this scope.
        for exposure in scope["exposures"]["pins"]:
            ep = {"kind": "pin", "scope": path, "component": exposure["id"], "id": exposure["id"], "public": True}
            pin(path, ep, block=("scope", path), kind="scope_pin")
        # The outer side is one complete child Block contact, irrespective of
        # how many Links or structure endpoints mention that same PinRef.
        for child in scope["children"]:
            child_path = tuple(child["path"])
            for exposure in child["exposures"]["pins"]:
                ep = {"kind": "pin", "scope": child_path, "component": exposure["id"], "id": exposure["id"], "public": True}
                terminals.add(endpoint_key(ep))
                if endpoint_key(ep) in outward_incidence[path]:
                    pin(path, ep, block=("scope", child_path))
        for occurrence, row in occurrences.items():
            if occurrence[:-1] != path or row["body_kind"] == "ordinary":
                continue
            names = tuple(row["public_pins"])
            if not names:
                names = tuple(ep["endpoint"]["id"] for ep in connectivity["endpoint_nets"] if ep["endpoint"].get("kind") == "pin" and tuple(ep["endpoint"].get("scope", ())) == path and ep["endpoint"].get("component") == occurrence[-1])
            selected = {
                member[pin_name]
                for structure in scope["structures"]
                for branch in structure.get("branches", (structure,))
                for member in branch.get("elements", ())
                if tuple(member["path"]) == occurrence
                for pin_name in ("pin_1", "pin_2")
            }
            for index, name in enumerate(names):
                ep = {"kind": "pin", "scope": path, "component": occurrence[-1], "id": name, "public": False}
                terminals.add(endpoint_key(ep))
                if name in selected:
                    # Selected complete-body pins alias that structure's true
                    # outer boundary; private series contacts stay private.
                    for structure in scope["structures"]:
                        if structure["kind"] not in {"series", "parallel", "branch"}:
                            continue
                        key = (structure["kind"], path, structure["id"])
                        for branch in structure.get("branches", (structure,)):
                            members = branch["elements"]
                            for member, field, boundary in ((members[0], "pin_1", "at" if structure["kind"] == "branch" else "start"), (members[-1], "pin_2", "end")):
                                if tuple(member["path"]) == occurrence and member[field] == name:
                                    boundary_ep = structure[boundary]
                                    aliases[endpoint_key(ep)].append((group_key(path, boundary_ep), ("boundary", key, boundary)))
                    terminals.add(endpoint_key(ep))
                    continue
                side = "left" if index < len(names) / 2 else "right"
                if endpoint_key(ep) in outward_incidence[path]:
                    pin(path, ep, block=("component", path, occurrence[-1]), side=side)

    for port in connectivity["ports"]:
        ep = {"kind": "port", "id": port["id"]}
        key = ("port", port["id"])
        port_keys.add(key)
        add((), key, ep, kind="port", side="left", alias=(key,))
    # Tap aliases reached through an exposure or Link identify the already
    # inventoried Block contact. They never invent an otherwise empty arm.
    for path, (scope, _) in scopes.items():
        for exposure in scope["exposures"]["pins"]:
            ep = exposure["intrinsic_endpoint"]
            if ep["kind"] == "tap":
                public = ("pin", path, exposure["id"], exposure["id"], True)
                aliases[endpoint_key(ep)].extend(entry for entry in aliases[public] if entry[0][0] == path)
        for structure in scope["structures"]:
            if structure["kind"] != "link":
                continue
            pins = [endpoint_key(ep) for ep in structure["endpoints"] if ep["kind"] == "pin"]
            for ep in structure["endpoints"]:
                if ep["kind"] == "tap":
                    aliases[endpoint_key(ep)].extend(entry for key in pins for entry in aliases[key] if entry[0][0] == path)
    for group in snapshot.source_provenance.get("ground_pins_call_groups", ()):
        for ep in group:
            key = endpoint_key(ep)
            path = tuple(ep.get("scope", ()))
            owner = path[:-1] if ep.get("public") else path
            ground_targets[key] = GroundTarget(key, owner, freeze(ep), None, None, scopes[owner][1])
    local_groups = {
        key: WiringGroup(key, key[0], next(iter(values.values())).net, tuple(values.values()), tuple(alias for alias, group in group_aliases.items() if group == key), scopes[key[0]][1] and not key[1].startswith("ground:"))
        for key, values in contacts.items()
    }
    groups = {
        key: group
        for key, group in local_groups.items()
        if len(group.attachments) >= 2
    }
    normalized_aliases = {
        alias: tuple(dict.fromkeys(entry for entry in entries if entry[1] in contacts[entry[0]]))
        for alias, entries in aliases.items()
    }
    return CompositionInventory(
        snapshot,
        sha256_hex(canonical_plan_snapshot(snapshot)),
        MappingProxyType(blocks),
        MappingProxyType(groups),
        MappingProxyType(local_groups),
        MappingProxyType(normalized_aliases),
        MappingProxyType(group_aliases),
        MappingProxyType(ground_targets),
        frozenset(terminals),
        frozenset(port_keys),
        MappingProxyType(taps),
        MappingProxyType({
            endpoint_key(contact.endpoint): contact
            for group in contacts.values() for contact in group.values()
            if contact.kind == "scope_pin"
        }),
    )
