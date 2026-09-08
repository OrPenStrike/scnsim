"""Reference-independent authoring evidence for symmetric two-terminal bodies.

This operates on either independently built manifest, never on a paired answer.
Terminal names in its output are canonical references, not observed authored
pin identities. Oriented source records remain in verified-source provenance.
Signed mutual coefficients transform with both branch references; floating
unlabelled twin nets admit a further reference reversal, fixed by a signed
spanning forest without discarding observable cycle or anchored coupling signs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal
from typing import cast

from .._canonical import canonical_json_bytes, float64_from_hex, float64_hex


_SYMMETRIC = {"resistor", "capacitor", "inductor", "josephson_junction"}


def coupling_sign(value: Mapping[str, object]) -> dict[str, object]:
    """Represent only the visible sign, without recovering a hidden magnitude."""
    result = dict(value)
    if "si_decimal" in value:
        number = Decimal(str(value["si_decimal"]))
        result["si_decimal"] = "1" if number > 0 else "-1" if number < 0 else "0"
        return result
    number = float64_from_hex(cast(str, value["si_value_f64"]))
    result["si_value_f64"] = float64_hex(1.0 if number > 0 else -1.0 if number < 0 else 0.0)
    return result


def normalize_authoring(
    electrical_manifest: Mapping[str, object],
    structural_manifest: Mapping[str, object],
    value_manifest: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Canonicalize one side's topology, reference convention and mutual signs."""
    electrical, structural, values = json.loads(
        canonical_json_bytes([electrical_manifest, structural_manifest, value_manifest])
    )
    bodies = {tuple(body["path"]): body for body in electrical["bodies"]}
    symmetric = {path for path, body in bodies.items() if body["model"] in _SYMMETRIC or body.get("line_kind") == "CPW"}

    # Distinguish nets using observable incidences, never ordinary pin numbers.
    # Boundary names and line terminal names remain observable and ordered.
    signatures = defaultdict(list)
    for row in electrical["nets"]:
        for text in row["contacts"]:
            contact = json.loads(text)
            if tuple(contact.get("component_path", ())) in symmetric:
                contact["pin_id"] = "symmetric_terminal"
            signatures[row["net"]].append(canonical_json_bytes(contact).decode())
    for row in structural["boundary_incidence"]:
        signatures[row["net"]].append(canonical_json_bytes({
            "region": row["region"], "boundary": row["boundary"],
        }).decode())
    groups = defaultdict(list)
    for net, contacts in signatures.items():
        if net != "ground":
            groups[canonical_json_bytes(sorted(contacts))].append(net)
    net_map = {"ground": "ground"}
    free_group = {}
    ordinal = 0
    for _, nets in sorted(groups.items()):
        # Distinct equal signatures can only be the two ends of the same
        # unlabelled parallel set: each native body has exactly two terminals.
        for net in sorted(nets):
            net_map[net] = f"netv-{ordinal}"
            if len(nets) == 2:
                free_group[net] = min(nets)
            ordinal += 1

    pin_maps = {}
    branch_signs = {}
    branch_groups = {}
    for path, body in bodies.items():
        terminals = body["terminals"]
        if path in symmetric:
            body["terminal_reference_basis"] = (
                "Canonical references derived from visible net incidence, "
                "not authored endpoint order"
            )
            body["observed_authored_terminal_order"] = False
            ordered = sorted(terminals, key=lambda row: net_map[row["net"]])
            pin_map = {row["pin_id"]: f"terminal_{index}" for index, row in enumerate(ordered, 1)}
            pin_maps[path] = pin_map
            for branch in body["oriented_branches"]:
                key = (path, branch["id"])
                branch_signs[key] = 1 if pin_map[branch["positive_pin"]] == "terminal_1" else -1
                branch_groups[key] = free_group.get(terminals[0]["net"])
                branch["positive_pin"], branch["negative_pin"] = "terminal_1", "terminal_2"
            body["pin_order"] = ["terminal_1", "terminal_2"]
            body["terminals"] = [
                {"pin_id": pin_map[row["pin_id"]], "net": net_map[row["net"]]}
                for row in ordered
            ]
        else:
            for row in terminals:
                row["net"] = net_map[row["net"]]

    def contact_reference(text: str) -> str:
        contact = json.loads(text)
        path = tuple(contact.get("component_path", ()))
        if path in pin_maps:
            contact["pin_id"] = pin_maps[path][contact["pin_id"]]
        return canonical_json_bytes(contact).decode()

    for row in electrical["nets"]:
        row["net"] = net_map[row["net"]]
        row["contacts"] = sorted(contact_reference(text) for text in row["contacts"])
    electrical["nets"].sort(key=canonical_json_bytes)
    for row in electrical["ports"]:
        row["node_net"] = net_map[row["node_net"]]
        row["reference_net"] = net_map[row["reference_net"]]
    for row in structural["boundary_incidence"]:
        row["net"] = net_map[row["net"]]
        for field in ("inside_contacts", "outside_contacts"):
            row[field] = sorted(contact_reference(text) for text in row[field])
    structural["boundary_incidence"].sort(key=canonical_json_bytes)

    couplings = []
    for row in electrical["couplings"]:
        identity = canonical_json_bytes({
            "kind": "coupling_coefficient", "coupling_id": row["coupling_id"],
        }).decode()
        keys = [(tuple(branch["path"]), branch["branch_id"]) for branch in row["branches"]]
        left, right = keys
        coefficient = Decimal(values[identity]["si_decimal"])
        if branch_signs[left] * branch_signs[right] < 0:
            coefficient = coefficient.copy_negate()
        couplings.append((identity, branch_groups[left], branch_groups[right], coefficient))

    # Fix only free reference reversals. Edges to None are anchored by visible
    # ground/Port/boundary/terminal incidence; cycles retain their relative sign.
    parents: dict[str | None, str | None] = {}
    forest = defaultdict(list)

    def root(node: str | None) -> str | None:
        parents.setdefault(node, node)
        while parents[node] != node:
            node = parents[node]
        return node

    for _, left, right, coefficient in sorted(couplings):
        if coefficient == 0 or root(left) == root(right):
            continue
        parents[root(left)] = root(right)
        sign = 1 if coefficient > 0 else -1
        forest[left].append((right, sign))
        forest[right].append((left, sign))
    gauges = {}
    for start in [None, *sorted(node for node in forest if node is not None)]:
        if start in gauges:
            continue
        gauges[start] = 1
        pending = [start]
        while pending:
            node = pending.pop()
            for peer, sign in forest[node]:
                if peer not in gauges:
                    gauges[peer] = gauges[node] * sign
                    pending.append(peer)
    for identity, left, right, coefficient in couplings:
        number = coefficient.copy_negate() if gauges.get(left, 1) * gauges.get(right, 1) < 0 else coefficient
        values[identity]["si_decimal"] = str(number) if number else "0"
    return electrical, structural, values
