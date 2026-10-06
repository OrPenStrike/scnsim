"""Normalize live authoring state into the immutable physical handoff.

Capture completes assembly and establishes the root-owned final net map before
record construction. Canonical hashing and numerical lowering consume it elsewhere."""

from __future__ import annotations

import json
from types import MappingProxyType
import numpy as np

from ..units import Quantity, registry
from .handles import _physical_inductive_branch
from .physical_values import RLGC, quantity_record
from .snapshot import AuthoringSnapshot, freeze


def _source_unit_identity(
    *,
    scope: str,
    component_path: tuple[str, ...] = (),
    parameter_id: str,
    field: str,
) -> str:
    return json.dumps(
        {
            "scope": scope,
            "component_path": list(component_path),
            "parameter_id": parameter_id,
            "field": field,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )



def _source_unit_record(
    identity: str, value: Quantity, si_unit: str
) -> dict[str, object]:
    magnitude = np.asarray(value.magnitude)
    probe = (
        value
        if magnitude.ndim == 0
        else registry.Quantity(float(magnitude.flat[0]), value.units)
    )
    envelope = quantity_record(probe, si_unit)
    return {
        "identity": identity,
        "source_unit": str(value.units),
        "canonical_si_unit": envelope["si_unit"],
        "canonical_dimensionality": envelope["dimensionality"],
    }



def _net_identity_map(plan):
    """Name final equivalence classes from their complete typed endpoint sets."""
    groups = {}

    def key(endpoint):
        return tuple(
            sorted(
                (name, tuple(value) if isinstance(value, list) else value)
                for name, value in endpoint.items()
            )
        )

    for scope, _ in plan._walk():
        for bus in scope.buses.values():
            groups.setdefault(bus.net.root(), []).append(key(scope.ep(bus)))
            for tap in bus.taps.values():
                groups.setdefault(tap.net.root(), []).append(key(scope.ep(tap)))
        for pin in scope.exposed_pins.values():
            groups.setdefault(pin.net.root(), []).append(key(scope.ep(pin)))
        for component in scope.components.values():
            for pin in component.pins.values():
                groups.setdefault(pin.net.root(), []).append(key(scope.ep(pin)))
    for port in plan.ports:
        groups.setdefault(port.net.root(), []).append(
            (("kind", "port"), ("id", port.id))
        )
    ordered = sorted(
        (tuple(sorted(keys)), root)
        for root, keys in groups.items()
        if not root.ground
    )
    result = {plan.ground_net.root(): "ground"}
    result.update(
        {root: f"net-{index:04d}" for index, (_, root) in enumerate(ordered, 1)}
    )
    return result



def capture_authoring_snapshot(plan) -> AuthoringSnapshot:
    plan.complete()
    plan._capture_net_ids = _net_identity_map(plan)
    occ = []
    leaves = []
    defs = {}
    edges = []
    eps = []
    endpoint_nets = []
    captured_fields = {}
    captured_refs = {}
    source_units = []
    source_identities = set()

    def source(identity, value, unit):
        if identity in source_identities:
            return
        source_identities.add(identity)
        source_units.append(_source_unit_record(identity, value, unit))

    couplings = []
    for s, scope_path in plan._walk():
        final = plan._final_net_id
        for bus in s.buses.values():
            endpoint_nets.append(
                {"endpoint": s.ep(bus), "final_net": final(bus.net)}
            )
            endpoint_nets.extend(
                {"endpoint": s.ep(tap), "final_net": final(tap.net)}
                for tap in bus.taps.values()
            )
        endpoint_nets.extend(
            {"endpoint": s.ep(pin), "final_net": final(pin.net)}
            for pin in s.exposed_pins.values()
        )
        for c in s.components.values():
            path = (*scope_path, c.id)
            net = lambda pin: plan._final_net_id(pin.net)
            boundary_pins = (
                {
                    name: {
                        "backing_endpoint": {
                            "kind": "pin",
                            "scope": list(path),
                            "component": c.id,
                            "id": name,
                        },
                        "final_net": net(pin),
                    }
                    for name, pin in c.pins.items()
                }
                if c.body
                else {}
            )
            boundary_coordinates = {
                name: {
                    "backing_endpoint": {
                        **coordinate.intrinsic_endpoint,
                        "scope": list(path),
                    },
                    "final_net": net(coordinate),
                }
                for name, coordinate in c.coordinates.items()
            }
            boundary_branches = {
                name: {
                    "oriented_physical_branch": {
                        "path": [
                            *branch.component.owner.path(), branch.component.id
                        ],
                        "id": branch.id,
                    }
                }
                for name, exposed in (
                    c.body.exposed_branches.items() if c.body else ()
                )
                for branch in (_physical_inductive_branch(exposed),)
            }
            occ.append(
                {
                    "path": list(path),
                    "factory": c.factory,
                    "catalog_id": c.catalog_id,
                    "catalog_source": dict(c.catalog_source),
                    "body_kind": c.kind,
                    "public_pins": boundary_pins,
                    "public_coordinates": boundary_coordinates,
                    "public_inductive_branches": boundary_branches,
                    "public_parameters": [
                        {"id": name, "parameter": parameter._key_record()}
                        for name, parameter in c.parameters.items()
                    ],
                }
            )
            if not c.body:
                fs = []
                for k, (base, b, r, u, positive, nonnegative) in c.fields.items():
                    captured_binding = dict(b)
                    if b["kind"] == "affine":
                        captured_binding["_affine_data"] = b["_affine_data"]
                    captured_fields[(path, k)] = (
                        freeze(base),
                        MappingProxyType(captured_binding),
                        r,
                        u,
                        positive,
                        nonnegative,
                    )
                    if r:
                        captured_refs[r._key()] = r
                    if isinstance(base, RLGC):
                        for (
                            rlgc_field,
                            rlgc_value,
                        ) in base._source_quantities.items():
                            source(
                                _source_unit_identity(
                                    scope="plan_rlgc",
                                    component_path=path,
                                    parameter_id="rlgc",
                                    field=rlgc_field,
                                ),
                                rlgc_value,
                                {
                                    "resistance_per_length": "ohm / meter",
                                    "inductance_per_length": "henry / meter",
                                    "conductance_per_length": "siemens / meter",
                                    "capacitance_per_length": "farad / meter",
                                    "extraction_frequency": "hertz",
                                }[rlgc_field],
                            )
                    elif b["kind"] != "affine":
                        source_value = base
                        if r is not None and r._source_unit is not None:
                            source_value = registry.Quantity(
                                base.to(r.spec.si_unit).magnitude, r._source_unit
                            )
                        source(
                            _source_unit_identity(
                                scope="plan_parameter",
                                component_path=path,
                                parameter_id=k,
                                field="baseline",
                            ),
                            source_value,
                            u,
                        )
                    if b["kind"] == "affine":
                        affine_source = b["_affine_source"]
                        input_baseline = r.baseline
                        if r._source_unit is not None:
                            input_baseline = registry.Quantity(
                                r.baseline.to(r.spec.si_unit).magnitude,
                                r._source_unit,
                            )
                        source(
                            _source_unit_identity(
                                scope="plan_affine",
                                component_path=path,
                                parameter_id=k,
                                field="input_baseline",
                            ),
                            input_baseline,
                            r.spec.si_unit,
                        )
                        source(
                            _source_unit_identity(
                                scope="plan_affine",
                                component_path=path,
                                parameter_id=k,
                                field="mapped_output_baseline",
                            ),
                            base.to(affine_source["intercept"].units),
                            u,
                        )
                        for affine_field, value, unit in (
                            (
                                "slope",
                                affine_source["slope"],
                                b["slope"]["si_unit"],
                            ),
                            (
                                "intercept",
                                affine_source["intercept"],
                                b["intercept"]["si_unit"],
                            ),
                            (
                                "support_lower",
                                affine_source["support_lower"],
                                r.spec.si_unit,
                            ),
                            (
                                "support_upper",
                                affine_source["support_upper"],
                                r.spec.si_unit,
                            ),
                        ):
                            source(
                                _source_unit_identity(
                                    scope="plan_affine",
                                    component_path=path,
                                    parameter_id=k,
                                    field=affine_field,
                                ),
                                value,
                                unit,
                            )
                    public_binding = {
                        x: y for x, y in b.items() if not x.startswith("_")
                    }
                    fs.append({"id": k, "unit": u, "binding": public_binding})
                    if r:
                        defs[r._key()] = r._definition_record()
                        edges.append(
                            {
                                "path": list(path),
                                "field": k,
                                "parameter": r._key_record(),
                            }
                        )
                leaves.append(
                    {
                        "path": list(path),
                        "model": c.factory,
                        "pin_order": list(c.pins),
                        "fields": fs,
                        "model_metadata": dict(c.metadata),
                        "oriented_branches": [
                            {
                                "id": k,
                                "positive_pin": v[0],
                                "negative_pin": v[1],
                                "value_field": v[2],
                            }
                            for k, v in c.branches.items()
                        ],
                    }
                )
            if not c.body:
                for p in c.pins.values():
                    eps.append(
                        {
                            "path": list(path),
                            "pin": p.name,
                            "net": plan._final_net_id(p.net),
                        }
                    )
                for p in c.pins.values():
                    endpoint_nets.append(
                        {"endpoint": s.ep(p), "final_net": final(p.net)}
                    )
            else:
                # A registered Composite has two distinct public-boundary
                # views: its containing-scope outer pins and the rebased
                # exposed pins in its immutable body overlay.  Both are
                # typed aliases of one final net and must be captured so
                # Link/Port consumers never reconstruct topology from a
                # live ComponentInstance.
                endpoint_nets.extend(
                    {"endpoint": s.ep(pin), "final_net": final(pin.net)}
                    for pin in c.pins.values()
                )
        couplings.extend(
            x
            for x in s.record(path=scope_path)["structures"]
            if x["kind"] == "coupling"
        )
    for port in plan.ports:
        endpoint_nets.append(
            {
                "endpoint": {"kind": "port", "id": port.id},
                "final_net": final(port.net),
            }
        )
    aliases = {}
    for scope, scope_path in plan._walk():
        if not scope_path:
            for bus in scope.buses.values():
                aliases.setdefault(final(bus.net), []).append(
                    {"kind": "root_bus", "id": bus.id}
                )
        for name, coordinate in scope.exposed_coordinates.items():
            aliases.setdefault(final(coordinate.net), []).append(
                {
                    "kind": "exposed_coordinate",
                    "scope": list(scope_path),
                    "id": name,
                }
            )
    for port in plan.ports:
        aliases.setdefault(final(port.net), []).append(
            {"kind": "port", "id": port.id}
        )
    nodes = [
        {
            "final_net": net,
            "compiler_node_id": net,
            "visibility": "public" if aliases.get(net) else "internal",
            "public_aliases": aliases.get(net, []),
        }
        for net in sorted(
            {
                row["final_net"]
                for row in endpoint_nets
                if row["final_net"] != "ground"
            }
        )
    ]
    semantic = {
        "schema": "scnsim.authoring_snapshot",
        "schema_version": 2,
        "plan_id": plan.id,
        "scope_hierarchy": plan.record(),
        "occurrences": occ,
        "physical_leaves": leaves,
        "connectivity": {
            "physical_endpoints": eps,
            "endpoint_nets": endpoint_nets,
            "node_coordinates": nodes,
            "canonical_ground": "ground",
            "ports": [
                {
                    "id": p.id,
                    "net": plan._final_net_id(p.net),
                    "role": p.role,
                    "reference_impedance": quantity_record(
                        p.reference_impedance, "ohm"
                    ),
                }
                for p in plan.ports
            ],
            "couplings": couplings,
        },
        "parameter_closure": {
            "definitions": list(defs.values()),
            "field_bindings": edges,
        },
    }
    for port in plan.ports:
        source(
            _source_unit_identity(
                scope="plan_port", parameter_id=port.id, field="reference_impedance"
            ),
            port.reference_impedance,
            "ohm",
        )
    source_units.sort(key=lambda row: row["identity"])
    prov = {
        "source_units": source_units,
        "ground_pins_call_groups": [
            [s.ep(p) for p in g] for s, _ in plan._walk() for g in s.ground_calls
        ],
    }
    return AuthoringSnapshot.create(
        semantic_record=semantic,
        source_provenance=prov,
        resolution_refs=captured_refs,
        resolution_fields=captured_fields,
    )
