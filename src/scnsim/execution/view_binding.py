"""Immutable root View declaration; public refs are weakly cached by their Run."""
from __future__ import annotations

from dataclasses import dataclass
from json import loads

from collections.abc import Mapping, Sequence
from ..authoring import ElectricNodeRef, CoordinateRef
from ..errors import CompilerInvariantError
from ..canonical import canonical_json_bytes, sha256_hex


@dataclass(frozen=True, slots=True)
class ViewDeclaration:
    lineage_bytes: bytes
    available_coordinates: tuple[str, ...]
    port_coordinates: tuple[tuple[str, str], ...]
    coordinate_load_states: tuple[tuple[str, str], ...]
    retained: tuple[str, ...] = ()

    @classmethod
    def create(cls, *, lineage, available_coordinates, port_coordinates,
               coordinate_load_states, retained=()):
        return cls(canonical_json_bytes(lineage), tuple(available_coordinates),
                   tuple(port_coordinates.items()), tuple(coordinate_load_states.items()), tuple(retained))

    def lineage(self):
        return loads(self.lineage_bytes)


def _coordinate_id(value: str | ElectricNodeRef | CoordinateRef) -> str:
    if isinstance(value, str):
        if not value:
            raise ValueError("coordinate IDs must not be empty")
        return value
    identifier = getattr(value, "id", None)
    if isinstance(identifier, str) and identifier:
        return identifier
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    raise TypeError("coordinate must be a public SCNSim coordinate handle or ID")

def coordinate_id(plan, coordinate_lookup, value: str | ElectricNodeRef | CoordinateRef) -> str:
        """Resolve a public alias to the snapshot's canonical compiler node."""

        if isinstance(value, ElectricNodeRef) and value.plan is not plan:
            raise ValueError("coordinate belongs to another Plan")
        if isinstance(value, CoordinateRef):
            if value.scope.root is not plan:
                raise ValueError("coordinate belongs to another Plan")
            key = canonical_json_bytes({"scope": list(value.scope.path()), "id": value.id}).decode("utf-8")
            resolved = coordinate_lookup.get(key)
        else:
            resolved = coordinate_lookup.get(_coordinate_id(value))
        if resolved is None:
            raise ValueError("coordinate is not a public alias in this Plan")
        return resolved

def _original_lineage_document(
    plan: Mapping[str, object],
    plan_sha256: str,
    runtime: Mapping[str, object],
    *,
    coordinate_order: tuple[str, ...],
) -> dict[str, object]:
    connectivity = plan.get("connectivity")
    ports = connectivity.get("ports") if isinstance(connectivity, Mapping) else None
    if not isinstance(ports, Sequence) or isinstance(ports, (str, bytes)):
        raise CompilerInvariantError("Plan Port inventory is malformed", stage="plan_seal")
    port_order = [port["id"] for port in ports]
    original = {
        "type": "original",
        "compiled_graph_sha256": sha256_hex(
            {
                "schema": "scnsim.compiled_graph_identity",
                "schema_version": 1,
                "plan_sha256": plan_sha256,
                "julia_source_sha256": runtime["julia_source_sha256"],
            }
        ),
        "coordinate_order": list(coordinate_order),
        "port_order": port_order,
        "port_realizable": bool(port_order),
    }
    record: dict[str, object] = {
        "type": "network_view_lineage",
        "original": original,
        "ptc": None,
        "transforms": [],
        "retain": None,
        "terminal_coordinates": port_order,
        "port_realizable": bool(port_order),
    }
    record["lineage_sha256"] = sha256_hex(record)
    return record

def derive_view(*, plan, coordinate_lookup, parent, pipeline):
        """Apply one immutable dev5 grammar suffix without executing it.

        Candidate-dependent transform weights and B/R/M realization remain a
        preflight responsibility; this Ref records only exact declarations and
        current coordinate identities.
        """

        if pipeline._retained is not None and parent._retained:
            raise ValueError("retain() is terminal and cannot be added to a retained View")
        if pipeline._ptc is not None and (
            parent._lineage["ptc"] is not None or parent._lineage["transforms"]
        ):
            raise ValueError("ptc() must be the first reduction in a View lineage")
        available = list(parent._available_coordinates)
        port_coordinates = dict(parent._port_coordinates)
        load_states = dict(parent._coordinate_load_states)
        ptc = parent._lineage["ptc"]
        if pipeline._ptc is not None:
            port_by_id = {port.id: port for port in plan.ports}
            requested: set[str] = set()
            for port in pipeline._ptc:
                if port.plan is not plan or port.id not in port_by_id or port_by_id[port.id] is not port:
                    raise ValueError("ptc() PortRef belongs to another Plan")
                if port.role != "nonloading_probe":
                    raise ValueError("ptc() accepts only nonloading_probe Ports")
                if port.id in requested:
                    raise ValueError("ptc() Ports must be unique")
                requested.add(port.id)
            ptc = {
                "type": "ptc",
                "selected_ports": [port.id for port in plan.ports if port.id in requested],
            }
            selected_ports = set(ptc["selected_ports"])
            for coordinate, port_id in port_coordinates.items():
                load_states[coordinate] = (
                    "compensated" if port_id in selected_ports else "raw"
                )
        transforms = [dict(value) for value in parent._lineage["transforms"]]

        def resolve_coordinate(value: str | ElectricNodeRef | CoordinateRef) -> str:
            # Derived coordinate IDs are already in the current basis. Every
            # other spelling must resolve through the snapshot's typed/public
            # alias table to one opaque compiler node ID.
            if isinstance(value, str) and value in available:
                return value
            return coordinate_id(plan, coordinate_lookup, value)

        for raw_left, raw_right, identifier in pipeline._transforms:
            left, right = resolve_coordinate(raw_left), resolve_coordinate(raw_right)
            if isinstance(raw_left, ElectricNodeRef) and raw_left.plan is not plan:
                raise ValueError("transform_pair node belongs to another Plan")
            if isinstance(raw_right, ElectricNodeRef) and raw_right.plan is not plan:
                raise ValueError("transform_pair node belongs to another Plan")
            if left == right or left not in available or right not in available:
                raise ValueError("transform_pair() requires two distinct current Public coordinates")
            common, differential = f"{identifier}.common", f"{identifier}.differential"
            if (
                common in available
                or differential in available
                or common in coordinate_lookup
                or differential in coordinate_lookup
            ):
                raise ValueError("transform_pair generated coordinate collides with the current basis")
            left_state = load_states.get(left, "not-port")
            right_state = load_states.get(right, "not-port")
            if (
                left_state != right_state
                and left_state != "not-port"
                and right_state != "not-port"
            ):
                raise ValueError("transform_pair Port inputs must share one PTC load state")
            generated_port = (
                identifier
                if left_state == right_state and left_state != "not-port"
                else None
            )
            left_index, right_index = available.index(left), available.index(right)
            insert_at = min(left_index, right_index)
            available = [value for value in available if value not in {left, right}]
            available[insert_at:insert_at] = [common, differential]
            port_coordinates.pop(left, None)
            port_coordinates.pop(right, None)
            load_states.pop(left, None)
            load_states.pop(right, None)
            if generated_port is not None:
                port_coordinates[common] = common
                port_coordinates[differential] = differential
                load_states[common] = left_state
                load_states[differential] = left_state
            else:
                load_states[common] = "not-port"
                load_states[differential] = "not-port"
            transforms.append(
                {
                    "type": "transform_pair",
                    "id": identifier,
                    "input_coordinates": [left, right],
                    "output_coordinates": [common, differential],
                }
            )
        retained: tuple[str, ...] = parent._retained
        retain_record = parent._lineage["retain"]
        if pipeline._retained is not None:
            resolved = tuple(resolve_coordinate(value) for value in pipeline._retained)
            if any(isinstance(value, ElectricNodeRef) and value.plan is not plan for value in pipeline._retained):
                raise ValueError("retained node belongs to another Plan")
            if len(set(resolved)) != len(resolved) or not resolved or any(value not in available for value in resolved):
                raise ValueError("retain() accepts only unique current Public coordinates")
            retained = resolved
            # Candidate-dependent B/R/M matrices are resolved by the Julia
            # preflight from this exact declarative lineage.
            retain_record = {
                "type": "retain",
                "retained_coordinates": list(resolved),
                "eliminated_coordinates": [value for value in available if value not in resolved],
                "output_coordinate_order": list(resolved),
            }
        terminal = list(retained) if retained else [port.id for port in plan.ports]
        # A transform without retain() changes the compiled physical basis but
        # not the raw public Direct boundary: logical Plan Ports remain the
        # terminal channels in their declaration order.
        port_realizable = (
            bool(terminal)
            if not retained
            else bool(terminal) and all(value in port_coordinates for value in terminal)
        )
        record: dict[str, object] = {
            "type": "network_view_lineage",
            "original": dict(parent._lineage["original"]),
            "ptc": ptc,
            "transforms": transforms,
            "retain": retain_record,
            "terminal_coordinates": terminal,
            "port_realizable": port_realizable,
        }
        record["lineage_sha256"] = sha256_hex(record)
        return ViewDeclaration.create(
            lineage=record,
            retained=retained,
            available_coordinates=tuple(available),
            port_coordinates=port_coordinates,
            coordinate_load_states=load_states,
        )
