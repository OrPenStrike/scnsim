"""Editable schematic composition over one captured CircuitPlan.

This owner holds the public composition declarations and caller-authored
presentation choices. Its wiring replaces owner-local inter-Block ink; it does
not edit electrical declarations, private Series contacts, or Parallel rails.
Capture, replay, and fixed default completion live in ``capture``; deterministic
automatic wiring helpers live in ``automatic``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from html import escape
from types import MappingProxyType
from typing import TYPE_CHECKING

from ...authoring.snapshot import AuthoringSnapshot
from ...authoring.identity import canonical_plan_snapshot
from ...canonical import sha256_hex
from ..diagram.composition_model import (
    Attachment, CapturedComposition, CompositionIntent, Connection, GroupKey,
    Junction, Key, WiringGroup, WiringRecipe, inventory,
)
from ...authoring import (
    BranchRef, BusRef, CircuitPlan, ComponentInstance, LinkRef, ParallelRef,
    PinRef, PortRef, SeriesRef, SubsystemPlan, TapRef,
)
from ...errors import SCNSimValidationError
from ..schematic import (
    DiagramAxis, SchematicLayout, _fail, _key, _owned_scope, _scope_path, _side,
)

if TYPE_CHECKING:
    from ...results import HtmlPresentation
    from ...specs import DiagramSide

_SIDES = ("left", "right", "top", "bottom")
_OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
_REF_TOKEN = object()

@dataclass(frozen=True, slots=True)
class _SnapshotTarget:
    owner: object
    key: Key


@dataclass(frozen=True, slots=True, init=False)
class SchematicEndpoint:
    """A presentation contact, not an ElectricNode or analytical Coordinate."""

    _owner: object
    _candidates: tuple[tuple[GroupKey, Key], ...]

    def __init__(self, *, _owner: object, _candidates: tuple, _token: object = None) -> None:
        if _token is not _REF_TOKEN:
            raise TypeError("presentation endpoints are created by composition.endpoint()")
        object.__setattr__(self, "_owner", _owner)
        object.__setattr__(self, "_candidates", _candidates)


@dataclass(frozen=True, slots=True, init=False)
class SchematicArm:
    """One eligible, single-use local junction arm."""

    _wiring: SchematicWiring
    _id: str
    _side: str

    def __init__(self, *, _wiring: SchematicWiring, _id: str, _side: str, _token: object = None) -> None:
        if _token is not _REF_TOKEN:
            raise TypeError("junction arms are obtained from a wiring junction")
        object.__setattr__(self, "_wiring", _wiring)
        object.__setattr__(self, "_id", _id)
        object.__setattr__(self, "_side", _side)


@dataclass(frozen=True, slots=True, init=False)
class SchematicJunction:
    """Opaque junction ref exposing only the conductive sides of its shape."""

    _wiring: SchematicWiring
    _junction: Junction

    def __init__(self, *, _wiring: SchematicWiring, _junction: Junction, _token: object = None) -> None:
        if _token is not _REF_TOKEN:
            raise TypeError("junctions are created by wiring shape operations")
        object.__setattr__(self, "_wiring", _wiring)
        object.__setattr__(self, "_junction", _junction)

    @property
    def id(self) -> str:
        return self._junction.id

    def _arm(self, side: str) -> SchematicArm:
        self._wiring._check()
        if side not in self._junction.sides:
            raise _fail("junction shape has no requested arm", junction=self.id, side=side)
        return SchematicArm(_wiring=self._wiring, _id=self.id, _side=side, _token=_REF_TOKEN)

    @property
    def left(self) -> SchematicArm:
        return self._arm("left")

    @property
    def right(self) -> SchematicArm:
        return self._arm("right")

    @property
    def top(self) -> SchematicArm:
        return self._arm("top")

    @property
    def bottom(self) -> SchematicArm:
        return self._arm("bottom")


class SchematicWiring:
    """The current replacement authority for one owner-local wiring group."""

    def __init__(self, *, _owner: object, _group: WiringGroup, _automatic: bool, _token: object = None) -> None:
        if _token is not _REF_TOKEN:
            raise TypeError("wiring is created by composition.replace_wiring()")
        self._owner = _owner
        self._group = _group
        self._automatic = _automatic
        self._junctions: dict[str, Junction] = {}
        self._connections: list[Connection] = []
        self._used: set[Key] = set()

    def _check(self) -> None:
        self._owner._assert_current()
        if self._owner._wiring.get(self._group.key) is not self:
            raise _fail("wiring replacement ref is stale", group=self._group.key)

    @property
    def junctions(self) -> tuple[Junction, ...]:
        self._check()
        return tuple(self._junctions.values())

    @property
    def connections(self) -> tuple[Connection, ...]:
        self._check()
        return tuple(self._connections)

    def _shape(self, id: str, kind: str, sides: tuple[str, ...]) -> SchematicJunction:
        self._check()
        if not isinstance(id, str):
            raise TypeError("junction id must be str")
        if not id or any(ord(char) < 32 or char in "/\\" for char in id):
            raise _fail("junction id must be nonempty and local", id=id)
        if id in self._junctions:
            raise _fail("junction id is already used in this wiring group", id=id)
        junction = Junction(id, kind, sides)
        self._junctions[id] = junction
        return SchematicJunction(_wiring=self, _junction=junction, _token=_REF_TOKEN)

    def straight(self, *, id: str, axis: DiagramAxis) -> SchematicJunction:
        if not isinstance(axis, DiagramAxis):
            raise TypeError("straight axis must be DiagramAxis")
        return self._shape(id, "straight", ("left", "right") if axis is DiagramAxis.HORIZONTAL else ("top", "bottom"))

    def elbow(self, *, id: str, sides: tuple[DiagramSide, DiagramSide]) -> SchematicJunction:
        if not isinstance(sides, tuple):
            raise TypeError("elbow sides must be a tuple")
        values = tuple(_side(side).value for side in sides)
        if len(values) != 2 or len(set(values)) != 2 or _OPPOSITE[values[0]] == values[1]:
            raise _fail("elbow needs exactly two adjacent sides")
        return self._shape(id, "elbow", tuple(side for side in _SIDES if side in values))

    def tee(self, *, id: str, branch: DiagramSide) -> SchematicJunction:
        value = _side(branch).value
        through = ("left", "right") if value in {"top", "bottom"} else ("top", "bottom")
        return self._shape(id, "tee", tuple(side for side in _SIDES if side in (*through, value)))

    def cross(self, *, id: str) -> SchematicJunction:
        return self._shape(id, "cross", _SIDES)

    def _connection_key(self, ref: object) -> Key:
        if isinstance(ref, SchematicArm):
            ref._wiring._check()
            if ref._wiring is not self:
                raise _fail("junction arm belongs to another wiring group or replacement")
            if ref._side not in self._junctions[ref._id].sides:
                raise _fail("junction arm is not part of its shape")
            return ("arm", ref._id, ref._side)
        if not isinstance(ref, SchematicEndpoint):
            raise TypeError("connect endpoints must be composition endpoints or junction arms")
        if ref._owner is not self._owner:
            raise _fail("presentation endpoint belongs to another composition")
        candidates = tuple(dict.fromkeys(key for group, key in ref._candidates if group == self._group.key))
        if not candidates:
            raise _fail("endpoint is foreign to this owner-local wiring group", group=self._group.key)
        if len(candidates) != 1:
            raise _fail("attachment alias is ambiguous; select endpoint(structure, boundary=...) explicitly", group=self._group.key)
        return ("attachment", candidates[0])

    def connect(self, a: SchematicEndpoint | SchematicArm, b: SchematicEndpoint | SchematicArm) -> None:
        self._check()
        left, right = self._connection_key(a), self._connection_key(b)
        if left == right or left in self._used or right in self._used:
            raise _fail("each external attachment and junction arm must be connected exactly once")
        if left[0] == right[0] == "arm" and left[1] == right[1]:
            raise _fail("a wire cannot join two arms of the same local junction")
        self._connections.append(Connection(left, right))
        self._used.update((left, right))

    def _capture(self, *, complete: bool = True, _checked: bool = False) -> WiringRecipe:
        if not _checked:
            self._check()
        if complete:
            expected = {("attachment", contact.key) for contact in self._group.attachments} | {
                ("arm", junction.id, side) for junction in self._junctions.values() for side in junction.sides
            }
            if self._used != expected:
                raise _fail("wiring replacement is incomplete", group=self._group.key, missing=tuple(sorted(expected - self._used, key=repr)))
            # A junction conducts between all of its own arms. Wires then
            # connect these local conductive nodes into exactly one group.
            def node(key: Key) -> Key:
                return ("junction", key[1]) if key[0] == "arm" else key

            neighbors: dict[Key, set[Key]] = {}
            for connection in self._connections:
                left, right = node(connection.a), node(connection.b)
                neighbors.setdefault(left, set()).add(right)
                neighbors.setdefault(right, set()).add(left)
            pending = [next(iter(neighbors))] if neighbors else []
            visited = set()
            while pending:
                current = pending.pop()
                if current not in visited:
                    visited.add(current)
                    pending.extend(neighbors[current] - visited)
            if visited != {node(key) for key in expected}:
                raise _fail("wiring replacement contains disconnected conductive groups", group=self._group.key)
        return WiringRecipe(self._group, tuple(self._junctions.values()), tuple(self._connections), self._automatic)


class SchematicComposition:
    """A mutable presentation recipe bound to one complete Plan declaration.

    Every field is an optional override of one fixed default. Replacing wiring
    requires a complete owner-local group; defaults never finish a caller's
    partial replacement. Built-Composite internals retain their authoring
    projection and remain nonaddressable.
    """

    def __init__(self, *, plan: CircuitPlan) -> None:
        if not isinstance(plan, CircuitPlan):
            raise TypeError("SchematicComposition requires CircuitPlan")
        self._plan = plan
        self._inventory = inventory(plan._capture_authoring_snapshot())
        self._initialize_declarations()

    def _initialize_declarations(self) -> None:
        self._axes: dict[Key, str] = {}
        self._branch_axes: dict[Key, str] = {}
        self._order: dict[Key, tuple[Key, ...]] = {}
        self._terminal_sides: dict[Key, str] = {}
        self._port_sides: dict[Key, str] = {}
        self._port_load_sides: dict[Key, str] = {}
        self._explicit_port_sides: set[Key] = set()
        self._tap_order: dict[Key, tuple[Key, ...]] = {}
        self._ground_sides: dict[Key, str] = {}
        self._wiring: dict[GroupKey, SchematicWiring] = {}
        self._building_automatic = False
        self._planning_group: GroupKey | None = None

    @classmethod
    def automatic(cls, *, plan: CircuitPlan, hints: SchematicLayout | None = None) -> SchematicComposition:
        if hints is not None and not isinstance(hints, SchematicLayout):
            raise TypeError("automatic composition hints must be SchematicLayout or None")
        result = cls(plan=plan)
        result._apply_hints(hints)
        return result

    def _apply_hints(self, hints: SchematicLayout | None) -> None:
        if hints is not None:
            # Legacy constraints are validated with their original typed-handle
            # eligibility, then applied through the one public mutator path.
            hints._capture_for(self._plan)
            for block, axis in hints.axes.items():
                self.axis(block, axis)
            for target, members in hints.order.items():
                self.order(target, members=members)
            for terminal, side in hints.terminal_sides.items():
                self.terminal_side(terminal, side)
            for port, side in hints.port_sides.items():
                self.port_side(port, side)
            for bus, taps in hints.tap_order.items():
                self.tap_order(bus, taps=taps)

    @classmethod
    def _automatic_from_snapshot(cls, *, plan: CircuitPlan, snapshot: AuthoringSnapshot, hints: SchematicLayout | None) -> SchematicComposition:
        """Use the render boundary's one captured source, without recapturing."""
        result = cls.__new__(cls)
        result._plan, result._inventory = plan, inventory(snapshot)
        result._initialize_declarations()
        result._building_automatic = True
        try:
            result._apply_hints(hints)
        finally:
            result._building_automatic = False
        return result

    def _assert_current(self, snapshot: AuthoringSnapshot | None = None) -> None:
        if self._building_automatic:
            return
        try:
            current = self._plan._capture_authoring_snapshot() if snapshot is None else snapshot
        except SCNSimValidationError as error:
            raise _fail("composition is stale or its bound Plan is no longer complete") from error
        if sha256_hex(canonical_plan_snapshot(current)) != self._inventory.plan_sha256:
            raise _fail("composition is stale after Plan changes; construct a new composition")

    def _target(self, key: Key) -> _SnapshotTarget:
        return _SnapshotTarget(self, key)

    def _key(self, ref: object) -> Key:
        if isinstance(ref, _SnapshotTarget):
            if ref.owner is not self or not self._building_automatic:
                raise _fail("private snapshot targets cannot be supplied by callers")
            return ref.key
        if isinstance(ref, LinkRef):
            _owned_scope(ref.scope, self._plan)
            if not any(row.get("kind") == "link" and row.get("id") == ref.id for row in ref.scope.structures):
                raise _fail("wiring LinkRef is stale")
            key = ("link", _scope_path(ref.scope), ref.id)
        else:
            key = _key(ref, self._plan)
        if key[0] != "port":
            scope = self._inventory.blocks.get(("scope", key[1]))
            if scope is not None and not scope.addressable:
                raise _fail("built-Composite implementation contents are not caller-addressable")
        return key

    @staticmethod
    def _require_ref(ref: object, allowed: tuple[type, ...], *, operation: str) -> None:
        if not isinstance(ref, (*allowed, _SnapshotTarget)):
            raise TypeError(f"{operation} requires an eligible typed authoring handle")

    def axis(self, block: object, axis: DiagramAxis) -> None:
        self._assert_current()
        self._require_ref(block, (CircuitPlan, SubsystemPlan, ComponentInstance, SeriesRef, ParallelRef, BranchRef), operation="axis")
        if not isinstance(axis, DiagramAxis):
            raise TypeError("Block axis must be DiagramAxis")
        key = self._key(block)
        if key not in self._inventory.blocks:
            if self._parallel_for_branch(key) is None:
                raise _fail("axis target is not a complete Block or assembly structure")
            self._branch_axes[key] = axis.value
            return
        self._axes[key] = axis.value

    def _parallel_for_branch(self, key: Key) -> Key | None:
        return next((parent for parent, block in self._inventory.blocks.items() if block.kind == "parallel" and key in block.order_members), None)

    def order(self, scope_or_parallel: object, *, members: tuple[object, ...]) -> None:
        self._assert_current()
        self._require_ref(scope_or_parallel, (CircuitPlan, SubsystemPlan, ParallelRef), operation="order")
        if not isinstance(members, tuple):
            raise TypeError("order members must be a tuple")
        for member in members:
            self._require_ref(member, (ComponentInstance, SubsystemPlan, SeriesRef), operation="order member")
        key = self._key(scope_or_parallel)
        block = self._inventory.blocks.get(key)
        values = tuple(self._key(member) for member in members)
        if block is None or block.order_members is None:
            raise _fail("order target must be a scope or Parallel")
        if len(values) != len(set(values)) or set(values) != set(block.order_members):
            raise _fail("order must name exactly the scope peers or Parallel branches")
        self._order[key] = values

    def terminal_side(self, pin_or_tap: PinRef | TapRef, side: DiagramSide) -> None:
        self._assert_current()
        self._require_ref(pin_or_tap, (PinRef, TapRef), operation="terminal_side")
        value = _side(side).value
        key = self._key(pin_or_tap)
        if key not in self._inventory.terminal_keys:
            raise _fail("terminal side requires a public Pin or authored Tap")
        self._terminal_sides[key] = value

    def port_side(self, port: PortRef, side: DiagramSide) -> None:
        """Set the marker boundary, preserving a previously explicit load side."""
        self._assert_current()
        self._require_ref(port, (PortRef,), operation="port_side")
        value = _side(side).value
        key = self._key(port)
        if key not in self._inventory.port_keys:
            raise _fail("port side requires a root PortRef")
        load = self._port_load_sides.get(key)
        if load is not None and not _perpendicular(value, load):
            raise _fail("Port boundary and load sides must be perpendicular", boundary_side=value, load_side=load)
        self._port_sides[key] = value
        if not self._building_automatic:
            self._explicit_port_sides.add(key)

    def port_orientation(self, port: PortRef, *, boundary_side: DiagramSide, load_side: DiagramSide) -> None:
        """Atomically select one of the eight complete root-frame Port poses."""
        self._assert_current()
        self._require_ref(port, (PortRef,), operation="port_orientation")
        boundary, load = _side(boundary_side).value, _side(load_side).value
        key = self._key(port)
        if key not in self._inventory.port_keys:
            raise _fail("Port orientation requires a root PortRef")
        if not _perpendicular(boundary, load):
            raise _fail("Port boundary and load sides must be perpendicular", boundary_side=boundary, load_side=load)
        self._port_sides[key], self._port_load_sides[key] = boundary, load
        if not self._building_automatic:
            self._explicit_port_sides.add(key)

    def tap_order(self, bus: BusRef, *, taps: tuple[TapRef, ...]) -> None:
        self._assert_current()
        self._require_ref(bus, (BusRef,), operation="tap_order")
        if not isinstance(taps, tuple):
            raise TypeError("tap order must be a tuple")
        for tap in taps:
            self._require_ref(tap, (TapRef,), operation="tap_order member")
        key = self._key(bus)
        values = tuple(self._key(tap) for tap in taps)
        if key not in self._inventory.taps or len(values) != len(set(values)) or set(values) != set(self._inventory.taps[key]):
            raise _fail("tap order must name exactly the authored taps of one Bus")
        self._tap_order[key] = values

    def ground_side(self, grounded_structure_or_eligible_parent_ground_pin: object, side: DiagramSide) -> None:
        self._assert_current()
        self._require_ref(grounded_structure_or_eligible_parent_ground_pin, (SeriesRef, ParallelRef, BranchRef, PinRef), operation="ground_side")
        value = _side(side).value
        key = self._key(grounded_structure_or_eligible_parent_ground_pin)
        if key not in self._inventory.ground_targets:
            raise _fail("ground side requires a grounded structure or eligible parent-grounded Pin")
        self._ground_sides[key] = value

    def endpoint(self, ref: object, *, boundary: str | None = None) -> SchematicEndpoint:
        self._assert_current()
        key = self._key(ref)
        if key[0] in {"series", "parallel", "branch"}:
            allowed = {"at", "end"} if key[0] == "branch" else {"start", "end"}
            if boundary not in allowed:
                raise _fail("structure endpoint needs an explicit eligible boundary", allowed=tuple(sorted(allowed)))
            key = (*key, boundary)
        elif key[0] in {"pin", "tap", "port"}:
            if boundary is not None:
                raise _fail("Pin, Port, and Tap already select their attachment; boundary is not allowed")
        else:
            raise TypeError("endpoint requires a structure, public Pin, Port, or authored Tap")
        candidates = self._inventory.endpoint_aliases.get(key, ())
        if not candidates:
            raise _fail("handle has no inter-Block wiring attachment; private or ground contacts cannot be rewired")
        return SchematicEndpoint(_owner=self, _candidates=candidates, _token=_REF_TOKEN)

    def replace_wiring(self, *, at: BusRef | LinkRef | PinRef) -> SchematicWiring:
        self._assert_current()
        key = self._key(at)
        if key[0] not in {"bus", "link", "pin", "wiring_group"}:
            raise TypeError("replace_wiring requires BusRef, LinkRef, or public PinRef")
        if key[0] == "pin":
            if key not in self._inventory.terminal_keys:
                raise _fail("wiring selection requires a public Pin")
            if key[4] and not key[1]:
                raise _fail("root public Pin has no parent-facing wiring group", pin=key)
            owner_scope = key[1][:-1] if key[4] else key[1]
            candidates = {group for group, _ in self._inventory.endpoint_aliases.get(key, ()) if group[0] == owner_scope and group in self._inventory.groups}
            if len(candidates) != 1:
                raise _fail("public Pin has no unique outward owner-local wiring group", pin=key)
            group_key = next(iter(candidates))
        else:
            group_key = key[1] if key[0] == "wiring_group" else self._inventory.group_aliases.get(key)
        if not self._building_automatic and group_key is not None and group_key[1].startswith("ground:"):
            raise _fail("ground networks do not support general wiring replacement")
        if group_key not in self._inventory.groups:
            raise _fail("selector has no inter-Block wiring group")
        result = SchematicWiring(_owner=self, _group=self._inventory.groups[group_key], _automatic=self._building_automatic, _token=_REF_TOKEN)
        self._wiring[group_key] = result
        return result

    def _preferred_side(self, contact: Attachment) -> str | None:
        from ..diagram.composition_model import endpoint_key

        key = endpoint_key(contact.endpoint)
        if contact.kind == "scope_pin":
            return self._scope_boundary_side(contact)
        if contact.kind == "port":
            return self._port_sides[key] if key in self._explicit_port_sides else self._default_port_side(contact)
        if contact.kind == "structure":
            direction = self._structure_boundary_side(contact)
            return self._parallel_exposure_side(contact, direction) or direction
        side = self._terminal_sides.get(key)
        if side is not None:
            return side if contact.kind == "scope_pin" else _OPPOSITE[side]
        if contact.preferred_side is not None:
            return _OPPOSITE[contact.preferred_side]
        return None

    def _parallel_exposure_side(self, contact: Attachment, direction: str) -> str | None:
        """Use a determined public exposure's lateral shared-rail access.

        This selects a contact on the existing rail, not a Parallel rotation.
        Multiple exposures must already agree; their names or unresolved sides
        never choose a preferred boundary. Explicit wiring bypasses this path.
        """
        if contact.block[0] != "parallel":
            return None
        group = self._inventory.groups.get(self._planning_group)
        if group is None or not any(row.key == contact.key for row in group.attachments):
            return None
        sides = tuple(
            self._scope_boundary_side(row) for row in group.attachments
            if row.kind == "scope_pin" and row.scope == contact.scope
        )
        if not sides or sides[0] is None or any(side != sides[0] for side in sides):
            return None
        return _OPPOSITE[sides[0]] if _perpendicular(sides[0], direction) else None

    def _structure_boundary_side(self, contact: Attachment) -> str:
        ground_side = self._ground_sides.get(contact.block)
        if ground_side is not None:
            grounded_boundary = self._inventory.ground_targets[contact.block].boundary
            return _OPPOSITE[ground_side] if contact.boundary == grounded_boundary else ground_side
        axis = self._axes.get(contact.block, self._inventory.blocks[contact.block].default_axis)
        selected = self._structure_recipe_side(contact, axis)
        if selected is not None:
            return selected
        at_start = contact.boundary != "end"
        return ("right" if at_start else "left") if axis == "horizontal" else ("bottom" if at_start else "top")

    def _structure_recipe_side(self, contact: Attachment, axis: str) -> str | None:
        """Use an already chosen compatible arm without treating a bend as rotation.

        A manual group can orient either end of a whole structure. Automatic
        Port groups additionally follow the non-Port group already planned at
        its other end; the converse is excluded to avoid cyclic preferences.
        """
        port_group = self._inventory.groups.get(self._planning_group)
        include_automatic = port_group is not None and port_group.scope == contact.scope and any(row.kind == "port" for row in port_group.attachments)
        ordered = sorted(self._wiring.values(), key=lambda row: row._automatic)
        for wiring in ordered:
            if wiring._group.key == self._planning_group:
                continue
            if wiring._automatic and (
                contact.block[0] == "parallel" or not include_automatic
                or any(row.kind == "port" for row in wiring._group.attachments)
            ):
                continue
            for other in wiring._group.attachments:
                if other.block != contact.block or other.kind != "structure":
                    continue
                side = self._attached_arm(wiring, other.key)
                if side is not None and (side in {"left", "right"}) == (axis == "horizontal"):
                    return side if other.boundary == contact.boundary else _OPPOSITE[side]
        return None

    @staticmethod
    def _attached_arm(wiring: SchematicWiring, key: Key) -> str | None:
        target = ("attachment", key)
        for connection in wiring._connections:
            other = connection.b if connection.a == target else connection.a if connection.b == target else None
            if other is not None and other[0] == "arm":
                return other[2]
        return None

    def _pin_incident_sides(self, contact: Attachment) -> tuple[str, ...]:
        """Read a public boundary's intrinsic incidences, not its name or role."""
        from ..diagram.composition_model import endpoint_key

        endpoint = contact.endpoint
        if endpoint.get("public"):
            path, key = tuple(endpoint["scope"]), endpoint_key(endpoint)
        else:
            path = (*endpoint["scope"], endpoint["component"])
            key = ("pin", path, endpoint["id"], endpoint["id"], True)
        sides = []
        for group_key, _ in self._inventory.endpoint_aliases.get(key, ()):
            if group_key[0] != path or group_key not in self._inventory.groups:
                continue
            for neighbor in self._inventory.groups[group_key].attachments:
                if neighbor.kind == "structure":
                    side = self._preferred_side(neighbor)
                    if side is not None:
                        sides.append(side)
        return tuple(sides)

    def _default_port_side(self, contact: Attachment) -> str:
        """Place an unspecified Port outward from its incident Block directions."""
        groups = {
            key for key, attachment in self._inventory.endpoint_aliases.get(contact.key, ())
            if key[0] == contact.scope and attachment == contact.key
            and key in self._inventory.groups
        }
        if len(groups) != 1:
            raise _fail("Port has no unique owner-local wiring group", port=contact.key)
        group = self._inventory.groups[next(iter(groups))]
        preferred = []
        for neighbor in group.attachments:
            if neighbor.kind == "port":
                continue
            side = self._preferred_side(neighbor)
            if side is not None:
                preferred.append(_OPPOSITE[side])
            elif neighbor.kind == "pin":
                preferred.extend(_OPPOSITE[side] for side in self._pin_incident_sides(neighbor))
        axis = self._axes.get(("scope", contact.scope), "horizontal")
        stable = ("left", "right", "top", "bottom") if axis == "horizontal" else ("top", "bottom", "left", "right")
        return preferred[0] if preferred and all(side == preferred[0] for side in preferred) else stable[0]

    def _commit_port_sides(self, wiring: SchematicWiring) -> None:
        for contact in wiring._group.attachments:
            if contact.kind != "port" or contact.key in self._explicit_port_sides:
                continue
            side = self._attached_arm(wiring, contact.key) or self._default_port_side(contact)
            from ...specs import DiagramSide

            self.port_side(self._target(contact.key), DiagramSide(side))

    def _ordered_groups(self):
        # Choose each owner-local non-Port assembly before its exterior Port
        # leads, and complete the parent before deriving a child's boundary.
        return sorted(
            self._inventory.groups.items(),
            key=lambda item: (
                len(item[0][0]), item[0][0],
                any(contact.kind == "port" for contact in item[1].attachments),
            ),
        )

    def _scope_boundary_side(self, contact: Attachment) -> str | None:
        """Derive a child boundary from its public constraint or parent recipe.

        The parent's explicit primitive arm determines where the whole child
        attaches. Its local automatic wiring is then authored through the same
        straight/elbow builders, rather than repaired by the lowerer.
        """
        from ..diagram.composition_model import endpoint_key

        path = contact.scope
        key = endpoint_key(contact.endpoint)
        outer = ("pin", path[:-1], path[-1], contact.endpoint["id"], False) if path else key
        from .automatic import _component_turns, _rotate_side

        turns = _component_turns(self, path)

        def child_side(side: str | None) -> str | None:
            # Whole Component orientation maps child-local ink to its parent.
            # Invert that transform for the private automatic child recipe;
            # inline Subsystem axes do not rotate their whole local frame.
            if side is None or not turns:
                return side
            return _rotate_side(side, -turns)

        for alias in (key, outer):
            side = self._terminal_sides.get(alias, self._ground_sides.get(alias))
            if side is not None:
                return child_side(side) if alias == outer else side
        candidates = tuple(dict.fromkeys(entry for alias in (key, outer) for entry in self._inventory.endpoint_aliases.get(alias, ()) if entry[0][0] == path[:-1] and entry[0][0] != path))
        for group_key, attachment_key in candidates:
            parent = self._wiring.get(group_key)
            if parent is None:
                continue
            target = ("attachment", attachment_key)
            for connection in parent._connections:
                other = connection.b if connection.a == target else connection.a if connection.b == target else None
                if other is None:
                    continue
                if other[0] == "arm":
                    return child_side(_OPPOSITE[other[2]])
                neighbor = next(row for row in parent._group.attachments if row.key == other[1])
                if neighbor.kind == "structure":
                    return child_side(self._preferred_side(neighbor))
        return None

    def _automatic_group(self, wiring: SchematicWiring) -> None:
        self._planning_group = wiring._group.key
        try:
            contacts = self._presentation_attachments(wiring._group)
            sides = tuple(self._preferred_side(contact) for contact in contacts)
            from .automatic import _populate_automatic

            _populate_automatic(wiring, contacts, sides)
            self._commit_port_sides(wiring)
        finally:
            self._planning_group = None

    def _presentation_attachments(self, group: WiringGroup) -> tuple[Attachment, ...]:
        """Conjoin explicit Tap precedence, retaining inventory as source truth.

        This is one stable topological order, not alternative permutations.
        Fixed recipes never call it; geometry still checks projected positions.
        """
        contacts = {contact.key: contact for contact in group.attachments}
        predecessors = {key: set() for key in contacts}
        for bus, taps in self._tap_order.items():
            if self._inventory.group_aliases.get(bus) != group.key:
                continue
            ordered = []
            for tap in taps:
                aliases = {
                    key for owner, key in self._inventory.endpoint_aliases.get(tap, ())
                    if owner == group.key and key in contacts
                }
                if len(aliases) != 1:
                    raise _fail("tap order requires one unambiguous actual attachment per tap", bus=bus, tap=tap)
                ordered.append(next(iter(aliases)))
            if len(set(ordered)) != len(ordered):
                raise _fail("tap order cannot order coincident attachment aliases", bus=bus)
            for before, after in zip(ordered, ordered[1:]):
                predecessors[after].add(before)
        remaining = list(contacts)
        result = []
        while remaining:
            key = next((key for key in remaining if not predecessors[key]), None)
            if key is None:
                raise _fail("explicit Tap orders have conflicting contact precedence", group=group.key)
            result.append(contacts[key])
            remaining.remove(key)
            for waiting in remaining:
                predecessors[waiting].discard(key)
        return tuple(result)

    @property
    def wiring(self) -> tuple[SchematicWiringRecord, ...]:
        """Read-only explicit edits and pending default groups; never plans."""
        self._assert_current()
        return tuple(
            _public_wiring(group, None if key not in self._wiring else self._wiring[key]._capture(complete=False),
                           "pending default")
            for key, group in self._inventory.groups.items() if group.addressable
        )

    def show(self) -> HtmlPresentation:
        """Summarize public Blocks, settings, and wiring without rendering."""
        from ...results import HtmlPresentation

        self._assert_current()
        lines = [f"SchematicComposition(plan={self._plan.id!r})"]
        for key, block in self._inventory.blocks.items():
            if not block.addressable:
                continue
            default = f"default {block.default_axis}"
            lines.append(f"{key!r}: axis={self._axes.get(key, default)}")
            if block.order_members is not None:
                lines.append(f"  order={self._order.get(key, 'source order')!r}")
        for key, group in self._inventory.groups.items():
            if group.addressable:
                builder = self._wiring.get(key)
                shapes = "pending default" if builder is None else tuple((item.id, item.kind, item.sides) for item in builder._junctions.values())
                lines.append(f"wiring {key!r}: {len(group.attachments)} attachments; shapes={shapes!r}")
        for name, values in (("branch_axes", self._branch_axes), ("terminal_sides", self._terminal_sides), ("port_sides", self._port_sides), ("port_load_sides", self._port_load_sides), ("tap_order", self._tap_order), ("ground_sides", self._ground_sides)):
            if values:
                lines.append(f"{name}={values!r}")
        return HtmlPresentation("<pre>" + escape("\n".join(lines)) + "</pre>")

    def _capture_intent_for(self, plan: CircuitPlan, *, snapshot: AuthoringSnapshot | None = None) -> CompositionIntent:
        if plan is not self._plan:
            raise _fail("composition belongs to a different CircuitPlan")
        self._assert_current(snapshot)
        for branch, axis in self._branch_axes.items():
            parent = self._parallel_for_branch(branch)
            ground = self._ground_sides.get(parent)
            effective = self._axes.get(parent, self._inventory.blocks[parent].default_axis) if ground is None else "horizontal" if ground in {"left", "right"} else "vertical"
            if axis != effective:
                raise _fail("Parallel branch axis conflicts with its containing Parallel", branch=branch, parallel=parent, requested_axis=axis, parallel_axis=effective)
        recipes = {key: wiring._capture(_checked=True) for key, wiring in self._wiring.items()}
        return CompositionIntent(
            inventory=self._inventory if snapshot is None else inventory(snapshot),
            axes=_frozen(self._axes), branch_axes=_frozen(self._branch_axes), order=_frozen(self._order),
            terminal_sides=_frozen(self._terminal_sides), port_sides=_frozen(self._port_sides),
            port_load_sides=_frozen(self._port_load_sides), tap_order=_frozen(self._tap_order),
            ground_sides=_frozen(self._ground_sides), fixed_wiring=_frozen(recipes),
        )

    def _capture_for(self, plan: CircuitPlan, *, snapshot: AuthoringSnapshot | None = None) -> CapturedComposition:
        """Internal diagnostic adapter to the same isolated materializer."""
        from .capture import materialize_composition

        return materialize_composition(self._capture_intent_for(plan, snapshot=snapshot))


@dataclass(frozen=True, slots=True)
class SchematicWiringRecord:
    """Public coordinate-free wiring inspection, without captured internals."""

    group: GroupKey
    aliases: tuple[Key, ...]
    attachments: tuple[Key, ...]
    junctions: tuple[Junction, ...]
    connections: tuple[Connection, ...]
    status: str


@dataclass(frozen=True, slots=True, init=False)
class SchematicCompositionSnapshot:
    """Detached public final recipe; reuse creates a new fixed composition."""

    _plan_sha256: str
    blocks: tuple[Key, ...]
    axes: Mapping[Key, str]
    order: Mapping[Key, tuple[Key, ...]]
    terminal_sides: Mapping[Key, str]
    port_sides: Mapping[Key, str]
    port_load_sides: Mapping[Key, str]
    tap_order: Mapping[Key, tuple[Key, ...]]
    ground_sides: Mapping[Key, str]
    wiring: tuple[SchematicWiringRecord, ...]

    def __init__(self, *, _token: object = None, **state: object) -> None:
        if _token is not _REF_TOKEN:
            raise TypeError("final composition snapshots are obtained from diagram.composition")
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, state[name])

    def show(self) -> HtmlPresentation:
        from ...results import HtmlPresentation

        lines = ["SchematicCompositionSnapshot (fixed public recipe)"]
        for key in self.blocks:
            lines.append(f"{key!r}: axis={self.axes[key]}")
            if key in self.order:
                lines.append(f"  order={self.order[key]!r}")
        for name in ("terminal_sides", "port_sides", "port_load_sides", "tap_order", "ground_sides"):
            if values := getattr(self, name):
                lines.append(f"{name}={dict(values)!r}")
        for recipe in self.wiring:
            lines.append(f"wiring {recipe.group!r}: {recipe.status}; shapes={recipe.junctions!r}")
        return HtmlPresentation("<pre>" + escape("\n".join(lines)) + "</pre>")

    def to_composition(self, *, plan: CircuitPlan) -> SchematicComposition:
        result = SchematicComposition(plan=plan)
        if result._inventory.plan_sha256 != self._plan_sha256:
            raise _fail("final composition snapshot does not match this exact Plan declaration")
        result._building_automatic = True
        try:
            from .capture import _apply_settings, _replay_wiring

            _apply_settings(result, self)
            for record in self.wiring:
                _replay_wiring(result, result._inventory.groups[record.group], record, automatic=False)
        finally:
            result._building_automatic = False
        return result


def _frozen(values: Mapping) -> Mapping:
    return MappingProxyType(dict(values))


def _perpendicular(first: str, second: str) -> bool:
    return (first in {"left", "right"}) != (second in {"left", "right"})


def _public_wiring(group: WiringGroup, recipe: WiringRecipe | None, missing: str = "unset") -> SchematicWiringRecord:
    return SchematicWiringRecord(
        group.key, group.aliases, tuple(contact.key for contact in group.attachments),
        () if recipe is None else recipe.junctions,
        () if recipe is None else recipe.connections,
        missing if recipe is None else "resolved default" if recipe.automatic else "explicit",
    )


# Preserve public class identity through the composition facade.
SchematicComposition.__module__ = "scnsim.composition"
SchematicCompositionSnapshot.__module__ = "scnsim.composition"

__all__ = ["SchematicComposition", "SchematicCompositionSnapshot"]
