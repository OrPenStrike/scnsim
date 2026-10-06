"""Capture, replay, and fixed completion for schematic compositions."""

from __future__ import annotations

from dataclasses import replace

from ...authoring.snapshot import AuthoringSnapshot
from ..diagram.composition_model import (
    CapturedComposition, CompositionIntent, WiringGroup, WiringRecipe, inventory,
)
from ..schematic import DiagramAxis, _fail
from ...authoring import CircuitPlan
from ...errors import SCNSimValidationError
from .api import (
    SchematicArm, SchematicComposition, SchematicCompositionSnapshot,
    SchematicEndpoint, SchematicWiringRecord, _OPPOSITE, _REF_TOKEN, _SIDES,
    _frozen, _public_wiring,
)
from .automatic import _component_turns, _rotate_side

def detached_composition(captured: CapturedComposition) -> SchematicCompositionSnapshot:
    """Copy only caller-addressable authoring records from the winning recipe."""
    visible_blocks = {key for key, block in captured.inventory.blocks.items() if block.addressable}

    def visible(key: Key) -> bool:
        if key[0] == "port":
            return True
        scope = captured.inventory.blocks.get(("scope", key[1]))
        return scope is not None and scope.addressable

    return SchematicCompositionSnapshot(
        _token=_REF_TOKEN, _plan_sha256=captured.inventory.plan_sha256,
        blocks=tuple(key for key in captured.inventory.blocks if key in visible_blocks),
        axes=_frozen({key: value for key, value in captured.axes.items() if key in visible_blocks}),
        order=_frozen({key: value for key, value in captured.order.items() if key in visible_blocks}),
        **{name: _frozen({key: value for key, value in getattr(captured, name).items() if visible(key)})
           for name in ("terminal_sides", "port_sides", "port_load_sides", "tap_order", "ground_sides")},
        wiring=tuple(_public_wiring(recipe.group, recipe) for recipe in captured.wiring.values() if recipe.group.addressable),
    )


def _apply_settings(work: SchematicComposition, settings: object) -> None:
    """Replay captured settings through the same validated public operations."""
    from ...specs import DiagramSide

    for key, axis in settings.axes.items():
        work.axis(work._target(key), DiagramAxis(axis))
    for key, members in settings.order.items():
        work.order(work._target(key), members=tuple(work._target(member) for member in members))
    for key, side in settings.terminal_sides.items():
        work.terminal_side(work._target(key), DiagramSide(side))
    for key, side in settings.port_sides.items():
        load = settings.port_load_sides.get(key)
        if load is None:
            work.port_side(work._target(key), DiagramSide(side))
        else:
            work.port_orientation(work._target(key), boundary_side=DiagramSide(side), load_side=DiagramSide(load))
    for key, members in settings.tap_order.items():
        work.tap_order(work._target(key), taps=tuple(work._target(member) for member in members))
    for key, side in settings.ground_sides.items():
        work.ground_side(work._target(key), DiagramSide(side))


def _replay_wiring(work: SchematicComposition, group: WiringGroup, recipe: WiringRecipe | SchematicWiringRecord, *, automatic: bool) -> None:
    from ...specs import DiagramSide

    builder = work.replace_wiring(at=work._target(("wiring_group", group.key)))
    builder._automatic = automatic
    junctions = {}
    for item in recipe.junctions:
        if item.kind == "straight":
            axis = DiagramAxis.HORIZONTAL if set(item.sides) == {"left", "right"} else DiagramAxis.VERTICAL
            ref = builder.straight(id=item.id, axis=axis)
        elif item.kind == "elbow":
            ref = builder.elbow(id=item.id, sides=tuple(DiagramSide(side) for side in item.sides))
        elif item.kind == "tee":
            missing = set(_SIDES) - set(item.sides)
            if len(missing) != 1:
                raise _fail("captured T has invalid arms")
            ref = builder.tee(id=item.id, branch=DiagramSide(_OPPOSITE[missing.pop()]))
        elif item.kind == "cross":
            ref = builder.cross(id=item.id)
        else:
            raise _fail("captured wiring contains an unknown primitive", kind=item.kind)
        if ref._junction.sides != item.sides:
            raise _fail("captured shape arms do not match its primitive", junction=item.id)
        junctions[item.id] = ref

    def endpoint(key: Key) -> SchematicEndpoint | SchematicArm:
        if key[0] == "attachment":
            if key[1] not in {contact.key for contact in group.attachments}:
                raise _fail("captured connection names a foreign attachment", group=group.key)
            return SchematicEndpoint(_owner=work, _candidates=((group.key, key[1]),), _token=_REF_TOKEN)
        if key[0] == "arm" and key[1] in junctions:
            return junctions[key[1]]._arm(key[2])
        raise _fail("captured connection names an unknown arm", group=group.key)

    for connection in recipe.connections:
        builder.connect(endpoint(connection.a), endpoint(connection.b))
    builder._capture()


def materialize_composition(intent: CompositionIntent) -> CapturedComposition:
    """Complete one fixed recipe in isolation, before any body measurement.

    Omitted fields use stable source defaults. Explicit traces replay the
    public operations unchanged. No live Plan reads, alternative candidates,
    geometry callbacks, rendering, or solver calls enter this path.
    """
    from ...specs import DiagramSide

    if not isinstance(intent, CompositionIntent):
        raise TypeError("default completion requires CompositionIntent")
    work = SchematicComposition.__new__(SchematicComposition)
    work._plan, work._inventory = None, intent.inventory
    work._initialize_declarations()
    work._building_automatic = True
    _apply_settings(work, intent)
    for key, block in intent.inventory.blocks.items():
        if key not in work._axes:
            work.axis(work._target(key), DiagramAxis(block.default_axis))
        if block.order_members is not None and key not in work._order:
            work.order(work._target(key), members=tuple(work._target(member) for member in block.order_members))
    work._explicit_port_sides = set(work._port_sides)
    for key, recipe in intent.fixed_wiring.items():
        _replay_wiring(work, intent.inventory.groups[key], recipe, automatic=False)
    for key, group in work._ordered_groups():
        if key in intent.fixed_wiring:
            work._commit_port_sides(work._wiring[key])
        else:
            builder = work.replace_wiring(at=work._target(("wiring_group", key)))
            work._automatic_group(builder)
    for key in sorted(intent.inventory.port_keys, key=repr):
        boundary = work._port_sides.get(key, "left")
        load = work._port_load_sides.get(key, "bottom" if boundary in {"left", "right"} else "right")
        work.port_orientation(work._target(key), boundary_side=DiagramSide(boundary), load_side=DiagramSide(load))
    _complete_boundary_and_ground_sides(work)
    return CapturedComposition(
        inventory=intent.inventory, axes=_frozen(work._axes), order=_frozen(work._order),
        terminal_sides=_frozen(work._terminal_sides), port_sides=_frozen(work._port_sides),
        port_load_sides=_frozen(work._port_load_sides), tap_order=_frozen(work._tap_order),
        ground_sides=_frozen(work._ground_sides),
        wiring=_frozen({key: builder._capture() for key, builder in work._wiring.items()}),
    )


def _complete_boundary_and_ground_sides(work: SchematicComposition) -> None:
    """Resolve source-frame sides from the already selected immutable trace."""
    from ..diagram.composition_model import endpoint_key
    from ...specs import DiagramSide

    for key, contact in work._inventory.scope_boundaries.items():
        side = work._scope_boundary_side(contact)
        if side is None:
            side = next((
                arm for group, attachment in work._inventory.endpoint_aliases.get(key, ())
                if group[0] == contact.scope and group in work._wiring
                and (arm := work._attached_arm(work._wiring[group], attachment)) is not None
            ), "left")
        work.terminal_side(work._target(key), DiagramSide(side))
        path = contact.scope
        if not path:
            continue
        outer = ("pin", path[:-1], path[-1], contact.endpoint["id"], False)
        if outer in work._inventory.terminal_keys:
            parent_side = _rotate_side(side, _component_turns(work, path))
            if outer in work._terminal_sides and work._terminal_sides[outer] != parent_side:
                raise _fail("one exposed physical pin has conflicting selected sides", pin=outer)
            work.terminal_side(work._target(outer), DiagramSide(parent_side))
    for key, target in work._inventory.ground_targets.items():
        if key in work._ground_sides:
            continue
        if target.block is not None:
            axis = work._axes[target.block]
            side = "right" if axis == "horizontal" else "bottom"
            found = False
            for wiring in work._wiring.values():
                for contact in wiring._group.attachments:
                    if contact.block != target.block:
                        continue
                    arm = work._attached_arm(wiring, contact.key)
                    direction = None if arm is None else _OPPOSITE[arm] if contact.boundary == "end" else arm
                    if direction is not None and (direction in {"left", "right"}) == (axis == "horizontal"):
                        side, found = direction, True
                        break
                if found:
                    break
        else:
            side = work._terminal_sides.get(endpoint_key(target.endpoint), "left")
        work.ground_side(work._target(key), DiagramSide(side))


def finalize_composition(
    captured: CapturedComposition, *, tap_order: Mapping[Key, tuple[Key, ...]],
) -> CapturedComposition:
    """Attach realized public Tap order without rebuilding wiring or geometry.

    Geometry supplies the complete strictly projected observed orders from its
    same scene. Here ownership, named membership, unique contact incidence and
    existing explicit constraints are checked through the public order builder.
    """
    if not isinstance(captured, CapturedComposition):
        raise TypeError("composition finalization requires CapturedComposition")
    work = SchematicComposition.__new__(SchematicComposition)
    work._plan, work._inventory = None, captured.inventory
    work._initialize_declarations()
    work._building_automatic = True
    work._tap_order = dict(captured.tap_order)
    for key, members in tap_order.items():
        scope = captured.inventory.blocks.get(("scope", key[1])) if key in captured.inventory.taps else None
        if scope is None or not scope.addressable:
            raise _fail("realized tap order requires a caller-addressable Bus", bus=key)
        if not isinstance(members, tuple):
            raise TypeError("realized tap order members must be a tuple")
        if key in captured.tap_order and captured.tap_order[key] != members:
            raise _fail("realized tap order conflicts with its explicit constraint", bus=key)
        work.tap_order(work._target(key), taps=tuple(work._target(member) for member in members))
        contacts = []
        for tap in members:
            aliases = {
                contact for group, contact in captured.inventory.endpoint_aliases.get(tap, ())
                if group[0] == key[1] and group in captured.inventory.groups
            }
            if len(aliases) != 1:
                raise _fail("realized tap order requires one unambiguous actual attachment per tap", bus=key, tap=tap)
            contacts.append(next(iter(aliases)))
        if len(set(contacts)) != len(contacts):
            raise _fail("realized tap order cannot order coincident attachment aliases", bus=key)
    return replace(captured, tap_order=_frozen(work._tap_order))


