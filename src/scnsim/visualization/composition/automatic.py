"""Deterministic fixed-default wiring and local orientation helpers."""

from __future__ import annotations

from ..diagram.composition_model import Attachment, WiringGroup
from ..schematic import DiagramAxis
from .api import (
    SchematicComposition, SchematicEndpoint, SchematicWiring, _OPPOSITE,
    _REF_TOKEN, _SIDES,
)

def _component_turns(work: SchematicComposition, path: tuple[str, ...]) -> int:
    key = ("component", path[:-1], path[-1]) if path else None
    block = work._inventory.blocks.get(key)
    axis = None if block is None else work._axes.get(key, block.default_axis)
    return 0 if block is None or axis == block.default_axis else 3 if axis == "vertical" else 1


def _rotate_side(side: str, turns: int) -> str:
    cardinal = ("right", "top", "left", "bottom")
    return cardinal[(cardinal.index(side) + turns) % 4]


def _assign_sides(preferred: tuple[str | None, ...], slots: tuple[str, ...]):
    """Assign exact directions, then source contacts to remaining stable arms."""
    available = list(range(len(slots)))
    assigned: dict[int, int] = {}
    for index, side in enumerate(preferred):
        slot = next((slot for slot in available if slots[slot] == side), None)
        if slot is not None:
            assigned[index] = slot
            available.remove(slot)
    for index in range(len(slots)):
        if index not in assigned:
            assigned[index] = available.pop(0)
    return tuple(assigned[index] for index in range(len(slots)))


def _populate_automatic(
    wiring: SchematicWiring, contacts: tuple[Attachment, ...], preferred_sides: tuple[str | None, ...],
) -> None:
    """Commit a deterministic topology using the public primitive operations.

    Exact three/four cardinal directions select a T/Cross. Otherwise use the
    scope's fixed T direction/chain. There is no tree or axis search.
    """
    from ...specs import DiagramSide

    group = wiring._group
    endpoints = tuple(SchematicEndpoint(_owner=wiring._owner, _candidates=((group.key, contact.key),), _token=_REF_TOKEN) for contact in contacts)
    count = len(endpoints)
    known = tuple(side for side in preferred_sides if side is not None)
    scope_axis = wiring._owner._axes[("scope", group.scope)]
    if count == 2:
        if len(known) == 2 and known[0] != known[1] and _OPPOSITE[known[0]] != known[1]:
            junction = wiring.elbow(id="automatic_elbow", sides=tuple(DiagramSide(side) for side in known))
        else:
            axis = DiagramAxis.VERTICAL if known and all(side in {"top", "bottom"} for side in known) else DiagramAxis.HORIZONTAL if known else DiagramAxis(scope_axis)
            junction = wiring.straight(id="automatic_straight", axis=axis)
        sides = junction._junction.sides
        assignment = _assign_sides(preferred_sides, sides)
        _connect_automatic_contacts(wiring, contacts, endpoints, tuple(getattr(junction, sides[slot]) for slot in assignment))
        return
    if count == 3:
        branch = _OPPOSITE[next(side for side in _SIDES if side not in known)] if len(set(known)) == 3 else "bottom" if scope_axis == "horizontal" else "right"
        junction = wiring.tee(id="automatic_tee", branch=DiagramSide(branch))
        sides = junction._junction.sides
    elif count == 4 and len(set(known)) == 4:
        sides = _SIDES
        junction = wiring.cross(id="automatic_cross")
    else:
        # The n-2 fixed Ts have n external arms and n-3 joining wires.
        through = ("left", "right") if scope_axis == "horizontal" else ("top", "bottom")
        branch = "bottom" if scope_axis == "horizontal" else "right"
        junctions = tuple(
            wiring.tee(id=f"automatic_tee_{index + 1}", branch=DiagramSide(branch))
            for index in range(count - 2)
        )
        for index in range(len(junctions) - 1):
            wiring.connect(getattr(junctions[index], through[1]), getattr(junctions[index + 1], through[0]))
        slots = tuple(
            (index, side) for index, junction in enumerate(junctions) for side in junction._junction.sides
            if not (side == through[0] and index > 0 or side == through[1] and index + 1 < len(junctions))
        )
        assignment = _assign_sides(preferred_sides, tuple(side for _, side in slots))
        _connect_automatic_contacts(wiring, contacts, endpoints, tuple(getattr(junctions[slots[slot][0]], slots[slot][1]) for slot in assignment))
        return
    assignment = _assign_sides(preferred_sides, sides)
    _connect_automatic_contacts(wiring, contacts, endpoints, tuple(getattr(junction, sides[slot]) for slot in assignment))


def _connect_automatic_contacts(
    wiring: SchematicWiring, contacts: tuple[Attachment, ...],
    endpoints: tuple[SchematicEndpoint, ...], arms: tuple[SchematicArm, ...],
) -> None:
    """Preserve Parallel reading direction in the public generated trace.

    A lateral shared-rail contact is valid. An opposite along-axis contact
    would instead reverse the whole Parallel, so prefer a compatible available
    arm or author a two-elbow return using the same single external contact.
    """
    from ...specs import DiagramSide

    required = tuple(
        wiring._owner._structure_boundary_side(contact)
        if contact.kind == "structure" and contact.block[0] == "parallel" else None
        for contact in contacts
    )
    selected = list(arms)
    for index, direction in enumerate(required):
        if direction is None or selected[index]._side != _OPPOSITE[direction]:
            continue
        alternatives = [
            other for other, arm in enumerate(selected)
            if other != index and arm._side != _OPPOSITE[direction]
            and (required[other] is None or selected[index]._side != _OPPOSITE[required[other]])
        ]
        if alternatives:
            other = next((item for item in alternatives if selected[item]._side == direction), alternatives[0])
            selected[index], selected[other] = selected[other], selected[index]
    for index, (endpoint, arm, direction) in enumerate(zip(endpoints, selected, required, strict=True)):
        if direction is None or arm._side != _OPPOSITE[direction]:
            wiring.connect(endpoint, arm)
            continue
        lateral = "right" if direction in {"top", "bottom"} else "top"
        entry = wiring.elbow(
            id=f"automatic_parallel_entry_{index + 1}",
            sides=(DiagramSide(direction), DiagramSide(lateral)),
        )
        returned = wiring.elbow(
            id=f"automatic_parallel_return_{index + 1}",
            sides=(DiagramSide(direction), DiagramSide(_OPPOSITE[lateral])),
        )
        wiring.connect(endpoint, getattr(entry, direction))
        wiring.connect(getattr(entry, lateral), getattr(returned, _OPPOSITE[lateral]))
        wiring.connect(getattr(returned, direction), arm)


