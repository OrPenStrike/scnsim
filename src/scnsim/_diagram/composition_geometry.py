"""Constructive owner-local placement and fixed wiring realization.

A captured recipe supplies every arm and connection. Measured whole Blocks
are arranged once by a stable spanning forest; non-tree connections remain
explicit closure routes. No authoring search, candidate ranking, or alternate
routing is implemented here. The caller owns the independently verified scene.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

from ..errors import SCNSimValidationError
from .composition_model import Key, WiringRecipe
from .metrics import DEFAULT_METRICS
from .scene import COORDINATE_TOLERANCE, Bounds, ConductivePolyline, JumpArc, Point

VECTORS = {"left": (-1, 0), "right": (1, 0), "top": (0, 1), "bottom": (0, -1)}
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_layout", evidence=evidence)


@dataclass(frozen=True, slots=True)
class BlockGeometry:
    key: Key
    bounds: Bounds
    order_anchor: Point | None = None
    occupied_bounds: Bounds | None = None

    def order_point(self) -> Point:
        if self.order_anchor is not None:
            return self.order_anchor
        return Point(
            (self.bounds.xmin + self.bounds.xmax) / 2,
            (self.bounds.ymin + self.bounds.ymax) / 2,
        )


@dataclass(frozen=True, slots=True)
class ContactGeometry:
    block: Key
    point: Point
    side: str


@dataclass(frozen=True, slots=True)
class BoundaryContactGeometry:
    """A placeholder Block and an exact label offset about its final anchor."""

    key: Key
    side: str
    label_bounds: Bounds | None = None


@dataclass(frozen=True, slots=True)
class PlacedComposition:
    origins: Mapping[Key, Point]
    contacts: Mapping[Key, ContactGeometry]
    primitives: tuple[ConductivePolyline, ...]
    primitive_bounds: tuple[Bounds, ...]
    edges: tuple[tuple[str, str, ContactGeometry, ContactGeometry], ...]
    bounds: Bounds
    edge_paths: tuple[tuple[Point, ...], ...]


@dataclass(frozen=True, slots=True)
class RoutingContext:
    obstacles: tuple[Bounds, ...]
    allowed: Bounds
    scope: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoutedComposition:
    placed: PlacedComposition
    conductive: tuple[ConductivePolyline, ...]
    jumps: tuple[JumpArc, ...]
    length: float
    bends: int
    crossings: int
    detour: float
    area: float


def _distance(a: Point, b: Point) -> float:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _union(bounds: Sequence[Bounds]) -> Bounds:
    if not bounds:
        return Bounds(0.0, 0.0, 0.0, 0.0)
    return Bounds(
        min(b.xmin for b in bounds),
        min(b.ymin for b in bounds),
        max(b.xmax for b in bounds),
        max(b.ymax for b in bounds),
    )


def _expand(bounds: Bounds, gap: float) -> Bounds:
    return Bounds(
        bounds.xmin - gap, bounds.ymin - gap, bounds.xmax + gap, bounds.ymax + gap
    )


def straight_segment(first: Point, second: Point) -> ConductivePolyline:
    if first == second or (first.x != second.x and first.y != second.y):
        raise _fail("straight wiring primitive requires distinct cardinal endpoints")
    return ConductivePolyline((first, second))


def elbow_segments(
    first: Point, corner: Point, second: Point
) -> tuple[ConductivePolyline, ...]:
    if (first.x == corner.x) == (corner.x == second.x):
        raise _fail("elbow wiring primitive requires adjacent cardinal arms")
    return straight_segment(first, corner), straight_segment(corner, second)


def primitive_segments(
    center: Point, sides: Sequence[str]
) -> tuple[ConductivePolyline, ...]:
    if len(set(sides)) != len(sides) or any(side not in VECTORS for side in sides):
        raise _fail("captured junction has invalid cardinal arms")
    span = DEFAULT_METRICS.terminal_stub
    return tuple(
        straight_segment(center, center.translated(dx * span, dy * span))
        for side in sides
        for dx, dy in (VECTORS[side],)
    )


def _path(points: Sequence[Point]) -> tuple[Point, ...]:
    """Remove zero-length seams, but retain escape/corridor segment ownership."""
    result = []
    coordinates: tuple[list[tuple[float, float]], list[tuple[float, float]]] = ([], [])
    for point in points:
        values = []
        for coordinate, rows in zip((point.x, point.y), coordinates):
            shared = next(
                (
                    snapped
                    for measured, snapped in rows
                    if abs(measured - coordinate) <= COORDINATE_TOLERANCE
                ),
                None,
            )
            if shared is None:
                shared = round(coordinate / COORDINATE_TOLERANCE) * COORDINATE_TOLERANCE
                rows.append((coordinate, shared))
            values.append(shared)
        point = Point(*values)
        if not result or _distance(result[-1], point) > COORDINATE_TOLERANCE:
            if result and result[-1].x != point.x and result[-1].y != point.y:
                raise _fail(
                    "fixed connector contains a non-cardinal segment",
                    first=result[-1],
                    second=point,
                )
            result.append(point)
    if len(result) < 2:
        raise _fail("fixed connector has coincident physical contacts")
    return tuple(result)


def _escape(contact: ContactGeometry, bounds: Bounds | None) -> Point:
    dx, dy = VECTORS[contact.side]
    at = contact.point
    end = at.translated(
        dx * DEFAULT_METRICS.terminal_stub, dy * DEFAULT_METRICS.terminal_stub
    )
    if bounds is None:
        return end
    expanded = _expand(bounds, DEFAULT_METRICS.obstacle_clearance)
    if (
        expanded.xmin <= at.x <= expanded.xmax
        and expanded.ymin <= at.y <= expanded.ymax
    ):
        if dx:
            return Point(
                max(end.x, expanded.xmax) if dx > 0 else min(end.x, expanded.xmin),
                end.y,
            )
        return Point(
            end.x, max(end.y, expanded.ymax) if dy > 0 else min(end.y, expanded.ymin)
        )
    return end


def _ordinary_path(
    a: ContactGeometry,
    b: ContactGeometry,
    ea: Point,
    eb: Point,
    track: tuple[str, float] | None = None,
) -> tuple[Point, ...] | None:
    """Classify the prescribed direct, L, or opposed-arm corridor construction."""
    ax, ay = VECTORS[a.side]
    bx, by = VECTORS[b.side]
    if OPPOSITE[a.side] == b.side:
        if (
            (ax and a.point.y == b.point.y or ay and a.point.x == b.point.x)
            and (b.point.x - a.point.x) * ax + (b.point.y - a.point.y) * ay
            > COORDINATE_TOLERANCE
        ):
            # One continuous conductor has no separate, overlapping escape
            # strokes. Its owning endpoint obstacles remain checked below.
            return _path((a.point, b.point))
        if ax and (eb.x - ea.x) * ax >= -COORDINATE_TOLERANCE:
            middle = (
                track[1] if track is not None and track[0] == "x" else (ea.x + eb.x) / 2
            )
            return _path(
                (a.point, ea, Point(middle, ea.y), Point(middle, eb.y), eb, b.point)
            )
        if ay and (eb.y - ea.y) * ay >= -COORDINATE_TOLERANCE:
            middle = (
                track[1] if track is not None and track[0] == "y" else (ea.y + eb.y) / 2
            )
            return _path(
                (a.point, ea, Point(ea.x, middle), Point(eb.x, middle), eb, b.point)
            )
    elif ax * bx + ay * by == 0:
        corner = Point(eb.x, ea.y) if ax else Point(ea.x, eb.y)
        if (corner.x - ea.x) * ax + (
            corner.y - ea.y
        ) * ay >= -COORDINATE_TOLERANCE and (corner.x - eb.x) * bx + (
            corner.y - eb.y
        ) * by >= -COORDINATE_TOLERANCE:
            return _path((a.point, ea, corner, eb, b.point))
    elif a.side == b.side:
        if ax and abs(ea.y - eb.y) > COORDINATE_TOLERANCE:
            x = max(ea.x, eb.x) if ax > 0 else min(ea.x, eb.x)
            return _path((a.point, ea, Point(x, ea.y), Point(x, eb.y), eb, b.point))
        if ay and abs(ea.x - eb.x) > COORDINATE_TOLERANCE:
            y = max(ea.y, eb.y) if ay > 0 else min(ea.y, eb.y)
            return _path((a.point, ea, Point(ea.x, y), Point(eb.x, y), eb, b.point))
    return None


def _perimeter_route(a, b, ea, eb, rectangle, boundary_keys):
    """The fixed clockwise arc and the exact sides it occupies."""
    corners = (
        Point(rectangle.xmin, rectangle.ymax),
        Point(rectangle.xmax, rectangle.ymax),
        Point(rectangle.xmax, rectangle.ymin),
        Point(rectangle.xmin, rectangle.ymin),
    )
    width, height = rectangle.xmax - rectangle.xmin, rectangle.ymax - rectangle.ymin
    perimeter = 2 * (width + height)

    def landing(contact, exterior):
        # Interior contacts leave along their actual outward ray. Boundary
        # contacts enter the near side along their actual inward ray. Neither
        # turns back across its own stem or cuts a perpendicular body column.
        side = OPPOSITE[contact.side] if contact.block in boundary_keys else contact.side
        if side == "top":
            return Point(exterior.x, rectangle.ymax), exterior.x - rectangle.xmin
        if side == "right":
            return Point(
                rectangle.xmax, exterior.y
            ), width + rectangle.ymax - exterior.y
        if side == "bottom":
            return Point(
                exterior.x, rectangle.ymin
            ), width + height + rectangle.xmax - exterior.x
        return Point(
            rectangle.xmin, exterior.y
        ), 2 * width + height + exterior.y - rectangle.ymin

    first, start = landing(a, ea)
    last, stop = landing(b, eb)
    if stop < start:
        stop += perimeter
    corners_at = (
        (width, corners[1]),
        (width + height, corners[2]),
        (2 * width + height, corners[3]),
        (perimeter, corners[0]),
    )
    between = [
        point
        for position, point in (
            *corners_at,
            *((position + perimeter, point) for position, point in corners_at),
        )
        if start < position < stop
    ]
    points = (first, *between, last)
    sides = set()
    for p, q in pairwise(points):
        if p.y == q.y == rectangle.ymax:
            sides.add("top")
        elif p.y == q.y == rectangle.ymin:
            sides.add("bottom")
        elif p.x == q.x == rectangle.xmin:
            sides.add("left")
        elif p.x == q.x == rectangle.xmax:
            sides.add("right")
        else:
            raise _fail("fixed perimeter landing lies beyond its allocated side", first=p, second=q)
    return points, sides


def _closure_path(
    a: ContactGeometry, b: ContactGeometry, ea: Point, eb: Point, rectangle: Bounds,
    boundary_keys=frozenset(),
) -> tuple[Point, ...]:
    """One clockwise perimeter track, allocated by declaration order."""
    points, _ = _perimeter_route(a, b, ea, eb, rectangle, boundary_keys)
    return _path((a.point, ea, *points, eb, b.point))


def place_composition(
    blocks: Mapping[Key, BlockGeometry],
    attachments: Mapping[Key, ContactGeometry],
    recipes: Sequence[WiringRecipe],
    *,
    peer_order: Sequence[Key],
    axis: str,
    contact_order: Sequence[Sequence[Key]] = (),
    boundary_contacts: Sequence[BoundaryContactGeometry] = (),
    frame_insets: tuple[float, float, float, float] = (0, 0, 0, 0),
    header_bounds: Bounds | None = None,
    header_rotation: int = 0,
) -> PlacedComposition:
    """Place measured subtrees once and preallocate all connection tracks.

    frame_insets are (left, top, right, bottom). Boundary keys identify
    placeholder Blocks; label bounds are exact local-to-final-anchor offsets.
    Every returned boundary position and edge path is final: callers do not
    clamp anchors, resize children, or retry a different route.
    """
    if axis not in {"horizontal", "vertical"}:
        raise _fail("fixed composition requires a cardinal scope axis", axis=axis)
    if len(frame_insets) != 4 or any(value < 0 for value in frame_insets):
        raise ValueError("frame_insets must be four nonnegative measurements")
    entities = dict(blocks)
    boundary = {row.key: row for row in boundary_contacts}
    if len(boundary) != len(boundary_contacts):
        raise _fail("one boundary Block has duplicate measured contacts")
    if any(row.side not in VECTORS for row in boundary_contacts):
        raise _fail("boundary contact has an invalid cardinal side")
    if len(set(peer_order)) != len(peer_order) or any(
        key not in blocks or key in boundary for key in peer_order
    ):
        raise _fail("fixed peer order does not identify distinct measured bodies")
    contacts = {}
    links = []
    junctions = []
    straight_keys = set()
    junction_keys = set()
    span = DEFAULT_METRICS.terminal_stub
    for recipe_index, recipe in enumerate(recipes):
        group = recipe.group.key
        for attachment in recipe.group.attachments:
            if attachment.key not in attachments:
                raise _fail(
                    "captured attachment lacks a measured Block contact",
                    attachment=attachment.key,
                )
            row = attachments[attachment.key]
            if row.block not in entities or row.side not in VECTORS:
                raise _fail(
                    "captured attachment has no valid measured owner",
                    attachment=attachment.key,
                )
            contacts[(group, "attachment", attachment.key)] = row
        for junction in recipe.junctions:
            key = ("junction", group, junction.id)
            ink = primitive_segments(Point(0, 0), junction.sides)
            entities[key] = BlockGeometry(
                key, _union(tuple(wire.bounds for wire in ink))
            )
            junctions.append((key, junction))
            junction_keys.add(key)
            if junction.kind == "straight":
                straight_keys.add(key)
            for side in junction.sides:
                dx, dy = VECTORS[side]
                contacts[(group, "arm", junction.id, side)] = ContactGeometry(
                    key, Point(dx * span, dy * span), side
                )
        for index, connection in enumerate(recipe.connections):
            first, second = (group, *connection.a), (group, *connection.b)
            if first not in contacts or second not in contacts:
                raise _fail(
                    "captured edge has no measured attachment or arm", group=group
                )
            links.append(
                (
                    f"wire-{recipe_index:04d}-{index:04d}",
                    recipe.group.net,
                    first,
                    second,
                )
            )
    neighbors = defaultdict(list)
    for ident, net, a, b in links:
        first, second = contacts[a], contacts[b]
        neighbors[first.block].append((second.block, first, second, ident))
        neighbors[second.block].append((first.block, second, first, ident))
    ordered_owners = list(peer_order)
    precedes = defaultdict(set)
    for first, second in pairwise(peer_order):
        precedes[second].add(first)
    for ordered in contact_order:
        owners = []
        for key in ordered:
            rows = [
                row
                for contact_key, row in contacts.items()
                if len(contact_key) == 3
                and contact_key[1] == "attachment"
                and contact_key[2] == key
            ]
            if len(rows) != 1:
                raise _fail(
                    "fixed named contact order requires one measured incidence",
                    attachment=key,
                )
            owner = rows[0].block
            owners.append(owner)
            if owner not in ordered_owners:
                ordered_owners.append(owner)
        for first, second in pairwise(owners):
            if first != second:
                precedes[second].add(first)
    # Contact precedence orders supported subtree slots, not ownership: a Tap
    # does not promote its Block to an independent scope peer. In particular,
    # replaying an observed contact order retains the same spanning forest.
    ordered_roots = []
    pending = set(ordered_owners)
    while pending:
        ready = next(
            (
                key
                for key in ordered_owners
                if key in pending and not (precedes[key] & pending)
            ),
            None,
        )
        if ready is None:
            raise _fail(
                "fixed row placement cannot represent combined peer/contact order",
                peers=tuple(peer_order),
                contacts=tuple(map(tuple, contact_order)),
                blocked_owners=tuple(key for key in ordered_owners if key in pending),
            )
        ordered_roots.append(ready)
        pending.remove(ready)
    order_rank = {key: index for index, key in enumerate(ordered_roots)}
    row_roots = [key for key in peer_order if key not in boundary]
    # Explicit peers remain distinct ordered roots; spanning edges never absorb
    # another peer into an earlier root's privately measured subtree.
    parent = {key: None for key in row_roots}
    children = defaultdict(list)
    roots = list(row_roots)
    visited = set()
    tree_edges = set()

    def traverse(key):
        visited.add(key)
        for other, own, remote, ident in neighbors[key]:
            if other in boundary or other in parent:
                continue
            parent[other] = key
            children[key].append((other, own, remote, ident))
            tree_edges.add(ident)
            traverse(other)

    for key in (*row_roots, *entities):
        if key in boundary or key in visited:
            continue
        if key not in parent:
            roots.append(key)
            parent[key] = None
        traverse(key)

    lane = DEFAULT_METRICS.routing_lane_pitch
    clearance = 2 * DEFAULT_METRICS.obstacle_clearance
    side_order = (
        ("right", "bottom", "left", "top")
        if axis == "horizontal"
        else ("bottom", "right", "top", "left")
    )
    measured_subtrees = {}

    def measure(key):
        origins = {key: Point(0, 0)}
        envelope = entities[key].bounds
        groups = defaultdict(list)
        for other, own, remote, ident in children[key]:
            child_origins, child_bounds = measure(other)
            groups[own.side].append((other, own, remote, child_origins, child_bounds))
        for side, rows in groups.items():
            if contact_order and (
                (axis == "horizontal" and side in {"top", "bottom"})
                or (axis == "vertical" and side in {"left", "right"})
            ):
                rows.sort(key=lambda row: order_rank.get(row[0], len(order_rank)))

        def transverse_extent(side):
            """Measured side branches reserve their exit width on a spine."""
            horizontal = side in {"left", "right"}
            intervals = []
            for branch_side in (("top", "bottom") if horizontal else ("left", "right")):
                end = None
                for other, own, remote, _, child_bounds in groups[branch_side]:
                    # A same-direction return has its own perimeter strip;
                    # unlike a direct/L branch it has no local spine exit.
                    if remote.side == branch_side:
                        continue
                    remote_escape = _escape(remote,
                        entities[other].occupied_bounds or entities[other].bounds)
                    turn_room = _distance(remote.point, remote_escape) + span
                    if horizontal:
                        offset = own.point.x - remote.point.x
                        offset -= VECTORS[remote.side][0] * turn_room
                        if end is not None:
                            offset = max(offset, end + clearance - child_bounds.xmin)
                        low, high = child_bounds.xmin + offset, child_bounds.xmax + offset
                        end = high
                    else:
                        offset = own.point.y - remote.point.y
                        offset -= VECTORS[remote.side][1] * turn_room
                        if end is not None:
                            offset = min(offset, end - clearance - child_bounds.ymax)
                        low, high = child_bounds.ymin + offset, child_bounds.ymax + offset
                        end = low
                    intervals.append((low, high))
            return intervals

        for side in side_order:
            rows = groups[side]
            if not rows:
                continue
            base = envelope
            group_bounds = []
            transverse_end = None
            for other, own, remote, child_origins, child_bounds in rows:
                if key in junction_keys and other in junction_keys and remote.side == OPPOSITE[side]:
                    intervals = transverse_extent(side)
                    if intervals:
                        low, high = min(row[0] for row in intervals), max(row[1] for row in intervals)
                        base = (Bounds(min(base.xmin,low),base.ymin,max(base.xmax,high),base.ymax)
                            if side in {"left","right"} else
                            Bounds(base.xmin,min(base.ymin,low),base.xmax,max(base.ymax,high)))
                gap = (
                    DEFAULT_METRICS.obstacle_clearance
                    if (key in straight_keys or other in straight_keys)
                    and OPPOSITE[own.side] == remote.side
                    else clearance
                ) + (len(rows) - 1) * lane
                remote_escape = _escape(
                    remote, entities[other].occupied_bounds or entities[other].bounds
                )
                turn_room = _distance(remote.point, remote_escape) + span
                if side in {"left", "right"}:
                    dx = (
                        base.xmax + gap - child_bounds.xmin
                        if side == "right"
                        else base.xmin - gap - child_bounds.xmax
                    )
                    dy = own.point.y - remote.point.y
                    if remote.side in {"top", "bottom"}:
                        dy -= VECTORS[remote.side][1] * turn_room
                    elif remote.side == side:
                        dy = base.ymax + clearance - child_bounds.ymin
                    if transverse_end is not None:
                        dy = min(dy, transverse_end - clearance - child_bounds.ymax)
                    transverse_end = child_bounds.ymin + dy
                else:
                    dy = (
                        base.ymax + gap - child_bounds.ymin
                        if side == "top"
                        else base.ymin - gap - child_bounds.ymax
                    )
                    dx = own.point.x - remote.point.x
                    if remote.side in {"left", "right"}:
                        dx -= VECTORS[remote.side][0] * turn_room
                    elif remote.side == side:
                        dx = base.xmax + clearance - child_bounds.xmin
                    if transverse_end is not None:
                        dx = max(dx, transverse_end + clearance - child_bounds.xmin)
                    transverse_end = child_bounds.xmax + dx
                origins.update(
                    (child_key, at.translated(dx, dy))
                    for child_key, at in child_origins.items()
                )
                group_bounds.append(child_bounds.translated(dx, dy))
            envelope = _union((envelope, *group_bounds))
        measured_subtrees[key] = envelope
        return origins, envelope

    origins = {}
    planned_tracks = {}
    envelopes = []
    previous_peer = None
    for root in roots:
        local, envelope = measure(root)
        shift = Point(0, 0)
        if envelopes:
            reference = entities[root].order_point()
            transverse = -reference.y if axis == "horizontal" else -reference.x
            contact_alignment = None
            transverse_side = None
            for ident, _, a, b in links:
                own, other = contacts[a], contacts[b]
                if own.block not in local:
                    own, other = other, own
                if own.block in local and other.block in origins:
                    own_at = own.point.translated(
                        local[own.block].x, local[own.block].y
                    )
                    other_at = other.point.translated(
                        origins[other.block].x, origins[other.block].y
                    )
                    transverse = (
                        other_at.y - own_at.y
                        if axis == "horizontal"
                        else other_at.x - own_at.x
                    )
                    # A peer on a transverse junction arm is a whole side
                    # branch, not a contact aligned through the main row.
                    # Reserve the already measured attachment subtree before
                    # placing that peer; all later columns retain this strip.
                    other_origin = origins[other.block]
                    branch = measured_subtrees[other.block].translated(
                        other_origin.x, other_origin.y
                    )
                    if axis == "horizontal" and other.side in {"top", "bottom"}:
                        contact_alignment = other_at.x - own_at.x
                        transverse_side = other.side
                        transverse = (
                            branch.ymax + clearance - envelope.ymin
                            if other.side == "top"
                            else branch.ymin - clearance - envelope.ymax
                        )
                        planned_tracks[ident] = (
                            "y",
                            branch.ymax + clearance / 2
                            if other.side == "top"
                            else branch.ymin - clearance / 2,
                        )
                    elif axis == "vertical" and other.side in {"left", "right"}:
                        contact_alignment = other_at.y - own_at.y
                        transverse_side = other.side
                        transverse = (
                            branch.xmax + clearance - envelope.xmin
                            if other.side == "right"
                            else branch.xmin - clearance - envelope.xmax
                        )
                        planned_tracks[ident] = (
                            "x",
                            branch.xmax + clearance / 2
                            if other.side == "right"
                            else branch.xmin - clearance / 2,
                        )
                    break
            previous = _union(envelopes)
            shift = (
                Point(previous.xmax + clearance - envelope.xmin, transverse)
                if axis == "horizontal"
                else Point(transverse, previous.ymin - clearance - envelope.ymax)
            )
            if contact_alignment is not None:
                # A side-hung whole peer occupies its transverse strip. Its
                # contact aligns with the actual parent arm, not the far end
                # of that arm's unrelated downstream connector envelope.
                if axis == "horizontal":
                    shift = Point(
                        max(contact_alignment, previous_peer + clearance - reference.x)
                        if previous_peer is not None
                        else contact_alignment,
                        transverse,
                    )
                else:
                    shift = Point(
                        transverse,
                        min(contact_alignment, -previous_peer - clearance - reference.y)
                        if previous_peer is not None
                        else contact_alignment,
                    )
                # The connected subtree is not the only occupant of this
                # side strip: an earlier sibling arm may already hang here.
                # Conjoin its measured envelope as a scalar separation bound
                # before finalizing this whole peer's origin.
                for key, at in origins.items():
                    prior = entities[key].bounds.translated(at.x, at.y)
                    if axis == "horizontal" and (
                        prior.xmax + clearance > envelope.xmin + shift.x
                        and prior.xmin - clearance < envelope.xmax + shift.x
                    ):
                        shift = Point(shift.x,
                            max(shift.y, prior.ymax + clearance - envelope.ymin)
                            if transverse_side == "top" else
                            min(shift.y, prior.ymin - clearance - envelope.ymax))
                    elif axis == "vertical" and (
                        prior.ymax + clearance > envelope.ymin + shift.y
                        and prior.ymin - clearance < envelope.ymax + shift.y
                    ):
                        shift = Point(
                            max(shift.x, prior.xmax + clearance - envelope.xmin)
                            if transverse_side == "right" else
                            min(shift.x, prior.xmin - clearance - envelope.xmax), shift.y)
        origins.update(
            (key, at.translated(shift.x, shift.y)) for key, at in local.items()
        )
        envelopes.append(envelope.translated(shift.x, shift.y))
        if root in peer_order:
            reference = entities[root].order_point()
            previous_peer = (
                origins[root].x + reference.x
                if axis == "horizontal"
                else -origins[root].y - reference.y
            )

    body_bounds = _union(envelopes)
    coordinate_rows: tuple[list[float], list[float]] = ([], [])

    def canonical_contact(point: Point) -> Point:
        values = []
        for coordinate, rows in zip((point.x, point.y), coordinate_rows):
            shared = next(
                (
                    value
                    for value in rows
                    if abs(value - coordinate) <= COORDINATE_TOLERANCE
                ),
                None,
            )
            if shared is None:
                shared = round(coordinate / COORDINATE_TOLERANCE) * COORDINATE_TOLERANCE
                rows.append(shared)
            values.append(shared)
        return Point(*values)

    world_contacts = {
        key: ContactGeometry(
            row.block,
            canonical_contact(
                row.point.translated(origins[row.block].x, origins[row.block].y)
            ),
            row.side,
        )
        for key, row in contacts.items()
        if row.block not in boundary
    }
    occupied = {
        key: (row.occupied_bounds or row.bounds).translated(
            origins[key].x, origins[key].y
        )
        for key, row in entities.items()
        if key not in boundary
    }
    provisional = {}
    closure_ids = []
    for ident, net, a, b in links:
        if contacts[a].block in boundary or contacts[b].block in boundary:
            continue
        first, second = world_contacts[a], world_contacts[b]
        ea, eb = (
            _escape(first, occupied[first.block]),
            _escape(second, occupied[second.block]),
        )
        ordinary = _ordinary_path(first, second, ea, eb, planned_tracks.get(ident))
        # Tree traversal is only a placement device. Every closure edge is
        # retained, including a directly representable cycle-closing stroke.
        if ordinary is not None:
            provisional[ident] = ordinary
        else:
            closure_ids.append(ident)
    left, top, right, bottom = frame_insets
    border = DEFAULT_METRICS.label_clearance
    label_depth = dict.fromkeys(VECTORS, 0.0)
    for row in boundary_contacts:
        if row.label_bounds is not None:
            label_depth[row.side] = max(
                label_depth[row.side],
                {
                    "left": row.label_bounds.xmax,
                    "right": -row.label_bounds.xmin,
                    "top": -row.label_bounds.ymin,
                    "bottom": row.label_bounds.ymax,
                }[row.side]
                + clearance,
            )
    boundary_sides = {row.side for row in boundary_contacts}
    padding = {
        side: max(
            inset + border,
            label_depth[side] + DEFAULT_METRICS.obstacle_clearance,
            clearance if side in boundary_sides else 0.0,
        )
        for side, inset in (("left", left), ("top", top), ("right", right), ("bottom", bottom))
    }
    # Tangential contact/label slots are independent of the final frame's
    # normal coordinates. Resolve them before claiming any perimeter lanes.
    frame = body_bounds
    by_side = defaultdict(list)
    for row in boundary_contacts:
        attached = next(
            (
                contacts[b] if contacts[a].block == row.key else contacts[a]
                for _, _, a, b in links
                if row.key in (contacts[a].block, contacts[b].block)
            ),
            None,
        )
        proposal = Point(0, 0)
        if attached is not None and attached.block in origins:
            proposal = attached.point.translated(
                origins[attached.block].x, origins[attached.block].y
            )
            # A perpendicular boundary entry needs a real transverse lane,
            # not the other arm's exact coordinate (which would force an
            # inward retrace or a full perimeter circuit). Put that lane past
            # the measured contents on the requested arm's outgoing side.
            if row.side in {"left", "right"}:
                if attached.side == "top":
                    proposal = Point(proposal.x, body_bounds.ymax + clearance)
                elif attached.side == "bottom":
                    proposal = Point(proposal.x, body_bounds.ymin - clearance)
            elif attached.side == "left":
                proposal = Point(body_bounds.xmin - clearance, proposal.y)
            elif attached.side == "right":
                proposal = Point(body_bounds.xmax + clearance, proposal.y)
        by_side[row.side].append((row, proposal))
    positions = {}
    for side, rows in by_side.items():
        if contact_order and (
            (axis == "horizontal" and side in {"top", "bottom"})
            or (axis == "vertical" and side in {"left", "right"})
        ):
            rows.sort(key=lambda item: order_rank.get(item[0].key, len(order_rank)))
        cursor = None
        for row, proposal in rows:
            label = row.label_bounds
            if side in {"left", "right"}:
                low, high = (label.ymin, label.ymax) if label else (0.0, 0.0)
                level = (
                    proposal.y
                    if cursor is None
                    else min(proposal.y, cursor - lane - high)
                )
                cursor = level + low
                positions[row.key] = level
                frame = Bounds(
                    frame.xmin,
                    min(frame.ymin, level + low),
                    frame.xmax,
                    max(frame.ymax, level + high),
                )
            else:
                low, high = (label.xmin, label.xmax) if label else (0.0, 0.0)
                level = (
                    proposal.x
                    if cursor is None
                    else max(proposal.x, cursor + lane - low)
                )
                cursor = level + high
                positions[row.key] = level
                frame = Bounds(
                    min(frame.xmin, level + low),
                    frame.ymin,
                    max(frame.xmax, level + high),
                    frame.ymax,
                )
    channel_body = frame
    if header_bounds is not None:
        sides = ("right", "top", "left", "bottom")
        local_top = sides[(1 - header_rotation // 90) % 4]
        local_left = sides[(2 - header_rotation // 90) % 4]
        def final_entry(row):
            level = positions[row.key]
            point = Point(0.0, level) if row.side in {"left", "right"} else Point(level, 0.0)
            label = row.label_bounds
            offset = min((Point(x,y).rotated(header_rotation).x
                for x in (label.xmin,label.xmax) for y in (label.ymin,label.ymax)),default=0.0) if label else 0.0
            return point.rotated(header_rotation).x + min(0.0, offset)
        top_entries = [
            final_entry(row) for row in boundary_contacts if row.side == local_top
        ]
        if top_entries:
            # Keep the upright corner title beside, not under, actual top
            # contact rays and captions. This is measured space, not a lane.
            final_xmin = min(Point(x,y).rotated(header_rotation).x
                for x in (channel_body.xmin,channel_body.xmax)
                for y in (channel_body.ymin,channel_body.ymax))
            padding[local_left] = max(
                padding[local_left],
                final_xmin - min(top_entries)
                + header_bounds.xmax - header_bounds.xmin
                + border + DEFAULT_METRICS.obstacle_clearance,
            )

    def framed(depths):
        return Bounds(
            channel_body.xmin - depths["left"],
            channel_body.ymin - depths["bottom"],
            channel_body.xmax + depths["right"],
            channel_body.ymax + depths["top"],
        )

    def boundary_point(key, row, bounds):
        level = positions[key]
        return (
            Point(bounds.xmin if row.side == "left" else bounds.xmax, level)
            if row.side in {"left", "right"}
            else Point(level, bounds.ymax if row.side == "top" else bounds.ymin)
        )

    # This minimal exterior establishes route classes, not candidate geometry.
    # Growing an exterior along its normal preserves direct/L/U feasibility;
    # there is one classification and no collision-driven alternative.
    nominal_frame = framed(padding)
    nominal_contacts = dict(world_contacts)
    nominal_occupied = dict(occupied)
    for key, row in boundary.items():
        at = boundary_point(key, row, nominal_frame)
        if row.label_bounds is not None:
            nominal_occupied[key] = row.label_bounds.translated(at.x, at.y)
        for contact_key, contact in contacts.items():
            if contact.block == key:
                nominal_contacts[contact_key] = ContactGeometry(key, at, OPPOSITE[row.side])
    for ident, _, a, b in links:
        if contacts[a].block not in boundary and contacts[b].block not in boundary:
            continue
        first, second = nominal_contacts[a], nominal_contacts[b]
        ea = _escape(first, nominal_occupied.get(first.block))
        eb = _escape(second, nominal_occupied.get(second.block))
        if _ordinary_path(first, second, ea, eb, planned_tracks.get(ident)) is None:
            closure_ids.append(ident)
    lane_counts = dict.fromkeys(VECTORS, 0)
    closure_rectangles = {}
    reference_rectangle = _expand(channel_body, clearance + lane)
    for ident, _, a, b in links:
        if ident not in closure_ids:
            continue
        first, second = nominal_contacts[a], nominal_contacts[b]
        ea = _escape(first, nominal_occupied.get(first.block))
        eb = _escape(second, nominal_occupied.get(second.block))
        _, used_sides = _perimeter_route(first, second, ea, eb, reference_rectangle, boundary)
        depths = dict.fromkeys(VECTORS, clearance)
        for side in used_sides:
            lane_counts[side] += 1
            depths[side] = clearance + (lane_counts[side] - 1) * lane
        closure_rectangles[ident] = framed(depths)
    frame = framed({
        side: padding[side] + (clearance + (count - 1) * lane if count else 0.0)
        for side, count in lane_counts.items()
    })
    for key, row in boundary.items():
        level = positions[key]
        at = boundary_point(key, row, frame)
        origins[key] = at
        for contact_key, contact in contacts.items():
            if contact.block == key:
                world_contacts[contact_key] = ContactGeometry(
                    key, canonical_contact(at), OPPOSITE[row.side]
                )
    for key, row in boundary.items():
        if row.label_bounds is not None:
            label = row.label_bounds.translated(origins[key].x, origins[key].y)
            occupied[key] = label
            if not frame.contains(label):
                raise _fail(
                    "fixed boundary label extends outside its measured frame",
                    block=key,
                    label_bounds=label,
                    frame=frame,
                )
    projected = [
        origins[key].x + entities[key].order_point().x
        if axis == "horizontal"
        else -origins[key].y - entities[key].order_point().y
        for key in peer_order
    ]
    if any(b - a <= COORDINATE_TOLERANCE for a, b in pairwise(projected)):
        raise _fail(
            "fixed subtree placement cannot represent declared peer order",
            peers=tuple(peer_order),
        )
    for ordered in contact_order:
        points = []
        for key in ordered:
            rows = [
                value
                for contact_key, value in world_contacts.items()
                if len(contact_key) == 3
                and contact_key[1] == "attachment"
                and contact_key[2] == key
            ]
            if len(rows) != 1:
                raise _fail(
                    "fixed named contact order requires one measured incidence",
                    attachment=key,
                )
            points.append(rows[0].point.x if axis == "horizontal" else -rows[0].point.y)
        if any(b - a <= COORDINATE_TOLERANCE for a, b in pairwise(points)):
            raise _fail(
                "fixed subtree placement cannot represent declared contact order",
                contacts=tuple(ordered),
                projected=tuple(points),
            )
    edges = tuple(
        (ident, net, world_contacts[a], world_contacts[b]) for ident, net, a, b in links
    )
    paths = []
    for ident, net, first, second in edges:
        if ident in provisional:
            paths.append(provisional[ident])
            continue
        ea = _escape(first, occupied.get(first.block))
        eb = _escape(second, occupied.get(second.block))
        if ident in closure_rectangles:
            paths.append(_closure_path(first, second, ea, eb, closure_rectangles[ident], boundary))
            continue
        ordinary = _ordinary_path(first, second, ea, eb, planned_tracks.get(ident))
        if ordinary is None:
            raise _fail("fixed boundary expansion changed its prescribed route class", edge=ident)
        paths.append(ordinary)
    primitives = tuple(
        wire
        for key, junction in junctions
        for wire in primitive_segments(origins[key], junction.sides)
    )
    primitive_bounds = tuple(
        entities[key].bounds.translated(origins[key].x, origins[key].y)
        for key, _ in junctions
    )
    contact_frame = _expand(frame, COORDINATE_TOLERANCE)
    if any(not contact_frame.contains(Bounds.around(path)) for path in paths):
        raise _fail(
            "fixed closure tracks do not fit the preallocated frame",
            connections=tuple(
                edge[0]
                for edge, path in zip(edges, paths)
                if not contact_frame.contains(Bounds.around(path))
            ),
        )
    return PlacedComposition(
        MappingProxyType(origins),
        MappingProxyType(world_contacts),
        primitives,
        primitive_bounds,
        edges,
        frame,
        tuple(paths),
    )


def _intersection(a: Point, b: Point, c: Point, d: Point) -> tuple[Point, Point] | None:
    """Closed orthogonal intersection, retaining a positive overlap interval."""
    av, cv = a.x == b.x, c.x == d.x
    if av == cv:
        if (a.x != c.x) if av else (a.y != c.y):
            return None
        low = (
            max(min(a.y, b.y), min(c.y, d.y))
            if av
            else max(min(a.x, b.x), min(c.x, d.x))
        )
        high = (
            min(max(a.y, b.y), max(c.y, d.y))
            if av
            else min(max(a.x, b.x), max(c.x, d.x))
        )
        if high < low - COORDINATE_TOLERANCE:
            return None
        return (
            (Point(a.x, low), Point(a.x, high))
            if av
            else (Point(low, a.y), Point(high, a.y))
        )
    vertical, horizontal = ((a, b), (c, d)) if av else ((c, d), (a, b))
    point = Point(vertical[0].x, horizontal[0].y)
    if (
        min(vertical[0].y, vertical[1].y) - COORDINATE_TOLERANCE
        <= point.y
        <= max(vertical[0].y, vertical[1].y) + COORDINATE_TOLERANCE
        and min(horizontal[0].x, horizontal[1].x) - COORDINATE_TOLERANCE
        <= point.x
        <= max(horizontal[0].x, horizontal[1].x) + COORDINATE_TOLERANCE
    ):
        return point, point
    return None


def route_composition(
    placed: PlacedComposition, *, context: RoutingContext
) -> RoutedComposition:
    """Validate and emit the one preset routing; never choose another path."""
    from .routing import (
        _compress,
        _deduplicate,
        _edge_interactions,
        _in_bounds,
        _inflate,
        _inside,
        _on_segment,
        _segment_clear,
        _split_jumps,
        _visible_bends,
    )

    if not isinstance(context, RoutingContext):
        raise TypeError("fixed routing requires RoutingContext")
    if len(placed.edges) != len(placed.edge_paths):
        raise _fail("fixed routing lost a declared connection", scope=context.scope)
    measured_obstacles = (*context.obstacles, *placed.primitive_bounds)
    obstacles = tuple(
        _inflate(row, DEFAULT_METRICS.obstacle_clearance) for row in measured_obstacles
    )
    obstacle_contacts = {
        inflated: tuple(
            dict.fromkeys(
                contact.block
                for edge in placed.edges
                for contact in edge[2:]
                if _in_bounds(contact.point, measured)
            )
        )
        for measured, inflated in zip(measured_obstacles, obstacles)
    }

    def endpoints(first: ContactGeometry, last: ContactGeometry) -> dict:
        return {
            "first_block": first.block,
            "first_side": first.side,
            "second_block": last.block,
            "second_side": last.side,
        }

    wires = []
    owners = []
    physical = {}
    for (ident, net, first, last), path in zip(placed.edges, placed.edge_paths):
        evidence = endpoints(first, last)
        if (
            _distance(path[0], first.point) > COORDINATE_TOLERANCE
            or _distance(path[-1], last.point) > COORDINATE_TOLERANCE
        ):
            raise _fail(
                "fixed route changed a measured attachment",
                scope=context.scope,
                edge=ident,
                **evidence,
            )
        if any(not _in_bounds(point, context.allowed) for point in path):
            raise _fail(
                "fixed route leaves its allocated frame",
                scope=context.scope,
                edge=ident,
                **evidence,
            )
        segments = tuple(pairwise(path))
        for index, (a, b) in enumerate(segments):
            for other_index, (c, d) in enumerate(segments[index + 1 :], index + 1):
                intersection = _intersection(a, b, c, d)
                if intersection is not None and (
                    other_index > index + 1
                    or _distance(*intersection) > COORDINATE_TOLERANCE
                ):
                    raise _fail(
                        "fixed connector crosses or retraces itself",
                        scope=context.scope,
                        edge=ident,
                        intersection=intersection,
                        **evidence,
                    )
        for contact, a, b in ((first, path[0], path[1]), (last, path[-1], path[-2])):
            dx, dy = VECTORS[contact.side]
            if (
                (b.x - a.x) * dx + (b.y - a.y) * dy <= COORDINATE_TOLERANCE
                or (dx and b.y != a.y)
                or (dy and b.x != a.x)
            ):
                raise _fail(
                    "fixed route cannot realize an explicit outward arm",
                    scope=context.scope,
                    edge=ident,
                    block=contact.block,
                    side=contact.side,
                    **evidence,
                )
        for index, (a, b) in enumerate(pairwise(path)):
            permitted = ()
            if index == 0:
                permitted += tuple(
                    row for row in obstacles if _in_bounds(first.point, row)
                )
            if index == len(path) - 2:
                permitted += tuple(
                    row for row in obstacles if _in_bounds(last.point, row)
                )
            remaining = tuple(row for row in obstacles if row not in permitted)
            if not _segment_clear(a, b, remaining):
                conflicts = tuple(
                    row for row in remaining if not _segment_clear(a, b, (row,))
                )
                raise _fail(
                    "fixed connector intersects occupied geometry",
                    scope=context.scope,
                    edge=ident,
                    first=a,
                    second=b,
                    obstacles=conflicts,
                    conflicting_blocks=tuple(
                        dict.fromkeys(
                            block
                            for row in conflicts
                            for block in obstacle_contacts[row]
                        )
                    ),
                    **evidence,
                )
            if (
                index == 0
                and len(path) > 2
                and any(_inside(b, row) for row in obstacles)
            ):
                raise _fail(
                    "fixed anchor escape does not leave occupied geometry",
                    scope=context.scope,
                    edge=ident,
                    block=first.block,
                    **evidence,
                )
            if (
                index == len(path) - 2
                and len(path) > 2
                and any(_inside(a, row) for row in obstacles)
            ):
                raise _fail(
                    "fixed anchor escape does not leave occupied geometry",
                    scope=context.scope,
                    edge=ident,
                    block=last.block,
                    **evidence,
                )
        wires.append(ConductivePolyline(path))
        owners.append(ident)
        physical[ident] = net
    # Validate intersections before deduplication; an undeclared overlap must
    # not disappear merely because its two edges share electrical identity.
    for index, (edge, path) in enumerate(zip(placed.edges, placed.edge_paths)):
        ident, net, first, last = edge
        for other, other_path in zip(
            placed.edges[index + 1 :], placed.edge_paths[index + 1 :]
        ):
            allowed = {a.point for a in (first, last) for b in other[2:] if a == b}
            for a, b in pairwise(_compress(path)):
                for c, d in pairwise(_compress(other_path)):
                    intersection = _intersection(a, b, c, d)
                    if intersection is None:
                        continue
                    low, high = intersection
                    if net == other[1]:
                        if _distance(low, high) > COORDINATE_TOLERANCE or not any(
                            _distance(low, p) <= COORDINATE_TOLERANCE for p in allowed
                        ):
                            raise _fail(
                                "fixed same-net routes introduce an undeclared junction or overlap",
                                scope=context.scope,
                                edges=(ident, other[0]),
                                intersection=intersection,
                                connections=(
                                    endpoints(first, last),
                                    endpoints(other[2], other[3]),
                                ),
                            )
                    elif _edge_interactions(
                        a, b, ((c, d),), jump_gap=DEFAULT_METRICS.jump_gap
                    )[0]:
                        raise _fail(
                            "fixed different-net crossing lacks jump clearance",
                            scope=context.scope,
                            edges=(ident, other[0]),
                            intersection=intersection,
                            connections=(
                                endpoints(first, last),
                                endpoints(other[2], other[3]),
                            ),
                        )
    wires, owners = _deduplicate(wires, owners)
    contacts = {contact.point for edge in placed.edges for contact in edge[2:]}
    split, split_owners = [], []
    for wire, owner in zip(wires, owners):
        a, b = wire.points
        points = sorted(
            {a, b, *(point for point in contacts if _on_segment(point, a, b))},
            key=lambda point: (point.x, point.y),
        )
        for first, last in pairwise(points):
            if _distance(first, last) > COORDINATE_TOLERANCE:
                split.append(ConductivePolyline((first, last)))
                split_owners.append(owner)
    conductive, jumps = _split_jumps(split, split_owners, DEFAULT_METRICS)
    length = sum(_distance(a, b) for wire in split for a, b in pairwise(wire.points))
    minimum = sum(_distance(a.point, b.point) for _, _, a, b in placed.edges)
    return RoutedComposition(
        placed,
        conductive,
        jumps,
        length,
        _visible_bends(split),
        len(jumps),
        max(0.0, length - minimum),
        (context.allowed.xmax - context.allowed.xmin)
        * (context.allowed.ymax - context.allowed.ymin),
    )
