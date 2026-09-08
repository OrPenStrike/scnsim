"""Independent visible-ink checks for captured wiring shapes and explicit sides.

The recipe supplies expectations, never observed positions. Electrical contact
signatures bind otherwise symmetric ordinary terminals; shape correspondence
then uses the complete split segment graph, preserving every branch vertex.
Applicable axes and reading order come from visible bodies, contours, and
uniquely bound contacts, with no placement tables or hidden terminal ordering.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from ..errors import SCNSimValidationError
from .audit import _conductive_paths, _is_ground_paths, _point_on_segment, _port_identity, _same_point, _segment_intersection, _verify_port_block_geometry
from .composition_model import Attachment, CapturedComposition, Key, WiringRecipe
from .reconstruction import _body_groups, _boundary_incidence, _boundary_segments, _inside_region, _junction_segments, _regions, _verify_junction_marks
from .scene import Bounds, COORDINATE_TOLERANCE, NeutralScene, Point


_OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
_SYMMETRIC = {"resistor", "capacitor", "inductor", "josephson_junction"}


def _symmetric_body(body: Any) -> bool:
    return body.model in _SYMMETRIC or body.model == "transmission_line" and body.n_sections is not None


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_layout", evidence=evidence)


def _direction(first: Point, second: Point) -> str:
    dx, dy = second.x - first.x, second.y - first.y
    if abs(dy) <= COORDINATE_TOLERANCE and abs(dx) > COORDINATE_TOLERANCE:
        return "right" if dx > 0 else "left"
    if abs(dx) <= COORDINATE_TOLERANCE and abs(dy) > COORDINATE_TOLERANCE:
        return "top" if dy > 0 else "bottom"
    raise _fail("composition incidence is not a positive-length cardinal ray")


@dataclass
class _Graph:
    points: list[Point]
    edges: dict[int, set[int]]
    external: set[frozenset[int]]
    jump_ends: set[int]
    components: dict[int, int]
    ground_components: set[int]

    def index(self, point: Point) -> int:
        for index, existing in enumerate(self.points):
            if _same_point(point, existing):
                return index
        self.points.append(point)
        return len(self.points) - 1

    def root(self, point: Point) -> int:
        index = self.index(point)
        if index not in self.components:
            raise _fail("composition contact does not meet continuous visible wire ink")
        return self.components[index]

    def is_ground(self, root: int) -> bool:
        return root in self.ground_components


def _build_graph(scene: NeutralScene, regions: Sequence[Any], bodies: Sequence[Any], straight_count: int) -> tuple[_Graph, tuple[tuple[Point, Point], ...]]:
    graph = _Graph([], defaultdict(set), set(), set(), {}, set())
    segments = tuple((a, b) for path in _conductive_paths(scene) for a, b in pairwise(path) if not _same_point(a, b))
    external = tuple((a, b) for wire in scene.conductive for a, b in pairwise(wire.points) if not _same_point(a, b))
    for a, b in segments:
        _direction(a, b)
        graph.index(a)
        graph.index(b)
    for index, first in enumerate(segments):
        for second in segments[index + 1:]:
            relation, point = _segment_intersection(first, second)
            if relation == "point" and point is not None:
                graph.index(point)
    for body in bodies:
        for _, point in (*body.pin_points, *body.auxiliary_pin_points):
            graph.index(point)
    for port in scene.ports:
        graph.index(port.circuit_anchor)
        graph.index(port.external_anchor)
        graph.index(port.ground_anchor)
    for mark in scene.node_marks:
        graph.index(mark.point)
    for region in regions:
        if not region.path:
            continue
        for boundary in _boundary_segments(region):
            for segment in segments:
                relation, point = _segment_intersection(boundary, segment)
                if relation == "point" and point is not None:
                    graph.index(point)
    # A straight has no visible center identity. Interior sample points allow
    # injective embeddings on a positive run even when serialized as one line.
    for a, b in external:
        for index in range(1, straight_count + 1):
            fraction = index / (straight_count + 1)
            graph.index(Point(a.x + fraction * (b.x - a.x), a.y + fraction * (b.y - a.y)))
    for a, b in segments:
        indexes = sorted(
            (index for index, point in enumerate(graph.points) if _point_on_segment(point, a, b)),
            key=lambda index: abs(graph.points[index].x - a.x) + abs(graph.points[index].y - a.y),
        )
        for first, second in pairwise(indexes):
            if first == second:
                continue
            graph.edges[first].add(second)
            graph.edges[second].add(first)
            midpoint = Point((graph.points[first].x + graph.points[second].x) / 2, (graph.points[first].y + graph.points[second].y) / 2)
            if any(_point_on_segment(midpoint, left, right) for left, right in external):
                graph.external.add(frozenset((first, second)))
    # A jump is continuous along its own arc, not at its under-wire crossing.
    # It is traversable subdivision geometry, never a straight/T/Cross center.
    for jump in scene.jumps:
        first, second = graph.index(jump.path.points[0]), graph.index(jump.path.points[-1])
        _direction(graph.points[first], graph.points[second])
        graph.edges[first].add(second)
        graph.edges[second].add(first)
        graph.external.add(frozenset((first, second)))
        graph.jump_ends.update((first, second))
    for start in range(len(graph.points)):
        if start in graph.components:
            continue
        pending = [start]
        while pending:
            current = pending.pop()
            if current in graph.components:
                continue
            graph.components[current] = start
            pending.extend(graph.edges[current] - graph.components.keys())
    ground_points = [guide.terminals[0] for guide in scene.guides if guide.label is None and len(guide.terminals) == 1 and _is_ground_paths(guide.paths, guide.terminals[0])]
    ground_points.extend(port.ground_anchor for port in scene.ports)
    graph.ground_components.update(graph.root(point) for point in ground_points)
    return graph, segments


def _scope_rows(snapshot: Mapping[str, object]) -> dict[tuple[str, ...], Any]:
    rows = {}

    def visit(scope: Any) -> None:
        rows[tuple(scope["path"])] = scope
        for child in scope["children"]:
            visit(child)
        for body in scope["component_bodies"]:
            visit(body["body"])

    visit(snapshot["scope_hierarchy"])
    return rows


class _Bindings:
    """Bind source attachment labels to independently observed contact sets."""

    def __init__(self, scene: NeutralScene, captured: CapturedComposition) -> None:
        self.scene = scene
        self.captured = captured
        self.semantic = captured.inventory.snapshot.semantic_record
        self.scopes = _scope_rows(self.semantic)
        self.regions = _regions(scene)
        shown = any(symbol.value is not None for symbol in scene.symbols) or any(box.length_label is not None for box in scene.boxes)
        self.bodies = _body_groups(scene, self.regions, show_values=shown)
        self.body_map = {body.path: body for body in self.bodies}
        count = sum(junction.kind == "straight" for recipe in captured.wiring.values() for junction in recipe.junctions)
        self.graph, self.segments = _build_graph(scene, self.regions, self.bodies, count)
        _verify_junction_marks(scene, _junction_segments(scene, self.segments))
        roots = {root: "ground" if self.graph.is_ground(root) else f"contact-{root}" for root in set(self.graph.components.values())}
        _boundary_incidence(scene, self.regions, self.bodies, self.graph, roots, self.segments)
        self.boundaries: dict[tuple[tuple[str, ...], str | None], int] = {}
        for region in self.regions:
            if not region.path:
                continue
            contacts = set()
            for boundary in _boundary_segments(region):
                for segment in self.segments:
                    relation, point = _segment_intersection(boundary, segment)
                    if relation == "point" and point is not None:
                        contacts.add(self.graph.index(point))
            for vertex in contacts:
                labels = [site.visible_label.text for site in scene.boundary_sites if site.visible_label is not None and _same_point(site.point, self.graph.points[vertex])]
                label = None if len(contacts) == 1 else labels[0]
                self.boundaries[(region.path, label)] = vertex
        self.endpoint_nets = {(tuple(row["path"]), row["pin"]): row["net"] for row in self.semantic["connectivity"]["physical_endpoints"]}
        self.structures = {(row["kind"], path, row["id"]): row for path, scope in self.scopes.items() for row in scope["structures"] if row["kind"] in {"series", "parallel", "branch"}}
        self.expected_signatures: dict[str, set[Key]] = defaultdict(set)
        self.observed_signatures: dict[int, set[Key]] = defaultdict(set)
        symmetric = {tuple(row["path"]) for row in self.semantic["physical_leaves"] if row["model"] in _SYMMETRIC or row["model"] == "transmission_line" and len(row["pin_order"]) == 2}
        for (path, pin), net in self.endpoint_nets.items():
            self.expected_signatures[net].add(("body", path, "symmetric" if path in symmetric else pin))
        for body in self.bodies:
            for pin, point in body.pin_points:
                self.observed_signatures[self.graph.root(point)].add(("body", body.path, "symmetric" if _symmetric_body(body) else pin))
        for row in self.semantic["connectivity"]["ports"]:
            self.expected_signatures[row["net"]].add(("port", row["id"]))
        self.ports = {_port_identity(port)[0]: port for port in scene.ports}
        for identifier, port in self.ports.items():
            self.observed_signatures[self.graph.root(port.external_anchor)].add(("port", identifier))
        for path, scope in self.scopes.items():
            pins = scope["exposures"]["pins"]
            for pin in pins:
                self.expected_signatures[pin["final_net"]].add(("boundary", path, None if len(pins) == 1 else pin["id"]))
        for (path, name), vertex in self.boundaries.items():
            self.observed_signatures[self.graph.components[vertex]].add(("boundary", path, name))

    def components(self, net: str) -> tuple[int, ...]:
        if net == "ground":
            return tuple(sorted(self.graph.ground_components))
        return tuple(root for root, signature in self.observed_signatures.items() if root not in self.graph.ground_components and signature == self.expected_signatures[net])

    def world_side(self, side: str, scope: tuple[str, ...]) -> str:
        """Transform local expectations by captured complete-Composite frames.

        This is a source-side coordinate convention, not a placement lookup.
        An enclosing Composite's H-to-V frame turns clockwise; its parent
        recipes and caller-facing outer Pin settings remain outside that frame.
        """
        degrees = 0
        for length in range(1, len(scope) + 1):
            path = scope[:length]
            key = ("component", path[:-1], path[-1])
            block = self.captured.inventory.blocks.get(key)
            if path in self.scopes and block is not None:
                axis = self.captured.axes[key]
                if axis != block.default_axis:
                    degrees += 270 if axis == "vertical" else 90
        cardinal = ("right", "top", "left", "bottom")
        return cardinal[(cardinal.index(side) + degrees // 90) % 4]

    def boundary(self, path: tuple[str, ...], name: str) -> int:
        if (path, name) in self.boundaries:
            return self.boundaries[(path, name)]
        if (path, None) in self.boundaries:
            return self.boundaries[(path, None)]
        raise _fail("requested composition pin has no visible named or sole contour crossing", path=path, pin=name)

    def pin(self, key: Key) -> tuple[int, tuple[str, ...]]:
        _, scope, component, name, public = key
        path = scope if public else (*scope, component)
        if path in self.scopes:
            return self.boundary(path, name), path
        body = self.body_map[path]
        if body.model == "transmission_line" and _symmetric_body(body):
            # CPW has no visible authored head/tail labels. Resolve a requested
            # pin only through its independently observed contact topology.
            net = self.endpoint_nets[(path, name)]
            vertices = self.member({"path": path}, name, set(self.components(net)))
            if len(vertices) != 1:
                raise _fail("CPW terminal side has ambiguous visible topology", pin=key)
            return next(iter(vertices)), path
        points = [point for pin, point in body.pin_points if pin == name]
        if len(points) != 1 or body.model in _SYMMETRIC:
            raise _fail("explicit terminal side lacks a visible labelled physical pin", pin=key)
        return self.graph.index(points[0]), path

    def member(self, member: Any, pin: str, components: set[int]) -> frozenset[int]:
        path = tuple(member["path"])
        if path not in self.body_map:
            vertex = self.boundary(path, pin)
            if self.graph.components[vertex] not in components:
                raise _fail("visible structure boundary pin belongs to a different contact network", path=path, pin=pin)
            return frozenset((vertex,))
        body = self.body_map[path]
        primary = [(name, point) for name, point in body.pin_points if (_symmetric_body(body) or name == pin) and self.graph.root(point) in components]
        if len(primary) != 1:
            raise _fail("source structure boundary has no unique visible symmetric-reference binding", path=path, pin=pin)
        points = [primary[0][1]]
        points.extend(point for _, point in body.auxiliary_pin_points if self.graph.root(point) in components)
        return frozenset(self.graph.index(point) for point in points)

    def structure(self, key: Key, boundary: str, components: set[int]) -> frozenset[int]:
        row = self.structures[key]
        ends = []
        for branch in row.get("branches", (row,)):
            member = branch["elements"][-1 if boundary == "end" else 0]
            ends.extend(self.member(member, member["pin_2" if boundary == "end" else "pin_1"], components))
        return frozenset(ends)

    def attachment(self, attachment: Attachment, component: int) -> frozenset[int]:
        endpoint = attachment.endpoint
        if attachment.kind == "structure":
            return self.structure(attachment.block, attachment.boundary, {component})
        if attachment.kind == "port":
            return frozenset((self.graph.index(self.ports[endpoint["id"]].external_anchor),))
        if endpoint["kind"] != "pin":
            raise _fail("captured composition attachment lacks a visible binding grammar", attachment=attachment.key)
        scope = tuple(endpoint["scope"])
        path = scope if endpoint.get("public") else (*scope, endpoint["component"])
        if path in self.scopes:
            return frozenset((self.boundary(path, endpoint["id"]),))
        return self.member({"path": path}, endpoint["id"], {component})

    def local_graph(self, scope: tuple[str, ...], component: int) -> dict[int, set[int]]:
        local: dict[int, set[int]] = defaultdict(set)
        own = next(region for region in self.regions if region.path == scope)
        children = [region for region in self.regions if region.path and region.path[:-1] == scope]
        for edge in self.graph.external:
            first, second = tuple(edge)
            if self.graph.components[first] != component:
                continue
            a, b = self.graph.points[first], self.graph.points[second]
            midpoint = Point((a.x + b.x) / 2, (a.y + b.y) / 2)
            if scope and not _inside_region(midpoint, own):
                continue
            if any(_inside_region(midpoint, child) for child in children):
                continue
            local[first].add(second)
            local[second].add(first)
        return local


def _rail(nodes: frozenset[int], edges: Mapping[int, set[int]], forbidden: set[int], points: Sequence[Point]) -> frozenset[int] | None:
    """Retain the contact-connecting common rail and visible straight overhangs."""
    if not nodes or any(node not in edges for node in nodes):
        return None
    start = min(nodes)
    previous: dict[int, int | None] = {start: None}
    pending = deque((start,))
    while pending:
        current = pending.popleft()
        for peer in sorted(edges[current]):
            if peer not in previous and peer not in forbidden:
                previous[peer] = current
                pending.append(peer)
    if not nodes <= previous.keys():
        return None
    result = set(nodes)
    for target in nodes:
        current = previous[target]
        while current is not None and current not in result:
            result.add(current)
            current = previous[current]
    # The visible common rail can continue past its last branch ingress.
    # Recover only straight free-end overhangs, never a bent route, another
    # physical contact, or a branching entrance to the requested wiring.
    for vertex in tuple(result):
        inward = {_direction(points[vertex], points[peer]) for peer in edges[vertex] if peer in result}
        for peer in edges[vertex] - result:
            direction = _direction(points[vertex], points[peer])
            if _OPPOSITE[direction] not in inward:
                continue
            extension = set()
            previous_vertex, current = vertex, peer
            while current not in forbidden and current not in result:
                extension.add(current)
                onward = edges[current] - {previous_vertex}
                if not onward:
                    result.update(extension)
                    break
                if len(onward) != 1:
                    break
                following = next(iter(onward))
                if following in extension or _direction(points[current], points[following]) != direction:
                    break
                previous_vertex, current = current, following
    return frozenset(result)


def _recipe_component(binding: _Bindings, recipe: WiringRecipe, component: int) -> tuple[dict[int, set[int]], dict[Key, frozenset[int]]] | None:
    """Select a connected owner-local ink component by its actual contacts.

    A global net may continue through a child's interior and return at another
    public pin. That continuity belongs to electrical A, not to the parent's
    local wiring recipe. No expected position or source group tag selects ink.
    """
    local = binding.local_graph(recipe.group.scope, component)
    try:
        contacts = {attachment.key: binding.attachment(attachment, component) for attachment in recipe.group.attachments}
    except SCNSimValidationError:
        # A candidate contact component can be the wrong member of an
        # indistinguishable embedding; no placement/source endpoint order
        # selects it. The complete graph still must match another candidate.
        return None
    all_contacts = set().union(*contacts.values())
    if not all_contacts or not all_contacts <= local.keys():
        return None
    selected = set()
    pending = [min(all_contacts)]
    while pending:
        vertex = pending.pop()
        if vertex not in selected:
            selected.add(vertex)
            pending.extend(local[vertex] - selected)
    if not all_contacts <= selected:
        return None
    return {vertex: local[vertex] for vertex in selected}, contacts


def _matches_recipe(binding: _Bindings, recipe: WiringRecipe, component: int) -> bool:
    graph = binding.graph
    selected = _recipe_component(binding, recipe, component)
    if selected is None:
        return False
    local, contacts = selected
    all_contacts = set().union(*contacts.values())
    physical_contacts = {
        graph.index(point) for body in binding.bodies
        for _, point in (*body.pin_points, *body.auxiliary_pin_points)
    } | set(binding.boundaries.values()) | {
        graph.index(port.external_anchor) for port in binding.scene.ports
    }
    contracted: dict[int, int] = {}
    attachment_nodes = {}
    for ordinal, (key, terminals) in enumerate(contacts.items(), 1):
        rail = _rail(terminals, local, (all_contacts | physical_contacts) - terminals, graph.points)
        if rail is None or any(vertex in contracted for vertex in rail):
            return False
        node = -ordinal
        attachment_nodes[key] = node
        contracted.update((vertex, node) for vertex in rail)
    reduced: dict[int, set[int]] = defaultdict(set)
    directions: dict[tuple[int, int], str] = {}
    for first in local:
        for second in local[first]:
            if first >= second:
                continue
            left, right = contracted.get(first, first), contracted.get(second, second)
            if left == right:
                continue
            if right in reduced[left]:
                # Two distinct external entrances are not one attachment arm.
                return False
            reduced[left].add(right)
            reduced[right].add(left)
            directions[left, right] = _direction(graph.points[first], graph.points[second])
            directions[right, left] = _OPPOSITE[directions[left, right]]
    if any(len(reduced[node]) != 1 for node in attachment_nodes.values()):
        return False
    dots = {graph.index(mark.point) for mark in binding.scene.node_marks}
    choices: dict[str, tuple[int, ...]] = {}
    for junction in recipe.junctions:
        choices[junction.id] = tuple(
            node for node, peers in reduced.items()
            if node >= 0 and node not in graph.jump_ends
            and len(peers) == len(junction.sides)
            and {directions[node, peer] for peer in peers} == {binding.world_side(side, recipe.group.scope) for side in junction.sides}
            and (len(peers) < 3 or node in dots)
        )
        if not choices[junction.id]:
            return False
    assigned: dict[str, int] = {}

    def endpoint(key: Key) -> int | None:
        return attachment_nodes[key[1]] if key[0] == "attachment" else assigned.get(key[1])

    def trace(a: Key, b: Key) -> set[frozenset[int]] | None:
        start, end = endpoint(a), endpoint(b)
        if start is None or end is None or start == end:
            return None
        neighbors = [peer for peer in reduced[start] if a[0] == "attachment" or directions[start, peer] == binding.world_side(a[2], recipe.group.scope)]
        if len(neighbors) != 1:
            return None
        previous, current = start, neighbors[0]
        used: set[frozenset[int]] = set()
        special = set(assigned.values()) | set(attachment_nodes.values())
        while True:
            edge = frozenset((previous, current))
            if edge in used:
                return None
            used.add(edge)
            if current == end:
                if b[0] == "arm" and directions[current, previous] != binding.world_side(b[2], recipe.group.scope):
                    return None
                return used
            if current in special or len(reduced[current]) != 2:
                return None
            following = next(peer for peer in reduced[current] if peer != previous)
            previous, current = current, following

    def consistent(*, complete: bool) -> bool:
        used: set[frozenset[int]] = set()
        for connection in recipe.connections:
            if endpoint(connection.a) is None or endpoint(connection.b) is None:
                continue
            path = trace(connection.a, connection.b)
            if path is None or path & used:
                return False
            used.update(path)
        return not complete or used == {frozenset((node, peer)) for node, peers in reduced.items() for peer in peers}

    def search() -> bool:
        if len(assigned) == len(choices):
            return consistent(complete=True)
        def priority(identifier: str) -> tuple[int, int, str]:
            anchored = sum(
                (connection.a[0] == "arm" and connection.a[1] == identifier and endpoint(connection.b) is not None)
                or (connection.b[0] == "arm" and connection.b[1] == identifier and endpoint(connection.a) is not None)
                for connection in recipe.connections
            )
            return -anchored, len(choices[identifier]), identifier
        identifier = min((key for key in choices if key not in assigned), key=priority)
        for node in choices[identifier]:
            if node in assigned.values():
                continue
            assigned[identifier] = node
            if consistent(complete=False) and search():
                return True
            del assigned[identifier]
        return False

    return search()


def _center(binding: _Bindings, vertices: frozenset[int]) -> Point:
    return Point(
        sum(binding.graph.points[vertex].x for vertex in vertices) / len(vertices),
        sum(binding.graph.points[vertex].y for vertex in vertices) / len(vertices),
    )


def _visible_bounds(binding: _Bindings, path: tuple[str, ...]) -> Bounds:
    if path in binding.scopes:
        return next(region.source.bounds for region in binding.regions if region.path == path)
    return binding.body_map[path].bounds


def _bounds_center(bounds: Bounds) -> Point:
    return Point((bounds.xmin + bounds.xmax) / 2, (bounds.ymin + bounds.ymax) / 2)


def _project(point: Point, direction: str) -> float:
    return {"right": point.x, "left": -point.x, "top": point.y, "bottom": -point.y}[direction]


def _ordered(points: Sequence[Point], direction: str) -> bool:
    return all(_project(right, direction) - _project(left, direction) > COORDINATE_TOLERANCE for left, right in pairwise(points))


def _effective_axis(binding: _Bindings, key: Key) -> str:
    side = binding.captured.ground_sides.get(key)
    return binding.captured.axes[key] if side is None else "horizontal" if side in {"left", "right"} else "vertical"


def _verify_axes_and_order(binding: _Bindings) -> None:
    """Check the existing local reading grammar only where it is observable.

    A lone peer does not reveal a scope-arrangement axis. A singleton Composite
    member with noncollinear selected contour pins likewise cannot reveal its
    enclosing structure axis; its own visible internal grammar is checked in
    its captured frame instead. Neither case invents ordered ordinary pins.
    """
    captured = binding.captured

    def axis_pair(first: Point, second: Point, direction: str, *, key: Key, allow_noncollinear: bool = False) -> None:
        if _same_point(first, second):
            raise _fail("visible body has no positive terminal-axis span", block=key)
        horizontal = abs(first.y - second.y) <= COORDINATE_TOLERANCE
        vertical = abs(first.x - second.x) <= COORDINATE_TOLERANCE
        if allow_noncollinear and not horizontal and not vertical:
            return
        if (direction in {"left", "right"} and not horizontal) or (direction in {"top", "bottom"} and not vertical):
            raise _fail("visible body or selected contour terminals violate the requested undirected axis", block=key)

    def member_axis(path: tuple[str, ...], direction: str, key: Key, member: Mapping[str, object] | None = None) -> None:
        if path in binding.scopes:
            if member is not None:
                first = binding.graph.points[binding.boundary(path, member["pin_1"])]
                second = binding.graph.points[binding.boundary(path, member["pin_2"])]
                axis_pair(first, second, direction, key=key, allow_noncollinear=True)
            return
        body = binding.body_map[path]
        points = dict(body.pin_points)
        if _symmetric_body(body):
            axis_pair(*points.values(), direction, key=key)
        else:
            for conductor in body.conductors:
                axis_pair(points[f"head.{conductor}"], points[f"tail.{conductor}"], direction, key=key)

    for key in captured.axes:
        if key[0] == "scope":
            continue
        direction = binding.world_side("right" if _effective_axis(binding, key) == "horizontal" else "bottom", key[1])
        if key[0] == "component":
            member_axis((*key[1], key[2]), direction, key)
            continue
        structure = binding.structures[key]
        for branch in structure.get("branches", (structure,)):
            members = branch["elements"]
            centers = []
            for member in members:
                path = tuple(member["path"])
                member_axis(path, direction, key, member)
                centers.append(_bounds_center(_visible_bounds(binding, path)))
            # Axis is undirected. The electrical witness separately binds the
            # ordered source chain, including a legitimately reflected frame.
            if not _ordered(centers, direction) and not _ordered(centers, _OPPOSITE[direction]):
                raise _fail("visible series members do not progress along their requested axis", block=key)

    for key, order in captured.order.items():
        if len(order) < 2:
            continue
        axis = _effective_axis(binding, key)
        if key[0] == "scope":
            scope = key[1]
            direction = binding.world_side("right" if axis == "horizontal" else "bottom", scope)
            centers = [_bounds_center(_visible_bounds(binding, peer[1] if peer[0] == "scope" else (*peer[1], peer[2]))) for peer in order]
        else:
            scope = key[1]
            direction = binding.world_side("bottom" if axis == "horizontal" else "right", scope)
            branches = {row["id"]: row for row in binding.structures[key]["branches"]}
            centers = []
            for peer in order:
                bounds = [_visible_bounds(binding, tuple(member["path"])) for member in branches[peer[2]]["elements"]]
                centers.append(_bounds_center(Bounds(min(item.xmin for item in bounds), min(item.ymin for item in bounds), max(item.xmax for item in bounds), max(item.ymax for item in bounds))))
        if not _ordered(centers, direction):
            raise _fail("visible peers or Parallel branches violate their requested local reading order", block=key)

    for bus, taps in captured.tap_order.items():
        points = []
        for tap in taps:
            aliases = set((group, contact) for group, contact in captured.inventory.endpoint_aliases.get(tap, ()) if group[0] == bus[1])
            if len(aliases) != 1:
                raise _fail("named Tap order requires one unambiguous visible attachment per Tap", tap=tap)
            group, contact = next(iter(aliases))
            recipe = captured.wiring[group]
            observed = set()
            for component in binding.components(recipe.group.net):
                if not _matches_recipe(binding, recipe, component):
                    continue
                contacts = {item.key: binding.attachment(item, component) for item in recipe.group.attachments}
                terminals = contacts[contact]
                if len(terminals) == 1:
                    observed.update(terminals)
                    continue
                selected = _recipe_component(binding, recipe, component)
                if selected is None:
                    continue
                local, _ = selected
                rail = _rail(terminals, local, set().union(*contacts.values()) - terminals, binding.graph.points)
                if rail is not None:
                    observed.update(vertex for vertex in rail if local[vertex] - rail)
            if len(observed) != 1:
                raise _fail("named Tap order has no unique observed contact or common-rail entrance", tap=tap)
            points.append(binding.graph.points[next(iter(observed))])
        direction = binding.world_side("right" if captured.axes[("scope", bus[1])] == "horizontal" else "bottom", bus[1])
        if not _ordered(points, direction):
            raise _fail("visible named Tap contacts violate owner-local projected order", bus=bus)


def _verify_sides(binding: _Bindings) -> None:
    captured, graph = binding.captured, binding.graph
    for key, wanted in captured.port_sides.items():
        port = binding.ports[key[1]]
        scope = next(contact.scope for group in captured.inventory.groups.values() for contact in group.attachments if contact.kind == "port" and contact.endpoint["id"] == key[1])
        wanted = binding.world_side(wanted, scope)
        _verify_port_block_geometry(port)
        if _direction(port.circuit_anchor, port.boundary_anchor) != wanted:
            raise _fail("visible Port lies on a different requested side", port=key[1], side=wanted)
        load_side = captured.port_load_sides[key]
        if _direction(port.circuit_anchor, port.load_anchor) != load_side:
            raise _fail("visible Port load lies on a different requested side", port=key[1], side=load_side)
    for key, wanted in captured.terminal_sides.items():
        if key[0] == "tap":
            # An unlabelled Tap's source identity has no newly selected visible
            # frame. Existing geometry checks retain this setting's authority.
            continue
        wanted = binding.world_side(wanted, key[1])
        vertex, path = binding.pin(key)
        at = graph.points[vertex]
        if path in binding.scopes:
            region = next(region for region in binding.regions if region.path == path)
            bounds = region.source.bounds
            sides = {
                side for side, distance in (
                    ("left", at.x - bounds.xmin), ("right", at.x - bounds.xmax),
                    ("top", at.y - bounds.ymax), ("bottom", at.y - bounds.ymin),
                ) if abs(distance) <= COORDINATE_TOLERANCE
            }
        else:
            body = binding.body_map[path]
            box = next(box for box in binding.scene.boxes if {point for _, point in box.anchors} == {point for _, point in body.pin_points})
            sides = set()
            for stroke in box.paths:
                if stroke.role != "wire":
                    continue
                for first, second in pairwise(stroke.points):
                    if _same_point(at, first):
                        sides.add(_direction(second, first))
                    elif _same_point(at, second):
                        sides.add(_direction(first, second))
        if sides != {wanted}:
            raise _fail("visible labelled or sole boundary pin violates its requested side", pin=key, side=wanted)
    ground_guides = [guide for guide in binding.scene.guides if guide.label is None and len(guide.terminals) == 1 and _is_ground_paths(guide.paths, guide.terminals[0])]
    for key, wanted in captured.ground_sides.items():
        target = captured.inventory.ground_targets[key]
        wanted = binding.world_side(wanted, target.scope)
        if target.block is not None:
            grounded = binding.structure(target.block, target.boundary, set(graph.ground_components))
            structure = binding.structures[target.block]
            other_boundary = ("at" if structure["kind"] == "branch" else "start") if target.boundary == "end" else "end"
            opposite = set()
            for branch in structure.get("branches", (structure,)):
                member = branch["elements"][-1 if other_boundary == "end" else 0]
                pin = member["pin_2" if other_boundary == "end" else "pin_1"]
                path = tuple(member["path"])
                if (path, pin) in binding.endpoint_nets:
                    net = binding.endpoint_nets[path, pin]
                else:
                    net = next(row["final_net"] for row in binding.scopes[path]["exposures"]["pins"] if row["id"] == pin)
                opposite.update(binding.member(member, pin, set(binding.components(net))))
            if not opposite or _direction(_center(binding, frozenset(opposite)), _center(binding, grounded)) != wanted:
                raise _fail("grounded structure's visible termination violates its requested side", block=key, side=wanted)
            if not any(graph.root(guide.terminals[0]) in {graph.components[vertex] for vertex in grounded} for guide in ground_guides):
                raise _fail("grounded structural termination has no actual local ground glyph", block=key)
        else:
            vertex, _ = binding.pin(key)
            local = binding.local_graph(target.scope, graph.components[vertex])
            matching = []
            for guide in ground_guides:
                end = graph.index(guide.terminals[0])
                if end == vertex:
                    stems = [point for stroke in guide.paths for a, point in pairwise(stroke.points) if _same_point(a, graph.points[vertex])]
                    if len(stems) == 1 and _direction(graph.points[vertex], stems[0]) == wanted:
                        matching.append(guide)
                    continue
                previous = {vertex: None}
                pending = deque((vertex,))
                while pending and end not in previous:
                    current = pending.popleft()
                    for peer in local.get(current, ()):
                        if peer not in previous:
                            previous[peer] = current
                            pending.append(peer)
                if end in previous:
                    first = end
                    while previous[first] != vertex:
                        first = previous[first]
                    if _direction(graph.points[vertex], graph.points[first]) == wanted:
                        matching.append(guide)
            if len(matching) != 1:
                raise _fail("parent ground has no unique visible path on its requested side", pin=key, side=wanted)


def verify_composition(scene: NeutralScene, captured_composition: CapturedComposition, *, point: object | None = None) -> None:
    """Verify requested wiring primitives/arms and observable explicit sides.

    Applicable axes/peer/branch order and uniquely bound Tap progression use
    their observed local frame. Unobservable singleton arrangement axes and
    hidden/ambiguous Tap identities are not claimed as observed facts.
    """
    if not isinstance(scene, NeutralScene) or not isinstance(captured_composition, CapturedComposition):
        raise TypeError("composition verification requires NeutralScene and CapturedComposition")
    if set(captured_composition.wiring) != set(captured_composition.inventory.groups):
        raise _fail("captured composition must cover every declared owner-local wiring group exactly once")
    if point is not None:
        from .._authoring_snapshot import ResolvedPlanPoint
        from .._canonical import canonical_plan_snapshot, sha256_hex

        if not isinstance(point, ResolvedPlanPoint):
            raise TypeError("composition point must be ResolvedPlanPoint")
        if sha256_hex(canonical_plan_snapshot(point.snapshot)) != captured_composition.inventory.plan_sha256:
            raise _fail("composition constraints belong to a different captured Plan")
    try:
        binding = _Bindings(scene, captured_composition)
        _verify_sides(binding)
        _verify_axes_and_order(binding)
        by_net: dict[str, list[WiringRecipe]] = defaultdict(list)
        ground_recipes = []
        for recipe in captured_composition.wiring.values():
            if recipe.group.net == "ground":
                if not recipe.automatic or not recipe.group.key[1].startswith("ground:"):
                    raise _fail("general composition replacement cannot target a ground network")
                # A parent-grounded exposed return retains its owner-local
                # wiring recipe. Electrical ground equality must not merge
                # separately drawn return components or their requested trees.
                ground_recipes.append(recipe)
                continue
            by_net[recipe.group.net].append(recipe)

        ground_claims: set[tuple[tuple[str, ...], frozenset[int]]] = set()

        def match_ground(index: int) -> bool:
            if index == len(ground_recipes):
                return True
            recipe = ground_recipes[index]
            for component in binding.components("ground"):
                selected = _recipe_component(binding, recipe, component)
                if selected is None or not _matches_recipe(binding, recipe, component):
                    continue
                local, _ = selected
                claim = (recipe.group.scope, frozenset(local))
                if claim in ground_claims:
                    continue
                ground_claims.add(claim)
                if match_ground(index + 1):
                    return True
                ground_claims.remove(claim)
            return False

        if not match_ground(0):
            raise _fail("visible parent-grounded return does not realize exclusive owner-local wiring recipes")
        choices = {net: binding.components(net) for net in by_net}
        if any(not candidates for candidates in choices.values()):
            raise _fail("composition attachment inventory does not match visible physical contacts")
        order = sorted(choices, key=lambda net: (len(choices[net]), net))
        used: set[int] = set()

        def match_local_recipes(recipes: Sequence[WiringRecipe], component: int) -> bool:
            claimed: set[tuple[tuple[str, ...], frozenset[int]]] = set()
            for recipe in recipes:
                selected = _recipe_component(binding, recipe, component)
                if selected is None or not _matches_recipe(binding, recipe, component):
                    return False
                local, _ = selected
                claim = (recipe.group.scope, frozenset(local))
                if claim in claimed:
                    return False
                claimed.add(claim)
            return True

        def match(index: int) -> bool:
            if index == len(order):
                return True
            net = order[index]
            for component in choices[net]:
                if component in used:
                    continue
                if match_local_recipes(by_net[net], component):
                    used.add(component)
                    if match(index + 1):
                        return True
                    used.remove(component)
            return False

        if not match(0):
            raise _fail("visible wiring does not realize the complete requested shapes, cardinal arms, and attachment graph")
    except SCNSimValidationError as error:
        if error.stage == "schematic_layout":
            raise
        raise _fail("visible ink cannot establish requested composition constraints", reason=str(error)) from error


__all__ = ["verify_composition"]
