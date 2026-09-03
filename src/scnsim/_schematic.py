"""Deterministic semantic authoring-schematic layout.

This module is deliberately presentation-only.  It reads an already authored
``CircuitPlan`` but never seals it, changes it, or contributes to its canonical
identity.  The layout is a deterministic heuristic: relational hints constrain
the graph; all remaining placement uses declaration order and stable IDs.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from math import isqrt
import re
from typing import TYPE_CHECKING, Any

import schemdraw
import schemdraw.elements as elm

if TYPE_CHECKING:  # Avoid importing authoring while it imports this renderer.
    from .authoring import CircuitPlan, ComponentInstance, ElectricNodeRef
    from .specs import CircuitDiagramSpec, DiagramSide, SchematicLayout


@dataclass(frozen=True, slots=True)
class _Edge:
    component: Any
    left: Any
    right: Any


@dataclass(frozen=True, slots=True)
class _DiagramMetrics:
    """Single source of drawing geometry for the authoring schematic."""

    unit_length: float = 2.2

    @property
    def native_span(self) -> float:
        return self.unit_length

    @property
    def label_clearance(self) -> float:
        return self.unit_length / 8

    @property
    def terminal_stub(self) -> float:
        return self.unit_length / 4

    @property
    def obstacle_clearance(self) -> float:
        return self.unit_length / 4

    @property
    def routing_lane_pitch(self) -> float:
        return self.unit_length / 2

    @property
    def port_lead(self) -> float:
        return 5 * self.unit_length / 8

    @property
    def port_load_span(self) -> float:
        return self.unit_length

    @property
    def grounded_branch_depth(self) -> float:
        return 2 * self.unit_length

    @property
    def panel_gap(self) -> float:
        return 3 * self.unit_length

    @property
    def element_gap(self) -> float:
        return 3 * self.unit_length / 4

    @property
    def junction_stagger(self) -> float:
        return self.unit_length / 6

    @property
    def jump_gap(self) -> float:
        return self.unit_length / 12

    @property
    def jump_height(self) -> float:
        return 3 * self.unit_length / 8


_METRICS = _DiagramMetrics()
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class _LabelBox:
    """Schemdraw-measured text extent used before final composition."""

    text: str
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class _LayoutBlock:
    """Occupied symbol-plus-label extent used by automatic placement."""

    symbol_width: float
    symbol_height: float
    label: _LabelBox
    label_side: str

    @property
    def width(self) -> float:
        if self.label_side in {"left", "right"}:
            return self.symbol_width + _METRICS.label_clearance + self.label.width
        return max(self.symbol_width, self.label.width)

    @property
    def height(self) -> float:
        if self.label_side in {"top", "bottom"}:
            return self.symbol_height + _METRICS.label_clearance + self.label.height
        return max(self.symbol_height, self.label.height)


@dataclass(frozen=True, slots=True)
class _NetBus:
    """One physical node projected as one bus with several geometric taps.

    ``ElectricNode`` remains the sole electrical identity.  Taps are private
    presentation anchors that let several components share that identity
    without forcing every symbol and label through one drawing coordinate.
    """

    node: Any
    start: tuple[float, float]
    end: tuple[float, float]
    identity_anchor: tuple[float, float]
    taps: tuple[tuple[tuple[str, int], tuple[float, float]], ...]

    def tap(self, key: tuple[str, int]) -> tuple[float, float]:
        for candidate, point in self.taps:
            if candidate == key:
                return point
        _layout_error(
            "automatic schematic requested an unallocated net tap",
            evidence={
                "node_id": getattr(self.node, "id", None),
                "tap_kind": key[0],
            },
        )
        raise AssertionError("unreachable")


@dataclass(frozen=True, slots=True, eq=False)
class _PresentationPin:
    """Unique pin used only when one physical leaf expands into parallel rows."""

    name: str


@dataclass(frozen=True, slots=True, eq=False)
class _PresentationComponent:
    """Leaf electrical element with an injective presentation path."""

    id: str
    display_id: str
    factory: str
    catalog_id: str
    _pins: dict[str, Any]
    _parameters: dict[str, Any]
    _realization: Any
    _ground_groups: tuple[tuple[Any, ...], ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class _PresentationNode:
    id: str
    visibility: str = "internal"
    endpoints: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class _PresentationPlan:
    _components: tuple[Any, ...]
    _nodes: tuple[Any, ...]
    _pin_nodes: dict[Any, Any]
    _ground_groups: tuple[tuple[Any, ...], ...]
    _ports: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _PresentationGroup:
    id: str
    members: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _PresentationLayout:
    resolved_paths: tuple[tuple[list[Any], list[Any]], ...]
    port_sides: Any
    branch_order: Any
    groups: tuple[_PresentationGroup, ...]


@lru_cache(maxsize=512)
def _label_box(text: str) -> _LabelBox:
    """Measure a label with the exact Schemdraw configuration used to render."""

    import schemdraw
    import schemdraw.elements as elm

    drawing = schemdraw.Drawing(show=False)
    drawing.config(unit=_METRICS.unit_length, fontsize=10)
    label = drawing.add(elm.Label().at((0, 0)).label(text, loc="center", ofst=0))
    bounds = label.get_bbox(transform=True, includetext=True)
    return _LabelBox(
        text=text,
        width=max(0.0, bounds.xmax - bounds.xmin),
        height=max(0.0, bounds.ymax - bounds.ymin),
    )


def _component_block(
    component: Any,
    *,
    orientation: str,
    show_values: bool,
) -> _LayoutBlock:
    label = _label_box(_label(component, show_values=show_values))
    if component.factory == "transmission_line":
        width, height = _block_size(component)
        return _LayoutBlock(width, height, label, "inside")
    if orientation == "horizontal":
        return _measured_native_block(
            component,
            orientation="right",
            show_values=show_values,
        )
    if orientation == "vertical":
        return _measured_native_block(
            component,
            orientation="down",
            show_values=show_values,
        )
    width, height = _block_size(component)
    return _LayoutBlock(width, height, label, "inside")


def _draw_label(
    drawing: Any,
    label: _LabelBox,
    at: tuple[float, float],
    *,
    color: str,
    horizontal: str = "center",
    vertical: str = "center",
) -> None:
    import schemdraw.elements as elm

    drawing.add(
        elm.Label()
        .at(at)
        .label(
            label.text,
            loc="center",
            ofst=0,
            halign=horizontal,
            valign=vertical,
            color=color,
        )
    )


def _layout_error(message: str, *, evidence: dict[str, object] | None = None) -> None:
    from .errors import SCNSimValidationError

    raise SCNSimValidationError(message, stage="schematic_layout", evidence=evidence)


def _nested_pin(component: Any, record: Any) -> Any:
    """Resolve one Composite ground-group record to its live leaf PinRef."""

    current = component
    for segment in record["component_path"]:
        matches = [
            child
            for child in current._realization["children"]
            if child.id == segment
        ]
        if len(matches) != 1:
            _layout_error("Composite presentation path is not uniquely resolvable")
        current = matches[0]
    return current._pins[record["pin_id"]]


def _presentation_plan(plan: Any, layout: Any | None) -> tuple[Any, Any | None]:
    """Recursively lower Composite snapshots to native electrical symbols.

    This presentation-only lowering reuses the already built immutable child
    graph.  It never invokes a factory, compiler, or solver.  Transmission
    lines remain the one dedicated CPW/MTL symbol family; every other leaf is
    a native Schemdraw R/L/C/Josephson element.
    """

    components: list[Any] = []
    nodes: list[Any] = list(plan._nodes)
    pin_nodes: dict[Any, Any] = {}
    leaf_by_top: dict[Any, list[Any]] = {
        component: [] for component in plan._components
    }
    presented_by_source_pin: dict[Any, list[Any]] = defaultdict(list)
    owner_by_pin: dict[Any, Any] = {}
    recorded_ground_groups: list[tuple[Any, ...]] = [
        tuple(group) for group in plan._ground_groups
    ]

    def display_path(path: tuple[str, ...]) -> str:
        if len(path) == 1:
            return path[0]
        return f"{path[0]}.{path[-1]}"

    def add_leaf(
        source: Any,
        path: tuple[str, ...],
        targets: dict[Any, Any],
        top: Any,
    ) -> None:
        identifier = "/".join(path)
        if source.factory == "josephson_junction":
            junction = _PresentationComponent(
                id=identifier,
                display_id=display_path(path),
                factory="josephson_junction",
                catalog_id=source.catalog_id,
                _pins=dict(source._pins),
                _parameters={
                    key: value
                    for key, value in source._parameters.items()
                    if key == "josephson_inductance"
                },
                _realization=source._realization,
            )
            components.append(junction)
            leaf_by_top[top].append(junction)
            for pin in source._pins.values():
                pin_nodes[pin] = targets[pin]
                presented_by_source_pin[pin].append(pin)
                owner_by_pin[pin] = junction

            capacitance = source._parameters.get("junction_capacitance")
            if (
                capacitance is not None
                and float(capacitance.baseline.to("farad").magnitude) != 0.0
            ):
                cj_pins = {
                    name: _PresentationPin(f"{identifier}.Cj.{name}")
                    for name in source._pins
                }
                parallel_capacitance = _PresentationComponent(
                    id=f"{identifier}/Cj",
                    display_id=f"{display_path(path)}.Cj",
                    factory="capacitor",
                    catalog_id=source.catalog_id,
                    _pins=cj_pins,
                    _parameters={"capacitance": capacitance},
                    _realization={"kind": "primitive"},
                )
                components.append(parallel_capacitance)
                leaf_by_top[top].append(parallel_capacitance)
                for name, source_pin in source._pins.items():
                    pin = cj_pins[name]
                    pin_nodes[pin] = targets[source_pin]
                    presented_by_source_pin[source_pin].append(pin)
                    owner_by_pin[pin] = parallel_capacitance
            return

        if source.factory not in {
            "resistor",
            "capacitor",
            "inductor",
            "transmission_line",
        }:
            _layout_error(
                "authoring schematic has no native leaf symbol",
                evidence={"component_path": list(path), "factory": source.factory},
            )
        leaf = _PresentationComponent(
            id=identifier,
            display_id=display_path(path),
            factory=source.factory,
            catalog_id=source.catalog_id,
            _pins=dict(source._pins),
            _parameters=dict(source._parameters),
            _realization=source._realization,
        )
        components.append(leaf)
        leaf_by_top[top].append(leaf)
        for pin in source._pins.values():
            pin_nodes[pin] = targets[pin]
            presented_by_source_pin[pin].append(pin)
            owner_by_pin[pin] = leaf

    def expand(
        source: Any,
        path: tuple[str, ...],
        targets: dict[Any, Any],
        top: Any,
    ) -> None:
        if source._realization.get("kind") != "composite":
            add_leaf(source, path, targets, top)
            return

        # A user Library owns its abstraction boundary.  Its hidden children
        # are not authoring-presentation data, even though the immutable
        # snapshot is available to the compiler.  Built-ins are the only
        # composites whose public contract defines a native-symbol expansion.
        if source.catalog_id != "scnsim.components":
            opaque = _PresentationComponent(
                id="/".join(path),
                display_id=display_path(path),
                factory=source.factory,
                catalog_id=source.catalog_id,
                _pins=dict(source._pins),
                _parameters=dict(source._parameters),
                _realization=source._realization,
                _ground_groups=tuple(source._ground_groups),
            )
            components.append(opaque)
            leaf_by_top[top].append(opaque)
            for pin in source._pins.values():
                pin_nodes[pin] = targets[pin]
                presented_by_source_pin[pin].append(pin)
                owner_by_pin[pin] = opaque
            return

        realization = source._realization
        public_targets = {
            record["private_node_id"]: targets[source._pins[record["public_id"]]]
            for record in realization["public_pin_map"]
        }
        internal_targets: dict[Any, Any] = {}
        for node in realization["nodes"]:
            target = public_targets.get(node.id)
            if target is None:
                target = _PresentationNode(
                    id=f"{'/'.join(path)}/node/{node.id}",
                )
                nodes.append(target)
            for pin in node.endpoints:
                internal_targets[pin] = target
        for pin in realization["grounded"]:
            internal_targets[pin] = "ground"

        for group in source._ground_groups:
            recorded_ground_groups.append(
                tuple(_nested_pin(source, record) for record in group)
            )
        for child in realization["children"]:
            child_targets = {
                pin: internal_targets[pin]
                for pin in child._pins.values()
            }
            expand(child, (*path, child.id), child_targets, top)

    for component in plan._components:
        targets = {
            pin: plan._pin_nodes[pin]
            for pin in component._pins.values()
        }
        expand(component, (component.id,), targets, component)

    ground_groups: list[tuple[Any, ...]] = []
    grouped_pins: set[Any] = set()
    for group in recorded_ground_groups:
        presented = tuple(
            pin
            for source_pin in group
            for pin in presented_by_source_pin.get(source_pin, ())
        )
        if presented:
            # A ground() call is one electrical reference declaration, not an
            # instruction to draw a long shared rail.  Parallel branches from
            # the same signal node share a compact local bus; shunts belonging
            # to different signal nodes receive independent ground glyphs.
            partitions: dict[object, list[Any]] = {}
            for pin in presented:
                owner = owner_by_pin[pin]
                opposite = tuple(
                    target
                    for other in owner._pins.values()
                    if other is not pin
                    for target in (pin_nodes.get(other),)
                    if target is not None and target != "ground"
                )
                key: object = (
                    ("node", id(opposite[0]))
                    if len(opposite) == 1
                    else ("branch", id(owner))
                )
                partitions.setdefault(key, []).append(pin)
            for partition in partitions.values():
                ground_groups.append(tuple(partition))
                grouped_pins.update(partition)
    for pin, target in pin_nodes.items():
        if target == "ground" and pin not in grouped_pins:
            ground_groups.append((pin,))

    presented_plan = _PresentationPlan(
        _components=tuple(components),
        _nodes=tuple(nodes),
        _pin_nodes=pin_nodes,
        _ground_groups=tuple(ground_groups),
        _ports=tuple(plan._ports),
    )
    if layout is None:
        return presented_plan, None

    edges = _edges(presented_plan)

    def leaf_path(
        start: Any,
        end: Any,
        allowed: tuple[Any, ...],
    ) -> tuple[list[Any], list[Any]]:
        allowed_set = set(allowed)
        adjacency: dict[int, list[tuple[Any, Any]]] = defaultdict(list)
        for edge in edges:
            if edge.component not in allowed_set:
                continue
            adjacency[id(edge.left)].append((edge.right, edge.component))
            adjacency[id(edge.right)].append((edge.left, edge.component))
        pending = deque([start])
        previous: dict[int, tuple[Any, Any] | None] = {id(start): None}
        while pending:
            current = pending.popleft()
            if current is end:
                break
            rows = sorted(
                adjacency.get(id(current), ()),
                key=lambda row: (_component_key(row[1]), _node_key(row[0])),
            )
            for other, component in rows:
                if id(other) in previous:
                    continue
                previous[id(other)] = (current, component)
                pending.append(other)
        if id(end) not in previous:
            _layout_error(
                "Composite path has no native-element route between its public terminals"
            )
        path_nodes = [end]
        path_components: list[Any] = []
        while path_nodes[-1] is not start:
            earlier, component = previous[id(path_nodes[-1])]  # type: ignore[misc]
            path_components.append(component)
            path_nodes.append(earlier)
        path_nodes.reverse()
        path_components.reverse()
        return path_nodes, path_components

    resolved_paths: list[tuple[list[Any], list[Any]]] = []
    for hinted in layout.primary_paths:
        outer_nodes = [_node_of(value, plan) for value in hinted.waypoints[::2]]
        outer_components = list(hinted.waypoints[1::2])
        path_nodes = [outer_nodes[0]]
        path_components: list[Any] = []
        for index, component in enumerate(outer_components):
            inner_nodes, inner_components = leaf_path(
                outer_nodes[index],
                outer_nodes[index + 1],
                tuple(leaf_by_top[component]),
            )
            path_nodes.extend(inner_nodes[1:])
            path_components.extend(inner_components)
        resolved_paths.append((path_nodes, path_components))

    branch_order: dict[Any, tuple[Any, ...]] = {}
    for node_ref, ordered in layout.branch_order.items():
        target = node_ref._node
        values: list[Any] = []
        for component in ordered:
            values.extend(
                leaf
                for leaf in leaf_by_top[component]
                if any(
                    presented_plan._pin_nodes.get(pin) is target
                    for pin in leaf._pins.values()
                )
            )
        branch_order[node_ref] = tuple(values)
    groups = tuple(
        _PresentationGroup(
            id=group.id,
            members=tuple(
                leaf
                for component in group.members
                for leaf in leaf_by_top[component]
            ),
        )
        for group in layout.groups
    )
    return presented_plan, _PresentationLayout(
        resolved_paths=tuple(resolved_paths),
        port_sides=layout.port_sides,
        branch_order=branch_order,
        groups=groups,
    )


def _node_of(value: object, plan: Any) -> Any:
    from .authoring import ElectricNodeRef, PortRef

    if isinstance(value, PortRef):
        if value._plan is not plan:
            _layout_error("layout handle belongs to a different Plan")
        return value.node._node
    if isinstance(value, ElectricNodeRef):
        if value._plan is not plan:
            _layout_error("layout handle belongs to a different Plan")
        return value._node
    _layout_error("schematic path endpoints must be PortRef or ElectricNodeRef")
    raise AssertionError("unreachable")


def _validate_layout(plan: Any, layout: Any | None) -> None:
    """Validate all relational hints against the authored Plan, fail closed."""

    if layout is None:
        return
    from .authoring import ComponentInstance, ElectricNodeRef, PortRef
    from .specs import DiagramSide

    components = set(plan._components)
    nodes = {id(node) for node in plan._nodes}
    ports = set(plan._ports)
    used_nodes: set[int] = set()
    path_components: set[ComponentInstance] = set()
    path_positions: dict[ComponentInstance, tuple[int, int]] = {}
    for path_index, path in enumerate(layout.primary_paths):
        values = path.waypoints
        for index, value in enumerate(values):
            is_terminal = index % 2 == 0
            if is_terminal:
                node = _node_of(value, plan)
                if id(node) not in nodes:
                    _layout_error("schematic path node is not declared by this Plan")
                if id(node) in used_nodes:
                    _layout_error(
                        "schematic path nodes cannot repeat",
                        evidence={"path_index": path_index, "node_id": node.id},
                    )
                used_nodes.add(id(node))
            elif not isinstance(value, ComponentInstance) or value not in components:
                _layout_error("schematic path component is not declared by this Plan")
        for index in range(1, len(values), 2):
            component = values[index]
            left = _node_of(values[index - 1], plan)
            right = _node_of(values[index + 1], plan)
            pins_left = [pin for pin in component._pins.values() if plan._pin_nodes.get(pin) is left]
            pins_right = [pin for pin in component._pins.values() if plan._pin_nodes.get(pin) is right]
            if not pins_left or not pins_right or (left is right and len(pins_left) < 2):
                _layout_error(
                    "schematic path requires directly adjacent component terminals",
                    evidence={"component_id": component.id},
                )
            # A multi-terminal component has no implicit pin-pair selection.
            if len(component._pins) > 2 and (len(pins_left) != 1 or len(pins_right) != 1):
                _layout_error(
                    "an ambiguous multi-terminal schematic path requires explicit unique nodes",
                    evidence={"component_id": component.id},
                )
            if component in path_components:
                _layout_error("one component cannot occur in more than one primary path")
            path_components.add(component)
            path_positions[component] = (path_index, index // 2)
    for port, side in layout.port_sides.items():
        if not isinstance(port, PortRef) or port not in ports:
            _layout_error("port_sides contains a foreign PortRef")
        if not isinstance(side, DiagramSide):
            _layout_error("port_sides values must be DiagramSide")
    ordered_components: set[ComponentInstance] = set()
    ordered_nodes: set[int] = set()
    for node_ref, members in layout.branch_order.items():
        if not isinstance(node_ref, ElectricNodeRef) or node_ref._plan is not plan:
            _layout_error("branch_order contains a foreign ElectricNodeRef")
        if id(node_ref._node) in ordered_nodes:
            _layout_error("branch_order cannot repeat one physical node")
        ordered_nodes.add(id(node_ref._node))
        if not isinstance(members, tuple) or len(set(members)) != len(members):
            _layout_error("branch_order members must be an unrepeated tuple")
        node = node_ref._node
        for component in members:
            if not isinstance(component, ComponentInstance) or component not in components:
                _layout_error("branch_order contains a foreign component")
            if not any(plan._pin_nodes.get(pin) is node for pin in component._pins.values()):
                _layout_error("branch_order component is not incident to its node")
            if component in ordered_components:
                _layout_error("a component may occur in only one branch_order hint")
            ordered_components.add(component)
    grouped: set[ComponentInstance] = set()
    for group in layout.groups:
        for component in group.members:
            if component not in components:
                _layout_error("SchematicGroup contains a foreign component")
            if component in grouped:
                _layout_error("SchematicGroup values cannot overlap or nest")
            grouped.add(component)
        on_path = [member for member in group.members if member in path_positions]
        if on_path and len(on_path) != len(group.members):
            _layout_error(
                "SchematicGroup cannot mix primary-path and branch components",
                evidence={"group_id": group.id},
            )
        if on_path:
            positions = [path_positions[member] for member in group.members]
            path_indexes = {path_index for path_index, _index in positions}
            member_indexes = [index for _path_index, index in positions]
            if (
                len(path_indexes) != 1
                or member_indexes != sorted(member_indexes)
                or member_indexes
                != list(range(member_indexes[0], member_indexes[0] + len(member_indexes)))
            ):
                _layout_error(
                    "primary-path SchematicGroup members must be contiguous and ordered",
                    evidence={"group_id": group.id},
                )


def _external_nodes(plan: Any, component: Any) -> tuple[Any, ...]:
    result: list[Any] = []
    for pin in component._pins.values():
        target = plan._pin_nodes.get(pin)
        if target != "ground" and all(target is not existing for existing in result):
            result.append(target)
    return tuple(result)


def _edges(plan: Any) -> tuple[_Edge, ...]:
    """Return only unambiguous two-node graph connections for backbone layout."""

    rows: list[_Edge] = []
    for component in plan._components:
        terminals = _external_nodes(plan, component)
        if (
            len(component._pins) == 2
            and len(terminals) == 2
            and terminals[0] is not terminals[1]
        ):
            rows.append(_Edge(component, terminals[0], terminals[1]))
    return tuple(rows)


def _node_key(node: Any) -> str:
    """Return the Plan-owned stable ID used for deterministic tie breaking."""

    return str(getattr(node, "id", ""))


def _component_key(component: Any) -> str:
    return component.id


def _paths_from_hints(
    plan: Any,
    layout: Any | None,
) -> tuple[tuple[list[Any], list[Any]], ...]:
    if isinstance(layout, _PresentationLayout):
        return layout.resolved_paths
    if layout is None or not layout.primary_paths:
        return ()
    return tuple(
        (
            [_node_of(value, plan) for value in path.waypoints[::2]],
            list(path.waypoints[1::2]),
        )
        for path in layout.primary_paths
    )


def _automatic_path(plan: Any, edges: tuple[_Edge, ...]) -> tuple[list[Any], list[Any]]:
    """Choose the declared two-Port route, else a stable graph diameter."""

    if not edges:
        return [], []
    adjacent: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    node_by_id: dict[int, Any] = {}
    for edge in edges:
        left, right = id(edge.left), id(edge.right)
        node_by_id[left], node_by_id[right] = edge.left, edge.right
        adjacent[left].append((right, edge.component))
        adjacent[right].append((left, edge.component))
    for values in adjacent.values():
        values.sort(key=lambda item: (_component_key(item[1]), _node_key(node_by_id[item[0]])))
    roots = [id(port.node._node) for port in plan._ports if id(port.node._node) in adjacent]
    root = roots[0] if roots else min(adjacent, key=lambda node: _node_key(node_by_id[node]))

    def route(start: int, end: int) -> tuple[list[Any], list[Any]] | None:
        pending: deque[int] = deque([start])
        previous: dict[int, tuple[int, Any] | None] = {start: None}
        while pending:
            current = pending.popleft()
            if current == end:
                break
            for other, component in adjacent[current]:
                if other in previous:
                    continue
                previous[other] = (current, component)
                pending.append(other)
        if end not in previous:
            return None
        node_ids = [end]
        components: list[Any] = []
        while node_ids[-1] != start:
            earlier, component = previous[node_ids[-1]]  # type: ignore[misc]
            components.append(component)
            node_ids.append(earlier)
        node_ids.reverse()
        components.reverse()
        return [node_by_id[node] for node in node_ids], components

    # Declaration order carries the author's boundary intent.  When the first
    # two logical Ports are connected, their shortest component route is the
    # main bus; later probe Ports remain peripheral branches instead of pulling
    # the backbone through the measured subsystem.
    if len(roots) >= 2 and roots[0] != roots[1]:
        declared_route = route(roots[0], roots[1])
        if declared_route is not None:
            return declared_route

    def traverse(start: int) -> tuple[int, dict[int, tuple[int, Any] | None]]:
        queue: deque[int] = deque([start])
        previous: dict[int, tuple[int, Any] | None] = {start: None}
        distance: dict[int, int] = {start: 0}
        while queue:
            current = queue.popleft()
            for other, component in adjacent[current]:
                if other in previous:
                    continue
                previous[other] = (current, component)
                distance[other] = distance[current] + 1
                queue.append(other)
        farthest = max(
            previous,
            key=lambda node: (distance[node], _node_key(node_by_id[node])),
        )
        return farthest, previous

    first_end, _root_tree = traverse(root)
    second_end, previous = traverse(first_end)
    node_ids: list[int] = [second_end]
    components: list[Any] = []
    while previous[node_ids[-1]] is not None:
        earlier, component = previous[node_ids[-1]]  # type: ignore[misc]
        components.append(component)
        node_ids.append(earlier)
    node_ids.reverse()
    components.reverse()
    if node_ids[-1] == root and node_ids[0] != root:
        node_ids.reverse()
        components.reverse()
    elif root not in {node_ids[0], node_ids[-1]} and _node_key(
        node_by_id[node_ids[-1]]
    ) < _node_key(node_by_id[node_ids[0]]):
        node_ids.reverse()
        components.reverse()
    return [node_by_id[node] for node in node_ids], components


def _label(component: Any, *, show_values: bool) -> str:
    display_id = getattr(component, "display_id", component.id)
    prefix = (
        "CPW  "
        if component.factory == "transmission_line" and len(component._pins) == 2
        else "MTL  "
        if component.factory == "transmission_line"
        else ""
    )
    rendered_id = _wrap_identifier(f"{prefix}{display_id}")
    if show_values and component._parameters:
        values = [
            f"{parameter.id} = {parameter.baseline:~P}"
            for parameter in component._parameters.values()
        ]
        if len(values) == 1:
            values[0] = f"{next(iter(component._parameters.values())).baseline:~P}"
        return "\n".join((rendered_id, *values))
    return rendered_id


@lru_cache(maxsize=512)
def _wrap_identifier(value: str) -> str:
    """Insert measured line breaks without changing identifier spelling."""

    if _label_box(value).width <= 2 * _METRICS.native_span:
        return value
    tokens = tuple(
        token for token in re.split(r"(?<=[._/])", value) if token
    )
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = f"{current}{token}"
        if current and _label_box(candidate).width > 2 * _METRICS.native_span:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _element(component: Any, *, color: str) -> Any:
    element_class = {
        "resistor": elm.Resistor,
        "capacitor": elm.Capacitor,
        "inductor": elm.Inductor,
        "josephson_junction": elm.Josephson,
    }.get(component.factory)
    if element_class is None:
        _layout_error(
            "authoring schematic has no native Schemdraw element",
            evidence={"component_id": component.id, "factory": component.factory},
        )
    return element_class(color=color)


def _native_element(
    component: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    show_values: bool,
) -> Any:
    """Build one native symbol whose own label participates in its bbox."""

    horizontal = abs(end[0] - start[0]) >= abs(end[1] - start[1])
    label_side = "top" if horizontal else "bottom"
    return (
        _element(component, color=color)
        .at(start)
        .to(end)
        .label(
            _label(component, show_values=show_values),
            loc=label_side,
            ofst=_METRICS.label_clearance,
            rotate=False,
            color=color,
        )
    )


@lru_cache(maxsize=512)
def _native_extent(
    factory: str,
    label: str,
    orientation: str,
) -> tuple[float, float, float, float]:
    """Return the placed full bbox for one native symbol at the origin."""

    element_class = {
        "resistor": elm.Resistor,
        "capacitor": elm.Capacitor,
        "inductor": elm.Inductor,
        "josephson_junction": elm.Josephson,
    }[factory]
    end = (
        (_METRICS.native_span, 0.0)
        if orientation == "right"
        else (-_METRICS.native_span, 0.0)
        if orientation == "left"
        else (0.0, -_METRICS.native_span)
        if orientation == "down"
        else (0.0, _METRICS.native_span)
    )
    horizontal = orientation in {"right", "left"}
    drawing = schemdraw.Drawing(show=False)
    drawing.config(unit=_METRICS.unit_length, lw=1.8, fontsize=10)
    placed = drawing.add(
        element_class()
        .at((0.0, 0.0))
        .to(end)
        .label(
            label,
            loc="top" if horizontal else "bottom",
            ofst=_METRICS.label_clearance,
            rotate=False,
        )
    )
    return tuple(placed.get_bbox(transform=True, includetext=True))


def _measured_native_block(
    component: Any,
    *,
    orientation: str,
    show_values: bool,
) -> _LayoutBlock:
    label_text = _label(component, show_values=show_values)
    xmin, ymin, xmax, ymax = _native_extent(
        component.factory,
        label_text,
        orientation,
    )
    return _LayoutBlock(
        xmax - xmin,
        ymax - ymin,
        _LabelBox("", 0.0, 0.0),
        "inside",
    )


class _TransmissionLineElement(elm.ElementCompound):
    """The sole custom authoring symbol: an anchor-owned CPW/MTL block."""

    def __init__(
        self,
        *,
        width: float,
        height: float,
        title_text: str,
        conductors: tuple[str, ...],
        color: str,
    ) -> None:
        self._width = width
        self._height = height
        self._title_text = title_text
        self._conductors = conductors
        self._color = color
        super().__init__(
            unit=_METRICS.unit_length,
            fontsize=10,
            lw=1.8,
            color=color,
        )

    def setup(self) -> None:
        if getattr(self, "_scnsim_setup_complete", False):
            return
        self._scnsim_setup_complete = True
        width = self._width
        height = self._height
        stub = _METRICS.terminal_stub
        body_left = stub
        body_right = width - stub
        row_count = max(1, len(self._conductors))
        if row_count == 1:
            rows = (0.0,)
        else:
            usable = height - 2 * _METRICS.terminal_stub
            step = usable / (row_count - 1)
            rows = tuple(usable / 2 - index * step for index in range(row_count))
        anchors: dict[str, tuple[float, float]] = {
            "start": (0.0, rows[0]),
            "end": (width, rows[0]),
        }
        for index, (conductor, y) in enumerate(zip(self._conductors, rows)):
            anchors[f"head.{conductor}"] = (0.0, y)
            anchors[f"tail.{conductor}"] = (width, y)
            self.add(
                elm.Line(color=self._color).endpoints((0.0, y), (body_left, y))
            )
            self.add(
                elm.Line(color=self._color).endpoints((body_right, y), (width, y))
            )
        self.anchors.update(anchors)
        self.add(
            elm.Rect(
                (body_left, -height / 2),
                (body_right, height / 2),
                color=self._color,
            ).at((0.0, 0.0))
        )
        from schemdraw.segments import SegmentText

        self.segments.append(
            SegmentText(
                (width / 2, 0.0),
                self._title_text,
                align=("center", "center"),
                color=self._color,
                fontsize=10,
            )
        )
        self.elmparams["drop"] = (width, rows[0])


def _wire(
    drawing: Any,
    points: list[tuple[float, float]],
    *,
    color: str,
    net: object,
    _detour_depth: int = 0,
) -> None:
    """Draw orthogonal wires, bridging a different-net geometric crossing.

    The renderer records only geometry here: it never treats an SVG crossing
    as an electrical join.  Authoring node dots are emitted separately from
    topology, so an unmarked crossing remains an explicit wire jump.
    """

    import schemdraw.elements as elm

    for start, end in zip(points, points[1:]):
        if start != end:
            existing = getattr(drawing, "_scnsim_wire_segments", ())
            horizontal = start[1] == end[1]
            collinear_conflict = False
            crossings: list[tuple[float, float]] = []
            same_net_crossings: list[
                tuple[
                    tuple[float, float],
                    tuple[float, float],
                    tuple[float, float],
                ]
            ] = []
            for other_start, other_end, other_net in existing:
                other_horizontal = other_start[1] == other_end[1]
                if horizontal == other_horizontal:
                    if horizontal and abs(start[1] - other_start[1]) < _EPSILON:
                        overlap = min(max(start[0], end[0]), max(other_start[0], other_end[0])) - max(min(start[0], end[0]), min(other_start[0], other_end[0]))
                    elif not horizontal and abs(start[0] - other_start[0]) < _EPSILON:
                        overlap = min(max(start[1], end[1]), max(other_start[1], other_end[1])) - max(min(start[1], end[1]), min(other_start[1], other_end[1]))
                    else:
                        overlap = 0.0
                    if overlap > _EPSILON and other_net != net:
                        collinear_conflict = True
                    continue
                h_start, h_end = (start, end) if horizontal else (other_start, other_end)
                v_start, v_end = (other_start, other_end) if horizontal else (start, end)
                x, y = v_start[0], h_start[1]
                if (
                    min(h_start[0], h_end[0]) < x < max(h_start[0], h_end[0])
                    and min(v_start[1], v_end[1]) < y < max(v_start[1], v_end[1])
                ):
                    if other_net == net:
                        same_net_crossings.append(
                            ((x, y), other_start, other_end)
                        )
                    else:
                        crossings.append((x, y))
            if collinear_conflict:
                if _detour_depth >= 12:
                    _layout_error(
                        "orthogonal router could not separate distinct nets",
                        evidence={
                            "component_id": getattr(
                                drawing, "_scnsim_current_component", None
                            ),
                            "start": list(start),
                            "end": list(end),
                        },
                    )
                count = getattr(drawing, "_scnsim_detour_count", 0)
                direction = 1.0 if count % 2 == 0 else -1.0
                drawing._scnsim_detour_count = count + 1
                if horizontal:
                    ordinates = [
                        point[1]
                        for other_start, other_end, _other_net in existing
                        for point in (other_start, other_end)
                    ]
                    lane = (
                        max(ordinates, default=start[1])
                        + _METRICS.routing_lane_pitch * (1 + count // 2)
                        if direction > 0
                        else min(ordinates, default=start[1])
                        - _METRICS.routing_lane_pitch * (1 + count // 2)
                    )
                    detour = [
                        start,
                        (start[0], lane),
                        (end[0], lane),
                        end,
                    ]
                else:
                    abscissas = [
                        point[0]
                        for other_start, other_end, _other_net in existing
                        for point in (other_start, other_end)
                    ]
                    lane = (
                        max(abscissas, default=start[0])
                        + _METRICS.routing_lane_pitch * (1 + count // 2)
                        if direction > 0
                        else min(abscissas, default=start[0])
                        - _METRICS.routing_lane_pitch * (1 + count // 2)
                    )
                    detour = [
                        start,
                        (lane, start[1]),
                        (lane, end[1]),
                        end,
                    ]
                _wire(
                    drawing,
                    detour,
                    color=color,
                    net=net,
                    _detour_depth=_detour_depth + 1,
                )
                continue
            coordinate = (lambda point: point[0]) if horizontal else (lambda point: point[1])
            if same_net_crossings:
                if _detour_depth >= 12:
                    _layout_error("orthogonal router could not stagger a same-net crossing")
                same_net_crossings.sort(
                    key=lambda item: coordinate(item[0]),
                    reverse=coordinate(end) < coordinate(start),
                )
                crossing, other_start, other_end = same_net_crossings[0]
                if horizontal:
                    low = min(other_start[1], other_end[1])
                    high = max(other_start[1], other_end[1])
                    positive_room = high - crossing[1]
                    negative_room = crossing[1] - low
                    sign = 1.0 if positive_room >= negative_room else -1.0
                    room = positive_room if sign > 0 else negative_room
                    offset = sign * min(_METRICS.junction_stagger, room / 2)
                    staggered = [
                        start,
                        crossing,
                        (crossing[0], crossing[1] + offset),
                        (end[0], crossing[1] + offset),
                        end,
                    ]
                else:
                    low = min(other_start[0], other_end[0])
                    high = max(other_start[0], other_end[0])
                    positive_room = high - crossing[0]
                    negative_room = crossing[0] - low
                    sign = 1.0 if positive_room >= negative_room else -1.0
                    room = positive_room if sign > 0 else negative_room
                    offset = sign * min(_METRICS.junction_stagger, room / 2)
                    staggered = [
                        start,
                        crossing,
                        (crossing[0] + offset, crossing[1]),
                        (crossing[0] + offset, end[1]),
                        end,
                    ]
                # A same-net four-way crossing has no distinct identity to
                # mark.  Follow the existing conductor briefly and leave it
                # at a second T so the topology stays legible without a dot.
                _wire(
                    drawing,
                    staggered,
                    color=color,
                    net=net,
                    _detour_depth=_detour_depth + 1,
                )
                continue
            crossings.sort(
                key=coordinate,
                reverse=coordinate(end) < coordinate(start),
            )
            cursor = start
            for crossing in crossings:
                gap = _METRICS.jump_gap
                direction = 1.0 if coordinate(end) > coordinate(start) else -1.0
                before = (
                    (crossing[0] - direction * gap, cursor[1])
                    if horizontal
                    else (cursor[0], crossing[1] - direction * gap)
                )
                after = (
                    (crossing[0] + direction * gap, cursor[1])
                    if horizontal
                    else (cursor[0], crossing[1] + direction * gap)
                )
                drawing.add(elm.Line(color=color).endpoints(cursor, before))
                # Arc2 is a real Schemdraw bridge, not an implicit junction.
                drawing.add(
                    elm.Arc2(k=_METRICS.jump_height, color=color)
                    .at(before)
                    .to(after)
                )
                cursor = after
            drawing.add(elm.Line(color=color).endpoints(cursor, end))
            drawing._scnsim_wire_segments = (*existing, (start, end, net))


def _dot(drawing: Any, at: tuple[float, float], *, color: str) -> None:
    import schemdraw.elements as elm

    drawing.add(elm.Dot(color=color, fill=color).at(at))


def _ground(
    drawing: Any,
    at: tuple[float, float],
    *,
    color: str,
    direction: str = "down",
) -> None:
    symbol = elm.Ground(color=color).at(at)
    if direction == "up":
        symbol = symbol.up()
    drawing.add(symbol)


def _draw_horizontal_component(
    drawing: Any,
    component: Any,
    left: tuple[float, float],
    right: tuple[float, float],
    *,
    color: str,
    show_values: bool,
) -> None:
    length = max(_METRICS.native_span, abs(right[0] - left[0]))
    if component.factory == "transmission_line":
        _width, height = _block_size(component)
        conductors = tuple(component._realization.get("pin_conductors", ()))
        drawing.add(
            _TransmissionLineElement(
                width=length,
                height=height,
                title_text=_label(component, show_values=show_values),
                conductors=conductors or ("signal",),
                color=color,
            ).at(left)
        )
    else:
        drawing.add(
            _native_element(
                component,
                left,
                right,
                color=color,
                show_values=show_values,
            )
        )


def _draw_vertical_component(
    drawing: Any,
    component: Any,
    top: tuple[float, float],
    bottom: tuple[float, float],
    *,
    color: str,
    show_values: bool,
) -> None:
    drawing.add(
        _native_element(
            component,
            top,
            bottom,
            color=color,
            show_values=show_values,
        )
    )


def _block_size(component: Any) -> tuple[float, float]:
    width = _METRICS.native_span
    if component.factory == "transmission_line":
        conductors = tuple(component._realization.get("pin_conductors", ()))
        conductor_count = max(1, len(conductors))
        pins = tuple(component._pins.values())
        left_labels = tuple(_label_box(pin.name) for pin in pins[:conductor_count])
        right_labels = tuple(_label_box(pin.name) for pin in pins[conductor_count:])
        left_width = max((label.width for label in left_labels), default=0.0)
        right_width = max((label.width for label in right_labels), default=0.0)
        title = _label_box(_label(component, show_values=True))
        width = max(
            2 * _METRICS.native_span,
            title.width + 2 * _METRICS.terminal_stub,
            left_width + right_width + _METRICS.native_span,
        )
        height = max(
            _METRICS.native_span,
            title.height
            + 2 * _METRICS.terminal_stub
            + _METRICS.routing_lane_pitch * conductor_count,
        )
        return width, height
    title = _label_box(_label(component, show_values=True))
    return (
        max(width, title.width + 2 * _METRICS.terminal_stub),
        max(
            _METRICS.native_span,
            title.height
            + 2 * _METRICS.terminal_stub
            + _METRICS.routing_lane_pitch * max(2, len(component._pins)),
        ),
    )


def _through_axis(
    node_id: int,
    point: tuple[float, float],
    edges: tuple[_Edge, ...],
    node_pos: dict[int, tuple[float, float]],
) -> str | None:
    """Return the axis only when a node has neighbors on both opposite sides."""

    left = right = above = below = False
    for edge in edges:
        if id(edge.left) == node_id:
            other = node_pos.get(id(edge.right))
        elif id(edge.right) == node_id:
            other = node_pos.get(id(edge.left))
        else:
            continue
        if other is None:
            continue
        if abs(other[1] - point[1]) < _EPSILON:
            left = left or other[0] < point[0]
            right = right or other[0] > point[0]
        if abs(other[0] - point[0]) < _EPSILON:
            below = below or other[1] < point[1]
            above = above or other[1] > point[1]
    if left and right and not (above or below):
        return "horizontal"
    if above and below and not (left or right):
        return "vertical"
    return None


def _draw_port(
    drawing: Any,
    port: Any,
    node_point: tuple[float, float],
    *,
    side: Any,
    color: str,
    background: str,
) -> None:
    """Draw the logical boundary and its actual raw-network reference branch."""

    from .specs import DiagramSide

    node_x, node_y = node_point
    if side in {DiagramSide.LEFT, DiagramSide.RIGHT}:
        direction = -1.0 if side is DiagramSide.LEFT else 1.0
        junction = node_point
        circle = (node_x + direction * _METRICS.port_lead, node_y)
        route = [circle, junction]
    elif side is DiagramSide.TOP:
        junction = (node_x, node_y + _METRICS.port_lead)
        circle = (node_x, junction[1] + _METRICS.port_lead)
        route = [circle, junction, node_point]
    else:
        junction = (node_x, node_y - _METRICS.port_lead)
        circle = (node_x, junction[1] - _METRICS.port_lead)
        route = [circle, junction, node_point]
    _wire(drawing, route, color=color, net=id(port.node._node))
    label = f"{_wrap_identifier(port.id)}\n{port.role}"
    if port.role == "nonloading_probe":
        label += " (PTC-removable)"
    drawing.add(elm.Dot(open=True, color=color, fill=background).at(circle))
    label_offset = {
        DiagramSide.LEFT: (-_METRICS.label_clearance, 0.0, "right", "center"),
        DiagramSide.RIGHT: (_METRICS.label_clearance, 0.0, "left", "center"),
        DiagramSide.TOP: (0.0, _METRICS.label_clearance, "center", "bottom"),
        DiagramSide.BOTTOM: (0.0, -_METRICS.label_clearance, "center", "top"),
    }[side]
    _draw_label(
        drawing,
        _label_box(label),
        (circle[0] + label_offset[0], circle[1] + label_offset[1]),
        color=color,
        horizontal=label_offset[2],
        vertical=label_offset[3],
    )
    # This is not a Library resistor: it is the visual projection of B/R/M.
    # DiagramSide rotates the three-anchor Port block rather than routing one
    # fixed symbol around the page.
    load_component = _PresentationComponent(
        id=f"{port.id}/reference_load",
        display_id=f"R = Z₀ = {port.reference_impedance:~P}",
        factory="resistor",
        catalog_id="scnsim.port",
        _pins={},
        _parameters={},
        _realization={"kind": "port_load"},
    )
    x, y = junction
    if side in {DiagramSide.LEFT, DiagramSide.RIGHT}:
        load_end = (x, y - _METRICS.port_load_span)
        # Keep the load annotation on the exterior side of the Port block so
        # its measured extent cannot intrude into an adjacent circuit block.
        load_label_side = "top" if side is DiagramSide.LEFT else "bottom"
        drawing.add(
            elm.Resistor(color=color)
            .at(junction)
            .to(load_end)
            .label(
                load_component.display_id,
                loc=load_label_side,
                ofst=_METRICS.label_clearance,
                rotate=False,
                color=color,
            )
        )
        _ground(drawing, load_end, color=color)
    else:
        load_end = (x + _METRICS.port_load_span, y)
        load_label_side = "top" if side is DiagramSide.TOP else "bottom"
        drawing.add(
            elm.Resistor(color=color)
            .at(junction)
            .to(load_end)
            .label(
                load_component.display_id,
                loc=load_label_side,
                ofst=_METRICS.label_clearance,
                rotate=False,
                color=color,
            )
        )
        ground_end = (load_end[0], load_end[1] - _METRICS.terminal_stub)
        _wire(
            drawing,
            [load_end, ground_end],
            color=color,
            net=("port_load", port.id),
        )
        _ground(drawing, ground_end, color=color)


def _incident_ground_groups(plan: Any) -> dict[Any, int]:
    groups: dict[Any, int] = {}
    for index, pins in enumerate(plan._ground_groups):
        for pin in pins:
            groups[pin] = index
    return groups


def _node_label_extent(node: Any) -> tuple[float, float]:
    if getattr(node, "visibility", None) != "public":
        return 0.0, 0.0
    box = _label_box(str(node.id))
    return box.width, box.height


def _component_tap_key(component: Any) -> tuple[str, int]:
    return ("component", id(component))


def _port_tap_key(port: Any) -> tuple[str, int]:
    return ("port", id(port))


def _net_attachments(
    plan: Any,
    layout: Any | None,
) -> dict[int, list[tuple[str, int]]]:
    """Return stable, declaration-ordered attachment keys for every net."""

    attachments: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for component in plan._components:
        key = _component_tap_key(component)
        seen: set[int] = set()
        for pin in component._pins.values():
            node = plan._pin_nodes.get(pin)
            if node == "ground" or node is None or id(node) in seen:
                continue
            seen.add(id(node))
            attachments[id(node)].append(key)
    for port in plan._ports:
        node = port.node._node
        keys = attachments[id(node)]
        # A promoted degree-one boundary is one junction: the Port and the
        # sole circuit branch deliberately share that one geometric anchor.
        if getattr(node, "visibility", None) != "port_promoted" or len(keys) != 1:
            keys.append(_port_tap_key(port))
    if layout is not None:
        for node_ref, ordered in layout.branch_order.items():
            node_id = id(node_ref._node)
            ranked = {
                _component_tap_key(component): index
                for index, component in enumerate(ordered)
            }
            original = attachments.get(node_id, [])
            attachments[node_id] = sorted(
                original,
                key=lambda key: (
                    0 if key in ranked else 1,
                    ranked.get(key, original.index(key)),
                ),
            )
    return attachments


def _build_net_buses(
    plan: Any,
    positions: dict[int, tuple[float, float]],
    *,
    show_values: bool,
    layout: Any | None,
) -> dict[int, _NetBus]:
    """Allocate several taps while preserving exactly one physical node."""

    attachments = _net_attachments(plan, layout)
    buses: dict[int, _NetBus] = {}
    for node in sorted(plan._nodes, key=_node_key):
        node_id = id(node)
        center_x, center_y = positions[node_id]
        keys = attachments.get(node_id, [])
        unique_keys = tuple(dict.fromkeys(keys))
        component_by_key = {
            _component_tap_key(component): component
            for component in plan._components
        }

        def occupied_width(key: tuple[str, int]) -> float:
            if key[0] == "port":
                # The Port block grows normal to the bus.  Its tap therefore
                # needs one native slot, not the full horizontal label width.
                return _METRICS.native_span
            component = component_by_key[key]
            targets = _external_nodes(plan, component)
            other = next((value for value in targets if value is not node), None)
            grounded = any(
                plan._pin_nodes.get(pin) == "ground"
                for pin in component._pins.values()
            )
            vertical = grounded
            if other is not None:
                other_x, other_y = positions[id(other)]
                vertical = abs(other_y - center_y) > abs(other_x - center_x)
            if not vertical:
                return _METRICS.routing_lane_pitch
            block = _component_block(
                component,
                orientation="vertical",
                show_values=show_values,
            )
            return max(_METRICS.routing_lane_pitch, block.width)

        if len(unique_keys) <= 1:
            points = ((center_x, center_y),)
        else:
            widths = [occupied_width(key) for key in unique_keys]
            centers = [0.0]
            for left_width, right_width in zip(widths, widths[1:]):
                centers.append(
                    centers[-1]
                    + left_width / 2
                    + _METRICS.element_gap
                    + right_width / 2
                )
            midpoint = (centers[0] + centers[-1]) / 2
            points = tuple(
                (center_x + value - midpoint, center_y) for value in centers
            )
        if getattr(node, "visibility", None) == "public":
            identity_anchor = points[0]
        else:
            identity_anchor = points[len(points) // 2]
        taps = list(zip(unique_keys, points))
        if getattr(node, "visibility", None) == "port_promoted" and len(keys) == 1:
            for port in plan._ports:
                if port.node._node is node:
                    taps.append((_port_tap_key(port), points[0]))
        buses[node_id] = _NetBus(
            node=node,
            start=points[0],
            end=points[-1],
            identity_anchor=identity_anchor,
            taps=tuple(taps),
        )
    return buses


def _draw_net_buses(
    drawing: Any,
    buses: dict[int, _NetBus],
    *,
    color: str,
) -> None:
    for node_id, bus in sorted(
        buses.items(), key=lambda row: _node_key(row[1].node)
    ):
        if bus.start != bus.end:
            _wire(
                drawing,
                [bus.start, bus.end],
                color=color,
                net=node_id,
            )


def _relation_span(component: Any) -> float:
    if component.factory == "transmission_line":
        return _block_size(component)[0]
    if component.factory in {
        "resistor",
        "capacitor",
        "inductor",
        "josephson_junction",
    }:
        return _METRICS.native_span
    return _block_size(component)[0]


def _expand_horizontal_layers(
    plan: Any,
    edges: tuple[_Edge, ...],
    positions: dict[int, tuple[float, float]],
    *,
    show_values: bool,
    layout: Any | None,
) -> dict[int, _NetBus]:
    """Monotonically widen graph layers until bus and element blocks fit."""

    for _iteration in range(max(1, len(plan._nodes) * 2)):
        buses = _build_net_buses(
            plan,
            positions,
            show_values=show_values,
            layout=layout,
        )
        changed = False
        for edge in edges:
            left_id, right_id = id(edge.left), id(edge.right)
            left_center = positions[left_id]
            right_center = positions[right_id]
            if abs(left_center[0] - right_center[0]) <= abs(
                left_center[1] - right_center[1]
            ):
                continue
            if left_center[0] > right_center[0]:
                left_id, right_id = right_id, left_id
                left_center, right_center = right_center, left_center
            required = _relation_span(edge.component) + 2 * _METRICS.element_gap
            available = buses[right_id].start[0] - buses[left_id].end[0]
            if available + _EPSILON >= required:
                continue
            shift = required - available
            threshold = right_center[0] - _EPSILON
            for node_id, (x, y) in tuple(positions.items()):
                if x >= threshold:
                    positions[node_id] = (x + shift, y)
            changed = True
            break
        if not changed:
            return buses
    _layout_error("automatic schematic spacing did not converge")
    raise AssertionError("unreachable")


def _place_primary_rows(
    rows: tuple[tuple[list[Any], list[Any]], ...],
    *,
    show_values: bool,
) -> tuple[dict[int, tuple[float, float]], set[Any]]:
    positions: dict[int, tuple[float, float]] = {}
    path_components: set[Any] = set()
    for row_index, (nodes, components) in enumerate(rows):
        if not nodes:
            continue
        y = -row_index * _METRICS.panel_gap
        x = 0.0
        positions[id(nodes[0])] = (x, y)
        for index, component in enumerate(components):
            block = _component_block(
                component,
                orientation="horizontal",
                show_values=show_values,
            )
            left_width, _left_height = _node_label_extent(nodes[index])
            right_width, _right_height = _node_label_extent(nodes[index + 1])
            occupied = max(
                _METRICS.native_span,
                block.width,
                (left_width + right_width) / 2,
            )
            x += occupied + 2 * _METRICS.element_gap
            positions[id(nodes[index + 1])] = (x, y)
            path_components.add(component)
    return positions, path_components


def _seed_multiconductor_lines(
    plan: Any,
    positions: dict[int, tuple[float, float]],
) -> set[Any]:
    """Place each N-trace line as one ordered multi-row layout block."""

    seeded: set[Any] = set()
    panel_y = min((point[1] for point in positions.values()), default=0.0)
    for component in plan._components:
        if component.factory != "transmission_line" or len(component._pins) <= 2:
            continue
        conductors = tuple(component._realization.get("pin_conductors", ()))
        count = len(conductors)
        pins = tuple(component._pins.values())
        if count == 0 or len(pins) != 2 * count:
            _layout_error(
                "transmission-line pin order does not match its conductor order",
                evidence={"component_id": component.id},
            )
        panel_y -= _METRICS.panel_gap
        width, height = _block_size(component)
        row_pitch = max(
            _METRICS.routing_lane_pitch,
            (height - 2 * _METRICS.terminal_stub) / max(1, count - 1),
        )
        midpoint = (count - 1) / 2
        left_x = 0.0
        right_x = width + 2 * _METRICS.element_gap
        for index in range(count):
            y = panel_y + (midpoint - index) * row_pitch
            head = plan._pin_nodes[pins[index]]
            tail = plan._pin_nodes[pins[count + index]]
            positions.setdefault(id(head), (left_x, y))
            positions.setdefault(id(tail), (right_x, y))
        seeded.add(component)
    return seeded


def _opaque_anchor_specs(
    component: Any,
) -> tuple[tuple[Any, tuple[float, float], tuple[float, float]], ...]:
    """Return clockwise public-pin anchors beginning at lower left."""

    width, height = _block_size(component)
    pins = tuple(component._pins.values())
    left_count = (len(pins) + 1) // 2
    right_count = len(pins) - left_count
    specs: list[tuple[Any, tuple[float, float], tuple[float, float]]] = []
    for index, pin in enumerate(pins[:left_count]):
        fraction = (index + 1) / (left_count + 1)
        specs.append(
            (
                pin,
                (-width / 2, -height / 2 + fraction * height),
                (-1.0, 0.0),
            )
        )
    for index, pin in enumerate(pins[left_count:]):
        fraction = (index + 1) / (right_count + 1)
        specs.append(
            (
                pin,
                (width / 2, height / 2 - fraction * height),
                (1.0, 0.0),
            )
        )
    return tuple(specs)


def _seed_opaque_blocks(
    plan: Any,
    positions: dict[int, tuple[float, float]],
) -> dict[Any, tuple[float, float]]:
    """Reserve one measured block for every N-terminal custom Composite."""

    centers: dict[Any, tuple[float, float]] = {}
    panel_y = min((point[1] for point in positions.values()), default=0.0)
    for component in plan._components:
        targets = _external_nodes(plan, component)
        if component.catalog_id == "scnsim.components" or len(targets) <= 2:
            continue
        specs = _opaque_anchor_specs(component)
        candidates: list[tuple[float, float]] = []
        for pin, anchor, _normal in specs:
            node = plan._pin_nodes.get(pin)
            if node != "ground" and node is not None and id(node) in positions:
                point = positions[id(node)]
                candidates.append((point[0] - anchor[0], point[1] - anchor[1]))
        if candidates:
            center = (
                sum(point[0] for point in candidates) / len(candidates),
                sum(point[1] for point in candidates) / len(candidates),
            )
        else:
            panel_y -= _METRICS.panel_gap
            center = (0.0, panel_y)
        centers[component] = center
        for pin, anchor, normal in specs:
            node = plan._pin_nodes.get(pin)
            if node == "ground" or node is None or id(node) in positions:
                continue
            positions[id(node)] = (
                center[0] + anchor[0] + normal[0] * _METRICS.element_gap,
                center[1] + anchor[1] + normal[1] * _METRICS.element_gap,
            )
    return centers


def _seed_unbounded_cyclic_panels(
    plan: Any,
    edges: tuple[_Edge, ...],
    positions: dict[int, tuple[float, float]],
) -> None:
    """Give port-free cyclic blocks a deterministic two-dimensional panel."""

    if plan._ports:
        return
    node_by_id = {id(node): node for node in plan._nodes}
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        left, right = id(edge.left), id(edge.right)
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(adjacency)
    panel_top = min((y for _x, y in positions.values()), default=0.0)
    while remaining:
        seed = min(remaining, key=lambda value: _node_key(node_by_id[value]))
        pending = [seed]
        connected: set[int] = set()
        while pending:
            current = pending.pop()
            if current in connected:
                continue
            connected.add(current)
            pending.extend(adjacency[current] - connected)
        remaining.difference_update(connected)
        edge_count = sum(len(adjacency[node]) for node in connected) // 2
        if edge_count < len(connected):
            continue
        ordered = sorted(
            connected,
            key=lambda value: (
                0 if value in positions else 1,
                -positions.get(value, (0.0, 0.0))[1],
                positions.get(value, (0.0, 0.0))[0],
                _node_key(node_by_id[value]),
            ),
        )
        columns = max(2, isqrt(len(ordered)))
        if columns * columns < len(ordered):
            columns += 1
        for index, node_id in enumerate(ordered):
            row, column = divmod(index, columns)
            positions[node_id] = (
                column * 2 * _METRICS.panel_gap,
                panel_top - row * 2 * _METRICS.panel_gap,
            )
        panel_top -= (
            ((len(ordered) - 1) // columns + 1) * 2 * _METRICS.panel_gap
        )


def _reserve_node_position(
    candidate: tuple[float, float],
    occupied: dict[int, tuple[float, float]],
) -> tuple[float, float]:
    x, y = candidate
    minimum = _METRICS.native_span + _METRICS.element_gap
    while any(
        abs(x - other_x) < minimum and abs(y - other_y) < minimum
        for other_x, other_y in occupied.values()
    ):
        y -= _METRICS.routing_lane_pitch
    return x, y


def _place_remaining_nodes(
    plan: Any,
    edges: tuple[_Edge, ...],
    positions: dict[int, tuple[float, float]],
    path_components: set[Any],
    *,
    show_values: bool,
    layout: Any | None,
) -> None:
    """Place off-bus connected components without fixture-specific geometry."""

    node_by_id = {id(node): node for node in plan._nodes}
    declaration = {
        component: index for index, component in enumerate(plan._components)
    }
    group_rank = {
        component: (group_index, member_index)
        for group_index, group in enumerate(layout.groups if layout is not None else ())
        for member_index, component in enumerate(group.members)
    }
    adjacency: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for edge in edges:
        if edge.component in path_components:
            continue
        left, right = id(edge.left), id(edge.right)
        adjacency[left].append((right, edge.component))
        adjacency[right].append((left, edge.component))
    for values in adjacency.values():
        values.sort(
            key=lambda row: (
                0 if row[1] in group_rank else 1,
                group_rank.get(row[1], (declaration[row[1]], 0)),
                declaration[row[1]],
                _node_key(node_by_id[row[0]]),
            )
        )

    unseen = set(adjacency)
    panel_y = min((point[1] for point in positions.values()), default=0.0)
    while unseen:
        seed = min(unseen, key=lambda node_id: _node_key(node_by_id[node_id]))
        pending = deque([seed])
        component_nodes: set[int] = set()
        while pending:
            current = pending.popleft()
            if current in component_nodes:
                continue
            component_nodes.add(current)
            pending.extend(other for other, _component in adjacency[current])
        unseen.difference_update(component_nodes)
        anchors = sorted(
            component_nodes.intersection(positions),
            key=lambda node_id: (
                positions[node_id][1],
                positions[node_id][0],
                _node_key(node_by_id[node_id]),
            ),
        )
        if anchors:
            roots = anchors
        else:
            panel_y -= _METRICS.panel_gap
            root = min(
                component_nodes,
                key=lambda node_id: _node_key(node_by_id[node_id]),
            )
            positions[root] = _reserve_node_position((0.0, panel_y), positions)
            roots = [root]

        previous: dict[int, int | None] = {root: None for root in roots}
        depth: dict[int, int] = {root: 0 for root in roots}
        queue = deque(roots)
        while queue:
            current = queue.popleft()
            for other, _component in adjacency[current]:
                if other in previous:
                    continue
                previous[other] = current
                depth[other] = depth[current] + 1
                queue.append(other)

        for level in range(1, max(depth.values(), default=0) + 1):
            parents: dict[int, list[int]] = defaultdict(list)
            for node_id, node_depth in depth.items():
                parent = previous[node_id]
                if node_depth == level and parent is not None and node_id not in positions:
                    parents[parent].append(node_id)
            for parent in sorted(
                parents,
                key=lambda node_id: (
                    positions[node_id][0],
                    positions[node_id][1],
                    _node_key(node_by_id[node_id]),
                ),
            ):
                child_set = set(parents[parent])
                children = [
                    other
                    for other, _component in adjacency[parent]
                    if other in child_set
                ]
                parent_x, parent_y = positions[parent]
                relation_blocks = [
                    _component_block(
                        component,
                        orientation="horizontal",
                        show_values=show_values,
                    )
                    for child in children
                    for other, component in adjacency[parent]
                    if other == child
                ]
                relation_width = max(
                    (block.width for block in relation_blocks),
                    default=_METRICS.native_span,
                )
                horizontal_step = max(
                    2 * _METRICS.panel_gap,
                    relation_width + 4 * _METRICS.element_gap,
                )
                vertical_pitch = max(
                    3 * _METRICS.grounded_branch_depth,
                    max(
                        (block.height for block in relation_blocks),
                        default=_METRICS.native_span,
                    )
                    + 4 * _METRICS.element_gap,
                )
                if depth[parent] == 0 and len(children) == 1:
                    candidates = (
                        (parent_x, parent_y - 2 * _METRICS.panel_gap),
                    )
                else:
                    x = parent_x + horizontal_step
                    midpoint = (len(children) - 1) / 2
                    candidates = tuple(
                        (x, parent_y + (midpoint - index) * vertical_pitch)
                        for index in range(len(children))
                    )
                for child, candidate in zip(children, candidates):
                    positions[child] = _reserve_node_position(candidate, positions)

    for node in sorted(plan._nodes, key=_node_key):
        node_id = id(node)
        if node_id in positions:
            continue
        panel_y -= _METRICS.panel_gap
        positions[node_id] = _reserve_node_position((0.0, panel_y), positions)


def _draw_inline_opaque(
    drawing: Any,
    component: Any,
    left: tuple[float, float],
    right: tuple[float, float],
    *,
    color: str,
    show_values: bool,
) -> None:
    label = _label_box(_label(component, show_values=show_values))
    center_x = (left[0] + right[0]) / 2
    body_width = max(
        _METRICS.native_span,
        label.width + 2 * _METRICS.terminal_stub,
    )
    body_height = max(
        _METRICS.native_span,
        label.height + 2 * _METRICS.terminal_stub,
    )
    body_left = center_x - body_width / 2
    body_right = center_x + body_width / 2
    _wire(drawing, [left, (body_left, left[1])], color=color, net=(component.id, "left"))
    drawing.add(
        elm.Rect(
            (body_left, left[1] - body_height / 2),
            (body_right, left[1] + body_height / 2),
            color=color,
        ).at((0.0, 0.0))
    )
    _draw_label(drawing, label, (center_x, left[1]), color=color)
    _wire(drawing, [(body_right, right[1]), right], color=color, net=(component.id, "right"))


def _draw_one_terminal_opaque(
    drawing: Any,
    component: Any,
    terminal: tuple[float, float],
    *,
    color: str,
    show_values: bool,
) -> None:
    """Draw one user-owned Composite boundary without exposing its children."""

    width, height = _block_size(component)
    body_left = terminal[0] + _METRICS.element_gap
    body_right = body_left + width
    body_bottom = terminal[1] - height / 2
    body_top = terminal[1] + height / 2
    _wire(
        drawing,
        [terminal, (body_left, terminal[1])],
        color=color,
        net=(component.id, "terminal"),
    )
    drawing.add(
        elm.Rect(
            (body_left, body_bottom),
            (body_right, body_top),
            color=color,
        ).at((0.0, 0.0))
    )
    _draw_label(
        drawing,
        _label_box(_label(component, show_values=show_values)),
        ((body_left + body_right) / 2, terminal[1]),
        color=color,
    )
    pin = next(iter(component._pins.values()))
    _draw_label(
        drawing,
        _label_box(pin.name),
        (
            body_left - _METRICS.label_clearance,
            terminal[1] + _METRICS.label_clearance,
        ),
        color=color,
        horizontal="right",
        vertical="bottom",
    )
    if component._ground_groups:
        ground_anchor = ((body_left + body_right) / 2, body_bottom)
        ground_end = (
            ground_anchor[0],
            ground_anchor[1] - _METRICS.terminal_stub,
        )
        _wire(
            drawing,
            [ground_anchor, ground_end],
            color=color,
            net=(component.id, "internal_ground"),
        )
        _ground(drawing, ground_end, color=color)


def _draw_multiconductor_line(
    drawing: Any,
    plan: Any,
    component: Any,
    buses: dict[int, _NetBus],
    *,
    color: str,
    show_values: bool,
) -> None:
    conductors = tuple(component._realization.get("pin_conductors", ()))
    count = len(conductors)
    pins = tuple(component._pins.values())
    if count == 0 or len(pins) != 2 * count:
        _layout_error(
            "transmission-line pin order does not match its conductor order",
            evidence={"component_id": component.id},
        )
    targets = tuple(plan._pin_nodes[pin] for pin in pins)
    target_points = tuple(
        buses[id(node)].tap(_component_tap_key(component)) for node in targets
    )
    width, height = _block_size(component)
    head_points = target_points[:count]
    tail_points = target_points[count:]
    center_x = (
        max(point[0] for point in head_points)
        + min(point[0] for point in tail_points)
    ) / 2
    center_y = sum(point[1] for point in target_points) / len(target_points)
    placed = drawing.add(
        _TransmissionLineElement(
            width=width,
            height=height,
            title_text=_label(component, show_values=show_values),
            conductors=conductors,
            color=color,
        ).at((center_x - width / 2, center_y))
    )
    for pin, node, target in zip(pins, targets, target_points):
        anchor = placed.absanchors[pin.name]
        anchor_point = (anchor.x, anchor.y)
        _wire(
            drawing,
            [target, (anchor_point[0], target[1]), anchor_point],
            color=color,
            net=id(node),
        )


def _draw_multiterminal_opaque(
    drawing: Any,
    plan: Any,
    component: Any,
    center: tuple[float, float],
    buses: dict[int, _NetBus],
    *,
    color: str,
    show_values: bool,
) -> None:
    width, height = _block_size(component)
    left = center[0] - width / 2
    right = center[0] + width / 2
    bottom = center[1] - height / 2
    top = center[1] + height / 2
    drawing.add(elm.Rect((left, bottom), (right, top), color=color).at((0.0, 0.0)))
    _draw_label(
        drawing,
        _label_box(_label(component, show_values=show_values)),
        center,
        color=color,
    )
    for pin, anchor, normal in _opaque_anchor_specs(component):
        node = plan._pin_nodes.get(pin)
        absolute = (center[0] + anchor[0], center[1] + anchor[1])
        if node == "ground":
            outside = (
                absolute[0] + normal[0] * _METRICS.terminal_stub,
                absolute[1] + normal[1] * _METRICS.terminal_stub,
            )
            _wire(
                drawing,
                [absolute, outside],
                color=color,
                net="ground",
            )
            _ground(drawing, outside, color=color)
            continue
        target = buses[id(node)].tap(_component_tap_key(component))
        outside = (
            absolute[0] + normal[0] * _METRICS.terminal_stub,
            absolute[1] + normal[1] * _METRICS.terminal_stub,
        )
        _wire(
            drawing,
            [target, (outside[0], target[1]), outside, absolute],
            color=color,
            net=id(node),
        )
        _draw_label(
            drawing,
            _label_box(_wrap_identifier(pin.name)),
            (
                absolute[0] - normal[0] * _METRICS.label_clearance,
                absolute[1] + _METRICS.label_clearance,
            ),
            color=color,
            horizontal="left" if normal[0] < 0 else "right",
            vertical="bottom",
        )


def _draw_relation(
    drawing: Any,
    component: Any,
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    lane_index: int,
    lane_count: int,
    color: str,
    show_values: bool,
    net_a: object,
    net_b: object,
) -> None:
    drawing._scnsim_current_component = component.id
    native = component.factory in {
        "resistor",
        "capacitor",
        "inductor",
        "josephson_junction",
        "transmission_line",
    }
    delta_x = abs(a[0] - b[0])
    delta_y = abs(a[1] - b[1])
    horizontal = delta_y < _EPSILON
    vertical = delta_x < _EPSILON or delta_y > delta_x
    block = _component_block(
        component,
        orientation="horizontal" if horizontal else "vertical" if vertical else "horizontal",
        show_values=show_values,
    )
    if horizontal:
        (left_node, left_net), (right_node, right_net) = sorted(
            ((a, net_a), (b, net_b)), key=lambda row: row[0][0]
        )
        lane_y = a[1] + (
            lane_index - (lane_count - 1) / 2
        ) * _METRICS.routing_lane_pitch
        span = (
            _block_size(component)[0]
            if component.factory == "transmission_line"
            else max(_METRICS.native_span, block.symbol_width)
        )
        center_x = (left_node[0] + right_node[0]) / 2
        left = (center_x - span / 2, lane_y)
        right = (center_x + span / 2, lane_y)
        _wire(
            drawing,
            [left_node, (left_node[0], lane_y), left],
            color=color,
            net=left_net,
        )
        _wire(
            drawing,
            [right, (right_node[0], lane_y), right_node],
            color=color,
            net=right_net,
        )
        if native:
            _draw_horizontal_component(
                drawing,
                component,
                left,
                right,
                color=color,
                show_values=show_values,
            )
        else:
            _draw_inline_opaque(
                drawing,
                component,
                left,
                right,
                color=color,
                show_values=show_values,
            )
        return

    if vertical and native and component.factory != "transmission_line":
        lane_x = (a[0] + b[0]) / 2
        if abs(a[0] - b[0]) < _EPSILON:
            route_lane = getattr(drawing, "_scnsim_vertical_lane", 0)
            drawing._scnsim_vertical_lane = route_lane + 1
            lane_x += route_lane * _METRICS.routing_lane_pitch
        (top_node, top_net), (bottom_node, bottom_net) = sorted(
            ((a, net_a), (b, net_b)),
            key=lambda row: row[0][1],
            reverse=True,
        )
        center_y = (top_node[1] + bottom_node[1]) / 2
        top = (lane_x, center_y + _METRICS.native_span / 2)
        bottom = (lane_x, center_y - _METRICS.native_span / 2)
        _wire(
            drawing,
            [top_node, (lane_x, top_node[1]), top],
            color=color,
            net=top_net,
        )
        _draw_vertical_component(
            drawing,
            component,
            top,
            bottom,
            color=color,
            show_values=show_values,
        )
        _wire(
            drawing,
            [bottom, (lane_x, bottom_node[1]), bottom_node],
            color=color,
            net=bottom_net,
        )
        return

    # A diagonal relation is routed orthogonally and keeps its symbol on the
    # horizontal child lane.  This is graph-derived, not a component-ID rule.
    (left_node, left_net), (right_node, right_net) = sorted(
        ((a, net_a), (b, net_b)), key=lambda row: row[0][0]
    )
    route_lane = getattr(drawing, "_scnsim_relation_lane", 0)
    drawing._scnsim_relation_lane = route_lane + 1
    lane_y = right_node[1] + (
        lane_index - (lane_count - 1) / 2 + route_lane
    ) * _METRICS.routing_lane_pitch
    span = (
        _block_size(component)[0]
        if component.factory == "transmission_line"
        else max(_METRICS.native_span, block.symbol_width)
    )
    center_x = (left_node[0] + right_node[0]) / 2
    left = (center_x - span / 2, lane_y)
    right = (center_x + span / 2, lane_y)
    entry = _METRICS.terminal_stub + route_lane * _METRICS.routing_lane_pitch
    left_lane_x = min(left[0], left_node[0] + entry)
    right_lane_x = max(right[0], right_node[0] - entry)
    _wire(
        drawing,
        [
            left_node,
            (left_lane_x, left_node[1]),
            (left_lane_x, lane_y),
            left,
        ],
        color=color,
        net=left_net,
    )
    if native:
        _draw_horizontal_component(
            drawing,
            component,
            left,
            right,
            color=color,
            show_values=show_values,
        )
    else:
        _draw_inline_opaque(
            drawing,
            component,
            left,
            right,
            color=color,
            show_values=show_values,
        )
    _wire(
        drawing,
        [
            right,
            (right_lane_x, lane_y),
            (right_lane_x, right_node[1]),
            right_node,
        ],
        color=color,
        net=right_net,
    )


def _draw_grounded_components(
    drawing: Any,
    plan: Any,
    positions: dict[int, tuple[float, float]],
    buses: dict[int, _NetBus],
    edges: tuple[_Edge, ...],
    *,
    color: str,
    show_values: bool,
) -> set[Any]:
    group_by_pin = _incident_ground_groups(plan)
    grouped: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for component in plan._components:
        ground_pins = tuple(
            pin
            for pin in component._pins.values()
            if plan._pin_nodes.get(pin) == "ground"
        )
        targets = _external_nodes(plan, component)
        if not ground_pins or len(targets) != 1:
            continue
        group_index = group_by_pin.get(ground_pins[0], id(component))
        grouped[(id(targets[0]), group_index)].append(component)

    drawn: set[Any] = set()
    for (node_id, _group), components in sorted(
        grouped.items(),
        key=lambda row: (
            positions[row[0][0]][1],
            positions[row[0][0]][0],
            row[0][1],
        ),
    ):
        node_x, node_y = positions[node_id]
        neighbors: list[tuple[float, float]] = []
        for edge in edges:
            if id(edge.left) == node_id:
                neighbors.append(positions[id(edge.right)])
            elif id(edge.right) == node_id:
                neighbors.append(positions[id(edge.left)])
        above = sum(point[1] > node_y + _EPSILON for point in neighbors)
        below = sum(point[1] < node_y - _EPSILON for point in neighbors)
        direction = "up" if above < below else "down"
        branch_xs = [
            buses[node_id].tap(_component_tap_key(component))[0]
            for component in components
        ]
        bus_y = node_y + (
            _METRICS.grounded_branch_depth
            if direction == "up"
            else -_METRICS.grounded_branch_depth
        )
        lead = max(
            0.0,
            (
                _METRICS.grounded_branch_depth
                - _METRICS.native_span
            )
            / 2,
        )
        for component, branch_x in zip(components, branch_xs):
            if direction == "down":
                start = (branch_x, node_y - lead)
                end = (branch_x, bus_y + lead)
            else:
                start = (branch_x, node_y + lead)
                end = (branch_x, bus_y - lead)
            _wire(
                drawing,
                [(branch_x, node_y), start],
                color=color,
                net=node_id,
            )
            drawing.add(
                _native_element(
                    component,
                    start,
                    end,
                    color=color,
                    show_values=show_values,
                )
            )
            _wire(
                drawing,
                [end, (branch_x, bus_y)],
                color=color,
                net="ground",
            )
            drawn.add(component)
        if len(branch_xs) > 1:
            _wire(
                drawing,
                [(min(branch_xs), bus_y), (max(branch_xs), bus_y)],
                color=color,
                net="ground",
            )
        _ground(drawing, ((min(branch_xs) + max(branch_xs)) / 2, bus_y), color=color, direction=direction)
    return drawn


def render_authoring_schematic(plan: Any, spec: Any) -> Any:
    """Return the current automatic semantic authoring schematic."""

    from .presentation import _palette, _themed_drawing
    from .specs import DiagramSide

    _validate_layout(plan, spec.layout)
    plan, layout = _presentation_plan(plan, spec.layout)
    palette = _palette(spec.theme)
    color, background = palette.foreground, palette.background
    drawing = _themed_drawing(spec.theme)
    drawing.config(
        unit=_METRICS.unit_length,
        color=color,
        bgcolor=background,
        lw=1.8,
        fontsize=10,
    )

    edges = _edges(plan)
    path_rows = _paths_from_hints(plan, layout)
    if not path_rows:
        path_rows = (_automatic_path(plan, edges),)
    positions, path_components = _place_primary_rows(
        path_rows,
        show_values=spec.show_parameter_values,
    )
    multiconductor_lines = _seed_multiconductor_lines(plan, positions)
    opaque_centers = _seed_opaque_blocks(plan, positions)
    _seed_unbounded_cyclic_panels(plan, edges, positions)
    _place_remaining_nodes(
        plan,
        edges,
        positions,
        path_components,
        show_values=spec.show_parameter_values,
        layout=layout,
    )
    buses = _expand_horizontal_layers(
        plan,
        edges,
        positions,
        show_values=spec.show_parameter_values,
        layout=layout,
    )
    _draw_net_buses(drawing, buses, color=color)

    declaration = {component: index for index, component in enumerate(plan._components)}
    pair_groups: dict[tuple[int, int], list[_Edge]] = defaultdict(list)
    for edge in edges:
        key = tuple(sorted((id(edge.left), id(edge.right))))
        pair_groups[key].append(edge)
    for group in pair_groups.values():
        group.sort(key=lambda edge: (declaration[edge.component], edge.component.id))

    drawn: set[Any] = set()
    for path_nodes, components in path_rows:
        for index, component in enumerate(components):
            a = buses[id(path_nodes[index])].tap(
                _component_tap_key(component)
            )
            b = buses[id(path_nodes[index + 1])].tap(
                _component_tap_key(component)
            )
            _draw_relation(
                drawing,
                component,
                a,
                b,
                lane_index=0,
                lane_count=1,
                color=color,
                show_values=spec.show_parameter_values,
                net_a=id(path_nodes[index]),
                net_b=id(path_nodes[index + 1]),
            )
            drawn.add(component)

    for key, group in sorted(pair_groups.items(), key=lambda row: row[0]):
        pending = [edge for edge in group if edge.component not in drawn]
        for lane_index, edge in enumerate(pending):
            _draw_relation(
                drawing,
                edge.component,
                buses[id(edge.left)].tap(_component_tap_key(edge.component)),
                buses[id(edge.right)].tap(_component_tap_key(edge.component)),
                lane_index=lane_index,
                lane_count=len(pending),
                color=color,
                show_values=spec.show_parameter_values,
                net_a=id(edge.left),
                net_b=id(edge.right),
            )
            drawn.add(edge.component)

    drawn.update(
        _draw_grounded_components(
            drawing,
            plan,
            positions,
            buses,
            edges,
            color=color,
            show_values=spec.show_parameter_values,
        )
    )

    for component in sorted(multiconductor_lines, key=_component_key):
        _draw_multiconductor_line(
            drawing,
            plan,
            component,
            buses,
            color=color,
            show_values=spec.show_parameter_values,
        )
        drawn.add(component)

    for component, center in sorted(
        opaque_centers.items(), key=lambda row: _component_key(row[0])
    ):
        _draw_multiterminal_opaque(
            drawing,
            plan,
            component,
            center,
            buses,
            color=color,
            show_values=spec.show_parameter_values,
        )
        drawn.add(component)

    # N-terminal opaque custom components and multi-row MTL blocks retain one
    # coherent block.  Their external nodes were already graph-placed.
    for component in plan._components:
        if component in drawn:
            continue
        targets = _external_nodes(plan, component)
        if len(targets) == 1 and component.catalog_id != "scnsim.components":
            terminal = buses[id(targets[0])].tap(
                _component_tap_key(component)
            )
            _draw_one_terminal_opaque(
                drawing,
                component,
                terminal,
                color=color,
                show_values=spec.show_parameter_values,
            )
            drawn.add(component)
            continue
        _layout_error(
            "automatic authoring layout did not classify a component block",
            evidence={
                "component_id": component.id,
                "factory": component.factory,
                "external_node_count": len(targets),
            },
        )

    primary_port_sides: dict[int, Any] = {}
    if path_rows and path_rows[0][0]:
        first, last = path_rows[0][0][0], path_rows[0][0][-1]
        primary_port_sides[id(first)] = DiagramSide.LEFT
        primary_port_sides[id(last)] = DiagramSide.RIGHT
    center_y = sum(point[1] for point in positions.values()) / max(1, len(positions))
    bounds = (
        min(point[0] for point in positions.values()),
        max(point[0] for point in positions.values()),
        min(point[1] for point in positions.values()),
        max(point[1] for point in positions.values()),
    )
    for port in plan._ports:
        node_id = id(port.node._node)
        point = buses[node_id].tap(_port_tap_key(port))
        through_axis = _through_axis(node_id, positions[node_id], edges, positions)
        requested = (
            layout.port_sides.get(port)
            if layout is not None and port in layout.port_sides
            else None
        )
        if requested is not None:
            allowed = (
                {DiagramSide.TOP, DiagramSide.BOTTOM}
                if through_axis == "horizontal"
                else {DiagramSide.LEFT, DiagramSide.RIGHT}
                if through_axis == "vertical"
                else None
            )
            if allowed is not None and requested not in allowed:
                _layout_error(
                    "Port side must be normal to an interior through bus",
                    evidence={
                        "port_id": port.id,
                        "through_axis": through_axis,
                        "requested_side": requested.value,
                    },
                )
            side = requested
        elif node_id in primary_port_sides:
            side = primary_port_sides[node_id]
        else:
            if through_axis == "horizontal":
                side = DiagramSide.TOP if point[1] >= center_y else DiagramSide.BOTTOM
            elif through_axis == "vertical":
                center_x = (bounds[0] + bounds[1]) / 2
                side = DiagramSide.RIGHT if point[0] >= center_x else DiagramSide.LEFT
            else:
                side = DiagramSide.TOP if point[1] >= center_y else DiagramSide.BOTTOM
        _draw_port(
            drawing,
            port,
            point,
            side=side,
            color=color,
            background=background,
        )

    for node in sorted(plan._nodes, key=_node_key):
        node_id = id(node)
        if node.visibility != "public":
            continue
        point = buses[node_id].identity_anchor
        _dot(drawing, point, color=color)
        _draw_label(
            drawing,
            _label_box(_wrap_identifier(node.id)),
            (point[0], point[1] + _METRICS.label_clearance),
            color=color,
            vertical="bottom",
        )

    return drawing


__all__ = ["render_authoring_schematic"]
