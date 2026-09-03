"""Deterministic semantic authoring-schematic layout.

This module is deliberately presentation-only.  It reads an already authored
``CircuitPlan`` but never seals it, changes it, or contributes to its canonical
identity.  The layout is a deterministic heuristic: relational hints constrain
the graph; all remaining placement uses declaration order and stable IDs.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Avoid importing authoring while it imports this renderer.
    from .authoring import CircuitPlan, ComponentInstance, ElectricNodeRef
    from .specs import CircuitDiagramSpec, DiagramSide, SchematicLayout


_BUILTIN_RESONATORS = frozenset(
    {
        "grounded_parallel_linear_lc_resonator",
        "floating_parallel_linear_lc_resonator",
        "grounded_parallel_single_junction_resonator",
        "floating_parallel_single_junction_resonator",
        "grounded_parallel_symmetric_squid_resonator",
        "floating_parallel_symmetric_squid_resonator",
    }
)


@dataclass(frozen=True, slots=True)
class _Edge:
    component: Any
    left: Any
    right: Any


@dataclass(frozen=True, slots=True)
class _LabelBox:
    """Deterministic text extent used by placement before anything is drawn."""

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
            return self.symbol_width + 0.46 + self.label.width
        return max(self.symbol_width, self.label.width)

    @property
    def height(self) -> float:
        if self.label_side in {"top", "bottom"}:
            return self.symbol_height + 0.46 + self.label.height
        return max(self.symbol_height, self.label.height)


def _label_box(text: str) -> _LabelBox:
    lines = text.splitlines() or [""]
    # Schemdraw ultimately delegates text measurement to the selected backend.
    # Layout cannot depend on that backend, so reserve a conservative extent in
    # drawing units from the fixed 10-point presentation font.
    return _LabelBox(
        text=text,
        width=max(0.8, 0.16 * max(len(line) for line in lines)),
        height=max(0.34, 0.34 * len(lines)),
    )


def _component_block(
    component: Any,
    *,
    orientation: str,
    show_values: bool,
) -> _LayoutBlock:
    label = _label_box(_label(component, show_values=show_values))
    if component.factory == "transmission_line":
        if orientation == "horizontal":
            width, _height = _block_size(component)
            return _LayoutBlock(width, 0.9, label, "inside")
        width, height = _block_size(component)
        return _LayoutBlock(width, height, label, "inside")
    if orientation == "horizontal":
        return _LayoutBlock(2.0, 0.55, label, "top")
    if orientation == "vertical":
        return _LayoutBlock(0.6, 2.2, label, "right")
    return _LayoutBlock(1.8, max(1.2, 0.45 * len(component._pins)), label, "top")


def _parameter_text(
    component: Any,
    parameter_id: str,
    symbol: str,
    *,
    show_values: bool,
) -> str:
    if not show_values:
        return symbol
    parameter = component._parameters[parameter_id]
    return f"{symbol}\n{parameter_id} = {parameter.baseline:~P}"


def _draw_labeled_vertical(
    drawing: Any,
    element: Any,
    top: tuple[float, float],
    bottom: tuple[float, float],
    label: str,
    *,
    color: str,
) -> None:
    drawing.add(element.at(top).down().length(top[1] - bottom[1]))
    _draw_label(
        drawing,
        _label_box(label),
        (top[0] + 0.46, (top[1] + bottom[1]) / 2),
        color=color,
        horizontal="left",
    )


def _draw_grounded_builtin_resonator(
    drawing: Any,
    component: Any,
    node_point: tuple[float, float],
    first_x: float,
    *,
    node_net: object,
    color: str,
    background: str,
    show_values: bool,
) -> tuple[tuple[float, float], ...]:
    """Draw the built-in grounded resonator's declared internal branches."""

    import schemdraw.elements as elm

    top_y = node_point[1]
    bottom_y = top_y - 4.2
    factory = component.factory
    title = (
        "SQUID resonator"
        if "squid" in factory
        else "JJ resonator"
        if "single_junction" in factory
        else "LC resonator"
    )
    _draw_label(
        drawing,
        _label_box(f"{title}  {component.id}"),
        (first_x, top_y + 0.46),
        color=color,
        horizontal="left",
        vertical="bottom",
    )

    rows: list[tuple[float, Any, str, float, float]] = []
    cursor = first_x

    def add_branch(
        element: Any,
        text: str,
        *,
        branch_top: float = top_y,
        branch_bottom: float = bottom_y,
    ) -> float:
        nonlocal cursor
        x = cursor
        rows.append((x, element, text, branch_top, branch_bottom))
        cursor += max(1.9, _label_box(text).width + 1.05)
        return x

    add_branch(
        elm.Capacitor(color=color),
        _parameter_text(
            component,
            "capacitance",
            "C",
            show_values=show_values,
        ),
    )
    if "linear" in factory:
        add_branch(
            elm.Inductor(color=color),
            _parameter_text(
                component,
                "inductance",
                "L",
                show_values=show_values,
            ),
        )
    elif "single_junction" in factory:
        add_branch(
            elm.Josephson(color=color),
            _parameter_text(
                component,
                "josephson_inductance",
                "JJ",
                show_values=show_values,
            ),
        )
        add_branch(
            elm.Capacitor(color=color),
            _parameter_text(
                component,
                "junction_capacitance",
                "Cj",
                show_values=show_values,
            ),
        )
    else:
        # Branch 1 is JJ1 || Cj1.  Branch 2 is loop-L in series with
        # (JJ2 || Cj2), matching the built-in symmetric-SQUID snapshot.
        add_branch(
            elm.Josephson(color=color),
            _parameter_text(
                component,
                "josephson_inductance",
                "JJ₁",
                show_values=show_values,
            ),
        )
        add_branch(
            elm.Capacitor(color=color),
            _parameter_text(
                component,
                "junction_capacitance",
                "Cj₁",
                show_values=show_values,
            ),
        )
        intermediate_y = top_y - 1.55
        loop_text = _parameter_text(
            component,
            "loop_inductance",
            "Lloop",
            show_values=show_values,
        )
        series_x = add_branch(
            elm.Inductor(color=color),
            loop_text,
            branch_bottom=intermediate_y,
        )
        jj2_text = _parameter_text(
            component,
            "josephson_inductance",
            "JJ₂",
            show_values=show_values,
        )
        rows.append(
            (
                series_x,
                elm.Josephson(color=color),
                jj2_text,
                intermediate_y,
                bottom_y,
            )
        )
        cursor = max(
            cursor,
            series_x
            + max(
                1.9,
                _label_box(loop_text).width + 1.05,
                _label_box(jj2_text).width + 1.05,
            ),
        )
        cj2_x = cursor
        cj2_text = _parameter_text(
            component,
            "junction_capacitance",
            "Cj₂",
            show_values=show_values,
        )
        cursor += max(1.9, _label_box(cj2_text).width + 1.05)
        rows.append(
            (
                cj2_x,
                elm.Capacitor(color=color),
                cj2_text,
                intermediate_y,
                bottom_y,
            )
        )
        _wire(
            drawing,
            [(series_x, intermediate_y), (cj2_x, intermediate_y)],
            color=color,
            net=(component.id, "loop_node"),
        )

    top_connected = [x for x, _element, _text, branch_top, _bottom in rows if branch_top == top_y]
    _wire(
        drawing,
        [node_point, (max(top_connected), top_y)],
        color=color,
        net=node_net,
    )
    for x, element, text, branch_top, branch_bottom in rows:
        _draw_labeled_vertical(
            drawing,
            element,
            (x, branch_top),
            (x, branch_bottom),
            text,
            color=color,
        )
    bottom_connected = [x for x, _element, _text, _top, bottom in rows if bottom == bottom_y]
    _wire(
        drawing,
        [(min(bottom_connected), bottom_y), (max(bottom_connected), bottom_y)],
        color=color,
        net="ground",
    )
    ground_x = (min(bottom_connected) + max(bottom_connected)) / 2
    _ground(drawing, (ground_x, bottom_y), color=color)
    return tuple((x, y) for x, _element, _text, top, bottom in rows for y in (top, bottom))


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
    """Choose one stable graph-diameter backbone in the first Port component."""

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
    is_builtin = component.catalog_id == "scnsim.components"
    kind = (
        {
            "interdigitated_capacitor": "IDC",
            "symmetric_squid": "SQUID",
            "transmission_line": "CPW" if len(component._pins) == 2 else "MTL",
        }.get(component.factory)
        if is_builtin
        else None
    )
    if (
        kind is None
        and is_builtin
        and component.factory in _BUILTIN_RESONATORS
    ):
        if "squid" in component.factory:
            kind = "SQUID resonator"
        elif "single_junction" in component.factory:
            kind = "JJ resonator"
        else:
            kind = "LC resonator"
    prefix = "" if kind is None else f"{kind}  "
    if show_values and component._parameters:
        values = [
            f"{parameter.id} = {parameter.baseline:~P}"
            for parameter in component._parameters.values()
        ]
        if len(values) == 1:
            values[0] = f"{next(iter(component._parameters.values())).baseline:~P}"
        return "\n".join((f"{prefix}{component.id}", *values))
    return f"{prefix}{component.id}"


def _element(component: Any, *, color: str) -> Any:
    import schemdraw.elements as elm
    from schemdraw.elements import Element2Term
    from schemdraw.segments import Segment, SegmentCircle, SegmentText

    class _TaggedComposite(Element2Term):
        """Compact dedicated symbol that keeps a Composite encapsulated."""

        def __init__(self, tag: str, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.segments.extend(
                (
                    Segment([(0.0, 0.0), (0.18, 0.0)]),
                    SegmentCircle((0.5, 0.0), 0.32),
                    SegmentText(
                        (0.5, 0.0),
                        tag,
                        align=("center", "center"),
                        fontsize=7,
                    ),
                    Segment([(0.82, 0.0), (1.0, 0.0)]),
                )
            )

    is_builtin = component.catalog_id == "scnsim.components"
    if component._realization.get("kind") == "composite" and not is_builtin:
        return elm.RBox(color=color)
    element_class = {
        "resistor": elm.Resistor,
        "capacitor": elm.Capacitor,
        "inductor": elm.Inductor,
        "josephson_junction": elm.Josephson,
        "interdigitated_capacitor": elm.Capacitor,
    }.get(component.factory)
    if is_builtin and component.factory == "symmetric_squid":
        return _TaggedComposite("SQ", color=color)
    if (
        is_builtin
        and component.factory in _BUILTIN_RESONATORS
    ):
        tag = "SQ" if "squid" in component.factory else "JJ" if "single_junction" in component.factory else "LC"
        return _TaggedComposite(tag, color=color)
    # Composite boundaries are intentionally opaque.  The factory name is
    # carried in the label rather than exposing a hidden child realization.
    return (element_class or elm.RBox)(color=color)


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
                    if horizontal and abs(start[1] - other_start[1]) < 1e-12:
                        overlap = min(max(start[0], end[0]), max(other_start[0], other_end[0])) - max(min(start[0], end[0]), min(other_start[0], other_end[0]))
                    elif not horizontal and abs(start[0] - other_start[0]) < 1e-12:
                        overlap = min(max(start[1], end[1]), max(other_start[1], other_end[1])) - max(min(start[1], end[1]), min(other_start[1], other_end[1]))
                    else:
                        overlap = 0.0
                    if overlap > 1e-12 and other_net != net:
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
                if _detour_depth >= 6:
                    _layout_error("orthogonal router could not separate distinct nets")
                count = getattr(drawing, "_scnsim_detour_count", 0)
                direction = 1.0 if count % 2 == 0 else -1.0
                offset = direction * 0.36 * (1 + count // 2)
                drawing._scnsim_detour_count = count + 1
                if horizontal:
                    detour = [
                        start,
                        (start[0], start[1] + offset),
                        (end[0], end[1] + offset),
                        end,
                    ]
                else:
                    detour = [
                        start,
                        (start[0] + offset, start[1]),
                        (end[0] + offset, end[1]),
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
                    offset = sign * min(0.36, room / 2.0)
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
                    offset = sign * min(0.36, room / 2.0)
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
                gap = 0.18
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
                drawing.add(elm.Arc2(k=0.75, color=color).at(before).to(after))
                cursor = after
            drawing.add(elm.Line(color=color).endpoints(cursor, end))
            drawing._scnsim_wire_segments = (*existing, (start, end, net))


def _dot(drawing: Any, at: tuple[float, float], *, color: str, background: str) -> None:
    import schemdraw.elements as elm

    drawing.add(elm.Dot(color=color, fill=color).at(at))


def _ground(drawing: Any, at: tuple[float, float], *, color: str) -> None:
    import schemdraw.elements as elm

    drawing.add(elm.Ground(color=color).at(at))


def _draw_horizontal_component(
    drawing: Any,
    component: Any,
    left: tuple[float, float],
    right: tuple[float, float],
    *,
    color: str,
    show_values: bool,
) -> None:
    import schemdraw.elements as elm

    length = max(1.2, right[0] - left[0])
    label = _label_box(_label(component, show_values=show_values))
    center = ((left[0] + right[0]) / 2, left[1])
    if component.factory == "transmission_line":
        box_width = min(length - 0.8, max(2.4, label.width + 0.7))
        box_height = max(0.9, label.height + 0.28)
        box_left = center[0] - box_width / 2
        box_right = center[0] + box_width / 2
        drawing.add(elm.Line(color=color).endpoints(left, (box_left, left[1])))
        drawing.add(
            elm.Rect(
                (box_left, left[1] - box_height / 2),
                (box_right, left[1] + box_height / 2),
                color=color,
            ).at((0.0, 0.0))
        )
        drawing.add(elm.Line(color=color).endpoints((box_right, left[1]), right))
        _draw_label(drawing, label, center, color=color)
    else:
        drawing.add(_element(component, color=color).at(left).right().length(length))
        _draw_label(
            drawing,
            label,
            (center[0], left[1] + 0.46),
            color=color,
            vertical="bottom",
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
    length = max(1.1, top[1] - bottom[1])
    drawing.add(_element(component, color=color).at(top).down().length(length))
    label = _label_box(_label(component, show_values=show_values))
    _draw_label(
        drawing,
        label,
        (top[0] + 0.46, (top[1] + bottom[1]) / 2),
        color=color,
        horizontal="left",
    )


def _draw_block(
    drawing: Any,
    plan: Any,
    component: Any,
    center: tuple[float, float],
    terminals: tuple[Any, ...],
    node_pos: dict[Any, tuple[float, float]],
    node_label_sides: dict[int, str],
    *,
    color: str,
    show_values: bool,
) -> None:
    import schemdraw.elements as elm

    x, y = center
    width, height = _block_size(component)
    drawing.add(
        elm.Rect(
            (x - width / 2, y - height / 2),
            (x + width / 2, y + height / 2),
            color=color,
        ).at((0.0, 0.0))
    )
    title = _label_box(_label(component, show_values=show_values))
    if component.factory == "transmission_line":
        _draw_label(
            drawing,
            title,
            (x, y + height / 2 - 0.18),
            color=color,
            vertical="top",
        )
    else:
        _draw_label(
            drawing,
            title,
            (x, y + height / 2 + 0.28),
            color=color,
            vertical="bottom",
        )
    pins = tuple(component._pins.values())
    anchors = _block_anchors(component, center)
    grounded: list[tuple[float, float]] = []
    for pin_index, (pin, anchor) in enumerate(zip(pins, anchors)):
        target = plan._pin_nodes.get(pin)
        if target == "ground":
            grounded.append(anchor)
            continue
        if target is not None:
            point = node_pos[id(target)]
            if abs(anchor[0] - (x - width / 2)) < 1e-12:
                if (
                    component.factory == "transmission_line"
                    and abs(point[1] - anchor[1]) < 1e-12
                    and point[0] < anchor[0]
                ):
                    route = [point, anchor]
                else:
                    outside = (anchor[0] - 0.45, anchor[1])
                    left_count = (len(pins) + 1) // 2
                    lane_y = (
                        y
                        + height / 2
                        + 0.55
                        + 0.4 * (left_count - pin_index)
                    )
                    route = [
                        point,
                        (point[0], lane_y),
                        (outside[0], lane_y),
                        outside,
                        anchor,
                    ]
                node_label_sides[id(target)] = "bottom"
                if component.factory == "transmission_line":
                    pin_label_at = (anchor[0] + 0.18, anchor[1])
                    pin_horizontal, pin_vertical = "left", "center"
                else:
                    pin_label_at = (anchor[0] - 0.18, anchor[1] + 0.16)
                    pin_horizontal, pin_vertical = "right", "bottom"
            elif abs(anchor[0] - (x + width / 2)) < 1e-12:
                if (
                    component.factory == "transmission_line"
                    and abs(point[1] - anchor[1]) < 1e-12
                    and point[0] > anchor[0]
                ):
                    route = [point, anchor]
                else:
                    outside = (anchor[0] + 0.45, anchor[1])
                    right_index = pin_index - len(pins) // 2
                    lane_y = y + height / 2 + 0.55 + 0.4 * right_index
                    route = [
                        point,
                        (point[0], lane_y),
                        (outside[0], lane_y),
                        outside,
                        anchor,
                    ]
                node_label_sides[id(target)] = "bottom"
                if component.factory == "transmission_line":
                    pin_label_at = (anchor[0] - 0.18, anchor[1])
                    pin_horizontal, pin_vertical = "right", "center"
                else:
                    pin_label_at = (anchor[0] + 0.18, anchor[1] + 0.16)
                    pin_horizontal, pin_vertical = "left", "bottom"
            elif abs(anchor[1] - (y + height / 2)) < 1e-12:
                outside = (anchor[0], anchor[1] + 0.45)
                route = [point, (point[0], outside[1]), outside, anchor]
                pin_label_at = (anchor[0] + 0.16, anchor[1] + 0.18)
                pin_horizontal, pin_vertical = "left", "bottom"
            else:
                outside = (anchor[0], anchor[1] - 0.45)
                route = [point, (point[0], outside[1]), outside, anchor]
                pin_label_at = (anchor[0] + 0.16, anchor[1] - 0.18)
                pin_horizontal, pin_vertical = "left", "top"
            _wire(drawing, route, color=color, net=id(target))
            _draw_label(
                drawing,
                _label_box(pin.name),
                pin_label_at,
                color=color,
                horizontal=pin_horizontal,
                vertical=pin_vertical,
            )
    if grounded:
        bus_y = min(anchor[1] for anchor in grounded) - 0.65
        for anchor in grounded:
            _wire(drawing, [anchor, (anchor[0], bus_y)], color=color, net="ground")
        _wire(
            drawing,
            [
                (min(anchor[0] for anchor in grounded), bus_y),
                (max(anchor[0] for anchor in grounded), bus_y),
            ],
            color=color,
            net="ground",
        )
        _ground(drawing, ((min(anchor[0] for anchor in grounded) + max(anchor[0] for anchor in grounded)) / 2, bus_y), color=color)


def _block_size(component: Any) -> tuple[float, float]:
    width = 1.8
    if component.factory == "transmission_line":
        pins = tuple(component._pins.values())
        conductor_count = len(pins) // 2
        left_labels = tuple(_label_box(pin.name) for pin in pins[:conductor_count])
        right_labels = tuple(_label_box(pin.name) for pin in pins[conductor_count:])
        left_width = max((label.width for label in left_labels), default=0.0)
        right_width = max((label.width for label in right_labels), default=0.0)
        title = _label_box(_label(component, show_values=True))
        width = max(
            2.4,
            title.width + 0.7,
            left_width + right_width + 1.25,
        )
        height = max(
            1.8,
            title.height + 0.5 + 0.58 * max(1, conductor_count),
        )
        return width, height
    return width, max(1.2, 0.45 * max(2, len(component._pins)))


def _block_anchors(
    component: Any,
    center: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    """Return clockwise pin anchors beginning at the lower-left boundary."""

    x, y = center
    width, height = _block_size(component)
    anchors: list[tuple[float, float]] = []
    pin_count = len(component._pins)
    if component.factory == "transmission_line" and pin_count % 2 == 0:
        conductor_count = pin_count // 2
        title = _label_box(_label(component, show_values=True))
        row_top = y + height / 2 - title.height - 0.42
        row_bottom = y - height / 2 + 0.28
        row_step = (
            0.0
            if conductor_count == 1
            else (row_top - row_bottom) / (conductor_count - 1)
        )
        for index in range(conductor_count):
            row_y = (
                (row_top + row_bottom) / 2
                if conductor_count == 1
                else row_top - row_step * index
            )
            anchors.append((x - width / 2, row_y))
        for index in range(conductor_count):
            row_y = (
                (row_top + row_bottom) / 2
                if conductor_count == 1
                else row_top - row_step * index
            )
            anchors.append((x + width / 2, row_y))
        return tuple(anchors)
    for index in range(pin_count):
        phase = 4.0 * (index + 0.5) / max(1, pin_count)
        side = int(phase) % 4
        fraction = phase - int(phase)
        if side == 0:
            anchors.append((x - width / 2, y - height / 2 + height * fraction))
        elif side == 1:
            anchors.append((x - width / 2 + width * fraction, y + height / 2))
        elif side == 2:
            anchors.append((x + width / 2, y + height / 2 - height * fraction))
        else:
            anchors.append((x + width / 2 - width * fraction, y - height / 2))
    return tuple(anchors)


def _port_side(
    layout: Any | None,
    port: Any,
    *,
    index: int,
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
    through_axis: str | None,
) -> Any:
    from .specs import DiagramSide

    requested = (
        layout.port_sides.get(port)
        if layout is not None and port in layout.port_sides
        else None
    )
    allowed = (
        {DiagramSide.TOP, DiagramSide.BOTTOM}
        if through_axis == "horizontal"
        else {DiagramSide.LEFT, DiagramSide.RIGHT}
        if through_axis == "vertical"
        else None
    )
    if requested is not None:
        if allowed is not None and requested not in allowed:
            _layout_error(
                "Port side must be normal to an interior through bus",
                evidence={
                    "port_id": port.id,
                    "through_axis": through_axis,
                    "requested_side": requested.value,
                },
            )
        return requested
    if through_axis == "horizontal":
        return DiagramSide.TOP if index % 2 == 0 else DiagramSide.BOTTOM
    if through_axis == "vertical":
        return DiagramSide.RIGHT if index % 2 == 0 else DiagramSide.LEFT
    # First declared Port supplies the left-rooted reading direction.  Other
    # ports are distributed to the nearest perimeter; ties rotate clockwise to
    # reduce coincident boundary labels deterministically.
    if index == 0:
        return DiagramSide.LEFT
    min_x, max_x, min_y, max_y = bounds
    distances = (
        (abs(point[0] - min_x), DiagramSide.LEFT),
        (abs(point[0] - max_x), DiagramSide.RIGHT),
        (abs(point[1] - max_y), DiagramSide.TOP),
        (abs(point[1] - min_y), DiagramSide.BOTTOM),
    )
    nearest = min(distances, key=lambda entry: (entry[0], entry[1].value))[1]
    if nearest is DiagramSide.LEFT:
        return (DiagramSide.RIGHT, DiagramSide.TOP, DiagramSide.BOTTOM)[index % 3]
    return nearest


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
        if abs(other[1] - point[1]) < 1e-12:
            left = left or other[0] < point[0]
            right = right or other[0] > point[0]
        if abs(other[0] - point[0]) < 1e-12:
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
    index: int,
    side: Any,
    through_axis: str | None,
    bounds: tuple[float, float, float, float],
    color: str,
    background: str,
) -> None:
    """Draw the logical boundary and its actual raw-network reference branch."""

    import schemdraw.elements as elm
    from .specs import DiagramSide

    node_x, node_y = node_point
    min_x, max_x, min_y, max_y = bounds
    if side in {DiagramSide.LEFT, DiagramSide.RIGHT}:
        direction = -1.0 if side is DiagramSide.LEFT else 1.0
        boundary_x = min_x - 1.65 if direction < 0 else max_x + 1.65
        if through_axis == "vertical":
            junction = (boundary_x, node_y)
            circle = (junction[0] + direction * 1.35, node_y)
            route = [circle, junction, node_point]
        else:
            junction = node_point
            requested_x = min_x if direction < 0 else max_x
            if abs(node_x - requested_x) < 1e-12:
                circle = (boundary_x, node_y)
                route = [circle, junction]
            else:
                outer_y = max_y + 1.65 + 1.25 * index
                circle = (boundary_x, outer_y)
                route = [circle, (node_x, outer_y), junction]
    elif side is DiagramSide.TOP:
        junction = (node_x, max_y + 1.65)
        circle = (node_x, junction[1] + 1.35)
        route = [circle, junction, node_point]
    else:
        junction = (node_x, min_y - 1.65)
        circle = (node_x, junction[1] - 1.35)
        route = [circle, junction, node_point]
    _wire(drawing, route, color=color, net=id(port.node._node))
    label = f"{port.id}\n{port.role}"
    if port.role == "nonloading_probe":
        label += " (PTC-removable)"
    drawing.add(elm.Dot(open=True, color=color, fill=background).at(circle))
    port_label = _label_box(label)
    label_offset = {
        DiagramSide.LEFT: (-0.25, 0.0, "right", "center"),
        DiagramSide.RIGHT: (0.25, 0.0, "left", "center"),
        DiagramSide.TOP: (0.0, 0.25, "center", "bottom"),
        DiagramSide.BOTTOM: (0.0, -0.25, "center", "top"),
    }[side]
    _draw_label(
        drawing,
        port_label,
        (circle[0] + label_offset[0], circle[1] + label_offset[1]),
        color=color,
        horizontal=label_offset[2],
        vertical=label_offset[3],
    )
    # This is not a Library resistor: it is the visual projection of B/R/M.
    # DiagramSide rotates the three-anchor Port block rather than routing one
    # fixed symbol around the page.
    x, y = junction
    load_label = _label_box(f"R = Z₀ = {port.reference_impedance:~P}")
    if side in {DiagramSide.LEFT, DiagramSide.RIGHT}:
        load_end = (x, y - 2.35)
        drawing.add(elm.Resistor(color=color).at(junction).down().length(2.35))
        _draw_label(
            drawing,
            load_label,
            (x + 0.46, y - 1.175),
            color=color,
            horizontal="left",
        )
    else:
        load_end = (x + 2.35, y)
        drawing.add(elm.Resistor(color=color).at(junction).right().length(2.35))
        _draw_label(
            drawing,
            load_label,
            (x + 1.175, y + 0.46),
            color=color,
            vertical="bottom",
        )
    _ground(drawing, load_end, color=color)


def _incident_ground_groups(plan: Any) -> dict[Any, int]:
    groups: dict[Any, int] = {}
    for index, pins in enumerate(plan._ground_groups):
        for pin in pins:
            groups[pin] = index
    return groups


def _packing_rank(layout: Any | None) -> dict[Any, tuple[int, int]]:
    """Keep each requested group contiguous without adding a group outline."""

    if layout is None:
        return {}
    ranks: dict[Any, tuple[int, int]] = {}
    for group_index, group in enumerate(layout.groups):
        for member_index, component in enumerate(group.members):
            ranks[component] = (group_index, member_index)
    return ranks


def _node_label_width(node: Any) -> float:
    if getattr(node, "visibility", None) != "public":
        return 0.0
    return _label_box(str(node.id)).width


def _place_path(
    nodes: list[Any],
    components: list[Any],
    *,
    y: float,
    show_values: bool,
    packing_rank: dict[Any, tuple[int, int]],
) -> dict[int, tuple[float, float]]:
    """Place one hard path after reserving component and node label extents."""

    if not nodes:
        return {}
    positions = {id(nodes[0]): (0.0, y)}
    x = 0.0
    for index, component in enumerate(components):
        orientation = "horizontal" if len(component._pins) == 2 else "block"
        block = _component_block(
            component,
            orientation=orientation,
            show_values=show_values,
        )
        adjacent_labels = (
            _node_label_width(nodes[index]) + _node_label_width(nodes[index + 1])
        ) / 2
        same_group = (
            index > 0
            and component in packing_rank
            and components[index - 1] in packing_rank
            and packing_rank[component][0] == packing_rank[components[index - 1]][0]
        )
        clearance = 0.8 if same_group else 1.5
        x += max(4.2, block.width + clearance, adjacent_labels + clearance)
        positions[id(nodes[index + 1])] = (x, y)
    return positions


def render_authoring_schematic(plan: Any, spec: Any) -> Any:
    """Return a real themed Schemdraw Drawing for an authored CircuitPlan."""

    from .presentation import _palette, _themed_drawing
    import schemdraw.elements as elm

    _validate_layout(plan, spec.layout)
    palette = _palette(spec.theme)
    color, background = palette.foreground, palette.background
    drawing = _themed_drawing(spec.theme)
    drawing.config(unit=2.2, color=color, bgcolor=background, lw=1.8, fontsize=10)

    edges = _edges(plan)
    packing_rank = _packing_rank(spec.layout)
    hinted_paths = _paths_from_hints(plan, spec.layout)
    if hinted_paths:
        path_rows = hinted_paths
    else:
        path_rows = (_automatic_path(plan, edges),)
    node_pos: dict[int, tuple[float, float]] = {}
    path_edges: list[tuple[Any, Any, Any]] = []
    for path_index, (path_nodes, path_components) in enumerate(path_rows):
        node_pos.update(
            _place_path(
                path_nodes,
                path_components,
                y=-7.0 * path_index,
                show_values=spec.show_parameter_values,
                packing_rank=packing_rank,
            )
        )
        path_edges.extend(
            (component, path_nodes[index], path_nodes[index + 1])
            for index, component in enumerate(path_components)
        )
    path_component_set = {component for component, _left, _right in path_edges}
    branch_hint_rank: dict[Any, int] = {}
    if spec.layout is not None:
        for members in spec.layout.branch_order.values():
            for index, component in enumerate(members):
                if component in path_component_set:
                    _layout_error(
                        "branch_order conflicts with a primary-path component",
                        evidence={"component_id": component.id},
                    )
                branch_hint_rank[component] = index
    # Seed every N-terminal component as one coherent layout unit.  Nodes on a
    # selected path constrain the block center; all remaining pins are then
    # placed from that same block's anchors.  They must never fall through to
    # the disconnected-panel pass as though they belonged to another object.
    fixed_block_centers: dict[Any, tuple[float, float]] = {}
    block_seed_right: float | None = None
    for component in plan._components:
        targets = _external_nodes(plan, component)
        if len(component._pins) <= 2:
            continue
        width, height = _block_size(component)
        pins = tuple(component._pins.values())
        relative_anchors = _block_anchors(component, (0.0, 0.0))
        center_candidates = []
        for pin, anchor in zip(pins, relative_anchors):
            target = plan._pin_nodes.get(pin)
            if target is not None and target != "ground" and id(target) in node_pos:
                point = node_pos[id(target)]
                center_candidates.append(
                    (point[0] - anchor[0], point[1] - anchor[1])
                )
        if center_candidates:
            center = (
                sum(point[0] for point in center_candidates)
                / len(center_candidates),
                sum(point[1] for point in center_candidates)
                / len(center_candidates),
            )
        else:
            center_x = (
                0.0
                if block_seed_right is None
                else block_seed_right + 3.6 + width / 2
            )
            center = (center_x, -7.0 * len(path_rows) if node_pos else 0.0)
            block_seed_right = center_x + width / 2
        fixed_block_centers[component] = center
        for pin, anchor in zip(pins, _block_anchors(component, center)):
            target = plan._pin_nodes.get(pin)
            if target is None or target == "ground" or id(target) in node_pos:
                continue
            if abs(anchor[0] - (center[0] - width / 2)) < 1e-12:
                point = (anchor[0] - 2.4, anchor[1])
            elif abs(anchor[0] - (center[0] + width / 2)) < 1e-12:
                point = (anchor[0] + 2.4, anchor[1])
            elif abs(anchor[1] - (center[1] + height / 2)) < 1e-12:
                point = (anchor[0], anchor[1] + 2.4)
            else:
                point = (anchor[0], anchor[1] - 2.4)
            node_pos[id(target)] = point

    if not node_pos and plan._nodes:
        node_pos[id(plan._nodes[0])] = (0.0, 0.0)

    # Complete the graph in deterministic breadth-first layers.  Branches use
    # independent rows; canonical ground never becomes a graph vertex.
    adjacency: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    node_by_id = {id(node): node for node in plan._nodes}
    for edge in edges:
        left, right = id(edge.left), id(edge.right)
        adjacency[left].append((right, edge.component))
        adjacency[right].append((left, edge.component))
    for values in adjacency.values():
        values.sort(key=lambda row: (_component_key(row[1]), _node_key(node_by_id[row[0]])))
    occupied = list(node_pos.values())

    def reserve(candidate: tuple[float, float]) -> tuple[float, float]:
        x, y = candidate
        while any(abs(x - ox) < 2.1 and abs(y - oy) < 2.1 for ox, oy in occupied):
            y += 3.0
        occupied.append((x, y))
        return x, y

    def expand(pending: deque[int]) -> None:
        while pending:
            node = pending.popleft()
            children = [row for row in adjacency[node] if row[0] not in node_pos]
            for child_index, (other, _component) in enumerate(children):
                parent = node_pos[node]
                distance = max(
                    4.2,
                    (
                        _node_label_width(node_by_id[node])
                        + _node_label_width(node_by_id[other])
                    )
                    / 2
                    + 1.5,
                )
                lane = (child_index // 2 + 1) * (1 if child_index % 2 == 0 else -1)
                node_pos[other] = reserve(
                    (
                        parent[0] + distance * (1.5 + child_index),
                        parent[1] + 3.0 * lane,
                    )
                )
                pending.append(other)

    expand(deque(sorted(node_pos, key=lambda node: _node_key(node_by_id[node]))))

    # Each disconnected graph receives its own panel and is then laid out as
    # one graph.  Nodes of a disconnected component are never mistaken for
    # independent page-wide rails.
    panel_y = min((point[1] for point in node_pos.values()), default=7.0) - 7.0
    for node in sorted(plan._nodes, key=_node_key):
        node_id = id(node)
        if node_id in node_pos:
            continue
        node_pos[node_id] = reserve((0.0, panel_y))
        panel_y -= 7.0
        expand(deque([node_id]))

    geometry_points = list(node_pos.values())
    node_label_sides: dict[int, str] = {}

    drawn: set[Any] = set()
    # The ordered series backbone is horizontal and remains the visual primary
    # path.  It is intentionally not a node rail.
    for component, left_node, right_node in path_edges:
        if len(component._pins) == 2:
            _draw_horizontal_component(
                drawing,
                component,
                node_pos[id(left_node)],
                node_pos[id(right_node)],
                color=color,
                show_values=spec.show_parameter_values,
            )
        else:
            left = node_pos[id(left_node)]
            right = node_pos[id(right_node)]
            _draw_block(
                drawing,
                plan,
                component,
                fixed_block_centers.get(
                    component,
                    ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2),
                ),
                _external_nodes(plan, component),
                node_pos,
                node_label_sides,
                color=color,
                show_values=spec.show_parameter_values,
            )
        drawn.add(component)

    ground_group = _incident_ground_groups(plan)
    declaration_index = {component: index for index, component in enumerate(plan._components)}
    shunts: dict[int, list[tuple[Any, object]]] = defaultdict(list)
    remaining: list[Any] = []
    for component in plan._components:
        if component in drawn:
            continue
        targets = _external_nodes(plan, component)
        ground_pins = [pin for pin in component._pins.values() if plan._pin_nodes.get(pin) == "ground"]
        internal_ground = bool(component._ground_groups)
        simple_grounded_symbol = len(component._pins) == 2 and bool(ground_pins)
        encapsulated_grounded_symbol = len(component._pins) == 1 and internal_ground
        if len(targets) == 1 and (
            simple_grounded_symbol or encapsulated_grounded_symbol
        ):
            group: object
            if ground_pins:
                group = ("explicit", ground_group.get(ground_pins[0], -1))
            else:
                group = ("internal", declaration_index[component])
            shunts[id(targets[0])].append((component, group))
        else:
            remaining.append(component)

    # Draw each Plan ground() call as a local short bus.  A component with an
    # internal Composite ground receives its own local glyph without exposing
    # the children that form it.
    shunt_by_group: dict[object, list[tuple[int, Any]]] = defaultdict(list)
    for node_id, values in shunts.items():
        ordered = list(values)
        if spec.layout is not None:
            hinted_order = next(
                (
                    members
                    for ref, members in spec.layout.branch_order.items()
                    if ref._node is node_by_id[node_id]
                ),
                (),
            )
            if hinted_order:
                lookup = {component: index for index, component in enumerate(hinted_order)}
                ordered.sort(key=lambda row: (lookup.get(row[0], len(lookup)), packing_rank.get(row[0], (len(packing_rank), declaration_index[row[0]])), declaration_index[row[0]], row[0].id))
            else:
                ordered.sort(key=lambda row: (packing_rank.get(row[0], (len(packing_rank), declaration_index[row[0]])), declaration_index[row[0]], row[0].id))
        for component, group in ordered:
            shunt_by_group[group].append((node_id, component))

    port_at_node = {id(port.node._node): port for port in plan._ports}
    node_branch_cursor: dict[int, float] = {}
    for node_id in shunts:
        clearance = 1.25
        port = port_at_node.get(node_id)
        if port is not None:
            load_label = _label_box(f"R = Z₀ = {port.reference_impedance:~P}")
            clearance = max(clearance, 0.46 + load_label.width + 0.75)
        node_branch_cursor[node_id] = node_pos[node_id][0] + clearance
    node_ground_depth: dict[int, int] = defaultdict(int)
    for group, entries in sorted(
        shunt_by_group.items(),
        key=lambda row: repr(row[0]),
    ):
        if (
            len(entries) == 1
            and group[0] == "internal"
            and entries[0][1].catalog_id == "scnsim.components"
            and entries[0][1].factory.startswith("grounded_parallel_")
            and entries[0][1].factory in _BUILTIN_RESONATORS
        ):
            node_id, component = entries[0]
            points = _draw_grounded_builtin_resonator(
                drawing,
                component,
                node_pos[node_id],
                node_branch_cursor[node_id],
                node_net=node_id,
                color=color,
                background=background,
                show_values=spec.show_parameter_values,
            )
            geometry_points.extend(points)
            node_branch_cursor[node_id] = max(point[0] for point in points) + 1.4
            node_ground_depth[node_id] += 1
            drawn.add(component)
            continue
        branch_rows: list[tuple[int, Any, float]] = []
        prior_at_node: dict[int, Any] = {}
        for node_id, component in entries:
            block = _component_block(
                component,
                orientation="vertical",
                show_values=spec.show_parameter_values,
            )
            previous = prior_at_node.get(node_id)
            same_group = (
                previous in packing_rank
                and component in packing_rank
                and packing_rank[previous][0] == packing_rank[component][0]
            )
            if previous is not None:
                node_branch_cursor[node_id] += 0.35 if same_group else 1.0
            branch_x = node_branch_cursor[node_id]
            branch_rows.append((node_id, component, branch_x))
            node_branch_cursor[node_id] += block.width
            prior_at_node[node_id] = component
        group_nodes = {node_id for node_id, _component, _x in branch_rows}
        depth = max(node_ground_depth[node_id] for node_id in group_nodes)
        bus_y = min(node_pos[node_id][1] for node_id in group_nodes) - 4.2 - depth
        for node_id in group_nodes:
            node_ground_depth[node_id] = depth + 1
            x, y = node_pos[node_id]
            furthest = max(
                branch_x
                for branch_node, _component, branch_x in branch_rows
                if branch_node == node_id
            )
            _wire(drawing, [(x, y), (furthest, y)], color=color, net=node_id)
        for node_id, component, branch_x in branch_rows:
            node_y = node_pos[node_id][1]
            geometry_points.extend(((branch_x, node_y), (branch_x, bus_y)))
            _draw_vertical_component(
                drawing,
                component,
                (branch_x, node_y),
                (branch_x, bus_y),
                color=color,
                show_values=spec.show_parameter_values,
            )
            drawn.add(component)
        branch_positions = [branch_x for _node, _component, branch_x in branch_rows]
        if len(branch_positions) > 1:
            _wire(
                drawing,
                [(min(branch_positions), bus_y), (max(branch_positions), bus_y)],
                color=color,
                net="ground",
            )
        _ground(
            drawing,
            ((min(branch_positions) + max(branch_positions)) / 2, bus_y),
            color=color,
        )

    # General fallback for cycles, non-planar connections, N-terminal blocks,
    # and disconnected panels.  Orthogonal lanes make crossings visible rather
    # than silently turning them into electrical junctions.
    fallback_top = max((point[1] for point in geometry_points), default=0.0) + 1.2
    block_slot = 0
    for component in sorted(
        remaining,
        key=lambda item: (
            branch_hint_rank.get(item, len(branch_hint_rank)),
            packing_rank.get(item, (len(packing_rank), declaration_index[item])),
            declaration_index[item],
            item.id,
        ),
    ):
        targets = _external_nodes(plan, component)
        if len(component._pins) == 2 and len(targets) == 2:
            left, right = targets
            a, b = node_pos[id(left)], node_pos[id(right)]
            route_block = _component_block(
                component,
                orientation="horizontal",
                show_values=spec.show_parameter_values,
            )
            route_y = fallback_top + route_block.height / 2
            fallback_top = route_y + route_block.height / 2 + 1.0
            center = ((a[0] + b[0]) / 2, route_y)
            half_width = max(
                0.8,
                (route_block.width + 0.8) / 2
                if component.factory == "transmission_line"
                else 0.8,
            )
            geometry_points.extend(((a[0], route_y), (b[0], route_y), center))
            _wire(
                drawing,
                [a, (a[0], route_y), (center[0] - half_width, route_y)],
                color=color,
                net=id(left),
            )
            _draw_horizontal_component(
                drawing,
                component,
                (center[0] - half_width, route_y),
                (center[0] + half_width, route_y),
                color=color, show_values=spec.show_parameter_values,
            )
            _wire(
                drawing,
                [(center[0] + half_width, route_y), (b[0], route_y), b],
                color=color,
                net=id(right),
            )
        else:
            if component in fixed_block_centers:
                center_x, center_y = fixed_block_centers[component]
            elif targets:
                center_x = sum(node_pos[id(node)][0] for node in targets) / len(targets)
                center_y = sum(node_pos[id(node)][1] for node in targets) / len(targets) + 2.2
            else:
                center_x = 3.8 * block_slot
                center_y = 0.0
                block_slot += 1
            block = _component_block(
                component,
                orientation="block",
                show_values=spec.show_parameter_values,
            )
            geometry_points.extend(
                (
                    (center_x - block.width / 2, center_y - block.height / 2),
                    (center_x + block.width / 2, center_y + block.height / 2),
                )
            )
            _draw_block(
                drawing,
                plan,
                component,
                (center_x, center_y),
                targets,
                node_pos,
                node_label_sides,
                color=color,
                show_values=spec.show_parameter_values,
            )
        drawn.add(component)

    # Identity dots and labels are emitted once after all geometry.  A filled
    # dot denotes a named Public-node identity, never merely a wire join.  A
    # port-promoted node is intentionally represented by its logical Port only.
    port_by_node = port_at_node
    bounds = (
        min((point[0] for point in geometry_points), default=0.0),
        max((point[0] for point in geometry_points), default=0.0),
        min((point[1] for point in geometry_points), default=0.0),
        max((point[1] for point in geometry_points), default=0.0),
    )
    for node in sorted(plan._nodes, key=_node_key):
        node_id = id(node)
        point = node_pos[node_id]
        port = port_by_node.get(node_id)
        if port is not None:
            through_axis = _through_axis(node_id, point, edges, node_pos)
            _draw_port(
                drawing,
                port,
                point,
                index=plan._ports.index(port),
                side=_port_side(
                    spec.layout,
                    port,
                    index=plan._ports.index(port),
                    point=point,
                    bounds=bounds,
                    through_axis=through_axis,
                ),
                through_axis=through_axis,
                bounds=bounds,
                color=color,
                background=background,
            )
        elif node.visibility == "public":
            _dot(drawing, point, color=color, background=background)
        if node.visibility == "public":
            if node_label_sides.get(node_id) == "bottom":
                label_at = (point[0], point[1] - 0.24)
                vertical = "top"
            else:
                label_at = (point[0], point[1] + 0.24)
                vertical = "bottom"
            _draw_label(
                drawing,
                _label_box(node.id),
                label_at,
                color=color,
                vertical=vertical,
            )

    return drawing


__all__ = ["render_authoring_schematic"]
