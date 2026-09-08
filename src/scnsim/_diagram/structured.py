"""Snapshot-only lowering for structured authoring schematics.

This is intentionally a small local composer, not a graph layout engine.  A
scope supplies its declared connector records and public endpoint bindings;
the lowerer places those records near the anchors they explicitly name and
routes only the resulting local contacts.  It never manufactures a
``main`` path, a semantic partition, or a page-wide electrical rail.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import pairwise

from .._authoring_snapshot import ResolvedPlanPoint
from .._physical_values import quantity_record
from ..errors import SCNSimValidationError
from .metrics import DEFAULT_METRICS, shape_text
from .native import (
    electrical_box,
    ground_mark,
    junction_mark,
    native_block,
    port_block,
    winding_dot,
)
from .routing import (
    _compress,
    _inflate,
    _on_segment,
    _segment_clear,
)
from .scene import (
    COORDINATE_TOLERANCE,
    BoundarySite,
    Bounds,
    ConductivePolyline,
    ElectricalBox,
    GuideMark,
    NativeSymbol,
    Path,
    Point,
    SceneBuilder,
    SubsystemRegion,
    TextRun,
)
from .values import format_envelope


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_layout", evidence=evidence)


def _record(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail("authoring snapshot has malformed structured record", field=name)
    return value


def _rows(value: object, *, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (tuple, list)):
        raise _fail("authoring snapshot has malformed structured rows", field=name)
    return tuple(_record(item, name=name) for item in value)


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail("authoring snapshot identifier is malformed", field=field)
    return value


def _path(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not all(
        isinstance(x, str) and x for x in value
    ):
        raise _fail("authoring snapshot path is malformed", field=field)
    return tuple(value)


def _endpoint_key(endpoint: Mapping[str, object]) -> str:
    """A private geometry key, never a persisted electrical identity."""

    def plain(value: object) -> object:
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise _fail("authoring endpoint has a non-string record key")
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        if value is None or isinstance(value, (str, int, bool)):
            return value
        raise _fail(
            "authoring endpoint has a non-serializable geometry value",
            value_type=type(value).__name__,
        )

    try:
        return json.dumps(
            plain(endpoint), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise _fail("authoring endpoint cannot be used for local geometry") from exc


def _bounds(items: Sequence[Bounds], *, padding: float) -> Bounds:
    if not items:
        return Bounds(-padding, -padding, padding, padding)
    return Bounds(
        min(item.xmin for item in items) - padding,
        min(item.ymin for item in items) - padding,
        max(item.xmax for item in items) + padding,
        max(item.ymax for item in items) + padding,
    )


def _translated(value: object, dx: float, dy: float):
    """Translate a measured fragment, including frozen glyph outlines."""
    if isinstance(value, (Point, Bounds)):
        return value.translated(dx, dy)
    if isinstance(value, tuple):
        return tuple(_translated(item, dx, dy) for item in value)
    if is_dataclass(value):
        return replace(
            value,
            **{
                field.name: _translated(getattr(value, field.name), dx, dy)
                for field in fields(value)
                if field.init
            },
        )
    return value


def _rigid_text(run: TextRun, degrees: int) -> TextRun:
    """Rotate already measured glyph ink without changing its reserved footprint."""
    degrees %= 360
    return replace(run, origin=run.origin.rotated(degrees),
        bounds=_oriented(run.bounds, degrees),
        glyphs=tuple(replace(glyph,
            vertices=tuple(point.rotated(degrees) for point in glyph.vertices),
            bounds=_oriented(glyph.bounds, degrees)) for glyph in run.glyphs))


def _oriented(value: object, degrees: int):
    """Orient physical geometry while reshaping literal text in an upright frame."""
    if isinstance(value, NativeSymbol):
        anchors = dict(value.anchors)
        start,end = anchors["terminal_1"].rotated(degrees),anchors["terminal_2"].rotated(degrees)
        label_center=Point((value.visible_name.bounds.xmin+value.visible_name.bounds.xmax)/2,(value.visible_name.bounds.ymin+value.visible_name.bounds.ymax)/2).rotated(degrees)
        side=("top" if label_center.y >= (start.y+end.y)/2 else "bottom") if abs(start.y-end.y)<=COORDINATE_TOLERANCE else ("right" if label_center.x >= (start.x+end.x)/2 else "left")
        return native_block(value.kind, start=start, end=end, name=value.visible_name.text, value=None if value.value is None else value.value.text, branch_label=None if value.branch_label is None else value.branch_label.text, correlation_key=value.correlation_key, label_side=side)
    if isinstance(value, Point):
        return value.rotated(degrees)
    if isinstance(value, Bounds):
        return Bounds.around(Point(x, y).rotated(degrees) for x in (value.xmin, value.xmax) for y in (value.ymin, value.ymax))
    if isinstance(value, TextRun):
        if value.role in {"public-pin", "tap-id", "region-id"}:
            return _rigid_text(value, degrees)
        center = Point((value.bounds.xmin + value.bounds.xmax) / 2, (value.bounds.ymin + value.bounds.ymax) / 2).rotated(degrees)
        measured = shape_text(value.text, at=Point(0.0, 0.0), size=value.size, role=value.role)
        return _translated(measured, center.x - (measured.bounds.xmin + measured.bounds.xmax) / 2, center.y - (measured.bounds.ymin + measured.bounds.ymax) / 2)
    if isinstance(value, tuple):
        return tuple(_oriented(item, degrees) for item in value)
    if isinstance(value, Mapping):
        return {key: _oriented(item, degrees) for key, item in value.items()}
    if is_dataclass(value):
        result = replace(value, **{field.name: _oriented(getattr(value, field.name), degrees) for field in fields(value)})
        if isinstance(result, NativeSymbol):
            result = replace(result, orientation=(value.orientation + degrees) % 360,
                occupied_bounds=_bounds((result.symbol_bounds, result.visible_name.bounds, *(() if result.value is None else (result.value.bounds,)), *(() if result.branch_label is None else (result.branch_label.bounds,))), padding=0.0))
        elif isinstance(result, ElectricalBox):
            b=result.outline.bounds
            outline=replace(result.outline,points=(Point(b.xmin,b.ymin),Point(b.xmax,b.ymin),Point(b.xmax,b.ymax),Point(b.xmin,b.ymax),Point(b.xmin,b.ymin)))
            result = _orient_box_labels(replace(result, orientation=(value.orientation + degrees) % 360,outline=outline))
        return result
    return value


def _orient_box_labels(box: ElectricalBox) -> ElectricalBox:
    """Place upright captions by one fixed row-major shelf in the physical frame."""
    runs = tuple(run for run in (box.title,box.kind_label,box.length_label,box.section_label,
        *box.anchor_labels,*box.conductor_rows,box.reference_label) if run is not None)
    frame = box.outline.bounds
    gap = DEFAULT_METRICS.label_clearance
    x,top,row_height = frame.xmin+gap,frame.ymax-gap,0.0
    placed = {}
    for run in runs:
        width,height = run.bounds.xmax-run.bounds.xmin,run.bounds.ymax-run.bounds.ymin
        if x+width > frame.xmax-gap and x > frame.xmin+gap:
            x,top,row_height = frame.xmin+gap,top-row_height-gap,0.0
        bound = Bounds(x,top-height,x+width,top)
        if not frame.contains(_inflate(bound,gap-COORDINATE_TOLERANCE)):
            raise _fail("fixed upright line captions exceed their physical frame",
                body=box.title.text,label=run.text,frame=frame)
        placed[id(run)] = _translated(run,x-run.bounds.xmin,top-height-run.bounds.ymin)
        x += width+gap
        row_height = max(row_height,height)
    def remap(run):
        return None if run is None else placed[id(run)]
    result = replace(box,title=remap(box.title),kind_label=remap(box.kind_label),
        length_label=remap(box.length_label),section_label=remap(box.section_label),anchor_labels=tuple(remap(run) for run in box.anchor_labels),
        conductor_rows=tuple(remap(run) for run in box.conductor_rows),reference_label=remap(box.reference_label))
    return replace(result,bounds=_bounds((result.outline.bounds,
        *(path.bounds for path in result.paths),*(run.bounds for run in placed.values())),padding=0.0))


@dataclass(frozen=True, slots=True)
class _Leaf:
    path: tuple[str, ...]
    model: str
    pins: tuple[str, ...]
    fields: Mapping[str, Mapping[str, object]]
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Placed:
    path: tuple[str, ...]
    anchors: Mapping[str, Point]
    bounds: Bounds
    identity_anchor: Point




class StructuredLowerer:
    """One-use snapshot projection with fixed measured composition."""

    def __init__(
        self, point: ResolvedPlanPoint, layout: object, *, show_values: bool,
    ) -> None:
        if not isinstance(point, ResolvedPlanPoint):
            raise TypeError("structured lowering requires ResolvedPlanPoint")
        self.point = point
        self.layout = layout
        self.show_values = show_values
        semantic = _record(point.snapshot.semantic_record, name="semantic_record")
        if (
            semantic.get("schema") != "scnsim.authoring_snapshot"
            or semantic.get("schema_version") != 2
        ):
            raise _fail("diagram lowering requires authoring snapshot schema version 2")
        self.semantic = semantic
        self.builder = SceneBuilder()
        self.leaves = self._leaves(
            _rows(semantic.get("physical_leaves"), name="physical_leaves")
        )
        self.occurrences = {
            _path(row.get("path"), field="occurrence.path"): row
            for row in _rows(semantic.get("occurrences"), name="occurrences")
        }
        connectivity = _record(semantic.get("connectivity"), name="connectivity")
        self.endpoint_nets = self._endpoint_nets(connectivity)
        self.endpoint_records = {
            _endpoint_key(
                _record(row.get("endpoint"), name="endpoint_nets.endpoint")
            ): _record(row.get("endpoint"), name="endpoint_nets.endpoint")
            for row in _rows(connectivity.get("endpoint_nets"), name="endpoint_nets")
        }
        self.placed: dict[tuple[str, ...], _Placed] = {}
        self.endpoint_anchors: dict[str, list[Point]] = defaultdict(list)
        self.scope_bounds: dict[tuple[str, ...], Bounds] = {}
        self.scope_origins: dict[tuple[str, ...], Point] = {(): Point(0.0, 0.0)}
        self.active_scope: tuple[str, ...] = ()
        self.prewired: list[tuple[str, ConductivePolyline]] = []
        self._validate_hints()

    @staticmethod
    def _leaves(rows: tuple[Mapping[str, object], ...]) -> dict[tuple[str, ...], _Leaf]:
        result: dict[tuple[str, ...], _Leaf] = {}
        for row in rows:
            path = _path(row.get("path"), field="physical_leaf.path")
            if path in result:
                raise _fail("authoring snapshot repeats physical leaf path", path=path)
            field_rows = _rows(row.get("fields"), name="physical_leaf.fields")
            fields = {
                _identifier(field.get("id"), field="physical_leaf.field.id"): field
                for field in field_rows
            }
            if len(fields) != len(field_rows):
                raise _fail("authoring snapshot repeats a physical field", path=path)
            result[path] = _Leaf(
                path,
                _identifier(row.get("model"), field="physical_leaf.model"),
                tuple(
                    _identifier(pin, field="physical_leaf.pin")
                    for pin in row.get("pin_order", ())
                ),
                fields,
                _record(
                    row.get("model_metadata", {}), name="physical_leaf.model_metadata"
                ),
            )
        return result

    @staticmethod
    def _endpoint_nets(connectivity: Mapping[str, object]) -> Mapping[str, str]:
        result: dict[str, str] = {}
        for row in _rows(connectivity.get("endpoint_nets"), name="endpoint_nets"):
            endpoint = _record(row.get("endpoint"), name="endpoint_nets.endpoint")
            key = _endpoint_key(endpoint)
            net = _identifier(row.get("final_net"), field="endpoint_nets.final_net")
            old = result.setdefault(key, net)
            if old != net:
                raise _fail(
                    "one endpoint has conflicting final nets", endpoint=endpoint
                )
        return result

    def _validate_hints(self) -> None:
        """Reject stale public-handle keys before any geometry is emitted."""
        if self.layout is None:
            return
        axes = getattr(self.layout, "axes", None)
        order = getattr(self.layout, "order", None)
        terminal_sides = getattr(self.layout, "terminal_sides", None)
        port_sides = getattr(self.layout, "port_sides", None)
        tap_order = getattr(self.layout, "tap_order", None)
        if not all(
            isinstance(item, Mapping)
            for item in (axes, order, terminal_sides, port_sides, tap_order)
        ):
            raise _fail("SchematicLayout has no frozen direct-handle mappings")
        scope_paths = set(self._scope_records())
        structure_keys = {
            (row.get("kind"), path, row.get("id"))
            for path, scope in self._scope_records().items()
            for row in _rows(scope.get("structures"), name="scope.structures")
        }
        for key in (*axes, *order):
            if not isinstance(key, tuple) or not key:
                raise _fail("layout key is malformed")
            if key[0] == "scope" and tuple(key[1]) not in scope_paths:
                raise _fail(
                    "layout references a scope absent from captured snapshot", key=key
                )
            if (
                key[0] in {"series", "parallel", "branch"}
                and (key[0], tuple(key[1]), key[2]) not in structure_keys
            ):
                raise _fail(
                    "layout references a structure absent from captured snapshot",
                    key=key,
                )

    def _scope_records(self) -> Mapping[tuple[str, ...], Mapping[str, object]]:
        root = _record(self.semantic.get("scope_hierarchy"), name="scope_hierarchy")
        result: dict[tuple[str, ...], Mapping[str, object]] = {}

        def visit(scope: Mapping[str, object]) -> None:
            path = _path(scope.get("path"), field="scope.path")
            if path in result:
                raise _fail("authoring snapshot repeats scope path", path=path)
            result[path] = scope
            for child in _rows(scope.get("children"), name="scope.children"):
                visit(child)
            for row in _rows(
                scope.get("component_bodies"), name="scope.component_bodies"
            ):
                visit(_record(row.get("body"), name="component_body.body"))

        visit(root)
        return result


    def _field_text(self, leaf: _Leaf, field_id: str) -> str | None:
        if not self.show_values:
            return None
        field = leaf.fields.get(field_id)
        if field is None:
            raise _fail(
                "physical leaf misses required displayed field",
                path=leaf.path,
                field=field_id,
            )
        unit = _identifier(field.get("unit"), field="physical_leaf.field.unit")
        value = self.point.resolved_fields.get((leaf.path, field_id))
        if value is None:
            raise _fail(
                "resolved point misses physical field", path=leaf.path, field=field_id
            )
        if unit == "rlgc":
            return None
        return format_envelope(quantity_record(value, unit))

    def _composition_direction(self, key: tuple[object, ...], axis: str) -> str:
        """Orient an ordered native Block toward its captured primitive arm."""
        from .composition_geometry import OPPOSITE

        default = "right" if axis == "horizontal" else "bottom"
        if self.layout is None or not hasattr(self.layout, "wiring"):
            return default
        ground = self.layout.ground_sides.get(key)
        if ground is not None:
            return ground
        for recipe in self.layout.wiring.values():
            for attachment in recipe.group.attachments:
                if attachment.block != key:
                    continue
                target = ("attachment", attachment.key)
                for connection in recipe.connections:
                    other = (
                        connection.b
                        if connection.a == target
                        else connection.a
                        if connection.b == target
                        else None
                    )
                    if other is None or other[0] != "arm":
                        continue
                    direction = (
                        other[2] if attachment.boundary != "end" else OPPOSITE[other[2]]
                    )
                    if (direction in {"left", "right"}) == (axis == "horizontal"):
                        return direction
        return default



    def _structure_contact_side(self, key, boundary) -> str:
        """A shared Parallel rail may expose either lateral measured end."""
        from .composition_geometry import OPPOSITE

        direction = self._composition_direction(key, self.layout.axes[key])
        default = direction if boundary == "end" else OPPOSITE[direction]
        if key[0] != "parallel":
            return default
        for recipe in self.layout.wiring.values():
            for attachment in recipe.group.attachments:
                if attachment.block != key or attachment.boundary != boundary:
                    continue
                target = ("attachment", attachment.key)
                for connection in recipe.connections:
                    other = connection.b if connection.a == target else connection.a if connection.b == target else None
                    if other is not None and other[0] == "arm":
                        return OPPOSITE[other[2]]
        return default


    def _emit_leaf(
        self,
        leaf: _Leaf,
        origin: Point,
        *,
        axis: str = "horizontal",
        label_side: str | None = None,
        direction: str | None = None,
    ) -> _Placed:
        """Emit one physical leaf exactly once at a local measured origin."""
        if leaf.path in self.placed:
            return self.placed[leaf.path]
        if leaf.model == "transmission_line":
            rlgc = self.point.resolved_fields.get((leaf.path, "rlgc"))
            if (
                rlgc is None
                or not hasattr(rlgc, "conductors")
                or not hasattr(rlgc, "reference_conductor")
            ):
                raise _fail(
                    "resolved transmission line has no RLGC value", path=leaf.path
                )
            box = electrical_box(
                kind="CPW" if len(rlgc.conductors) == 1 else "MTL",
                origin=origin,
                orientation={"right": 0, "top": 90, "left": 180, "bottom": 270}[
                    direction or ("right" if axis == "horizontal" else "bottom")
                ],
                title=leaf.path[-1],
                conductors=tuple(rlgc.conductors),
                reference=rlgc.reference_conductor,
                length=self._field_text(leaf, "length"),
                n_sections=leaf.metadata.get("n_sections"),
            )
            self.builder.boxes.append(box)
            anchors = dict(box.anchors)
            # The source model pin vocabulary is head.<conductor>/tail.<conductor>.
            placed = _Placed(
                leaf.path,
                anchors,
                box.bounds,
                Point(box.title.bounds.xmin, box.title.bounds.ymax),
            )
        else:
            kind, field = {
                "resistor": ("R", "resistance"),
                "capacitor": ("C", "capacitance"),
                "inductor": ("L", "inductance"),
                "josephson_junction": ("JJ", "josephson_inductance"),
            }.get(leaf.model, (None, None))
            if kind is None or field is None:
                raise _fail(
                    "structured diagram cannot lower unknown physical model",
                    path=leaf.path,
                    model=leaf.model,
                )
            dx, dy = {
                "right": (1, 0),
                "left": (-1, 0),
                "top": (0, 1),
                "bottom": (0, -1),
            }[direction or ("right" if axis == "horizontal" else "bottom")]
            end = origin.translated(
                dx * DEFAULT_METRICS.native_span, dy * DEFAULT_METRICS.native_span
            )
            symbol = native_block(
                kind,
                start=origin,
                end=end,
                name=leaf.path[-1],
                value=self._field_text(leaf, field),
                label_side=label_side,
            )
            self.builder.symbols.append(symbol)
            placed = _Placed(
                leaf.path,
                dict(symbol.anchors),
                symbol.occupied_bounds,
                Point(symbol.visible_name.bounds.xmin, symbol.visible_name.bounds.ymax),
            )
            # A junction capacitance is a real separately visible physical field.
            if (
                leaf.model == "josephson_junction"
                and "junction_capacitance" in leaf.fields
            ):
                dx, dy = (
                    (0.0, -2 * DEFAULT_METRICS.native_span)
                    if axis == "horizontal"
                    else (2 * DEFAULT_METRICS.native_span, 0.0)
                )
                shunt_start, shunt_end = (
                    origin.translated(dx, dy),
                    end.translated(dx, dy),
                )
                capacitance = native_block(
                    "C",
                    start=shunt_start,
                    end=shunt_end,
                    name=leaf.path[-1],
                    value=self._field_text(leaf, "junction_capacitance"),
                    branch_label="Cj",
                )
                self.builder.symbols.append(capacitance)
                # The parallel element gets contacts back to the same explicit pins.
                self.builder.conductive.extend(
                    (
                        ConductivePolyline((origin, shunt_start)),
                        ConductivePolyline((end, shunt_end)),
                    )
                )
        self.placed[leaf.path] = placed
        for pin, anchor in placed.anchors.items():
            endpoint = {
                "kind": "pin",
                "scope": list(leaf.path[:-1]),
                "component": leaf.path[-1],
                "id": pin,
                "public": False,
            }
            self.endpoint_anchors[_endpoint_key(endpoint)].append(anchor)
        return placed

    def _element_paths(
        self, structure: Mapping[str, object]
    ) -> tuple[tuple[str, ...], ...]:
        def one(rows: object) -> tuple[tuple[str, ...], ...]:
            return tuple(
                _path(
                    _record(row, name="structure.element").get("path"),
                    field="structure.element.path",
                )
                for row in _rows(rows, name="structure.elements")
            )

        kind = _identifier(structure.get("kind"), field="structure.kind")
        if kind == "parallel":
            return tuple(
                path
                for branch in _rows(structure.get("branches"), name="parallel.branches")
                for path in one(branch.get("elements"))
            )
        if kind in {"series", "branch"}:
            return one(structure.get("elements"))
        return ()

    def _occupied(self, *, labels: bool = True) -> list[Bounds]:
        occupied = [item.occupied_bounds for item in self.builder.symbols]
        occupied.extend(item.bounds for item in self.builder.boxes)
        occupied.extend(item.occupied_bounds for item in self.builder.ports)
        # Child contours are rigid obstacles at the parent level.
        occupied.extend(region.bounds for region in self.builder.regions)
        if labels:
            occupied.extend(
                site.visible_label.bounds
                for site in self.builder.boundary_sites
                if site.visible_label is not None
            )
            occupied.extend(
                guide.label.bounds
                for guide in self.builder.guides
                if guide.label is not None
            )
            occupied.extend(run.bounds for run in self.builder.text)
        return occupied

    def _extent(self) -> Bounds:
        occupied = self._occupied()
        occupied.extend(guide.bounds for guide in self.builder.guides)
        occupied.extend(wire.bounds for wire in self.builder.conductive)
        occupied.extend(jump.path.bounds for jump in self.builder.jumps)
        occupied.extend(wire.bounds for _, wire in self.prewired)
        return _bounds(occupied, padding=0.0)

    def _placement_extent(self) -> Bounds:
        """Reserve the measured public rays without adding visible circuitry."""
        corridors = []
        for path, frame in self.scope_bounds.items():
            placed = self.placed.get(path)
            if placed is None:
                continue
            for point in placed.anchors.values():
                dx = -1 if abs(point.x-frame.xmin) <= COORDINATE_TOLERANCE else 1 if abs(point.x-frame.xmax) <= COORDINATE_TOLERANCE else 0
                dy = -1 if abs(point.y-frame.ymin) <= COORDINATE_TOLERANCE else 1 if abs(point.y-frame.ymax) <= COORDINATE_TOLERANCE else 0
                if dx or dy:
                    corridors.append(_inflate(Bounds.around((point, point.translated(dx*DEFAULT_METRICS.terminal_stub,dy*DEFAULT_METRICS.terminal_stub))),DEFAULT_METRICS.obstacle_clearance))
        return _bounds((self._extent(), *corridors), padding=0.0)

    def _peer_order_anchor(self, key) -> Point:
        """Visible body/frame center, separate from its reserved access space."""
        path = key[1] if key[0] == "scope" else (*key[1],key[2])
        if path in self.scope_bounds:
            bounds = self.scope_bounds[path]
        else:
            bounds = _bounds(tuple(symbol.occupied_bounds for symbol in self.builder.symbols)
                             + tuple(box.bounds for box in self.builder.boxes), padding=0.0)
        return Point((bounds.xmin+bounds.xmax)/2,(bounds.ymin+bounds.ymax)/2)

    def _adopt(self, child: StructuredLowerer, dx: float, dy: float) -> None:
        for name, rows in vars(child.builder).items():
            getattr(self.builder, name).extend(
                _translated(item, dx, dy) for item in rows
            )
        for path, placed in child.placed.items():
            self.placed[path] = _Placed(
                path,
                {
                    pin: point.translated(dx, dy)
                    for pin, point in placed.anchors.items()
                },
                placed.bounds.translated(dx, dy),
                placed.identity_anchor.translated(dx, dy),
            )
        for key, points in child.endpoint_anchors.items():
            self.endpoint_anchors[key].extend(
                point.translated(dx, dy) for point in points
            )
        self.scope_bounds.update(
            {
                key: bounds.translated(dx, dy)
                for key, bounds in child.scope_bounds.items()
            }
        )

    def _orient(self, degrees: int) -> None:
        """Change one complete measured body's frame without changing its graph."""
        for rows in vars(self.builder).values():
            rows[:] = [_oriented(row, degrees) for row in rows]
        self.placed = {path: _oriented(placed, degrees) for path, placed in self.placed.items()}
        self.scope_bounds = {path: _oriented(bound, degrees) for path, bound in self.scope_bounds.items()}
        headers = {region.bounds:region.header for region in self.builder.regions
            if region.header is not None}
        for path,bounds in self.scope_bounds.items():
            if bounds in headers and path in self.placed:
                self.placed[path] = replace(self.placed[path],identity_anchor=headers[bounds].origin)
        for points in self.endpoint_anchors.values():
            points[:] = [_oriented(point, degrees) for point in points]
        self.prewired = [(net, _oriented(wire, degrees)) for net, wire in self.prewired]

    def _place_structure(self, scope, structure, ordinal, *, children=None) -> None:
        """Construct one measured ordered chain or shared-rail Parallel body."""
        kind = structure["kind"]
        if kind in {"link", "coupling"}:
            return
        children = {} if children is None else children
        key = (kind, scope, structure["id"])
        axis = self.layout.axes[key]
        side = self.layout.ground_sides.get(key)
        if side is not None:
            axis = "horizontal" if side in {"left", "right"} else "vertical"
        direction = self._composition_direction(key, axis)
        self._structure_axis = axis
        branches = tuple(structure.get("branches", (structure,)))
        if kind == "parallel":
            by_key = {("series", scope, row["id"]): row for row in branches}
            branches = tuple(by_key[item] for item in self.layout.order[key])
        across = 0.0
        for branch_index, branch in enumerate(branches):
            chain = StructuredLowerer(self.point, self.layout, show_values=self.show_values)
            chain.active_scope = scope
            cursor = 0.0
            for member_index, row in enumerate(branch["elements"]):
                path = tuple(row["path"])
                if path in children:
                    body = children[path]
                elif path in self.leaves:
                    body = StructuredLowerer(self.point, self.layout, show_values=self.show_values)
                    body.active_scope = scope
                    body._emit_leaf(self.leaves[path], Point(0.0, 0.0),
                        axis=axis, direction=direction,
                        label_side="right" if axis == "vertical" and len(branches)>1
                        and branch_index == len(branches)-1 else None)
                else:
                    raise _fail("structure member has no measured body", path=path)
                bounds = body._placement_extent()
                start = body.placed[path].anchors[row["pin_1"]]
                if axis == "horizontal":
                    dx = cursor-bounds.xmin if direction == "right" else cursor-bounds.xmax
                    dy = -start.y
                else:
                    dx = -start.x
                    dy = cursor-bounds.ymax if direction == "bottom" else cursor-bounds.ymin
                chain._adopt(body, dx, dy)
                moved = bounds.translated(dx, dy)
                gap = 2 * DEFAULT_METRICS.terminal_stub + DEFAULT_METRICS.obstacle_clearance
                cursor = (moved.xmax+gap if direction == "right" else moved.xmin-gap) if axis == "horizontal" else (moved.ymin-gap if direction == "bottom" else moved.ymax+gap)
            bounds = chain._extent()
            dx,dy = (across-bounds.xmin,0.0) if axis == "vertical" else (0.0,across-bounds.ymax)
            self._adopt(chain,dx,dy)
            across = bounds.xmax+dx+DEFAULT_METRICS.routing_lane_pitch if axis == "vertical" else bounds.ymin+dy-DEFAULT_METRICS.routing_lane_pitch
        self._connect_structure(scope, structure, branches, axis, direction)

    def _connect_structure(self, scope, structure, branches, axis, direction) -> None:
        kind = structure["kind"]
        endpoints = (structure.get("start", structure.get("at")), structure.get("end"))
        self._structure_edges = []
        # Connector strokes are direct consequences of declared ordered
        # members.  Parallel branches share measured rails; short chains gain
        # plain lead length, never altered component geometry or pin order.
        chains = [branch["elements"] for branch in branches]
        starts = [
            self.placed[tuple(rows[0]["path"])].anchors[rows[0]["pin_1"]]
            for rows in chains
        ]
        ends = [
            self.placed[tuple(rows[-1]["path"])].anchors[rows[-1]["pin_2"]]
            for rows in chains
        ]
        for rows in chains:
            for left, right in pairwise(rows):
                a = self.placed[tuple(left["path"])].anchors[left["pin_2"]]
                b = self.placed[tuple(right["path"])].anchors[right["pin_1"]]
                net = self.endpoint_nets[
                    _endpoint_key(
                        {
                            "kind": "pin",
                            "scope": tuple(left["path"])[:-1],
                            "component": left["path"][-1],
                            "id": left["pin_2"],
                            "public": False,
                        }
                    )
                ]
                self._structure_wire(net, a, b)
        for index, (endpoint, points) in enumerate(
            zip(endpoints, (starts, ends), strict=True)
        ):
            local = (
                {**endpoint, "scope": scope}
                if endpoint.get("kind") == "ground"
                else endpoint
            )
            net = (
                "ground"
                if endpoint.get("kind") == "ground"
                else self.endpoint_nets[_endpoint_key(endpoint)]
            )
            if kind == "parallel":
                level = (
                    (
                        max(point.y for point in points)
                        if (index == 0) == (direction == "bottom")
                        else min(point.y for point in points)
                    )
                    if axis == "vertical"
                    else (
                        min(point.x for point in points)
                        if (index == 0) == (direction == "right")
                        else max(point.x for point in points)
                    )
                )
                composite_bounds = [
                    self.placed[member].bounds
                    for member in self._element_paths(structure)
                    if member not in self.leaves or axis == "horizontal"
                ]
                if composite_bounds:
                    margin = DEFAULT_METRICS.terminal_stub
                    if axis == "vertical":
                        level = (
                            max(
                                level,
                                *(bound.ymax + margin for bound in composite_bounds),
                            )
                            if (index == 0) == (direction == "bottom")
                            else min(
                                level,
                                *(bound.ymin - margin for bound in composite_bounds),
                            )
                        )
                    else:
                        level = (
                            min(
                                level,
                                *(bound.xmin - margin for bound in composite_bounds),
                            )
                            if (index == 0) == (direction == "right")
                            else max(
                                level,
                                *(bound.xmax + margin for bound in composite_bounds),
                            )
                        )
                rail = [
                    Point(point.x, level)
                    if axis == "vertical"
                    else Point(level, point.y)
                    for point in points
                ]
                for point, landing in zip(points, rail, strict=True):
                    self._structure_wire(net, point, landing)
                contact = (
                    Point((rail[0].x + rail[-1].x) / 2, level)
                    if endpoint.get("kind") == "ground" and axis == "vertical"
                    else min(rail, key=lambda point: point.y)
                    if endpoint.get("kind") == "ground"
                    else rail[0]
                )
                if endpoint.get("kind") != "ground":
                    side = self._structure_contact_side((kind,scope,structure["id"]), "start" if index == 0 else "end")
                    measured = self._extent()
                    # This is the existing shared rail extended past measured
                    # member ink, not a new endpoint or a parent-side relocation.
                    if axis == "vertical" and side in {"left","right"}:
                        contact = Point(measured.xmin-DEFAULT_METRICS.terminal_stub if side == "left" else measured.xmax+DEFAULT_METRICS.terminal_stub, level)
                    elif axis == "horizontal" and side in {"top","bottom"}:
                        contact = Point(level, measured.ymax+DEFAULT_METRICS.terminal_stub if side == "top" else measured.ymin-DEFAULT_METRICS.terminal_stub)
                rail = sorted({*rail, contact}, key=lambda point: (point.x, point.y))
                for a, b in pairwise(rail):
                    self._structure_wire(net, a, b)
            else:
                contact = points[0]
                member = tuple(chains[0][0 if index == 0 else -1]["path"])
                if member not in self.leaves:
                    measured = self._extent()
                    outward = direction if index else {"left":"right","right":"left","top":"bottom","bottom":"top"}[direction]
                    margin = DEFAULT_METRICS.routing_lane_pitch
                    exterior = Point(measured.xmin-margin if outward=="left" else measured.xmax+margin, contact.y) if outward in {"left","right"} else Point(contact.x, measured.ymin-margin if outward=="bottom" else measured.ymax+margin)
                    self._structure_wire(net,contact,exterior)
                    contact = exterior
            self.endpoint_anchors[_endpoint_key(local)].append(contact)
        self._route_structure_edges()

    def _structure_wire(self, net: str, a: Point, b: Point) -> None:
        """Complete each authored internal edge; an obstructed edge is not omitted."""
        if a != b:
            self._structure_edges.append((net, a, b))

    def _route_structure_edges(self) -> None:
        """Emit the arithmetic channels reserved between ordered measured members."""
        axis = self._structure_axis
        for net, a, b in self._structure_edges:
            if a.x == b.x or a.y == b.y:
                points = (a,b)
            elif axis == "horizontal":
                middle = (a.x+b.x)/2
                points = (a,Point(middle,a.y),Point(middle,b.y),b)
            else:
                middle = (a.y+b.y)/2
                points = (a,Point(a.x,middle),Point(b.x,middle),b)
            points = _compress(points)
            if not self._prepared_path_clear(points):
                raise _fail("fixed structure channel intersects measured ink",
                    scope=self.active_scope, net=net, points=points)
            self.prewired.extend((net,ConductivePolyline((left,right)))
                for left,right in pairwise(points) if left != right)


    def _prepared_path_clear(self, points: Sequence[Point]) -> bool:
        # A rigid child's contour is not a common rail. Its prescribed
        # connector must meet the boundary outward, never run along the frame.
        for a, b in pairwise(points):
            if not _segment_clear(a,b,tuple(label.bounds for symbol in self.builder.symbols for label in (symbol.visible_name,symbol.value,symbol.branch_label) if label is not None)):
                return False
            for symbol in self.builder.symbols:
                first, second = (point for _, point in symbol.anchors)
                bounds = symbol.symbol_bounds
                stub = DEFAULT_METRICS.terminal_stub
                # A shortcut may meet terminal leads, but may not run through
                # native branch-body ink even when that ink is on its bounds'
                # edge (an unmarked horizontal inductor is one such case).
                core = (
                    Bounds(
                        bounds.xmin + stub, bounds.ymin, bounds.xmax - stub, bounds.ymax
                    )
                    if abs(first.y - second.y) <= COORDINATE_TOLERANCE
                    else Bounds(
                        bounds.xmin, bounds.ymin + stub, bounds.xmax, bounds.ymax - stub
                    )
                )
                if not _segment_clear(
                    a, b, (_inflate(core, DEFAULT_METRICS.obstacle_clearance),)
                ):
                    return False
            for region in self.builder.regions:
                bounds = region.bounds
                if not _segment_clear(a, b, (bounds,)):
                    return False
                if (
                    abs(a.x - b.x) <= COORDINATE_TOLERANCE
                    and min(abs(a.x - bounds.xmin), abs(a.x - bounds.xmax))
                    <= COORDINATE_TOLERANCE
                    and min(a.y, b.y) < bounds.ymax - COORDINATE_TOLERANCE
                    and max(a.y, b.y) > bounds.ymin + COORDINATE_TOLERANCE
                ) or (
                    abs(a.y - b.y) <= COORDINATE_TOLERANCE
                    and min(abs(a.y - bounds.ymin), abs(a.y - bounds.ymax))
                    <= COORDINATE_TOLERANCE
                    and min(a.x, b.x) < bounds.xmax - COORDINATE_TOLERANCE
                    and max(a.x, b.x) > bounds.xmin + COORDINATE_TOLERANCE
                ):
                    return False
        return True





    def _contact_mark(self, text, at, *, role, side="top") -> TextRun:
        """Shape a caption in its fixed, previously measured contact slot."""
        orientation = self._scope_orientation(self.active_scope)
        at = at.rotated(orientation)
        side = self._rotate_side(side, orientation)
        run = shape_text(text, at=Point(0.0,0.0),
            size=DEFAULT_METRICS.tertiary_text_size, role=role)
        width,height = run.bounds.xmax-run.bounds.xmin,run.bounds.ymax-run.bounds.ymin
        gap = DEFAULT_METRICS.label_clearance
        x,y = {
            "left": (at.x-gap-width, at.y+gap),
            "right": (at.x+gap, at.y+gap),
            "top": (at.x+gap, at.y+gap),
            "bottom": (at.x+gap, at.y-gap-height),
        }[side]
        return _rigid_text(_translated(run,x-run.bounds.xmin,y-run.bounds.ymin),
            -orientation)


    def _coupling_label(self, row) -> TextRun:
        label = "COUPLING:" + row["id"]
        if self.show_values:
            label += "; k=" + format_envelope(row["coupling_coefficient"])
        else:
            from .._canonical import float64_from_hex
            coefficient = float64_from_hex(row["coupling_coefficient"]["si_value_f64"])
            label += "; k" + ("<0" if coefficient < 0 else ">0" if coefficient > 0 else "=0")
        return shape_text(label,at=Point(0.0,0.0),
            size=DEFAULT_METRICS.tertiary_text_size,role="structure-id")

    def _emit_couplings(self, scope) -> None:
        """Draw signed mutual guides in preallocated owner-local caption strips."""
        gap = DEFAULT_METRICS.label_clearance
        frame = self.scope_bounds[tuple(scope["path"])]
        top = frame.ymax-getattr(self,"_header_height",0.0)-gap
        for row in scope["structures"]:
            if row["kind"] != "coupling":
                continue
            dots = []
            for key in ("inductor_a","inductor_b"):
                placed = self.placed[tuple(row[key]["path"])]
                terminal,other = placed.anchors["terminal_1"],placed.anchors["terminal_2"]
                symbol = next(symbol for symbol in self.builder.symbols
                    if symbol.kind in {"L","JJ"} and dict(symbol.anchors)["terminal_1"] == terminal)
                dots.append(winding_dot(terminal,other,symbol.symbol_bounds))
            a,b = (center for center,_ in dots)
            caption = self._coupling_label(row)
            height = caption.bounds.ymax-caption.bounds.ymin
            caption = _translated(caption,frame.xmin+gap-caption.bounds.xmin,
                top-caption.bounds.ymax)
            level = top-height-gap
            points = _compress((a,Point(a.x,level),Point(b.x,level),b))
            if len(points)<2:
                raise _fail("mutual guide has coincident physical terminals",coupling=row["id"])
            self.builder.guides.append(GuideMark("coupling",
                (Path(points,"coupling"),*(dot for _,dot in dots)),caption,(a,b)))
            top = level-gap



    def _validate_visible_layout(self) -> None:
        """Reject a completed scene whose visible evidence is illegible.

        This final check includes every translated child header and public-pin caption. It
        neither repacks a child nor guesses a new electrical path.
        """
        labels = (
            [
                ("guide", guide.label)
                for guide in self.builder.guides
                if guide.label is not None
            ]
            + [
                ("boundary", site.visible_label)
                for site in self.builder.boundary_sites
                if site.visible_label is not None
            ]
            + [
                ("region", region.header)
                for region in self.builder.regions
                if region.header is not None
            ]
        )
        physical = [symbol.occupied_bounds for symbol in self.builder.symbols]
        physical.extend(box.bounds for box in self.builder.boxes)
        physical.extend(port.occupied_bounds for port in self.builder.ports)
        physical.extend(
            label.bounds for port in self.builder.ports for label in port.labels
        )
        for role, label in labels:
            assert label is not None
            if any(
                label.bounds.overlaps(
                    bounds,
                    clearance=DEFAULT_METRICS.label_clearance - COORDINATE_TOLERANCE,
                )
                for bounds in physical
            ):
                raise _fail(
                    "completed scene has a visible label over physical ink",
                    role=role,
                    label=label.text,
                )
        for index, (left_role, left) in enumerate(labels):
            assert left is not None
            for right_role, right in labels[index + 1 :]:
                assert right is not None
                if left.bounds.overlaps(
                    right.bounds,
                    clearance=DEFAULT_METRICS.label_clearance - COORDINATE_TOLERANCE,
                ):
                    raise _fail(
                        "completed scene has overlapping visible labels",
                        left_role=left_role,
                        left=left.text,
                        right_role=right_role,
                        right=right.text,
                    )

    def lower(self):
        from .composition_model import CapturedComposition

        if not isinstance(self.layout, CapturedComposition):
            raise _fail("authoring lowering requires a complete captured composition")
        self._compose_scope(self._scope_records()[()])
        self._emit_junctions()
        self._validate_visible_layout()
        return self.builder.freeze()

    def selected_tap_orders(self):
        """Actual strict local contact progression for editable, unambiguous buses.

        This completes the presentation recipe only. The scene carries no
        placement answer, and independent conformance still observes its ink.
        """
        selected = {}
        for bus, taps in self.layout.inventory.taps.items():
            if len(taps) < 2 or not self.layout.inventory.blocks[("scope",bus[1])].addressable:
                continue
            positions = []
            for tap in taps:
                key = _endpoint_key({"kind":"tap","scope":tap[1],"bus":tap[2],"id":tap[3]})
                points = self.endpoint_anchors.get(key, ())
                if len(points) != 1:
                    break
                at = points[0]
                value = at.x if self.layout.axes[("scope",bus[1])] == "horizontal" else -at.y
                positions.append((value,tap))
            if len(positions) != len(taps):
                continue
            ordered = sorted(positions)
            if any(right[0]-left[0] <= COORDINATE_TOLERANCE for left,right in pairwise(ordered)):
                continue
            selected[bus] = tuple(tap for _,tap in ordered)
        return selected

    def _body_orientation(self, path: tuple[str, ...]) -> int:
        if not path:
            return 0
        key = ("component", path[:-1], path[-1])
        block = self.layout.inventory.blocks.get(key)
        if block is None or self.layout.axes[key] == block.default_axis:
            return 0
        return 270 if self.layout.axes[key] == "vertical" else 90

    def _scope_orientation(self, path: tuple[str, ...]) -> int:
        """Total later rigid rotations, including containing built bodies."""
        return sum(self._body_orientation(path[:depth])
            for depth in range(1, len(path)+1)) % 360

    @staticmethod
    def _rotate_side(side: str, degrees: int) -> str:
        sides = ("right", "top", "left", "bottom")
        return sides[(sides.index(side) + degrees // 90) % 4]

    def _composition_boundary_side(self, endpoint: Mapping[str, object]) -> str:
        from .composition_geometry import OPPOSITE
        from .composition_model import endpoint_key

        key = endpoint_key(endpoint)
        child_path = tuple(endpoint["scope"])
        keys = {key}
        if child_path in self.occurrences:
            keys.add(("pin",child_path[:-1],child_path[-1],endpoint["id"],False))
        orientation = self._body_orientation(child_path)
        def local(side, alias):
            return self._rotate_side(side, -orientation) if alias != key else side
        explicit_values = {local(self.layout.terminal_sides[item], item) for item in keys if item in self.layout.terminal_sides}
        if len(explicit_values)>1:
            raise _fail("one exposed physical pin has conflicting selected sides", pin=key)
        explicit = next(iter(explicit_values),None)
        explicit = explicit or next((local(self.layout.ground_sides[item], item) for item in keys if item in self.layout.ground_sides),None)
        if explicit is not None:
            return explicit
        for recipe in self.layout.wiring.values():
            if recipe.group.scope != child_path[:-1]:
                continue
            for attachment in recipe.group.attachments:
                if endpoint_key(attachment.endpoint) not in keys:
                    continue
                target = ("attachment", attachment.key)
                for connection in recipe.connections:
                    other = (
                        connection.b
                        if connection.a == target
                        else connection.a
                        if connection.b == target
                        else None
                    )
                    if other is not None and other[0] == "arm":
                        return self._rotate_side(OPPOSITE[other[2]], -orientation)
        for recipe in self.layout.wiring.values():
            if recipe.group.scope != child_path:
                continue
            for attachment in recipe.group.attachments:
                if attachment.kind != "scope_pin" or endpoint_key(attachment.endpoint) != key:
                    continue
                target = ("attachment", attachment.key)
                for connection in recipe.connections:
                    other = connection.b if connection.a == target else connection.a if connection.b == target else None
                    if other is not None and other[0] == "arm":
                        return other[2]
        return "left"

    def _compose_scope(self, scope) -> None:
        """Measure each child once, then translate it into one fixed composition."""
        path = tuple(scope["path"])
        self.active_scope = path
        records = {tuple(child["path"]):child for child in scope["children"]}
        records.update({tuple(row["body"]["path"]):row["body"] for row in scope["component_bodies"]})
        children = {}
        for child_path,record in records.items():
            child = StructuredLowerer(self.point,self.layout,show_values=self.show_values)
            child._compose_scope(record)
            orientation = self._body_orientation(child_path)
            if orientation:
                child._orient(orientation)
            child._validate_visible_layout()
            children[child_path] = child
        self._compose_with_children(scope, children)

    def _compose_with_children(self, scope, selected_children):
        from .composition_geometry import (
            OPPOSITE,
            BlockGeometry,
            BoundaryContactGeometry,
            ContactGeometry,
            place_composition,
        )
        from .composition_model import endpoint_key

        path = tuple(scope["path"])
        structures = tuple(row for row in scope["structures"] if row["kind"] in {"series","parallel","branch"})
        children = selected_children
        fragments: dict[tuple[object, ...], StructuredLowerer] = {}
        boundaries: dict[tuple[tuple[object, ...], str], Point] = {}
        member_owners = {}
        for row in structures:
            key = (row["kind"], path, row["id"])
            fragment = StructuredLowerer(
                self.point, self.layout, show_values=self.show_values
            )
            fragment.active_scope = path
            for member in self._element_paths(row):
                member_owners[member] = key
            fragment._place_structure(path, row, 0, children=children)
            branches = row.get("branches", (row,))
            for boundary, member, pin in (
                (
                    "at" if row["kind"] == "branch" else "start",
                    branches[0]["elements"][0],
                    "pin_1",
                ),
                ("end", branches[-1]["elements"][-1], "pin_2"),
            ):
                endpoint = row[boundary]
                contacts = fragment.endpoint_anchors.get(_endpoint_key(endpoint), ())
                anchor = (
                    contacts[0]
                    if contacts
                    else fragment.placed[tuple(member["path"])].anchors[member[pin]]
                )
                boundaries[key, boundary] = anchor
            for target in self.layout.inventory.ground_targets.values():
                if target.block != key:
                    continue
                anchor = boundaries[key, target.boundary]
                side = self.layout.ground_sides.get(key) or self._composition_direction(
                    key, self.layout.axes[key]
                )
                fragment.builder.guides.append(ground_mark(anchor, side=side))
            fragment.builder.conductive.extend(wire for _, wire in fragment.prewired)
            fragment.prewired.clear()
            fragments[key] = fragment

        for child_path, child in children.items():
            if child_path in member_owners:
                continue
            key = (
                ("component", path, child_path[-1])
                if child_path in self.occurrences
                else ("scope", child_path)
            )
            fragments[key] = child
        for leaf in self.leaves.values():
            if leaf.path[:-1] != path or leaf.path in member_owners:
                continue
            key = ("component", path, leaf.path[-1])
            fragment = StructuredLowerer(
                self.point, self.layout, show_values=self.show_values
            )
            fragment.active_scope = path
            fragment._emit_leaf(leaf, Point(0.0, 0.0), axis=self.layout.axes[key])
            fragments[key] = fragment

        recipes = tuple(
            recipe
            for recipe in self.layout.wiring.values()
            if recipe.group.scope == path
        )
        attachments = {}
        for target in self.layout.inventory.ground_targets.values():
            if target.scope != path or target.block is not None:
                continue
            endpoint = target.endpoint
            member_path = tuple(endpoint["scope"]) if endpoint.get("public") else (*endpoint["scope"], endpoint["component"])
            owner = member_owners.get(member_path, ("component", path, member_path[-1]) if member_path in self.occurrences else ("scope", member_path))
            fragment = fragments[owner]
            member = fragment.placed[member_path]
            anchor = member.anchors[endpoint["id"]]
            side = self.layout.ground_sides.get(target.key) or (
                "left" if abs(anchor.x - member.bounds.xmin) <= COORDINATE_TOLERANCE else
                "right" if abs(anchor.x - member.bounds.xmax) <= COORDINATE_TOLERANCE else
                "bottom" if abs(anchor.y - member.bounds.ymin) <= COORDINATE_TOLERANCE else "top"
            )
            dx, dy = {"left": (-1,0), "right": (1,0), "top": (0,1), "bottom": (0,-1)}[side]
            ground_at = anchor.translated(dx * DEFAULT_METRICS.terminal_stub, dy * DEFAULT_METRICS.terminal_stub)
            glyph = ground_mark(ground_at, side=side)
            if any(glyph.bounds.overlaps(bound, clearance=DEFAULT_METRICS.label_clearance - COORDINATE_TOLERANCE) for bound in fragment._occupied()):
                raise _fail("parent ground outward placement intersects another measured Block", endpoint=endpoint)
            fragment.builder.conductive.append(ConductivePolyline((anchor,ground_at)))
            fragment.builder.guides.append(glyph)
        blocks = {
            key: BlockGeometry(key, fragment._placement_extent(),
                order_anchor=fragment._peer_order_anchor(key) if key[0] in {"scope","component"} else None,
                occupied_bounds=fragment._extent())
            for key, fragment in fragments.items()
        }
        scope_pin_keys = {}
        for boundary in self.layout.inventory.scope_boundaries.values():
            if boundary.scope != path:
                continue
            key = ("scope_boundary", path, boundary.endpoint["id"])
            scope_pin_keys[key] = boundary
            blocks[key] = BlockGeometry(key, Bounds(0.0, 0.0, 0.0, 0.0))
            side = self._composition_boundary_side(boundary.endpoint)
            attachments[boundary.key] = ContactGeometry(key, Point(0.0, 0.0), OPPOSITE[side])
        for recipe in recipes:
            for attachment in recipe.group.attachments:
                key = attachment.block
                if attachment.kind == "structure":
                    side = self._structure_contact_side(key, attachment.boundary)
                    attachments[attachment.key] = ContactGeometry(
                        key, boundaries[key, attachment.boundary], side
                    )
                elif attachment.kind == "scope_pin":
                    side = self._composition_boundary_side(attachment.endpoint)
                    key = ("scope_boundary", path, attachment.endpoint["id"])
                    blocks[key] = BlockGeometry(key, Bounds(0.0, 0.0, 0.0, 0.0))
                    attachments[attachment.key] = ContactGeometry(
                        key, Point(0.0, 0.0), OPPOSITE[side]
                    )
                    scope_pin_keys[key] = attachment
                elif attachment.kind == "port":
                    key = attachment.key
                    row = next(
                        row
                        for row in self.semantic["connectivity"]["ports"]
                        if row["id"] == attachment.endpoint["id"]
                    )
                    side = self.layout.port_sides.get(key, "left")
                    fragment = StructuredLowerer(
                        self.point, self.layout, show_values=self.show_values
                    )
                    fragment.active_scope = path
                    port = port_block(
                        port_id=row["id"],
                        role=row["role"],
                        reference_impedance=format_envelope(row["reference_impedance"]),
                        boundary_anchor=Point(0.0, 0.0),
                        side=side,
                        load_side=self.layout.port_load_sides[key],
                    )
                    fragment.builder.ports.append(port)
                    fragments[key] = fragment
                    blocks[key] = BlockGeometry(key, port.occupied_bounds)
                    attachments[attachment.key] = ContactGeometry(
                        key, port.external_anchor, OPPOSITE[side]
                    )
                else:
                    member_path = (
                        tuple(key[1]) if key[0] == "scope" else (*key[1], key[2])
                    )
                    owner = member_owners.get(member_path, key)
                    fragment = fragments[owner]
                    anchor = fragment.placed[member_path].anchors[
                        attachment.endpoint["id"]
                    ]
                    bounds = fragment.scope_bounds.get(
                        member_path, fragment.placed[member_path].bounds
                    )
                    side = self.layout.terminal_sides.get(
                        endpoint_key(attachment.endpoint)
                    ) or (
                        "left"
                        if abs(anchor.x - bounds.xmin) <= COORDINATE_TOLERANCE
                        else "right"
                        if abs(anchor.x - bounds.xmax) <= COORDINATE_TOLERANCE
                        else "bottom"
                        if abs(anchor.y - bounds.ymin) <= COORDINATE_TOLERANCE
                        else "top"
                    )
                    attachments[attachment.key] = ContactGeometry(owner, anchor, side)

        tap_sites = {}
        contact_order = []
        for bus_key, taps in self.layout.inventory.taps.items():
            if bus_key[1] != path:
                continue
            for tap_key in taps:
                aliases = tuple(key for group, key in self.layout.inventory.endpoint_aliases.get(tap_key, ()) if group[0] == path and key in attachments)
                if len(aliases) == 1:
                    tap_sites[tap_key] = aliases[0]
            if bus_key in self.layout.tap_order:
                ordered = self.layout.tap_order[bus_key]
                if any(tap not in tap_sites for tap in ordered):
                    raise _fail("named tap order requires one unambiguous actual attachment per tap", bus=bus_key, taps=ordered)
                contact_order.append(tuple(tap_sites[tap] for tap in ordered))
        peer_order = self.layout.order[("scope", path)]
        boundary_rows = []
        boundary_labels = {}
        boundary_tap_labels = defaultdict(list)
        for key, attachment in scope_pin_keys.items():
            side = self._composition_boundary_side(attachment.endpoint)
            label = None
            if len(scope["exposures"]["pins"]) > 1:
                label = self._contact_mark(attachment.endpoint["id"],Point(0.0,0.0),
                    role="public-pin",side=OPPOSITE[side])
            boundary_labels[key] = label
        # Named labels are measured decorations of their contact's owning fragment.
        # They reserve a fixed exterior caption strip before any parent placement.
        for ordinal,(tap,contact_key) in enumerate(tap_sites.items()):
            contact = attachments[contact_key]
            if contact.block in scope_pin_keys:
                key = contact.block
                side = self._composition_boundary_side(scope_pin_keys[key].endpoint)
                label = self._contact_mark(tap[-1],Point(0.0,0.0),role="tap-id",side=OPPOSITE[side])
                previous = [*(() if boundary_labels[key] is None else (boundary_labels[key],)),*boundary_tap_labels[key]]
                if previous:
                    gap = DEFAULT_METRICS.label_clearance
                    dy = min(run.bounds.ymin for run in previous)-gap-label.bounds.ymax if side == "top" else max(run.bounds.ymax for run in previous)+gap-label.bounds.ymin
                    label = _translated(label,0.0,dy)
                boundary_tap_labels[key].append(label)
                continue
            fragment = fragments[contact.block]
            extent = fragment._extent()
            label = self._contact_mark(tap[-1],Point(contact.point.x,extent.ymax),
                role="tap-id")
            fragment.builder.text.append(label)
        for key,attachment in scope_pin_keys.items():
            labels = (*(() if boundary_labels[key] is None else (boundary_labels[key],)),*boundary_tap_labels[key])
            boundary_rows.append(BoundaryContactGeometry(key,
                self._composition_boundary_side(attachment.endpoint),
                _bounds(tuple(run.bounds for run in labels),padding=0.0) if labels else None))
        for key,fragment in fragments.items():
            blocks[key] = BlockGeometry(key,fragment._placement_extent(),
                order_anchor=fragment._peer_order_anchor(key) if key[0] in {"scope","component"} else None,
                occupied_bounds=fragment._extent())
        gap = DEFAULT_METRICS.label_clearance
        header = shape_text(path[-1],at=Point(0.0,0.0),
            size=1.25*DEFAULT_METRICS.primary_text_size,role="region-id") if path else None
        header_height = 0.0 if header is None else header.bounds.ymax-header.bounds.ymin+2*gap
        coupling_labels = tuple(self._coupling_label(row) for row in scope["structures"] if row["kind"] == "coupling")
        coupling_height = sum(label.bounds.ymax-label.bounds.ymin+3*gap for label in coupling_labels)
        text_width = max((label.bounds.xmax-label.bounds.xmin for label in
            (*(() if header is None else (header,)),*coupling_labels)), default=0.0)
        body_width = max((block.bounds.xmax-block.bounds.xmin for block in blocks.values()),default=0.0)
        width_inset = gap+max(0.0,text_width-body_width)/2
        orientation = self._scope_orientation(path)
        insets = {"left":width_inset,"top":coupling_height+gap,
            "right":width_inset,"bottom":gap}
        title_side = self._rotate_side("top", -orientation)
        if title_side == "top":
            insets[title_side] += header_height
        else:
            insets[title_side] = max(insets[title_side], header_height+gap)
        if orientation % 180:
            body_height = max((block.bounds.ymax-block.bounds.ymin for block in blocks.values()),default=0.0)
            title_width_inset = gap+max(0.0,text_width-body_height)/2
            for side in ("top", "bottom"):
                insets[side] = max(insets[side], title_width_inset)
        self._header_height = header_height
        placement = place_composition(
            blocks, attachments, recipes, peer_order=peer_order,
            axis=self.layout.axes[("scope",path)],contact_order=tuple(contact_order),
            boundary_contacts=tuple(boundary_rows),
            frame_insets=tuple(insets[side] for side in ("left","top","right","bottom")),
            header_bounds=None if header is None else header.bounds,
            header_rotation=orientation)
        self._complete_scope(scope,fragments,attachments,scope_pin_keys,tap_sites,
            placement,boundary_labels,boundary_tap_labels,header)

    def _complete_scope(self, scope, fragments, attachments, scope_pin_keys,
                        tap_sites, placement, boundary_labels, boundary_tap_labels, header) -> None:
        """Emit the one completed frame and prescribed routes without relocating ink."""
        from .composition_geometry import (
            OPPOSITE,
            ContactGeometry,
            RoutingContext,
            route_composition,
        )
        path = tuple(scope["path"])
        self.active_scope = path
        for key,fragment in fragments.items():
            origin = placement.origins[key]
            self._adopt(fragment,origin.x,origin.y)
        self.builder.conductive.extend(placement.primitives)
        bounds = placement.bounds
        self.scope_bounds[path] = bounds
        public_anchors = {}
        replacements = {}
        for key,attachment in scope_pin_keys.items():
            anchor = placement.origins[key]
            side = self._composition_boundary_side(attachment.endpoint)
            public_anchors[attachment.endpoint["id"]] = anchor
            replacements[key] = ContactGeometry(key,anchor,OPPOSITE[side])
            self.endpoint_anchors[_endpoint_key(attachment.endpoint)].append(anchor)
            label = boundary_labels[key]
            if label is not None:
                label = _translated(label,anchor.x,anchor.y)
            self.builder.boundary_sites.append(BoundarySite(anchor,label))
            self.builder.text.extend(_translated(run,anchor.x,anchor.y)
                for run in boundary_tap_labels[key])
        if header is not None:
            gap = DEFAULT_METRICS.label_clearance
            orientation = self._scope_orientation(path)
            final_frame = _oriented(bounds, orientation)
            header = _rigid_text(_translated(header,
                final_frame.xmin+gap-header.bounds.xmin,
                final_frame.ymax-gap-header.bounds.ymax), -orientation)
        obstacles = [fragment._extent().translated(placement.origins[key].x,placement.origins[key].y)
            for key,fragment in fragments.items()]
        obstacles.extend(site.visible_label.bounds for site in self.builder.boundary_sites if site.visible_label is not None)
        obstacles.extend(run.bounds for run in self.builder.text)
        if header is not None:
            obstacles.append(header.bounds)
        route = route_composition(placement,context=RoutingContext(tuple(obstacles),bounds,path))
        self.builder.conductive.extend(route.conductive)
        self.builder.jumps.extend(route.jumps)
        self._finish_scope(scope,attachments,scope_pin_keys,tap_sites,placement,
            replacements,public_anchors,bounds,header)

    def _finish_scope(self, scope, attachments, scope_pin_keys, tap_sites,
                      placement, replacements, public_anchors, bounds, header) -> None:
        from .composition_geometry import VECTORS

        path = tuple(scope["path"])
        for attachment in scope_pin_keys.values():
            pin_id = attachment.endpoint["id"]
            inward_connected = any(
                recipe.group.scope == path and any(contact.key == attachment.key
                    for contact in recipe.group.attachments)
                for recipe in self.layout.wiring.values())
            if not inward_connected:
                at = public_anchors[pin_id]
                dx,dy = VECTORS[self._composition_boundary_side(attachment.endpoint)]
                self.builder.conductive.append(ConductivePolyline((
                    at.translated(-dx*DEFAULT_METRICS.terminal_stub,-dy*DEFAULT_METRICS.terminal_stub),at)))
            connected = any(
                recipe.group.scope == path[:-1] and any(
                    contact.kind in {"pin","scope_pin"} and contact.endpoint["id"] == pin_id and (
                        tuple(contact.endpoint["scope"]) == path if contact.endpoint.get("public") else (*contact.endpoint["scope"],contact.endpoint.get("component")) == path
                    ) for contact in recipe.group.attachments
                ) for recipe in self.layout.wiring.values()
            )
            parent = self._scope_records().get(path[:-1])
            if parent is not None:
                connected = connected or any(tuple(member["path"]) == path and pin_id in (member["pin_1"],member["pin_2"]) for structure in parent["structures"] for branch in structure.get("branches",(structure,)) for member in branch.get("elements",()))
            connected = connected or any(target.block is None and target.endpoint["id"] == pin_id and (tuple(target.endpoint["scope"]) if target.endpoint.get("public") else (*target.endpoint["scope"],target.endpoint["component"])) == path for target in self.layout.inventory.ground_targets.values())
            if not connected:
                at=public_anchors[pin_id]
                dx,dy=VECTORS[self._composition_boundary_side(attachment.endpoint)]
                self.builder.conductive.append(ConductivePolyline((at,at.translated(dx*DEFAULT_METRICS.terminal_stub,dy*DEFAULT_METRICS.terminal_stub))))
        actual_tap_sites = {}
        for tap_key, attachment_key in tap_sites.items():
            contact = attachments[attachment_key]
            origin = placement.origins[contact.block]
            at = replacements[contact.block].point if contact.block in replacements else contact.point.translated(origin.x, origin.y)
            actual_tap_sites[tap_key] = at
            self.endpoint_anchors[_endpoint_key({"kind":"tap", "scope":path, "bus":tap_key[2], "id":tap_key[3]})] = [at]
        for bus_key, taps in self.layout.tap_order.items():
            if bus_key[1] != path:
                continue
            positions = tuple(actual_tap_sites[tap].x if self.layout.axes[("scope",path)] == "horizontal" else -actual_tap_sites[tap].y for tap in taps)
            if any(right-left <= COORDINATE_TOLERANCE for left,right in pairwise(positions)):
                raise _fail("projected visible named-tap positions do not realize the requested order", bus=bus_key, positions=positions)
        self._emit_couplings(scope)
        if not path:
            bounds = _bounds((bounds, self._extent()), padding=DEFAULT_METRICS.label_clearance)
            self.scope_bounds[path] = bounds
        self._append_composed_region(path, bounds, header)
        self.placed[path] = _Placed(
            path,
            public_anchors,
            bounds,
            header.origin if header is not None else Point(bounds.xmin, bounds.ymax),
        )

    def _append_composed_region(
        self, path: tuple[str, ...], bounds: Bounds, header: TextRun | None
    ) -> None:
        r = DEFAULT_METRICS.label_clearance
        x0, y0, x1, y1 = bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax
        outline = Path(
            (
                Point(x0 + r, y0),
                Point(x1 - r, y0),
                Point(x1, y0),
                Point(x1, y0),
                Point(x1, y0 + r),
                Point(x1, y1 - r),
                Point(x1, y1),
                Point(x1, y1),
                Point(x1 - r, y1),
                Point(x0 + r, y1),
                Point(x0, y1),
                Point(x0, y1),
                Point(x0, y1 - r),
                Point(x0, y0 + r),
                Point(x0, y0),
                Point(x0, y0),
                Point(x0 + r, y0),
            ),
            "region",
            True,
            (1, 2, 4, 4, 4, 2, 4, 4, 4, 2, 4, 4, 4, 2, 4, 4, 4),
        )
        self.builder.regions.append(
            SubsystemRegion(
                bounds,
                outline,
                header,
                header.origin
                if header is not None
                else Point(bounds.xmin, bounds.ymax),
                "composite" if path else "root",
            )
        )

    def _emit_junctions(self) -> None:
        """Mark actual three-way conductive incidence, without semantic tags."""
        segments = [
            segment
            for wire in self.builder.conductive
            for segment in pairwise(wire.points)
        ]
        for port in self.builder.ports:
            segments.extend(
                segment
                for path in port.paths
                if path.role == "wire"
                for segment in pairwise(path.points)
            )
        candidates = {point for segment in segments for point in segment}
        for symbol in self.builder.symbols:
            anchors = tuple(point for _, point in symbol.anchors)
            candidates.update(anchors)
            for anchor in anchors:
                for path in symbol.paths:
                    for endpoint, neighbor in (
                        (path.points[0], path.points[1]),
                        (path.points[-1], path.points[-2]),
                    ):
                        if (
                            abs(anchor.x - endpoint.x) <= COORDINATE_TOLERANCE
                            and abs(anchor.y - endpoint.y) <= COORDINATE_TOLERANCE
                        ):
                            segments.append((endpoint, neighbor))
        for guide in self.builder.guides:
            if guide.kind == "ground":
                candidates.update(guide.terminals)
                segments.extend(
                    segment for path in guide.paths for segment in pairwise(path.points)
                )
        # Deduplicate the actual contacts before any visibility-grid rounding:
        # two contacts within tolerance can round to adjacent grid coordinates.
        unique: list[Point] = []
        for point in sorted(candidates, key=lambda item: (item.x, item.y)):
            if any(
                abs(point.x - existing.x) <= COORDINATE_TOLERANCE
                and abs(point.y - existing.y) <= COORDINATE_TOLERANCE
                for existing in unique
            ):
                continue
            unique.append(point)
        for point in unique:
            directions = set()
            for a, b in segments:
                if not _on_segment(point, a, b):
                    continue
                for other in (a, b):
                    dx, dy = other.x - point.x, other.y - point.y
                    if abs(dx) > COORDINATE_TOLERANCE:
                        directions.add((1 if dx > 0 else -1, 0))
                    if abs(dy) > COORDINATE_TOLERANCE:
                        directions.add((0, 1 if dy > 0 else -1))
            if len(directions) >= 3:
                self.builder.node_marks.append(junction_mark(point))


def lower_authoring(point: ResolvedPlanPoint, layout: object, *, show_values: bool):
    """Lower one resolved authoring point to visible immutable scene facts."""
    return StructuredLowerer(point, layout, show_values=show_values).lower()


__all__ = ["StructuredLowerer", "lower_authoring"]
