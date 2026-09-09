"""Independent reconstruction of structured V2 facts from visible scene ink.

The routines in this module never consume renderer correlation keys, placement
records, captured net names, or an expected manifest.  Full identities are
recovered from local visible labels plus measured region containment.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import cast

from .._canonical import canonical_json_bytes
from ..errors import SCNSimValidationError
from .scene import (
    COORDINATE_TOLERANCE,
    Bounds,
    BoundarySite,
    GuideMark,
    NativeSymbol,
    NeutralScene,
    Point,
    SubsystemRegion,
    TextRun,
)
from .values import parse_display_scalar

_Path = tuple[str, ...]


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_audit", evidence=evidence)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _token(kind: str, **fields: object) -> str:
    return canonical_json_bytes({"kind": kind, **fields}).decode("utf-8")


def _contact(*, path: _Path | None = None, pin: str | None = None, port: str | None = None) -> str:
    if port is not None:
        return _token("visible_port_contact", port_id=port)
    assert path is not None and pin is not None
    return _token("visible_physical_terminal", component_path=list(path), pin_id=pin)


def _same(left: float, right: float) -> bool:
    return abs(left - right) <= COORDINATE_TOLERANCE


def _same_point(left: Point, right: Point) -> bool:
    return _same(left.x, right.x) and _same(left.y, right.y)


def _area(bounds: Bounds) -> float:
    return (bounds.xmax - bounds.xmin) * (bounds.ymax - bounds.ymin)


def _strictly_contains(outer: Bounds, inner: Bounds) -> bool:
    return outer.contains(inner) and outer != inner


def _region_contains_bounds(region: _Region, bounds: Bounds, *, include_boundary: bool = False) -> bool:
    if not region.path:
        return region.source.bounds.contains(bounds)
    import numpy as np

    boundary = region.source.boundary
    segments: list[tuple[Point, ...]] = []
    if boundary.codes is None:
        segments.extend((left, right) for left, right in zip(boundary.points, boundary.points[1:]))
    else:
        current = boundary.points[0]
        start = current
        index = 1
        while index < len(boundary.points):
            code = boundary.codes[index]
            if code == 1:
                current = boundary.points[index]
                start = current
                index += 1
            elif code == 2:
                following = boundary.points[index]
                segments.append((current, following))
                current = following
                index += 1
            elif code == 3:
                if index + 1 >= len(boundary.points) or boundary.codes[index + 1] != 3:
                    raise _fail("visible region has incomplete quadratic contour geometry")
                control, following = boundary.points[index : index + 2]
                segments.append((current, control, following))
                current = following
                index += 2
            elif code == 4:
                if index + 2 >= len(boundary.points) or boundary.codes[index : index + 3] != (4, 4, 4):
                    raise _fail("visible region has incomplete cubic contour geometry")
                control_1, control_2, following = boundary.points[index : index + 3]
                segments.append((current, control_1, control_2, following))
                current = following
                index += 3
            elif code == 79:
                segments.append((current, start))
                current = start
                index += 1
            else:
                raise _fail("visible region uses an unsupported contour code", code=code)
        if boundary.closed and not _same_point(current, start):
            segments.append((current, start))

    def evaluate(values: Sequence[float], t: float) -> float:
        if len(values) == 2:
            return (1.0 - t) * values[0] + t * values[1]
        if len(values) == 3:
            return (1.0 - t) ** 2 * values[0] + 2.0 * (1.0 - t) * t * values[1] + t**2 * values[2]
        return (
            (1.0 - t) ** 3 * values[0]
            + 3.0 * (1.0 - t) ** 2 * t * values[1]
            + 3.0 * (1.0 - t) * t**2 * values[2]
            + t**3 * values[3]
        )

    def roots_at(values: Sequence[float], target: float) -> tuple[float, ...]:
        if len(values) == 2:
            denominator = values[1] - values[0]
            return () if abs(denominator) <= COORDINATE_TOLERANCE else ((target - values[0]) / denominator,)
        if len(values) == 3:
            coefficients = [
                values[0] - 2.0 * values[1] + values[2],
                -2.0 * values[0] + 2.0 * values[1],
                values[0] - target,
            ]
        else:
            coefficients = [
                -values[0] + 3.0 * values[1] - 3.0 * values[2] + values[3],
                3.0 * values[0] - 6.0 * values[1] + 3.0 * values[2],
                -3.0 * values[0] + 3.0 * values[1],
                values[0] - target,
            ]
        while len(coefficients) > 1 and abs(coefficients[0]) <= COORDINATE_TOLERANCE:
            coefficients.pop(0)
        roots = [
            float(root.real)
            for root in np.roots(coefficients)
            if abs(float(root.imag)) <= COORDINATE_TOLERANCE
        ]
        unique: list[float] = []
        for root in sorted(roots):
            if not unique or abs(root - unique[-1]) > COORDINATE_TOLERANCE:
                unique.append(root)
        return tuple(unique)

    def derivative(values: Sequence[float], t: float) -> float:
        if len(values) == 2:
            return values[1] - values[0]
        if len(values) == 3:
            return 2.0 * ((1.0 - t) * (values[1] - values[0]) + t * (values[2] - values[1]))
        return 3.0 * (
            (1.0 - t) ** 2 * (values[1] - values[0])
            + 2.0 * (1.0 - t) * t * (values[2] - values[1])
            + t**2 * (values[3] - values[2])
        )

    def inside(point: Point) -> bool:
        if include_boundary:
            from .audit import _point_on_segment

            if any(len(segment) == 2 and _point_on_segment(point, *segment) for segment in segments):
                return True
        crossings = 0
        for segment in segments:
            xs = tuple(item.x for item in segment)
            ys = tuple(item.y for item in segment)
            for root in roots_at(ys, point.y):
                if root < -COORDINATE_TOLERANCE or root >= 1.0 - COORDINATE_TOLERANCE:
                    continue
                t = max(0.0, root)
                x = evaluate(xs, t)
                if abs(x - point.x) <= COORDINATE_TOLERANCE:
                    return include_boundary
                if x > point.x and abs(derivative(ys, t)) > COORDINATE_TOLERANCE:
                    crossings += 1
        return crossings % 2 == 1

    return all(
        inside(point)
        for point in (
            Point(bounds.xmin, bounds.ymin),
            Point(bounds.xmin, bounds.ymax),
            Point(bounds.xmax, bounds.ymin),
            Point(bounds.xmax, bounds.ymax),
        )
    )


def _point_on_boundary(point: Point, bounds: Bounds) -> bool:
    horizontal = bounds.xmin - COORDINATE_TOLERANCE <= point.x <= bounds.xmax + COORDINATE_TOLERANCE and (
        _same(point.y, bounds.ymin) or _same(point.y, bounds.ymax)
    )
    vertical = bounds.ymin - COORDINATE_TOLERANCE <= point.y <= bounds.ymax + COORDINATE_TOLERANCE and (
        _same(point.x, bounds.xmin) or _same(point.x, bounds.xmax)
    )
    return horizontal or vertical


def _point_on_bounds(point: Point, bounds: Bounds) -> bool:
    return (
        bounds.xmin - COORDINATE_TOLERANCE <= point.x <= bounds.xmax + COORDINATE_TOLERANCE
        and bounds.ymin - COORDINATE_TOLERANCE <= point.y <= bounds.ymax + COORDINATE_TOLERANCE
    )


@dataclass(frozen=True, slots=True)
class _Region:
    source: SubsystemRegion
    path: _Path
    parent: _Path | None


@dataclass(frozen=True, slots=True)
class _Body:
    path: _Path
    owner: _Path
    model: str
    pin_points: tuple[tuple[str, Point], ...]
    auxiliary_pin_points: tuple[tuple[str, Point], ...]
    bounds: Bounds
    identity_bounds: Bounds
    fields: tuple[dict[str, object], ...]
    oriented_branches: tuple[dict[str, object], ...]
    conductors: tuple[str, ...] = ()
    reference_conductor: str | None = None
    n_sections: int | None = None


@dataclass(frozen=True, slots=True)
class _Marker:
    label: str
    point: Point
    incidence: Point
    endpoint: Mapping[str, object]
    owner: _Path


def _regions(scene: NeutralScene) -> tuple[_Region, ...]:
    from .audit import _same_bounds

    roots = [
        region
        for region in scene.regions
        if all(
            region is candidate or region.bounds.contains(candidate.bounds)
            for candidate in scene.regions
        )
    ]
    if len(roots) != 1 or roots[0].header is not None:
        raise _fail("scene bounds do not establish one unlabelled root envelope")
    root = roots[0]

    def contains(candidate: SubsystemRegion, bounds: Bounds) -> bool:
        if candidate is root:
            return candidate.bounds.contains(bounds)
        # The path value merely selects contour-aware containment; identity is
        # resolved independently below from the visible header hierarchy.
        return _region_contains_bounds(_Region(candidate, ("visible",), None), bounds)

    for region in scene.regions:
        if not root.bounds.contains(region.bounds):
            raise _fail("visible region lies outside the root envelope")
        if region is root:
            continue
        if (
            region.boundary.role != "region"
            or not region.boundary.closed
            or not _same_bounds(region.boundary.bounds, region.bounds)
        ):
            raise _fail("visible child region lacks one exact closed contour")
        if region.header is None or not contains(region, region.header.bounds):
            raise _fail("ownership header is not wholly visible inside its region")
    parents: dict[int, SubsystemRegion] = {}
    for region in scene.regions:
        if region is root:
            continue
        if region.header is None:
            raise _fail("non-root ownership region lacks one visible local identity")
        containers = [
            candidate
            for candidate in scene.regions
            if candidate is not region
            and _strictly_contains(candidate.bounds, region.bounds)
            and contains(candidate, region.bounds)
        ]
        if not containers:
            raise _fail("ownership region has no visible parent")
        parents[id(region)] = min(containers, key=lambda item: _area(item.bounds))
    paths: dict[int, _Path] = {id(root): ()}

    def resolve(region: SubsystemRegion) -> _Path:
        if id(region) in paths:
            return paths[id(region)]
        parent = parents[id(region)]
        assert region.header is not None
        local_id = region.header.text
        if not local_id or "/" in local_id or local_id.startswith("REGION:"):
            raise _fail("ownership header is not one literal local authored identity", label=local_id)
        result = (*resolve(parent), local_id)
        paths[id(region)] = result
        return result

    for region in scene.regions:
        resolve(region)
    for index, left in enumerate(scene.regions):
        for right in scene.regions[index + 1 :]:
            if left is root or right is root:
                continue
            nested = contains(left, right.bounds) or contains(right, left.bounds)
            if not nested and left.bounds.overlaps(right.bounds):
                raise _fail("sibling ownership regions visibly overlap")
    siblings: dict[_Path, set[str]] = defaultdict(set)
    for region in scene.regions:
        path = paths[id(region)]
        if path:
            if path[-1] in siblings[path[:-1]]:
                raise _fail("sibling ownership regions repeat a visible local ID", id=path[-1])
            siblings[path[:-1]].add(path[-1])
    return tuple(
        _Region(region, paths[id(region)], None if region is root else paths[id(parents[id(region)])])
        for region in scene.regions
    )


def _deepest(bounds: Bounds, regions: Sequence[_Region]) -> _Region:
    containers = [region for region in regions if _region_contains_bounds(region, bounds)]
    if not containers:
        raise _fail("complete visible object lies outside the root envelope")
    return min(containers, key=lambda item: _area(item.source.bounds))


def _quantity(text: str, unit: str) -> Mapping[str, object]:
    try:
        parsed = parse_display_scalar(text, si_unit=unit)
        return {"si_decimal": parsed["si_decimal"], "si_unit": parsed["si_unit"]}
    except Exception as error:
        raise _fail("visible scalar is not a complete display-projected quantity", label=text, unit=unit) from error


def _body_groups(
    scene: NeutralScene,
    regions: Sequence[_Region],
    *,
    show_values: bool,
) -> tuple[_Body, ...]:
    from .audit import _branch_role, _box_model, _verify_native_geometry

    grouped: dict[tuple[_Path, str], list[NativeSymbol]] = defaultdict(list)
    for symbol in scene.symbols:
        _verify_native_geometry(symbol, "authoring")
        _branch_role(symbol)
        owner = _deepest(symbol.occupied_bounds, regions).path
        name = symbol.visible_name.text
        if not name or "/" in name or name.startswith("REGION:"):
            raise _fail("native body name is not one local authored identity", name=name)
        grouped[(owner, name)].append(symbol)
    bodies: list[_Body] = []
    for (owner, name), symbols in sorted(grouped.items()):
        kinds = tuple(symbol.kind for symbol in symbols)
        if kinds == ("R",):
            model, field_specs = "resistor", ((symbols[0], "resistance", "ohm"),)
        elif kinds == ("L",):
            model, field_specs = "inductor", ((symbols[0], "inductance", "henry"),)
        elif kinds == ("C",):
            model, field_specs = "capacitor", ((symbols[0], "capacitance", "farad"),)
        elif sorted(kinds) == ["C", "JJ"] and len(symbols) == 2:
            junction = next(symbol for symbol in symbols if symbol.kind == "JJ")
            capacitance = next(symbol for symbol in symbols if symbol.kind == "C")
            if capacitance.branch_label is None or capacitance.branch_label.text != "Cj":
                raise _fail("junction capacitance lacks its visible Cj branch identity", body=name)
            model = "josephson_junction"
            field_specs = (
                (junction, "josephson_inductance", "henry"),
                (capacitance, "junction_capacitance", "farad"),
            )
        else:
            raise _fail("visible glyph multiplicity does not identify one physical body model", owner=list(owner), body=name, glyphs=kinds)
        primary = next((symbol for symbol in symbols if symbol.kind != "C" or model != "josephson_junction"), symbols[0])
        # The glyph supplies two geometric contacts, not visible authored
        # terminal numbers. This local reference is normalized from topology.
        pins = tuple(
            (f"terminal_{index}", point)
            for index, point in enumerate(
                sorted((point for _, point in primary.anchors), key=lambda point: (point.x, point.y)), 1
            )
        )
        if len(pins) != 2 or len({item[0] for item in pins}) != 2:
            raise _fail("visible native body has malformed terminal anchors", body=name)
        auxiliary: tuple[tuple[str, Point], ...] = ()
        for symbol in symbols:
            if len(symbol.anchors) != 2 or {item[0] for item in symbol.anchors} != {item[0] for item in pins}:
                raise _fail("parallel native glyphs disagree on one body's terminal vocabulary", body=name)
            if symbol is not primary:
                auxiliary = tuple(
                    (f"terminal_{index}", point)
                    for index, point in enumerate(
                        sorted((point for _, point in symbol.anchors), key=lambda point: (point.x, point.y)), 1
                    )
                )
        fields: list[dict[str, object]] = []
        for symbol, field, unit in field_specs:
            if show_values and symbol.value is None:
                raise _fail("required visible scalar label is absent", body=name, field=field)
            if not show_values and symbol.value is not None:
                raise _fail("physical scalar is visible although values were not requested", body=name, field=field)
            value = None if symbol.value is None else _plain(_quantity(symbol.value.text, unit))
            fields.append({"id": field, "unit": unit, "value": value, "displayed_text": None if symbol.value is None else symbol.value.text})
        branches: list[dict[str, object]] = []
        if model in {"inductor", "josephson_junction"}:
            role = _branch_role(primary)
            branch_id = role.removeprefix("inductor:") if model == "inductor" else "self"
            branches.append(
                {
                    "id": branch_id,
                    "positive_pin": pins[0][0],
                    "negative_pin": pins[1][0],
                    "value_field": "inductance" if model == "inductor" else "josephson_inductance",
                }
            )
        union = Bounds(
            min(symbol.occupied_bounds.xmin for symbol in symbols),
            min(symbol.occupied_bounds.ymin for symbol in symbols),
            max(symbol.occupied_bounds.xmax for symbol in symbols),
            max(symbol.occupied_bounds.ymax for symbol in symbols),
        )
        if not _deepest(union, regions).path == owner:
            raise _fail("one body's complete glyph/label occupancy crosses an ownership boundary", body=name)
        bodies.append(
            _Body(
                (*owner, name), owner, model, pins, auxiliary, union, primary.visible_name.bounds,
                tuple(fields), tuple(branches),
            )
        )
    _box_model(scene, regions)  # type: ignore[arg-type]
    for box in scene.boxes:
        owner = _deepest(box.bounds, regions).path
        name = box.title.text
        if not name or "/" in name:
            raise _fail("line body title is not one local authored identity", title=name)
        rows = tuple(run.text for run in box.conductor_rows)
        if box.kind_label.text != box.kind:
            raise _fail("line body kind label disagrees with its visible native outline")
        sections = None
        if box.kind == "CPW":
            if rows or box.anchor_labels or box.reference_label is not None or len(box.anchors) != 2:
                raise _fail("CPW requires exactly two unlabelled visible terminals", body=name)
            label = box.section_label
            match = None if label is None else re.fullmatch(r"([1-9][0-9]*) (section|sections)", label.text)
            if match is None or label.role != "line-sections" or (match[1] == "1") != (match[2] == "section"):
                raise _fail("CPW omits its positive visible section count", body=name)
            sections = int(match[1])
            # Native anchor names carry source-only head/tail identity. Observe
            # only their actual lead-tip coordinates, then normalize topology.
            pins = tuple((f"terminal_{index}", point) for index, point in enumerate(sorted((point for _, point in box.anchors), key=lambda point: (point.x, point.y)), 1))
            reference = None
        else:
            if len(rows) < 2 or len(set(rows)) != len(rows):
                raise _fail("MTL does not visibly preserve ordered conductor multiplicity")
            wanted = tuple(f"{end}.{conductor}" for end in ("head", "tail") for conductor in rows)
            if tuple(name for name, _ in box.anchors) != wanted or tuple(run.text for run in box.anchor_labels) != wanted:
                raise _fail("MTL loses complete ordered head/tail conductor anchors", body=name)
            if box.reference_label is None:
                raise _fail("MTL omits its visible reference conductor", body=name)
            pins, reference = tuple(box.anchors), box.reference_label.text
        fields: list[dict[str, object]] = [{"id": "rlgc", "unit": "rlgc", "value": None}]
        if show_values and box.length_label is None:
            raise _fail("line body omits its required selected length", body=name)
        if not show_values and box.length_label is not None:
            raise _fail("line length is visible although values were not requested", body=name)
        fields.insert(0, {"id": "length", "unit": "meter", "value": None if box.length_label is None else _plain(_quantity(box.length_label.text, "meter")), "displayed_text": None if box.length_label is None else box.length_label.text})
        bodies.append(
            _Body(
                (*owner, name), owner, "transmission_line", pins, (), box.bounds,
                box.title.bounds, tuple(fields), (), rows, reference, sections,
            )
        )
    paths = [body.path for body in bodies]
    if len(paths) != len(set(paths)):
        raise _fail("visible physical-body identities are duplicated")
    return tuple(sorted(bodies, key=lambda body: body.path))


def _observed_nets(
    scene: NeutralScene,
    bodies: Sequence[_Body],
) -> tuple[object, dict[int, str], object, tuple[tuple[Point, Point], ...]]:
    from .audit import _build_point_graph, _point_on_segment, _port_identity

    graph, segments = _build_point_graph(scene)
    # Structured semantic guides may terminate at an interior point of one
    # visible wire segment.  Register that literal geometric incidence before
    # naming equivalence classes; segmentation of the same continuous ink
    # cannot change the witness.  Public analysis labels are observations and
    # therefore never register or union electrical graph points.
    for guide in scene.guides:
        if guide.kind == "analysis_label":
            continue
        for terminal in guide.terminals:
            contacted = [segment for segment in segments if _point_on_segment(terminal, *segment)]
            if contacted:
                graph.union(
                    graph.index(terminal),
                    *(graph.index(segment[0]) for segment in contacted),
                )
    contacts: dict[int, set[str]] = defaultdict(set)
    for body in bodies:
        for pin, point in body.pin_points:
            contacts[graph.root(point)].add(_contact(path=body.path, pin=pin))
    for port in scene.ports:
        port_id, _, _, _ = _port_identity(port)
        contacts[graph.root(port.external_anchor)].add(_contact(port=port_id))
    for body in bodies:
        if not body.auxiliary_pin_points:
            continue
        if len(body.auxiliary_pin_points) != len(body.pin_points) or any(
            auxiliary_pin != primary_pin
            or graph.root(auxiliary_point) != graph.root(primary_point)
            for (auxiliary_pin, auxiliary_point), (primary_pin, primary_point) in zip(
                body.auxiliary_pin_points, body.pin_points, strict=True
            )
        ):
            raise _fail(
                "junction capacitance is not visibly parallel to the corresponding JJ terminals",
                path=list(body.path),
            )
    roots: dict[int, str] = {}
    rows: list[dict[str, object]] = []
    ordered_roots = sorted(
        (root for root in contacts if not graph.is_ground(root)),
        key=lambda root: canonical_json_bytes(sorted(contacts[root])),
    )
    ordinals = {root: f"netv-{ordinal}" for ordinal, root in enumerate(ordered_roots)}
    for root, values in contacts.items():
        visible = "ground" if graph.is_ground(root) else ordinals[root]
        roots[root] = visible
        rows.append({"net": visible, "contacts": sorted(values)})
    stray = {
        graph.root(point)
        for segment in segments
        for point in segment
        if not graph.is_ground(graph.root(point)) and graph.root(point) not in contacts
    }
    if stray:
        raise _fail("visible conductive ink forms a net with no physical terminal or Port")
    return graph, roots, tuple(sorted(rows, key=canonical_json_bytes)), segments


def _net(graph: object, roots: Mapping[int, str], point: Point) -> str:
    root = graph.root(point)
    if graph.is_ground(root):
        return "ground"
    if root not in roots:
        raise _fail("visible contact net contains no physical terminal or Port")
    return roots[root]


def _ray_count(point: Point, segments: Sequence[tuple[Point, Point]]) -> int:
    from .audit import _point_on_segment

    directions: list[tuple[float, float]] = []
    for left, right in segments:
        if not _point_on_segment(point, left, right):
            continue
        for other in (left, right):
            # Use the graph's point equivalence before normalizing a ray;
            # coordinate-equivalent endpoints cannot create a third direction.
            if _same_point(point, other):
                continue
            dx = 0.0 if _same(other.x, point.x) else other.x - point.x
            dy = 0.0 if _same(other.y, point.y) else other.y - point.y
            length = (dx * dx + dy * dy) ** 0.5
            if length <= COORDINATE_TOLERANCE:
                continue
            direction = (dx / length, dy / length)
            if any(
                abs(direction[0] * existing[1] - direction[1] * existing[0])
                <= COORDINATE_TOLERANCE
                and direction[0] * existing[0] + direction[1] * existing[1] > 0.0
                for existing in directions
            ):
                continue
            directions.append(direction)
    return len(directions)


def _junction_segments(
    scene: NeutralScene,
    conductive: Sequence[tuple[Point, Point]],
) -> tuple[tuple[Point, Point], ...]:
    """Add visible ground stems without treating a glyph body as a wire."""

    from .audit import _is_ground_paths

    segments = list(conductive)
    for guide in scene.guides:
        if (
            guide.label is None
            and len(guide.terminals) == 1
            and _is_ground_paths(guide.paths, guide.terminals[0])
        ):
            terminal = guide.terminals[0]
            for path in guide.paths:
                if _same_point(terminal, path.points[0]):
                    segments.append((path.points[0], path.points[1]))
                elif _same_point(terminal, path.points[-1]):
                    segments.append((path.points[-1], path.points[-2]))
    return tuple(segments)


def _verify_junction_marks(
    scene: NeutralScene,
    segments: Sequence[tuple[Point, Point]],
) -> None:
    """Require visible dots for true junctions without letting dots join ink."""

    from .audit import _same_path, _segment_intersection
    from .native import junction_mark

    by_point: list[Point] = []
    for mark in scene.node_marks:
        if mark.label is not None:
            raise _fail("authoring junction dot carries a hidden node identity")
        reference = junction_mark(mark.point)
        if (
            not mark.filled
            or len(mark.paths) != len(reference.paths)
            or any(
                not _same_path(actual, wanted)
                for actual, wanted in zip(mark.paths, reference.paths, strict=True)
            )
        ):
            raise _fail("authoring junction dot does not match its visible native grammar")
        if any(_same_point(mark.point, point) for point in by_point):
            raise _fail("authoring scene duplicates one visible junction dot")
        if _ray_count(mark.point, segments) < 2:
            raise _fail("filled junction dot is isolated from real connected ink")
        by_point.append(mark.point)

    candidates = [point for segment in segments for point in segment]
    for index, first in enumerate(segments):
        for second in segments[index + 1 :]:
            relation, point = _segment_intersection(first, second)
            if relation == "point" and point is not None:
                candidates.append(point)
    unique: list[Point] = []
    for point in candidates:
        if not any(_same_point(point, existing) for existing in unique):
            unique.append(point)
    for point in unique:
        if _ray_count(point, segments) < 3:
            continue
        matches = sum(_same_point(point, marked) for marked in by_point)
        if matches != 1:
            raise _fail(
                "true conductive junction lacks exactly one visible filled dot",
                x=point.x,
                y=point.y,
            )


def _boundary_segments(region: _Region) -> tuple[tuple[Point, Point], ...]:
    path = region.source.boundary
    if path.codes is None:
        return tuple(zip(path.points, path.points[1:]))
    segments: list[tuple[Point, Point]] = []
    current = path.points[0]
    start = current
    index = 1
    while index < len(path.points):
        code = path.codes[index]
        if code == 1:
            current = path.points[index]
            start = current
            index += 1
        elif code == 2:
            following = path.points[index]
            segments.append((current, following))
            current = following
            index += 1
        elif code == 3:
            if index + 1 >= len(path.points) or path.codes[index + 1] != 3:
                raise _fail("visible region has incomplete quadratic contour geometry")
            current = path.points[index + 1]
            index += 2
        elif code == 4:
            if index + 2 >= len(path.points) or path.codes[index : index + 3] != (4, 4, 4):
                raise _fail("visible region has incomplete cubic contour geometry")
            current = path.points[index + 2]
            index += 3
        elif code == 79:
            segments.append((current, start))
            current = start
            index += 1
        else:
            raise _fail("visible region uses an unsupported contour code", code=code)
    return tuple(segments)


def _inside_region(point: Point, region: _Region) -> bool:
    bounds = region.source.bounds
    return (
        bounds.xmin + COORDINATE_TOLERANCE < point.x < bounds.xmax - COORDINATE_TOLERANCE
        and bounds.ymin + COORDINATE_TOLERANCE < point.y < bounds.ymax - COORDINATE_TOLERANCE
    )


def _open_boundary_stub(
    scene: NeutralScene,
    region: _Region,
    point: Point,
    regions: Sequence[_Region],
    bodies: Sequence[_Body],
    segments: Sequence[tuple[Point, Point]],
) -> bool:
    """Observe a straight exterior normal run ending free, not a bound flag.

    Collinear subdivisions are immaterial. A bend, branch, jump or any outside
    physical/Port/ground/other-contour contact makes this connected incidence;
    there is deliberately no maximum stub length or electrical-net shortcut.
    """
    from .audit import _is_ground_paths, _point_on_segment, _segment_intersection

    bounds = region.source.bounds
    normals = [
        (dx, dy) for coordinate, edge, dx, dy in (
            (point.x, bounds.xmin, -1, 0), (point.x, bounds.xmax, 1, 0),
            (point.y, bounds.ymin, 0, -1), (point.y, bounds.ymax, 0, 1),
        ) if abs(coordinate - edge) <= COORDINATE_TOLERANCE
    ]
    if len(normals) != 1:
        return False
    dx, dy = normals[0]

    def distance(other: Point) -> float:
        return (other.x - point.x) * dx + (other.y - point.y) * dy

    def collinear(other: Point) -> bool:
        return abs((other.x - point.x) * dy - (other.y - point.y) * dx) <= COORDINATE_TOLERANCE

    intervals = [sorted((distance(a), distance(b))) for a, b in segments if collinear(a) and collinear(b)]
    extent = 0.0
    while True:
        reached = max((end for start, end in intervals if start <= extent + COORDINATE_TOLERANCE and end > 0), default=extent)
        if reached <= extent:
            break
        extent = reached
    if extent <= COORDINATE_TOLERANCE:
        return False
    tip = Point(point.x + dx * extent, point.y + dy * extent)
    run = (point, tip)
    for a, b in segments:
        if collinear(a) and collinear(b):
            continue
        relation, crossing = _segment_intersection(run, (a, b))
        # Intrinsic ink is excluded, including its contact at the contour.
        if relation == "point" and crossing is not None and (
            distance(crossing) > COORDINATE_TOLERANCE or any(
                not _inside_region(other, region) and not _same_point(other, point)
                for other in (a, b)
            )
        ):
            return False
    # A terminal on the contour is not exterior merely because strict point
    # containment excludes edges: inspect the complete observed body ink.
    contacts = [contact for body in bodies if not _region_contains_bounds(region, body.bounds, include_boundary=True) for _, contact in (*body.pin_points, *body.auxiliary_pin_points)]
    contacts.extend(contact for port in scene.ports for contact in (port.external_anchor, port.circuit_anchor, port.boundary_anchor, port.load_anchor, port.ground_anchor))
    contacts.extend(guide.terminals[0] for guide in scene.guides if guide.label is None and len(guide.terminals) == 1 and _is_ground_paths(guide.paths, guide.terminals[0]) and any(not _inside_region(at, region) and not _point_on_boundary(at, region.source.bounds) for path in guide.paths for at in path.points))
    contacts.extend(contact for jump in scene.jumps for contact in (jump.path.points[0], jump.path.points[-1]))
    if any(_point_on_segment(contact, *run) for contact in contacts):
        return False
    for other in regions:
        if not other.path or other.path == region.path:
            continue
        for boundary in _boundary_segments(other):
            relation, crossing = _segment_intersection(run, boundary)
            if relation == "overlap" or relation == "point" and crossing is not None:
                return False
    return True


def _boundary_incidence(
    scene: NeutralScene,
    regions: Sequence[_Region],
    bodies: Sequence[_Body],
    graph: object,
    roots: Mapping[int, str],
    segments: Sequence[tuple[Point, Point]],
) -> list[dict[str, object]]:
    from .audit import _point_on_segment, _port_identity, _segment_intersection

    contacts: list[tuple[str, _Path, Point]] = [
        (_contact(path=body.path, pin=pin), body.owner, point)
        for body in bodies
        for pin, point in body.pin_points
    ]
    contacts.extend(
        (_contact(port=_port_identity(port)[0]), (), port.external_anchor)
        for port in scene.ports
    )
    rows: list[dict[str, object]] = []
    used_labels: set[int] = set()
    for region in (item for item in regions if item.path):
        crossings: list[Point] = []
        for boundary in _boundary_segments(region):
            for conductor in segments:
                relation, point = _segment_intersection(boundary, conductor)
                if relation == "overlap":
                    if _same_point(conductor[0], conductor[1]):
                        continue
                    raise _fail(
                        "conductive ink runs along a visible ownership contour",
                        region=list(region.path),
                    )
                if relation == "point" and point is not None and not any(
                    _same_point(point, existing) for existing in crossings
                ):
                    crossings.append(point)
        crossings.sort(key=lambda point: (point.x, point.y))
        labelled: dict[int, BoundarySite] = {}
        parent_incidence: dict[Point, str] = {}
        for point in crossings:
            sites = [
                (index, site)
                for index, site in enumerate(scene.boundary_sites)
                if site.visible_label is not None and _same_point(point, site.point)
            ]
            if not any(_point_on_segment(point, *segment) for segment in segments):
                raise _fail("visible region crossing does not lie on actual conductive ink")
            incident = [segment for segment in segments if _point_on_segment(point, *segment)]
            inside_ink = any(
                _inside_region(other, region)
                for segment in incident
                for other in segment
                if not _same_point(other, point)
            )
            outside_ink = any(
                not _inside_region(other, region)
                and not _point_on_boundary(other, region.source.bounds)
                for segment in incident
                for other in segment
                if not _same_point(other, point)
            )
            if not inside_ink:
                raise _fail(
                    "public boundary incidence lacks continuous intrinsic ink",
                    region=list(region.path),
                )
            parent_incidence[point] = (
                "open" if not outside_ink or _open_boundary_stub(
                    scene, region, point, regions, bodies, segments
                ) else "connected"
            )
            if len(sites) > 1:
                raise _fail("one contour crossing carries duplicate visible boundary names")
            if sites:
                index, site = sites[0]
                labelled[len(labelled)] = site
                used_labels.add(index)
        if len(crossings) == 1:
            if labelled:
                raise _fail("single public boundary must use its unlabelled only-boundary form")
            boundary_rows = [(crossings[0], {"mode": "only"})]
        else:
            if len(labelled) != len(crossings):
                raise _fail("multi-pin boundary omits a required visible local name")
            labels = [labelled[index].visible_label for index in range(len(crossings))]
            names = [cast(TextRun, label).text for label in labels]
            if len(names) != len(set(names)) or any(
                not name or "/" in name or name.startswith("pin:") for name in names
            ):
                raise _fail("multi-pin boundary names are missing, duplicated, or prefixed")
            boundary_rows = [
                (
                    point,
                    {
                        "mode": "named",
                        "id": cast(TextRun, labelled[index].visible_label).text,
                    },
                )
                for index, point in enumerate(crossings)
            ]
        for point, boundary in boundary_rows:
            root = graph.root(point)
            visible_contacts = [
                (identity, owner)
                for identity, owner, contact_point in contacts
                if graph.root(contact_point) == root
            ]
            inside = sorted(
                identity
                for identity, owner in visible_contacts
                if owner[: len(region.path)] == region.path
            )
            outside = sorted(
                identity
                for identity, owner in visible_contacts
                if owner[: len(region.path)] != region.path
            )
            rows.append(
                {
                    "region": list(region.path),
                    "boundary": boundary,
                    "parent_incidence": parent_incidence[point],
                    "net": _net(graph, roots, point),
                    "inside_contacts": inside,
                    "outside_contacts": outside,
                }
            )
    unmatched = [
        site
        for index, site in enumerate(scene.boundary_sites)
        if site.visible_label is not None and index not in used_labels
    ]
    if unmatched:
        raise _fail("visible boundary name is not attached to actual child-contour ink")
    return sorted(rows, key=canonical_json_bytes)


def _verify_local_grounds(
    scene: NeutralScene,
    regions: Sequence[_Region],
    bodies: Sequence[_Body],
    graph: object,
    raw_graph: object,
    roots: Mapping[int, str],
) -> None:
    from .audit import _is_ground_paths

    ground_guides = [
        guide
        for guide in scene.guides
        if guide.label is None
        and len(guide.terminals) == 1
        and _is_ground_paths(guide.paths, guide.terminals[0])
    ]
    guide_roots: dict[int, list[GuideMark]] = defaultdict(list)
    for guide in ground_guides:
        _deepest(guide.bounds, regions)
        guide_roots[raw_graph.root(guide.terminals[0])].append(guide)
    if any(len(guides) != 1 for guides in guide_roots.values()):
        raise _fail("one local conductive return carries duplicate visible ground glyphs")
    port_ground_roots = {raw_graph.root(port.ground_anchor) for port in scene.ports}
    if set(guide_roots) & port_ground_roots:
        raise _fail("Port-local and circuit-local ground graphics are physically shared")
    grounded_terminal_roots = {
        raw_graph.root(point)
        for body in bodies
        for _, point in body.pin_points
        if _net(graph, roots, point) == "ground"
    }
    if grounded_terminal_roots != set(guide_roots):
        raise _fail(
            "grounded physical terminals do not have exactly one local visible return",
            grounded_components=len(grounded_terminal_roots),
            ground_glyphs=len(guide_roots),
        )


def _combined_bounds(*bounds: Bounds) -> Bounds:
    return Bounds(
        min(item.xmin for item in bounds),
        min(item.ymin for item in bounds),
        max(item.xmax for item in bounds),
        max(item.ymax for item in bounds),
    )


def _guide_owner(guide: GuideMark, regions: Sequence[_Region]) -> _Path:
    occupied = guide.bounds
    if guide.label is not None:
        occupied = _combined_bounds(occupied, guide.label.bounds)
    return _deepest(occupied, regions).path


def _markers(
    scene: NeutralScene,
    regions: Sequence[_Region],
    graph: object,
    roots: Mapping[int, str],
) -> tuple[_Marker, ...]:
    markers: list[_Marker] = []
    for guide in scene.guides:
        if guide.label is None:
            continue
        text = guide.label.text
        if not text.startswith(("bus:", "tap:")):
            continue
        _guide_connected(guide)
        if len(guide.terminals) != 2:
            raise _fail("bus/tap identity leader must expose contact and unique identity incidence", label=text)
        owner = _guide_owner(guide, regions)
        if text.startswith("bus:"):
            identifier = text.removeprefix("bus:")
            endpoint = {"kind": "bus", "scope": list(owner), "id": identifier}
        else:
            pair = text.removeprefix("tap:").split("/", 1)
            if len(pair) != 2 or not all(pair):
                raise _fail("visible tap marker is malformed", label=text)
            identifier = pair[1]
            endpoint = {
                "kind": "tap",
                "scope": list(owner),
                "bus": pair[0],
                "id": pair[1],
            }
        contacts = [
            point
            for point in guide.terminals
            if graph.is_ground(graph.root(point)) or graph.root(point) in roots
        ]
        if len(contacts) != 1:
            raise _fail("bus/tap leader does not distinguish one electrical contact", label=text)
        point = contacts[0]
        incidence = next(
            terminal for terminal in guide.terminals if not _same_point(terminal, point)
        )
        markers.append(_Marker(text, point, incidence, endpoint, owner))

    boundary_by_point: dict[Point, BoundarySite] = {}
    for site in scene.boundary_sites:
        if site.visible_label is not None:
            raise _fail("public boundary identity must be carried by its selecting leader")
        if site.point in boundary_by_point:
            raise _fail("public boundary sites duplicate one visible contact")
        boundary_by_point[site.point] = site
        _net(graph, roots, site.point)
    used_boundaries: set[Point] = set()
    for guide in scene.guides:
        if guide.label is None or not guide.label.text.startswith("pin:"):
            continue
        _guide_connected(guide)
        if len(guide.terminals) != 2:
            raise _fail("public pin leader must select its boundary and backing contact")
        edges = [point for point in guide.terminals if point in boundary_by_point]
        if len(edges) != 1:
            raise _fail("public pin leader does not touch exactly one BoundarySite")
        edge = edges[0]
        matches = [
            region
            for region in regions
            if region.path and _point_on_boundary(edge, region.source.bounds)
        ]
        if len(matches) != 1:
            raise _fail("public pin BoundarySite does not lie on one owner boundary")
        owner = matches[0].path
        if _guide_owner(guide, regions) != owner:
            raise _fail("public pin leader/label is not wholly owned by its region")
        identifier = guide.label.text.removeprefix("pin:")
        if not identifier or "/" in identifier:
            raise _fail("visible public pin has an invalid local ID", label=guide.label.text)
        markers.append(
            _Marker(
                guide.label.text,
                edge,
                edge,
                {"kind": "boundary_pin", "scope": list(owner), "id": identifier},
                owner,
            )
        )
        used_boundaries.add(edge)
    if used_boundaries != set(boundary_by_point):
        raise _fail("a public BoundarySite lacks exactly one visible pin leader")
    identities = [(marker.owner, canonical_json_bytes(marker.endpoint)) for marker in markers]
    if len(identities) != len(set(identities)):
        raise _fail("visible endpoint markers duplicate one structural handle")
    return tuple(markers)


def _guide_connected(guide: GuideMark) -> None:
    from .audit import _verify_guide_terminal_incidence

    if any(path.role not in {"guide", "coupling"} or path.closed for path in guide.paths):
        raise _fail("semantic guide contains conductive or closed ink")
    _verify_guide_terminal_incidence(guide)


def _guide_order(guide: GuideMark) -> tuple[Point, ...]:
    """Order incidences along one visible simple guide path.

    The GuideMark terminal tuple is used only as the set of declared visible
    stroke contacts.  Ordering comes from the actual connected stroke graph and
    the contract's left-to-right/top-to-bottom reading convention.
    """

    _guide_connected(guide)
    points: list[Point] = []

    def index(point: Point) -> int:
        for item, existing in enumerate(points):
            if _same_point(point, existing):
                return item
        points.append(point)
        return len(points) - 1

    adjacency: dict[int, set[int]] = defaultdict(set)
    for path in guide.paths:
        if path.codes is not None:
            raise _fail("structured incidence guide must use visible linear strokes")
        for left, right in zip(path.points, path.points[1:]):
            a, b = index(left), index(right)
            if a == b:
                raise _fail("structured incidence guide has a zero-length stroke")
            adjacency[a].add(b)
            adjacency[b].add(a)
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        raise _fail("ordered structured incidence guide is not one simple visible path")
    ends = [node for node, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(ends) != 2:
        raise _fail("ordered structured incidence guide has no unique visible ends")
    left, right = (points[node] for node in ends)
    horizontal = abs(left.x - right.x) >= abs(left.y - right.y)
    start = min(ends, key=lambda node: (points[node].x, -points[node].y)) if horizontal else max(ends, key=lambda node: (points[node].y, -points[node].x))
    sequence: list[int] = []
    previous: int | None = None
    current = start
    while True:
        sequence.append(current)
        following = [node for node in adjacency[current] if node != previous]
        if not following:
            break
        if len(following) != 1:
            raise _fail("ordered structured incidence guide traversal is ambiguous")
        previous, current = current, following[0]
    terminal_nodes = {index(point) for point in guide.terminals}
    return tuple(points[node] for node in sequence if node in terminal_nodes)


def _marker_at(point: Point, markers: Sequence[_Marker]) -> _Marker | None:
    matches = [marker for marker in markers if _same_point(point, marker.incidence)]
    if not matches:
        matches = [marker for marker in markers if _same_point(point, marker.point)]
    if len(matches) > 1:
        raise _fail("one guide incidence is ambiguous between visible endpoint markers")
    return None if not matches else matches[0]


def _member_at(
    point: Point,
    bodies: Sequence[_Body],
    regions: Sequence[_Region],
    *,
    owner: _Path,
) -> _Body | _Region | None:
    matches: list[_Body | _Region] = [
        body
        for body in bodies
        if body.owner == owner and _point_on_bounds(point, body.identity_bounds)
    ]
    matches.extend(
        region
        for region in regions
        if region.parent == owner
        and region.source.header is not None
        and _point_on_bounds(point, region.source.header.bounds)
    )
    if len(matches) > 1:
        raise _fail("one guide incidence is ambiguous between visible member identities")
    return None if not matches else matches[0]


def _ground_at(point: Point, grounds: Sequence[GuideMark]) -> GuideMark | None:
    matches = [guide for guide in grounds if any(_same_point(point, terminal) for terminal in guide.terminals)]
    if len(matches) > 1:
        raise _fail("one guide incidence is ambiguous between local ground glyphs")
    return None if not matches else matches[0]


def _physical_pin_at(point: Point, bodies: Sequence[_Body]) -> Mapping[str, object] | None:
    matches = [
        (body, pin)
        for body in bodies
        for pin, terminal in body.pin_points
        if _same_point(point, terminal)
    ]
    if len(matches) > 1:
        raise _fail("one guide incidence is ambiguous between physical terminals")
    if not matches:
        return None
    body, pin = matches[0]
    return MappingProxyType(
        {"kind": "physical_pin", "path": list(body.path), "pin_id": pin}
    )


def _endpoint_at(
    point: Point,
    markers: Sequence[_Marker],
    grounds: Sequence[GuideMark],
    bodies: Sequence[_Body],
    *,
    owner: _Path,
) -> Mapping[str, object] | None:
    marker = _marker_at(point, markers)
    ground = _ground_at(point, grounds)
    if marker is not None:
        kind = marker.endpoint.get("kind")
        if kind in {"bus", "tap"} and marker.owner != owner:
            raise _fail("structured endpoint selects a bus/tap outside its owning scope")
        if kind == "boundary_pin" and marker.owner[:-1] != owner:
            raise _fail("structured endpoint selects a boundary pin outside its parent scope")
        return marker.endpoint
    if ground is not None:
        return MappingProxyType({"kind": "ground"})
    physical = _physical_pin_at(point, bodies)
    if physical is not None:
        path = tuple(cast(Sequence[str], physical["path"]))
        if path[:-1] != owner:
            raise _fail("structured endpoint selects a physical pin outside its owning scope")
    return physical


def _member_choices(
    member: _Body | _Region,
    start_net: str,
    markers: Sequence[_Marker],
    graph: object,
    roots: Mapping[int, str],
) -> tuple[tuple[dict[str, object], str], ...]:
    if isinstance(member, _Body):
        path = member.path
        pins = tuple(
            (pin, _net(graph, roots, point)) for pin, point in member.pin_points
        )
    else:
        path = member.path
        pins = tuple(
            (cast(str, marker.endpoint["id"]), _net(graph, roots, marker.point))
            for marker in markers
            if marker.endpoint.get("kind") == "boundary_pin"
            and marker.owner == member.path
        )
        if len(pins) < 2:
            raise _fail(
                "visible complete Composite member has fewer than two public boundary pins",
                path=list(path),
            )
    choices = [
        (
            {"path": list(path), "pin_1": first, "pin_2": second},
            second_net,
        )
        for first, first_net in pins
        if first_net == start_net
        for second, second_net in pins
        if second != first
    ]
    return tuple(choices)


def _member_chain(
    members: Sequence[_Body | _Region],
    *,
    start_net: str,
    end_net: str,
    markers: Sequence[_Marker],
    graph: object,
    roots: Mapping[int, str],
) -> list[dict[str, object]]:
    states: list[tuple[str, list[dict[str, object]]]] = [(start_net, [])]
    for member in members:
        following: list[tuple[str, list[dict[str, object]]]] = []
        for current, records in states:
            for record, next_net in _member_choices(
                member, current, markers, graph, roots
            ):
                following.append((next_net, [*records, record]))
        states = following
        if not states:
            break
    matches = [records for final, records in states if final == end_net]
    unique = {
        canonical_json_bytes(records): records
        for records in matches
    }
    if len(unique) != 1:
        paths = [
            list(member.path)
            for member in members
        ]
        raise _fail(
            "visible member incidences do not select one ordered pin-pair chain",
            paths=paths,
            choices=len(unique),
        )
    return next(iter(unique.values()))


def _endpoint_net(
    endpoint: Mapping[str, object],
    incidence: Point,
    markers: Sequence[_Marker],
    graph: object,
    roots: Mapping[int, str],
) -> str:
    if endpoint.get("kind") == "ground":
        return "ground"
    marker = _marker_at(incidence, markers)
    return _net(graph, roots, marker.point if marker is not None else incidence)


def _structure_guides(
    scene: NeutralScene,
    regions: Sequence[_Region],
    bodies: Sequence[_Body],
    markers: Sequence[_Marker],
    graph: object,
    roots: Mapping[int, str],
) -> tuple[list[dict[str, object]], set[int]]:
    from .audit import _is_ground_paths

    grounds = [
        guide
        for guide in scene.guides
        if guide.label is None
        and len(guide.terminals) == 1
        and _is_ground_paths(guide.paths, guide.terminals[0])
    ]
    records: list[dict[str, object]] = []
    intrinsic_ground_ids: set[int] = set()
    parallel: dict[tuple[_Path, str], list[dict[str, object]]] = defaultdict(list)
    patterns = (
        (re.compile(r"^S:(.+)$"), "series"),
        (re.compile(r"^B:(.+)$"), "branch"),
        (re.compile(r"^P:([^/]+)/(.+)$"), "parallel_branch"),
        (re.compile(r"^LINK:(.+)$"), "link"),
    )
    for guide in scene.guides:
        if guide.label is None:
            continue
        matched: tuple[str, re.Match[str]] | None = None
        for pattern, kind in patterns:
            result = pattern.fullmatch(guide.label.text)
            if result is not None:
                matched = (kind, result)
                break
        if matched is None:
            continue
        kind, match = matched
        owner = _guide_owner(guide, regions)
        if kind == "link":
            _guide_connected(guide)
            endpoints: list[Mapping[str, object]] = []
            for terminal in guide.terminals:
                endpoint = _endpoint_at(
                    terminal, markers, grounds, bodies, owner=owner
                )
                if endpoint is None or endpoint.get("kind") == "ground":
                    raise _fail("visible LINK incidence is not a complete non-ground endpoint set", link=match.group(1))
                endpoints.append(endpoint)
            if len(endpoints) < 2 or len({canonical_json_bytes(item) for item in endpoints}) != len(endpoints):
                raise _fail("visible LINK incidence is incomplete or duplicated", link=match.group(1))
            records.append({"scope": list(owner), "kind": "link", "id": match.group(1), "endpoints": sorted((_plain(item) for item in endpoints), key=canonical_json_bytes)})
            continue
        ordered = _guide_order(guide)
        if len(ordered) < 3:
            raise _fail("structured operator guide lacks endpoint/member incidence", label=guide.label.text)
        start = _endpoint_at(ordered[0], markers, grounds, bodies, owner=owner)
        end = _endpoint_at(ordered[-1], markers, grounds, bodies, owner=owner)
        if start is None or end is None:
            raise _fail("structured operator guide lacks visible endpoint marks", label=guide.label.text)
        member_bodies = [
            _member_at(point, bodies, regions, owner=owner)
            for point in ordered[1:-1]
        ]
        if any(body is None for body in member_bodies):
            raise _fail("structured operator guide has an incidence not attached to one body identity", label=guide.label.text)
        final_net = _endpoint_net(end, ordered[-1], markers, graph, roots)
        members = _member_chain(
            cast(list[_Body | _Region], member_bodies),
            start_net=_endpoint_net(start, ordered[0], markers, graph, roots),
            end_net=final_net,
            markers=markers,
            graph=graph,
            roots=roots,
        )
        for endpoint, point in ((start, ordered[0]), (end, ordered[-1])):
            if endpoint.get("kind") == "ground":
                ground = _ground_at(point, grounds)
                assert ground is not None
                if _deepest(ground.bounds, regions).path != owner:
                    raise _fail(
                        "intrinsic ground glyph lies outside its structured owner",
                        label=guide.label.text,
                    )
                intrinsic_ground_ids.add(id(ground))
        if kind == "series":
            records.append({"scope": list(owner), "kind": "series", "id": match.group(1), "start": _plain(start), "members": members, "end": _plain(end)})
        elif kind == "branch":
            records.append({"scope": list(owner), "kind": "branch", "id": match.group(1), "at": _plain(start), "members": members, "end": _plain(end)})
        else:
            parallel[(owner, match.group(1))].append({"id": f"{match.group(1)}.{match.group(2)}", "members": members, "start": _plain(start), "end": _plain(end)})
    for (owner, identifier), branches in parallel.items():
        starts = {canonical_json_bytes(branch.pop("start")) for branch in branches}
        ends = {canonical_json_bytes(branch.pop("end")) for branch in branches}
        if len(starts) != 1 or len(ends) != 1 or len(branches) < 2:
            raise _fail("visible parallel branches do not share one complete endpoint pair", parallel=identifier)
        records.append(
            {
                "scope": list(owner), "kind": "parallel", "id": identifier,
                "start": _plain(__import__("json").loads(next(iter(starts)))),
                "branches": sorted(branches, key=lambda row: cast(str, row["id"])),
                "end": _plain(__import__("json").loads(next(iter(ends)))),
            }
        )
    return records, intrinsic_ground_ids


def _exposure_guides(
    scene: NeutralScene,
    regions: Sequence[_Region],
    markers: Sequence[_Marker],
    bodies: Sequence[_Body],
    graph: object,
    roots: Mapping[int, str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    pin_markers = [marker for marker in markers if marker.label.startswith("pin:")]
    for guide in scene.guides:
        if guide.label is None:
            continue
        text = guide.label.text
        if not text.startswith(("pin:", "coord:", "branch:")):
            continue
        _guide_connected(guide)
        identifier = text.split(":", 1)[1]
        if not identifier or "/" in identifier:
            raise _fail("visible exposure has an invalid local ID", label=text)
        if text.startswith("pin:"):
            candidates = [
                marker
                for marker in pin_markers
                if marker.label == text
                and any(_same_point(marker.point, point) for point in guide.terminals)
            ]
            if len(candidates) != 1 or len(guide.terminals) != 2:
                raise _fail("public pin exposure leader omits its conductive boundary site", label=text)
            marker = candidates[0]
            backing = next(point for point in guide.terminals if not _same_point(marker.point, point))
            target = _marker_at(backing, markers)
            if target is None or target.endpoint.get("kind") not in {"bus", "tap"} or target.owner != marker.owner:
                raise _fail("public pin leader does not select one same-scope backing bus/tap", label=text)
            net = _net(graph, roots, marker.point)
            if net != _net(graph, roots, target.point):
                raise _fail("public pin backing leader disagrees with visible electrical continuity", label=text)
            result.append({"scope": list(marker.owner), "kind": "pin", "id": identifier, "target": _plain(target.endpoint), "net": net, "ground_role": "ground" if net == "ground" else "ungrounded"})
        elif text.startswith("coord:"):
            if len(guide.terminals) != 1:
                raise _fail("coordinate exposure leader must select one backing contact", label=text)
            target = _marker_at(guide.terminals[0], markers)
            if target is None or target.endpoint.get("kind") not in {"bus", "tap"}:
                raise _fail("coordinate exposure leader does not select one bus/tap", label=text)
            owner = _guide_owner(guide, regions)
            if target.owner != owner:
                raise _fail("coordinate exposure leader crosses its ownership boundary", label=text)
            net = _net(graph, roots, target.point)
            result.append({"scope": list(owner), "kind": "coordinate", "id": identifier, "target": _plain(target.endpoint), "net": net, "ground_role": "ground" if net == "ground" else "ungrounded"})
        else:
            if len(guide.terminals) != 1:
                raise _fail("branch exposure leader must select one inductive body center", label=text)
            terminal = guide.terminals[0]
            candidates = [body for body in bodies if body.oriented_branches and _point_on_bounds(terminal, body.bounds)]
            if len(candidates) != 1:
                raise _fail("branch exposure leader does not select one inductive body", label=text)
            body = candidates[0]
            owner = _guide_owner(guide, regions)
            if body.owner != owner:
                raise _fail("branch exposure leader crosses its ownership boundary", label=text)
            branch = body.oriented_branches[0]
            result.append({"scope": list(owner), "kind": "branch", "id": identifier, "target": {"path": list(body.path), "id": branch["id"]}})
    return result


def _couplings(
    scene: NeutralScene,
    regions: Sequence[_Region],
    bodies: Sequence[_Body],
    *,
    show_values: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for guide in scene.guides:
        if guide.kind != "coupling" or guide.label is None:
            continue
        from dataclasses import replace
        from .audit import _same_path
        from .native import winding_dot

        dots = tuple(path for path in guide.paths if path.closed)
        if len(dots) != 2:
            raise _fail("visible mutual coupling requires exactly two winding dots")
        _guide_connected(replace(guide, paths=tuple(path for path in guide.paths if not path.closed)))
        if len(guide.terminals) != 2:
            raise _fail("visible mutual coupling lacks label/two oriented incidences")
        if not guide.label.text.startswith("COUPLING:"):
            raise _fail(
                "visible mutual coupling label is malformed",
                label=guide.label.text,
            )
        payload = guide.label.text.removeprefix("COUPLING:")
        if show_values:
            identifier, separator, coefficient_text = payload.rpartition("; k=")
            if not separator or not identifier or not coefficient_text:
                raise _fail("visible mutual coupling label is malformed", label=guide.label.text)
        else:
            identifier, separator, sign_text = payload.rpartition("; k")
            if not separator or not identifier or sign_text not in {"<0", ">0", "=0"}:
                raise _fail("visible mutual coupling label is malformed", label=guide.label.text)
            coefficient_text = {"<0": "-1 dimensionless", ">0": "1 dimensionless", "=0": "0 dimensionless"}[sign_text]
        selected: list[_Body] = []
        reference_sign = 1
        for terminal in guide.terminals:
            candidates = []
            for body in bodies:
                if len(body.oriented_branches) != 1:
                    continue
                symbol = next(
                    symbol for symbol in scene.symbols
                    if symbol.kind in {"L", "JJ"}
                    and set(point for _, point in symbol.anchors) == set(point for _, point in body.pin_points)
                )
                for pin, point in body.pin_points:
                    other = next(other for _, other in body.pin_points if other != point)
                    center, dot = winding_dot(point, other, symbol.symbol_bounds)
                    if _same_point(terminal, center) and sum(_same_path(dot, actual) for actual in dots) == 1:
                        candidates.append((body, pin))
            if len(candidates) != 1:
                raise _fail("mutual guide terminal does not select one visible inductive terminal")
            body, pin = candidates[0]
            reference_sign *= 1 if pin == body.oriented_branches[0]["positive_pin"] else -1
            selected.append(body)
        if selected[0].path == selected[1].path:
            raise _fail("mutual guide attaches twice to one body")
        from decimal import Decimal

        coefficient = dict(_quantity(coefficient_text, "dimensionless"))
        comparison_coefficient = dict(coefficient)
        number = Decimal(str(coefficient["si_decimal"]))
        if reference_sign < 0:
            number = number.copy_negate()
        comparison_coefficient["si_decimal"] = str(number) if number else "0"
        branches = sorted(
            (
                {"path": list(body.path), "branch_id": body.oriented_branches[0]["id"]}
                for body in selected
            ),
            key=canonical_json_bytes,
        )
        record = {
            "scope": list(_guide_owner(guide, regions)),
            "kind": "coupling",
            "id": identifier,
            "branches": branches,
        }
        records.append(record)
        rows.append({"category": "observed_scene", "kind": "mutual_coupling", "identity": identifier, "facts": record})
        if coefficient is not None:
            rows.append(
                {
                    "category": "observed_scene",
                    "kind": "displayed_parameter_value" if show_values else "displayed_coupling_polarity",
                    "identity": _token("coupling_coefficient", coupling_id=identifier),
                    "value": _plain(coefficient),
                    "displayed_text": coefficient_text if show_values else guide.label.text,
                    "comparison_value": _plain(comparison_coefficient),
                    "comparison_reference": "topology-normalized branch references",
                }
            )
    return records, rows


def reconstruct_authoring(
    scene: NeutralScene,
    *,
    show_values: bool,
) -> tuple[Mapping[str, object], Mapping[str, object], tuple[Mapping[str, object], ...]]:
    """Reconstruct authoring A/B solely from emitted glyph/stroke geometry."""

    if not isinstance(scene, NeutralScene):
        raise TypeError("authoring reconstruction requires NeutralScene")
    if not isinstance(show_values, bool):
        raise TypeError("show_values must be boolean")
    from .audit import _is_ground_paths

    for guide in scene.guides:
        if guide.label is None:
            if not (
                len(guide.terminals) == 1
                and _is_ground_paths(guide.paths, guide.terminals[0])
            ):
                raise _fail("unlabelled guide lies outside the closed visible grammar")
        elif guide.kind == "coupling":
            continue
        elif not (
            guide.label.role == "analysis-label"
            and guide.kind == "analysis_label"
            and len(guide.paths) == 1
            and guide.paths[0].role == "analysis-label"
            and len(guide.paths[0].points) == 2
            and guide.terminals == (guide.paths[0].points[0],)
        ):
            raise _fail(
                "authoring scene contains a superseded semantic debug guide",
                label=guide.label.text,
            )
    regions = _regions(scene)
    bodies = _body_groups(scene, regions, show_values=show_values)
    graph, roots, nets, segments = _observed_nets(scene, bodies)
    _verify_junction_marks(scene, _junction_segments(scene, segments))
    from .audit import _build_point_graph

    raw_graph, _ = _build_point_graph(scene, normalize_ground=False)
    _verify_local_grounds(scene, regions, bodies, graph, raw_graph, roots)
    coupling_structures, coupling_rows = _couplings(
        scene, regions, bodies, show_values=show_values
    )

    analysis_labels = []
    analysis_rows = []
    from .audit import _point_on_segment

    for guide in scene.guides:
        if guide.kind != "analysis_label":
            continue
        assert guide.label is not None
        wire_point, free_end = guide.paths[0].points
        if not any(_same_point(wire_point, point) for point in graph.points):
            raise _fail(
                "public analysis label leader is not attached to visible electrical ink",
                label=guide.label.text,
            )
        if any(_same_point(free_end, point) for point in graph.points) or any(
            _point_on_segment(free_end, *segment) for segment in segments
        ):
            raise _fail(
                "public analysis label leader falsely terminates on electrical ink",
                label=guide.label.text,
            )
        contact_root = graph.root(wire_point)
        owner = _deepest(guide.label.bounds, regions).path
        record = {
            "scope": list(owner),
            "text": guide.label.text,
            "net": "ground" if graph.is_ground(contact_root) else roots[contact_root],
        }
        analysis_labels.append(record)
        analysis_rows.append(
            {
                "category": "observed_scene",
                "kind": "public_analysis_label",
                "identity": guide.label.text,
                "facts": record,
            }
        )

    body_rows: list[dict[str, object]] = []
    value_rows: list[dict[str, object]] = []
    for body in bodies:
        fields = [{"id": row["id"], "unit": row["unit"]} for row in body.fields]
        for row in body.fields:
            if row["unit"] != "rlgc" and row["value"] is not None:
                identity = _token("physical_field", component_path=list(body.path), field=row["id"])
                value_rows.append({"category": "observed_scene", "kind": "displayed_parameter_value", "identity": identity, "value": row["value"], "displayed_text": row["displayed_text"]})
        record: dict[str, object] = {
            "path": list(body.path), "model": body.model,
            "pin_order": [pin for pin, _ in body.pin_points],
            "terminals": [{"pin_id": pin, "net": _net(graph, roots, point)} for pin, point in body.pin_points],
            "fields": fields, "oriented_branches": _plain(body.oriented_branches),
        }
        if body.model == "transmission_line":
            record["conductors"] = list(body.conductors)
            record["reference_conductor"] = body.reference_conductor
            record["line_kind"] = "CPW" if body.n_sections is not None else "MTL"
            if body.n_sections is not None:
                record["n_sections"] = body.n_sections
        body_rows.append(record)

    ports: list[dict[str, object]] = []
    from .audit import _port_identity, _verify_port_block_geometry
    for port in scene.ports:
        _verify_port_block_geometry(port)
        port_id, role, impedance_text, _ = _port_identity(port)
        if _deepest(port.occupied_bounds, regions).path or any(
            region.path and region.source.bounds.overlaps(port.occupied_bounds)
            for region in regions
        ):
            raise _fail("root Port/load occupancy lies inside a child ownership region", port_id=port_id)
        if any(graph.root(anchor) != graph.root(port.external_anchor) for anchor in (port.circuit_anchor, port.boundary_anchor, port.load_anchor)):
            raise _fail("Port circle/load is not continuously attached to its visible circuit node", port_id=port_id)
        if not graph.is_ground(graph.root(port.ground_anchor)):
            raise _fail("Port raw load lacks its visible local reference return", port_id=port_id)
        impedance = _plain(_quantity(impedance_text, "ohm"))
        ports.append({"port_id": port_id, "role": role, "node_net": _net(graph, roots, port.external_anchor), "reference_net": "ground", "reference_impedance": impedance, "orientation": "node_to_reference", "load_kind": "raw_reference_impedance"})
        value_rows.append({"category": "observed_scene", "kind": "port_impedance", "identity": _token("port_impedance", port_id=port_id), "value": impedance, "displayed_text": impedance_text})

    coupling_electrical = [
        {"coupling_id": row["id"], "branches": row["branches"]}
        for row in coupling_structures
    ]
    electrical = cast(Mapping[str, object], _freeze({
        "schema": "scnsim.diagram_electrical_manifest", "schema_version": 2,
        "representation": "authoring", "nets": list(nets),
        "bodies": sorted(body_rows, key=lambda row: cast(list[str], row["path"])),
        "ports": sorted(ports, key=lambda row: cast(str, row["port_id"])),
        "couplings": sorted(coupling_electrical, key=lambda row: cast(str, row["coupling_id"])),
    }))

    region_rows = [
        {
            "path": list(region.path),
            "parent": [] if region.parent is None else list(region.parent),
            "local_id": region.path[-1],
        }
        for region in regions
        if region.path
    ]
    leaf_ownership = [{"path": list(body.path), "owner": list(body.owner), "model": body.model} for body in bodies]
    port_ownership = [{"port_id": row["port_id"], "owner": []} for row in ports]
    boundary_incidence = _boundary_incidence(
        scene, regions, bodies, graph, roots, segments
    )
    structural = cast(Mapping[str, object], _freeze({
        "schema": "scnsim.diagram_structural_manifest", "schema_version": 2,
        "representation": "authoring",
        "regions": sorted(region_rows, key=lambda row: cast(list[str], row["path"])),
        "leaf_ownership": sorted(leaf_ownership, key=lambda row: cast(list[str], row["path"])),
        "port_ownership": sorted(port_ownership, key=lambda row: cast(str, row["port_id"])),
        "boundary_incidence": boundary_incidence,
        "public_analysis_labels": sorted(analysis_labels, key=canonical_json_bytes),
    }))
    from .equivalence import normalize_authoring

    details = [*value_rows, *coupling_rows, *analysis_rows]
    values = {
        row["identity"]: row.get("comparison_value", row["value"]) for row in details
        if row["kind"] in {"displayed_parameter_value", "displayed_coupling_polarity", "port_impedance"}
    }
    electrical, structural, values = normalize_authoring(electrical, structural, values)
    for row in details:
        if "comparison_value" in row:
            row["comparison_value"] = values[row["identity"]]
    detail = tuple(cast(Mapping[str, object], _freeze(row)) for row in details)
    return electrical, structural, detail


__all__ = ["reconstruct_authoring"]
