"""Independent electrical and semantic witnesses for emitted diagram scenes.

Only visible neutral-scene facts are reconstructed here.  In particular this
module never reads renderer correlation keys, semantic-IR records, placement
ownership, hidden net names, or ``GuideMark.kind``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
import json
import math
import re
from types import MappingProxyType
from typing import Literal, cast

from .. import units
from .._canonical import canonical_json_bytes, float64_from_hex, quantity_envelope, sha256_hex
from ..errors import SCNSimValidationError
from .expected import _ExpectedManifest
from .scene import (
    COORDINATE_TOLERANCE,
    Bounds,
    BoundarySite,
    ElectricalBox,
    GuideMark,
    NativeSymbol,
    NeutralScene,
    Path,
    Point,
    PortBlock,
    SubsystemRegion,
    TextRun,
    scene_digest,
)
from .snapshot import _CapturedPlan

_Representation = Literal["authoring", "compiled"]
_Path = tuple[str, ...]
_Endpoint = tuple[_Path, str]
_GROUND = -1
_NATIVE_KINDS = frozenset({"R", "L", "C", "JJ", "G"})


def _audit_fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_audit", evidence=evidence)


def _layout_fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_layout", evidence=evidence)


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


def _contact_token(kind: str, *, path: _Path = (), identity: str, terminal: str = "") -> str:
    return _token(
        "diagram_contact",
        contact_kind=kind,
        component_path=list(path),
        identity=identity,
        terminal=terminal,
    )


def _endpoint_record(endpoint: _Endpoint) -> dict[str, object]:
    return {"component_path": list(endpoint[0]), "pin_id": endpoint[1]}


def _net_key(endpoints: Iterable[_Endpoint], *, ground: bool) -> str:
    if ground:
        return "ground"
    records = [_endpoint_record(endpoint) for endpoint in sorted(set(endpoints))]
    if not records:
        raise _audit_fail("an observed non-ground net has no visible electrical endpoint")
    return "net-" + sha256_hex(
        {
            "schema": "scnsim.diagram_net_equivalence",
            "schema_version": 1,
            "endpoints": records,
        }
    )


def _same(left: float, right: float) -> bool:
    return abs(left - right) <= COORDINATE_TOLERANCE


def _same_point(left: Point, right: Point) -> bool:
    return _same(left.x, right.x) and _same(left.y, right.y)


def _finite_point(point: Point) -> bool:
    return math.isfinite(point.x) and math.isfinite(point.y)


def _finite_bounds(bounds: Bounds) -> bool:
    return all(math.isfinite(value) for value in (bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax))


def _strictly_contains(outer: Bounds, inner: Bounds) -> bool:
    return outer.contains(inner) and outer != inner


def _area(bounds: Bounds) -> float:
    return (bounds.xmax - bounds.xmin) * (bounds.ymax - bounds.ymin)


def _point_in_bounds(point: Point, bounds: Bounds) -> bool:
    return (
        bounds.xmin - COORDINATE_TOLERANCE <= point.x <= bounds.xmax + COORDINATE_TOLERANCE
        and bounds.ymin - COORDINATE_TOLERANCE <= point.y <= bounds.ymax + COORDINATE_TOLERANCE
    )


def _point_on_rect_boundary(point: Point, bounds: Bounds) -> bool:
    horizontal = (
        bounds.xmin - COORDINATE_TOLERANCE <= point.x <= bounds.xmax + COORDINATE_TOLERANCE
        and (_same(point.y, bounds.ymin) or _same(point.y, bounds.ymax))
    )
    vertical = (
        bounds.ymin - COORDINATE_TOLERANCE <= point.y <= bounds.ymax + COORDINATE_TOLERANCE
        and (_same(point.x, bounds.xmin) or _same(point.x, bounds.xmax))
    )
    return horizontal or vertical


def _point_on_segment(point: Point, left: Point, right: Point) -> bool:
    length = math.hypot(right.x - left.x, right.y - left.y)
    if length <= COORDINATE_TOLERANCE:
        return _same_point(point, left)
    cross = (point.x - left.x) * (right.y - left.y) - (point.y - left.y) * (right.x - left.x)
    if abs(cross) / length > COORDINATE_TOLERANCE:
        return False
    return (
        min(left.x, right.x) - COORDINATE_TOLERANCE <= point.x <= max(left.x, right.x) + COORDINATE_TOLERANCE
        and min(left.y, right.y) - COORDINATE_TOLERANCE <= point.y <= max(left.y, right.y) + COORDINATE_TOLERANCE
    )


def _segment_intersection(
    first: tuple[Point, Point], second: tuple[Point, Point]
) -> tuple[Literal["none", "point", "overlap"], Point | None]:
    a, b = first
    c, d = second
    first_length = math.hypot(b.x - a.x, b.y - a.y)
    second_length = math.hypot(d.x - c.x, d.y - c.y)
    if first_length <= COORDINATE_TOLERANCE:
        return ("point", a) if _point_on_segment(a, c, d) else ("none", None)
    if second_length <= COORDINATE_TOLERANCE:
        return ("point", c) if _point_on_segment(c, a, b) else ("none", None)
    denominator = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x)
    numerator_first = (c.x - a.x) * (d.y - c.y) - (c.y - a.y) * (d.x - c.x)
    numerator_second = (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
    if abs(denominator) / (first_length * second_length) <= COORDINATE_TOLERANCE:
        if (
            abs(numerator_first) / second_length > COORDINATE_TOLERANCE
            or abs(numerator_second) / first_length > COORDINATE_TOLERANCE
        ):
            return "none", None
        if any(_point_on_segment(point, *first) for point in second) or any(
            _point_on_segment(point, *second) for point in first
        ):
            return "overlap", None
        return "none", None
    first_fraction = numerator_first / denominator
    second_fraction = numerator_second / denominator
    first_tolerance = COORDINATE_TOLERANCE / first_length
    second_tolerance = COORDINATE_TOLERANCE / second_length
    if (
        -first_tolerance <= first_fraction <= 1.0 + first_tolerance
        and -second_tolerance <= second_fraction <= 1.0 + second_tolerance
    ):
        return "point", Point(
            a.x + first_fraction * (b.x - a.x),
            a.y + first_fraction * (b.y - a.y),
        )
    return "none", None


class _PointGraph:
    """Geometric union-find whose keys are scene points, never hidden net IDs."""

    def __init__(self) -> None:
        self.points: list[Point] = []
        self.parent: list[int] = []
        self.ground_roots: set[int] = set()

    def index(self, point: Point) -> int:
        if not _finite_point(point):
            raise _layout_fail("diagram geometry contains a non-finite point")
        for index, existing in enumerate(self.points):
            if _same_point(point, existing):
                return index
        index = len(self.points)
        self.points.append(point)
        self.parent.append(index)
        return index

    def find(self, index: int) -> int:
        parent = self.parent[index]
        if parent != index:
            self.parent[index] = self.find(parent)
        return self.parent[index]

    def union(self, *indexes: int) -> None:
        if not indexes:
            return
        roots = [self.find(index) for index in indexes]
        root = min(roots)
        for item in roots:
            self.parent[item] = root

    def mark_ground(self, point: Point) -> None:
        self.ground_roots.add(self.find(self.index(point)))

    def normalize_ground(self) -> None:
        roots = {self.find(root) for root in self.ground_roots}
        if roots:
            self.union(*roots)
            self.ground_roots = {self.find(next(iter(roots)))}

    def root(self, point: Point) -> int:
        return self.find(self.index(point))

    def is_ground(self, root: int) -> bool:
        return self.find(root) in {self.find(item) for item in self.ground_roots}


@dataclass(frozen=True, slots=True)
class _Region:
    source: SubsystemRegion
    path: _Path
    parent: _Path | None


@dataclass(frozen=True, slots=True)
class _Box:
    source: ElectricalBox
    path: _Path
    parent: _Path


@dataclass(frozen=True, slots=True)
class _ObservedBranch:
    source: NativeSymbol
    path: _Path
    owner: _Path
    role: str
    pins: tuple[str, str]
    points: tuple[Point, Point]
    nets: tuple[str, str] = ("", "")

    @property
    def key(self) -> str:
        return _token(
            "authoring_branch",
            component_path=list(self.path),
            branch_role=self.role,
        )

    @property
    def reciprocal(self) -> bool:
        return self.source.kind in {"R", "C"}

    @property
    def unordered_nets(self) -> tuple[str, str]:
        return cast(tuple[str, str], tuple(sorted(self.nets)))


@dataclass(frozen=True, slots=True)
class _ObservedSite:
    source: BoundarySite
    path: _Path
    pin_id: str
    net: str = ""
    peer_kind: str = ""
    peer_id: str = ""

    @property
    def key(self) -> str:
        return _token(
            "boundary_site",
            component_path=list(self.path),
            pin_id=self.pin_id,
        )

    @property
    def peer_key(self) -> str:
        return _token("diagram_peer", peer_kind=self.peer_kind, peer_id=self.peer_id)


@dataclass(frozen=True, slots=True)
class DiagramAuditData:
    """Immutable internal certificate data consumed by the public audit facade."""

    representation: _Representation
    plan_id: str
    plan_sha256: str
    connectivity_sha256: str
    semantic_sha256: str
    compiled_graph_sha256: str | None
    expanded_graph_sha256: str | None
    presentation_sha256: str | None
    observed_electrical: Mapping[str, object]
    observed_semantic: Mapping[str, object]
    observed_rows: tuple[Mapping[str, object], ...]
    verified_rows: tuple[Mapping[str, object], ...]


def _visible_bounds(bounds: Iterable[Bounds]) -> Bounds:
    materialized = tuple(bounds)
    if not materialized:
        raise _layout_fail("visible object has no geometric extent")
    return Bounds(
        min(item.xmin for item in materialized),
        min(item.ymin for item in materialized),
        max(item.xmax for item in materialized),
        max(item.ymax for item in materialized),
    )


def _verify_rectangular_frame(path: Path, bounds: Bounds, *, role: str) -> None:
    wanted = (
        Point(bounds.xmin, bounds.ymin),
        Point(bounds.xmax, bounds.ymin),
        Point(bounds.xmax, bounds.ymax),
        Point(bounds.xmin, bounds.ymax),
        Point(bounds.xmin, bounds.ymin),
    )
    if (
        path.role != role
        or not path.closed
        or path.codes is not None
        or len(path.points) != len(wanted)
        or any(not _same_point(actual, expected) for actual, expected in zip(path.points, wanted, strict=True))
    ):
        raise _layout_fail("visible rectangular frame disagrees with its occupied bounds", role=role)


def _region_model(scene: NeutralScene) -> tuple[_Region, ...]:
    if not scene.regions:
        raise _layout_fail("diagram scene has no root envelope")
    if any(not _finite_bounds(region.bounds) for region in scene.regions):
        raise _layout_fail("diagram region contains non-finite bounds")
    for region in scene.regions:
        _verify_rectangular_frame(region.boundary, region.bounds, role="region")
        if not _point_in_bounds(region.identity_anchor, region.bounds):
            raise _layout_fail("region identity anchor lies outside its visible boundary")
        if region.header is not None and (
            not _same_point(region.header.origin, region.identity_anchor)
            or not region.bounds.contains(region.header.bounds)
        ):
            raise _layout_fail("Composite header is not visibly anchored inside its region")
    outer = [
        region
        for region in scene.regions
        if all(region is other or region.bounds.contains(other.bounds) for other in scene.regions)
    ]
    if len(outer) != 1:
        raise _audit_fail("visible geometry does not establish one unique root region")
    root = outer[0]
    if root.kind != "root" or root.header is not None:
        raise _audit_fail("the root envelope cannot emit a duplicate always-visible identity header")
    parent_by_source: dict[int, SubsystemRegion] = {}
    for region in scene.regions:
        if region is root:
            continue
        if region.kind != "composite":
            raise _audit_fail("a non-root region does not carry the visible Composite role")
        if region.header is None:
            raise _audit_fail("every visible Composite region needs an identity header")
        containers = [
            candidate
            for candidate in scene.regions
            if candidate is not region and _strictly_contains(candidate.bounds, region.bounds)
        ]
        if not containers:
            raise _layout_fail("a Composite region lies outside the root envelope")
        parent_by_source[id(region)] = min(containers, key=lambda item: _area(item.bounds))
    paths: dict[int, _Path] = {id(root): ()}

    def resolve(region: SubsystemRegion) -> _Path:
        existing = paths.get(id(region))
        if existing is not None:
            return existing
        parent = parent_by_source[id(region)]
        parent_path = resolve(parent)
        assert region.header is not None
        local_id = region.header.text
        if not local_id:
            raise _audit_fail("Composite region identity is visibly empty")
        result = (*parent_path, local_id)
        paths[id(region)] = result
        return result

    for region in scene.regions:
        resolve(region)
    by_parent: dict[_Path, list[str]] = defaultdict(list)
    for region in scene.regions:
        if region is not root:
            path = paths[id(region)]
            by_parent[path[:-1]].append(path[-1])
    if any(len(names) != len(set(names)) for names in by_parent.values()):
        raise _audit_fail("sibling Composite headers do not establish unique local identities")
    for first_index, first in enumerate(scene.regions):
        for second in scene.regions[first_index + 1 :]:
            if first is root or second is root:
                continue
            nested = first.bounds.contains(second.bounds) or second.bounds.contains(first.bounds)
            if not nested and first.bounds.overlaps(second.bounds):
                raise _layout_fail("sibling Composite regions overlap")
    return tuple(
        _Region(
            region,
            paths[id(region)],
            None if region is root else paths[id(parent_by_source[id(region)])],
        )
        for region in scene.regions
    )


def _deepest_region(bounds: Bounds, regions: Sequence[_Region]) -> _Region:
    containing = [region for region in regions if region.source.bounds.contains(bounds)]
    if not containing:
        raise _layout_fail("a complete occupied object lies outside the root envelope")
    return min(containing, key=lambda region: _area(region.source.bounds))


def _box_model(scene: NeutralScene, regions: Sequence[_Region]) -> tuple[_Box, ...]:
    boxes: list[_Box] = []
    paths: set[_Path] = set()
    for box in scene.boxes:
        if box.kind not in {"CPW", "MTL"} or box.orientation not in (0, 90, 180, 270):
            raise _audit_fail("a transmission-line box is not visibly native CPW/MTL geometry")
        _verify_rectangular_frame(box.outline, box.outline.bounds, role="box")
        visible = _visible_bounds(
            (
                box.outline.bounds,
                *(path.bounds for path in box.paths),
                box.title.bounds,
                box.kind_label.bounds,
                *((box.length_label.bounds,) if box.length_label is not None else ()),
                *((box.section_label.bounds,) if box.section_label is not None else ()),
                *(row.bounds for row in box.conductor_rows),
                *(label.bounds for label in box.anchor_labels),
                *((box.reference_label.bounds,) if box.reference_label is not None else ()),
            )
        )
        if not _same_bounds(visible, box.bounds):
            raise _layout_fail("transmission-line occupied bounds do not match its actual visible ink")
        endpoints = tuple(
            point
            for path in box.paths
            for point in (path.points[0], path.points[-1])
        )
        if any(
            sum(_same_point(anchor, endpoint) for endpoint in endpoints) != 1
            for _, anchor in box.anchors
        ):
            raise _audit_fail("transmission-line anchor is not one actual visible lead endpoint")
        owner = _deepest_region(box.bounds, regions)
        path = (*owner.path, box.title.text)
        if path in paths:
            raise _audit_fail("visible transmission-line IDs are not unique in their owning region")
        paths.add(path)
        boxes.append(_Box(box, path, owner.path))
    return tuple(boxes)


def _branch_role(symbol: NativeSymbol) -> str:
    if symbol.kind not in _NATIVE_KINDS:
        raise _audit_fail("scene contains a symbol outside the closed native whitelist")
    branch_label = None if symbol.branch_label is None else symbol.branch_label.text
    if symbol.kind == "R":
        if branch_label not in (None, "resistance"):
            raise _audit_fail("resistor carries an unsupported visible branch badge")
        return "resistance"
    if symbol.kind == "C":
        if branch_label is None or branch_label == "capacitance":
            return "capacitance"
        if branch_label == "Cj":
            return "junction_capacitance"
        raise _audit_fail("capacitor carries an unsupported visible branch badge")
    if symbol.kind == "L":
        if branch_label in (None, "self", "inductor:self"):
            return "inductor:self"
        if branch_label.startswith("inductor:") and len(branch_label) > len("inductor:"):
            return branch_label
        raise _audit_fail("inductor carries an unsupported visible branch badge")
    if symbol.kind == "JJ":
        if branch_label not in (None, "josephson_inductance"):
            raise _audit_fail("Josephson symbol carries an unsupported visible branch badge")
        return "josephson_inductance"
    if branch_label is None:
        return "conductance"
    return branch_label


def _same_bounds(left: Bounds, right: Bounds) -> bool:
    return all(
        _same(first, second)
        for first, second in zip(
            (left.xmin, left.ymin, left.xmax, left.ymax),
            (right.xmin, right.ymin, right.xmax, right.ymax),
            strict=True,
        )
    )


def _same_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left.role == right.role
        and left.closed == right.closed
        and left.codes == right.codes
        and len(left.points) == len(right.points)
        and all(_same_point(first, second) for first, second in zip(left.points, right.points, strict=True))
    )


def _verify_text_run(run: object) -> None:
    from .metrics import shape_text
    from .scene import TextRun

    if not isinstance(run, TextRun):
        raise _audit_fail("scene text fact is not one immutable TextRun")
    try:
        reference = shape_text(run.text, at=run.origin, size=run.size, role=run.role)
    except Exception as error:
        raise _audit_fail("visible text cannot be reconstructed by the trusted font grammar") from error
    if (
        not _same_bounds(run.bounds, reference.bounds)
        or not _same(run.ascent, reference.ascent)
        or not _same(run.descent, reference.descent)
        or len(run.glyphs) != len(reference.glyphs)
    ):
        raise _audit_fail("visible text bounds/metrics do not match its literal glyphs", text=run.text)
    for actual, wanted in zip(run.glyphs, reference.glyphs, strict=True):
        if (
            actual.character != wanted.character
            or actual.font_sha256 != wanted.font_sha256
            or actual.codes != wanted.codes
            or not _same_bounds(actual.bounds, wanted.bounds)
            or not _same(actual.advance, wanted.advance)
            or len(actual.vertices) != len(wanted.vertices)
            or any(
                not _same_point(first, second)
                for first, second in zip(actual.vertices, wanted.vertices, strict=True)
            )
        ):
            raise _audit_fail("visible glyph outlines do not encode their claimed text", text=run.text)


_GraphicOwner = tuple[str, int]


def _scene_text_groups(scene: NeutralScene) -> tuple[tuple[_GraphicOwner, TextRun], ...]:
    groups: list[tuple[_GraphicOwner, TextRun]] = []

    def extend(owner: _GraphicOwner, runs: Iterable[TextRun | None]) -> None:
        groups.extend((owner, run) for run in runs if run is not None)

    for index, run in enumerate(scene.text):
        extend(("scene_text", index), (run,))
    for index, symbol in enumerate(scene.symbols):
        extend(("symbol", index), (symbol.visible_name, symbol.value, symbol.branch_label))
    for index, port in enumerate(scene.ports):
        extend(
            ("port", index),
            (
                *port.labels,
                port.reference_load.visible_name,
                port.reference_load.value,
                port.reference_load.branch_label,
            ),
        )
    for index, box in enumerate(scene.boxes):
        extend(
            ("box", index),
            (
                box.title,
                box.kind_label,
                box.length_label,
                box.section_label,
                *box.conductor_rows,
                *box.anchor_labels,
                box.reference_label,
            ),
        )
    for index, region in enumerate(scene.regions):
        extend(("region", index), (region.header,))
    for index, site in enumerate(scene.boundary_sites):
        extend(("boundary_site", index), (site.visible_label,))
    for index, mark in enumerate(scene.node_marks):
        extend(("node_mark", index), (mark.label,))
    for index, guide in enumerate(scene.guides):
        extend(("guide", index), (guide.label,))
    if scene.provenance_band is not None:
        extend(("provenance", 0), scene.provenance_band.lines)
    return tuple(groups)


def _verify_scene_text(scene: NeutralScene) -> None:
    for _, run in _scene_text_groups(scene):
        _verify_text_run(run)


def _inked_glyph_bounds(run: TextRun) -> tuple[Bounds, ...]:
    """Return actual character ink boxes, excluding advancing whitespace."""

    return tuple(glyph.bounds for glyph in run.glyphs if glyph.vertices)


def _path_stroke_bounds(path: Path) -> tuple[Bounds, ...]:
    """Bound each emitted line/Bézier stroke without filling its contour."""

    if path.codes is None:
        return tuple(
            Bounds.around((left, right))
            for left, right in zip(path.points, path.points[1:])
        )

    def bezier(coordinates: Sequence[float]) -> tuple[float, float]:
        candidates = [0.0, 1.0]
        if len(coordinates) == 3:
            first, control, last = coordinates
            denominator = first - 2.0 * control + last
            if abs(denominator) > COORDINATE_TOLERANCE:
                candidates.append((first - control) / denominator)
            value = lambda t: (1.0 - t) ** 2 * first + 2.0 * (1.0 - t) * t * control + t**2 * last
        elif len(coordinates) == 4:
            first, control_1, control_2, last = coordinates
            a = -first + 3.0 * control_1 - 3.0 * control_2 + last
            b = 2.0 * (first - 2.0 * control_1 + control_2)
            c = control_1 - first
            if abs(a) <= COORDINATE_TOLERANCE:
                if abs(b) > COORDINATE_TOLERANCE:
                    candidates.append(-c / b)
            else:
                discriminant = b * b - 4.0 * a * c
                if discriminant >= 0.0:
                    root = math.sqrt(discriminant)
                    candidates.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
            value = lambda t: (
                (1.0 - t) ** 3 * first
                + 3.0 * (1.0 - t) ** 2 * t * control_1
                + 3.0 * (1.0 - t) * t**2 * control_2
                + t**3 * last
            )
        else:
            raise _audit_fail("visible curved path has an unsupported Bézier order")
        values = [value(t) for t in candidates if 0.0 <= t <= 1.0]
        return min(values), max(values)

    points, codes = path.points, path.codes
    if not points or codes[0] != 1:
        raise _audit_fail("visible coded path does not begin with MOVETO")
    strokes: list[Bounds] = []
    current = points[0]
    subpath_start = current
    index = 1
    while index < len(points):
        code = codes[index]
        if code == 1:
            current = points[index]
            subpath_start = current
            index += 1
        elif code == 2:
            following = points[index]
            strokes.append(Bounds.around((current, following)))
            current = following
            index += 1
        elif code == 3:
            if index + 1 >= len(points) or codes[index + 1] != 3:
                raise _audit_fail("visible quadratic path has incomplete control points")
            control, following = points[index], points[index + 1]
            xmin, xmax = bezier((current.x, control.x, following.x))
            ymin, ymax = bezier((current.y, control.y, following.y))
            strokes.append(Bounds(xmin, ymin, xmax, ymax))
            current = following
            index += 2
        elif code == 4:
            if index + 2 >= len(points) or codes[index : index + 3] != (4, 4, 4):
                raise _audit_fail("visible cubic path has incomplete control points")
            control_1, control_2, following = points[index : index + 3]
            xmin, xmax = bezier((current.x, control_1.x, control_2.x, following.x))
            ymin, ymax = bezier((current.y, control_1.y, control_2.y, following.y))
            strokes.append(Bounds(xmin, ymin, xmax, ymax))
            current = following
            index += 3
        elif code == 79:
            strokes.append(Bounds.around((current, subpath_start)))
            current = subpath_start
            index += 1
        else:
            raise _audit_fail("visible path uses an unsupported drawing code", code=code)
    return tuple(strokes)


def _verify_label_group_clearance(
    *,
    object_kind: str,
    identity: str,
    runs: Sequence[object],
    paths: Sequence[Path],
) -> None:
    """Reject local label collisions using only emitted glyph and stroke bounds."""

    materialized = [run for run in runs if isinstance(run, TextRun)]
    labelled_glyphs = [
        (run_index, run.text, glyph_bounds)
        for run_index, run in enumerate(materialized)
        for glyph_bounds in _inked_glyph_bounds(run)
    ]
    for _, text, glyph_bounds in labelled_glyphs:
        for path in paths:
            if any(
                glyph_bounds.overlaps(bounds) for bounds in _path_stroke_bounds(path)
            ):
                raise _layout_fail(
                    "visible label glyph overlaps its owning object's non-text ink",
                    object_kind=object_kind,
                    identity=identity,
                    text=text,
                )
    for index, (left_run, left_text, left) in enumerate(labelled_glyphs):
        for right_run, right_text, right in labelled_glyphs[index + 1 :]:
            if left_run != right_run and left.overlaps(right):
                raise _layout_fail(
                    "visible labels overlap inside one graphical object",
                    object_kind=object_kind,
                    identity=identity,
                    first_text=left_text,
                    second_text=right_text,
                )


def _verify_global_clearance(scene: NeutralScene) -> None:
    """Reject cross-object text/text and text/geometry collisions."""

    groups = _scene_text_groups(scene)
    inked = tuple(
        (owner, run, _inked_glyph_bounds(run)) for owner, run in groups
    )
    for index, (left_owner, left_run, left_glyphs) in enumerate(inked):
        for right_owner, right_run, right_glyphs in inked[index + 1 :]:
            if not left_run.bounds.overlaps(right_run.bounds):
                continue
            if any(
                left.overlaps(right)
                for left in left_glyphs
                for right in right_glyphs
            ):
                raise _layout_fail(
                    "visible TextRuns overlap in the emitted scene",
                    first_owner=list(left_owner),
                    first_text=left_run.text,
                    second_owner=list(right_owner),
                    second_text=right_run.text,
                )

    occupied: list[tuple[_GraphicOwner, Bounds]] = []
    strokes: list[tuple[_GraphicOwner | None, Bounds]] = []
    for index, symbol in enumerate(scene.symbols):
        owner = ("symbol", index)
        occupied.append((owner, symbol.occupied_bounds))
        for path in (*symbol.paths, *((symbol.reference_polarity,) if symbol.reference_polarity else ())):
            strokes.extend((owner, bounds) for bounds in _path_stroke_bounds(path))
    for index, port in enumerate(scene.ports):
        owner = ("port", index)
        occupied.append((owner, port.occupied_bounds))
        load = port.reference_load
        paths = (
            port.circle,
            *port.paths,
            *load.paths,
            *((load.reference_polarity,) if load.reference_polarity else ()),
        )
        for path in paths:
            strokes.extend((owner, bounds) for bounds in _path_stroke_bounds(path))
    for index, box in enumerate(scene.boxes):
        owner = ("box", index)
        occupied.append((owner, box.bounds))
        for path in (box.outline, *box.paths):
            strokes.extend((owner, bounds) for bounds in _path_stroke_bounds(path))
    for wire in scene.conductive:
        strokes.extend((None, bounds) for bounds in _path_stroke_bounds(Path(wire.points, "wire")))
    for jump in scene.jumps:
        strokes.extend((None, bounds) for bounds in _path_stroke_bounds(jump.path))
    for index, region in enumerate(scene.regions):
        owner = ("region", index)
        strokes.extend((owner, bounds) for bounds in _path_stroke_bounds(region.boundary))
    for index, mark in enumerate(scene.node_marks):
        owner = ("node_mark", index)
        for path in mark.paths:
            strokes.extend((owner, bounds) for bounds in _path_stroke_bounds(path))
    for index, guide in enumerate(scene.guides):
        owner = ("guide", index)
        for path in guide.paths:
            strokes.extend((owner, bounds) for bounds in _path_stroke_bounds(path))

    for owner, run, glyphs in inked:
        occupied_candidates = [
            (other_owner, bounds)
            for other_owner, bounds in occupied
            if other_owner != owner and run.bounds.overlaps(bounds)
        ]
        stroke_candidates = [
            (other_owner, bounds)
            for other_owner, bounds in strokes
            if other_owner != owner and run.bounds.overlaps(bounds)
        ]
        for glyph in glyphs:
            occupied_collisions = [
                other_owner
                for other_owner, bounds in occupied_candidates
                if glyph.overlaps(bounds)
            ]
            if occupied_collisions:
                raise _layout_fail(
                    "visible text overlaps a foreign graphical object",
                    owner=list(owner),
                    text=run.text,
                    foreign_owner=list(occupied_collisions[0]),
                )
            stroke_collisions = [
                other_owner
                for other_owner, bounds in stroke_candidates
                if glyph.overlaps(bounds)
            ]
            if stroke_collisions:
                raise _layout_fail(
                    "visible text overlaps a foreign non-text stroke",
                    owner=list(owner),
                    text=run.text,
                    foreign_owner=(
                        None
                        if stroke_collisions[0] is None
                        else list(stroke_collisions[0])
                    ),
                )


def _verify_internal_clearance(scene: NeutralScene) -> None:
    """Check local visual clearance not established by trusted shape identity."""

    for symbol in scene.symbols:
        visible = _visible_bounds(
            (
                symbol.symbol_bounds,
                symbol.visible_name.bounds,
                *((symbol.value.bounds,) if symbol.value is not None else ()),
                *((symbol.branch_label.bounds,) if symbol.branch_label is not None else ()),
                *((symbol.reference_polarity.bounds,) if symbol.reference_polarity is not None else ()),
            )
        )
        if not _same_bounds(visible, symbol.occupied_bounds):
            raise _layout_fail(
                "native symbol occupied bounds do not match its actual visible ink",
                identity=symbol.visible_name.text,
            )
        paths = (*symbol.paths, *((symbol.reference_polarity,) if symbol.reference_polarity is not None else ()))
        runs = (symbol.visible_name, symbol.value, symbol.branch_label)
        _verify_label_group_clearance(
            object_kind="native_symbol",
            identity=symbol.visible_name.text,
            runs=runs,
            paths=paths,
        )
        if symbol.reference_polarity is not None and symbol.reference_polarity.bounds.overlaps(symbol.symbol_bounds):
            raise _layout_fail(
                "native reference-polarity signs overlap their owning symbol body",
                identity=symbol.visible_name.text,
            )

    for port in scene.ports:
        load = port.reference_load
        load_visible = _visible_bounds(
            (
                load.symbol_bounds,
                load.visible_name.bounds,
                *((load.value.bounds,) if load.value is not None else ()),
                *((load.branch_label.bounds,) if load.branch_label is not None else ()),
                *((load.reference_polarity.bounds,) if load.reference_polarity is not None else ()),
            )
        )
        if not _same_bounds(load_visible, load.occupied_bounds):
            raise _layout_fail("Port load occupied bounds do not match its actual visible ink")
        if any(
            sum(_same_path(ground_path, path) for path in port.paths) != 1
            for ground_path in port.ground_glyph
        ):
            raise _audit_fail("Port ground glyph is not part of its actual emitted stroke set")
        port_visible = _visible_bounds(
            (
                port.circle.bounds,
                load.occupied_bounds,
                *(path.bounds for path in port.paths),
                *(label.bounds for label in port.labels),
            )
        )
        if not _same_bounds(port_visible, port.occupied_bounds):
            raise _layout_fail("Port occupied bounds do not match its actual visible ink")
        load_paths = (*load.paths, *((load.reference_polarity,) if load.reference_polarity is not None else ()))
        port_paths = (port.circle, *port.paths, *load_paths)
        port_runs = (
            *port.labels,
            load.visible_name,
            load.value,
            load.branch_label,
        )
        identity = port.labels[0].text if port.labels else "<missing-port-label>"
        _verify_label_group_clearance(
            object_kind="port",
            identity=identity,
            runs=port_runs,
            paths=port_paths,
        )

    for box in scene.boxes:
        visible = _visible_bounds(
            (
                box.outline.bounds,
                *(path.bounds for path in box.paths),
                box.title.bounds,
                box.kind_label.bounds,
                *((box.length_label.bounds,) if box.length_label is not None else ()),
                *((box.section_label.bounds,) if box.section_label is not None else ()),
                *(row.bounds for row in box.conductor_rows),
                *(label.bounds for label in box.anchor_labels),
                *((box.reference_label.bounds,) if box.reference_label is not None else ()),
            )
        )
        if not _same_bounds(visible, box.bounds):
            raise _layout_fail("transmission-line occupied bounds do not match its actual visible ink")
        runs = (
            box.title,
            box.kind_label,
            box.length_label,
            box.section_label,
            *box.conductor_rows,
            *box.anchor_labels,
            box.reference_label,
        )
        _verify_label_group_clearance(
            object_kind="transmission_line_box",
            identity=box.title.text,
            runs=runs,
            paths=(box.outline, *box.paths),
        )


def _verify_native_geometry(symbol: NativeSymbol, representation: _Representation) -> None:
    """Match actual ink against the trusted native-template grammar."""

    anchors = tuple(symbol.anchors)
    if len(anchors) != 2 or {name for name, _ in anchors} != {"terminal_1", "terminal_2"}:
        raise _audit_fail("native symbol does not expose the closed terminal_1/terminal_2 grammar")
    start, end = anchors[0][1], anchors[1][1]
    from .metrics import DEFAULT_METRICS
    from .native import native_block

    try:
        references = tuple(
            native_block(
                symbol.kind,
                start=start,
                end=end,
                name="native-geometry-reference",
                value=None,
                metrics=DEFAULT_METRICS,
                label_side=side,
            )
            for start, end in ((start, end), (end, start))
            for side in ("top", "bottom", "left", "right")
        )
    except Exception as error:
        raise _layout_fail("native symbol anchors do not form one exact trusted 1U primitive") from error
    if not any(
        symbol.orientation == reference.orientation
        and len(symbol.paths) == len(reference.paths)
        and all(
            _same_path(actual, wanted)
            for actual, wanted in zip(symbol.paths, reference.paths, strict=True)
        )
        and _same_bounds(symbol.symbol_bounds, reference.symbol_bounds)
        and _same_path(symbol.reference_polarity, reference.reference_polarity)
        for reference in references
    ):
        raise _audit_fail(
            "native symbol kind/anchors do not match its trusted visible glyph geometry",
            visible_name=symbol.visible_name.text,
            native_kind=symbol.kind,
        )


def _branch_model(
    scene: NeutralScene, regions: Sequence[_Region], representation: _Representation
) -> tuple[_ObservedBranch, ...]:
    branches: list[_ObservedBranch] = []
    seen: set[str] = set()
    occupied: list[tuple[str, Bounds]] = []
    for symbol in scene.symbols:
        _verify_native_geometry(symbol, representation)
        if not _finite_bounds(symbol.occupied_bounds) or not symbol.occupied_bounds.contains(symbol.symbol_bounds):
            raise _layout_fail("native symbol occupied bounds are malformed")
        if symbol.orientation not in (0, 90, 180, 270):
            raise _layout_fail("native symbol orientation is not cardinal")
        anchors = tuple(symbol.anchors)
        if len(anchors) != 2 or len({name for name, _ in anchors}) != 2:
            raise _audit_fail("every visible native branch needs two named electrical anchors")
        owner = _deepest_region(symbol.occupied_bounds, regions)
        path = (*owner.path, symbol.visible_name.text)
        role = _branch_role(symbol)
        branch = _ObservedBranch(
            symbol,
            path,
            owner.path,
            role,
            cast(tuple[str, str], tuple(name for name, _ in anchors)),
            cast(tuple[Point, Point], tuple(point for _, point in anchors)),
        )
        if branch.key in seen:
            raise _audit_fail("a visible native branch identity is duplicated", branch_id=branch.key)
        seen.add(branch.key)
        branches.append(branch)
        occupied.append((branch.key, symbol.occupied_bounds))
        if representation == "authoring" and symbol.kind == "G":
            raise _audit_fail("authoring projection contains a compiled-only conductance symbol")
    for index, (left_id, left) in enumerate(occupied):
        for right_id, right in occupied[index + 1 :]:
            if left.overlaps(right):
                raise _layout_fail(
                    "native symbol occupied bounds overlap",
                    first_branch=left_id,
                    second_branch=right_id,
                )
    return tuple(branches)


def _conductive_paths(scene: NeutralScene) -> tuple[tuple[Point, ...], ...]:
    paths: list[tuple[Point, ...]] = [wire.points for wire in scene.conductive]
    # Native bodies are branches, never wires through their glyph.  Their
    # endpoint-adjacent lead strokes are nevertheless real conductor ink and
    # must support an intentionally singleton terminal as well as a visible T.
    # Add only the one segment incident to each validated terminal anchor.
    for symbol in scene.symbols:
        if symbol.kind == "JJ":
            # Schemdraw's Josephson cross includes two axial strokes meeting
            # at its center. That is branch-body ink, not a galvanic short.
            # Only the verified native terminal stubs enter the wire graph.
            from .metrics import DEFAULT_METRICS

            left, right = (point for _, point in symbol.anchors)
            span = math.hypot(right.x - left.x, right.y - left.y)
            for at, other in ((left, right), (right, left)):
                fraction = DEFAULT_METRICS.terminal_stub / span
                paths.append((at, at.translated(
                    (other.x - at.x) * fraction, (other.y - at.y) * fraction
                )))
            continue
        for _, anchor in symbol.anchors:
            for path in symbol.paths:
                if path.codes is not None or len(path.points) < 2:
                    continue
                if _same_point(anchor, path.points[0]):
                    paths.append(path.points[:2])
                elif _same_point(anchor, path.points[-1]):
                    paths.append(path.points[-2:])
    for box in scene.boxes:
        paths.extend(path.points for path in box.paths if path.role == "wire")
    for port in scene.ports:
        paths.extend(path.points for path in port.paths if path.role == "wire")
    return tuple(paths)


def _is_ground_paths(paths: Sequence[Path], terminal: Point) -> bool:
    if (
        len(paths) != 4
        or any(path.role not in {"symbol", "glyph"} or path.closed or len(path.points) != 2 for path in paths)
    ):
        return False
    stems = [path for path in paths if any(_same_point(terminal, point) for point in path.points)]
    if len(stems) != 1:
        return False
    stem = stems[0]
    stem_end = next(point for point in stem.points if not _same_point(point, terminal))
    vx, vy = stem_end.x - terminal.x, stem_end.y - terminal.y
    length = math.hypot(vx, vy)
    if length <= COORDINATE_TOLERANCE:
        return False
    ux, uy = vx / length, vy / length
    bars = [path for path in paths if path is not stem]
    if any(
        abs((path.points[1].x - path.points[0].x) * ux + (path.points[1].y - path.points[0].y) * uy)
        > COORDINATE_TOLERANCE
        for path in bars
    ):
        return False
    centers = [
        Point(
            (path.points[0].x + path.points[1].x) / 2.0,
            (path.points[0].y + path.points[1].y) / 2.0,
        )
        for path in bars
    ]
    if any(
        abs((center.x - terminal.x) * uy - (center.y - terminal.y) * ux)
        > COORDINATE_TOLERANCE
        for center in centers
    ):
        return False
    ordered = sorted(
        zip(bars, centers),
        key=lambda item: (item[1].x - terminal.x) * ux + (item[1].y - terminal.y) * uy,
    )
    widths = [
        math.hypot(path.points[1].x - path.points[0].x, path.points[1].y - path.points[0].y)
        for path, _ in ordered
    ]
    distances = [
        (center.x - terminal.x) * ux + (center.y - terminal.y) * uy
        for _, center in ordered
    ]
    return (
        _same_point(ordered[0][1], stem_end)
        and widths[0] > widths[1] > widths[2] > COORDINATE_TOLERANCE
        and 0.0 < distances[0] < distances[1] < distances[2]
    )


def _build_point_graph(
    scene: NeutralScene,
    *,
    normalize_ground: bool = True,
) -> tuple[_PointGraph, tuple[tuple[Point, Point], ...]]:
    graph = _PointGraph()
    conductive = _conductive_paths(scene)
    segments: list[tuple[Point, Point]] = []
    for points in conductive:
        if len(points) < 2:
            raise _layout_fail("a conductive path has fewer than two points")
        indexes = [graph.index(point) for point in points]
        graph.union(*indexes)
        segments.extend(zip(points, points[1:]))
    for jump in scene.jumps:
        points = jump.path.points
        if (
            jump.path.role != "jump"
            or jump.path.closed
            or jump.path.codes != (1, 4, 4, 4)
            or len(points) != 4
        ):
            raise _layout_fail("wire-jump geometry is not one visible open cubic path")
        start, end = points[0], points[-1]
        if _same_point(start, end) or not (_same(start.x, end.x) or _same(start.y, end.y)):
            raise _layout_fail("wire-jump endpoints do not span one cardinal conductor gap")
        if not all(any(_same_point(endpoint, tip) for segment in segments for tip in segment) for endpoint in (start, end)):
            raise _layout_fail("wire-jump endpoints do not terminate actual split conductive strokes")
        midpoint = Point((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)
        horizontal = _same(start.y, end.y)
        if any(
            _point_on_segment(midpoint, *segment)
            and ((_same(segment[0].y, segment[1].y)) if horizontal else (_same(segment[0].x, segment[1].x)))
            for segment in segments
        ):
            raise _layout_fail("wire-jump straight interval still contains conductive ink")
        graph.union(graph.index(start), graph.index(end))
    for first_index, first in enumerate(segments):
        for second in segments[first_index + 1 :]:
            relation, point = _segment_intersection(first, second)
            if relation == "none":
                continue
            if relation == "overlap":
                shared = [
                    endpoint
                    for endpoint in (*first, *second)
                    if _point_on_segment(endpoint, *first) and _point_on_segment(endpoint, *second)
                ]
                graph.union(*(graph.index(endpoint) for endpoint in shared))
                continue
            assert point is not None
            first_interior = not any(_same_point(point, endpoint) for endpoint in first)
            second_interior = not any(_same_point(point, endpoint) for endpoint in second)
            if first_interior and second_interior:
                raise _layout_fail(
                    "different conductive strokes form an ambiguous implicit crossing",
                    first_segment=[{"x": item.x, "y": item.y} for item in first],
                    second_segment=[{"x": item.x, "y": item.y} for item in second],
                    crossing={"x": point.x, "y": point.y},
                )
            graph.union(graph.index(first[0]), graph.index(second[0]), graph.index(point))
    visible_contact_points = [
        point
        for branch in scene.symbols
        for _, point in branch.anchors
    ]
    visible_contact_points.extend(site.point for site in scene.boundary_sites)
    visible_contact_points.extend(mark.point for mark in scene.node_marks)
    visible_contact_points.extend(
        point
        for box in scene.boxes
        for _, point in box.anchors
    )
    visible_contact_points.extend(
        point
        for port in scene.ports
        for point in (
            port.boundary_anchor,
            port.circuit_anchor,
            port.external_anchor,
            port.load_anchor,
            port.ground_anchor,
        )
    )
    visible_contact_points.extend(
        point
        for port in scene.ports
        for _, point in port.reference_load.anchors
    )
    for port in scene.ports:
        if not _is_ground_paths(port.ground_glyph, port.ground_anchor):
            raise _audit_fail("Port local return is not a recognized visible ground glyph")
    for point in visible_contact_points:
        contacts = [segment for segment in segments if _point_on_segment(point, *segment)]
        ground_contact = any(
            guide.label is None
            and guide.paths
            and all(path.role in {"symbol", "glyph"} for path in guide.paths)
            and any(_same_point(point, terminal) for terminal in guide.terminals)
            for guide in scene.guides
        ) or any(_same_point(point, port.ground_anchor) for port in scene.ports)
        if not contacts and not ground_contact:
            raise _layout_fail("a registered electrical anchor has no actual conductive contact")
        graph.union(graph.index(point), *(graph.index(segment[0]) for segment in contacts))
    ground_guides = [
        guide
        for guide in scene.guides
        if guide.label is None
        and len(guide.terminals) == 1
        and _is_ground_paths(guide.paths, guide.terminals[0])
    ]
    duplicated_port_ground = [
        guide
        for guide in ground_guides
        if any(
            _same_point(guide.terminals[0], port.ground_anchor)
            for port in scene.ports
        )
    ]
    if duplicated_port_ground:
        raise _audit_fail(
            "Port local return duplicates its intrinsic visible ground glyph"
        )
    suspicious_ground_ink = [
        guide
        for guide in scene.guides
        if guide.label is None
        and guide.paths
        and all(path.role in {"symbol", "glyph"} for path in guide.paths)
        and guide not in ground_guides
    ]
    if suspicious_ground_ink:
        raise _audit_fail("an unlabeled native-symbol guide is not a recognized ground glyph")
    for guide in ground_guides:
        terminal = guide.terminals[0]
        anchor_contact = any(
            _same_point(terminal, point)
            for symbol in scene.symbols
            for _, point in symbol.anchors
        ) or any(
            _same_point(terminal, point)
            for port in scene.ports
            for _, point in port.reference_load.anchors
        )
        contacts = [segment for segment in segments if _point_on_segment(terminal, *segment)]
        if not contacts and not anchor_contact:
            raise _layout_fail("a ground glyph does not contact a conductive return")
        graph.union(
            graph.index(terminal),
            *(graph.index(segment[0]) for segment in contacts),
        )
        graph.mark_ground(terminal)
    for port in scene.ports:
        graph.mark_ground(port.ground_anchor)
    if normalize_ground:
        graph.normalize_ground()
    return graph, tuple(segments)


def _contacted_net(
    graph: _PointGraph,
    point: Point,
    endpoints_by_root: Mapping[int, set[_Endpoint]],
) -> str:
    root = graph.root(point)
    return _net_key(endpoints_by_root.get(root, ()), ground=graph.is_ground(root))


def _infer_sites(
    scene: NeutralScene,
    regions: Sequence[_Region],
    boxes: Sequence[_Box],
) -> tuple[_ObservedSite, ...]:
    sites: list[_ObservedSite] = []
    seen: dict[str, Point] = {}

    def add_site(observed: _ObservedSite) -> None:
        previous = seen.get(observed.key)
        if previous is not None:
            if not _same_point(previous, observed.source.point):
                raise _audit_fail("a visible boundary-site identity is duplicated")
            return
        seen[observed.key] = observed.source.point
        sites.append(observed)

    for site in scene.boundary_sites:
        if site.visible_label is None:
            raise _audit_fail("every boundary site needs its visible local pin identity")
        matches: list[tuple[_Path, str]] = []
        for region in regions:
            if region.path and _point_on_rect_boundary(site.point, region.source.bounds):
                matches.append((region.path, site.visible_label.text))
        for box in boxes:
            for anchor, point in box.source.anchors:
                if _same_point(site.point, point):
                    matches.append((box.path, anchor))
        unique = sorted(set(matches))
        if len(unique) != 1:
            raise _audit_fail(
                "visible boundary-site geometry does not select one Subsystem boundary",
                label=site.visible_label.text,
                match_count=len(unique),
            )
        path, pin_id = unique[0]
        if site.visible_label.text != pin_id:
            raise _audit_fail("boundary-site label disagrees with its visible electrical anchor")
        observed = _ObservedSite(site, path, pin_id)
        add_site(observed)
    for box in boxes:
        if len(box.source.anchor_labels) != len(box.source.anchors):
            raise _audit_fail("CPW/MTL box omits a visible anchor label")
        for (pin_id, point), label in zip(
            box.source.anchors, box.source.anchor_labels, strict=True
        ):
            if label.text != pin_id:
                raise _audit_fail("CPW/MTL boundary label disagrees with its visible anchor")
            add_site(_ObservedSite(BoundarySite(point, label), box.path, pin_id))
    return tuple(sites)


def _port_identity(port: PortBlock) -> tuple[str, str, str, bool]:
    if len(port.labels) not in (1, 2):
        raise _audit_fail("Port block must visibly carry its ID and optional PTC mark")
    port_id = port.labels[0].text
    removable = len(port.labels) == 2
    if removable and port.labels[1].text != "PTC-removable":
        raise _audit_fail("nonloading Port carries an unrecognized PTC-removable mark")
    role = "nonloading_probe" if removable else "terminated"
    if (
        port.reference_load.value is not None
        or not port.reference_load.visible_name.text
    ):
        raise _audit_fail("Port omits its sole visible raw Z0 native load", port_id=port_id)
    return port_id, role, port.reference_load.visible_name.text, removable


def _same_text_placement(left: TextRun | None, right: TextRun | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left.text == right.text
        and left.role == right.role
        and _same_point(left.origin, right.origin)
        and _same(left.size, right.size)
        and _same_bounds(left.bounds, right.bounds)
        and _same(left.ascent, right.ascent)
        and _same(left.descent, right.descent)
    )


def _verify_port_block_geometry(port: PortBlock) -> None:
    """Match a Port's actual visible ink to the closed native Port grammar."""

    from .metrics import DEFAULT_METRICS
    from .native import port_block

    port_id, role, impedance, _ = _port_identity(port)
    dx = port.boundary_anchor.x - port.circuit_anchor.x
    dy = port.boundary_anchor.y - port.circuit_anchor.y
    if _same(dy, 0.0) and not _same(dx, 0.0):
        side = "right" if dx > 0.0 else "left"
    elif _same(dx, 0.0) and not _same(dy, 0.0):
        side = "top" if dy > 0.0 else "bottom"
    else:
        raise _audit_fail("Port boundary-to-circuit incidence is not cardinal")
    load_dx = port.load_anchor.x - port.circuit_anchor.x
    load_dy = port.load_anchor.y - port.circuit_anchor.y
    if _same(load_dy, 0.0) and not _same(load_dx, 0.0):
        load_side = "right" if load_dx > 0.0 else "left"
    elif _same(load_dx, 0.0) and not _same(load_dy, 0.0):
        load_side = "top" if load_dy > 0.0 else "bottom"
    else:
        raise _audit_fail("Port circuit-to-load incidence is not cardinal")
    if (side in {"left", "right"}) == (load_side in {"left", "right"}):
        raise _audit_fail("Port boundary and load incidence are not perpendicular")
    reference = port_block(
        port_id=port_id,
        role=cast(Literal["terminated", "nonloading_probe"], role),
        reference_impedance=impedance,
        boundary_anchor=port.boundary_anchor,
        side=cast(Literal["left", "right", "top", "bottom"], side),
        load_side=cast(Literal["left", "right", "top", "bottom"], load_side),
        metrics=DEFAULT_METRICS,
    )
    load = port.reference_load
    wanted_load = reference.reference_load
    if (
        not _same_path(port.circle, reference.circle)
        or not _same_point(port.circle_center, reference.circle_center)
        or any(
            not _same_point(actual, wanted)
            for actual, wanted in zip(
                (
                    port.boundary_anchor,
                    port.circuit_anchor,
                    port.external_anchor,
                    port.load_anchor,
                    port.ground_anchor,
                ),
                (
                    reference.boundary_anchor,
                    reference.circuit_anchor,
                    reference.external_anchor,
                    reference.load_anchor,
                    reference.ground_anchor,
                ),
                strict=True,
            )
        )
        or len(port.paths) != len(reference.paths)
        or any(
            not _same_path(actual, wanted)
            for actual, wanted in zip(port.paths, reference.paths, strict=True)
        )
        or len(port.ground_glyph) != len(reference.ground_glyph)
        or any(
            not _same_path(actual, wanted)
            for actual, wanted in zip(
                port.ground_glyph, reference.ground_glyph, strict=True
            )
        )
        or len(port.labels) != len(reference.labels)
        or any(
            not _same_text_placement(actual, wanted)
            for actual, wanted in zip(port.labels, reference.labels, strict=True)
        )
        or not _same_bounds(port.occupied_bounds, reference.occupied_bounds)
        or load.kind != "R"
        or len(load.anchors) != len(wanted_load.anchors)
        or any(
            actual_name != wanted_name or not _same_point(actual_point, wanted_point)
            for (actual_name, actual_point), (wanted_name, wanted_point) in zip(
                load.anchors, wanted_load.anchors, strict=True
            )
        )
        or not _same_bounds(load.symbol_bounds, wanted_load.symbol_bounds)
        or not _same_bounds(load.occupied_bounds, wanted_load.occupied_bounds)
        or not _same_text_placement(load.visible_name, wanted_load.visible_name)
        or not _same_text_placement(load.value, wanted_load.value)
        or not _same_text_placement(load.branch_label, wanted_load.branch_label)
    ):
        raise _audit_fail(
            "Port body/load ink does not match its visible native grammar",
            port_id=port_id,
        )
    _verify_native_geometry(load, "authoring")


def _parse_quantity(text: str, *, si_unit: str) -> Mapping[str, object]:
    try:
        quantity = units.registry.Quantity(text)
        return cast(
            Mapping[str, object],
            quantity_envelope(
                quantity,
                si_unit=si_unit,
                registry=units.registry,
            ),
        )
    except Exception as error:
        raise _audit_fail("visible parameter label is not a complete physical quantity", label=text) from error


def _native_value_rows(branches: Sequence[_ObservedBranch]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for branch in branches:
        if branch.source.value is None:
            continue
        si_unit = {
            "R": "ohm",
            "L": "henry",
            "C": "farad",
            "JJ": "henry",
            "G": "siemens",
        }[branch.source.kind]
        observed = _parse_quantity(branch.source.value.text, si_unit=si_unit)
        rows.append(
            {
                "category": "observed_scene",
                "kind": "displayed_baseline_value",
                "identity": branch.key,
                "value": _plain(observed),
            }
        )
    return rows


def _electrical_authoring(
    scene: NeutralScene,
    regions: Sequence[_Region],
    boxes: Sequence[_Box],
) -> tuple[Mapping[str, object], tuple[_ObservedBranch, ...], tuple[_ObservedSite, ...], list[dict[str, object]]]:
    branches = _branch_model(scene, regions, "authoring")
    sites = _infer_sites(scene, regions, boxes)
    graph, _ = _build_point_graph(scene)
    endpoints_by_root: dict[int, set[_Endpoint]] = defaultdict(set)
    for branch in branches:
        for pin, point in zip(branch.pins, branch.points, strict=True):
            endpoints_by_root[graph.root(point)].add((branch.path, pin))
    for site in sites:
        endpoints_by_root[graph.root(site.source.point)].add((site.path, site.pin_id))
    observed_branches = tuple(
        _ObservedBranch(
            branch.source,
            branch.path,
            branch.owner,
            branch.role,
            branch.pins,
            branch.points,
            cast(
                tuple[str, str],
                tuple(_contacted_net(graph, point, endpoints_by_root) for point in branch.points),
            ),
        )
        for branch in branches
    )
    observed_sites = tuple(
        _ObservedSite(
            site.source,
            site.path,
            site.pin_id,
            _contacted_net(graph, site.source.point, endpoints_by_root),
        )
        for site in sites
    )
    net_contacts: dict[str, set[str]] = defaultdict(set)
    for branch in observed_branches:
        for pin, net in zip(branch.pins, branch.nets, strict=True):
            net_contacts[net].add(
                _contact_token(
                    "native_terminal", path=branch.path, identity=branch.role, terminal=pin
                )
            )
    for site in observed_sites:
        net_contacts[site.net].add(
            _contact_token("boundary_site", path=site.path, identity=site.pin_id)
        )

    value_rows = _native_value_rows(observed_branches)
    line_records: list[dict[str, object]] = []
    for box in boxes:
        conductor_rows = tuple(row.text for row in box.source.conductor_rows)
        if not conductor_rows or len(set(conductor_rows)) != len(conductor_rows):
            raise _audit_fail("CPW/MTL visible conductor-row order is empty or duplicated")
        if (box.source.kind == "CPW") != (len(conductor_rows) == 1):
            raise _audit_fail("CPW/MTL native box kind disagrees with its visible conductor rows")
        if box.source.reference_label is None:
            raise _audit_fail("CPW/MTL box omits its visible reference conductor")
        if box.source.kind_label.text != box.source.kind:
            raise _audit_fail("CPW/MTL visible kind label disagrees with its native outline")
        anchors = dict(box.source.anchors)
        wanted_anchors = tuple(
            f"{end}.{conductor}" for end in ("head", "tail") for conductor in conductor_rows
        )
        if tuple(name for name, _ in box.source.anchors) != wanted_anchors:
            raise _audit_fail("CPW/MTL box does not retain ordered head/tail conductor anchors")
        if tuple(label.text for label in box.source.anchor_labels) != wanted_anchors:
            raise _audit_fail("CPW/MTL box omits an ordered visible anchor label")
        if box.source.length_label is not None:
            value_key = _token("transmission_line_length", component_path=list(box.path))
            observed_length = _parse_quantity(box.source.length_label.text, si_unit="meter")
            value_rows.append(
                {
                    "category": "observed_scene",
                    "kind": "transmission_line_length",
                    "identity": value_key,
                    "value": _plain(observed_length),
                }
            )
        line_records.append(
            {
                "component_path": list(box.path),
                "conductors": list(conductor_rows),
                "reference_conductor": box.source.reference_label.text,
                "orientation": "extractor_positive_z_is_head_to_tail",
                "rows": [
                    {
                        "conductor": conductor,
                        "row_ordinal": ordinal,
                        "head_net": _contacted_net(graph, anchors[f"head.{conductor}"], endpoints_by_root),
                        "tail_net": _contacted_net(graph, anchors[f"tail.{conductor}"], endpoints_by_root),
                        "head_site": _contact_token(
                            "boundary_site", path=box.path, identity=f"head.{conductor}"
                        ),
                        "tail_site": _contact_token(
                            "boundary_site", path=box.path, identity=f"tail.{conductor}"
                        ),
                    }
                    for ordinal, conductor in enumerate(conductor_rows)
                ],
            }
        )

    port_records: list[dict[str, object]] = []
    root_children = {
        region.path for region in regions if region.parent == ()
    } | {box.path for box in boxes if box.parent == ()}
    root_subsystem_nets = {
        site.net for site in observed_sites if site.path in root_children
    }
    for port in scene.ports:
        _verify_native_geometry(port.reference_load, "authoring")
        if port.reference_load.kind != "R":
            raise _audit_fail("Port raw reference load is not a trusted native resistor")
        port_id, role, impedance, _ = _port_identity(port)
        parsed_impedance = _parse_quantity(impedance, si_unit="ohm")
        node_net = _contacted_net(graph, port.external_anchor, endpoints_by_root)
        if graph.root(port.external_anchor) != graph.root(port.circuit_anchor):
            raise _audit_fail("Port external lead is not continuously attached to its circuit T")
        if graph.root(port.boundary_anchor) != graph.root(port.circuit_anchor):
            raise _audit_fail("Port boundary circle is not continuously attached to its circuit anchor")
        if graph.root(port.load_anchor) != graph.root(port.circuit_anchor):
            raise _audit_fail("Port raw load is not attached to its circuit node")
        if not graph.is_ground(graph.root(port.ground_anchor)):
            raise _audit_fail("Port raw reference load has no visible local ground return")
        reference_anchors = dict(port.reference_load.anchors)
        if len(reference_anchors) != 2 or not all(
            any(_same_point(point, anchor) for anchor in reference_anchors.values())
            for point in (port.load_anchor, port.ground_anchor)
        ):
            raise _audit_fail("Port raw load symbol does not span its visible load and reference anchors")
        record = {
            "port_id": port_id,
            "role": role,
            "orientation": "node_to_reference",
            "reference_impedance": _plain(parsed_impedance),
            "node_net": node_net,
            "reference_net": "ground",
            "circuit_contact": _contact_token("port", identity=port_id, terminal="circuit"),
            "boundary_contact": _contact_token("port", identity=port_id, terminal="boundary"),
            "load_signal_contact": _contact_token("port", identity=port_id, terminal="load_signal"),
            "load_reference_contact": _contact_token("port", identity=port_id, terminal="load_reference"),
            "load_kind": "raw_reference_impedance",
        }
        port_records.append(record)
        net_contacts[node_net].update(
            cast(str, record[field])
            for field in ("circuit_contact", "boundary_contact", "load_signal_contact")
        )
        if node_net not in root_subsystem_nets:
            net_contacts[node_net].add(
                _contact_token("boundary_site", identity=port_id)
            )
        net_contacts["ground"].add(cast(str, record["load_reference_contact"]))
        value_rows.append(
            {
                "category": "observed_scene",
                "kind": "port",
                "identity": port_id,
                "role": role,
                "reference_impedance": _plain(parsed_impedance),
                "orientation": "node_to_reference",
            }
        )

    node_records: list[dict[str, object]] = []
    node_marks = getattr(scene, "node_marks", ())
    for mark in node_marks:
        label = getattr(mark, "label", None)
        point = getattr(mark, "point", None)
        filled = getattr(mark, "filled", None)
        paths = getattr(mark, "paths", ())
        if (
            label is None
            or not isinstance(point, Point)
            or filled is not True
            or not paths
            or any(path.role not in {"glyph", "symbol"} or not path.closed for path in paths)
        ):
            raise _audit_fail("Public-node glyph is not a visible filled dot with an identity label")
        net = _contacted_net(graph, point, endpoints_by_root)
        node_records.append(
            {"node_id": label.text, "visibility": "public", "visible_mark": "filled_dot", "net": net}
        )
        net_contacts[net].add(_contact_token("public_node", identity=label.text))
    for port in port_records:
        if not any(item["net"] == port["node_net"] for item in node_records):
            node_records.append(
                {
                    "node_id": port["port_id"],
                    "visibility": "port_promoted",
                    "visible_mark": "port_circle",
                    "net": port["node_net"],
                }
            )

    branch_records = [
        {
            "branch_id": branch.key,
            "component_path": list(branch.path),
            "branch_role": branch.role,
            "native_kind": branch.source.kind,
            "owner_scope": list(branch.owner),
            "terminals": [
                {
                    "pin_id": pin,
                    "net": net,
                    "contact": _contact_token(
                        "native_terminal", path=branch.path, identity=branch.role, terminal=pin
                    ),
                }
                for pin, net in zip(branch.pins, branch.nets, strict=True)
            ],
            "reciprocal_terminal_swap": branch.reciprocal,
        }
        for branch in observed_branches
    ]
    electrical = {
        "schema": "scnsim.diagram_connectivity_manifest",
        "schema_version": 1,
        "representation": "authoring",
        "nets": [
            {"net": net, "contacts": sorted(contacts)}
            for net, contacts in sorted(net_contacts.items())
        ],
        "public_nodes": sorted(node_records, key=lambda item: cast(str, item["node_id"])),
        "branches": sorted(branch_records, key=lambda item: cast(str, item["branch_id"])),
        "transmission_lines": sorted(line_records, key=lambda item: cast(list[str], item["component_path"])),
        "ports": sorted(port_records, key=lambda item: cast(str, item["port_id"])),
        "couplings": [],
        "omissions": [],
    }
    return cast(Mapping[str, object], _freeze(electrical)), observed_branches, observed_sites, value_rows


@dataclass(frozen=True, slots=True)
class _Unit:
    id: str
    members: tuple[str, ...]
    nets: frozenset[str]
    grounded: bool


@dataclass(frozen=True, slots=True)
class _ScopedSite:
    observed: _ObservedSite
    peer_kind: str
    peer_id: str

    @property
    def key(self) -> str:
        return self.observed.key

    @property
    def net(self) -> str:
        return self.observed.net

    @property
    def peer_key(self) -> str:
        return _token("diagram_peer", peer_kind=self.peer_kind, peer_id=self.peer_id)

    def record(self) -> dict[str, object]:
        return {
            "site_id": self.key,
            "component_path": list(self.observed.path),
            "pin_id": self.observed.pin_id,
            "net": self.net,
            "peer": self.peer_key,
        }


def _subsystem_parents(
    regions: Sequence[_Region], boxes: Sequence[_Box]
) -> dict[_Path, _Path]:
    parents = {
        region.path: cast(_Path, region.parent)
        for region in regions
        if region.path
    }
    parents.update({box.path: box.parent for box in boxes})
    return parents


def _scoped_sites(
    scope: _Path,
    sites: Sequence[_ObservedSite],
    parents: Mapping[_Path, _Path],
    port_records: Sequence[Mapping[str, object]],
    port_points: Mapping[str, Point],
) -> tuple[_ScopedSite, ...]:
    result: list[_ScopedSite] = []
    subsystem_nets: set[str] = set()
    for site in sites:
        if parents.get(site.path) == scope:
            peer_id = canonical_json_bytes({"component_path": list(site.path)}).decode("utf-8")
            result.append(_ScopedSite(site, "subsystem", peer_id))
            subsystem_nets.add(site.net)
    if scope:
        for site in sites:
            if site.path == scope:
                peer_id = canonical_json_bytes(
                    {
                        "parent_scope": list(parents[scope]),
                        "component_path": list(scope),
                        "pin_id": site.pin_id,
                    }
                ).decode("utf-8")
                result.append(_ScopedSite(site, "parent_boundary", peer_id))
    else:
        for port in port_records:
            port_id = cast(str, port["port_id"])
            net = cast(str, port["node_net"])
            if net in subsystem_nets:
                continue
            point = port_points[port_id]
            synthetic = _ObservedSite(
                BoundarySite(point, None), (), port_id, net
            )
            result.append(_ScopedSite(synthetic, "port", port_id))
    unique = {site.key + "\x1f" + site.peer_key: site for site in result}
    return tuple(unique[key] for key in sorted(unique))


def _relations_and_units(
    branches: Sequence[_ObservedBranch], scope: _Path
) -> tuple[list[dict[str, object]], list[_Unit]]:
    owned = tuple(branch for branch in branches if branch.owner == scope)
    groups: dict[tuple[str, str], list[_ObservedBranch]] = defaultdict(list)
    for branch in owned:
        groups[branch.unordered_nets].append(branch)
    relations: list[dict[str, object]] = []
    units: list[_Unit] = []
    consumed: set[str] = set()
    for nets in sorted(groups):
        members = sorted(groups[nets], key=lambda item: item.key)
        if len(members) < 2:
            continue
        member_ids = tuple(member.key for member in members)
        relation_id = _token(
            "parallel_relation",
            owner_scope=list(scope),
            endpoints=list(nets),
            members=list(member_ids),
        )
        relations.append(
            {
                "relation_id": relation_id,
                "owner_scope": list(scope),
                "endpoint_nets": list(nets),
                "members": list(member_ids),
            }
        )
        units.append(_Unit(relation_id, member_ids, frozenset(nets), "ground" in nets))
        consumed.update(member_ids)
    for branch in sorted(owned, key=lambda item: item.key):
        if branch.key not in consumed:
            units.append(
                _Unit(branch.key, (branch.key,), frozenset(branch.nets), "ground" in branch.nets)
            )
    return relations, units


def _unit_components(units: Sequence[_Unit], boundary_nets: set[str]) -> list[list[_Unit]]:
    active = [unit for unit in units if not unit.grounded]
    neighbors: dict[int, set[int]] = defaultdict(set)
    for left, first in enumerate(active):
        for right in range(left + 1, len(active)):
            shared = first.nets & active[right].nets
            if any(net != "ground" and net not in boundary_nets for net in shared):
                neighbors[left].add(right)
                neighbors[right].add(left)
    groups: list[list[_Unit]] = []
    unseen = set(range(len(active)))
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        pending = [start]
        indexes: list[int] = []
        while pending:
            current = pending.pop()
            indexes.append(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
        groups.append([active[index] for index in sorted(indexes)])
    return groups


def _scope_semantics(
    scope: _Path,
    branches: Sequence[_ObservedBranch],
    sites: Sequence[_ScopedSite],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    relations, units = _relations_and_units(branches, scope)
    sites_by_net: dict[str, list[_ScopedSite]] = defaultdict(list)
    for site in sites:
        sites_by_net[site.net].append(site)
    boundary_nets = set(sites_by_net)
    candidates: list[dict[str, object]] = []
    internal = {unit.id for unit in units if unit.grounded}
    for component in _unit_components(units, boundary_nets):
        nets = frozenset(net for unit in component for net in unit.nets)
        incidence = tuple(
            sorted(
                (site for net in nets for site in sites_by_net.get(net, ())),
                key=lambda item: item.key,
            )
        )
        peers = tuple(sorted({site.peer_key for site in incidence}))
        members = tuple(sorted({member for unit in component for member in unit.members}))
        if len(peers) < 2:
            internal.update(unit.id for unit in component)
            continue
        candidates.append(
            {
                "units": tuple(sorted(unit.id for unit in component)),
                "members": members,
                "nets": tuple(sorted(nets)),
                "sites": incidence,
                "peers": peers,
            }
        )
    pair_indexes = [index for index, item in enumerate(candidates) if len(item["peers"]) == 2]
    pair_parent = {index: index for index in pair_indexes}

    def find(index: int) -> int:
        parent = pair_parent[index]
        if parent != index:
            pair_parent[index] = find(parent)
        return pair_parent[index]

    def union(left: int, right: int) -> None:
        first, second = find(left), find(right)
        pair_parent[max(first, second)] = min(first, second)

    for offset, left in enumerate(pair_indexes):
        for right in pair_indexes[offset + 1 :]:
            if candidates[left]["peers"] == candidates[right]["peers"] and set(candidates[left]["nets"]) & set(candidates[right]["nets"]):
                union(left, right)
    pair_groups: dict[int, list[int]] = defaultdict(list)
    for index in pair_indexes:
        pair_groups[find(index)].append(index)
    selected = [item for item in candidates if len(item["peers"]) >= 3]
    for indexes in pair_groups.values():
        rows = [candidates[index] for index in indexes]
        selected.append(
            {
                "units": tuple(sorted({value for row in rows for value in row["units"]})),
                "members": tuple(sorted({value for row in rows for value in row["members"]})),
                "nets": tuple(sorted({value for row in rows for value in row["nets"]})),
                "sites": tuple(
                    sorted(
                        {site.key: site for row in rows for site in row["sites"]}.values(),
                        key=lambda item: item.key,
                    )
                ),
                "peers": rows[0]["peers"],
            }
        )
    interfaces: list[dict[str, object]] = []
    classified: dict[str, str] = {}
    for item in sorted(selected, key=lambda value: (value["peers"], value["members"], value["nets"])):
        member_ids = cast(tuple[str, ...], item["members"])
        nets = cast(tuple[str, ...], item["nets"])
        peers = cast(tuple[str, ...], item["peers"])
        incidence = cast(tuple[_ScopedSite, ...], item["sites"])
        counts = {peer: sum(site.peer_key == peer for site in incidence) for peer in peers}
        identity = _token(
            "interface_bundle",
            parent_scope=list(scope),
            peer_signature=list(peers),
            sites=[site.key for site in incidence],
            members=list(member_ids),
            nets=list(nets),
        )
        interfaces.append(
            {
                "interface_id": identity,
                "parent_scope": list(scope),
                "relation_type": "P" if len(peers) == 2 else "M",
                "peer_signature": list(peers),
                "peer_count": len(peers),
                "sites": [site.record() for site in incidence],
                "members": list(member_ids),
                "nets": list(nets),
                "terminal_counts": counts,
                "terminal_class": "multi_terminal" if any(value > 1 for value in counts.values()) else "single_terminal",
            }
        )
        classified.update({member: identity for member in member_ids})
    stitches: list[dict[str, object]] = []
    for net, net_sites in sorted(sites_by_net.items()):
        unique = tuple(sorted({site.key: site for site in net_sites}.values(), key=lambda item: item.key))
        if len(unique) < 2:
            continue
        peers = tuple(sorted({site.peer_key for site in unique}))
        identity = _token(
            "memberless_stitch",
            parent_scope=list(scope),
            peer_signature=list(peers),
            net=net,
            sites=[site.key for site in unique],
            members=[],
        )
        stitches.append(
            {
                "stitch_id": identity,
                "parent_scope": list(scope),
                "relation_type": "S",
                "net": net,
                "peer_signature": list(peers),
                "peer_count": len(peers),
                "sites": [site.record() for site in unique],
                "members": [],
            }
        )
    classification: list[dict[str, object]] = []
    for unit in sorted(units, key=lambda item: item.id):
        if unit.id in internal:
            role = "parent_internal_or_peripheral"
            interface_id: str | None = None
        else:
            targets = {classified[member] for member in unit.members if member in classified}
            if len(targets) != 1:
                raise _audit_fail("observed member incidence is incomplete or duplicated")
            role = "interface_member"
            interface_id = next(iter(targets))
        classification.append(
            {
                "unit_id": unit.id,
                "members": list(unit.members),
                "classification": role,
                "interface_id": interface_id,
            }
        )
    return relations, interfaces, stitches, classification


def _visible_relation_guides(scene: NeutralScene) -> tuple[GuideMark, ...]:
    guides: list[GuideMark] = []
    for guide in scene.guides:
        if guide.label is None:
            continue
        match = re.fullmatch(r"([SPM]):(.+)", guide.label.text)
        if match is None:
            continue
        if any(path.closed or path.role != "guide" for path in guide.paths):
            raise _layout_fail("relation guide is not visibly open and non-conductive")
        if not guide.terminals or len(guide.terminals) != len(set(guide.terminals)):
            raise _audit_fail("visible relation guide lacks distinct actual attachment terminals")
        if any(
            not any(
                _same_point(terminal, path.points[0])
                or _same_point(terminal, path.points[-1])
                for path in guide.paths
            )
            for terminal in guide.terminals
        ):
            raise _audit_fail("visible relation guide terminal is not an actual stroke endpoint")
        guides.append(guide)
    return tuple(guides)


def _verify_guide_terminal_incidence(guide: GuideMark) -> None:
    """Prove every declared guide terminal belongs to one visible connector."""

    if not guide.paths or not guide.terminals:
        raise _audit_fail("visible semantic guide has no actual connector incidence")
    if len(guide.terminals) != len(set(guide.terminals)):
        raise _audit_fail("visible semantic guide duplicates a connector terminal")
    if any(
        not any(
            _same_point(terminal, path.points[0])
            or _same_point(terminal, path.points[-1])
            for path in guide.paths
        )
        for terminal in guide.terminals
    ):
        raise _audit_fail("visible semantic guide terminal is not an actual stroke endpoint")

    graph = _PointGraph()
    segments: list[tuple[Point, Point]] = []
    for path in guide.paths:
        graph.union(*(graph.index(point) for point in path.points))
        if path.codes is None:
            segments.extend(zip(path.points, path.points[1:]))
    for first_index, first in enumerate(segments):
        for second in segments[first_index + 1 :]:
            relation, point = _segment_intersection(first, second)
            if relation == "point":
                assert point is not None
                graph.union(graph.index(first[0]), graph.index(second[0]), graph.index(point))
            elif relation == "overlap":
                shared = [
                    endpoint
                    for endpoint in (*first, *second)
                    if _point_on_segment(endpoint, *first)
                    and _point_on_segment(endpoint, *second)
                ]
                graph.union(*(graph.index(endpoint) for endpoint in shared))
    root = graph.root(guide.terminals[0])
    if any(graph.root(terminal) != root for terminal in guide.terminals) or any(
        graph.root(path.points[0]) != root for path in guide.paths
    ):
        raise _audit_fail("visible semantic guide strokes do not form one connected incidence")


def _verify_relation_guides(
    scene: NeutralScene,
    interfaces: Sequence[Mapping[str, object]],
    stitches: Sequence[Mapping[str, object]],
    sites: Sequence[_ObservedSite],
    branches: Sequence[_ObservedBranch],
    port_points: Mapping[str, Point],
    regions: Sequence[_Region],
) -> list[dict[str, object]]:
    relations = [*stitches, *interfaces]
    guides = _visible_relation_guides(scene)
    if len(guides) != len(relations):
        raise _audit_fail(
            "visible relation-guide count disagrees with reconstructed incidence",
            expected_from_scene_incidence=len(relations),
            visible_guides=len(guides),
        )
    site_points = {site.key: site.source.point for site in sites}
    site_points.update(
        {
            _token("boundary_site", component_path=[], pin_id=port_id): point
            for port_id, point in port_points.items()
        }
    )
    branches_by_id = {branch.key: branch for branch in branches}

    def same_point_set(actual: Sequence[Point], required: Sequence[Point]) -> bool:
        if len(actual) != len(required):
            return False
        unused = set(range(len(actual)))
        for point in required:
            matches = [index for index in unused if _same_point(actual[index], point)]
            if len(matches) != 1:
                return False
            unused.remove(matches[0])
        return not unused

    def visible_peer(value: object) -> tuple[str, str]:
        if not isinstance(value, str):
            raise _audit_fail("reconstructed relation peer identity is malformed")
        try:
            record = json.loads(value)
        except Exception as error:
            raise _audit_fail("reconstructed relation peer identity is malformed") from error
        if not isinstance(record, dict) or record.get("kind") != "diagram_peer":
            raise _audit_fail("reconstructed relation peer lies outside the closed grammar")
        kind, identity = record.get("peer_kind"), record.get("peer_id")
        if kind == "port" and isinstance(identity, str) and identity:
            return f"port:{identity}", identity
        if kind not in {"subsystem", "parent_boundary"} or not isinstance(identity, str):
            raise _audit_fail("reconstructed relation peer lies outside the closed grammar")
        try:
            structured = json.loads(identity)
        except Exception as error:
            raise _audit_fail("reconstructed structured relation peer is malformed") from error
        if not isinstance(structured, dict):
            raise _audit_fail("reconstructed structured relation peer is malformed")
        path = structured.get("component_path")
        if (
            not isinstance(path, list)
            or not path
            or any(not isinstance(part, str) or not part for part in path)
        ):
            raise _audit_fail("reconstructed relation peer has no visible component path")
        visible = ".".join(path)
        if kind == "subsystem" and set(structured) == {"component_path"}:
            return f"subsystem:{visible}", visible
        pin = structured.get("pin_id")
        parent = structured.get("parent_scope")
        if (
            kind != "parent_boundary"
            or set(structured) != {"parent_scope", "component_path", "pin_id"}
            or not isinstance(parent, list)
            or any(not isinstance(part, str) or not part for part in parent)
            or not isinstance(pin, str)
            or not pin
        ):
            raise _audit_fail("reconstructed parent-boundary peer is malformed")
        suffix = f"{visible}:{pin}"
        return f"boundary:{suffix}", suffix

    relation_by_id: dict[str, Mapping[str, object]] = {}
    label_groups: dict[tuple[tuple[str, ...], str, tuple[str, ...]], list[Mapping[str, object]]] = defaultdict(list)
    for relation in relations:
        identity = cast(str, relation.get("stitch_id", relation.get("interface_id")))
        relation_by_id[identity] = relation
        relation_type = cast(str, relation["relation_type"])
        peer_signature = cast(Sequence[object], relation["peer_signature"])
        visible_peers = tuple(
            suffix for _, suffix in sorted(visible_peer(peer) for peer in peer_signature)
        )
        parent_scope = tuple(cast(Sequence[str], relation["parent_scope"]))
        label_groups[(parent_scope, relation_type, visible_peers)].append(relation)

    compact_labels: dict[str, str] = {}
    for (_, relation_type, visible_peers), grouped in label_groups.items():
        ordered = sorted(
            grouped,
            key=lambda relation: canonical_json_bytes(
                {
                    "members": _plain(relation["members"]),
                    "nets": _plain(
                        relation.get("nets", (relation.get("net"),))
                    ),
                    "sites": [
                        site["site_id"]
                        for site in cast(Sequence[Mapping[str, object]], relation["sites"])
                    ],
                }
            ),
        )
        base = relation_type + ":" + "-".join(visible_peers)
        for ordinal, relation in enumerate(ordered, start=1):
            identity = cast(str, relation.get("stitch_id", relation.get("interface_id")))
            compact_labels[identity] = base + (f"-{ordinal}" if len(ordered) > 1 else "")

    guides_by_scope_label: dict[tuple[_Path, str], list[GuideMark]] = defaultdict(list)
    for guide in guides:
        assert guide.label is not None
        guide_bounds = _visible_bounds(
            (guide.label.bounds, *(path.bounds for path in guide.paths))
        )
        guide_scope = _deepest_region(guide_bounds, regions).path
        guides_by_scope_label[(guide_scope, guide.label.text)].append(guide)
    expected_scope_labels = {
        (
            tuple(cast(Sequence[str], relation_by_id[identity]["parent_scope"])),
            label,
        )
        for identity, label in compact_labels.items()
    }
    if set(guides_by_scope_label) != expected_scope_labels or any(
        len(group) != 1 for group in guides_by_scope_label.values()
    ):
        raise _audit_fail(
            "visible relation marks do not exactly match independently derived compact identities",
            derived_labels=[list(scope) + [label] for scope, label in sorted(expected_scope_labels)],
            visible_labels=[list(scope) + [label] for scope, label in sorted(guides_by_scope_label)],
        )

    region_by_scope = {region.path: region for region in regions}
    rows: list[dict[str, object]] = []
    for identity, relation in sorted(relation_by_id.items()):
        relation_sites: list[tuple[str, Point]] = []
        for site in cast(Sequence[Mapping[str, object]], relation["sites"]):
            site_id = cast(str, site["site_id"])
            point = site_points.get(site_id)
            if point is None:
                raise _audit_fail(
                    "reconstructed semantic relation has no visible boundary attachment",
                    relation_id=identity,
                    site_id=site_id,
                )
            relation_sites.append((site_id, point))
        member_branches: list[_ObservedBranch] = []
        for member_id in cast(Sequence[str], relation["members"]):
            branch = branches_by_id.get(member_id)
            if branch is None:
                raise _audit_fail(
                    "semantic relation names a member without one visible native branch",
                    relation_id=identity,
                    member_id=member_id,
                )
            member_branches.append(branch)
        if not relation_sites:
            raise _audit_fail("reconstructed semantic relation has no visible incidence")
        owner_scope = tuple(cast(Sequence[str], relation["parent_scope"]))
        guide = guides_by_scope_label[(owner_scope, compact_labels[identity])][0]
        assert guide.label is not None
        derived_type = cast(str, relation["relation_type"])
        if derived_type == "S":
            if member_branches or len(relation_sites) < 2:
                raise _audit_fail("reconstructed stitch does not retain empty-member same-net incidence")
            matching_pairs = [
                (left_id, right_id)
                for left_index, (left_id, left_point) in enumerate(relation_sites)
                for right_id, right_point in relation_sites[left_index + 1 :]
                if same_point_set(guide.terminals, (left_point, right_point))
            ]
            if len(matching_pairs) != 1:
                raise _audit_fail(
                    "visible stitch bracket is not bound to two actual sites of its same-net relation",
                    relation_id=identity,
                    matching_site_pairs=len(matching_pairs),
                )
        else:
            matches = [
                branch
                for branch in member_branches
                if same_point_set(guide.terminals, branch.points)
            ]
            if len(matches) != 1:
                raise _audit_fail(
                    "visible pair/multiparty bracket is not bound to one actual relation member",
                    relation_id=identity,
                    matching_members=len(matches),
                )
        owner_region = region_by_scope.get(owner_scope)
        if owner_region is None or not all(
            owner_region.source.bounds.contains(path.bounds) for path in guide.paths
        ) or not owner_region.source.bounds.contains(guide.label.bounds):
            raise _audit_fail(
                "visible relation mark lies outside its independently reconstructed owner scope",
                relation_id=identity,
            )
        rows.append(
            {
                "category": "observed_scene",
                "kind": "semantic_relation",
                "identity": identity,
                "relation_type": derived_type,
                "visible_mark": guide.label.text,
                "peer_count": relation["peer_count"],
                "sites": relation["sites"],
                "members": relation["members"],
            }
        )
    return rows


def _branch_reference(branch: _ObservedBranch) -> dict[str, object]:
    branch_id = branch.role.removeprefix("inductor:")
    return {"component_path": list(branch.path), "branch_id": branch_id}


def _quantity_after_marker(text: str, marker: str, unit: str) -> Mapping[str, object] | None:
    match = re.search(
        rf"(?:^|;)\s*{re.escape(marker)}\s*=\s*([^;]+?)\s*(?=;|$)",
        text,
    )
    if match is None:
        return None
    return _parse_quantity(match.group(1), si_unit=unit)


def _lca(left: _Path, right: _Path) -> _Path:
    result: list[str] = []
    for first, second in zip(left, right, strict=False):
        if first != second:
            break
        result.append(first)
    return tuple(result)


def _visible_couplings(
    scene: NeutralScene,
    branches: Sequence[_ObservedBranch],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    visible = [
        guide
        for guide in scene.guides
        if guide.label is not None
        and guide.paths
        and all(path.role == "coupling" for path in guide.paths)
    ]
    inductive = [branch for branch in branches if branch.source.kind in {"L", "JJ"}]
    centers = {
        branch.key: Point(
            (branch.points[0].x + branch.points[1].x) / 2.0,
            (branch.points[0].y + branch.points[1].y) / 2.0,
        )
        for branch in inductive
    }
    observed: list[dict[str, object]] = []
    semantic: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    used_pairs: set[tuple[str, str]] = set()
    for guide in visible:
        assert guide.label is not None
        if any(path.closed for path in guide.paths):
            raise _layout_fail("mutual-coupling annotation must be visibly non-conductive and open")
        if len(guide.terminals) != 2:
            raise _audit_fail("visible coupling must have exactly two oriented branch attachments")
        _verify_guide_terminal_incidence(guide)
        selected_list: list[_ObservedBranch] = []
        for terminal in guide.terminals:
            matches = [
                branch
                for branch in inductive
                if _same_point(terminal, centers[branch.key])
            ]
            if len(matches) != 1:
                raise _audit_fail(
                    "visible coupling terminal does not contact one actual inductive-symbol center",
                    match_count=len(matches),
                )
            selected_list.append(matches[0])
        selected = (selected_list[0], selected_list[1])
        if selected[0].key == selected[1].key:
            raise _audit_fail("visible coupling attaches twice to one inductive branch")
        pair_key = tuple(sorted((selected[0].key, selected[1].key)))
        if pair_key in used_pairs:
            raise _audit_fail("visible coupling annotations duplicate one physical branch pair")
        used_pairs.add(pair_key)
        coefficient = _quantity_after_marker(guide.label.text, "k", "dimensionless")
        if coefficient is None:
            raise _audit_fail("mutual-coupling mark omits its visible signed k")
        identity_match = re.search(r"(?:^|;)\s*id\s*=\s*([^;]+?)\s*(?=;|$)", guide.label.text)
        if identity_match is None or not identity_match.group(1):
            raise _audit_fail("mutual-coupling mark omits its visible identity")
        coupling_id = identity_match.group(1)
        branch_a, branch_b = selected
        electrical_record = {
            "coupling_id": coupling_id,
            "branch_a": _branch_reference(branch_a),
            "branch_b": _branch_reference(branch_b),
            "coupling_coefficient": _plain(coefficient),
        }
        owner = _lca(branch_a.owner, branch_b.owner)
        semantic_record = {"owner_scope": list(owner), **electrical_record}
        observed.append(electrical_record)
        semantic.append(semantic_record)
        row: dict[str, object] = {
            "category": "observed_scene",
            "kind": "mutual_coupling",
            "identity": coupling_id,
            "branch_a": electrical_record["branch_a"],
            "branch_b": electrical_record["branch_b"],
            "coupling_coefficient": _plain(coefficient),
            "presentation": "visible_mark_or_legend",
        }
        derived = _quantity_after_marker(guide.label.text, "M", "henry")
        if derived is not None:
            if branch_a.source.value is None or branch_b.source.value is None:
                raise _audit_fail(
                    "visible derived M cannot be witnessed without both visible inductive values",
                    coupling_id=coupling_id,
                )
            inductance_a = _parse_quantity(branch_a.source.value.text, si_unit="henry")
            inductance_b = _parse_quantity(branch_b.source.value.text, si_unit="henry")
            coefficient_value = float64_from_hex(cast(str, coefficient["si_value_f64"]))
            inductance_a_value = float64_from_hex(cast(str, inductance_a["si_value_f64"]))
            inductance_b_value = float64_from_hex(cast(str, inductance_b["si_value_f64"]))
            # Evaluate the physical product from the three independently
            # visible binary64 values without an extra rounded a*b operation.
            with localcontext() as context:
                context.prec = 200
                mutual_value = float(
                    Decimal.from_float(coefficient_value)
                    * (
                        Decimal.from_float(inductance_a_value)
                        * Decimal.from_float(inductance_b_value)
                    ).sqrt()
                )
            reconstructed = quantity_envelope(
                units.registry.Quantity(mutual_value, "henry"),
                si_unit="henry",
                registry=units.registry,
            )
            if canonical_json_bytes(derived) != canonical_json_bytes(reconstructed):
                raise _audit_fail(
                    "visible derived M disagrees with visible inductance and signed-k evidence",
                    coupling_id=coupling_id,
                )
            row["derived_mutual_inductance"] = _plain(derived)
        audit_rows.append(row)
    observed.sort(key=lambda item: cast(str, item["coupling_id"]))
    semantic.sort(key=lambda item: cast(str, item["coupling_id"]))
    return observed, semantic, audit_rows


def _visible_omissions(
    scene: NeutralScene,
    branches: Sequence[_ObservedBranch],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    visible = [
        guide
        for guide in scene.guides
        if guide.label is not None and guide.label.text == "Cj=0 F (omitted)"
    ]
    junctions = [branch for branch in branches if branch.source.kind == "JJ"]
    records: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    used: set[str] = set()
    for guide in visible:
        _verify_guide_terminal_incidence(guide)
        if (
            len(guide.terminals) != 2
            or len(guide.paths) != 2
            or any(path.role != "guide" or path.closed for path in guide.paths)
            or any(
                sum(
                    any(_same_point(terminal, point) for point in path.points)
                    for path in guide.paths
                )
                != 1
                for terminal in guide.terminals
            )
        ):
            raise _audit_fail("visible Cj omission lacks two actual open guide arms")
        matches = [
            branch
            for branch in junctions
            if len(branch.points) == len(guide.terminals)
            and all(
                sum(_same_point(point, terminal) for terminal in guide.terminals) == 1
                for point in branch.points
            )
        ]
        if len(matches) != 1:
            raise _audit_fail(
                "visible Cj omission is not incident to one actual Josephson branch",
                match_count=len(matches),
            )
        junction = matches[0]
        omission_id = _token(
            "authoring_branch",
            component_path=list(junction.path),
            branch_role="junction_capacitance",
        )
        if omission_id in used:
            raise _audit_fail("visible omission records duplicate one exact-zero branch")
        used.add(omission_id)
        record = {
            "branch_id": omission_id,
            "component_path": list(junction.path),
            "branch_role": "junction_capacitance",
            "native_kind": "C",
            "endpoint_nets": list(junction.nets),
            "reason": "exact_zero",
            "surviving_continuity": list(junction.nets),
        }
        records.append(record)
        assert guide.label is not None
        rows.append(
            {
                "category": "observed_scene",
                "kind": "exact_zero_omission",
                "identity": omission_id,
                "value": _plain(_parse_quantity("0 F", si_unit="farad")),
                "visible_mark": guide.label.text,
                "endpoint_nets": list(junction.nets),
                "continuity_branch": junction.key,
            }
        )
    records.sort(key=lambda item: cast(str, item["branch_id"]))
    return records, rows


def _semantic_authoring(
    scene: NeutralScene,
    regions: Sequence[_Region],
    boxes: Sequence[_Box],
    branches: Sequence[_ObservedBranch],
    sites: Sequence[_ObservedSite],
    electrical: Mapping[str, object],
) -> tuple[Mapping[str, object], list[dict[str, object]], Mapping[str, object]]:
    parents = _subsystem_parents(regions, boxes)
    scopes = tuple(sorted({(), *(region.path for region in regions if region.path)}))
    port_records = cast(Sequence[Mapping[str, object]], electrical["ports"])
    port_points = {
        _port_identity(port)[0]: port.boundary_anchor
        for port in scene.ports
    }
    global_sites: list[dict[str, object]] = []
    for site in sites:
        peer_id = canonical_json_bytes({"component_path": list(site.path)}).decode("utf-8")
        global_sites.append(
            {
                "site_id": site.key,
                "component_path": list(site.path),
                "pin_id": site.pin_id,
                "net": site.net,
                "peer": _token("diagram_peer", peer_kind="subsystem", peer_id=peer_id),
            }
        )
    root_subsystem_nets = {
        site.net for site in sites if parents.get(site.path) == ()
    }
    for port in port_records:
        port_id = cast(str, port["port_id"])
        net = cast(str, port["node_net"])
        if net in root_subsystem_nets:
            continue
        global_sites.append(
            {
                "site_id": _token("boundary_site", component_path=[], pin_id=port_id),
                "component_path": [],
                "pin_id": port_id,
                "net": net,
                "peer": _token("diagram_peer", peer_kind="port", peer_id=port_id),
            }
        )
    relations: list[dict[str, object]] = []
    interfaces: list[dict[str, object]] = []
    stitches: list[dict[str, object]] = []
    classifications: list[dict[str, object]] = []
    for scope in scopes:
        scoped = _scoped_sites(scope, sites, parents, port_records, port_points)
        relation_rows, interface_rows, stitch_rows, classification_rows = _scope_semantics(
            scope, branches, scoped
        )
        relations.extend(relation_rows)
        interfaces.extend(interface_rows)
        stitches.extend(stitch_rows)
        classifications.extend(classification_rows)
    coupling_electrical, coupling_semantic, coupling_rows = _visible_couplings(
        scene, branches
    )
    omission_records, omission_rows = _visible_omissions(scene, branches)
    relation_rows = _verify_relation_guides(
        scene, interfaces, stitches, sites, branches, port_points, regions
    )
    region_records = [
        {
            "component_path": list(region.path),
            "parent_scope": None if region.parent is None else list(region.parent),
            "region_role": "root_envelope" if not region.path else "composite_region",
            "visible_id": None if not region.path else region.path[-1],
        }
        for region in sorted(regions, key=lambda item: item.path)
    ]
    region_records.extend(
        {
            "component_path": list(box.path),
            "parent_scope": list(box.parent),
            "region_role": "electrical_box",
            "visible_id": box.path[-1],
        }
        for box in boxes
    )
    region_records.sort(key=lambda item: cast(list[str], item["component_path"]))
    leaf_ownership = [
        {
            "branch_id": branch.key,
            "component_path": list(branch.path),
            "owner_scope": list(branch.owner),
            "visible_local_name": branch.source.visible_name.text,
            "branch_role": branch.role,
        }
        for branch in sorted(branches, key=lambda item: (item.path, item.role))
    ]
    port_boundaries = [
        {
            "port_id": port["port_id"],
            "node_net": port["node_net"],
            "colocated_subsystem_peers": sorted(
                {
                    _token(
                        "diagram_peer",
                        peer_kind="subsystem",
                        peer_id=canonical_json_bytes(
                            {"component_path": list(site.path)}
                        ).decode("utf-8"),
                    )
                    for site in sites
                    if site.net == port["node_net"]
                }
            ),
        }
        for port in port_records
    ]
    semantic = {
        "schema": "scnsim.diagram_semantic_manifest",
        "schema_version": 1,
        "representation": "authoring",
        "regions": region_records,
        "leaf_ownership": leaf_ownership,
        "boundary_sites": sorted(global_sites, key=lambda item: cast(str, item["site_id"])),
        "relations": sorted(relations, key=lambda item: cast(str, item["relation_id"])),
        "interfaces": sorted(interfaces, key=lambda item: cast(str, item["interface_id"])),
        "stitches": sorted(stitches, key=lambda item: cast(str, item["stitch_id"])),
        "member_classification": sorted(classifications, key=lambda item: cast(str, item["unit_id"])),
        "port_boundaries": sorted(port_boundaries, key=lambda item: cast(str, item["port_id"])),
        "couplings": coupling_semantic,
    }
    patched_electrical = dict(_plain(electrical))
    patched_electrical["couplings"] = coupling_electrical
    patched_electrical["omissions"] = omission_records
    return (
        cast(Mapping[str, object], _freeze(semantic)),
        [*relation_rows, *coupling_rows, *omission_rows],
        cast(Mapping[str, object], _freeze(patched_electrical)),
    )


def _replace_net_tokens(value: object, replacements: Mapping[str, str]) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _replace_net_tokens(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_replace_net_tokens(item, replacements) for item in value]
    if isinstance(value, str):
        result = value
        for source, target in sorted(replacements.items()):
            result = result.replace(source, target)
        return result
    return value


def _normalize_manifests(
    electrical: Mapping[str, object], semantic: Mapping[str, object]
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    raw = cast(dict[str, object], _plain(electrical))
    contact_replacements: dict[str, str] = {}
    for branch in cast(list[dict[str, object]], raw.get("branches", [])):
        if branch.get("reciprocal_terminal_swap") is not True:
            continue
        path = tuple(cast(Sequence[str], branch["component_path"]))
        role = cast(str, branch["branch_role"])
        replacement = _contact_token(
            "native_terminal", path=path, identity=role, terminal="reciprocal"
        )
        for terminal in cast(list[dict[str, object]], branch["terminals"]):
            contact_replacements[cast(str, terminal["contact"])] = replacement
    net_replacements: dict[str, str] = {"ground": "ground"}
    normalized_nets: list[dict[str, object]] = []
    for net in cast(list[dict[str, object]], raw.get("nets", [])):
        old = cast(str, net["net"])
        contacts = sorted(
            contact_replacements.get(cast(str, contact), cast(str, contact))
            for contact in cast(Sequence[object], net["contacts"])
        )
        new = "ground" if old == "ground" else "netv-" + sha256_hex(
            {
                "schema": "scnsim.visible_net_equivalence",
                "schema_version": 1,
                "contacts": contacts,
            }
        )
        net_replacements[old] = new
        normalized_nets.append({"net": new, "contacts": contacts})
    normalized = cast(dict[str, object], _replace_net_tokens(raw, net_replacements))
    normalized["nets"] = sorted(
        normalized_nets, key=lambda item: canonical_json_bytes(item)
    )
    for branch in cast(list[dict[str, object]], normalized.get("branches", [])):
        terminals = cast(list[dict[str, object]], branch["terminals"])
        for terminal in terminals:
            terminal["contact"] = contact_replacements.get(
                cast(str, terminal["contact"]), cast(str, terminal["contact"])
            )
        if branch.get("reciprocal_terminal_swap") is True:
            branch["terminal_pin_ids"] = sorted(cast(str, item["pin_id"]) for item in terminals)
            branch["terminal_nets"] = sorted(cast(str, item["net"]) for item in terminals)
            branch["terminal_contacts"] = sorted(cast(str, item["contact"]) for item in terminals)
            del branch["terminals"]
    normalized_semantic = _replace_net_tokens(_plain(semantic), net_replacements)
    return (
        cast(Mapping[str, object], _freeze(normalized)),
        cast(Mapping[str, object], _freeze(normalized_semantic)),
    )


def _compare_manifest(
    name: str,
    expected: Mapping[str, object],
    observed: Mapping[str, object],
) -> None:
    expected_bytes = canonical_json_bytes(expected)
    observed_bytes = canonical_json_bytes(observed)
    if expected_bytes != observed_bytes:
        raise _audit_fail(
            f"observed scene {name} differs from its independent expected manifest",
            expected_sha256=sha256_hex(expected_bytes),
            observed_sha256=sha256_hex(observed_bytes),
        )


def _observed_rows(
    electrical: Mapping[str, object],
    semantic: Mapping[str, object],
    detail_rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = [*detail_rows]
    collections = (
        ("public_node", electrical.get("public_nodes", ())),
        ("native_branch", electrical.get("branches", ())),
        ("transmission_line", electrical.get("transmission_lines", ())),
        ("port", electrical.get("ports", ())),
        ("parallel_relation", semantic.get("relations", ())),
        ("interface_bundle", semantic.get("interfaces", ())),
        ("memberless_stitch", semantic.get("stitches", ())),
    )
    for kind, values in collections:
        for value in cast(Sequence[Mapping[str, object]], values):
            rows.append(
                {
                    "category": "observed_scene",
                    "kind": kind,
                    "facts": _plain(value),
                }
            )
    return tuple(cast(Mapping[str, object], _freeze(row)) for row in rows)


def _verified_rows(
    expected: _ExpectedManifest,
    *,
    connectivity_sha256: str,
    semantic_sha256: str,
) -> tuple[Mapping[str, object], ...]:
    verified = expected.verified
    rows: list[dict[str, object]] = [
        {
            "category": "verified_snapshot_provenance",
            "kind": "identity",
            "plan_id": verified["plan_id"],
            "plan_sha256": verified["plan_sha256"],
            "connectivity_sha256": connectivity_sha256,
            "semantic_sha256": semantic_sha256,
            "compiled_graph_sha256": expected.compiled_graph_sha256,
            "expanded_graph_sha256": expected.expanded_graph_sha256,
        },
        {
            "category": "verified_snapshot_provenance",
            "kind": "canonical_baseline_values",
            "values": _plain(verified.get("canonical_values")),
        },
        {
            "category": "verified_snapshot_provenance",
            "kind": "source_units",
            "values": _plain(
                cast(Mapping[str, object], verified["source_provenance"]).get("source_units")
            ),
        },
        {
            "category": "verified_snapshot_provenance",
            "kind": "ground_call_groups",
            "digest_participation": "excluded",
            "values": _plain(
                cast(Mapping[str, object], verified["source_provenance"]).get("ground_call_groups")
            ),
        },
    ]
    if expected.representation == "compiled":
        rows.append(
            {
                "category": "verified_snapshot_provenance",
                "kind": "compiled_expansion",
                "resolved_bindings": _plain(verified.get("resolved_bindings")),
            }
        )
    return tuple(cast(Mapping[str, object], _freeze(row)) for row in rows)


def _verify_visible_values(
    expected: _ExpectedManifest,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Compare independently parsed visible quantities only after reconstruction."""

    observed: dict[str, Mapping[str, object]] = {}

    def add(identity: str, value: object) -> None:
        if not isinstance(value, Mapping):
            raise _audit_fail("observed value row does not contain a canonical quantity", identity=identity)
        if identity in observed:
            raise _audit_fail("visible value evidence duplicates one canonical identity", identity=identity)
        observed[identity] = value

    for row in rows:
        kind = row.get("kind")
        identity = row.get("identity")
        if kind in {"displayed_baseline_value", "transmission_line_length", "exact_zero_omission"}:
            if not isinstance(identity, str):
                raise _audit_fail("visible baseline row lacks an identity")
            add(identity, row.get("value"))
        elif kind == "port":
            if not isinstance(identity, str):
                raise _audit_fail("visible Port value row lacks an identity")
            add(_token("port_impedance", port_id=identity), row.get("reference_impedance"))
        elif kind == "mutual_coupling":
            if not isinstance(identity, str):
                raise _audit_fail("visible coupling value row lacks an identity")
            add(
                _token("coupling_coefficient", coupling_id=identity),
                row.get("coupling_coefficient"),
            )
            if "derived_mutual_inductance" in row:
                add(
                    _token("derived_mutual_inductance", coupling_id=identity),
                    row.get("derived_mutual_inductance"),
                )

    for identity, value in observed.items():
        wanted = expected.expected_values.get(identity)
        if not isinstance(wanted, Mapping):
            raise _audit_fail(
                "visible quantity has no captured baseline identity",
                identity=identity,
            )
        if canonical_json_bytes(value) != canonical_json_bytes(wanted):
            raise _audit_fail(
                "visible quantity disagrees with the captured Plan baseline",
                identity=identity,
                expected_sha256=sha256_hex(wanted),
                observed_sha256=sha256_hex(value),
            )


def _certify_authoring(
    capture: _CapturedPlan,
    scene: NeutralScene,
    expected: _ExpectedManifest,
) -> DiagramAuditData:
    regions = _region_model(scene)
    boxes = _box_model(scene, regions)
    electrical, branches, sites, value_rows = _electrical_authoring(
        scene, regions, boxes
    )
    semantic, semantic_rows, electrical = _semantic_authoring(
        scene, regions, boxes, branches, sites, electrical
    )
    _verify_visible_values(expected, [*value_rows, *semantic_rows])
    normalized_expected_electrical, normalized_expected_semantic = _normalize_manifests(
        expected.electrical, expected.semantic
    )
    normalized_observed_electrical, normalized_observed_semantic = _normalize_manifests(
        electrical, semantic
    )
    _compare_manifest(
        "electrical reconstruction",
        normalized_expected_electrical,
        normalized_observed_electrical,
    )
    _compare_manifest(
        "semantic reconstruction",
        normalized_expected_semantic,
        normalized_observed_semantic,
    )
    connectivity_sha256 = sha256_hex(normalized_observed_electrical)
    semantic_sha256 = sha256_hex(normalized_observed_semantic)
    verified_rows = _verified_rows(
        expected,
        connectivity_sha256=connectivity_sha256,
        semantic_sha256=semantic_sha256,
    )
    return DiagramAuditData(
        representation="authoring",
        plan_id=cast(str, capture.document["plan_id"]),
        plan_sha256=capture.plan_sha256,
        connectivity_sha256=connectivity_sha256,
        semantic_sha256=semantic_sha256,
        compiled_graph_sha256=None,
        expanded_graph_sha256=None,
        presentation_sha256=scene_digest(scene),
        observed_electrical=normalized_observed_electrical,
        observed_semantic=normalized_observed_semantic,
        observed_rows=_observed_rows(
            normalized_observed_electrical,
            normalized_observed_semantic,
            [*value_rows, *semantic_rows],
        ),
        verified_rows=verified_rows,
    )


_COMPILED_PREFIX = "SCNSIM-COMPILED-V1 "
_COMPILED_FIELDS: Mapping[str, tuple[tuple[str, Literal["json", "quantity"]], ...]] = {
    "LINE": (
        ("path", "json"),
        ("conductors", "json"),
        ("reference", "json"),
        ("n_sections", "json"),
        ("length", "quantity"),
        ("dx", "quantity"),
        ("orientation", "json"),
        ("stations", "json"),
        ("rlgc_source", "json"),
    ),
    "TERM": (
        ("kind", "json"),
        ("path", "json"),
        ("section", "json"),
        ("station", "json"),
        ("end", "json"),
        ("row_conductor", "json"),
        ("column_conductor", "json"),
        ("branch_id", "json"),
        ("value", "quantity"),
        ("omitted_as_zero", "json"),
        ("terminal_1_to_terminal_2", "json"),
        ("incidence", "json"),
        ("Bplus", "json"),
        ("Bminus", "json"),
    ),
    "OMIT": (
        ("kind", "json"),
        ("path", "json"),
        ("section", "json"),
        ("station", "json"),
        ("end", "json"),
        ("row_conductor", "json"),
        ("column_conductor", "json"),
        ("branch_id", "json"),
        ("value", "quantity"),
        ("omitted_as_zero", "json"),
        ("terminal_1_to_terminal_2", "json"),
        ("incidence", "json"),
        ("Bplus", "json"),
        ("Bminus", "json"),
    ),
    "MUTUAL": (
        ("coupling_id", "json"),
        ("branch_a", "json"),
        ("branch_b", "json"),
        ("k", "quantity"),
        ("M", "quantity"),
        ("omitted_as_zero", "json"),
    ),
}


def _parse_visible_json(text: str, position: int) -> tuple[object, int]:
    try:
        value, end = json.JSONDecoder().raw_decode(text, position)
    except Exception as error:
        raise _audit_fail("compiled review row contains malformed visible JSON") from error
    literal = text[position:end]
    if canonical_json_bytes(value).decode("utf-8") != literal:
        raise _audit_fail("compiled review row JSON token is not canonical")
    return value, end


def _parse_visible_quantity(text: str, position: int) -> tuple[object, int]:
    if text.startswith("null", position):
        return None, position + 4
    if not text.startswith("q=", position):
        raise _audit_fail("compiled review row quantity token lacks its exact q= prefix")
    separator = text.find("@", position + 2)
    if separator < 0:
        raise _audit_fail("compiled review row quantity token lacks a unit separator")
    encoded = text[position + 2 : separator]
    try:
        magnitude = float64_from_hex(encoded)
    except Exception as error:
        raise _audit_fail("compiled review row contains a malformed Float64 token") from error
    unit, end = _parse_visible_json(text, separator + 1)
    if not isinstance(unit, str) or not unit:
        raise _audit_fail("compiled review row quantity has no canonical SI unit")
    try:
        envelope = quantity_envelope(
            units.registry.Quantity(magnitude, unit),
            si_unit=unit,
            registry=units.registry,
        )
    except Exception as error:
        raise _audit_fail("compiled review row quantity has an invalid SI unit") from error
    if envelope["si_value_f64"] != encoded:
        raise _audit_fail("compiled review row quantity token is not canonical")
    return envelope, end


def _parse_compiled_fields(tag: str, payload: str) -> dict[str, object]:
    result: dict[str, object] = {}
    position = 0
    for index, (key, field_type) in enumerate(_COMPILED_FIELDS[tag]):
        prefix = key + "="
        if not payload.startswith(prefix, position):
            raise _audit_fail(
                "compiled review row has missing, reordered, or extra columns",
                tag=tag,
                required_column=key,
            )
        position += len(prefix)
        if field_type == "quantity":
            value, position = _parse_visible_quantity(payload, position)
        else:
            value, position = _parse_visible_json(payload, position)
        result[key] = value
        last = index == len(_COMPILED_FIELDS[tag]) - 1
        if last:
            if position != len(payload):
                raise _audit_fail("compiled review row has trailing visible fields", tag=tag)
        elif position >= len(payload) or payload[position] != " ":
            raise _audit_fail("compiled review row columns are not visibly separated", tag=tag)
        else:
            position += 1
    return result


def _visible_compiled_rows(
    scene: NeutralScene,
) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...]]:
    headers = [run.text for run in scene.text if run.text.startswith(_COMPILED_PREFIX + "NODE_ORDER ")]
    if len(headers) != 1:
        raise _audit_fail(
            "compiled scene needs one visible ordered-node header", visible_headers=len(headers)
        )
    header_payload = headers[0].removeprefix(_COMPILED_PREFIX + "NODE_ORDER ")
    node_order_value, end = _parse_visible_json(header_payload, 0)
    if end != len(header_payload) or not isinstance(node_order_value, list) or not node_order_value:
        raise _audit_fail("compiled ordered-node header is malformed")
    if any(not isinstance(item, str) or not item for item in node_order_value):
        raise _audit_fail("compiled ordered-node header contains an invalid identity")
    node_order = tuple(cast(list[str], node_order_value))
    if len(node_order) != len(set(node_order)):
        raise _audit_fail("compiled ordered-node header duplicates an identity")

    tagged: dict[int, Mapping[str, object]] = {}
    pattern = re.compile(r"^SCNSIM-COMPILED-V1 #(\d+) (LINE|TERM|OMIT|MUTUAL) (.+)$")
    for run in scene.text:
        if run.text == headers[0]:
            continue
        match = pattern.fullmatch(run.text)
        if match is None:
            if run.text.startswith(_COMPILED_PREFIX):
                raise _audit_fail("compiled scene contains an unparseable competing evidence row")
            continue
        ordinal, tag = int(match.group(1)), match.group(2)
        if ordinal in tagged:
            raise _audit_fail("compiled review table duplicates a visible row ordinal")
        payload = match.group(3)
        field_tag = "MUTUAL" if tag == "OMIT" and payload.startswith("coupling_id=") else tag
        fields = _parse_compiled_fields(field_tag, payload)
        if field_tag == "LINE":
            row = {
                "kind": "transmission_line_audit",
                "component_path": fields["path"],
                "conductors": fields["conductors"],
                "reference_conductor": fields["reference"],
                "n_sections": fields["n_sections"],
                "length": fields["length"],
                "dx": fields["dx"],
                "orientation": fields["orientation"],
                "stations": fields["stations"],
                "rlgc_source": fields["rlgc_source"],
            }
        elif field_tag == "MUTUAL":
            row = {
                "kind": "mutual_inductance",
                "coupling_id": fields["coupling_id"],
                "branch_a": fields["branch_a"],
                "branch_b": fields["branch_b"],
                "coupling_coefficient": fields["k"],
                "derived_mutual_inductance": fields["M"],
                "omitted_as_zero": fields["omitted_as_zero"],
            }
            if (tag == "OMIT") != (fields["omitted_as_zero"] is True):
                raise _audit_fail("compiled mutual tag disagrees with its visible exact-zero field")
        else:
            row = {
                "kind": fields["kind"],
                "component_path": fields["path"],
                "section": fields["section"],
                "station": fields["station"],
                "end": fields["end"],
                "row_conductor": fields["row_conductor"],
                "column_conductor": fields["column_conductor"],
                "branch_id": fields["branch_id"],
                "value": fields["value"],
                "omitted_as_zero": fields["omitted_as_zero"],
                "terminal_1_to_terminal_2": fields["terminal_1_to_terminal_2"],
                "incidence_f64": fields["incidence"],
                "physical_positive_incidence_f64": fields["Bplus"],
                "physical_negative_incidence_f64": fields["Bminus"],
            }
            if (tag == "OMIT") != (fields["omitted_as_zero"] is True):
                raise _audit_fail("compiled row tag disagrees with its visible exact-zero field")
        tagged[ordinal] = cast(Mapping[str, object], _freeze(row))
    if not tagged or set(tagged) != set(range(len(tagged))):
        raise _audit_fail("compiled review table row ordinals are incomplete or non-contiguous")
    return node_order, tuple(tagged[index] for index in range(len(tagged)))


def _compiled_manifests(
    node_order: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    ports: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    audits = tuple(row for row in rows if row["kind"] == "transmission_line_audit")
    branch_rows = tuple(row for row in rows if row["kind"] != "transmission_line_audit")
    active = tuple(row for row in branch_rows if row["omitted_as_zero"] is not True)
    omissions = tuple(row for row in branch_rows if row["omitted_as_zero"] is True)
    couplings = tuple(row for row in active if row["kind"] == "mutual_inductance")
    physical = tuple(row for row in active if row["kind"] != "mutual_inductance")
    electrical = {
        "schema": "scnsim.diagram_connectivity_manifest",
        "schema_version": 1,
        "representation": "compiled",
        "node_order": list(node_order),
        "expanded_terms": [_plain(row) for row in physical],
        "transmission_lines": [_plain(row) for row in audits],
        "couplings": [_plain(row) for row in couplings],
        "omissions": [_plain(row) for row in omissions],
        "ports": [_plain(port) for port in ports],
    }
    hierarchy = [
        {
            "component_path": _plain(row["component_path"]),
            "kind": "pi_ladder",
            "n_sections": row["n_sections"],
            "conductors": _plain(row["conductors"]),
        }
        for row in audits
    ]
    membership = [
        {
            "component_path": _plain(row.get("component_path")),
            "compiled_kind": row["kind"],
            "section": row.get("section"),
            "station": row.get("station"),
            "row_conductor": row.get("row_conductor"),
            "column_conductor": row.get("column_conductor"),
            "omitted_as_zero": row.get("omitted_as_zero"),
        }
        for row in branch_rows
    ]
    semantic = {
        "schema": "scnsim.diagram_semantic_manifest",
        "schema_version": 1,
        "representation": "compiled",
        "compiler_hierarchy": hierarchy,
        "expanded_membership": membership,
        "matrix_evidence": [_plain(row) for row in audits],
        "couplings": [_plain(row) for row in couplings],
        "omissions": [_plain(row) for row in omissions],
    }
    return cast(Mapping[str, object], _freeze(electrical)), cast(
        Mapping[str, object], _freeze(semantic)
    )


def _compiled_endpoint(
    row: Mapping[str, object], field: str, node_order: Sequence[str]
) -> str:
    values = row.get(field)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != len(node_order):
        raise _audit_fail("visible compiled primitive has malformed physical incidence", field=field)
    selected: list[str] = []
    for node, encoded in zip(node_order, values, strict=True):
        if not isinstance(encoded, str):
            raise _audit_fail("visible compiled incidence lacks an exact Float64 token", field=field)
        try:
            scalar = float64_from_hex(encoded)
        except Exception as error:
            raise _audit_fail("visible compiled incidence has a malformed Float64 token") from error
        if scalar != 0.0:
            if abs(scalar) != 1.0:
                raise _audit_fail("visible compiled physical incidence is not an endpoint selector")
            selected.append(node)
    if not selected:
        return "ground"
    if len(selected) != 1:
        raise _audit_fail("visible compiled physical incidence selects multiple nodes")
    return selected[0]


_COMPILED_MATRIX_KINDS: Mapping[str, str] = {
    "series_resistance": "ohm",
    "series_inductance": "henry",
    "shunt_capacitance_half": "farad",
    "shunt_conductance_half": "siemens",
}
_COMPILED_PRIMITIVE_KINDS: Mapping[str, str] = {
    "resistor": "ohm",
    "capacitor": "farad",
    "inductor": "henry",
    "josephson_inductance": "henry",
    "junction_capacitance": "farad",
}


def _compiled_scalar(row: Mapping[str, object], *, unit: str) -> float:
    value = row.get("value")
    if not isinstance(value, Mapping) or value.get("si_unit") != unit:
        raise _audit_fail(
            "visible compiled coefficient has an invalid canonical SI unit",
            expected_unit=unit,
        )
    encoded = value.get("si_value_f64")
    if not isinstance(encoded, str):
        raise _audit_fail("visible compiled coefficient lacks an exact Float64 token")
    try:
        return float64_from_hex(encoded)
    except Exception as error:
        raise _audit_fail("visible compiled coefficient has a malformed Float64 token") from error


def _selector_matches(
    values: object,
    *,
    selected: str | None,
    node_order: Sequence[str],
) -> bool:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != len(node_order)
    ):
        return False
    for node, encoded in zip(node_order, values, strict=True):
        if not isinstance(encoded, str):
            return False
        try:
            scalar = float64_from_hex(encoded)
        except Exception:
            return False
        if scalar != (1.0 if node == selected else 0.0):
            return False
    return True


def _verify_compiled_matrix_structure(
    node_order: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    """Validate the complete visible matrix basis independently of geometry."""

    known_kinds = {
        "transmission_line_audit",
        "mutual_inductance",
        *_COMPILED_MATRIX_KINDS,
        *_COMPILED_PRIMITIVE_KINDS,
    }
    if any(row.get("kind") not in known_kinds for row in rows):
        raise _audit_fail("visible compiled row lies outside the closed projection grammar")
    for row in rows:
        kind = cast(str, row.get("kind"))
        if kind in _COMPILED_PRIMITIVE_KINDS:
            scalar = _compiled_scalar(
                row, unit=_COMPILED_PRIMITIVE_KINDS[kind]
            )
            omitted = row.get("omitted_as_zero")
            if not isinstance(omitted, bool) or omitted != (scalar == 0.0):
                raise _audit_fail(
                    "visible compiled primitive exact-zero flag disagrees with its value"
                )
            _compiled_endpoint(row, "physical_positive_incidence_f64", node_order)
            _compiled_endpoint(row, "physical_negative_incidence_f64", node_order)
        elif kind == "mutual_inductance":
            coupling_id = row.get("coupling_id")
            branch_a, branch_b = row.get("branch_a"), row.get("branch_b")
            coefficient = row.get("coupling_coefficient")
            derived = row.get("derived_mutual_inductance")
            if (
                not isinstance(coupling_id, str)
                or not coupling_id
                or not isinstance(branch_a, Mapping)
                or not isinstance(branch_b, Mapping)
                or canonical_json_bytes(branch_a) == canonical_json_bytes(branch_b)
                or not isinstance(coefficient, Mapping)
                or coefficient.get("si_unit") != "dimensionless"
                or not isinstance(derived, Mapping)
                or derived.get("si_unit") != "henry"
            ):
                raise _audit_fail("visible compiled mutual-inductance row is malformed")
            try:
                coefficient_scalar = float64_from_hex(
                    cast(str, coefficient["si_value_f64"])
                )
                derived_scalar = float64_from_hex(cast(str, derived["si_value_f64"]))
            except Exception as error:
                raise _audit_fail("visible compiled mutual row has malformed values") from error
            omitted = row.get("omitted_as_zero")
            if (
                not isinstance(omitted, bool)
                or omitted != (coefficient_scalar == 0.0)
                or (derived_scalar == 0.0) != omitted
            ):
                raise _audit_fail("visible compiled mutual exact-zero evidence disagrees")

    line_rows = [row for row in rows if row.get("kind") == "transmission_line_audit"]
    audits: dict[_Path, Mapping[str, object]] = {}
    for audit in line_rows:
        raw_path = audit.get("component_path")
        if (
            not isinstance(raw_path, Sequence)
            or isinstance(raw_path, (str, bytes))
            or not raw_path
            or any(not isinstance(item, str) or not item for item in raw_path)
        ):
            raise _audit_fail("visible compiled line audit has an invalid component path")
        path = tuple(cast(Sequence[str], raw_path))
        if path in audits:
            raise _audit_fail("visible compiled line audits duplicate a component path")
        audits[path] = audit

    matrix_rows: dict[_Path, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        kind = row.get("kind")
        if kind not in _COMPILED_MATRIX_KINDS:
            continue
        raw_path = row.get("component_path")
        if not isinstance(raw_path, Sequence) or isinstance(raw_path, (str, bytes)):
            raise _audit_fail("visible compiled matrix row has no component path")
        path = tuple(cast(Sequence[str], raw_path))
        if path not in audits:
            raise _audit_fail("visible compiled matrix row has no line-audit basis")
        matrix_rows[path].append(row)
    if set(matrix_rows) != set(audits):
        raise _audit_fail("visible compiled line audits and matrix row blocks disagree")

    for path, audit in audits.items():
        raw_conductors = audit.get("conductors")
        sections = audit.get("n_sections")
        reference = audit.get("reference_conductor")
        if (
            not isinstance(raw_conductors, Sequence)
            or isinstance(raw_conductors, (str, bytes))
            or not raw_conductors
            or any(not isinstance(item, str) or not item for item in raw_conductors)
        ):
            raise _audit_fail("visible compiled line has an invalid ordered conductor basis")
        conductors = tuple(cast(Sequence[str], raw_conductors))
        if len(conductors) != len(set(conductors)):
            raise _audit_fail("visible compiled line duplicates an ordered conductor")
        if not isinstance(reference, str) or not reference or reference in conductors:
            raise _audit_fail("visible compiled line has an invalid reference conductor")
        if isinstance(sections, bool) or not isinstance(sections, int) or sections < 1:
            raise _audit_fail("visible compiled line has an invalid section count")
        if audit.get("orientation") != "extractor_positive_z_is_head_to_tail":
            raise _audit_fail("visible compiled line loses its head-to-tail orientation")
        for field in ("length", "dx"):
            value = audit.get(field)
            if not isinstance(value, Mapping) or value.get("si_unit") != "meter":
                raise _audit_fail("visible compiled line length evidence is malformed", field=field)

        stations = audit.get("stations")
        if not isinstance(stations, Sequence) or isinstance(stations, (str, bytes)):
            raise _audit_fail("visible compiled line station ledger is malformed")
        station_keys: list[tuple[int, str]] = []
        station_nodes: dict[tuple[int, str], str] = {}
        for raw in stations:
            if not isinstance(raw, Mapping):
                raise _audit_fail("visible compiled station row is malformed")
            station, conductor, node = raw.get("station"), raw.get("conductor"), raw.get("compiled_node_id")
            if (
                isinstance(station, bool)
                or not isinstance(station, int)
                or conductor not in conductors
                or not isinstance(node, str)
                or node not in node_order
            ):
                raise _audit_fail("visible compiled station lies outside its node/conductor basis")
            key = (station, cast(str, conductor))
            if key in station_nodes:
                raise _audit_fail("visible compiled station row is duplicated")
            station_keys.append(key)
            station_nodes[key] = node
        wanted_station_keys = [
            (station, conductor)
            for station in range(sections + 1)
            for conductor in conductors
        ]
        if station_keys != wanted_station_keys:
            raise _audit_fail("visible compiled stations lose exact section/conductor order")

        wanted_keys = {
            (kind, section, station, end, row_conductor, column_conductor)
            for section in range(1, sections + 1)
            for row_conductor in conductors
            for column_conductor in conductors
            for kind, station, end in (
                ("series_resistance", None, None),
                ("series_inductance", None, None),
                ("shunt_capacitance_half", section - 1, "left"),
                ("shunt_capacitance_half", section, "right"),
                ("shunt_conductance_half", section - 1, "left"),
                ("shunt_conductance_half", section, "right"),
            )
        }
        actual: dict[tuple[object, ...], Mapping[str, object]] = {}
        for row in matrix_rows[path]:
            key = (
                row.get("kind"),
                row.get("section"),
                row.get("station"),
                row.get("end"),
                row.get("row_conductor"),
                row.get("column_conductor"),
            )
            if key in actual:
                raise _audit_fail("visible compiled matrix row is duplicated")
            if key not in wanted_keys:
                raise _audit_fail("visible compiled matrix row lies outside its exact basis")
            actual[key] = row
            unit = _COMPILED_MATRIX_KINDS[cast(str, row.get("kind"))]
            scalar = _compiled_scalar(row, unit=unit)
            omitted = row.get("omitted_as_zero")
            if not isinstance(omitted, bool) or omitted != (scalar == 0.0):
                raise _audit_fail("visible compiled exact-zero flag disagrees with its coefficient")
            if row.get("terminal_1_to_terminal_2") is not None or row.get("incidence_f64") is not None:
                raise _audit_fail("visible matrix coefficient invents a scalar branch incidence")
            kind, section, station, _, row_conductor, column_conductor = key
            if cast(str, kind).startswith("series_"):
                if row.get("physical_positive_incidence_f64") is not None or row.get("physical_negative_incidence_f64") is not None:
                    raise _audit_fail("visible series-matrix coefficient invents physical selector vectors")
            else:
                selected_row = station_nodes[(cast(int, station), cast(str, row_conductor))]
                selected_column = (
                    None
                    if row_conductor == column_conductor
                    else station_nodes[(cast(int, station), cast(str, column_conductor))]
                )
                if not _selector_matches(
                    row.get("physical_positive_incidence_f64"),
                    selected=selected_row,
                    node_order=node_order,
                ) or not _selector_matches(
                    row.get("physical_negative_incidence_f64"),
                    selected=selected_column,
                    node_order=node_order,
                ):
                    raise _audit_fail("visible B_t selector vectors disagree with their station basis")
        if set(actual) != wanted_keys:
            raise _audit_fail(
                "visible compiled matrix rows are incomplete or outside their exact basis",
                expected=len(wanted_keys),
                actual=len(actual),
            )
        for key, row in actual.items():
            kind, section, station, end, left, right = key
            if left == right:
                continue
            mirror = actual[(kind, section, station, end, right, left)]
            if canonical_json_bytes(row["value"]) != canonical_json_bytes(mirror["value"]):
                raise _audit_fail("visible ordered matrix entries are not exactly symmetric")


def _verify_compiled_geometry(
    scene: NeutralScene,
    node_order: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    _verify_compiled_matrix_structure(node_order, rows)
    graph, _ = _build_point_graph(scene)
    marks: dict[str, Point] = {}
    for mark in scene.node_marks:
        if mark.label is None:
            raise _audit_fail("compiled node identity mark lacks its required visible label")
        if mark.label.text in marks:
            raise _audit_fail("compiled scene duplicates a visible node identity")
        if not mark.filled or not mark.paths or any(not path.closed for path in mark.paths):
            raise _audit_fail("compiled node identity is not carried by a filled visible mark")
        marks[mark.label.text] = mark.point
    if tuple(marks) != tuple(node_order):
        raise _audit_fail("compiled visible node marks disagree with ordered-node evidence")
    node_roots = {node: graph.root(point) for node, point in marks.items()}
    if any(graph.is_ground(root) for root in node_roots.values()):
        raise _audit_fail("a visible compiled node is physically shorted to ground")
    if len(set(node_roots.values())) != len(node_roots):
        raise _audit_fail("distinct visible compiled nodes are physically shorted together")

    port_records: list[dict[str, object]] = []
    seen_ports: set[str] = set()
    for port in scene.ports:
        _verify_port_block_geometry(port)
        port_id, role, impedance, _ = _port_identity(port)
        if port_id in seen_ports:
            raise _audit_fail("compiled scene duplicates a visible Port identity")
        seen_ports.add(port_id)
        circuit_root = graph.root(port.external_anchor)
        matching_nodes = [node for node, root in node_roots.items() if root == circuit_root]
        if len(matching_nodes) != 1:
            raise _audit_fail("compiled Port attachment does not select one visible compiler node")
        if any(graph.root(anchor) != circuit_root for anchor in (port.circuit_anchor, port.boundary_anchor, port.load_anchor)):
            raise _audit_fail("compiled Port circle/load is not continuously attached to its node")
        if not graph.is_ground(graph.root(port.ground_anchor)):
            raise _audit_fail("compiled Port raw load lacks an actual visible ground glyph")
        reference_anchors = tuple(point for _, point in port.reference_load.anchors)
        if len(reference_anchors) != 2 or not all(
            any(_same_point(point, anchor) for anchor in reference_anchors)
            for point in (port.load_anchor, port.ground_anchor)
        ):
            raise _audit_fail("compiled Port raw load does not span its visible anchors")
        parsed_impedance = _parse_quantity(impedance, si_unit="ohm")
        port_records.append(
            {
                "port_id": port_id,
                "node_id": matching_nodes[0],
                "role": role,
                "orientation": "node_to_reference",
                "reference_impedance": _plain(parsed_impedance),
                "reference_node": "ground",
                "load_kind": "raw_reference_impedance",
            }
        )
    port_records.sort(key=lambda item: cast(str, item["port_id"]))

    glyphs: dict[int, NativeSymbol] = {}
    for symbol in scene.symbols:
        if symbol.branch_label is None:
            raise _audit_fail("compiled coefficient glyph lacks its visible row ordinal")
        match = re.fullmatch(r"#(\d+)", symbol.branch_label.text)
        if match is None:
            raise _audit_fail("compiled coefficient glyph has an invalid visible row ordinal")
        ordinal = int(match.group(1))
        if ordinal in glyphs:
            raise _audit_fail("compiled coefficient row is depicted by duplicate native glyphs")
        glyphs[ordinal] = symbol

    expected_glyphs: set[int] = set()
    line_by_path = {
        tuple(cast(Sequence[str], row["component_path"])): row
        for row in rows
        if row["kind"] == "transmission_line_audit"
    }
    native_by_kind = {
        "resistor": "R",
        "capacitor": "C",
        "inductor": "L",
        "josephson_inductance": "JJ",
        "junction_capacitance": "C",
        "series_resistance": "R",
        "series_inductance": "L",
        "shunt_capacitance_half": "C",
        "shunt_conductance_half": "G",
    }
    for ordinal, row in enumerate(rows):
        kind = cast(str, row["kind"])
        if kind in native_by_kind and row["omitted_as_zero"] is not True:
            diagonal = kind not in {
                "series_resistance",
                "series_inductance",
                "shunt_capacitance_half",
                "shunt_conductance_half",
            } or row["row_conductor"] == row["column_conductor"]
            if diagonal:
                expected_glyphs.add(ordinal)
                symbol = glyphs.get(ordinal)
                if symbol is None or symbol.kind != native_by_kind[kind]:
                    raise _audit_fail(
                        "compiled matrix coefficient lacks its correct native glyph",
                        ordinal=ordinal,
                        matrix_kind=kind,
                    )
                expected_name = (
                    f"{native_by_kind[kind]} coefficient"
                    if kind in {
                        "series_resistance", "series_inductance",
                        "shunt_capacitance_half", "shunt_conductance_half",
                    }
                    else f"{'.'.join(cast(Sequence[str], row['component_path']))}:{kind}"
                )
                if symbol.visible_name.text != expected_name:
                    raise _audit_fail(
                        "compiled coefficient glyph lacks its exact visible matrix/primitive identity",
                        ordinal=ordinal,
                    )
                if symbol.value is not None:
                    value = cast(Mapping[str, object], row["value"])
                    observed = _parse_quantity(
                        symbol.value.text, si_unit=cast(str, value["si_unit"])
                    )
                    if canonical_json_bytes(observed) != canonical_json_bytes(value):
                        raise _audit_fail("compiled coefficient glyph value disagrees with its visible row")
    if set(glyphs) != expected_glyphs:
        raise _audit_fail(
            "compiled scene has missing or extra coefficient glyphs",
            expected_ordinals=sorted(expected_glyphs),
            actual_ordinals=sorted(glyphs),
        )

    def root_matches(point: Point, identity: str) -> bool:
        return graph.is_ground(graph.root(point)) if identity == "ground" else graph.root(point) == node_roots[identity]

    primitive_kinds = frozenset({"resistor", "capacitor", "inductor", "josephson_inductance", "junction_capacitance"})
    for ordinal, row in enumerate(rows):
        if row["kind"] not in primitive_kinds or row["omitted_as_zero"] is True:
            continue
        positive = _compiled_endpoint(row, "physical_positive_incidence_f64", node_order)
        negative = _compiled_endpoint(row, "physical_negative_incidence_f64", node_order)
        symbol = glyphs[ordinal]
        points = tuple(point for _, point in symbol.anchors)
        direct = root_matches(points[0], positive) and root_matches(points[1], negative)
        reciprocal = symbol.kind in {"R", "C"} and root_matches(points[1], positive) and root_matches(points[0], negative)
        if not direct and not reciprocal:
            raise _audit_fail("compiled primitive glyph disagrees with its visible physical incidence", ordinal=ordinal)

    for path, audit in line_by_path.items():
        station_nodes = {
            (item["station"], item["conductor"]): item["compiled_node_id"]
            for item in cast(Sequence[Mapping[str, object]], audit["stations"])
        }
        conductors = cast(Sequence[str], audit["conductors"])
        sections = cast(int, audit["n_sections"])
        indexed = {
            (
                row["kind"], row["section"], row["station"], row["end"],
                row["row_conductor"], row["column_conductor"],
            ): ordinal
            for ordinal, row in enumerate(rows)
            if tuple(cast(Sequence[str], row.get("component_path") or ())) == path
            and row["kind"] in {
                "series_resistance", "series_inductance",
                "shunt_capacitance_half", "shunt_conductance_half",
            }
        }
        for section in range(1, sections + 1):
            for conductor in conductors:
                series_ordinals = [
                    indexed[(kind, section, None, None, conductor, conductor)]
                    for kind in ("series_resistance", "series_inductance")
                ]
                active_symbols = [glyphs[item] for item in series_ordinals if item in glyphs]
                left = cast(str, station_nodes[(section - 1, conductor)])
                right = cast(str, station_nodes[(section, conductor)])
                if active_symbols:
                    points = [tuple(point for _, point in symbol.anchors) for symbol in active_symbols]
                    if not root_matches(points[0][0], left) or not root_matches(points[-1][1], right):
                        raise _audit_fail("compiled series coefficient block reverses its visible B_s incidence")
                    if any(graph.root(first[1]) != graph.root(second[0]) for first, second in zip(points, points[1:])):
                        raise _audit_fail("compiled series R/L coefficient block is visibly discontinuous")
                elif node_roots[left] != node_roots[right]:
                    raise _audit_fail("all-zero series block does not preserve its declared continuity")
                for kind in ("shunt_capacitance_half", "shunt_conductance_half"):
                    for end, station in (("left", section - 1), ("right", section)):
                        item = indexed[(kind, section, station, end, conductor, conductor)]
                        if item not in glyphs:
                            continue
                        points = tuple(point for _, point in glyphs[item].anchors)
                        node = cast(str, station_nodes[(station, conductor)])
                        if not (
                            root_matches(points[0], node) and root_matches(points[1], "ground")
                            or root_matches(points[1], node) and root_matches(points[0], "ground")
                        ):
                            raise _audit_fail("compiled half-shunt glyph is not attached to its visible B_t station")

    guide_labels = [guide.label.text for guide in scene.guides if guide.label is not None]
    for ordinal, row in enumerate(rows):
        if row.get("omitted_as_zero") is True and f"#{ordinal}" not in guide_labels:
            raise _audit_fail("compiled exact-zero row lacks its visible omission marker", ordinal=ordinal)
        if row["kind"] == "mutual_inductance" and row.get("omitted_as_zero") is not True and f"#{ordinal}" not in guide_labels:
            raise _audit_fail("compiled mutual row lacks its visible coupling marker", ordinal=ordinal)

    # Each symmetric off-diagonal matrix pair has one nonconductive visible
    # mark; two ordered ledger entries remain one coefficient relationship.
    for path, audit in line_by_path.items():
        conductors = cast(Sequence[str], audit["conductors"])
        sections = cast(int, audit["n_sections"])
        for section in range(1, sections + 1):
            attachments = [
                ("series_resistance", None, None),
                ("series_inductance", None, None),
                ("shunt_capacitance_half", section - 1, "left"),
                ("shunt_capacitance_half", section, "right"),
                ("shunt_conductance_half", section - 1, "left"),
                ("shunt_conductance_half", section, "right"),
            ]
            for kind, station, end in attachments:
                prefix = (
                    "L offdiagonal coupling "
                    if kind == "series_inductance"
                    else f"{kind} offdiagonal matrix coefficient "
                )
                for left_index, left in enumerate(conductors):
                    for right in conductors[left_index + 1 :]:
                        matches = [
                            index
                            for index, row in enumerate(rows)
                            if tuple(cast(Sequence[str], row.get("component_path") or ())) == path
                            and row["kind"] == kind
                            and row["section"] == section
                            and row["station"] == station
                            and row["end"] == end
                            and (row["row_conductor"], row["column_conductor"])
                            in {(left, right), (right, left)}
                        ]
                        if len(matches) != 2:
                            raise _audit_fail("compiled matrix is missing an ordered symmetric entry pair")
                        forward = next(index for index in matches if rows[index]["row_conductor"] == left)
                        reverse = next(index for index in matches if rows[index]["row_conductor"] == right)
                        forward_row, reverse_row = rows[forward], rows[reverse]
                        forward_zero = forward_row["omitted_as_zero"] is True
                        reverse_zero = reverse_row["omitted_as_zero"] is True
                        if forward_zero and reverse_zero:
                            continue
                        if forward_zero != reverse_zero:
                            raise _audit_fail("compiled symmetric matrix entries disagree on exact zero")
                        candidates = [
                            label
                            for label in guide_labels
                            if label.startswith(prefix)
                            and f"#{forward}" in label
                            and f"#{reverse}" in label
                        ]
                        if len(candidates) != 1:
                            raise _audit_fail("compiled off-diagonal pair lacks one visible matrix mark")
                        forward_value = cast(Mapping[str, object], forward_row["value"])
                        reverse_value = cast(Mapping[str, object], reverse_row["value"])
                        if (
                            f"#{forward} signed_f64={forward_value['si_value_f64']}" not in candidates[0]
                            or f"#{reverse} signed_f64={reverse_value['si_value_f64']}" not in candidates[0]
                        ):
                            raise _audit_fail("compiled off-diagonal mark loses an exact signed coefficient")
                        if f"orientation={left}→{right}" not in candidates[0]:
                            raise _audit_fail("compiled off-diagonal mark loses conductor orientation")
                        if canonical_json_bytes(forward_value) != canonical_json_bytes(reverse_value):
                            raise _audit_fail("compiled symmetric matrix entries visibly disagree")
    return [
        {
            "category": "observed_scene",
            "kind": "compiled_matrix_row",
            "ordinal": ordinal,
            "facts": _plain(row),
        }
        for ordinal, row in enumerate(rows)
    ], port_records


def _certify_compiled(
    capture: _CapturedPlan,
    scene: NeutralScene,
    expected: _ExpectedManifest,
) -> DiagramAuditData:
    node_order, rows = _visible_compiled_rows(scene)
    detail_rows, ports = _verify_compiled_geometry(scene, node_order, rows)
    electrical, semantic = _compiled_manifests(node_order, rows, ports)
    _compare_manifest("compiled electrical reconstruction", expected.electrical, electrical)
    _compare_manifest("compiled semantic reconstruction", expected.semantic, semantic)
    connectivity_sha256 = sha256_hex(electrical)
    semantic_sha256 = sha256_hex(semantic)
    return DiagramAuditData(
        representation="compiled",
        plan_id=cast(str, capture.document["plan_id"]),
        plan_sha256=capture.plan_sha256,
        connectivity_sha256=connectivity_sha256,
        semantic_sha256=semantic_sha256,
        compiled_graph_sha256=expected.compiled_graph_sha256,
        expanded_graph_sha256=expected.expanded_graph_sha256,
        presentation_sha256=scene_digest(scene),
        observed_electrical=electrical,
        observed_semantic=semantic,
        observed_rows=_observed_rows(electrical, semantic, detail_rows),
        verified_rows=_verified_rows(
            expected,
            connectivity_sha256=connectivity_sha256,
            semantic_sha256=semantic_sha256,
        ),
    )


def _point_observed_rows(
    electrical: Mapping[str, object],
    structural: Mapping[str, object],
    detail: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = [*detail]
    collections = (
        ("electrical_net", electrical.get("nets", ())),
        ("physical_body", electrical.get("bodies", ())),
        ("port", electrical.get("ports", ())),
        ("mutual_coupling", electrical.get("couplings", ())),
        ("ownership_region", structural.get("regions", ())),
        ("leaf_ownership", structural.get("leaf_ownership", ())),
        ("port_ownership", structural.get("port_ownership", ())),
        ("boundary_incidence", structural.get("boundary_incidence", ())),
        ("compiled_term", electrical.get("expanded_terms", ())),
        ("compiled_line", electrical.get("transmission_lines", ())),
        ("compiled_omission", electrical.get("omissions", ())),
    )
    for kind, values in collections:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise _audit_fail("reconstructed manifest collection is malformed", kind=kind)
        rows.extend(
            {
                "category": "observed_scene",
                "kind": kind,
                "facts": _plain(value),
            }
            for value in values
            if isinstance(value, Mapping)
        )
    return tuple(cast(Mapping[str, object], _freeze(row)) for row in rows)


def _point_verified_rows(expected: object) -> tuple[Mapping[str, object], ...]:
    verified = getattr(expected, "verified", None)
    if not isinstance(verified, Mapping):
        raise _audit_fail("point expectation lacks verified provenance")
    source_rows = verified.get("source_rows")
    if not isinstance(source_rows, Sequence) or isinstance(source_rows, (str, bytes)):
        raise _audit_fail("point expectation lacks complete verified source rows")
    if any(
        not isinstance(row, Mapping)
        or row.get("category") != "verified_source"
        or not isinstance(row.get("kind"), str)
        for row in source_rows
    ):
        raise _audit_fail("point expectation has malformed verified source rows")
    rows: list[dict[str, object]] = [
        {
            "category": "verified_snapshot_provenance",
            "kind": "identity",
            "plan_id": verified.get("plan_id"),
            "plan_sha256": verified.get("plan_sha256"),
            "connectivity_sha256": verified.get("connectivity_sha256"),
            "semantic_sha256": verified.get("semantic_sha256"),
            "parameters_sha256": verified.get("parameters_sha256"),
            "compiled_graph_sha256": getattr(expected, "compiled_graph_sha256", None),
            "expanded_graph_sha256": getattr(expected, "expanded_graph_sha256", None),
        },
        {
            "category": "verified_snapshot_provenance",
            "kind": "canonical_point_values",
            "values": _plain(verified.get("canonical_values")),
        },
    ]
    if getattr(expected, "representation", None) == "compiled":
        rows.append(
            {
                "category": "verified_snapshot_provenance",
                "kind": "compiled_expansion",
                "resolved_bindings": _plain(verified.get("resolved_bindings")),
            }
        )
    rows.extend(dict(row) for row in source_rows)
    return tuple(cast(Mapping[str, object], _freeze(row)) for row in rows)


def certify_scene(
    point: object,
    scene: NeutralScene,
    *,
    representation: _Representation = "authoring",
    show_values: bool,
    compiled_evidence: Mapping[str, object] | None = None,
) -> DiagramAuditData:
    """Certify one frozen scene by independent expected/visible reconstruction."""

    from .._authoring_snapshot import ResolvedPlanPoint
    from .expected import build_point_expected
    from .witness import witness_authoring, witness_compiled

    if not isinstance(point, ResolvedPlanPoint):
        raise TypeError("certify_scene() requires an immutable ResolvedPlanPoint")
    if not isinstance(scene, NeutralScene):
        raise TypeError("certify_scene() requires a NeutralScene")
    if not isinstance(show_values, bool):
        raise TypeError("show_values must be boolean")
    if representation not in ("authoring", "compiled"):
        raise ValueError("representation must be 'authoring' or 'compiled'")
    _verify_scene_text(scene)
    _verify_internal_clearance(scene)
    _verify_global_clearance(scene)
    expected = build_point_expected(
        point,
        representation=representation,
        compiled_evidence=compiled_evidence,
    )
    if representation == "authoring":
        electrical, structural, detail = witness_authoring(
            point, scene, show_values=show_values
        )
    else:
        if compiled_evidence is None:
            raise TypeError("compiled certification requires exact compiler evidence")
        electrical, structural, detail = witness_compiled(
            point,
            scene,
            compiled_evidence,
            show_values=show_values,
        )
    verified = expected.verified
    required = {
        key: verified.get(key)
        for key in (
            "plan_id",
            "plan_sha256",
            "connectivity_sha256",
            "semantic_sha256",
        )
    }
    if any(not isinstance(value, str) or not value for value in required.values()):
        raise _audit_fail("verified point identity is incomplete")
    return DiagramAuditData(
        representation=representation,
        plan_id=cast(str, required["plan_id"]),
        plan_sha256=cast(str, required["plan_sha256"]),
        connectivity_sha256=cast(str, required["connectivity_sha256"]),
        semantic_sha256=cast(str, required["semantic_sha256"]),
        compiled_graph_sha256=expected.compiled_graph_sha256,
        expanded_graph_sha256=expected.expanded_graph_sha256,
        presentation_sha256=scene_digest(scene),
        observed_electrical=electrical,
        observed_semantic=structural,
        observed_rows=_point_observed_rows(electrical, structural, detail),
        verified_rows=_point_verified_rows(expected),
    )
