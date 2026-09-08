"""Immutable neutral facts emitted by the schematic renderer.

This module is deliberately not an audit model.  It stores only visible ink,
text, anchors, and containment needed for an independent later witness.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from typing import Literal

COORDINATE_TOLERANCE = 1e-9
"""Exact-scene comparison tolerance for later geometric reconstruction."""


@dataclass(frozen=True, slots=True, order=True)
class Point:
    x: float
    y: float

    def translated(self, dx: float, dy: float) -> Point:
        return Point(self.x + dx, self.y + dy)

    def rotated(self, cardinal_degrees: int, *, origin: Point | None = None) -> Point:
        if cardinal_degrees not in (0, 90, 180, 270):
            raise ValueError("scene rotation must be cardinal")
        pivot = Point(0.0, 0.0) if origin is None else origin
        x, y = self.x - pivot.x, self.y - pivot.y
        rotated = {
            0: (x, y),
            90: (-y, x),
            180: (-x, -y),
            270: (y, -x),
        }[cardinal_degrees]
        return Point(pivot.x + rotated[0], pivot.y + rotated[1])


@dataclass(frozen=True, slots=True)
class Bounds:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmin > self.xmax or self.ymin > self.ymax:
            raise ValueError("scene bounds must be ordered")

    @classmethod
    def around(cls, points: Iterable[Point]) -> Bounds:
        materialized = tuple(points)
        if not materialized:
            raise ValueError("scene bounds require at least one point")
        return cls(
            min(p.x for p in materialized),
            min(p.y for p in materialized),
            max(p.x for p in materialized),
            max(p.y for p in materialized),
        )

    def translated(self, dx: float, dy: float) -> Bounds:
        return Bounds(self.xmin + dx, self.ymin + dy, self.xmax + dx, self.ymax + dy)

    def overlaps(self, other: Bounds, *, clearance: float = 0.0) -> bool:
        return not (
            self.xmax + clearance <= other.xmin
            or other.xmax + clearance <= self.xmin
            or self.ymax + clearance <= other.ymin
            or other.ymax + clearance <= self.ymin
        )

    def contains(self, other: Bounds) -> bool:
        return (
            self.xmin <= other.xmin
            and self.ymin <= other.ymin
            and self.xmax >= other.xmax
            and self.ymax >= other.ymax
        )


@dataclass(frozen=True, slots=True)
class Path:
    points: tuple[Point, ...]
    role: Literal[
        "symbol", "wire", "guide", "region", "box", "coupling", "jump", "glyph"
    ]
    closed: bool = False
    codes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("a visible path requires at least two points")
        if self.codes is not None and len(self.codes) != len(self.points):
            raise ValueError("path codes must align with every path point")

    @property
    def bounds(self) -> Bounds:
        return Bounds.around(self.points)

    def translated(self, dx: float, dy: float) -> Path:
        return Path(
            tuple(point.translated(dx, dy) for point in self.points),
            self.role,
            self.closed,
            self.codes,
        )

    def rotated(self, cardinal_degrees: int, *, origin: Point | None = None) -> Path:
        return Path(
            tuple(
                point.rotated(cardinal_degrees, origin=origin) for point in self.points
            ),
            self.role,
            self.closed,
            self.codes,
        )


@dataclass(frozen=True, slots=True)
class GlyphPath:
    character: str
    font_sha256: str
    vertices: tuple[Point, ...]
    codes: tuple[int, ...]
    bounds: Bounds
    advance: float


@dataclass(frozen=True, slots=True)
class TextRun:
    text: str
    origin: Point
    size: float
    role: str
    glyphs: tuple[GlyphPath, ...]
    bounds: Bounds
    ascent: float
    descent: float

    def __post_init__(self) -> None:
        if not self.text or len(self.text) != len(self.glyphs):
            raise ValueError("text runs must retain every literal glyph")


@dataclass(frozen=True, slots=True)
class NativeSymbol:
    kind: Literal["R", "L", "C", "JJ", "G"]
    visible_name: TextRun
    value: TextRun | None
    anchors: tuple[tuple[str, Point], ...]
    paths: tuple[Path, ...]
    symbol_bounds: Bounds
    occupied_bounds: Bounds
    orientation: int
    reference_polarity: Path | None = None
    branch_label: TextRun | None = None
    correlation_key: str | None = None


@dataclass(frozen=True, slots=True)
class ConductivePolyline:
    points: tuple[Point, ...]
    correlation_key: str | None = None

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("conductive polylines require two endpoints")

    @property
    def bounds(self) -> Bounds:
        return Bounds.around(self.points)


@dataclass(frozen=True, slots=True)
class JumpArc:
    path: Path
    correlation_key: str | None = None


@dataclass(frozen=True, slots=True)
class PortBlock:
    """Visible Port facts: circle, circuit T, load, and local return anchors."""

    circle: Path
    circle_center: Point
    boundary_anchor: Point
    circuit_anchor: Point
    external_anchor: Point
    load_anchor: Point
    ground_anchor: Point
    reference_load: NativeSymbol
    paths: tuple[Path, ...]
    ground_glyph: tuple[Path, ...]
    labels: tuple[TextRun, ...]
    occupied_bounds: Bounds
    correlation_key: str | None = None


@dataclass(frozen=True, slots=True)
class ElectricalBox:
    """Compact CPW facts or ordered MTL conductor evidence and physical leads."""

    kind: Literal["CPW", "MTL"]
    orientation: int
    title: TextRun
    kind_label: TextRun
    length_label: TextRun | None
    conductor_rows: tuple[TextRun, ...]
    anchor_labels: tuple[TextRun, ...]
    reference_label: TextRun | None
    anchors: tuple[tuple[str, Point], ...]
    outline: Path
    paths: tuple[Path, ...]
    bounds: Bounds
    correlation_key: str | None = None
    section_label: TextRun | None = None


@dataclass(frozen=True, slots=True)
class SubsystemRegion:
    bounds: Bounds
    boundary: Path
    header: TextRun | None
    identity_anchor: Point
    kind: Literal["root", "composite"]
    correlation_key: str | None = None


@dataclass(frozen=True, slots=True)
class BoundarySite:
    point: Point
    visible_label: TextRun | None
    correlation_key: str | None = None


@dataclass(frozen=True, slots=True)
class NodeMark:
    """Visible filled junction, optionally carrying a compiled-node label."""

    point: Point
    label: TextRun | None
    paths: tuple[Path, ...]
    filled: bool = True
    correlation_key: str | None = None

    def __post_init__(self) -> None:
        if not self.paths or (
            self.filled and not any(path.closed for path in self.paths)
        ):
            raise ValueError(
                "a filled Public node requires a closed visible glyph path"
            )


@dataclass(frozen=True, slots=True)
class GuideMark:
    kind: Literal[
        "bracket", "stitch", "pair", "multiparty", "coupling", "omission", "ground"
    ]
    paths: tuple[Path, ...]
    label: TextRun | None
    terminals: tuple[Point, ...]
    correlation_key: str | None = None

    @property
    def bounds(self) -> Bounds:
        return Bounds.around(point for path in self.paths for point in path.points)

    def __post_init__(self) -> None:
        if not self.paths or any(
            not any(
                terminal == endpoint
                for path in self.paths
                for endpoint in (path.points[0], path.points[-1])
            )
            for terminal in self.terminals
        ):
            raise ValueError("guide terminals must end on actual visible guide strokes")


@dataclass(frozen=True, slots=True)
class ProvenanceBand:
    """Presentation-only certificate text appended after pre-certification."""

    bounds: Bounds
    background: Path
    lines: tuple[TextRun, ...]

    def __post_init__(self) -> None:
        if (
            self.background.role != "region"
            or not self.background.closed
            or not self.lines
        ):
            raise ValueError(
                "provenance requires a visible closed band and nonempty lines"
            )


@dataclass(frozen=True, slots=True)
class NeutralScene:
    """One fully emitted presentation scene, immutable after construction."""

    symbols: tuple[NativeSymbol, ...] = ()
    conductive: tuple[ConductivePolyline, ...] = ()
    jumps: tuple[JumpArc, ...] = ()
    ports: tuple[PortBlock, ...] = ()
    boxes: tuple[ElectricalBox, ...] = ()
    text: tuple[TextRun, ...] = ()
    regions: tuple[SubsystemRegion, ...] = ()
    boundary_sites: tuple[BoundarySite, ...] = ()
    node_marks: tuple[NodeMark, ...] = ()
    guides: tuple[GuideMark, ...] = ()
    provenance_band: ProvenanceBand | None = None
    bounds: Bounds | None = None

    def __post_init__(self) -> None:
        for name in (
            "symbols",
            "conductive",
            "jumps",
            "ports",
            "boxes",
            "text",
            "regions",
            "boundary_sites",
            "node_marks",
            "guides",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"neutral scene {name} must be an immutable tuple")
        roots = tuple(region for region in self.regions if region.kind == "root")
        if len(roots) != 1:
            raise ValueError("every neutral scene retains exactly one root envelope")
        all_bounds = [roots[0].bounds]
        all_bounds.extend(symbol.occupied_bounds for symbol in self.symbols)
        all_bounds.extend(port.occupied_bounds for port in self.ports)
        all_bounds.extend(box.bounds for box in self.boxes)
        all_bounds.extend(run.bounds for run in self.text)
        all_bounds.extend(wire.bounds for wire in self.conductive)
        all_bounds.extend(path.bounds for jump in self.jumps for path in (jump.path,))
        all_bounds.extend(guide.bounds for guide in self.guides)
        all_bounds.extend(
            site.visible_label.bounds
            for site in self.boundary_sites
            if site.visible_label is not None
        )
        all_bounds.extend(
            mark.label.bounds for mark in self.node_marks if mark.label is not None
        )
        all_bounds.extend(
            path.bounds for mark in self.node_marks for path in mark.paths
        )
        corners = tuple(Point(bound.xmin, bound.ymin) for bound in all_bounds) + tuple(
            Point(bound.xmax, bound.ymax) for bound in all_bounds
        )
        physical_bounds = Bounds.around(corners)
        visible_bounds = (
            physical_bounds
            if self.provenance_band is None
            else Bounds.around(
                (
                    Point(physical_bounds.xmin, physical_bounds.ymin),
                    Point(physical_bounds.xmax, physical_bounds.ymax),
                    Point(
                        self.provenance_band.bounds.xmin,
                        self.provenance_band.bounds.ymin,
                    ),
                    Point(
                        self.provenance_band.bounds.xmax,
                        self.provenance_band.bounds.ymax,
                    ),
                )
            )
        )
        if self.bounds is None:
            object.__setattr__(self, "bounds", visible_bounds)
        elif not self.bounds.contains(visible_bounds):
            raise ValueError("scene bounds must contain every emitted fact")
        if not roots[0].bounds.contains(physical_bounds):
            raise ValueError("root scene envelope must contain every emitted fact")


class SceneBuilder:
    """Private construction helper; its frozen output has no answer tables."""

    def __init__(self) -> None:
        self.symbols: list[NativeSymbol] = []
        self.conductive: list[ConductivePolyline] = []
        self.jumps: list[JumpArc] = []
        self.ports: list[PortBlock] = []
        self.boxes: list[ElectricalBox] = []
        self.text: list[TextRun] = []
        self.regions: list[SubsystemRegion] = []
        self.boundary_sites: list[BoundarySite] = []
        self.node_marks: list[NodeMark] = []
        self.guides: list[GuideMark] = []

    def freeze(self) -> NeutralScene:
        return NeutralScene(
            symbols=tuple(self.symbols),
            conductive=tuple(self.conductive),
            jumps=tuple(self.jumps),
            ports=tuple(self.ports),
            text=tuple(self.text),
            boxes=tuple(self.boxes),
            regions=tuple(self.regions),
            boundary_sites=tuple(self.boundary_sites),
            node_marks=tuple(self.node_marks),
            guides=tuple(self.guides),
        )


def _json_value(value: object) -> object:
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
            if field.name != "correlation_key"
        }
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def scene_digest(scene: NeutralScene) -> str:
    """Canonical presentation identity, excluding private correlation/provenance."""

    if not isinstance(scene, NeutralScene):
        raise TypeError("scene_digest requires a NeutralScene")
    # The root envelope remains a physical scene fact.  The outer output extent
    # and late certificate band are intentionally presentation-only and cannot
    # make a digest self-referential.
    identity = {
        field.name: _json_value(getattr(scene, field.name))
        for field in fields(scene)
        if field.name not in {"bounds", "provenance_band"}
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def append_provenance_band(scene: NeutralScene, lines: tuple[str, ...]) -> NeutralScene:
    """Append measured certificate text without changing physical scene facts."""

    from .metrics import DEFAULT_METRICS, shape_text

    if not isinstance(scene, NeutralScene):
        raise TypeError("append_provenance_band requires a NeutralScene")
    if scene.provenance_band is not None:
        raise ValueError("a scene can have one provenance band")
    if (
        not isinstance(lines, tuple)
        or not lines
        or any(not isinstance(line, str) or not line for line in lines)
    ):
        raise TypeError("provenance lines must be a nonempty tuple of literal strings")
    root = next(region for region in scene.regions if region.kind == "root")
    metrics = DEFAULT_METRICS
    font_size = metrics.secondary_text_size
    measured = tuple(
        shape_text(line, at=Point(0.0, 0.0), size=font_size, role="provenance")
        for line in lines
    )
    padding = metrics.terminal_stub
    width = max(
        root.bounds.xmax - root.bounds.xmin,
        max(run.bounds.xmax - run.bounds.xmin for run in measured) + 2 * padding,
    )
    height = sum(run.bounds.ymax - run.bounds.ymin for run in measured) + padding * (
        len(measured) + 1
    )
    center_x = (root.bounds.xmin + root.bounds.xmax) / 2
    top = root.bounds.ymin - padding
    bottom = top - height
    left, right = center_x - width / 2, center_x + width / 2
    runs: list[TextRun] = []
    cursor = top - padding
    for line, measure in zip(lines, measured):
        line_height = measure.bounds.ymax - measure.bounds.ymin
        provisional = shape_text(
            line, at=Point(0.0, cursor - line_height), size=font_size, role="provenance"
        )
        text_width = provisional.bounds.xmax - provisional.bounds.xmin
        runs.append(
            shape_text(
                line,
                at=Point(center_x - text_width / 2, cursor - line_height),
                size=font_size,
                role="provenance",
            )
        )
        cursor -= line_height + padding
    band_bounds = Bounds(left, bottom, right, top)
    band = ProvenanceBand(
        band_bounds,
        Path(
            (
                Point(left, bottom),
                Point(right, bottom),
                Point(right, top),
                Point(left, top),
                Point(left, bottom),
            ),
            "region",
            True,
        ),
        tuple(runs),
    )
    return NeutralScene(
        symbols=scene.symbols,
        conductive=scene.conductive,
        jumps=scene.jumps,
        ports=scene.ports,
        boxes=scene.boxes,
        text=scene.text,
        regions=scene.regions,
        boundary_sites=scene.boundary_sites,
        node_marks=scene.node_marks,
        guides=scene.guides,
        provenance_band=band,
    )
