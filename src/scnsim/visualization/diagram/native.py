"""Native-symbol and transmission-line scene emission.

Schemdraw remains the trusted source for R/L/C/JJ/G glyph geometry.  We bind
labels to a real placed element before extracting its transformed anchors and
segments, then freeze equivalent literal text as exact Matplotlib glyph paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, cos, isclose, isfinite, pi, radians, sin, tan
from typing import Literal

from .metrics import DEFAULT_METRICS, DiagramMetrics, shape_text
from .scene import (
    COORDINATE_TOLERANCE,
    Bounds,
    ConductivePolyline,
    ElectricalBox,
    GuideMark,
    JumpArc,
    NativeSymbol,
    NodeMark,
    Path,
    Point,
    PortBlock,
    TextRun,
)
from .values import format_envelope

NativeKind = Literal["R", "L", "C", "JJ", "G"]
_NATIVE_WHITELIST = frozenset({"R", "L", "C", "JJ", "G"})


def electrical_resolution_label(policy: Mapping[str, object]) -> str:
    """The exact visible authoring policy; no candidate N is inferred here."""

    if not isinstance(policy, Mapping) or set(policy) != {"kind", "max_frequency", "sections_per_wavelength"} or policy["kind"] != "electrical_resolution":
        raise ValueError("line resolution policy is malformed")
    count = policy["sections_per_wavelength"]
    frequency = policy["max_frequency"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or not isinstance(frequency, Mapping):
        raise ValueError("line resolution policy is malformed")
    return f"M={count}, fmax={format_envelope(frequency)}"


def _literal(text: str, *, controlled_math: bool = False) -> str:
    if not isinstance(text, str) or not text:
        raise ValueError("native labels must be nonempty literal text")
    # Plan identifiers are visible literals, not TeX.  Preserve their Unicode
    # bytes exactly; only a truly unavailable font glyph is a render failure.
    del controlled_math
    return text


def _bounds(*items: Bounds) -> Bounds:
    return Bounds(
        min(item.xmin for item in items),
        min(item.ymin for item in items),
        max(item.xmax for item in items),
        max(item.ymax for item in items),
    )


def _orientation(start: Point, end: Point) -> int:
    dx, dy = end.x - start.x, end.y - start.y
    if abs(dy) <= COORDINATE_TOLERANCE and abs(dx) > COORDINATE_TOLERANCE:
        return 0 if dx > 0 else 180
    if abs(dx) <= COORDINATE_TOLERANCE and abs(dy) > COORDINATE_TOLERANCE:
        return 90 if dy > 0 else 270
    raise ValueError("native symbols require cardinal anchors")


def _symbol_element(kind: NativeKind, *, color: str):
    import schemdraw.elements as elm

    classes = {
        "R": elm.Resistor,
        "L": elm.Inductor,
        "C": elm.Capacitor,
        "JJ": elm.Josephson,
        "G": elm.Resistor,
    }
    if kind not in _NATIVE_WHITELIST:
        raise ValueError(f"untrusted native symbol kind {kind!r}")
    return classes[kind](color=color)


def _text_centered(text: str, center: Point, *, size: float, role: str) -> TextRun:
    provisional = shape_text(text, at=Point(0.0, 0.0), size=size, role=role)
    midpoint = Point(
        (provisional.bounds.xmin + provisional.bounds.xmax) / 2,
        (provisional.bounds.ymin + provisional.bounds.ymax) / 2,
    )
    return shape_text(
        text,
        at=Point(center.x - midpoint.x, center.y - midpoint.y),
        size=size,
        role=role,
    )


def _extent(run: TextRun, *, horizontal: bool) -> float:
    return (
        run.bounds.xmax - run.bounds.xmin
        if horizontal
        else run.bounds.ymax - run.bounds.ymin
    )


def _point_on_path_end(point: Point, paths: tuple[Path, ...]) -> bool:
    return any(
        abs(point.x - endpoint.x) <= 1e-9 and abs(point.y - endpoint.y) <= 1e-9
        for path in paths
        for endpoint in (path.points[0], path.points[-1])
    )


def _arc_paths(segment: object) -> tuple[Path, ...]:
    """Freeze Schemdraw SegmentArc geometry as cubic paths, not omitted leads."""

    center = segment.center
    width, height = (
        float(segment.width) / 2,
        float(segment.height) / 2,
    )
    rotation = radians(float(segment.angle))
    begin, finish = (
        radians(float(segment.theta1)),
        radians(float(segment.theta2)),
    )
    count = max(1, ceil(abs(finish - begin) / (pi / 2)))
    paths: list[Path] = []

    def point(theta: float) -> Point:
        x, y = width * cos(theta), height * sin(theta)
        return Point(
            float(center[0]) + x * cos(rotation) - y * sin(rotation),
            float(center[1]) + x * sin(rotation) + y * cos(rotation),
        )

    for index in range(count):
        a, b = (
            begin + (finish - begin) * index / count,
            begin + (finish - begin) * (index + 1) / count,
        )
        tangent = 4 * tan((b - a) / 4) / 3
        derivative_a = Point(-width * sin(a), height * cos(a))
        derivative_b = Point(-width * sin(b), height * cos(b))

        def rotate_derivative(item: Point) -> Point:
            return Point(
                item.x * cos(rotation) - item.y * sin(rotation),
                item.x * sin(rotation) + item.y * cos(rotation),
            )

        da, db = rotate_derivative(derivative_a), rotate_derivative(derivative_b)
        start, end = point(a), point(b)
        paths.append(
            Path(
                (
                    start,
                    Point(start.x + tangent * da.x, start.y + tangent * da.y),
                    Point(end.x - tangent * db.x, end.y - tangent * db.y),
                    end,
                ),
                "symbol",
                False,
                (1, 4, 4, 4),
            )
        )
    return tuple(paths)


def native_block(
    kind: NativeKind,
    *,
    start: Point,
    end: Point,
    name: str,
    value: str | None,
    color: str = "#243044",
    metrics: DiagramMetrics = DEFAULT_METRICS,
    correlation_key: str | None = None,
    branch_label: str | None = None,
    label_side: Literal["top", "bottom", "left", "right"] | None = None,
) -> NativeSymbol:
    """Emit one actual native Schemdraw symbol and its frozen visible facts."""

    if kind not in _NATIVE_WHITELIST:
        raise ValueError(f"untrusted native symbol kind {kind!r}")
    if (
        abs((end.x - start.x) ** 2 + (end.y - start.y) ** 2 - metrics.native_span**2)
        > 1e-7
    ):
        raise ValueError("native electrical anchor span must equal one U")
    name = _literal(name)
    if value is not None:
        value = _literal(value)
    if branch_label is not None:
        branch_label = _literal(branch_label)
    orientation = _orientation(start, end)
    horizontal = orientation in {0, 180}
    label_loc = ("top" if horizontal else "left") if label_side is None else label_side
    if label_loc not in {"top", "bottom", "left", "right"}:
        raise ValueError("native label side must be top, bottom, left, or right")
    name_measure = _text_centered(
        name,
        Point(0.0, 0.0),
        size=metrics.primary_text_size,
        role="symbol-name",
    )
    value_measure = (
        _text_centered(
            value,
            Point(0.0, 0.0),
            size=metrics.primary_text_size,
            role="symbol-value",
        )
        if value is not None
        else None
    )
    branch_measure = (
        _text_centered(
            branch_label,
            Point(0.0, 0.0),
            size=metrics.tertiary_text_size,
            role="branch-label",
        )
        if branch_label is not None
        else None
    )
    label_text = name if value is None else f"{name}\n{value}"
    # Schemdraw label locations rotate with the native element.  Select its
    # local location so ``label_loc`` stays a global/external scene side.
    bound_loc = {
        0: {"top": "top", "bottom": "bottom", "left": "left", "right": "right"},
        90: {"top": "right", "right": "bottom", "bottom": "left", "left": "top"},
        180: {"top": "top", "bottom": "bottom", "left": "left", "right": "right"},
        270: {"top": "right", "right": "bottom", "bottom": "left", "left": "top"},
    }[orientation][label_loc]
    # Label binding happens before placement/extraction.  The separately
    # frozen TextRuns below carry the glyph paths used by the neutral scene.
    import schemdraw

    drawing = schemdraw.Drawing(show=False)
    drawing.config(unit=metrics.unit_length, fontsize=10, color=color)
    element = (
        _symbol_element(kind, color=color).at((start.x, start.y)).to((end.x, end.y))
    )
    element.label(
        label_text,
        loc=bound_loc,
        ofst=metrics.label_clearance,
        rotate=False,
        color=color,
    )
    placed = drawing.add(element)
    extracted: list[Path] = []
    label_anchors: dict[str, Point] = {}
    for segment in placed.segments:
        if getattr(segment, "text", None) is not None:
            transformed_text = segment.xform(placed.transform, **placed.elmparams)
            label_anchors[str(transformed_text.text)] = Point(
                float(transformed_text.xy[0]), float(transformed_text.xy[1])
            )
            continue
        raw_path = getattr(segment, "path", None)
        if raw_path is None:
            if type(segment).__name__ == "SegmentArc":
                extracted.extend(
                    _arc_paths(segment.xform(placed.transform, **placed.elmparams))
                )
            continue
        transformed = segment.xform(placed.transform, **placed.elmparams).path
        current: list[Point] = []
        for point in transformed:
            x, y = float(point[0]), float(point[1])
            if not (isfinite(x) and isfinite(y)):
                if len(current) >= 2:
                    extracted.append(Path(tuple(current), "symbol"))
                current = []
                continue
            current.append(Point(x, y))
        if len(current) >= 2:
            extracted.append(Path(tuple(current), "symbol"))
    if not extracted:
        raise RuntimeError("Schemdraw emitted no native symbol path")
    frozen_paths = tuple(extracted)
    if not all(_point_on_path_end(anchor, frozen_paths) for anchor in (start, end)):
        raise RuntimeError(
            "native electrical anchor does not terminate an actual Schemdraw lead"
        )
    symbol_bounds = Bounds.around(point for path in extracted for point in path.points)
    try:
        label_anchor = label_anchors[label_text]
    except KeyError as exc:
        raise RuntimeError("Schemdraw failed to retain a bound native label") from exc
    name_width = _extent(name_measure, horizontal=True)
    name_height = _extent(name_measure, horizontal=False)
    value_width = (
        0.0 if value_measure is None else _extent(value_measure, horizontal=True)
    )
    value_height = (
        0.0 if value_measure is None else _extent(value_measure, horizontal=False)
    )
    if label_loc == "top":
        value_center = Point(
            label_anchor.x,
            symbol_bounds.ymax + metrics.label_clearance + value_height / 2,
        )
        name_center = Point(
            label_anchor.x,
            symbol_bounds.ymax + metrics.label_clearance + name_height / 2
            if value_measure is None
            else value_center.y
            + value_height / 2
            + metrics.label_clearance
            + name_height / 2,
        )
    elif label_loc == "bottom":
        name_center = Point(
            label_anchor.x,
            symbol_bounds.ymin - metrics.label_clearance - name_height / 2,
        )
        value_center = Point(
            label_anchor.x,
            name_center.y
            - name_height / 2
            - metrics.label_clearance
            - value_height / 2,
        )
    else:
        rows_height = name_height + (
            0.0 if value_measure is None else metrics.label_clearance + value_height
        )
        name_center_y = label_anchor.y + rows_height / 2 - name_height / 2
        value_center_y = (
            name_center_y - name_height / 2 - metrics.label_clearance - value_height / 2
        )
        edge = (
            symbol_bounds.xmin - metrics.label_clearance
            if label_loc == "left"
            else symbol_bounds.xmax + metrics.label_clearance
        )
        name_center = Point(
            edge - name_width / 2 if label_loc == "left" else edge + name_width / 2,
            name_center_y,
        )
        value_center = Point(
            edge - value_width / 2 if label_loc == "left" else edge + value_width / 2,
            value_center_y,
        )
    name_run = _text_centered(
        name,
        name_center,
        size=metrics.primary_text_size,
        role="symbol-name",
    )
    value_run = (
        _text_centered(
            value,
            value_center,
            size=metrics.primary_text_size,
            role="symbol-value",
        )
        if value is not None
        else None
    )
    occupied_items = (
        (symbol_bounds, name_run.bounds)
        if value_run is None
        else (symbol_bounds, name_run.bounds, value_run.bounds)
    )
    occupied = _bounds(*occupied_items)
    branch_run = None
    if branch_label is not None and branch_measure is not None:
        badge_width = branch_measure.bounds.xmax - branch_measure.bounds.xmin
        badge_center = Point(
            name_run.bounds.xmin - metrics.label_clearance - badge_width / 2
            if label_loc == "left"
            else name_run.bounds.xmax + metrics.label_clearance + badge_width / 2,
            name_center.y,
        )
        branch_run = _text_centered(
            branch_label,
            badge_center,
            size=metrics.tertiary_text_size,
            role="branch-label",
        )
    if branch_run is not None:
        occupied = _bounds(occupied, branch_run.bounds)
    return NativeSymbol(
        kind,
        name_run,
        value_run,
        (("terminal_1", start), ("terminal_2", end)),
        frozen_paths,
        symbol_bounds,
        occupied,
        orientation,
        None,
        branch_run,
        correlation_key,
    )


def _cpw_box(
    *, origin: Point, title: str, conductor: str, length: str | None,
    section_text: str, section_role: str, minimum_axis: float, orientation: int,
    metrics: DiagramMetrics, correlation_key: str | None,
) -> ElectricalBox:
    """One-conductor line with upright measured facts and two physical leads."""
    labels = [("CPW", metrics.secondary_text_size, "line-kind"),
              (title, metrics.primary_text_size, "line-title")]
    if length is not None:
        labels.append((length, metrics.secondary_text_size, "line-length"))
    labels.append((section_text, metrics.secondary_text_size, section_role))
    measured = tuple(shape_text(text, at=Point(0.0, 0.0), size=size, role=role)
                     for text, size, role in labels)
    padding = metrics.label_clearance
    rows = (measured[:2], measured[2:])
    row_widths = tuple(sum(_extent(run, horizontal=True) for run in row)
                       + padding * (len(row) - 1) for row in rows)
    row_limits = tuple((min(run.bounds.ymin for run in row),
                        max(run.bounds.ymax for run in row)) for row in rows)
    horizontal = orientation in {0, 180}
    body_width = max(minimum_axis if horizontal else metrics.native_span,
                     max(row_widths) + 2 * padding)
    body_height = max(metrics.native_span if horizontal else minimum_axis,
                      sum(high - low for low, high in row_limits) + 3 * padding)
    dx, dy = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[orientation]
    center = origin.translated(dx * body_width / 2, dy * body_height / 2)
    body = Bounds(center.x - body_width / 2, center.y - body_height / 2,
                  center.x + body_width / 2, center.y + body_height / 2)
    outline = Path((Point(body.xmin, body.ymin), Point(body.xmax, body.ymin),
                    Point(body.xmax, body.ymax), Point(body.xmin, body.ymax),
                    Point(body.xmin, body.ymin)), "box", closed=True)
    content_height = sum(high - low for low, high in row_limits) + padding
    top = center.y + content_height / 2
    placed: dict[str, TextRun] = {}
    for row, row_width, (low, high) in zip(rows, row_widths, row_limits, strict=True):
        left = center.x - row_width / 2
        baseline = top - high
        for run in row:
            placed[run.role] = shape_text(run.text,
                at=Point(left - run.bounds.xmin, baseline), size=run.size, role=run.role)
            left += _extent(run, horizontal=True) + padding
        top -= high - low + padding
    tail_contact = origin.translated(dx * body_width, dy * body_height)
    head = origin.translated(-dx * metrics.terminal_stub, -dy * metrics.terminal_stub)
    tail = tail_contact.translated(dx * metrics.terminal_stub, dy * metrics.terminal_stub)
    paths = (Path((head, origin), "wire"), Path((tail_contact, tail), "wire"))
    return ElectricalBox(
        "CPW", orientation, placed["line-title"], placed["line-kind"],
        placed.get("line-length"), (), (), None,
        ((f"head.{conductor}", head), (f"tail.{conductor}", tail)),
        outline, paths, _bounds(body, *(path.bounds for path in paths)),
        correlation_key, section_label=placed[section_role],
    )


def electrical_box(
    *,
    kind: Literal["CPW", "MTL"],
    origin: Point,
    title: str,
    conductors: tuple[str, ...],
    reference: str | None = None,
    length: str | None = None,
    n_sections: int | None = None,
    resolution: Mapping[str, object] | None = None,
    width: float | None = None,
    orientation: Literal[0, 90, 180, 270] = 0,
    metrics: DiagramMetrics = DEFAULT_METRICS,
    correlation_key: str | None = None,
) -> ElectricalBox:
    """Emit compact CPW facts or ordered MTL rows with real physical anchors."""

    if kind not in {"CPW", "MTL"} or not conductors:
        raise ValueError("electrical boxes require CPW/MTL and ordered conductors")
    if orientation not in {0, 90, 180, 270}:
        raise ValueError("electrical box orientation must be cardinal")
    title = _literal(title)
    if reference is not None:
        reference = _literal(reference)
    if length is not None:
        length = _literal(length)
    if len(set(conductors)) != len(conductors):
        raise ValueError("MTL conductor rows must be unique")
    if any(_literal(item) != item for item in conductors):
        raise AssertionError("unreachable")
    minimum_axis = 2 * metrics.native_span if width is None else float(width)
    if not isfinite(minimum_axis) or minimum_axis <= 0:
        raise ValueError("electrical box width must be positive")
    if (n_sections is None) == (resolution is None):
        raise ValueError("line box requires exactly one fixed count or resolution policy")
    policy_text = None if resolution is None else electrical_resolution_label(resolution)
    if kind == "CPW":
        if len(conductors) != 1:
            raise ValueError("CPW boxes require exactly one conductor")
        if n_sections is not None and (isinstance(n_sections, bool) or not isinstance(n_sections, int) or n_sections <= 0):
            raise ValueError("CPW boxes require the captured positive section count")
        section_text = policy_text if policy_text is not None else f"{n_sections} section" + ("s" if n_sections != 1 else "")
        return _cpw_box(origin=origin, title=title, conductor=conductors[0], length=length,
                        section_text=section_text, section_role="line-policy" if policy_text is not None else "line-sections",
                        minimum_axis=minimum_axis,
                        orientation=orientation, metrics=metrics,
                        correlation_key=correlation_key)

    padding = metrics.label_clearance
    gap = metrics.label_clearance
    title_measure = _text_centered(
        title,
        Point(0.0, 0.0),
        size=metrics.primary_text_size,
        role="line-title",
    )
    kind_measure = _text_centered(
        kind,
        Point(0.0, 0.0),
        size=metrics.secondary_text_size,
        role="line-kind",
    )
    length_measure = (
        _text_centered(
            length,
            Point(0.0, 0.0),
            size=metrics.secondary_text_size,
            role="line-length",
        )
        if length is not None
        else None
    )
    row_measures = tuple(
        _text_centered(
            conductor,
            Point(0.0, 0.0),
            size=metrics.secondary_text_size,
            role="conductor-row",
        )
        for conductor in conductors
    )
    head_measures = tuple(
        _text_centered(
            f"head.{conductor}",
            Point(0.0, 0.0),
            size=metrics.tertiary_text_size,
            role="line-head-anchor",
        )
        for conductor in conductors
    )
    tail_measures = tuple(
        _text_centered(
            f"tail.{conductor}",
            Point(0.0, 0.0),
            size=metrics.tertiary_text_size,
            role="line-tail-anchor",
        )
        for conductor in conductors
    )
    reference_measure = (
        _text_centered(
            reference,
            Point(0.0, 0.0),
            size=metrics.secondary_text_size,
            role="reference-conductor",
        )
        if reference is not None
        else None
    )
    policy_measure = (
        _text_centered(policy_text, Point(0.0, 0.0), size=metrics.secondary_text_size, role="line-policy")
        if policy_text is not None else None
    )

    header_measures = tuple(
        item
        for item in (kind_measure, title_measure, length_measure)
        if item is not None
    )
    header_width = sum(_extent(item, horizontal=True) for item in header_measures)
    header_width += gap * (len(header_measures) - 1) + 2 * padding
    header_height = max(_extent(item, horizontal=False) for item in header_measures)
    footer_width = max((_extent(item, horizontal=True) + 2 * padding for item in (reference_measure, policy_measure) if item is not None), default=0.0)
    footer_height = sum(_extent(item, horizontal=False) for item in (reference_measure, policy_measure) if item is not None) + (gap if reference_measure is not None and policy_measure is not None else 0.0)
    horizontal = orientation in {0, 180}
    if horizontal:
        row_widths = tuple(
            _extent(head, horizontal=True)
            + _extent(row, horizontal=True)
            + _extent(tail, horizontal=True)
            + 2 * gap
            + 2 * padding
            for head, row, tail in zip(
                head_measures, row_measures, tail_measures, strict=True
            )
        )
        row_heights = tuple(
            max(
                _extent(head, horizontal=False),
                _extent(row, horizontal=False),
                _extent(tail, horizontal=False),
            )
            for head, row, tail in zip(
                head_measures, row_measures, tail_measures, strict=True
            )
        )
        # Text packing must also leave neighboring physical lead rays outside
        # the existing obstacle clearance. Reuse these gaps in body and anchor
        # placement, so their measurements describe the same scene.
        row_gaps = tuple(
            max(gap, metrics.obstacle_clearance - (first + second) / 2)
            for first, second in zip(row_heights, row_heights[1:])
        )
        body_width = max(minimum_axis, header_width, footer_width, *row_widths)
        content_height = (
            2 * padding
            + header_height
            + gap
            + sum(row_heights)
            + sum(row_gaps)
            + (gap + footer_height if footer_height else 0.0)
        )
        body_height = max(metrics.native_span, content_height)
    else:
        column_widths = tuple(
            max(
                _extent(head, horizontal=True),
                _extent(row, horizontal=True),
                _extent(tail, horizontal=True),
            )
            for head, row, tail in zip(
                head_measures, row_measures, tail_measures, strict=True
            )
        )
        column_gaps = tuple(
            max(gap, metrics.obstacle_clearance - (first + second) / 2)
            for first, second in zip(column_widths, column_widths[1:])
        )
        endpoint_height = max(
            *(_extent(item, horizontal=False) for item in head_measures),
            *(_extent(item, horizontal=False) for item in tail_measures),
        )
        row_height = max(_extent(item, horizontal=False) for item in row_measures)
        body_width = max(
            metrics.native_span,
            header_width,
            footer_width,
            2 * padding + sum(column_widths) + sum(column_gaps),
        )
        content_height = (
            2 * padding
            + header_height
            + endpoint_height
            + row_height
            + footer_height
            + gap * (4 if footer_height else 3)
        )
        body_height = max(minimum_axis, content_height)

    if orientation == 0:
        left, right = origin.x, origin.x + body_width
        bottom, top = origin.y - body_height / 2, origin.y + body_height / 2
    elif orientation == 180:
        left, right = origin.x - body_width, origin.x
        bottom, top = origin.y - body_height / 2, origin.y + body_height / 2
    elif orientation == 90:
        left, right = origin.x - body_width / 2, origin.x + body_width / 2
        bottom, top = origin.y, origin.y + body_height
    else:
        left, right = origin.x - body_width / 2, origin.x + body_width / 2
        bottom, top = origin.y - body_height, origin.y

    outline = Path(
        (
            Point(left, bottom),
            Point(right, bottom),
            Point(right, top),
            Point(left, top),
            Point(left, bottom),
        ),
        "box",
        True,
    )

    header_total = sum(_extent(item, horizontal=True) for item in header_measures)
    header_total += gap * (len(header_measures) - 1)
    header_cursor = (left + right - header_total) / 2
    header_runs: list[TextRun] = []
    header_y = top - padding - header_height / 2
    for measure in header_measures:
        item_width = _extent(measure, horizontal=True)
        header_runs.append(
            _text_centered(
                measure.text,
                Point(header_cursor + item_width / 2, header_y),
                size=measure.size,
                role=measure.role,
            )
        )
        header_cursor += item_width + gap
    kind_run = next(run for run in header_runs if run.role == "line-kind")
    title_run = next(run for run in header_runs if run.role == "line-title")
    length_run = next((run for run in header_runs if run.role == "line-length"), None)
    reference_run = (
        None
        if reference_measure is None
        else _text_centered(
            reference_measure.text,
            Point((left + right) / 2, bottom + padding + _extent(reference_measure, horizontal=False) / 2),
            size=reference_measure.size,
            role=reference_measure.role,
        )
    )
    policy_run = (
        None if policy_measure is None else _text_centered(
            policy_measure.text,
            Point((left + right) / 2, bottom + padding + footer_height - _extent(policy_measure, horizontal=False) / 2),
            size=policy_measure.size, role=policy_measure.role,
        )
    )

    rows: list[TextRun] = []
    placed_heads: list[TextRun] = []
    placed_tails: list[TextRun] = []
    anchors_by_end: dict[str, list[tuple[str, Point]]] = {"head": [], "tail": []}
    paths: list[Path] = []
    if horizontal:
        available_top = header_y - header_height / 2 - gap
        available_bottom = (
            (policy_run or reference_run).bounds.ymax + gap
            if policy_run is not None or reference_run is not None
            else bottom + padding
        )
        packed_height = sum(row_heights) + sum(row_gaps)
        cursor = (available_top + available_bottom + packed_height) / 2
        row_centers = [0.0] * len(row_heights)
        row_order = (
            range(len(row_heights))
            if orientation == 0
            else reversed(range(len(row_heights)))
        )
        ordered_gaps = row_gaps if orientation == 0 else row_gaps[::-1]
        for position, index in enumerate(row_order):
            row_height_value = row_heights[index]
            row_centers[index] = cursor - row_height_value / 2
            cursor -= row_height_value
            if position < len(ordered_gaps):
                cursor -= ordered_gaps[position]
        direction = 1.0 if orientation == 0 else -1.0
        head_border_x = origin.x
        tail_border_x = origin.x + direction * body_width
        for conductor, head, row, tail, y in zip(
            conductors,
            head_measures,
            row_measures,
            tail_measures,
            row_centers,
            strict=True,
        ):
            measures = (head, row, tail)
            free_gap = (
                body_width
                - 2 * padding
                - sum(_extent(item, horizontal=True) for item in measures)
            ) / 2
            distance = padding
            placed: list[TextRun] = []
            for measure in measures:
                item_width = _extent(measure, horizontal=True)
                placed.append(
                    _text_centered(
                        measure.text,
                        Point(
                            head_border_x + direction * (distance + item_width / 2), y
                        ),
                        size=measure.size,
                        role=measure.role,
                    )
                )
                distance += item_width + free_gap
            placed_heads.append(placed[0])
            rows.append(placed[1])
            placed_tails.append(placed[2])
            head_contact = Point(head_border_x, y)
            tail_contact = Point(tail_border_x, y)
            head_anchor = head_contact.translated(
                -direction * metrics.terminal_stub, 0.0
            )
            tail_anchor = tail_contact.translated(
                direction * metrics.terminal_stub, 0.0
            )
            anchors_by_end["head"].append((f"head.{conductor}", head_anchor))
            anchors_by_end["tail"].append((f"tail.{conductor}", tail_anchor))
            paths.extend(
                (
                    Path((head_anchor, head_contact), "wire"),
                    Path((tail_contact, tail_anchor), "wire"),
                )
            )
    else:
        packed_width = sum(column_widths) + sum(column_gaps)
        cursor = (left + right - packed_width) / 2
        column_centers = [0.0] * len(column_widths)
        column_order = (
            range(len(column_widths))
            if orientation == 90
            else reversed(range(len(column_widths)))
        )
        ordered_gaps = column_gaps if orientation == 90 else column_gaps[::-1]
        for position, index in enumerate(column_order):
            column_width = column_widths[index]
            column_centers[index] = cursor + column_width / 2
            cursor += column_width
            if position < len(ordered_gaps):
                cursor += ordered_gaps[position]

        footer_top = (
            (policy_run or reference_run).bounds.ymax if policy_run is not None or reference_run is not None else bottom + padding
        )
        low_height = max(
            _extent(item, horizontal=False)
            for item in (head_measures if orientation == 90 else tail_measures)
        )
        high_height = max(
            _extent(item, horizontal=False)
            for item in (tail_measures if orientation == 90 else head_measures)
        )
        conductor_height = max(_extent(item, horizontal=False) for item in row_measures)
        low_y = footer_top + gap + low_height / 2
        high_y = header_y - header_height / 2 - gap - high_height / 2
        conductor_y = (low_y + low_height / 2 + high_y - high_height / 2) / 2
        if conductor_y - conductor_height / 2 < low_y + low_height / 2 + gap:
            raise RuntimeError("measured vertical line-box rows do not fit")
        if conductor_y + conductor_height / 2 > high_y - high_height / 2 - gap:
            raise RuntimeError("measured vertical line-box rows do not fit")

        head_y, tail_y = (low_y, high_y) if orientation == 90 else (high_y, low_y)
        head_border_y = origin.y
        vertical_direction = 1.0 if orientation == 90 else -1.0
        tail_border_y = origin.y + vertical_direction * body_height
        for conductor, head, row, tail, x in zip(
            conductors,
            head_measures,
            row_measures,
            tail_measures,
            column_centers,
            strict=True,
        ):
            placed_heads.append(
                _text_centered(
                    head.text,
                    Point(x, head_y),
                    size=head.size,
                    role=head.role,
                )
            )
            rows.append(
                _text_centered(
                    row.text,
                    Point(x, conductor_y),
                    size=row.size,
                    role=row.role,
                )
            )
            placed_tails.append(
                _text_centered(
                    tail.text,
                    Point(x, tail_y),
                    size=tail.size,
                    role=tail.role,
                )
            )
            head_contact = Point(x, head_border_y)
            tail_contact = Point(x, tail_border_y)
            head_anchor = head_contact.translated(
                0.0, -vertical_direction * metrics.terminal_stub
            )
            tail_anchor = tail_contact.translated(
                0.0, vertical_direction * metrics.terminal_stub
            )
            anchors_by_end["head"].append((f"head.{conductor}", head_anchor))
            anchors_by_end["tail"].append((f"tail.{conductor}", tail_anchor))
            paths.extend(
                (
                    Path((head_anchor, head_contact), "wire"),
                    Path((tail_contact, tail_anchor), "wire"),
                )
            )

    anchor_labels = (*placed_heads, *placed_tails)
    anchors = (*anchors_by_end["head"], *anchors_by_end["tail"])
    visible_bounds = [
        outline.bounds,
        *(path.bounds for path in paths),
        title_run.bounds,
        kind_run.bounds,
        *(row.bounds for row in rows),
        *(label.bounds for label in anchor_labels),
    ]
    if reference_run is not None:
        visible_bounds.append(reference_run.bounds)
    if policy_run is not None:
        visible_bounds.append(policy_run.bounds)
    if length_run is not None:
        visible_bounds.append(length_run.bounds)
    bounds = _bounds(*visible_bounds)
    return ElectricalBox(
        kind,
        orientation,
        title_run,
        kind_run,
        length_run,
        tuple(rows),
        anchor_labels,
        reference_run,
        anchors,
        outline,
        tuple(paths),
        bounds,
        correlation_key,
        section_label=policy_run,
    )


def conductive_wire(
    *points: Point, correlation_key: str | None = None
) -> ConductivePolyline:
    """Return one complete visible conductive polyline, without a hidden net ID."""

    return ConductivePolyline(tuple(points), correlation_key)


def junction_mark(
    at: Point,
    *,
    metrics: DiagramMetrics = DEFAULT_METRICS,
    correlation_key: str | None = None,
) -> NodeMark:
    """Emit one unlabelled, filled electrical T/junction dot from visible ink."""

    radius = metrics.port_circle_radius * 0.7
    outline = Path(
        tuple(
            Point(
                at.x + radius * cos(2 * pi * index / 16),
                at.y + radius * sin(2 * pi * index / 16),
            )
            for index in range(16)
        ),
        "symbol",
        True,
    )
    return NodeMark(at, None, (outline,), True, correlation_key)


def winding_dot(at: Point, other: Point, bounds: Bounds, *, orientation: int = 0) -> tuple[Point, Path]:
    """Place a nonconductive winding dot beside one geometric terminal lead.

    Unlike an electrical junction dot, its center is off the wire. A dotted
    mutual leader terminates at this mark, not at a conductive graph node.
    """
    if orientation not in (0, 90, 180, 270):
        raise ValueError("winding-dot orientation must be cardinal")
    if orientation:
        inverse = (-orientation) % 360
        canonical_bounds = Bounds.around(Point(x,y).rotated(inverse)
            for x in (bounds.xmin,bounds.xmax) for y in (bounds.ymin,bounds.ymax))
        center, outline = winding_dot(at.rotated(inverse),other.rotated(inverse),canonical_bounds)
        return center.rotated(orientation), outline.rotated(orientation)
    radius = DEFAULT_METRICS.port_circle_radius * 0.5
    inset = DEFAULT_METRICS.terminal_stub / 2
    gap = DEFAULT_METRICS.label_clearance + radius
    if abs(at.y - other.y) <= COORDINATE_TOLERANCE:
        center = Point(at.x + (inset if other.x > at.x else -inset), bounds.ymin - gap)
    else:
        center = Point(bounds.xmax + gap, at.y + (inset if other.y > at.y else -inset))
    outline = Path(tuple(
        center.translated(radius * cos(2 * pi * index / 16), radius * sin(2 * pi * index / 16))
        for index in range(16)
    ), "coupling", True)
    return center, outline


def jump_arc(
    start: Point,
    end: Point,
    *,
    metrics: DiagramMetrics = DEFAULT_METRICS,
    correlation_key: str | None = None,
) -> JumpArc:
    """Emit one cubic Bezier wire-jump; the crossing wire is split around it."""

    if start.y == end.y:
        if not isclose(
            abs(end.x - start.x),
            metrics.jump_gap,
            rel_tol=0.0,
            abs_tol=COORDINATE_TOLERANCE,
        ):
            raise ValueError("horizontal jump endpoints must span jump_gap")
        sign = 1.0 if end.x > start.x else -1.0
        controls = (
            Point(start.x + sign * metrics.jump_gap / 3, start.y + metrics.jump_height),
            Point(end.x - sign * metrics.jump_gap / 3, end.y + metrics.jump_height),
        )
    elif start.x == end.x:
        if not isclose(
            abs(end.y - start.y),
            metrics.jump_gap,
            rel_tol=0.0,
            abs_tol=COORDINATE_TOLERANCE,
        ):
            raise ValueError("vertical jump endpoints must span jump_gap")
        sign = 1.0 if end.y > start.y else -1.0
        controls = (
            Point(start.x - metrics.jump_height, start.y + sign * metrics.jump_gap / 3),
            Point(end.x - metrics.jump_height, end.y - sign * metrics.jump_gap / 3),
        )
    else:
        raise ValueError("wire jumps are cardinal")
    return JumpArc(
        Path((start, controls[0], controls[1], end), "jump", False, (1, 4, 4, 4)),
        correlation_key,
    )


def split_wire_for_jump(
    start: Point,
    end: Point,
    jump_start: Point,
    jump_end: Point,
    *,
    correlation_key: str | None = None,
) -> tuple[ConductivePolyline, ConductivePolyline]:
    """Split the straight conductive wire so its jump span contains no ink."""

    if (
        start.y == end.y == jump_start.y == jump_end.y
        or start.x == end.x == jump_start.x == jump_end.x
    ):
        return (
            ConductivePolyline((start, jump_start), correlation_key),
            ConductivePolyline((jump_end, end), correlation_key),
        )
    raise ValueError("jump split points must lie on one cardinal wire")


def ground_mark(
    at: Point,
    *,
    side: Literal["left", "right", "top", "bottom"] = "bottom",
    metrics: DiagramMetrics = DEFAULT_METRICS,
    correlation_key: str | None = None,
) -> GuideMark:
    """Orient one local return glyph without changing its reference endpoint.

    The same native stem/bar construction serves structural returns and Port
    loads. Cardinal orientation is visible ink, not a second ground domain.
    """
    directions = {
        "left": Point(-1.0, 0.0),
        "right": Point(1.0, 0.0),
        "top": Point(0.0, 1.0),
        "bottom": Point(0.0, -1.0),
    }
    if side not in directions:
        raise ValueError("local ground side must be a cardinal direction")
    paths = _ground_glyph(at, directions[side], metrics=metrics)
    return GuideMark("ground", paths, None, (at,), correlation_key)


def _ground_glyph(
    at: Point, direction: Point, *, metrics: DiagramMetrics
) -> tuple[Path, ...]:
    """Return one local ground symbol whose stem begins at its actual return."""

    lateral = Point(-direction.y, direction.x)
    stem = metrics.ground_stem
    step = metrics.ground_bar_step
    half_width = metrics.ground_half_width
    stem_end = at.translated(direction.x * stem, direction.y * stem)

    def bar(distance: float, scale: float) -> Path:
        center = at.translated(direction.x * distance, direction.y * distance)
        return Path(
            (
                center.translated(
                    -lateral.x * half_width * scale,
                    -lateral.y * half_width * scale,
                ),
                center.translated(
                    lateral.x * half_width * scale,
                    lateral.y * half_width * scale,
                ),
            ),
            "symbol",
        )

    return (
        Path((at, stem_end), "symbol"),
        bar(stem, 1.0),
        bar(stem + step, 0.7),
        bar(stem + 2 * step, 0.2),
    )


def port_block(
    *,
    port_id: str,
    role: Literal["terminated", "nonloading_probe"],
    reference_impedance: str,
    boundary_anchor: Point,
    side: Literal["left", "right", "top", "bottom"],
    load_side: Literal["left", "right", "top", "bottom"] | None = None,
    metrics: DiagramMetrics = DEFAULT_METRICS,
    correlation_key: str | None = None,
) -> PortBlock:
    """Emit one fully measured boundary/load pose for a logical Port."""

    port_id, reference_impedance = _literal(port_id), _literal(reference_impedance)
    directions = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "top": (0.0, 1.0),
        "bottom": (0.0, -1.0),
    }
    if side not in directions:
        raise ValueError("Port side must be left, right, top, or bottom")
    dx, dy = directions[side]
    # Existing compiled lowering still asks only for the boundary side while
    # the authoring composition capture is introduced.  Retain precisely its
    # former deterministic transverse placement in that private bridge; all
    # final authoring poses supply ``load_side`` explicitly.
    if load_side is None:
        load_side = {
            "left": "bottom",
            "right": "bottom",
            "top": "left",
            "bottom": "right",
        }[side]
    if load_side not in directions:
        raise ValueError("Port load side must be left, right, top, or bottom")
    load_dx, load_dy = directions[load_side]
    if dx * load_dx + dy * load_dy != 0.0:
        raise ValueError("Port load side must be perpendicular to its boundary side")
    if role not in {"terminated", "nonloading_probe"}:
        raise ValueError("Port role must be terminated or nonloading_probe")
    radius = metrics.port_circle_radius
    # The Plan boundary meets the circle on its inward rim.  The circuit T is
    # on the interior side, so no conductive path crosses the open circle.
    center = boundary_anchor.translated(dx * radius, dy * radius)
    circuit = boundary_anchor.translated(
        -dx * metrics.port_lead, -dy * metrics.port_lead
    )
    normal = Point(load_dx, load_dy)
    load_anchor = circuit.translated(
        normal.x * metrics.terminal_stub,
        normal.y * metrics.terminal_stub,
    )
    ground_anchor = load_anchor.translated(
        normal.x * metrics.port_load_span,
        normal.y * metrics.port_load_span,
    )
    circle = Path(
        tuple(
            Point(
                center.x + radius * cos(2 * pi * index / 24),
                center.y + radius * sin(2 * pi * index / 24),
            )
            for index in range(25)
        ),
        "symbol",
        True,
    )
    label_side = {
        "left": "right",
        "right": "left",
        "top": "bottom",
        "bottom": "top",
    }[side]
    load = native_block(
        "R",
        start=load_anchor,
        end=ground_anchor,
        # A Port-owned shunt is identified by its real glyph and attachment;
        # one literal impedance label is cleaner than repeating a role formula
        # next to the same physical value.
        name=reference_impedance,
        value=None,
        label_side=label_side,
        metrics=metrics,
        correlation_key=correlation_key,
    )
    branch = Path((circuit, load_anchor), "wire")
    boundary = Path((boundary_anchor, circuit), "wire")
    inward_tip = circuit.translated(
        -dx * metrics.terminal_stub,
        -dy * metrics.terminal_stub,
    )
    inward_lead = Path((circuit, inward_tip), "wire")
    ground_glyph = _ground_glyph(ground_anchor, normal, metrics=metrics)

    # The Port's authored ID and its actual local load are sufficient clean
    # visible evidence for a terminated boundary.  A nonloading probe alone
    # needs an explicit removable marker because it intentionally lacks that
    # physical loading interpretation.
    outward_measures = [
        _text_centered(
            port_id,
            Point(0.0, 0.0),
            size=metrics.port_id_text_size,
            role="port-id",
        ),
    ]
    if role == "nonloading_probe":
        outward_measures.append(
            _text_centered(
                "PTC-removable",
                Point(0.0, 0.0),
                size=metrics.tertiary_text_size,
                role="port-ptc",
            )
        )
    stack_width = max(_extent(item, horizontal=True) for item in outward_measures)
    row_heights = tuple(_extent(item, horizontal=False) for item in outward_measures)
    stack_height = sum(row_heights) + metrics.label_clearance * (len(row_heights) - 1)
    if side == "left":
        stack_center = Point(
            circle.bounds.xmin - metrics.label_clearance - stack_width / 2,
            center.y,
        )
    elif side == "right":
        stack_center = Point(
            circle.bounds.xmax + metrics.label_clearance + stack_width / 2,
            center.y,
        )
    elif side == "top":
        stack_center = Point(
            center.x,
            circle.bounds.ymax + metrics.label_clearance + stack_height / 2,
        )
    else:
        stack_center = Point(
            center.x,
            circle.bounds.ymin - metrics.label_clearance - stack_height / 2,
        )
    row_cursor = stack_center.y + stack_height / 2
    outward_runs: list[TextRun] = []
    for measure, row_height in zip(outward_measures, row_heights, strict=True):
        outward_runs.append(
            _text_centered(
                measure.text,
                Point(stack_center.x, row_cursor - row_height / 2),
                size=measure.size,
                role=measure.role,
            )
        )
        row_cursor -= row_height + metrics.label_clearance

    by_role = {run.role: run for run in outward_runs}
    labels: tuple[TextRun, ...] = (by_role["port-id"],)
    if role == "nonloading_probe":
        labels += (by_role["port-ptc"],)
    actual_label_bounds = tuple(run.bounds for run in outward_runs)
    paths = (boundary, inward_lead, branch, *ground_glyph)
    occupied = _bounds(
        circle.bounds,
        load.occupied_bounds,
        *(path.bounds for path in paths),
        *actual_label_bounds,
    )
    return PortBlock(
        circle,
        center,
        boundary_anchor,
        circuit,
        inward_tip,
        load_anchor,
        ground_anchor,
        load,
        paths,
        ground_glyph,
        labels,
        occupied,
        correlation_key,
    )


# Explicit authoring shares the native templates above with compiled diagrams.
# These helpers move immutable measured facts; they never pack bodies or route.
def transform_fragment(scene, *, origin: Point, rotation: int = 0):
    """Rigidly adopt the same glyph/stroke geometry in a parent coordinate frame."""
    from dataclasses import fields, is_dataclass, replace
    from .scene import NeutralScene

    if rotation not in (0, 90, 180, 270):
        raise ValueError("fragment rotation must be cardinal")

    def move(value):
        if isinstance(value, Point):
            return value.rotated(rotation).translated(origin.x, origin.y)
        if isinstance(value, Bounds):
            return Bounds.around(move(Point(x, y))
                                for x in (value.xmin, value.xmax)
                                for y in (value.ymin, value.ymax))
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if is_dataclass(value):
            changes = {field.name: move(getattr(value, field.name))
                       for field in fields(value) if field.init}
            if isinstance(value, (NativeSymbol, ElectricalBox, TextRun)):
                changes["orientation"] = (value.orientation + rotation) % 360
            return replace(value, **changes)
        return value

    if not isinstance(scene, NeutralScene):
        raise TypeError("fragment adoption requires an immutable NeutralScene")
    return move(scene)


def fragment_text(scene):
    """Return stable structural paths to native text, without copied glyph data."""
    from dataclasses import fields, is_dataclass
    from types import MappingProxyType

    result = {}

    def visit(value, path):
        if isinstance(value, TextRun):
            result[path] = value
        elif isinstance(value, tuple):
            for index, item in enumerate(value):
                visit(item, (*path, index))
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name), (*path, field.name))

    visit(scene, ())
    return MappingProxyType(result)


def fragment_scene(*, symbols=(), boxes=(), ports=(), conductive=(), jumps=(),
                   guides=(), text=(), regions=(), boundary_sites=(), node_marks=()):
    """Enclose measured ink locally; this envelope is not a Drawing certificate."""
    from .scene import NeutralScene, SubsystemRegion

    bounds = [symbol.occupied_bounds for symbol in symbols]
    bounds += [box.bounds for box in boxes]
    bounds += [port.occupied_bounds for port in ports]
    bounds += [wire.bounds for wire in conductive]
    bounds += [jump.path.bounds for jump in jumps]
    bounds += [guide.bounds for guide in guides]
    bounds += [run.bounds for run in text]
    bounds += [region.bounds for region in regions]
    bounds += [site.visible_label.bounds for site in boundary_sites
               if site.visible_label is not None]
    bounds += [Bounds.around((site.point,)) for site in boundary_sites]
    bounds += [path.bounds for mark in node_marks for path in mark.paths]
    bounds += [mark.label.bounds for mark in node_marks if mark.label is not None]
    envelope = _bounds(*bounds) if bounds else Bounds(0.0, 0.0, 0.0, 0.0)
    root = SubsystemRegion(envelope, Path((Point(envelope.xmin, envelope.ymin),
        Point(envelope.xmax, envelope.ymin), Point(envelope.xmax, envelope.ymax),
        Point(envelope.xmin, envelope.ymax), Point(envelope.xmin, envelope.ymin)),
        "region", True), None, Point(envelope.xmin, envelope.ymax), "root")
    return NeutralScene(symbols=tuple(symbols), boxes=tuple(boxes), ports=tuple(ports),
        conductive=tuple(conductive), jumps=tuple(jumps), guides=tuple(guides),
        text=tuple(text), regions=(*tuple(regions), root),
        boundary_sites=tuple(boundary_sites), node_marks=tuple(node_marks))


def emit_native(measurement, *, pose, captions):
    """Translate the selected cardinal variant and apply explicit caption anchors.

    Caption keys are the stable paths returned by fragment_text; anchors are
    in the receiving scope. The selected variant already includes rotation.
    """
    from dataclasses import fields, is_dataclass, replace
    from .scene import NeutralScene

    if pose.rotation != measurement.orientation:
        raise ValueError("native pose must select its measured cardinal variant")
    scene = transform_fragment(measurement.scene, origin=Point(*pose.origin))
    known = fragment_text(scene)
    extra = set(captions) - set(known)
    if extra:
        raise ValueError(f"native caption sources are unknown: {extra!r}")

    def relocate(value, path):
        if isinstance(value, TextRun):
            if path not in captions:
                return value
            at = captions[path]
            delta = Point(at.x - value.origin.x, at.y - value.origin.y)
            # Translation preserves the measured glyphs and exact literal text.
            return transform_text(value, origin=delta)
        if isinstance(value, tuple):
            return tuple(relocate(item, (*path, index)) for index, item in enumerate(value))
        if is_dataclass(value):
            changes = {field.name: relocate(getattr(value, field.name), (*path, field.name))
                       for field in fields(value) if field.init}
            if isinstance(value, NativeSymbol):
                runs = [changes[name] for name in ("visible_name", "value", "branch_label")
                        if changes[name] is not None]
                changes["occupied_bounds"] = _bounds(value.symbol_bounds, *(run.bounds for run in runs))
            elif isinstance(value, ElectricalBox):
                runs = [changes[name] for name in ("title", "kind_label", "length_label",
                        "section_label", "reference_label") if changes[name] is not None]
                runs += [*changes["anchor_labels"], *changes["conductor_rows"]]
                changes["bounds"] = _bounds(value.outline.bounds,
                    *(stroke.bounds for stroke in value.paths), *(run.bounds for run in runs))
            elif isinstance(value, PortBlock):
                changes["occupied_bounds"] = _bounds(value.circle.bounds,
                    *(stroke.bounds for stroke in value.paths),
                    *(stroke.bounds for stroke in value.ground_glyph),
                    changes["reference_load"].occupied_bounds,
                    *(run.bounds for run in changes["labels"]))
            return replace(value, **changes)
        return value

    # Rebuild only the temporary envelope after moving text. Native body/stroke
    # coordinates are untouched; the enclosing scope supplies its final frame.
    payload = {field.name: relocate(getattr(scene, field.name), (field.name,))
               for field in fields(NeutralScene) if field.name not in
               {"regions", "bounds", "provenance_band"}}
    payload["regions"] = tuple(region for region in scene.regions if region.kind != "root")
    return fragment_scene(**payload)


def transform_text(run: TextRun, *, origin: Point):
    """Translate already shaped text without changing its font/metrics authority."""
    from dataclasses import replace
    return replace(run, origin=run.origin.translated(origin.x, origin.y),
        bounds=run.bounds.translated(origin.x, origin.y),
        glyphs=tuple(replace(glyph,
            vertices=tuple(point.translated(origin.x, origin.y) for point in glyph.vertices),
            bounds=glyph.bounds.translated(origin.x, origin.y)) for glyph in run.glyphs))


def measure_native(point, occurrence_ref, *, orientation, show_values, metrics):
    """Measure one captured physical target, without placement or routing.

    Cardinal variants select the native primitive before its upright labels are
    shaped. Contacts and incident-ink access are intrinsic to that same scene;
    the parent only translates this selected variant.
    """
    import json
    from ..schematic import DiagramRef, MeasuredFragment
    from ...authoring.physical_values import quantity_record

    if orientation not in (0, 90, 180, 270):
        raise ValueError("native orientation must be cardinal")
    source = json.loads(occurrence_ref.key)
    path = tuple(source["path"])
    direction = ("right", "top", "left", "bottom")[orientation // 90]
    origin = Point(0.0, 0.0)
    symbols, boxes, ports, guides, wires, node_marks = [], [], [], [], [], []
    anchors = {}
    identity = source
    markers = {}
    if occurrence_ref.kind == "occurrence":
        leaf = next(row for row in point.snapshot.semantic_record["physical_leaves"]
                    if tuple(row["path"]) == path)
        identity = leaf
        fields = {row["id"]: row for row in leaf["fields"]}

        def field_text(name):
            if not show_values:
                return None
            field = fields[name]
            return format_envelope(quantity_record(point.resolved_fields[path, name], field["unit"]))

        if leaf["model"] == "transmission_line":
            rlgc = point.resolved_fields[path, "rlgc"]
            metadata = leaf["model_metadata"]
            box = electrical_box(kind="CPW" if len(rlgc.conductors) == 1 else "MTL",
                origin=origin, orientation=orientation, title=path[-1],
                conductors=tuple(rlgc.conductors), reference=rlgc.reference_conductor,
                length=field_text("length"), n_sections=metadata.get("n_sections"),
                resolution=metadata.get("discretization"), metrics=metrics)
            boxes.append(box)
            anchors.update(box.anchors)
        else:
            kind, field = {"resistor": ("R", "resistance"),
                "capacitor": ("C", "capacitance"),
                "inductor": ("L", "inductance"),
                "josephson_junction": ("JJ", "josephson_inductance")}[leaf["model"]]
            end = Point(metrics.native_span, 0.0).rotated(orientation)
            symbol = native_block(kind, start=origin, end=end, name=path[-1],
                                  value=field_text(field), metrics=metrics)
            symbols.append(symbol)
            anchors.update(symbol.anchors)
            if leaf["model"] in {"inductor", "josephson_junction"}:
                for branch in leaf["oriented_branches"]:
                    if branch["value_field"] not in {"inductance", "josephson_inductance"}:
                        continue
                    sides = {}
                    for sign,pin,other in (("positive",branch["positive_pin"],branch["negative_pin"]),
                                           ("negative",branch["negative_pin"],branch["positive_pin"])):
                        center,stroke = winding_dot(anchors[pin],anchors[other],symbol.symbol_bounds,
                                                    orientation=symbol.orientation)
                        sides[sign] = {"point":center,"path":stroke}
                    markers[branch["id"]] = sides
            if leaf["model"] == "josephson_junction" and "junction_capacitance" in fields:
                offset = Point(0.0, -2 * metrics.native_span).rotated(orientation)
                first, last = offset, end.translated(offset.x, offset.y)
                symbols.append(native_block("C", start=first, end=last,
                    name=path[-1], value=field_text("junction_capacitance"),
                    branch_label="Cj", metrics=metrics))
                wires.extend((ConductivePolyline((origin, first)), ConductivePolyline((end, last))))
        contact_rows = [(pin, anchor, "left" if pin.startswith("head.") or pin == "terminal_1" else "right")
                        for pin, anchor in anchors.items()]
        contact_rows = [(pin, anchor, ("right", "top", "left", "bottom")[(
            ("right", "top", "left", "bottom").index(side) + orientation // 90) % 4])
                        for pin, anchor, side in contact_rows]
        refs = [(DiagramRef._create(occurrence_ref.preparation_sha256,
            json.dumps({"kind": "contact", "path": list(path), "contact_kind": "pin", "id": pin},
                       sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            "contact", occurrence_ref.scope_path), anchor, side)
                for pin, anchor, side in contact_rows]
    elif occurrence_ref.kind == "port":
        row = next(row for row in point.snapshot.semantic_record["connectivity"]["ports"]
                   if row["id"] == source["id"])
        identity = row
        port = port_block(port_id=row["id"], role=row["role"],
            reference_impedance=format_envelope(row["reference_impedance"]),
            boundary_anchor=origin, side=direction,
            load_side=("bottom", "right", "top", "left")[orientation // 90], metrics=metrics)
        ports.append(port)
        # The Port primitive owns this three-ray electrical T. Its filled dot
        # travels with the body, rather than requiring an authored hidden node.
        node_marks.append(junction_mark(port.circuit_anchor,metrics=metrics))
        refs = [(occurrence_ref, port.external_anchor,
                 ("left", "bottom", "right", "top")[orientation // 90])]
    elif occurrence_ref.kind == "ground":
        # Rotation selects a ground glyph extending along its declared side.
        guides.append(ground_mark(origin, side=direction, metrics=metrics))
        refs = [(occurrence_ref, origin, ("left", "bottom", "right", "top")[orientation // 90])]
    else:
        raise ValueError("native measurement requires occurrence, port, or ground ref")
    scene = fragment_scene(symbols=symbols, boxes=boxes, ports=ports,
                           guides=guides, conductive=wires, node_marks=node_marks)
    obstacles, access = measured_obstacles(scene, owner=occurrence_ref,
        scope=occurrence_ref.scope_path, contacts=refs, metrics=metrics)
    contacts = {ref: {"point": anchor, "side": side, "source": ref.key}
                for ref, anchor, side in refs}
    bodies = tuple(row.bounds for row in obstacles if row.kind != "text")
    texts = {key: run.bounds for key, run in fragment_text(scene).items()}
    return MeasuredFragment(occurrence_ref, identity, orientation,
        _bounds(*bodies), scene.bounds, texts, contacts, obstacles, access, markers, scene)


def measured_obstacles(scene, *, owner, scope, contacts, metrics):
    """Inventory actual ink and only its declared incident terminal sources."""
    from .composition_obstacles import OwnedObstacle, terminal_access

    rows, incident = [], {ref: [] for ref, _, _ in contacts}

    def add(key, bounds, kind, endpoints=()):
        source = (owner.key, *key)
        rows.append(OwnedObstacle((owner.key,), scope, kind, source, bounds))
        for ref, anchor, _ in contacts:
            if anchor in endpoints:
                incident[ref].append((source, bounds))

    for index, symbol in enumerate(scene.symbols):
        add(("symbols", index, "body"), symbol.symbol_bounds, "symbol_body",
            tuple(anchor for _, anchor in symbol.anchors))
        if symbol.reference_polarity is not None:
            add(("symbols", index, "polarity"), symbol.reference_polarity.bounds, "glyph")
    for index, box in enumerate(scene.boxes):
        add(("boxes", index, "outline"), box.outline.bounds, "box_body")
        for item, stroke in enumerate(box.paths):
            add(("boxes", index, "paths", item), stroke.bounds, "conductive",
                (stroke.points[0], stroke.points[-1]))
    for index, port in enumerate(scene.ports):
        add(("ports", index, "circle"), port.circle.bounds, "port_body")
        add(("ports", index, "load"), port.reference_load.symbol_bounds, "symbol_body")
        for item, stroke in enumerate(port.paths):
            add(("ports", index, "paths", item), stroke.bounds, "conductive",
                (stroke.points[0], stroke.points[-1]))
        for item, stroke in enumerate(port.ground_glyph):
            add(("ports", index, "ground", item), stroke.bounds, "glyph")
    for index, guide in enumerate(scene.guides):
        # A Ground's bars share its declared contact access: their clearance
        # envelopes can reach the lawful approach ray without touching ink.
        # The access consumer still clips permission to that terminal ray.
        ground_terminals = guide.terminals if owner.kind == "ground" and guide.kind == "ground" else ()
        for item, stroke in enumerate(guide.paths):
            add(("guides", index, "paths", item), stroke.bounds, "glyph",
                (stroke.points[0], stroke.points[-1], *ground_terminals))
    for index, wire in enumerate(scene.conductive):
        add(("conductive", index), wire.bounds, "conductive", (wire.points[0], wire.points[-1]))
    for index, mark in enumerate(scene.node_marks):
        for item, stroke in enumerate(mark.paths):
            add(("node_marks", index, "paths", item), stroke.bounds, "conductive")
    for key, run in fragment_text(scene).items():
        add(key, run.bounds, "text")
    access = []
    for ref, anchor, side in contacts:
        physical = incident[ref]
        body = _bounds(*(bounds for _, bounds in physical)) if physical else Bounds.around((anchor,))
        access.append(terminal_access(owner=(owner.key,), source=(ref.key,), side=side,
            point=anchor, body_bounds=body, obstacle_sources=tuple(key for key, _ in physical),
            stub=metrics.terminal_stub, clearance=metrics.obstacle_clearance))
    return tuple(rows), tuple(access)
