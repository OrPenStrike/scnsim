"""Native-symbol and transmission-line scene emission.

Schemdraw remains the trusted source for R/L/C/JJ/G glyph geometry.  We bind
labels to a real placed element before extracting its transformed anchors and
segments, then freeze equivalent literal text as exact Matplotlib glyph paths.
"""

from __future__ import annotations

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

NativeKind = Literal["R", "L", "C", "JJ", "G"]
_NATIVE_WHITELIST = frozenset({"R", "L", "C", "JJ", "G"})


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
    n_sections: int, minimum_axis: float, orientation: int,
    metrics: DiagramMetrics, correlation_key: str | None,
) -> ElectricalBox:
    """One-conductor line with upright measured facts and two physical leads."""
    labels = [("CPW", metrics.secondary_text_size, "line-kind"),
              (title, metrics.primary_text_size, "line-title")]
    if length is not None:
        labels.append((length, metrics.secondary_text_size, "line-length"))
    labels.append((f"{n_sections} section" + ("s" if n_sections != 1 else ""),
                   metrics.secondary_text_size, "line-sections"))
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
        correlation_key, section_label=placed["line-sections"],
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
    if kind == "CPW":
        if len(conductors) != 1:
            raise ValueError("CPW boxes require exactly one conductor")
        if isinstance(n_sections, bool) or not isinstance(n_sections, int) or n_sections <= 0:
            raise ValueError("CPW boxes require the captured positive section count")
        return _cpw_box(origin=origin, title=title, conductor=conductors[0], length=length,
                        n_sections=n_sections, minimum_axis=minimum_axis,
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

    header_measures = tuple(
        item
        for item in (kind_measure, title_measure, length_measure)
        if item is not None
    )
    header_width = sum(_extent(item, horizontal=True) for item in header_measures)
    header_width += gap * (len(header_measures) - 1) + 2 * padding
    header_height = max(_extent(item, horizontal=False) for item in header_measures)
    footer_width = (
        0.0
        if reference_measure is None
        else _extent(reference_measure, horizontal=True) + 2 * padding
    )
    footer_height = (
        0.0
        if reference_measure is None
        else _extent(reference_measure, horizontal=False)
    )
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
        body_width = max(minimum_axis, header_width, footer_width, *row_widths)
        content_height = (
            2 * padding
            + header_height
            + gap
            + sum(row_heights)
            + gap * (len(row_heights) - 1)
            + (gap + footer_height if reference_measure is not None else 0.0)
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
        endpoint_height = max(
            *(_extent(item, horizontal=False) for item in head_measures),
            *(_extent(item, horizontal=False) for item in tail_measures),
        )
        row_height = max(_extent(item, horizontal=False) for item in row_measures)
        body_width = max(
            metrics.native_span,
            header_width,
            footer_width,
            2 * padding + sum(column_widths) + gap * (len(column_widths) - 1),
        )
        content_height = (
            2 * padding
            + header_height
            + endpoint_height
            + row_height
            + (footer_height if reference_measure is not None else 0.0)
            + gap * (4 if reference_measure is not None else 3)
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
            Point((left + right) / 2, bottom + padding + footer_height / 2),
            size=reference_measure.size,
            role=reference_measure.role,
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
            reference_run.bounds.ymax + gap
            if reference_run is not None
            else bottom + padding
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
        packed_height = sum(row_heights) + gap * (len(row_heights) - 1)
        cursor = (available_top + available_bottom + packed_height) / 2
        row_centers: list[float] = []
        for row_height_value in row_heights:
            row_centers.append(cursor - row_height_value / 2)
            cursor -= row_height_value + gap
        if orientation == 180:
            row_centers.reverse()
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
        packed_width = sum(column_widths) + gap * (len(column_widths) - 1)
        cursor = (left + right - packed_width) / 2
        column_centers = [0.0] * len(column_widths)
        column_order = (
            range(len(column_widths))
            if orientation == 90
            else reversed(range(len(column_widths)))
        )
        for index in column_order:
            column_width = column_widths[index]
            column_centers[index] = cursor + column_width / 2
            cursor += column_width + gap

        footer_top = (
            reference_run.bounds.ymax if reference_run is not None else bottom + padding
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


def winding_dot(at: Point, other: Point, bounds: Bounds) -> tuple[Point, Path]:
    """Place a nonconductive winding dot beside one geometric terminal lead.

    Unlike an electrical junction dot, its center is off the wire. A dotted
    mutual leader terminates at this mark, not at a conductive graph node.
    """
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
