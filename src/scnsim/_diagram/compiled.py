"""Complete visible compiled matrix-stamp projection.

This projection consumes one immutable resolved point plus standalone compiler
audit evidence.  RLGC entries remain coefficients of disclosed matrix
stamps; no scalar-equivalent R/L/C/G network is invented.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .._authoring_snapshot import ResolvedPlanPoint
from .._canonical import (
    canonical_diagram_digests,
    canonical_expanded_graph_sha256,
    canonical_json_bytes,
    canonical_parameters_sha256,
    float64_from_hex,
)
from ..errors import SCNSimValidationError
from .metrics import DEFAULT_METRICS, DiagramMetrics, shape_text
from .native import conductive_wire, ground_mark, native_block, port_block
from .routing import WireDemand, route_demands
from .scene import (
    Bounds,
    ConductivePolyline,
    GuideMark,
    NativeSymbol,
    NeutralScene,
    NodeMark,
    Path,
    Point,
    SceneBuilder,
    SubsystemRegion,
)

_LINE = frozenset(
    {
        "series_resistance",
        "series_inductance",
        "shunt_capacitance_half",
        "shunt_conductance_half",
    }
)
_PRIMITIVE = frozenset(
    {
        "resistor",
        "capacitor",
        "inductor",
        "josephson_inductance",
        "junction_capacitance",
    }
)
_KNOWN = _LINE | _PRIMITIVE | {"transmission_line_audit", "mutual_inductance"}


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_layout", evidence=evidence)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail("compiled projection requires mapping evidence", field=field)
    return value


def _items(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail("compiled projection requires ordered evidence", field=field)
    return tuple(value)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail("compiled projection requires nonempty identity", field=field)
    return value


def _path(value: object) -> tuple[str, ...]:
    result = tuple(
        _text(item, "component_path") for item in _items(value, "component_path")
    )
    if not result:
        raise _fail("compiled component path is empty")
    return result


def _compact(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _quantity(value: object) -> tuple[float, str, str]:
    record = _mapping(value, "quantity")
    token, unit = record.get("si_value_f64"), record.get("si_unit")
    if not isinstance(token, str) or not isinstance(unit, str) or not unit:
        raise _fail("compiled quantity is malformed")
    try:
        return float64_from_hex(token), unit, token
    except Exception as error:
        raise _fail("compiled quantity has malformed Float64 evidence") from error


def _value(value: object) -> str:
    magnitude, unit, _ = _quantity(value)
    return f"{format(magnitude, '.17g')} {unit}"


def _qtoken(value: object | None) -> str:
    if value is None:
        return "null"
    _, unit, token = _quantity(value)
    return f"q={token}@{_compact(unit)}"


def _size(metrics: DiagramMetrics, points: float) -> float:
    return metrics.unit_length * points / 36.0


def _run(text: str, at: Point, role: str, metrics: DiagramMetrics, points: float):
    return shape_text(text, at=at, size=_size(metrics, points), role=role)


def _center(text: str, at: Point, role: str, metrics: DiagramMetrics, points: float):
    measured = _run(text, Point(0.0, at.y), role, metrics, points)
    return _run(
        text,
        Point(at.x - (measured.bounds.xmax - measured.bounds.xmin) / 2, at.y),
        role,
        metrics,
        points,
    )


def _append(
    builder: SceneBuilder,
    text: str,
    at: Point,
    role: str,
    metrics: DiagramMetrics,
    points: float,
):
    run = _run(text, at, role, metrics, points)
    builder.text.append(run)
    return run


def _tag(row: Mapping[str, object], ordinal: int) -> str:
    if row.get("kind") == "transmission_line_audit":
        return "LINE"
    if row.get("omitted_as_zero") is True:
        return "OMIT"
    if row.get("kind") == "mutual_inductance":
        return "MUTUAL"
    return "TERM"


def _tag_id(ordinal: int) -> str:
    return f"#{ordinal}"


def _term_row(ordinal: int, row: Mapping[str, object]) -> str:
    """Compact independently parseable actual-row evidence, not hidden metadata."""

    kind = _text(row.get("kind"), "kind")
    path = _compact(row.get("component_path")) if "component_path" in row else "null"
    if kind == "transmission_line_audit":
        fields = (
            ("path", path),
            ("conductors", _compact(row.get("conductors"))),
            ("reference", _compact(row.get("reference_conductor"))),
            ("n_sections", _compact(row.get("n_sections"))),
            ("length", _qtoken(row.get("length"))),
            ("dx", _qtoken(row.get("dx"))),
            ("orientation", _compact(row.get("orientation"))),
            ("stations", _compact(row.get("stations"))),
            ("rlgc_source", _compact(row.get("rlgc_source"))),
        )
    elif kind == "mutual_inductance":
        fields = (
            ("coupling_id", _compact(row.get("coupling_id"))),
            ("branch_a", _compact(row.get("branch_a"))),
            ("branch_b", _compact(row.get("branch_b"))),
            ("k", _qtoken(row.get("coupling_coefficient"))),
            ("M", _qtoken(row.get("derived_mutual_inductance"))),
            ("omitted_as_zero", _compact(row.get("omitted_as_zero"))),
        )
    else:
        fields = (
            ("kind", _compact(kind)),
            ("path", path),
            ("section", _compact(row.get("section"))),
            ("station", _compact(row.get("station"))),
            ("end", _compact(row.get("end"))),
            ("row_conductor", _compact(row.get("row_conductor"))),
            ("column_conductor", _compact(row.get("column_conductor"))),
            ("branch_id", _compact(row.get("branch_id"))),
            ("value", _qtoken(row.get("value"))),
            ("omitted_as_zero", _compact(row.get("omitted_as_zero"))),
            ("terminal_1_to_terminal_2", _compact(row.get("terminal_1_to_terminal_2"))),
            ("incidence", _compact(row.get("incidence_f64"))),
            ("Bplus", _compact(row.get("physical_positive_incidence_f64"))),
            ("Bminus", _compact(row.get("physical_negative_incidence_f64"))),
        )
    return (
        "SCNSIM-COMPILED-V1 "
        + _tag_id(ordinal)
        + " "
        + _tag(row, ordinal)
        + " "
        + " ".join(f"{key}={value}" for key, value in fields)
    )


def _node_mark(
    builder: SceneBuilder,
    node: str,
    point: Point,
    metrics: DiagramMetrics,
    *,
    port_occupied: Sequence[Bounds],
) -> None:
    """Place the node identity clear of the actual Port ink at its boundary.

    A Port's role label is part of its occupied native block, not metadata.
    Choose a deterministic alternate node-label side when the historical
    above-node placement would overlap it.  This changes no anchor or graph
    fact; it only keeps two independently emitted visible identities legible.
    """

    radius = metrics.terminal_stub / 2
    paths = (
        Path(
            (
                Point(point.x - radius, point.y),
                Point(point.x, point.y + radius),
                Point(point.x + radius, point.y),
                Point(point.x, point.y - radius),
                Point(point.x - radius, point.y),
            ),
            "symbol",
            True,
        ),
    )
    candidates = [
        Point(point.x, point.y + metrics.terminal_stub),
        Point(point.x, point.y - 2 * metrics.terminal_stub),
        Point(point.x - 2 * metrics.native_span, point.y),
        Point(point.x + 2 * metrics.native_span, point.y),
    ]
    measured = _center(node, Point(0.0, 0.0), "compiled-node", metrics, 8.0)
    for bounds in port_occupied:
        candidates.extend(
            (
                Point(
                    point.x,
                    bounds.ymin - metrics.label_clearance - measured.bounds.ymax,
                ),
                Point(
                    point.x,
                    bounds.ymax + metrics.label_clearance - measured.bounds.ymin,
                ),
                Point(
                    bounds.xmin - metrics.label_clearance - measured.bounds.xmax,
                    point.y,
                ),
                Point(
                    bounds.xmax + metrics.label_clearance - measured.bounds.xmin,
                    point.y,
                ),
            )
        )
    candidates.sort(
        key=lambda at: (abs(at.x - point.x) + abs(at.y - point.y), at.x, at.y)
    )
    label = next(
        (
            candidate
            for candidate in (
                _center(node, at, "compiled-node", metrics, 8.0) for at in candidates
            )
            if not any(
                candidate.bounds.overlaps(occupied, clearance=metrics.label_clearance)
                for occupied in port_occupied
            )
        ),
        None,
    )
    if label is None:
        raise _fail(
            "compiled node identity has no clear Port-label placement", node=node
        )
    builder.node_marks.append(
        NodeMark(point, label, paths, True, f"compiled-node:{node}")
    )


def _node_positions(
    nodes: tuple[str, ...], metrics: DiagramMetrics
) -> dict[str, Point]:
    cursor = 0.0
    positions: dict[str, Point] = {}
    for node in nodes:
        label = _run(node, Point(0.0, 0.0), "measure-node", metrics, 8.0)
        width = label.bounds.xmax - label.bounds.xmin
        cursor += width / 2 + metrics.native_span
        positions[node] = Point(cursor, 0.0)
        cursor += width / 2 + metrics.native_span
    return positions


def _endpoint(row: Mapping[str, object], field: str, nodes: tuple[str, ...]) -> str:
    vector = _items(row.get(field), field)
    if len(vector) != len(nodes):
        raise _fail(
            "compiled primitive incidence does not match node order", field=field
        )
    selected: list[str] = []
    for node, token in zip(nodes, vector, strict=True):
        if not isinstance(token, str):
            raise _fail("compiled primitive incidence token is malformed", field=field)
        value = float64_from_hex(token)
        if value != 0.0:
            if abs(value) != 1.0:
                raise _fail(
                    "compiled primitive incidence is not an endpoint selector",
                    field=field,
                )
            selected.append(node)
    if not selected:
        return "ground"
    if len(selected) != 1:
        raise _fail("compiled primitive incidence selects multiple nodes", field=field)
    return selected[0]


@dataclass(frozen=True, slots=True)
class _Line:
    audit_ordinal: int
    audit: Mapping[str, object]
    path: tuple[str, ...]
    conductors: tuple[str, ...]
    sections: int
    stations: Mapping[tuple[int, str], Mapping[str, object]]


def _lines(rows: tuple[Mapping[str, object], ...]) -> tuple[_Line, ...]:
    result: list[_Line] = []
    for ordinal, audit in enumerate(rows):
        if audit.get("kind") != "transmission_line_audit":
            continue
        path = _path(audit.get("component_path"))
        conductors = tuple(
            _text(item, "conductors")
            for item in _items(audit.get("conductors"), "conductors")
        )
        sections = audit.get("n_sections")
        if (
            not conductors
            or len(set(conductors)) != len(conductors)
            or isinstance(sections, bool)
            or not isinstance(sections, int)
            or sections < 1
        ):
            raise _fail("compiled line audit is malformed", component_path=path)
        expected = {
            (station, conductor)
            for station in range(sections + 1)
            for conductor in conductors
        }
        stations: dict[tuple[int, str], Mapping[str, object]] = {}
        for record in (
            _mapping(item, "stations")
            for item in _items(audit.get("stations"), "stations")
        ):
            station, conductor = record.get("station"), record.get("conductor")
            if (
                isinstance(station, bool)
                or not isinstance(station, int)
                or not isinstance(conductor, str)
                or (station, conductor) not in expected
                or (station, conductor) in stations
            ):
                raise _fail("compiled station record is malformed", component_path=path)
            _text(record.get("compiled_node_id"), "compiled_node_id")
            stations[(station, conductor)] = record
        if set(stations) != expected:
            raise _fail(
                "compiled line lacks an ordered station record", component_path=path
            )
        result.append(_Line(ordinal, audit, path, conductors, sections, stations))
    if len({line.path for line in result}) != len(result):
        raise _fail("compiled line audit repeats a component path")
    return tuple(sorted(result, key=lambda item: item.path))


def _matrix_rows(
    rows: tuple[Mapping[str, object], ...], line: _Line
) -> dict[tuple[object, ...], tuple[int, Mapping[str, object]]]:
    found: dict[tuple[object, ...], tuple[int, Mapping[str, object]]] = {}
    for ordinal, row in enumerate(rows):
        if (
            tuple(row.get("component_path", ())) != line.path
            or row.get("kind") not in _LINE
        ):
            continue
        kind = _text(row.get("kind"), "kind")
        section, left, right = (
            row.get("section"),
            row.get("row_conductor"),
            row.get("column_conductor"),
        )
        if (
            isinstance(section, bool)
            or not isinstance(section, int)
            or section not in range(1, line.sections + 1)
            or left not in line.conductors
            or right not in line.conductors
        ):
            raise _fail(
                "compiled matrix entry is outside its ordered line basis",
                component_path=line.path,
            )
        station, end = row.get("station"), row.get("end")
        if kind.startswith("series_"):
            if station is not None or end is not None:
                raise _fail(
                    "series matrix entry has shunt attachment fields",
                    component_path=line.path,
                )
        elif (
            not isinstance(station, int)
            or station not in {section - 1, section}
            or end not in {"left", "right"}
        ):
            raise _fail(
                "half-shunt matrix entry lacks exact station/end evidence",
                component_path=line.path,
            )
        key = (kind, section, station, end, left, right)
        if key in found:
            raise _fail("compiled matrix entry is duplicated", component_path=line.path)
        found[key] = (ordinal, row)
    expected = 6 * line.sections * len(line.conductors) ** 2
    if len(found) != expected:
        raise _fail(
            "compiled matrix entries are incomplete",
            component_path=line.path,
            expected=expected,
            actual=len(found),
        )
    return found


def _compound(
    builder: SceneBuilder,
    glyphs: list[NativeSymbol],
    label: str,
    key: str,
    metrics: DiagramMetrics,
    *,
    fallback: Bounds,
) -> Bounds:
    # A complete matrix stamp can have every diagonal exactly zero while an
    # off-diagonal coefficient remains nonzero.  Its bracket is still a real
    # visible matrix region; a missing diagonal glyph must not erase it.
    occupied = [glyph.occupied_bounds for glyph in glyphs] or [fallback]
    bounds = Bounds(
        min(item.xmin for item in occupied) - metrics.native_span,
        min(item.ymin for item in occupied) - metrics.native_span,
        max(item.xmax for item in occupied) + metrics.native_span,
        max(item.ymax for item in occupied) + metrics.native_span,
    )
    outline = Path(
        (
            Point(bounds.xmin, bounds.ymin),
            Point(bounds.xmax, bounds.ymin),
            Point(bounds.xmax, bounds.ymax),
            Point(bounds.xmin, bounds.ymax),
            Point(bounds.xmin, bounds.ymin),
        ),
        "guide",
        True,
    )
    builder.guides.append(GuideMark("bracket", (outline,), None, (), key))
    builder.text.append(
        _run(
            label,
            Point(bounds.xmin, bounds.ymax + metrics.label_clearance),
            "compiled-matrix-compound",
            metrics,
            8.0,
        )
    )
    return bounds


def _offdiagonal_guides(
    builder: SceneBuilder,
    *,
    kind: str,
    entries: Mapping[tuple[object, ...], tuple[int, Mapping[str, object]]],
    section: int,
    station: int | None,
    end: str | None,
    conductors: tuple[str, ...],
    bounds: Bounds,
    metrics: DiagramMetrics,
) -> None:
    """Draw nonconductive, signed matrix-entry links for every off-diagonal.

    These strokes are deliberately coupling marks, never conductive paths:
    off-diagonal R/L/C/G entries are compiler coefficients, not invented
    scalar-equivalent two-terminal components.
    """

    visible = 0
    for left_index, left in enumerate(conductors):
        for right in conductors[left_index + 1 :]:
            forward_ordinal, forward = entries[
                (kind, section, station, end, left, right)
            ]
            reverse_ordinal, reverse = entries[
                (kind, section, station, end, right, left)
            ]
            forward_zero, reverse_zero = (
                forward.get("omitted_as_zero") is True,
                reverse.get("omitted_as_zero") is True,
            )
            if forward_zero and reverse_zero:
                continue
            if forward_zero != reverse_zero:
                raise _fail(
                    "symmetric off-diagonal matrix pair has inconsistent zero evidence",
                    kind=kind,
                    section=section,
                    left=left,
                    right=right,
                )
            _, _, forward_value = _quantity(forward["value"])
            _, _, reverse_value = _quantity(reverse["value"])
            y = bounds.ymin - metrics.label_clearance * (2 + visible * 3)
            path = Path((Point(bounds.xmin, y), Point(bounds.xmax, y)), "coupling")
            prefix = (
                "L offdiagonal coupling"
                if kind == "series_inductance"
                else f"{kind} offdiagonal matrix coefficient"
            )
            label = (
                f"{prefix} {_tag_id(forward_ordinal)} signed_f64={forward_value} "
                f"{_tag_id(reverse_ordinal)} signed_f64={reverse_value} orientation={left}→{right} "
                "nonconductive matrix stamp"
            )
            builder.guides.append(
                GuideMark(
                    "coupling",
                    (path,),
                    _center(
                        label,
                        Point(
                            (bounds.xmin + bounds.xmax) / 2, y - metrics.terminal_stub
                        ),
                        "compiled-offdiagonal",
                        metrics,
                        7.0,
                    ),
                    (),
                    f"compiled-offdiagonal:{kind}:{section}:{station}:{end}:{left}:{right}",
                )
            )
            visible += 1


def _series_block(
    builder: SceneBuilder,
    line: _Line,
    section: int,
    rows: Mapping[tuple[object, ...], tuple[int, Mapping[str, object]]],
    origin: Point,
    terms: dict[str, list[Point]],
    metrics: DiagramMetrics,
    show_values: bool,
) -> float:
    glyphs: list[NativeSymbol] = []
    panel_points: list[Point] = []
    row_pitch = 3 * metrics.native_span
    track_x = origin.x + metrics.native_span
    for index, conductor in enumerate(line.conductors):
        y = origin.y - index * row_pitch
        x0 = track_x
        r_ord, r = rows[
            ("series_resistance", section, None, None, conductor, conductor)
        ]
        l_ord, l = rows[
            ("series_inductance", section, None, None, conductor, conductor)
        ]
        left = _text(
            line.stations[(section - 1, conductor)].get("compiled_node_id"),
            "compiled_node_id",
        )
        right = _text(
            line.stations[(section, conductor)].get("compiled_node_id"),
            "compiled_node_id",
        )
        left_tip = Point(x0, y)
        entries = ((r_ord, r, "R"), (l_ord, l, "L"))

        def placed(
            shift: float, *, left_tip=left_tip, entries=entries, y=y
        ) -> tuple[
            Point, list[tuple[int, Mapping[str, object], NativeSymbol, Point, Point]]
        ]:
            cursor = left_tip
            result: list[
                tuple[int, Mapping[str, object], NativeSymbol, Point, Point]
            ] = []
            for ordinal, row, native in entries:
                if row.get("omitted_as_zero") is True:
                    continue
                start = Point(
                    cursor.x + metrics.terminal_stub + (shift if not result else 0.0), y
                )
                finish = Point(start.x + metrics.native_span, y)
                glyph = native_block(
                    native,
                    start=start,
                    end=finish,
                    name=f"{native} coefficient",
                    value=_value(row["value"]) if show_values else None,
                    branch_label=_tag_id(ordinal),
                    label_side="top",
                    metrics=metrics,
                )
                result.append((ordinal, row, glyph, cursor, start))
                cursor = finish
            return cursor, result

        cursor, built = placed(0.0)
        if built:
            left_clearance = min(
                glyph.occupied_bounds.xmin for _, _, glyph, _, _ in built
            )
            shift = max(0.0, left_tip.x + metrics.label_clearance - left_clearance)
            if shift:
                cursor, built = placed(shift)
        right_tip = Point(
            max(
                x0 + 6 * metrics.native_span,
                cursor.x + metrics.native_span,
                *(
                    glyph.occupied_bounds.xmax + metrics.label_clearance
                    for _, _, glyph, _, _ in built
                ),
            ),
            y,
        )
        panel_points.extend((left_tip, right_tip))
        terms[left].append(left_tip)
        terms[right].append(right_tip)
        # Each incidence label belongs to its exposed terminal.  Putting both
        # labels outside their respective tips keeps long compiled identities
        # from crossing in the middle of one matrix row.
        left_text = f"B_s + s{section - 1}.{conductor} → {left}"
        right_text = f"B_s − s{section}.{conductor} → {right}"
        left_measure = _run(
            left_text,
            Point(0.0, y + metrics.terminal_stub),
            "compiled-Bs",
            metrics,
            7.0,
        )
        terminal_labels = (
            _run(
                left_text,
                Point(
                    left_tip.x
                    - metrics.label_clearance
                    - (left_measure.bounds.xmax - left_measure.bounds.xmin),
                    y + metrics.terminal_stub,
                ),
                "compiled-Bs",
                metrics,
                7.0,
            ),
            _run(
                right_text,
                Point(right_tip.x + metrics.label_clearance, y + metrics.terminal_stub),
                "compiled-Bs",
                metrics,
                7.0,
            ),
        )
        builder.text.extend(terminal_labels)
        conductor_glyphs: list[NativeSymbol] = []
        for ordinal, _, glyph, lead_start, start in built:
            builder.conductive.append(conductive_wire(lead_start, start))
            builder.symbols.append(glyph)
            glyphs.append(glyph)
            conductor_glyphs.append(glyph)
        builder.conductive.append(conductive_wire(cursor, right_tip))
        # Track widths come from the actual frozen B labels and native symbol
        # occupied boxes.  No string-length proxy may decide matrix columns.
        track_x = max(
            right_tip.x + metrics.panel_gap,
            *(label.bounds.xmax + metrics.panel_gap for label in terminal_labels),
            *(
                glyph.occupied_bounds.xmax + metrics.panel_gap
                for glyph in conductor_glyphs
            ),
        )
    panel = Bounds.around(panel_points)
    bounds = _compound(
        builder,
        glyphs,
        f"MATRIX B_s(R+sL)^-1B_sT · {'.'.join(line.path)} · section={section} · diagonal glyphs are coefficients, never scalar branches",
        f"compiled-series:{'.'.join(line.path)}:{section}",
        metrics,
        fallback=panel,
    )
    _offdiagonal_guides(
        builder,
        kind="series_resistance",
        entries=rows,
        section=section,
        station=None,
        end=None,
        conductors=line.conductors,
        bounds=bounds,
        metrics=metrics,
    )
    _offdiagonal_guides(
        builder,
        kind="series_inductance",
        entries=rows,
        section=section,
        station=None,
        end=None,
        conductors=line.conductors,
        bounds=bounds,
        metrics=metrics,
    )
    return bounds.ymin - metrics.panel_gap


def _shunt_block(
    builder: SceneBuilder,
    line: _Line,
    section: int,
    end: Literal["left", "right"],
    rows: Mapping[tuple[object, ...], tuple[int, Mapping[str, object]]],
    origin: Point,
    terms: dict[str, list[Point]],
    metrics: DiagramMetrics,
    show_values: bool,
) -> float:
    station = section - 1 if end == "left" else section
    glyphs: list[NativeSymbol] = []
    panel_points: list[Point] = []
    track_x = origin.x + metrics.native_span
    for index, conductor in enumerate(line.conductors):
        y = origin.y - index * 4 * metrics.native_span
        x0 = track_x
        node = _text(
            line.stations[(station, conductor)].get("compiled_node_id"),
            "compiled_node_id",
        )
        tip = Point(x0, y)
        terms[node].append(tip)
        panel_points.extend(
            (tip, Point(x0 + 4 * metrics.native_span, y - metrics.native_span))
        )
        terminal_label = _center(
            f"B_t s{station}.{conductor} → {node}",
            Point(tip.x, y + metrics.terminal_stub),
            "compiled-Bt",
            metrics,
            7.0,
        )
        builder.text.append(terminal_label)
        conductor_glyphs: list[NativeSymbol] = []
        for offset, kind, key in (
            (2 * metrics.native_span, "C", "shunt_capacitance_half"),
            (4 * metrics.native_span, "G", "shunt_conductance_half"),
        ):
            ordinal, row = rows[(key, section, station, end, conductor, conductor)]
            if row.get("omitted_as_zero") is True:
                continue
            x = x0 + offset
            top, bottom = Point(x, y), Point(x, y - metrics.native_span)
            builder.conductive.append(conductive_wire(tip, top))
            glyph = native_block(
                kind,
                start=bottom,
                end=top,
                name=f"{kind} coefficient",
                value=_value(row["value"]) if show_values else None,
                branch_label=_tag_id(ordinal),
                label_side="right",
                metrics=metrics,
            )
            builder.symbols.append(glyph)
            glyphs.append(glyph)
            conductor_glyphs.append(glyph)
            builder.guides.append(
                ground_mark(
                    bottom,
                    metrics=metrics,
                    correlation_key=f"compiled-half-ground:{ordinal}",
                )
            )
        track_x = max(
            x0 + 4 * metrics.native_span + metrics.panel_gap,
            terminal_label.bounds.xmax + metrics.panel_gap,
            *(
                glyph.occupied_bounds.xmax + metrics.panel_gap
                for glyph in conductor_glyphs
            ),
        )
    bounds = _compound(
        builder,
        glyphs,
        f"MATRIX B_t(G/2+sC/2)B_tT · {'.'.join(line.path)} · section={section} end={end} · half-stamp coefficient glyphs",
        f"compiled-shunt:{'.'.join(line.path)}:{section}:{end}",
        metrics,
        fallback=Bounds.around(panel_points),
    )
    _offdiagonal_guides(
        builder,
        kind="shunt_capacitance_half",
        entries=rows,
        section=section,
        station=station,
        end=end,
        conductors=line.conductors,
        bounds=bounds,
        metrics=metrics,
    )
    _offdiagonal_guides(
        builder,
        kind="shunt_conductance_half",
        entries=rows,
        section=section,
        station=station,
        end=end,
        conductors=line.conductors,
        bounds=bounds,
        metrics=metrics,
    )
    return bounds.ymin - metrics.panel_gap


def _line_blocks(
    builder: SceneBuilder,
    line: _Line,
    rows: tuple[Mapping[str, object], ...],
    origin: Point,
    terms: dict[str, list[Point]],
    metrics: DiagramMetrics,
    show_values: bool,
) -> float:
    matrix = _matrix_rows(rows, line)
    label = f"COMPILED PI {'.'.join(line.path)} · ordered conductors={','.join(line.conductors)} · reference={_text(line.audit.get('reference_conductor'), 'reference_conductor')} · sections={line.sections}"
    title = _append(builder, label, origin, "compiled-line-title", metrics, 9.0)
    y = title.bounds.ymin - metrics.native_span
    for section in range(1, line.sections + 1):
        y = _series_block(
            builder,
            line,
            section,
            matrix,
            Point(origin.x, y),
            terms,
            metrics,
            show_values,
        )
        y = _shunt_block(
            builder,
            line,
            section,
            "left",
            matrix,
            Point(origin.x, y),
            terms,
            metrics,
            show_values,
        )
        y = _shunt_block(
            builder,
            line,
            section,
            "right",
            matrix,
            Point(origin.x, y),
            terms,
            metrics,
            show_values,
        )
    return y


def _primitives(
    builder: SceneBuilder,
    rows: tuple[Mapping[str, object], ...],
    nodes: tuple[str, ...],
    origin: Point,
    terms: dict[str, list[Point]],
    metrics: DiagramMetrics,
    show_values: bool,
) -> float:
    y = origin.y
    active = [
        (ordinal, row)
        for ordinal, row in enumerate(rows)
        if row.get("kind") in _PRIMITIVE and row.get("omitted_as_zero") is not True
    ]
    if active:
        title = _append(
            builder,
            "COMPILED PHYSICAL PRIMITIVES",
            Point(origin.x, y),
            "compiled-primitives",
            metrics,
            9.0,
        )
        y = title.bounds.ymin - metrics.native_span
    for ordinal, row in active:
        kind = _text(row.get("kind"), "kind")
        positive, negative = (
            _endpoint(row, "physical_positive_incidence_f64", nodes),
            _endpoint(row, "physical_negative_incidence_f64", nodes),
        )
        native: Literal["R", "L", "C", "JJ", "G"] = {
            "resistor": "R",
            "capacitor": "C",
            "inductor": "L",
            "josephson_inductance": "JJ",
            "junction_capacitance": "C",
        }[kind]
        start, finish = (
            Point(origin.x + metrics.native_span, y),
            Point(origin.x + 2 * metrics.native_span, y),
        )
        symbol = native_block(
            native,
            start=start,
            end=finish,
            name=f"{'.'.join(_path(row.get('component_path')))}:{kind}",
            value=_value(row["value"]) if show_values else None,
            branch_label=_tag_id(ordinal),
            metrics=metrics,
        )
        builder.symbols.append(symbol)
        if positive == "ground":
            builder.guides.append(
                ground_mark(
                    start,
                    metrics=metrics,
                    correlation_key=f"compiled-ground:{ordinal}:positive",
                )
            )
        else:
            terms[positive].append(start)
        if negative == "ground":
            builder.guides.append(
                ground_mark(
                    finish,
                    metrics=metrics,
                    correlation_key=f"compiled-ground:{ordinal}:negative",
                )
            )
        else:
            terms[negative].append(finish)
        y = min(y - metrics.panel_gap, symbol.occupied_bounds.ymin - metrics.panel_gap)
    return y


def _ports(
    builder: SceneBuilder,
    audit_ports: Mapping[str, object],
    semantic_ports: object,
    positions: Mapping[str, Point],
    metrics: DiagramMetrics,
    show_values: bool,
) -> None:
    """Emit snapshot-owned physical Ports and bind them to audit node IDs."""
    ids = tuple(
        _text(item, "ports.ids") for item in _items(audit_ports.get("ids"), "ports.ids")
    )
    rows = tuple(
        _mapping(item, "semantic.connectivity.ports")
        for item in _items(semantic_ports, "semantic.connectivity.ports")
    )
    if len(set(ids)) != len(ids) or {
        _text(row.get("id"), "port.id") for row in rows
    } != set(ids):
        raise _fail("compiler audit Port ids disagree with captured resolved point")
    seen: set[str] = set()
    by_id = {_text(row.get("id"), "port.id"): row for row in rows}
    for index, port_id in enumerate(ids):
        row = by_id[port_id]
        node, role = _text(row.get("net"), "port.net"), row.get("role")
        if (
            port_id in seen
            or node not in positions
            or role not in {"terminated", "nonloading_probe"}
        ):
            raise _fail("captured resolved-point Port is malformed", port_id=port_id)
        seen.add(port_id)
        # A Port's load is an actual raw reference impedance, not a matrix
        # coefficient.  It consequently remains legible even when the
        # coefficient value labels are suppressed; the review ledger does not
        # substitute for this physical Port fact.
        block = port_block(
            port_id=port_id,
            role=role,
            reference_impedance=_value(row["reference_impedance"]),
            boundary_anchor=positions[node],
            side="top" if index % 2 == 0 else "bottom",
            metrics=metrics,
        )
        builder.ports.append(block)


def _escape_to_exterior(
    anchor: Point, other: Point, occupied: Bounds, metrics: DiagramMetrics
) -> Point:
    """Return the registered orthogonal stub outside one occupied native box."""

    dx, dy = anchor.x - other.x, anchor.y - other.y
    if dx and dy:
        raise _fail(
            "native terminal anchors are not cardinal", anchor=anchor, other=other
        )
    if dx > 0:
        return Point(occupied.xmax + metrics.obstacle_clearance, anchor.y)
    if dx < 0:
        return Point(occupied.xmin - metrics.obstacle_clearance, anchor.y)
    if dy > 0:
        return Point(anchor.x, occupied.ymax + metrics.obstacle_clearance)
    if dy < 0:
        return Point(anchor.x, occupied.ymin - metrics.obstacle_clearance)
    raise _fail("native terminal anchors coincide", anchor=anchor)


def _escape_toward(
    anchor: Point, other: Point, occupied: Bounds, metrics: DiagramMetrics
) -> Point:
    """Extend a registered Port route along its emitted inward circuit lead."""

    dx, dy = other.x - anchor.x, other.y - anchor.y
    if dx and dy:
        raise _fail(
            "native terminal anchors are not cardinal", anchor=anchor, other=other
        )
    if dx > 0:
        return Point(occupied.xmax + metrics.obstacle_clearance, anchor.y)
    if dx < 0:
        return Point(occupied.xmin - metrics.obstacle_clearance, anchor.y)
    if dy > 0:
        return Point(anchor.x, occupied.ymax + metrics.obstacle_clearance)
    if dy < 0:
        return Point(anchor.x, occupied.ymin - metrics.obstacle_clearance)
    raise _fail("native terminal anchors coincide", anchor=anchor)


def _registered_escapes(
    builder: SceneBuilder, terminals: tuple[Point, ...], metrics: DiagramMetrics
) -> tuple[tuple[Point, Point], ...]:
    """Bind every routed terminal inside a native/Port box to a visible stub.

    The router is intentionally unaware of component ownership.  This layout
    step derives the only allowed escape from each emitted anchor, preserving
    all occupied boxes as obstacles rather than silently piercing them.
    """

    escapes: dict[Point, Point] = {}
    terminal_set = set(terminals)
    for mark in builder.node_marks:
        if mark.point in terminal_set:
            # A node's nearby caption is reserved ink too.  The bottom ray
            # leaves the diamond without entering the caption above it.
            escapes[mark.point] = Point(
                mark.point.x,
                mark.point.y - metrics.terminal_stub - metrics.obstacle_clearance,
            )
    for symbol in builder.symbols:
        first, second = (point for _, point in symbol.anchors)
        if first in terminal_set:
            escapes[first] = _escape_to_exterior(
                first, second, symbol.occupied_bounds, metrics
            )
        if second in terminal_set:
            escapes[second] = _escape_to_exterior(
                second, first, symbol.occupied_bounds, metrics
            )
    for port in builder.ports:
        anchor = port.boundary_anchor
        if anchor not in terminal_set:
            continue
        # The port circle is deliberately on the exterior side of its Plan
        # boundary.  Its inward circuit lead is already visible in port_block:
        # use that exact owned corridor to leave the complete Port ink, rather
        # than cutting outward through the Port's role caption.
        escapes[anchor] = _escape_toward(
            anchor, port.circuit_anchor, port.occupied_bounds, metrics
        )
    for wire in builder.conductive:
        # Each B_s/B_t tip is the exposed end of a pre-existing local matrix
        # lead.  Register its outward stub so the generic router starts in a
        # free lane rather than trying to grow through the coefficient box.
        # The lead itself is passed via ``existing`` below and remains a real
        # stroke, not an obstacle or a hidden net substitution.
        first, second = wire.points[0], wire.points[1]
        if first in terminal_set:
            escapes[first] = _escape_to_exterior(first, second, wire.bounds, metrics)
        last, before = wire.points[-1], wire.points[-2]
        if last in terminal_set:
            escapes[last] = _escape_to_exterior(last, before, wire.bounds, metrics)
    return tuple(
        sorted(
            escapes.items(),
            key=lambda item: (item[0].x, item[0].y, item[1].x, item[1].y),
        )
    )


def _existing_strokes(
    builder: SceneBuilder, terms: Mapping[str, list[Point]]
) -> tuple[tuple[str, ConductivePolyline], ...]:
    """Assign transient router ownership to already-emitted matrix strokes.

    Ownership exists only while routing.  The frozen scene retains exactly the
    returned visible geometry, never this private mapping.  An exposed B
    terminal determines a stroke's node; an entirely internal coefficient
    lead is kept distinct so it cannot become an invented global connection.
    """

    owners_by_point: dict[Point, set[str]] = defaultdict(set)
    for node, points in terms.items():
        for point in points:
            owners_by_point[point].add(node)
    result: list[tuple[str, ConductivePolyline]] = []
    for index, wire in enumerate(builder.conductive):
        owners = owners_by_point[wire.points[0]] | owners_by_point[wire.points[-1]]
        if len(owners) > 1:
            raise _fail(
                "one pre-existing matrix stroke joins distinct compiled nodes",
                wire=index,
                nodes=sorted(owners),
            )
        owner = next(iter(owners), f"\x00compiled-internal:{index}")
        result.append((owner, wire))
    return tuple(result)


def _route(
    builder: SceneBuilder,
    positions: Mapping[str, Point],
    terms: Mapping[str, list[Point]],
    metrics: DiagramMetrics,
) -> None:
    # Native/Port occupied bounds already include their bound labels.  Every
    # other text run and guide caption exists before global routing and is
    # actual visible ink, including the projection/section headings.  Reserve
    # it all rather than allowing a foreign net to cross a readable fact.  The
    # review ledger is appended only after routing, outside this field.
    obstacles = (
        tuple(symbol.occupied_bounds for symbol in builder.symbols)
        + tuple(port.occupied_bounds for port in builder.ports)
        + tuple(run.bounds for run in builder.text)
        + tuple(
            mark.label.bounds for mark in builder.node_marks if mark.label is not None
        )
        + tuple(
            guide.label.bounds for guide in builder.guides if guide.label is not None
        )
    )
    pad = 4 * metrics.panel_gap + metrics.obstacle_clearance
    demands: list[WireDemand] = []
    for node, anchor in positions.items():
        terminals = tuple(dict.fromkeys((anchor, *terms.get(node, ()))))
        if len(terminals) < 2:
            continue
        # The routing domain is local to the node's actual expanded
        # incidences and their measured nearby ink.  This does not introduce
        # a presentation rail or clip the full compiler evidence table.
        terminal_extent = Bounds.around(terminals)
        relevant = tuple(
            bound
            for bound in obstacles
            if bound.overlaps(terminal_extent, clearance=pad)
        )
        local_extent = Bounds.around(
            (
                *terminals,
                *(Point(bound.xmin, bound.ymin) for bound in relevant),
                *(Point(bound.xmax, bound.ymax) for bound in relevant),
            )
        )
        local_allowed = Bounds(
            local_extent.xmin - pad,
            local_extent.ymin - pad,
            local_extent.xmax + pad,
            local_extent.ymax + pad,
        )
        demands.append(
            WireDemand(
                node,
                (),
                terminals,
                local_allowed,
                escapes=_registered_escapes(builder, terminals, metrics),
            )
        )
    try:
        routed = route_demands(
            demands,
            obstacles=obstacles,
            metrics=metrics,
            reverse_order=False,
            existing=_existing_strokes(builder, terms),
        )
    except SCNSimValidationError as forward_error:
        try:
            routed = route_demands(
                demands,
                obstacles=obstacles,
                metrics=metrics,
                reverse_order=True,
                existing=_existing_strokes(builder, terms),
            )
        except SCNSimValidationError as reverse_error:
            raise _fail(
                "finite compiled-node router found no valid normal or reversed routing candidate",
                forward=str(forward_error),
                reverse=str(reverse_error),
            ) from reverse_error
    builder.conductive[:] = routed.conductive
    builder.jumps[:] = routed.jumps


def _ledger(
    builder: SceneBuilder,
    rows: tuple[Mapping[str, object], ...],
    origin: Point,
    nodes: tuple[str, ...],
    metrics: DiagramMetrics,
) -> None:
    title = _append(
        builder,
        "COMPILED REVIEW TABLE — coefficient glyphs are matrix terms; each compact row preserves exact compiler evidence",
        origin,
        "compiled-ledger-title",
        metrics,
        9.0,
    )
    y = title.bounds.ymin - metrics.native_span
    node_order = f"SCNSIM-COMPILED-V1 NODE_ORDER {_compact(list(nodes))}"
    for literal in (
        node_order,
        "TABLE COLUMNS: #ordinal TAG fixed canonical fields; q=f64hex@canonical-si-unit; Bplus/Bminus are complete ordered incidence vectors",
    ):
        run = _append(
            builder, literal, Point(origin.x, y), "compiled-ledger-legend", metrics, 7.0
        )
        y = run.bounds.ymin - metrics.label_clearance - _size(metrics, 7.0)
    for ordinal, row in enumerate(rows):
        guide_start = len(builder.guides)
        run = _append(
            builder,
            _term_row(ordinal, row),
            Point(origin.x, y),
            "compiled-ledger-row",
            metrics,
            7.0,
        )
        if _tag(row, ordinal) == "OMIT":
            line = Path(
                (
                    Point(run.bounds.xmin, run.bounds.ymin - metrics.label_clearance),
                    Point(run.bounds.xmax, run.bounds.ymin - metrics.label_clearance),
                ),
                "guide",
            )
            builder.guides.append(
                GuideMark(
                    "omission",
                    (line,),
                    _center(
                        _tag_id(ordinal),
                        Point(
                            (line.points[0].x + line.points[-1].x) / 2,
                            line.points[0].y - metrics.label_clearance,
                        ),
                        "compiled-omission",
                        metrics,
                        7.0,
                    ),
                    (),
                    f"compiled-omission:{ordinal}",
                )
            )
        elif _tag(row, ordinal) == "MUTUAL":
            line = Path(
                (
                    Point(run.bounds.xmin, run.bounds.ymin - metrics.label_clearance),
                    Point(run.bounds.xmax, run.bounds.ymin - metrics.label_clearance),
                ),
                "coupling",
            )
            builder.guides.append(
                GuideMark(
                    "coupling",
                    (line,),
                    _center(
                        _tag_id(ordinal),
                        Point(
                            (line.points[0].x + line.points[-1].x) / 2,
                            line.points[0].y - metrics.label_clearance,
                        ),
                        "compiled-mutual",
                        metrics,
                        7.0,
                    ),
                    (),
                    f"compiled-mutual:{ordinal}",
                )
            )
        row_floor = min(
            run.bounds.ymin,
            *(guide.bounds.ymin for guide in builder.guides[guide_start:]),
        ) if len(builder.guides) > guide_start else run.bounds.ymin
        y = row_floor - metrics.label_clearance - _size(metrics, 7.0)


def _finish(
    builder: SceneBuilder, plan_id: str, plan_sha: str, metrics: DiagramMetrics
) -> None:
    bounds = [
        *(symbol.occupied_bounds for symbol in builder.symbols),
        *(port.occupied_bounds for port in builder.ports),
        *(run.bounds for run in builder.text),
        *(mark.label.bounds for mark in builder.node_marks if mark.label is not None),
        *(path.bounds for mark in builder.node_marks for path in mark.paths),
        *(guide.bounds for guide in builder.guides),
        *(wire.bounds for wire in builder.conductive),
    ]
    if not bounds:
        raise _fail("compiled projection emitted no visible facts")
    pad = metrics.panel_gap
    outer = Bounds(
        min(item.xmin for item in bounds) - pad,
        min(item.ymin for item in bounds) - pad,
        max(item.xmax for item in bounds) + pad,
        max(item.ymax for item in bounds) + pad,
    )
    path = Path(
        (
            Point(outer.xmin, outer.ymin),
            Point(outer.xmax, outer.ymin),
            Point(outer.xmax, outer.ymax),
            Point(outer.xmin, outer.ymax),
            Point(outer.xmin, outer.ymin),
        ),
        "region",
        True,
    )
    builder.regions.append(
        SubsystemRegion(
            outer,
            path,
            None,
            Point(outer.xmin, outer.ymax),
            "root",
            f"compiled-root:{plan_id}:{plan_sha}",
        )
    )


def layout_compiled(
    point: ResolvedPlanPoint,
    compiled: Mapping[str, object],
    *,
    show_values: bool = True,
) -> NeutralScene:
    """Emit a complete routed compiled scene without rerunning the compiler."""
    if not isinstance(show_values, bool):
        raise TypeError("show_values must be boolean")
    if not isinstance(point, ResolvedPlanPoint):
        raise TypeError("layout_compiled requires ResolvedPlanPoint")
    if not isinstance(compiled, Mapping):
        raise TypeError("layout_compiled requires compiler-audit mapping")
    required = {
        "schema",
        "schema_version",
        "plan_sha256",
        "parameters_sha256",
        "node_order",
        "matrix_order",
        "resolved_bindings",
        "expanded_branch_rows",
        "c_matrix",
        "k_matrix",
        "g_matrix",
        "ports",
        "compiled_graph_sha256",
        "expanded_graph_sha256",
    }
    if (
        set(compiled) != required
        or compiled.get("schema") != "scnsim.compiler_audit"
        or compiled.get("schema_version") != 2
        or compiled.get("matrix_order") != "canonical_node_id"
    ):
        raise _fail("compiled schematic requires the complete compiler-audit v2 frame")
    digests = canonical_diagram_digests(point.snapshot, representation="compiled")
    plan_sha = digests["plan_sha256"]
    parameters_sha = canonical_parameters_sha256(point.parameter_record)
    if (
        compiled.get("plan_sha256") != plan_sha
        or compiled.get("parameters_sha256") != parameters_sha
    ):
        raise _fail("compiler audit binds a different captured resolved point")
    semantic = getattr(point.snapshot, "semantic_record", None)
    if not isinstance(semantic, Mapping):
        raise _fail("captured resolved point has no semantic record")
    plan_id = _text(semantic.get("plan_id"), "plan_id")
    nodes = tuple(
        _text(item, "node_order")
        for item in _items(compiled.get("node_order"), "node_order")
    )
    if not nodes or len(set(nodes)) != len(nodes):
        raise _fail("compiled node order is empty or duplicated")
    rows = tuple(
        _mapping(item, "expanded_branch_rows")
        for item in _items(compiled.get("expanded_branch_rows"), "expanded_branch_rows")
    )
    bindings = tuple(
        _mapping(item, "resolved_bindings")
        for item in _items(compiled.get("resolved_bindings"), "resolved_bindings")
    )
    if compiled.get("expanded_graph_sha256") != canonical_expanded_graph_sha256(
        plan_sha256=plan_sha,
        node_order=nodes,
        resolved_bindings=bindings,
        expanded_branch_rows=rows,
    ):
        raise _fail("compiler audit expanded graph identity is inconsistent")
    if not isinstance(compiled.get("compiled_graph_sha256"), str):
        raise _fail("compiler audit has no compiled graph identity")
    unknown = sorted({_text(row.get("kind"), "kind") for row in rows} - _KNOWN)
    if unknown:
        raise _fail("compiled row kind lacks visual grammar", row_kinds=unknown)
    lines = _lines(rows)
    if {line.path for line in lines} != {
        _path(row.get("component_path")) for row in rows if row.get("kind") in _LINE
    }:
        raise _fail("compiled line audits and matrix terms disagree")
    metrics, builder = DEFAULT_METRICS, SceneBuilder()
    connectivity = _mapping(semantic.get("connectivity"), "semantic.connectivity")
    port_nets = {row["net"] for row in _items(connectivity.get("ports"), "ports")}
    positions = _node_positions(
        tuple(node for node in nodes if node in port_nets), metrics
    )
    _ports(
        builder,
        _mapping(compiled.get("ports"), "ports"),
        connectivity.get("ports"),
        positions,
        metrics,
        show_values,
    )
    _append(
        builder,
        f"COMPILED MATRIX-STAMP PROJECTION · CircuitPlan {plan_id} · plan {plan_sha}",
        Point(0.0, metrics.native_span),
        "compiled-title",
        metrics,
        9.0,
    )
    terms: dict[str, list[Point]] = defaultdict(list)
    y = -2 * metrics.panel_gap
    for line in lines:
        y = _line_blocks(
            builder, line, rows, Point(0.0, y), terms, metrics, show_values
        )
    y = _primitives(builder, rows, nodes, Point(0.0, y), terms, metrics, show_values)
    # A compiled station is a local expanded incidence, not a demand to run
    # every internal node to a remote presentation rail.  External Port nodes
    # keep their native boundaries; other diamonds sit at the first measured
    # incidence and their exact IDs remain visible and independently audited.
    for node in nodes:
        if node not in positions:
            if not terms.get(node):
                raise _fail(
                    "compiled node has no visible expanded incidence", node=node
                )
            positions[node] = terms[node][0]
        _node_mark(
            builder,
            node,
            positions[node],
            metrics,
            port_occupied=(
                *(symbol.occupied_bounds for symbol in builder.symbols),
                *(port.occupied_bounds for port in builder.ports),
                *(run.bounds for run in builder.text),
                *(
                    mark.label.bounds
                    for mark in builder.node_marks
                    if mark.label is not None
                ),
                *(
                    guide.label.bounds
                    for guide in builder.guides
                    if guide.label is not None
                ),
            ),
        )
    _route(builder, positions, terms, metrics)
    floor = (
        min((wire.bounds.ymin for wire in builder.conductive), default=y)
        - 2 * metrics.panel_gap
    )
    _ledger(builder, rows, Point(0.0, floor), nodes, metrics)
    _finish(builder, plan_id, plan_sha, metrics)
    return builder.freeze()


__all__ = ["layout_compiled"]
