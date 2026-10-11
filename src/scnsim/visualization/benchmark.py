"""Pure HTML presentation of current Workspace operation diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
import json
from typing import Any

from ..benchmark.models import BenchmarkResult
from ..numeric_encoding import record_document
from .plots.common import _plotly, _style, report_palette
from .presentation import Theme, _report_html


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"operation record {label} must be an object")
    return value


def _required_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"operation record {label} must be an array")
    return value


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    return value


def _native_compact(value: object) -> str:
    return json.dumps(
        _json_value(value), ensure_ascii=False, sort_keys=False, separators=(",", ":")
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{escape(label)}</th>" for label in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _details(title: str, value: object) -> str:
    return (
        f"<details><summary>{escape(title)}</summary>"
        f"<pre style=\"white-space:pre-wrap;overflow-wrap:anywhere\">"
        f"{escape(json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, indent=2))}</pre></details>"
    )


def _recorded(value: object) -> object:
    return "not recorded" if value is None else value


@dataclass(frozen=True, slots=True)
class _TimelineOperation:
    operation_id: str
    method: str | None
    operation: str | None
    root_kind: str
    parent_span_id: str | None
    backend: str | None
    precision: str | None
    status: str
    request_sha256: str | None
    span_id: str
    clock: Mapping[str, object] | None
    start_ns: int | None
    end_ns: int | None
    details: object
    numerical_refs: object


@dataclass(frozen=True, slots=True)
class _TimelineSpan:
    operation_id: str
    span_id: str
    parent_span_id: str | None
    kind: str
    clock: Mapping[str, object] | None
    start_ns: int | None
    end_ns: int | None
    status: str
    details: object
    numerical_refs: object
    root: bool = False
    depth: int = 0


def _timeline_clock_id(clock: Mapping[str, object] | None) -> str | None:
    value = None if clock is None else clock.get("id")
    return value if isinstance(value, str) else None


def _timeline_clock_label(clock: Mapping[str, object] | None) -> str:
    if clock is None:
        return "Clock unavailable; alignment unknown"
    return "; ".join((
        f"id={_recorded(clock.get('id'))}",
        f"source={_recorded(clock.get('source'))}",
        f"unit={_recorded(clock.get('unit'))}",
        f"origin_ns={_recorded(clock.get('monotonic_origin_ns'))}",
    ))


def _timeline_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"operation timeline {label} must be an integer or null")
    return value


def _timeline_operation_label(operation: _TimelineOperation) -> str:
    if operation.operation is not None:
        return operation.operation
    return "request kind unavailable before preparation"


def _timeline_operations(document: Mapping[str, object]) -> tuple[
    list[_TimelineOperation], list[_TimelineSpan]
]:
    """Normalize the verified operation projection for presentation only."""

    raw_operations = _required_sequence(document["operations"], "operations")
    raw_spans = _required_sequence(document["spans"], "spans")
    operations: list[_TimelineOperation] = []
    spans: list[_TimelineSpan] = []
    operations_by_id: dict[str, _TimelineOperation] = {}

    for index, raw_value in enumerate(raw_operations):
        raw = _required_mapping(raw_value, f"operations[{index}]")
        operation_id = raw["operation_id"]
        method = raw["method"]
        operation_kind = raw["operation"]
        root_kind = raw["kind"]
        parent_span_id = raw["parent_span_id"]
        backend = raw["backend"]
        precision = raw["precision"]
        status = raw["status"]
        request_sha256 = raw["request_sha256"]
        span_id = raw["span_id"]
        clock_value = raw["clock"]
        if not isinstance(operation_id, str) or not isinstance(span_id, str):
            raise TypeError(f"operations[{index}] identity fields must be strings")
        if operation_kind is not None and not isinstance(operation_kind, str):
            raise TypeError(f"operations[{index}].operation must be a string or null")
        if not isinstance(root_kind, str) or not isinstance(status, str):
            raise TypeError(f"operations[{index}] kind and status must be strings")
        if parent_span_id is not None and not isinstance(parent_span_id, str):
            raise TypeError(f"operations[{index}].parent_span_id must be a string or null")
        if method is not None and not isinstance(method, str):
            raise TypeError(f"operations[{index}].method must be a string or null")
        if backend is not None and not isinstance(backend, str):
            raise TypeError(f"operations[{index}].backend must be a string or null")
        if precision is not None and not isinstance(precision, str):
            raise TypeError(f"operations[{index}].precision must be a string or null")
        if request_sha256 is not None and not isinstance(request_sha256, str):
            raise TypeError(f"operations[{index}].request_sha256 must be a string or null")
        clock = (
            _required_mapping(clock_value, f"operations[{index}].clock")
            if clock_value is not None else None
        )
        operation = _TimelineOperation(
            operation_id=operation_id,
            method=method,
            operation=operation_kind,
            root_kind=root_kind,
            parent_span_id=parent_span_id,
            backend=backend,
            precision=precision,
            status=status,
            request_sha256=request_sha256,
            span_id=span_id,
            clock=clock,
            start_ns=_timeline_integer(raw["start_ns"], f"operations[{index}].start_ns"),
            end_ns=_timeline_integer(raw["end_ns"], f"operations[{index}].end_ns"),
            details=raw["details"],
            numerical_refs=raw["numerical_refs"],
        )
        if operation_id in operations_by_id:
            raise ValueError(f"operation timeline repeats operation id {operation_id!r}")
        operations.append(operation)
        operations_by_id[operation_id] = operation
        spans.append(_TimelineSpan(
            operation_id=operation_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            kind=root_kind,
            clock=clock,
            start_ns=operation.start_ns,
            end_ns=operation.end_ns,
            status=status,
            details=operation.details,
            numerical_refs=operation.numerical_refs,
            root=True,
        ))

    for index, raw_value in enumerate(raw_spans):
        raw = _required_mapping(raw_value, f"spans[{index}]")
        operation_id = raw["operation_id"]
        span_id = raw["span_id"]
        parent_span_id = raw["parent_span_id"]
        kind = raw["kind"]
        status = raw["status"]
        clock_value = raw["clock"]
        if not isinstance(operation_id, str) or operation_id not in operations_by_id:
            raise ValueError(f"spans[{index}] references an unknown operation")
        if not isinstance(span_id, str) or not isinstance(kind, str) or not isinstance(status, str):
            raise TypeError(f"spans[{index}] identity, kind, and status must be strings")
        if parent_span_id is not None and not isinstance(parent_span_id, str):
            raise TypeError(f"spans[{index}].parent_span_id must be a string or null")
        clock = (
            _required_mapping(clock_value, f"spans[{index}].clock")
            if clock_value is not None else None
        )
        spans.append(_TimelineSpan(
            operation_id=operation_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            kind=kind,
            clock=clock,
            start_ns=_timeline_integer(raw["start_ns"], f"spans[{index}].start_ns"),
            end_ns=_timeline_integer(raw["end_ns"], f"spans[{index}].end_ns"),
            status=status,
            details=raw["details"],
            numerical_refs=raw["numerical_refs"],
        ))

    spans_by_id: dict[tuple[str, str], _TimelineSpan] = {}
    for row in spans:
        key = (row.operation_id, row.span_id)
        if key in spans_by_id:
            raise ValueError(f"operation timeline repeats span id {row.span_id!r}")
        spans_by_id[key] = row

    def depth_of(row: _TimelineSpan) -> int:
        depth = 1
        parent_id = row.parent_span_id
        visited = {row.span_id}
        while parent_id is not None:
            parent = spans_by_id.get((row.operation_id, parent_id))
            if parent is None:
                break
            if parent.span_id in visited:
                raise ValueError(f"operation timeline contains a parent cycle at {parent.span_id!r}")
            if parent.root:
                return depth
            visited.add(parent.span_id)
            depth += 1
            parent_id = parent.parent_span_id
        return depth

    normalized: list[_TimelineSpan] = []
    for row in spans:
        if row.root:
            normalized.append(row)
        else:
            normalized.append(_TimelineSpan(
                operation_id=row.operation_id,
                span_id=row.span_id,
                parent_span_id=row.parent_span_id,
                kind=row.kind,
                clock=row.clock,
                start_ns=row.start_ns,
                end_ns=row.end_ns,
                status=row.status,
                details=row.details,
                numerical_refs=row.numerical_refs,
                depth=depth_of(row),
            ))
    return operations, normalized


def _timeline_filter_values(operations: Sequence[_TimelineOperation], key: str) -> tuple[str, ...]:
    values = {
        str(getattr(operation, key))
        for operation in operations
        if getattr(operation, key) is not None
    }
    if any(getattr(operation, key) is None for operation in operations):
        values.add("__scnsim_missing__")
    return tuple(sorted(values))


def _timeline_short_id(value: str) -> str:
    return value if len(value) <= 8 else value[:8]


def _timeline_row_key(operation_id: str, span_id: str) -> str:
    return json.dumps((operation_id, span_id), ensure_ascii=False, separators=(",", ":"))


def _timeline_filters(operations: Sequence[_TimelineOperation]) -> str:
    fields = (
        ("operation_id", "Operation"),
        ("method", "Method"),
        ("operation", "Operation kind"),
        ("backend", "Backend"),
        ("precision", "Precision"),
        ("status", "Status"),
    )
    controls: list[str] = []
    for field, label in fields:
        values = (
            tuple(sorted({operation.operation_id for operation in operations}))
            if field == "operation_id" else _timeline_filter_values(operations, field)
        )
        options = "".join(
            f'<option value="{escape(value, quote=True)}">'
            f'{escape("(not recorded)" if value == "__scnsim_missing__" else value)}</option>'
            for value in values
        )
        controls.append(
            f'<label class="scnsim-filter">{escape(label)}'
            f'<select multiple size="{min(max(len(values), 2), 5)}" data-filter="{field}">{options}</select>'
            "</label>"
        )
    return (
        '<fieldset class="scnsim-timeline-filters"><legend>Timeline filters</legend>'
        + "".join(controls)
        + '<label class="scnsim-filter scnsim-toggle"><input type="checkbox" '
          'data-toggle-details> Show nested detail spans</label>'
        + '<span class="scnsim-filter-count" data-visible-count></span>'
        + '<p>Select one or more values; an empty selection includes all recorded values.</p>'
        + "</fieldset>"
    )


def _timeline_groups(
    operations: Sequence[_TimelineOperation],
    spans: Sequence[_TimelineSpan],
) -> tuple[dict[tuple[str, str | None], list[_TimelineSpan]], dict[str, _TimelineOperation]]:
    operation_by_id = {operation.operation_id: operation for operation in operations}
    groups: dict[tuple[str, str | None], list[_TimelineSpan]] = {}
    for span in spans:
        groups.setdefault((span.operation_id, _timeline_clock_id(span.clock)), []).append(span)
    for rows in groups.values():
        rows.sort(key=lambda row: (0 if row.root else row.depth, row.span_id))
    return groups, operation_by_id


def _timeline_root_summaries(
    operations: Sequence[_TimelineOperation],
) -> tuple[str, list[tuple[object, ...]]]:
    """Summarize root durations and same-clock wall envelopes only."""

    completed = [
        operation for operation in operations
        if operation.status != "running"
        and operation.start_ns is not None
        and operation.end_ns is not None
    ]
    incomplete = len(operations) - len(completed)
    cumulative_ns = sum(
        operation.end_ns - operation.start_ns
        for operation in completed
        if operation.start_ns is not None and operation.end_ns is not None
    )
    cumulative_label = (
        f"{cumulative_ns} ns ({cumulative_ns / 1e9:.9g} s) across "
        f"{len(completed)} closed root operation(s)"
    )
    if incomplete:
        cumulative_label += f"; {incomplete} root duration(s) remain unknown"
    if not operations:
        cumulative_label = "unavailable; no operation roots were recorded"

    by_clock: dict[str, list[_TimelineOperation]] = {}
    unaligned_roots = 0
    for operation in operations:
        clock_id = _timeline_clock_id(operation.clock)
        if clock_id is None:
            unaligned_roots += 1
            continue
        by_clock.setdefault(clock_id, []).append(operation)

    rows: list[tuple[object, ...]] = []
    for clock_id, clock_operations in by_clock.items():
        bounded = [
            operation for operation in clock_operations
            if operation.status != "running"
            and operation.start_ns is not None
            and operation.end_ns is not None
        ]
        unknown = len(clock_operations) - len(bounded)
        if unknown:
            wall_label = f"unknown; {unknown} root endpoint(s) unavailable or still open"
        else:
            start_ns = min(operation.start_ns for operation in bounded if operation.start_ns is not None)
            end_ns = max(operation.end_ns for operation in bounded if operation.end_ns is not None)
            wall_ns = end_ns - start_ns
            wall_label = f"{wall_ns} ns ({wall_ns / 1e9:.9g} s)"
        clock_record = next(
            operation.clock for operation in clock_operations if operation.clock is not None
        )
        rows.append((
            clock_id,
            len(clock_operations),
            wall_label,
            _timeline_clock_label(clock_record),
        ))

    if unaligned_roots:
        rows.append((
            "clock unavailable",
            unaligned_roots,
            "unknown; roots cannot be placed in a recorded clock domain",
            "clock identity unavailable",
        ))
    if len(by_clock) > 1:
        rows.append((
            "cross-domain",
            len(operations),
            "unavailable; clock origins are not aligned",
            "no cross-clock wall span recorded",
        ))
    if not rows:
        rows.append(("unavailable", 0, "no recorded operation roots", "clock unavailable"))
    return cumulative_label, rows


_GENERATION_PERFORMANCE_KEYS = (
    "generation",
    "population_size",
    "population_evaluation_wall_ns",
    "average_candidate_wall_ns",
    "new_unique_candidates",
    "cache_hit_occurrences",
    "same_generation_duplicate_occurrences",
    "numerical_failure_occurrences",
    "worker_capacity",
    "active_worker_count",
)


def _generation_performance_section(
    operations: Sequence[_TimelineOperation],
    generation_records: Sequence[tuple[str, Mapping[str, object]]],
) -> str | None:
    """Present only generation summaries released after the numerical commit ACK."""

    if not generation_records:
        if any(operation.operation == "optimize_direct" for operation in operations):
            return (
                "<h2>Generation population performance</h2>"
                "<p>No acknowledged complete-generation summary was recorded. "
                "Partial or uncommitted population timing remains unreported; "
                "no missing generation is filled with zero.</p>"
            )
        return None

    operation_by_id = {operation.operation_id: operation for operation in operations}
    operation_order = {
        operation.operation_id: index
        for index, operation in enumerate(operations, 1)
    }
    summaries_by_operation: dict[str, list[Mapping[str, object]]] = {}
    for operation_id, summary in generation_records:
        if operation_id not in operation_by_id:
            raise ValueError("generation timing summary references an unknown operation")
        summaries_by_operation.setdefault(operation_id, []).append(summary)

    table_rows: list[tuple[object, ...]] = []
    weighted_rows: list[tuple[object, ...]] = []
    plot_traces: list[tuple[str, list[int], list[float], list[str], bool]] = []
    all_summaries: list[dict[str, object]] = []

    for operation_id, summaries in summaries_by_operation.items():
        run_label = f"Run {operation_order[operation_id]}"
        ordered: list[tuple[int | None, int | None, int | None, Mapping[str, object]]] = []
        for summary in summaries:
            for key in _GENERATION_PERFORMANCE_KEYS:
                if key not in summary:
                    raise KeyError(f"generation summary is missing required field {key!r}")
            generation = _timeline_integer(summary["generation"], "generation summary generation")
            population = _timeline_integer(summary["population_size"], "generation summary population_size")
            wall = _timeline_integer(
                summary["population_evaluation_wall_ns"],
                "generation summary population_evaluation_wall_ns",
            )
            average = _timeline_integer(
                summary["average_candidate_wall_ns"],
                "generation summary average_candidate_wall_ns",
            )
            extras = {
                str(key): value for key, value in summary.items()
                if key not in _GENERATION_PERFORMANCE_KEYS
            }
            table_rows.append((
                run_label,
                _recorded(generation),
                _recorded(population),
                f"{wall} ns ({wall / 1e6:.9g} ms)" if wall is not None else "not recorded",
                f"{average} ns ({average / 1e6:.9g} ms)" if average is not None else "not recorded",
                _recorded(summary["new_unique_candidates"]),
                _recorded(summary["cache_hit_occurrences"]),
                _recorded(summary["same_generation_duplicate_occurrences"]),
                _recorded(summary["numerical_failure_occurrences"]),
                _recorded(summary["worker_capacity"]),
                _recorded(summary["active_worker_count"]),
                _native_compact(extras) if extras else "not recorded",
            ))
            ordered.append((generation, population, wall, summary))
            all_summaries.append({
                "operation_id": operation_id,
                "run": run_label,
                "summary": dict(summary),
            })

        ordered.sort(key=lambda item: (item[0] is None, item[0] or 0))
        total_wall_ns = 0
        total_population = 0
        complete_denominator_count = 0
        x_values: list[int] = []
        y_values: list[float] = []
        hover: list[str] = []
        segments: list[tuple[list[int], list[float], list[str]]] = []
        previous_generation: int | None = None

        def finish_segment() -> None:
            nonlocal x_values, y_values, hover
            if x_values:
                segments.append((x_values, y_values, hover))
            x_values, y_values, hover = [], [], []

        for generation, population, wall, summary in ordered:
            average = summary["average_candidate_wall_ns"]
            if generation is None or not isinstance(average, int) or isinstance(average, bool):
                finish_segment()
                previous_generation = None
            else:
                if previous_generation is not None and generation != previous_generation + 1:
                    finish_segment()
                x_values.append(generation)
                y_values.append(average / 1e6)
                hover.append("<br>".join((
                    f"{escape(run_label)} · generation {generation}",
                    f"population size: {escape(str(_recorded(population)))}",
                    f"population evaluation wall: {escape(str(_recorded(wall)))} ns",
                    f"recorded wall per population slot: {average} ns",
                    f"new unique candidates: {escape(str(summary['new_unique_candidates']))}",
                    f"cache-hit occurrences: {escape(str(summary['cache_hit_occurrences']))}",
                    f"same-generation duplicates: {escape(str(summary['same_generation_duplicate_occurrences']))}",
                    f"numerical-failure occurrences: {escape(str(summary['numerical_failure_occurrences']))}",
                    f"worker capacity: {escape(str(summary['worker_capacity']))}",
                    f"peak active callbacks (not CPU cores): {escape(str(summary['active_worker_count']))}",
                )))
                previous_generation = generation
            if wall is not None and population is not None:
                total_wall_ns += wall
                total_population += population
                complete_denominator_count += 1

        if complete_denominator_count and total_population:
            weighted_value = (
                f"{total_wall_ns}/{total_population} ns/slot "
                f"({total_wall_ns / total_population / 1e6:.9g} ms/slot)"
            )
            wall_text = f"{total_wall_ns} ns ({total_wall_ns / 1e9:.9g} s)"
            population_text: object = total_population
        else:
            weighted_value = "unavailable; no recorded population wall/slot denominator"
            wall_text = "not recorded"
            population_text = "not recorded"
        weighted_rows.append((
            run_label,
            len(ordered),
            population_text,
            wall_text,
            weighted_value,
        ))
        finish_segment()
        plot_traces.extend(
            (run_label, xs, ys, labels, index == 0)
            for index, (xs, ys, labels) in enumerate(segments)
        )

    go, pio, _ = _plotly()
    figure = go.Figure()
    for run_label, x_generation, y_average_ms, hover, show_legend in plot_traces:
        figure.add_trace(go.Scatter(
            x=x_generation,
            y=y_average_ms,
            mode="lines+markers",
            name=run_label,
            legendgroup=run_label,
            showlegend=show_legend,
            connectgaps=False,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))
    figure.update_xaxes(title_text="Recorded committed generation", rangemode="tozero")
    figure.update_yaxes(title_text="Recorded population wall / population size (ms)")
    _style(figure, Theme.AUTO, title="Generation population evaluation")
    figure.update_layout(hovermode="closest")

    body = [
        "<h2>Generation population performance</h2>",
        "<p>These summaries are included only through generations whose required "
        "numerical commit was acknowledged. Each row describes one complete "
        "population evaluation. Wall divided by population size is a generation "
        "aggregate, not an individual candidate duration; it may include dispatch, "
        "waits, and parallel overlap.</p>",
        "<h3>Weighted population wall per population slot</h3>",
        _table(
            ("run", "recorded generations", "population slots", "summed population wall", "weighted wall per slot"),
            weighted_rows,
        ),
        "<h3>Recorded generation summaries</h3>",
        _table(
            (
                "run", "generation", "population size",
                "population-evaluation wall", "recorded wall per slot",
                "new unique candidates", "cache-hit occurrences",
                "same-generation duplicate occurrences", "numerical-failure occurrences",
                "worker capacity", "peak active callbacks (not CPU cores)",
                "additional assembly/JIT/cache counters",
            ),
            table_rows,
        ),
        "<p>Worker capacity is configured candidate-worker capacity. Peak active "
        "candidate callbacks may include jobs waiting for assembly; it does not "
        "establish simultaneous native execution. PJRT pool settings are a separate "
        "runtime fact.</p>",
    ]
    if plot_traces:
        body.append(pio.to_html(
            figure,
            include_plotlyjs=True,
            full_html=False,
            auto_play=False,
            div_id="scnsim-generation-performance",
            config={"responsive": True, "scrollZoom": True},
        ))
    else:
        body.append("<p>No generation curve points have complete recorded timing values.</p>")
    body.append(_details("Recorded generation performance summaries", all_summaries))
    return "".join(body)


def _timeline_hover(operation: _TimelineOperation, span: _TimelineSpan) -> str:
    start = "unavailable" if span.start_ns is None else f"{span.start_ns} ns ({span.start_ns / 1e9:.9g} s from origin)"
    if span.start_ns is None or span.end_ns is None:
        duration = "unavailable: one or both endpoints not recorded"
    else:
        duration_ns = span.end_ns - span.start_ns
        duration = f"{duration_ns} ns ({duration_ns / 1e9:.9g} s)"
    return "<br>".join((
        f"operation: {escape(operation.operation_id)}",
        f"method: {escape(str(_recorded(operation.method)))}",
        f"operation kind: {escape(_timeline_operation_label(operation))}",
        f"backend / precision: {escape(str(_recorded(operation.backend)))} / {escape(str(_recorded(operation.precision)))}",
        f"operation status: {escape(operation.status)}; span status: {escape(span.status)}",
        f"span: {escape(span.kind)} · {escape(span.span_id)}",
        f"parent span: {escape(str(_recorded(span.parent_span_id)))}",
        f"clock: {escape(_timeline_clock_label(span.clock))}",
        f"start: {escape(start)}",
        f"end: {escape(str(_recorded(span.end_ns)))} ns",
        f"recorded span duration: {escape(duration)}",
        f"details: {escape(_native_compact(span.details))}",
        f"numerical refs: {escape(_native_compact(span.numerical_refs))}",
    ))


def _timeline_figure(
    operations: Sequence[_TimelineOperation],
    spans: Sequence[_TimelineSpan],
) -> tuple[Any | None, int]:
    """Build clock-segmented operation bars from recorded endpoints only."""

    groups, operation_by_id = _timeline_groups(operations, spans)
    timed_groups = {
        key: rows for key, rows in groups.items()
        if key[1] is not None and any(row.start_ns is not None for row in rows)
    }
    if not timed_groups:
        return None, 0

    go, _, make_subplots = _plotly()
    clock_ids = list(dict.fromkeys(clock_id for _, clock_id in timed_groups))
    figure = make_subplots(
        rows=len(clock_ids),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=min(0.04, 0.3 / max(len(clock_ids), 1)),
        subplot_titles=[f"Clock domain {clock_id}" for clock_id in clock_ids],
    )
    visible_spans = 0
    initial_category_count = 0
    color = report_palette(Theme.AUTO)[0].accent
    for row_index, clock_id in enumerate(clock_ids, 1):
        category_rows: list[str] = []
        category_labels: list[str] = []
        for (operation_id, selected_clock), rows in timed_groups.items():
            if selected_clock != clock_id:
                continue
            operation = operation_by_id[operation_id]
            method = str(_recorded(operation.method))
            precision = str(_recorded(operation.precision))
            for span in rows:
                if span.start_ns is None:
                    continue
                detail_prefix = "↳ " * max(span.depth, 1)
                if span.root:
                    label = (
                        f"{method}/{precision} · {_timeline_operation_label(operation)} · "
                        f"{_timeline_short_id(operation_id)}"
                    )
                else:
                    label = (
                        f"{method}/{precision} · {_timeline_short_id(operation_id)} · "
                        f"{detail_prefix}{span.kind} · {_timeline_short_id(span.span_id)}"
                    )
                row_key = _timeline_row_key(operation_id, span.span_id)
                if span.root:
                    category_rows.append(row_key)
                    category_labels.append(label)
                visible_spans += 1

                custom = _timeline_hover(operation, span)
                trace_meta = {
                    "timeline": {
                        "operation_id": operation.operation_id,
                        "method": operation.method,
                        "operation": operation.operation,
                        "backend": operation.backend,
                        "precision": operation.precision,
                        "status": operation.status,
                        "detail": not span.root,
                        "span_count": 1,
                        "axis_index": row_index,
                        "row_label": label,
                    }
                }
                if span.end_ns is None or span.end_ns == span.start_ns:
                    symbol = "circle-open" if span.end_ns is None else "circle"
                    figure.add_trace(go.Scatter(
                        x=[span.start_ns / 1e9],
                        y=[row_key],
                        mode="markers",
                        name=_timeline_operation_label(operation),
                        legendgroup=operation.operation_id,
                        meta=trace_meta,
                        text=[custom],
                        hovertemplate="%{text}<extra></extra>",
                        marker={"symbol": symbol, "size": 9, "color": color},
                        showlegend=False,
                        visible=span.root,
                    ), row=row_index, col=1)
                    continue

                duration_ns = span.end_ns - span.start_ns
                figure.add_trace(go.Bar(
                    x=[duration_ns / 1e9],
                    base=[span.start_ns / 1e9],
                    y=[row_key],
                    orientation="h",
                    name=_timeline_operation_label(operation),
                    legendgroup=operation.operation_id,
                    meta=trace_meta,
                    text=[custom],
                    hovertemplate="%{text}<extra></extra>",
                    marker={"color": color, "opacity": 0.9 if span.root else 0.65},
                    showlegend=False,
                    visible=span.root,
                ), row=row_index, col=1)

        figure.update_yaxes(
            categoryorder="array",
            categoryarray=category_rows,
            tickmode="array",
            tickvals=category_rows,
            ticktext=category_labels,
            autorange=True,
            title_text="Nested spans above operation roots",
            row=row_index,
            col=1,
        )
        initial_category_count += len(category_rows)
        figure.update_xaxes(
            title_text="Offset from this clock origin (s)",
            tickformat=".6g",
            row=row_index,
            col=1,
        )

    _style(figure, Theme.AUTO, title="Recorded operation timeline")
    figure.update_layout(
        barmode="overlay",
        hovermode="closest",
        showlegend=False,
        height=max(480, 170 + 80 * len(clock_ids) + 24 * initial_category_count),
        margin={"l": 240, "r": 32, "t": 80, "b": 72},
    )
    return figure, visible_spans


def _timeline_controls_script(div_id: str) -> str:
    return (
        "<script>(()=>{"
        f"const plot=document.getElementById('{div_id}');"
        "if(!plot||!window.Plotly)return;"
        "const controls=document.querySelector('.scnsim-timeline-filters');"
        "if(!controls)return;"
        "const selected=select=>new Set(Array.from(select.selectedOptions).map(o=>o.value));"
        "const applies=(values,value)=>values.size===0||values.has(value===null?'__scnsim_missing__':String(value));"
        "function update(){"
        "const filters={};controls.querySelectorAll('select[data-filter]').forEach(s=>filters[s.dataset.filter]=selected(s));"
        "const showDetails=controls.querySelector('[data-toggle-details]').checked;"
        "const indices=[];const visibility=[];const categories=new Map();let shown=0;const shownOps=new Set();"
        "Object.keys(plot.layout||{}).forEach(key=>{const match=key.match(/^yaxis([0-9]*)$/);"
        "if(match)categories.set(Number(match[1]||1),[]);});"
        "plot.data.forEach((trace,index)=>{const meta=trace.meta&&trace.meta.timeline||{};"
        "const visible=(!meta.detail||showDetails)&&Object.entries(filters).every(([key,values])=>applies(values,meta[key]));"
        "indices.push(index);visibility.push(visible);if(visible){shown+=Number(meta.span_count||0);if(meta.operation_id)shownOps.add(meta.operation_id);"
        "const axis=Number(meta.axis_index||1);const rows=categories.get(axis)||[];"
        "const keys=Array.from(trace.y||[]);const rowKey=keys[0];"
        "if(rowKey!==undefined&&!rows.some(row=>row.key===rowKey))rows.push({key:rowKey,label:meta.row_label||String(rowKey)});"
        "categories.set(axis,rows);}});"
        "Plotly.restyle(plot,{visible:visibility},indices);"
        "const layout={};let rowCount=0;let clockCount=0;"
        "categories.forEach((rows,axis)=>{const key=axis===1?'yaxis':`yaxis${axis}`;"
        "const rowKeys=rows.map(row=>row.key);layout[`${key}.categoryarray`]=rowKeys;"
        "layout[`${key}.tickmode`]='array';layout[`${key}.tickvals`]=rowKeys;layout[`${key}.ticktext`]=rows.map(row=>row.label);"
        "layout[`${key}.autorange`]=true;rowCount+=rows.length;clockCount++;});"
        "layout.height=Math.max(480,170+80*clockCount+24*rowCount);Plotly.relayout(plot,layout);"
        "const counter=controls.querySelector('[data-visible-count]');"
        "if(counter)counter.textContent=`${shownOps.size} operation(s) · ${shown} span row(s) shown`;"
        "}controls.querySelectorAll('select,[data-toggle-details]').forEach(el=>el.addEventListener('change',update));update();"
        "})();</script>"
    )


def _aggregate_timing_section(
    operations: Sequence[_TimelineOperation],
    batches: Sequence[Mapping[str, object]],
) -> str | None:
    """Combine acknowledged aggregate deltas without inventing intervals."""

    operation_ids = {operation.operation_id for operation in operations}
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for batch_index, batch in enumerate(batches):
        operation_id = batch.get("operation_id")
        if not isinstance(operation_id, str) or operation_id not in operation_ids:
            raise ValueError(f"timing_batches[{batch_index}] references an unknown operation")
        raw_groups = _required_sequence(batch.get("aggregates", []), f"timing_batches[{batch_index}].aggregates")
        for group_index, raw_value in enumerate(raw_groups):
            raw = _required_mapping(raw_value, f"timing_batches[{batch_index}].aggregates[{group_index}]")
            key_value = raw.get("group_key_sha256")
            kind = raw.get("kind")
            logical_parent = raw.get("logical_parent")
            if not isinstance(key_value, str) or not isinstance(kind, str) or not isinstance(logical_parent, str):
                raise TypeError("aggregate timing identity fields must be strings")
            key = (operation_id, key_value)
            count = _timeline_integer(raw.get("count"), "aggregate count")
            elapsed_sum = _timeline_integer(raw.get("elapsed_ns_sum"), "aggregate elapsed_ns_sum")
            elapsed_min = _timeline_integer(raw.get("elapsed_ns_min"), "aggregate elapsed_ns_min")
            elapsed_max = _timeline_integer(raw.get("elapsed_ns_max"), "aggregate elapsed_ns_max")
            if count is None or elapsed_sum is None or elapsed_min is None or elapsed_max is None:
                raise TypeError("aggregate timing counts and elapsed values must be integers")
            status_counts = _required_mapping(raw.get("status_counts"), "aggregate status_counts")
            context = _required_mapping(raw.get("context"), "aggregate context")
            max_context_value = raw.get("max_context")
            max_context = (
                _required_mapping(max_context_value, "aggregate max_context")
                if max_context_value is not None else None
            )
            occurrence = _timeline_integer(raw.get("max_occurrence"), "aggregate max_occurrence")
            current = groups.get(key)
            if current is None:
                groups[key] = {
                    "operation_id": operation_id,
                    "group_key_sha256": key_value,
                    "kind": kind,
                    "logical_parent": logical_parent,
                    "context": dict(context),
                    "count": count,
                    "status_counts": {str(name): _timeline_integer(value, "aggregate status count") for name, value in status_counts.items()},
                    "elapsed_ns_sum": elapsed_sum,
                    "elapsed_ns_min": elapsed_min,
                    "elapsed_ns_max": elapsed_max,
                    "max_context": None if max_context is None else dict(max_context),
                    "max_occurrence": occurrence,
                }
                continue
            current["count"] = int(current["count"]) + count
            current["elapsed_ns_sum"] = int(current["elapsed_ns_sum"]) + elapsed_sum
            current["elapsed_ns_min"] = min(int(current["elapsed_ns_min"]), elapsed_min)
            previous_max = int(current["elapsed_ns_max"])
            previous_occurrence = current["max_occurrence"]
            if (
                elapsed_max > previous_max
                or (
                    elapsed_max == previous_max
                    and occurrence is not None
                    and isinstance(previous_occurrence, int)
                    and occurrence < previous_occurrence
                )
            ):
                current["elapsed_ns_max"] = elapsed_max
                current["max_context"] = None if max_context is None else dict(max_context)
                current["max_occurrence"] = occurrence
            merged_status = current["status_counts"]
            assert isinstance(merged_status, dict)
            for name, value in status_counts.items():
                parsed_count = _timeline_integer(value, "aggregate status count")
                if parsed_count is None:
                    raise TypeError("aggregate status count must be an integer")
                merged_status[str(name)] = int(merged_status.get(str(name), 0)) + parsed_count

    if not groups:
        return None
    operation_order = {operation.operation_id: index for index, operation in enumerate(operations, 1)}
    rows: list[tuple[object, ...]] = []
    details: list[dict[str, object]] = []
    for key in sorted(groups, key=lambda item: (operation_order[item[0]], item[1])):
        group = groups[key]
        count = int(group["count"])
        elapsed_sum = int(group["elapsed_ns_sum"])
        mean = elapsed_sum / count
        details.append(group)
        rows.append((
            f"Run {operation_order[key[0]]}",
            group["kind"],
            group["logical_parent"],
            count,
            _native_compact(group["status_counts"]),
            f"{elapsed_sum} ns ({elapsed_sum / 1e9:.9g} s)",
            f"{mean:.9g} ns",
            f"{group['elapsed_ns_min']} ns",
            f"{group['elapsed_ns_max']} ns",
            _native_compact(group["max_context"]) if group["max_context"] is not None else "not recorded",
        ))
    return (
        "<h2>Aggregate phase timing</h2>"
        "<p>Rows combine committed aggregate deltas by operation, phase, logical parent, "
        "and stable execution context. No per-invocation intervals are reconstructed. "
        "Mean is derived from recorded elapsed sum and count.</p>"
        + _table(
            ("run", "phase", "logical parent", "count", "status counts",
             "elapsed sum", "mean", "minimum", "maximum", "maximum context"),
            rows,
        )
        + _details("Aggregate timing records", details)
    )


def _timeline_section(document: Mapping[str, object]) -> str:
    """Render one verified operation projection without loading its workspace."""

    raw_batches = _required_sequence(document.get("timing_batches", []), "timing_batches")
    batches = [
        _required_mapping(value, f"timing_batches[{index}]")
        for index, value in enumerate(raw_batches)
    ]
    derived_spans: list[dict[str, object]] = []
    generation_records: list[tuple[str, Mapping[str, object]]] = []
    for batch_index, batch in enumerate(batches):
        operation_id = batch.get("operation_id")
        if not isinstance(operation_id, str):
            raise TypeError(f"timing_batches[{batch_index}].operation_id must be a string")
        for span_index, raw_span in enumerate(_required_sequence(
            batch.get("spans", []), f"timing_batches[{batch_index}].spans"
        )):
            span = dict(_required_mapping(raw_span, f"timing_batches[{batch_index}].spans[{span_index}]"))
            if span.get("operation_id", operation_id) != operation_id:
                raise ValueError("detailed timing span operation identity differs from its batch")
            span["operation_id"] = operation_id
            span.setdefault("clock", batch.get("clock"))
            span.setdefault("numerical_refs", [])
            derived_spans.append(span)
        for summary_index, raw_summary in enumerate(_required_sequence(
            batch.get("generations", []), f"timing_batches[{batch_index}].generations"
        )):
            generation_records.append((
                operation_id,
                _required_mapping(raw_summary, f"timing_batches[{batch_index}].generations[{summary_index}]"),
            ))

    timeline_document = dict(document)
    timeline_document["spans"] = derived_spans
    operations, spans = _timeline_operations(timeline_document)
    figure, visible_spans = _timeline_figure(operations, spans)
    body: list[str] = [
        "<h1>SCNSim operation timing</h1>",
        "<p>Operation roots use their recorded start/end offsets. Nested timeline bars "
        "are shown only for detailed-mode intervals actually stored in timing batches. "
        "Aggregate-mode measurements remain summary tables and cannot reconstruct an "
        "interval timeline. Separate clock domains have separate axes; missing endpoints "
        "remain unknown. Parent spans are inclusive and are never added to child spans.</p>",
        _timeline_filters(operations),
    ]
    cumulative_label, wall_rows = _timeline_root_summaries(operations)
    body.append("<h2>Root-operation timing</h2>" + _table(
        ("measure", "recorded observation"),
        (("Cumulative closed-root duration", cumulative_label),),
    ))
    body.append("<h3>Wall envelope by clock domain</h3>" + _table(
        ("clock domain", "root operations", "wall envelope", "clock binding"),
        wall_rows,
    ))
    generation_section = _generation_performance_section(operations, generation_records)
    if generation_section is not None:
        body.append(generation_section)
    aggregate_section = _aggregate_timing_section(operations, batches)
    if aggregate_section is not None:
        body.append(aggregate_section)
    if figure is None:
        body.append("<h2>Detailed timeline</h2><p>No clock-aligned root or detailed intervals were recorded.</p>")
    else:
        div_id = "scnsim-operation-timeline"
        _, pio, _ = _plotly()
        body.append(
            "<h2>Root and detailed timing intervals</h2>"
            + pio.to_html(
                figure,
                include_plotlyjs=True,
                full_html=False,
                auto_play=False,
                div_id=div_id,
                config={"responsive": True, "scrollZoom": True},
            )
            + _timeline_controls_script(div_id)
        )
        unaligned = sum(
            span.start_ns is None or _timeline_clock_id(span.clock) is None
            for span in spans
        )
        body.append(
            f"<p>{visible_spans} root/detailed rows have same-clock start offsets; "
            f"{unaligned} rows with missing start or clock id remain unaligned.</p>"
        )

    operation_rows = []
    for operation in operations:
        if operation.status == "running" or operation.start_ns is None or operation.end_ns is None:
            root_duration = "unknown; endpoint unavailable or operation still open"
        else:
            elapsed_ns = operation.end_ns - operation.start_ns
            root_duration = f"{elapsed_ns} ns ({elapsed_ns / 1e9:.9g} s)"
        operation_rows.append((
            operation.operation_id,
            operation.method,
            _timeline_operation_label(operation),
            operation.backend,
            operation.precision,
            operation.status,
            root_duration,
            _timeline_clock_label(operation.clock),
            operation.request_sha256,
        ))
    if operation_rows:
        body.append("<h2>Operation roots</h2>" + _table(
            ("operation id", "method", "operation", "backend", "precision", "status", "root duration", "clock domain", "request SHA-256"),
            operation_rows,
        ))

    identity = (
        document["schema"],
        document["schema_version"],
        document["plan_sha256"],
        document["workspace_instance_id"],
    )
    body.append(_table(
        ("record", "schema version", "Plan SHA-256", "workspace instance"),
        (identity,),
    ))
    operation_status_counts: dict[str, int] = {}
    span_counts: dict[str, int] = {}
    batch_modes: dict[str, int] = {}
    for operation in operations:
        operation_status_counts[operation.status] = operation_status_counts.get(operation.status, 0) + 1
    for span in spans:
        if not span.root:
            span_counts[span.kind] = span_counts.get(span.kind, 0) + 1
    for batch in batches:
        mode = batch.get("timing_mode")
        if isinstance(mode, str):
            batch_modes[mode] = batch_modes.get(mode, 0) + 1
    count_rows = (
        [("operation status", key, value) for key, value in sorted(operation_status_counts.items())]
        + [("detailed span kind", key, value) for key, value in sorted(span_counts.items())]
        + [("timing mode", key, value) for key, value in sorted(batch_modes.items())]
    )
    body.append("<h2>Recorded row counts</h2>" + _table(
        ("category", "key", "recorded rows"),
        count_rows,
    ))
    task_states_value = document.get("task_states", {})
    task_states = _required_mapping(task_states_value, "task_states")
    task_state_rows = []
    for operation_id in sorted(task_states):
        state = _required_mapping(task_states[operation_id], f"task_states[{operation_id}]")
        association = state.get("association")
        task_status = state.get("task_status")
        task_state_rows.append((
            operation_id,
            _native_compact(association) if association is not None else "not associated",
            _recorded(task_status),
            _native_compact(state.get("latest_state_reference")),
            _native_compact(state.get("checkpoint_reference")),
        ))
    if task_state_rows:
        body.append("<h2>Independent task state and checkpoint references</h2>" + _table(
            ("operation id", "association", "task status", "latest state reference", "checkpoint reference"),
            task_state_rows,
        ))
    body.append(_details("Timing batch records", batches))
    body.append(_details("Task state references", [
        {
            "operation_id": operation_id,
            "association": state.get("association"),
            "task_status": state.get("task_status"),
            "latest_state_reference": state.get("latest_state_reference"),
            "checkpoint_reference": state.get("checkpoint_reference"),
        }
        for operation_id, raw_state in sorted(task_states.items())
        for state in [_required_mapping(raw_state, f"task_states[{operation_id}]")]
    ]))
    body.append(_details("Clock domains recorded on roots and detailed intervals", [
        {"operation_id": operation.operation_id, "clock": operation.clock}
        for operation in operations
    ] + [
        {"operation_id": span.operation_id, "span_id": span.span_id, "clock": span.clock}
        for span in spans if not span.root
    ]))
    body.append("<h2>Operation and detailed interval details</h2>")
    for operation in operations:
        related = [span for span in spans if span.operation_id == operation.operation_id]
        body.append(_details(
            f"{_recorded(operation.method)} · {_timeline_operation_label(operation)} · "
            f"{operation.operation_id} · {operation.status}",
            {
                "operation": {
                    "method": operation.method,
                    "operation": operation.operation,
                    "backend": operation.backend,
                    "precision": operation.precision,
                    "kind": operation.root_kind,
                    "parent_span_id": operation.parent_span_id,
                    "request_sha256": operation.request_sha256,
                    "clock": operation.clock,
                    "start_ns": operation.start_ns,
                    "end_ns": operation.end_ns,
                    "details": operation.details,
                    "numerical_refs": operation.numerical_refs,
                },
                "spans": [
                    {
                        "span_id": span.span_id,
                        "parent_span_id": span.parent_span_id,
                        "kind": span.kind,
                        "clock": span.clock,
                        "start_ns": span.start_ns,
                        "end_ns": span.end_ns,
                        "status": span.status,
                        "details": span.details,
                        "numerical_refs": span.numerical_refs,
                    }
                    for span in related
                ],
            },
        ))
    return "".join(body)

def render_benchmark(result: BenchmarkResult) -> str:
    """Render the current immutable Workspace operation report as HTML."""

    if not isinstance(result, BenchmarkResult):
        raise TypeError("render_benchmark requires BenchmarkResult")
    document = record_document(result.manifest_bytes)
    if (
        document.get("schema") != "scnsim.operation_benchmark"
        or document.get("schema_version") != 2
    ):
        raise ValueError("unsupported current Workspace operation evidence")
    return _report_html(_timeline_section(document), Theme.AUTO)
