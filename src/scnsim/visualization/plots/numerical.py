"""Frequency matrices, scalar quantities, traces, and operators."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from html import escape
from math import ceil
from typing import Any, Literal

import numpy as np
from pint import Quantity

from ..presentation import Theme
from ...results.selection import _channel_index, _frequency_index
from .common import (
    Channel, Component, MatrixKind, _channel_label, _channel_record,
    _component_values, _display_axis, _family, _frequency_axis,
    _frequency_unit_for_axis, _matrix_meta, _matrix_values, _plotly,
    _presentation_channel, _presentation_source,
    _require_axis_semantics, _response_axis, _set_db_viewport, _style,
    _subplot_type, _trace, _trace_meta, _unit_label,
)

def _validate_matrix_arguments(
    *,
    kind: MatrixKind,
    input_channel: Channel | None,
    output_channel: Channel | None,
    frequency: Quantity | None,
    component: Component | None,
    magnitude: str,
    family: str,
) -> None:
    if kind not in {"response", "table", "heatmap"}:
        raise ValueError("kind must be 'response', 'table', or 'heatmap'")
    if magnitude not in {"linear", "db"}:
        raise ValueError("magnitude must be 'linear' or 'db'")
    if family != "S" and magnitude != "linear":
        raise ValueError("dimensionful Y/Z magnitude has no dB presentation")
    if kind == "response":
        if frequency is not None:
            raise ValueError("frequency is valid only for table or heatmap")
        if component is not None:
            _component_values(np.asarray([1 + 0j]), component, family=family, magnitude=magnitude)
        return
    if input_channel is not None or output_channel is not None:
        raise ValueError("table and heatmap use the whole stored matrix; channel selectors are invalid")
    if frequency is None:
        raise ValueError("frequency is required for table or heatmap")
    if kind == "table":
        if component is not None or magnitude != "linear":
            raise ValueError("table retains complex values and accepts neither component nor dB magnitude")
        return
    if component is None:
        raise ValueError("heatmap requires an explicit component")
    _component_values(np.asarray([1 + 0j]), component, family=family, magnitude=magnitude)


def _matrix_table(view: Any, *, family: str, frequency: Quantity) -> Any:
    go, _, _ = _plotly()
    index = _frequency_index(view, frequency)
    values = _matrix_values(view)[index]
    unit = str(view.matrix.units)
    shown_frequency = frequency.to_compact()
    headers = [
        f"output \\ input<br>{shown_frequency:~P}",
        *(_channel_label(channel) for channel in view.input_channels),
    ]
    columns: list[list[str]] = [[_channel_label(channel) for channel in view.output_channels]]
    for input_index in range(len(view.input_channels)):
        columns.append([
            f"{value.real:.9g} {value.imag:+.9g}j {unit}"
            for value in values[:, input_index]
        ])
    return go.Table(header={"values": headers}, cells={"values": columns})


def _matrix_heatmap(view: Any, *, family: str, frequency: Quantity, component: Component, magnitude: str) -> Any:
    go, _, _ = _plotly()
    index = _frequency_index(view, frequency)
    values, _ = _component_values(
        _matrix_values(view)[index], component, family=family, magnitude=magnitude
    )
    unit = _unit_label(view, family, component, magnitude)
    return go.Heatmap(
        x=[_channel_label(channel) for channel in view.input_channels],
        y=[_channel_label(channel) for channel in view.output_channels],
        z=np.array(values, copy=True),
        colorbar={"title": unit},
        hovertemplate="output=%{y}<br>input=%{x}<br>value=%{z}<br>unit=" + unit + "<extra></extra>",
    )


def matrix_plot(
    result: Any,
    *,
    input_channel: Channel | None = None,
    output_channel: Channel | None = None,
    kind: MatrixKind = "response",
    frequency: Quantity | None = None,
    component: Component | None = None,
    magnitude: Literal["linear", "db"] = "linear",
    theme: Theme = Theme.AUTO,
) -> Any:
    view = result.view
    family = _family(view)
    _validate_matrix_arguments(
        kind=kind,
        input_channel=input_channel,
        output_channel=output_channel,
        frequency=frequency,
        component=component,
        magnitude=magnitude,
        family=family,
    )
    go, _, make_subplots = _plotly()
    selected: dict[str, object] = {"kind": kind}
    if kind == "response":
        input_index = _channel_index(view.input_channels, input_channel, role="input", default=True)
        output_index = _channel_index(view.output_channels, output_channel, role="output", default=True)
        components: tuple[Component, ...] = ("magnitude", "phase") if component is None else (component,)
        figure = make_subplots(
            rows=len(components),
            cols=1,
            shared_xaxes=True,
            subplot_titles=tuple(item for item in components),
        )
        for row, selected_component in enumerate(components, 1):
            trace = _trace(
                view,
                family=family,
                input_index=input_index,
                output_index=output_index,
                component=selected_component,
                magnitude=magnitude,
                source=_presentation_source(
                    result,
                    input_channel=_channel_record(view.input_channels[input_index]),
                    output_channel=_channel_record(view.output_channels[output_index]),
                ),
            )
            figure.add_trace(trace, row=row, col=1)
            figure.update_yaxes(
                title_text=_unit_label(
                    view,
                    family,
                    selected_component,
                    magnitude if selected_component == "magnitude" else "linear",
                ),
                row=row,
                col=1,
            )
            if selected_component == "magnitude" and magnitude == "db":
                _set_db_viewport(figure, trace.y, row=row)
        _, frequency_unit = _display_axis(view.frequencies)
        figure.update_xaxes(title_text=f"frequency ({frequency_unit})", row=len(components), col=1)
        selected.update(
            input_channel=_channel_record(view.input_channels[input_index]),
            output_channel=_channel_record(view.output_channels[output_index]),
            components=list(components),
        )
    elif kind == "table":
        assert frequency is not None
        figure = go.Figure(data=[_matrix_table(view, family=family, frequency=frequency)])
        figure.data[0].meta = {
            "scnsim": {"source": _presentation_source(result, frequency=str(frequency))}
        }
        figure.update_layout(height=max(240, 104 + 28 * len(view.output_channels)))
        selected.update(frequency=str(frequency), component="complex")
    else:
        assert frequency is not None and component is not None
        figure = go.Figure(data=[
            _matrix_heatmap(
                view,
                family=family,
                frequency=frequency,
                component=component,
                magnitude=magnitude,
            )
        ])
        figure.update_xaxes(title_text="input channel")
        figure.update_yaxes(title_text="output channel")
        unit = _unit_label(view, family, component, magnitude)
        figure.data[0].meta = _trace_meta(
            x_axis={
                "quantity": "input_channel",
                "channels": [_channel_record(channel) for channel in view.input_channels],
            },
            y_axis={
                "quantity": "output_channel",
                "channels": [_channel_record(channel) for channel in view.output_channels],
                "value": _response_axis(
                    family=family, component=component, magnitude=magnitude, unit=unit
                ),
            },
            source=_presentation_source(result, frequency=str(frequency)),
        )
        selected.update(frequency=str(frequency), component=component)
    layout_meta = _matrix_meta(view, family, selected=selected)
    layout_meta["scnsim"]["source"] = _presentation_source(result)
    figure.update_layout(meta=layout_meta)
    title = f"{family} matrix {kind}"
    if kind in {"table", "heatmap"}:
        assert frequency is not None
        title += f" at {frequency.to_compact():~P}"
    return _style(figure, theme, title=title)


def matrix_add_to(
    result: Any,
    figure: Any,
    *,
    row: int,
    col: int,
    kind: Literal["trace", "table", "heatmap"],
    input_channel: Channel | None = None,
    output_channel: Channel | None = None,
    frequency: Quantity | None = None,
    component: Component | None = None,
    magnitude: Literal["linear", "db"] = "linear",
) -> Any:
    go, _, _ = _plotly()
    if not isinstance(figure, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")
    view = result.view
    family = _family(view)
    expected_subplot = "domain" if kind == "table" else "xy"
    if _subplot_type(figure, row, col) != expected_subplot:
        raise ValueError(f"{kind} requires a compatible {expected_subplot} subplot")
    if kind == "trace":
        if frequency is not None:
            raise ValueError("trace does not accept an exact frequency")
        if component is None:
            raise ValueError("trace requires one explicit component")
        _component_values(np.asarray([1 + 0j]), component, family=family, magnitude=magnitude)
        input_index = _channel_index(view.input_channels, input_channel, role="input", default=False)
        output_index = _channel_index(view.output_channels, output_channel, role="output", default=False)
        unit = _unit_label(view, family, component, magnitude if component == "magnitude" else "linear")
        y_axis = _response_axis(
            family=family,
            component=component,
            magnitude=magnitude if component == "magnitude" else "linear",
            unit=unit,
        )
        frequency_unit = _frequency_unit_for_axis(
            figure, row, col, frequencies=view.frequencies, y_axis=y_axis
        )
        trace = _trace(
            view,
            family=family,
            input_index=input_index,
            output_index=output_index,
            component=component,
            magnitude=magnitude,
            frequency_unit=frequency_unit,
            source=_presentation_source(
                result,
                input_channel=_channel_record(view.input_channels[input_index]),
                output_channel=_channel_record(view.output_channels[output_index]),
            ),
        )
    elif kind == "table":
        _validate_matrix_arguments(
            kind="table", input_channel=input_channel, output_channel=output_channel,
            frequency=frequency, component=component, magnitude=magnitude, family=family,
        )
        assert frequency is not None
        trace = _matrix_table(view, family=family, frequency=frequency)
    elif kind == "heatmap":
        _validate_matrix_arguments(
            kind="heatmap", input_channel=input_channel, output_channel=output_channel,
            frequency=frequency, component=component, magnitude=magnitude, family=family,
        )
        assert frequency is not None and component is not None
        trace = _matrix_heatmap(
            view, family=family, frequency=frequency, component=component, magnitude=magnitude
        )
        unit = _unit_label(view, family, component, magnitude)
        x_axis = {
            "quantity": "input_channel",
            "channels": [_channel_record(channel) for channel in view.input_channels],
        }
        y_axis = {
            "quantity": "output_channel",
            "channels": [_channel_record(channel) for channel in view.output_channels],
            "value": _response_axis(
                family=family, component=component, magnitude=magnitude, unit=unit
            ),
        }
        trace.meta = _trace_meta(
            x_axis=x_axis,
            y_axis=y_axis,
            source=_presentation_source(result, frequency=str(frequency)),
        )
        _require_axis_semantics(
            figure, row, col, x_axis=x_axis, y_axis=y_axis
        )
    else:
        raise ValueError("kind must be 'trace', 'table', or 'heatmap'")
    figure.add_trace(trace, row=row, col=col)
    return figure


def _scalar_cell_text(value: str) -> str:
    """Escape a shared presentation string and insert renderer-only line breaks."""

    paragraphs = value.splitlines() or [value]
    lines = [
        line
        for paragraph in paragraphs
        for line in (textwrap.wrap(paragraph, width=36, break_long_words=True, break_on_hyphens=False) or [""])
    ]
    return "<br>".join(escape(line) for line in lines)


def scalar_plot(result: Any, *, theme: Theme = Theme.AUTO, detailed: bool = False) -> Any:
    from ..quantity_presentation import build_scalar_presentation

    presentation = build_scalar_presentation(result, detailed=detailed)
    tables = presentation.tables
    go, _, make_subplots = _plotly()

    prepared: list[tuple[Any, list[list[str]], int, int]] = []
    for table in tables:
        rendered_rows = [tuple(_scalar_cell_text(value) for value in row) for row in table.rows]
        columns = [list(column) for column in zip(*rendered_rows, strict=True)] if rendered_rows else [
            [] for _ in table.columns
        ]
        line_count = max(
            (value.count("<br>") + 1 for row in rendered_rows for value in row),
            default=1,
        )
        cell_height = max(30, 20 * line_count + 8)
        table_height = 42 + cell_height * max(1, len(rendered_rows))
        prepared.append((table, columns, cell_height, table_height))

    total_table_height = sum(item[3] for item in prepared)
    row_heights = [item[3] / total_table_height for item in prepared]
    vertical_spacing = min(0.08, 0.28 / len(prepared))
    figure = make_subplots(
        rows=len(prepared),
        cols=1,
        specs=[[{"type": "domain"}] for _ in prepared],
        row_heights=row_heights,
        subplot_titles=tuple(table.title for table, _, _, _ in prepared),
        vertical_spacing=vertical_spacing,
    )
    for row_index, (table, columns, cell_height, _) in enumerate(prepared, start=1):
        figure.add_trace(
            go.Table(
                header={
                    "values": [_scalar_cell_text(value) for value in table.columns],
                    "align": "left",
                    "height": 42,
                    "font": {"size": 13},
                },
                cells={
                    "values": columns,
                    "align": "left",
                    "height": cell_height,
                    "font": {"size": 12},
                },
            ),
            row=row_index,
            col=1,
        )
    figure.update_annotations(font_size=14)
    _style(figure, theme, title=presentation.title)
    margins = figure.layout.margin
    top_margin = int(margins.t or 0)
    bottom_margin = int(margins.b or 0)
    domain_fraction = 1.0 - vertical_spacing * max(0, len(prepared) - 1)
    # Domain rows share only the drawable area after Plotly's margins and gaps.
    plot_area_height = ceil(total_table_height / domain_fraction)
    figure.update_layout(
        height=max(300, top_margin + bottom_margin + plot_area_height),
        meta={
            "scnsim": {
                "kind": "scalar_quantity",
                "table_titles": [table.title for table, _, _, _ in prepared],
                "detailed": detailed,
            }
        },
    )
    return figure


def scalar_add_to(result: Any, figure: Any, *, row: int, col: int) -> Any:
    go, _, _ = _plotly()
    if not isinstance(figure, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")
    if _subplot_type(figure, row, col) != "domain":
        raise ValueError("scalar table requires a compatible domain subplot")
    source = scalar_plot(result)
    figure.add_trace(source.data[0], row=row, col=col)
    return figure


def trace_plot(
    result: Any,
    *,
    component: Component | None = None,
    magnitude: Literal["linear", "db"] = "linear",
    theme: Theme = Theme.AUTO,
) -> Any:
    if component is not None:
        _component_values(np.asarray([1 + 0j]), component, family="S", magnitude=magnitude)
    go, _, make_subplots = _plotly()
    values = np.asarray(result.value.magnitude)
    if values.ndim != 1 or values.shape != np.asarray(result.frequencies.magnitude).shape:
        raise ValueError("trace values disagree with their frequency grid")
    components: tuple[Component, ...] = ("magnitude", "phase") if component is None else (component,)
    figure = make_subplots(rows=len(components), cols=1, shared_xaxes=True, subplot_titles=components)
    frequencies, frequency_unit = _display_axis(result.frequencies)
    label = escape(str(getattr(result, "_presentation", {}).get("id", "trace")))
    for row, selected in enumerate(components, 1):
        selected_magnitude = magnitude if selected == "magnitude" else "linear"
        shown, _ = _component_values(values, selected, family="S", magnitude=selected_magnitude)
        unit = "dB" if selected == "magnitude" and magnitude == "db" else (
            "degree (exact zero undefined)" if selected == "phase" else str(result.value.units)
        )
        y_axis = _response_axis(
            family="S", component=selected, magnitude=selected_magnitude, unit=unit
        )
        figure.add_trace(
            go.Scatter(
                x=np.array(frequencies, copy=True),
                y=np.array(shown, copy=True),
                mode="lines",
                name=f"{label} {selected}",
                meta=_trace_meta(
                    x_axis=_frequency_axis(frequency_unit),
                    y_axis=y_axis,
                    source=_presentation_source(result),
                ),
                hovertemplate=f"frequency=%{{x}} {frequency_unit}<br>value=%{{y}} {unit}<extra>%{{fullData.name}}</extra>",
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text=unit, row=row, col=1)
        if selected == "magnitude" and magnitude == "db":
            _set_db_viewport(figure, shown, row=row)
    figure.update_xaxes(title_text=f"frequency ({frequency_unit})", row=len(components), col=1)
    presentation = getattr(result, "_presentation", {})
    lineage = presentation.get("ref_lineage") if isinstance(presentation, Mapping) else None
    figure.update_layout(meta={"scnsim": {
        "kind": "trace",
        "id": str(presentation.get("id", "trace")) if isinstance(presentation, Mapping) else "trace",
        "family": str(presentation.get("family", "S")) if isinstance(presentation, Mapping) else "S",
        "input_channel": (
            _presentation_channel(presentation.get("input_channel"))
            if isinstance(presentation, Mapping) else None
        ),
        "output_channel": (
            _presentation_channel(presentation.get("output_channel"))
            if isinstance(presentation, Mapping) else None
        ),
        "view_lineage": (
            str(lineage.get("lineage_sha256"))
            if isinstance(lineage, Mapping) and isinstance(lineage.get("lineage_sha256"), str)
            else None
        ),
    }})
    return _style(figure, theme, title=label)


def _trace_result_trace(
    result: Any,
    *,
    component: Component,
    magnitude: Literal["linear", "db"],
    name: str | None,
    frequency_unit: str | None = None,
    source: Mapping[str, object] | None = None,
) -> Any:
    go, _, _ = _plotly()
    selected_magnitude = magnitude if component == "magnitude" else "linear"
    if component != "magnitude" and magnitude != "linear":
        raise ValueError("magnitude is incompatible with the selected component")
    values, _ = _component_values(
        np.asarray(result.value.magnitude), component, family="S", magnitude=selected_magnitude
    )
    unit = "dB" if component == "magnitude" and magnitude == "db" else (
        "degree (exact zero undefined)" if component == "phase" else str(result.value.units)
    )
    frequencies, frequency_unit = _display_axis(result.frequencies, frequency_unit)
    y_axis = _response_axis(
        family="S", component=component, magnitude=selected_magnitude, unit=unit
    )
    label = escape(name or str(getattr(result, "_presentation", {}).get("id", "trace")))
    return go.Scatter(
        x=np.array(frequencies, copy=True),
        y=np.array(values, copy=True),
        mode="lines",
        name=label,
        meta=_trace_meta(
            x_axis=_frequency_axis(frequency_unit),
            y_axis=y_axis,
            source=source or _presentation_source(result),
        ),
        hovertemplate=(
            f"frequency=%{{x}} {frequency_unit}<br>value=%{{y}} {unit}"
            "<extra>%{fullData.name}</extra>"
        ),
    )


def trace_add_to(
    result: Any,
    figure: Any,
    *,
    row: int,
    col: int,
    component: Component,
    magnitude: Literal["linear", "db"] = "linear",
    name: str | None = None,
) -> Any:
    go, _, _ = _plotly()
    if not isinstance(figure, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")
    if _subplot_type(figure, row, col) != "xy":
        raise ValueError("trace requires a compatible xy subplot")
    unit = "dB" if component == "magnitude" and magnitude == "db" else (
        "degree (exact zero undefined)" if component == "phase" else str(result.value.units)
    )
    selected_magnitude = magnitude if component == "magnitude" else "linear"
    y_axis = _response_axis(
        family="S", component=component, magnitude=selected_magnitude, unit=unit
    )
    frequency_unit = _frequency_unit_for_axis(
        figure, row, col, frequencies=result.frequencies, y_axis=y_axis
    )
    trace = _trace_result_trace(
        result,
        component=component,
        magnitude=magnitude,
        name=name,
        frequency_unit=frequency_unit,
    )
    figure.add_trace(trace, row=row, col=col)
    return figure


def operator_plot(
    result: Any,
    *,
    frequency: Quantity,
    kind: Literal["table", "heatmap"] = "table",
    component: Component | None = None,
    theme: Theme = Theme.AUTO,
) -> Any:
    point = result.at(frequency)
    matrix = np.asarray(point.matrix.magnitude)
    if matrix.shape != (len(point.coordinates), len(point.coordinates)):
        raise ValueError("operator matrix shape disagrees with its coordinate inventory")
    go, _, _ = _plotly()
    if kind == "table":
        if component is not None:
            raise ValueError("operator table retains complex values and does not accept component")
        labels = [escape(coordinate) for coordinate in point.coordinates]
        headers = [f"output \\ input<br>{point.frequency.to_compact():~P}", *labels]
        columns: list[list[str]] = [labels]
        unit = str(point.matrix.units)
        for input_index in range(len(point.coordinates)):
            columns.append([f"{value.real:.9g} {value.imag:+.9g}j {unit}" for value in matrix[:, input_index]])
        trace = go.Table(header={"values": headers}, cells={"values": columns})
    elif kind == "heatmap":
        if component is None:
            raise ValueError("operator heatmap requires an explicit component")
        values, _ = _component_values(matrix, component, family="Y", magnitude="linear")
        unit = "degree (exact zero undefined)" if component == "phase" else str(point.matrix.units)
        trace = go.Heatmap(
            x=[escape(coordinate) for coordinate in point.coordinates],
            y=[escape(coordinate) for coordinate in point.coordinates], z=values,
            colorbar={"title": unit},
            hovertemplate=f"output=%{{y}}<br>input=%{{x}}<br>value=%{{z}} {unit}<extra></extra>",
        )
    else:
        raise ValueError("kind must be 'table' or 'heatmap'")
    if kind == "table":
        trace.meta = {
            "scnsim": {"source": _presentation_source(result, frequency=str(point.frequency))}
        }
    else:
        x_axis = {"quantity": "operator_input", "coordinates": list(point.coordinates)}
        y_axis = {
            "quantity": "operator_output",
            "coordinates": list(point.coordinates),
            "value": {"component": component, "unit": unit},
        }
        trace.meta = _trace_meta(
            x_axis=x_axis,
            y_axis=y_axis,
            source=_presentation_source(result, frequency=str(point.frequency)),
        )
    figure = go.Figure(data=[trace])
    figure.update_layout(meta={"scnsim": {
        "kind": "operator",
        "frequency": str(point.frequency),
        "coordinates": list(point.coordinates),
        "source": _presentation_source(result, frequency=str(point.frequency)),
    }})
    return _style(figure, theme, title=f"Operator {kind} at {point.frequency.to_compact():~P}")


def operator_add_to(
    result: Any,
    figure: Any,
    *,
    row: int,
    col: int,
    frequency: Quantity,
    kind: Literal["table", "heatmap"],
    component: Component | None = None,
) -> Any:
    go, _, _ = _plotly()
    if not isinstance(figure, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")
    expected = "domain" if kind == "table" else "xy"
    if _subplot_type(figure, row, col) != expected:
        raise ValueError(f"operator {kind} requires a compatible {expected} subplot")
    source = operator_plot(result, frequency=frequency, kind=kind, component=component)
    if kind == "heatmap":
        trace = source.data[0]
        point = result.at(frequency)
        unit = "degree (exact zero undefined)" if component == "phase" else str(point.matrix.units)
        x_axis = {"quantity": "operator_input", "coordinates": list(point.coordinates)}
        y_axis = {
            "quantity": "operator_output",
            "coordinates": list(point.coordinates),
            "value": {"component": component, "unit": unit},
        }
        trace.meta = _trace_meta(
            x_axis=x_axis,
            y_axis=y_axis,
            source=_presentation_source(result, frequency=str(point.frequency)),
        )
        _require_axis_semantics(
            figure, row, col, x_axis=x_axis, y_axis=y_axis
        )
    figure.add_trace(source.data[0], row=row, col=col)
    return figure
