"""Lazy Plotly presentation for already-materialized numerical Results.

This module owns no calculation or evidence.  Public Result methods import it
only when a Figure is requested, keeping Plotly and optional static-export
support outside solve/evaluate/resolve startup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Any, Literal

import numpy as np
from pint import Quantity

from .presentation import Theme, _Palette, _palette, _require_theme


Channel = str | tuple[str, tuple[int, ...]]
Component = Literal["magnitude", "phase", "real", "imag"]
MatrixKind = Literal["response", "table", "heatmap"]


def _plotly() -> tuple[Any, Any, Any]:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    return go, pio, make_subplots


def _template(theme: Theme) -> str:
    checked = _require_theme(theme)
    if checked is Theme.LIGHT:
        return "plotly_white"
    if checked is Theme.DARK:
        return "plotly_dark"
    _, pio, _ = _plotly()
    return str(pio.templates.default or "plotly")


def _style(figure: Any, theme: Theme, *, title: str | None = None) -> Any:
    checked = _require_theme(theme)
    layout: dict[str, object] = {
        "template": _template(checked),
        "title": title,
        "legend_title_text": "series",
        "margin": {"l": 72, "r": 32, "t": 72 if title else 40, "b": 64},
    }
    if checked is not Theme.AUTO:
        layout["colorway"] = list(_palette(checked).cycle)
    figure.update_layout(**layout)
    return figure


def report_palette(theme: Theme) -> tuple[_Palette, Literal["light", "dark"]]:
    """Resolve report colors from the same fixed Plotly template as Figures."""

    checked = _require_theme(theme)
    if checked is not Theme.AUTO:
        return _palette(checked), checked.value
    _, pio, _ = _plotly()
    layout = pio.templates[_template(checked)].layout
    fallback = _palette(Theme.LIGHT)
    background = str(layout.paper_bgcolor or layout.plot_bgcolor or fallback.background)
    foreground = str(layout.font.color or fallback.foreground)
    grid = str(layout.xaxis.gridcolor or layout.yaxis.gridcolor or fallback.grid)
    colorway = tuple(str(color) for color in (layout.colorway or fallback.cycle))
    accent = colorway[0] if colorway else fallback.accent
    resolved = _Palette(
        background=background,
        foreground=foreground,
        secondary=foreground,
        grid=grid,
        accent=accent,
        cycle=colorway or fallback.cycle,
    )
    # Plotly's built-in dark templates state that choice in their resolved
    # paper background. Unknown custom color syntaxes retain a light control
    # scheme while using the exact template colors above.
    compact = background.lower().replace(" ", "")
    scheme: Literal["light", "dark"] = "dark" if compact in {
        "#000", "#000000", "#111", "#111111", "rgb(0,0,0)", "black"
    } or "dark" in _template(checked).lower() else "light"
    return resolved, scheme


def show_figure(figure: Any) -> None:
    """Display one native Figure and suppress a second notebook MIME display."""

    figure.show()


def _family(view: Any) -> Literal["S", "Y", "Z"]:
    matrix = view.matrix
    if matrix.dimensionless:
        return "S"
    if matrix.is_compatible_with("siemens"):
        return "Y"
    if matrix.is_compatible_with("ohm"):
        return "Z"
    raise ValueError("matrix family has no supported S, Y, or Z unit")


def _channel_label(channel: tuple[str, tuple[int, ...]]) -> str:
    coordinate, mode = channel
    label = coordinate if not mode else f"{coordinate}{mode}"
    return escape(label)


def _channel_record(channel: tuple[str, tuple[int, ...]]) -> dict[str, object]:
    coordinate, mode = channel
    return {"coordinate": coordinate, "mode": list(mode)}


def _presentation_channel(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping) or not isinstance(value.get("coordinate"), str):
        return None
    mode = value.get("mode")
    if not isinstance(mode, Sequence) or isinstance(mode, (str, bytes)):
        return None
    return {"coordinate": value["coordinate"], "mode": [int(item) for item in mode]}


def _channel_index(
    channels: Sequence[tuple[str, tuple[int, ...]]],
    selector: Channel | None,
    *,
    role: str,
    default: bool,
) -> int:
    if not channels:
        raise ValueError(f"matrix has no {role} channels")
    if selector is None:
        if default:
            return 0
        raise ValueError(f"{role}_channel is required")
    if isinstance(selector, str):
        matches = [index for index, (coordinate, _) in enumerate(channels) if coordinate == selector]
        if not matches:
            raise ValueError(f"unknown {role} channel coordinate: {selector}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous {role} channel coordinate; use (coordinate, mode)")
        return matches[0]
    if (
        not isinstance(selector, tuple)
        or len(selector) != 2
        or not isinstance(selector[0], str)
        or not isinstance(selector[1], tuple)
        or any(not isinstance(value, int) or isinstance(value, bool) for value in selector[1])
    ):
        raise TypeError(f"{role}_channel must be a coordinate string or (coordinate, mode) tuple")
    try:
        return channels.index(selector)
    except ValueError as error:
        raise ValueError(f"unknown {role} channel: {selector!r}") from error


def _matrix_values(view: Any) -> np.ndarray:
    matrix = np.asarray(view.matrix.magnitude)
    if matrix.ndim != 3:
        raise ValueError("matrix must have [frequency, output, input] axes")
    if matrix.shape != (
        np.asarray(view.frequencies.magnitude).size,
        len(view.output_channels),
        len(view.input_channels),
    ):
        raise ValueError("matrix shape disagrees with its stored channel inventory")
    return matrix


def _display_axis(values: Quantity, unit: str | None = None) -> tuple[np.ndarray, str]:
    hertz = values.to("hertz")
    if unit is None:
        magnitudes = np.asarray(hertz.magnitude)
        maximum = float(np.max(np.abs(magnitudes))) if magnitudes.size else 0.0
        unit = (
            "gigahertz" if maximum >= 1e9
            else "megahertz" if maximum >= 1e6
            else "kilohertz" if maximum >= 1e3
            else "hertz"
        )
    try:
        displayed = hertz.to(unit)
    except Exception as error:
        raise ValueError("frequency display unit is incompatible with frequency") from error
    return np.asarray(displayed.magnitude), f"{displayed.units:~P}"


def _frequency_axis(unit: str) -> dict[str, str]:
    return {"quantity": "frequency", "unit": unit}


def _response_axis(
    *, family: str, component: Component, magnitude: str, unit: str
) -> dict[str, str]:
    return {
        "quantity": "network_response",
        "family": family,
        "component": component,
        "magnitude": magnitude,
        "unit": unit,
    }


def _trace_meta(
    *,
    x_axis: Mapping[str, object],
    y_axis: Mapping[str, object],
    source: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "scnsim": {
            "unit": y_axis.get("unit"),
            "x_axis": dict(x_axis),
            "y_axis": dict(y_axis),
            "source": dict(source or {}),
        }
    }


def _metadata_value(value: object) -> object:
    """Detach frozen decoder metadata into Plotly's JSON-shaped values."""

    if isinstance(value, Mapping):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_metadata_value(item) for item in value]
    return value


def _presentation_source(result: Any, **selected: object) -> dict[str, object]:
    source: dict[str, object] = dict(selected)
    identity = getattr(result, "identity", None)
    if identity is None:
        identity = getattr(result, "_parent_identity", None)
    point_batch = getattr(identity, "batch", None)
    if point_batch is not None:
        source["source_index"] = _metadata_value(getattr(identity, "source_index", None))
        source["parameters_sha256"] = getattr(identity, "parameters_sha256", None)
        identity = point_batch
    for name in ("plan_sha256", "request_sha256", "attempt_sha256", "result_sha256"):
        value = getattr(identity, name, None)
        if isinstance(value, str):
            source[name] = value
    presentation = getattr(result, "_presentation", None)
    if isinstance(presentation, Mapping):
        for name in (
            "id", "family", "case_id", "input_channel", "output_channel", "view",
        ):
            if name in presentation and name not in source:
                source[name] = _metadata_value(presentation[name])
    lineage = presentation.get("ref_lineage") if isinstance(presentation, Mapping) else None
    if isinstance(lineage, Mapping) and isinstance(lineage.get("lineage_sha256"), str):
        source["view_lineage_sha256"] = lineage["lineage_sha256"]
    return source


def _frequency_index(view: Any, frequency: Quantity | None) -> int:
    if not isinstance(frequency, Quantity) or np.asarray(frequency.magnitude).ndim != 0:
        raise TypeError("frequency must be one scalar Quantity")
    try:
        wanted = float(frequency.to(view.frequencies.units).magnitude)
    except Exception as error:
        raise ValueError("frequency is incompatible with the stored grid") from error
    values = np.asarray(view.frequencies.magnitude)
    matches = np.flatnonzero(values == wanted)
    if matches.size != 1:
        raise KeyError("frequency was not materialized exactly once")
    return int(matches[0])


def _component_values(values: np.ndarray, component: Component, *, family: str, magnitude: str) -> tuple[np.ndarray, str]:
    if component not in {"magnitude", "phase", "real", "imag"}:
        raise ValueError("component must be 'magnitude', 'phase', 'real', or 'imag'")
    if magnitude not in {"linear", "db"}:
        raise ValueError("magnitude must be 'linear' or 'db'")
    if magnitude == "db" and (family != "S" or component != "magnitude"):
        raise ValueError("magnitude='db' is supported only for S magnitude")
    if component != "magnitude" and magnitude != "linear":
        raise ValueError("magnitude is incompatible with the selected component")
    if component == "magnitude":
        result = np.abs(values)
        if magnitude == "db":
            with np.errstate(divide="ignore"):
                result = 20.0 * np.log10(result)
            return result, "dB"
        return result, "dimensionless" if family == "S" else family
    if component == "phase":
        result = np.where(np.abs(values) == 0.0, np.nan, np.angle(values, deg=True))
        return result, "degree (exact zero undefined)"
    if component == "real":
        return np.real(values), "dimensionless" if family == "S" else family
    return np.imag(values), "dimensionless" if family == "S" else family


def _unit_label(view: Any, family: str, component: Component, magnitude: str) -> str:
    if component == "phase":
        return "degree (exact zero undefined)"
    if family == "S":
        return "dB" if component == "magnitude" and magnitude == "db" else "dimensionless"
    return str(view.matrix.units)


def _set_db_viewport(figure: Any, values: object, *, row: int) -> None:
    """Avoid magnifying roundoff while leaving exact trace values untouched."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size and float(np.max(finite) - np.min(finite)) < 0.1:
        center = float((np.max(finite) + np.min(finite)) / 2.0)
        figure.update_yaxes(range=[center - 0.05, center + 0.05], row=row, col=1)


def _trace(
    view: Any,
    *,
    family: str,
    input_index: int,
    output_index: int,
    component: Component,
    magnitude: str,
    name: str | None = None,
    frequency_unit: str | None = None,
    source: Mapping[str, object] | None = None,
) -> Any:
    go, _, _ = _plotly()
    matrix = _matrix_values(view)
    component_magnitude = magnitude if component == "magnitude" else "linear"
    values, _ = _component_values(
        matrix[:, output_index, input_index], component, family=family, magnitude=component_magnitude
    )
    input_label = _channel_label(view.input_channels[input_index])
    output_label = _channel_label(view.output_channels[output_index])
    unit = _unit_label(view, family, component, component_magnitude)
    label = name or f"{family}[{output_label} <- {input_label}] {component}"
    frequencies, frequency_unit = _display_axis(view.frequencies, frequency_unit)
    y_axis = _response_axis(
        family=family,
        component=component,
        magnitude=component_magnitude,
        unit=unit,
    )
    return go.Scatter(
        x=np.array(frequencies, copy=True),
        y=np.array(values, copy=True),
        mode="lines",
        name=label,
        meta=_trace_meta(
            x_axis=_frequency_axis(frequency_unit), y_axis=y_axis, source=source
        ),
        customdata=np.full(values.shape, unit, dtype=object),
        hovertemplate=f"frequency=%{{x}} {frequency_unit}<br>value=%{{y}}<br>unit=%{{customdata}}<extra>%{{fullData.name}}</extra>",
    )


def _matrix_meta(view: Any, family: str, *, selected: Mapping[str, object] | None = None) -> dict[str, object]:
    return {
        "scnsim": {
            "kind": "matrix_family",
            "family": family,
            "matrix_unit": str(view.matrix.units),
            "frequency_unit": str(view.frequencies.units),
            "input_channels": [_channel_record(channel) for channel in view.input_channels],
            "output_channels": [_channel_record(channel) for channel in view.output_channels],
            "selected": dict(selected or {}),
        }
    }


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


def _subplot_type(figure: Any, row: int, col: int) -> str:
    if not isinstance(row, int) or isinstance(row, bool) or row < 1 or not isinstance(col, int) or isinstance(col, bool) or col < 1:
        raise TypeError("row and col must be positive integers")
    grid = getattr(figure, "_grid_ref", None)
    try:
        refs = grid[row - 1][col - 1]
        return refs[0].subplot_type
    except (TypeError, IndexError, AttributeError) as error:
        raise ValueError("figure does not contain the requested Plotly subplot") from error


def _axis_traces(figure: Any, row: int, col: int) -> tuple[Any, ...]:
    """Return traces assigned to one existing xy subplot."""

    reference = figure._grid_ref[row - 1][col - 1][0]
    target_x = reference.trace_kwargs.get("xaxis")
    target_y = reference.trace_kwargs.get("yaxis")
    return tuple(
        trace
        for trace in figure.data
        if getattr(trace, "xaxis", None) == target_x
        and getattr(trace, "yaxis", None) == target_y
    )


def _axis_semantics(trace: Any) -> tuple[Mapping[str, object], Mapping[str, object]]:
    meta = getattr(trace, "meta", None)
    recorded = meta.get("scnsim") if isinstance(meta, Mapping) else None
    x_axis = recorded.get("x_axis") if isinstance(recorded, Mapping) else None
    y_axis = recorded.get("y_axis") if isinstance(recorded, Mapping) else None
    if not isinstance(x_axis, Mapping) or not isinstance(y_axis, Mapping):
        raise ValueError(
            "target subplot already contains a trace without SCNSim axis semantics; use a separate axis"
        )
    return x_axis, y_axis


def _require_axis_semantics(
    figure: Any,
    row: int,
    col: int,
    *,
    x_axis: Mapping[str, object],
    y_axis: Mapping[str, object],
) -> None:
    """Reject insertion into an occupied axis with different physical semantics."""

    for trace in _axis_traces(figure, row, col):
        recorded_x, recorded_y = _axis_semantics(trace)
        if dict(recorded_x) != dict(x_axis) or dict(recorded_y) != dict(y_axis):
            raise ValueError(
                "target subplot already contains a trace with different axis semantics; use a separate axis"
            )


def _frequency_unit_for_axis(
    figure: Any,
    row: int,
    col: int,
    *,
    frequencies: Quantity,
    y_axis: Mapping[str, object],
) -> str:
    """Select one frequency display unit, honoring an occupied compatible axis."""

    existing = _axis_traces(figure, row, col)
    if not existing:
        _, unit = _display_axis(frequencies)
        return unit
    selected_unit: str | None = None
    for trace in existing:
        recorded_x, recorded_y = _axis_semantics(trace)
        if recorded_x.get("quantity") != "frequency" or recorded_y != y_axis:
            raise ValueError(
                "target subplot already contains a trace with different axis semantics; use a separate axis"
            )
        unit = recorded_x.get("unit")
        if not isinstance(unit, str):
            raise ValueError("target frequency axis has no display-unit evidence")
        if selected_unit is not None and selected_unit != unit:
            raise ValueError("target frequency axis contains inconsistent display units")
        selected_unit = unit
    assert selected_unit is not None
    _display_axis(frequencies, selected_unit)
    return selected_unit


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


def scalar_plot(result: Any, *, theme: Theme = Theme.AUTO) -> Any:
    go, _, _ = _plotly()
    fields = (
        "root", "frequency", "linewidth", "slope", "value", "magnitude",
        "real", "imag", "zero", "numerator_slope", "denominator", "coupling",
        "branch_a_residue", "branch_b_residue", "family",
    )
    names = ["definition"]
    values = [type(result).__name__]
    for name in fields:
        value = getattr(result, name, None)
        if value is not None:
            names.append(name)
            values.append(escape(str(value)))
    identity = getattr(result, "identity", None)
    for name in ("plan_sha256", "request_sha256", "attempt_sha256", "result_sha256"):
        value = getattr(identity, name, None)
        if value is not None:
            names.append(name.removesuffix("_sha256"))
            values.append(escape(str(value)))
    presentation = getattr(result, "_presentation", {})
    if isinstance(presentation, Mapping):
        lineage = presentation.get("ref_lineage")
        if isinstance(lineage, Mapping) and isinstance(lineage.get("lineage_sha256"), str):
            names.append("view_lineage")
            values.append(escape(lineage["lineage_sha256"]))
        spec = presentation.get("spec")
        if isinstance(spec, Mapping) and isinstance(spec.get("type"), str):
            names.append("quantity_spec")
            values.append(escape(spec["type"]))
            if spec["type"] in {"diagonal_root", "operator_element_root"}:
                basis = lineage.get("terminal_coordinates") if isinstance(lineage, Mapping) else None
                if isinstance(basis, list):
                    names.append("final_view_basis")
                    values.append(escape(", ".join(str(item) for item in basis)))
                row = spec.get("coordinate") if spec["type"] == "diagonal_root" else spec.get("row")
                column = spec.get("coordinate") if spec["type"] == "diagonal_root" else spec.get("column")
                names.extend(("root_equation", "root_hint", "frequency_interpretation"))
                values.extend((
                    escape(f"F_View[{row}, {column}](omega) = 0"),
                    escape(_quantity_text(spec.get("root_hint"))),
                    "Re(omega) / (2 pi); slope = dF_View[row,column] / d omega",
                ))
    figure = go.Figure(data=[go.Table(header={"values": ["field", "value"]}, cells={"values": [names, values]})])
    figure.update_layout(meta={"scnsim": {"kind": "scalar_quantity", "fields": names}})
    return _style(figure, theme, title="Direct scalar quantity")


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


def _parameter_field_data(result: Any, *, x: Any, y: Any | None) -> dict[str, Any]:
    from .authoring import ParameterRef, ParameterSet

    if not isinstance(x, ParameterRef) or (y is not None and not isinstance(y, ParameterRef)):
        raise TypeError("x and y must be ParameterRef values")
    if y is x or (y is not None and y == x):
        raise ValueError("x and y must name distinct parameters")
    if not result._samples:
        raise ValueError("an empty ParameterField has no plottable samples")
    displayed = {x} if y is None else {x, y}
    varying: set[Any] = set()
    first = result._samples[0]["parameters"]
    if not isinstance(first, ParameterSet):
        raise TypeError("ParameterField sample parameters are malformed")
    from .results import _parameter_value_bytes

    for parameter in first.values:
        values = {
            _parameter_value_bytes(sample["parameters"].values[parameter], parameter)
            for sample in result._samples
        }
        if len(values) > 1:
            varying.add(parameter)
    if varying - displayed:
        raise ValueError("every varying non-displayed parameter must be fixed by selection")
    quantity_values = [sample["value"] for sample in result._samples if sample["value"] is not None]
    if any(
        not isinstance(value, Quantity) or np.asarray(value.magnitude).ndim != 0
        for value in quantity_values
    ):
        raise ValueError("ParameterField plot requires scalar quantity samples")
    unit = quantity_values[0].units if quantity_values else None
    values = np.asarray(
        [
            np.nan
            if sample["value"] is None
            else float(sample["value"].to(unit).magnitude)
            for sample in result._samples
        ]
    )
    x_values = np.asarray([
        float(sample["parameters"].values[x].to(x.spec.si_unit).magnitude)
        for sample in result._samples
    ])
    source_indices = [str(sample["source_index"]) for sample in result._samples]
    identities: list[str] = []
    for sample in result._samples:
        identity = getattr(sample["identity"], "parameters_sha256", None)
        if not isinstance(identity, str) or not identity:
            raise ValueError("ParameterField sample parameter identity is malformed")
        identities.append(identity)
    statuses = ["success" if sample["failure"] is None else "failure" for sample in result._samples]
    failures = [
        "—"
        if sample["failure"] is None
        else f"{getattr(sample['failure'], 'kind', type(sample['failure']).__name__)}: {sample['failure']}"
        for sample in result._samples
    ]
    data: dict[str, Any] = {
        "values": values,
        "unit": None if unit is None else str(unit),
        "x": x_values,
        "sample_x": x_values,
        "x_label": escape(f"{x.definitions_id}.{x.id} ({x.spec.si_unit})"),
        "x_axis": {
            "quantity": "parameter",
            "definitions_id": x.definitions_id,
            "parameter_id": x.id,
            "unit": str(x.spec.si_unit),
        },
        "source_indices": source_indices,
        "identities": identities,
        "statuses": statuses,
        "failures": failures,
        "failure_indices": [index for index, status in enumerate(statuses) if status == "failure"],
    }
    if y is not None:
        if result._kind != "grid":
            raise ValueError("listed or scattered parameter samples do not define a Cartesian heatmap")
        if x not in result._axis_parameters or y not in result._axis_parameters:
            raise ValueError("heatmap axes must be declared ParameterSpace grid axes")
        data["sample_y"] = np.asarray([
            float(sample["parameters"].values[y].to(y.spec.si_unit).magnitude)
            for sample in result._samples
        ])
    if not quantity_values:
        data["kind"] = "status"
        return data
    if y is None:
        data["kind"] = "line"
        data["y_axis"] = {
            "quantity": "parameter_field",
            "selector": str(result.quantity),
            "unit": str(unit),
        }
        return data
    y_values = data["sample_y"]
    xs = tuple(dict.fromkeys(x_values.tolist()))
    ys = tuple(dict.fromkeys(y_values.tolist()))
    cells: dict[tuple[float, float], int] = {}
    for index, cell in enumerate(zip(x_values, y_values, strict=True)):
        if cell in cells:
            raise ValueError("repeated parameter cells remain indexed and cannot form a grid")
        cells[cell] = index
    if len(cells) != len(xs) * len(ys):
        raise ValueError("listed, scattered, or incomplete samples cannot form a grid")
    grid = np.empty((len(ys), len(xs)))
    grid_custom = np.empty((len(ys), len(xs), 4), dtype=object)
    for row_index, y_value in enumerate(ys):
        for column, x_value in enumerate(xs):
            sample_index = cells[(x_value, y_value)]
            grid[row_index, column] = values[sample_index]
            grid_custom[row_index, column] = (
                statuses[sample_index],
                source_indices[sample_index],
                identities[sample_index],
                failures[sample_index],
            )
    data.update(
        kind="heatmap",
        y=np.asarray(ys),
        sample_y=y_values,
        y_label=escape(f"{y.definitions_id}.{y.id} ({y.spec.si_unit})"),
        y_axis={
            "quantity": "parameter",
            "definitions_id": y.definitions_id,
            "parameter_id": y.id,
            "unit": str(y.spec.si_unit),
            "value": {
                "quantity": "parameter_field",
                "selector": str(result.quantity),
                "unit": str(unit),
            },
        },
        x=np.asarray(xs),
        values=grid,
        customdata=grid_custom,
    )
    return data


def _parameter_field_trace(result: Any, *, data: Mapping[str, Any], y: Any | None) -> Any:
    go, _, _ = _plotly()
    if y is None:
        return go.Scatter(
            x=data["x"], y=data["values"], mode="lines+markers",
            name=escape(str(result.quantity)), connectgaps=False,
            meta=_trace_meta(
                x_axis=data["x_axis"],
                y_axis=data["y_axis"],
                source={
                    "samples": [
                        {"source_index": source, "parameters_sha256": identity, "status": status}
                        for source, identity, status in zip(
                            data["source_indices"], data["identities"], data["statuses"], strict=True
                        )
                    ]
                },
            ),
            customdata=np.asarray(
                list(zip(data["statuses"], data["source_indices"], data["identities"], data["failures"], strict=True)),
                dtype=object,
            ),
            hovertemplate=(
                f"{data['x_label']}=%{{x}}<br>value=%{{y}} {data['unit']}"
                "<br>status=%{customdata[0]}<br>source=%{customdata[1]}"
                "<br>parameters=%{customdata[2]}<br>%{customdata[3]}<extra></extra>"
            ),
        )
    return go.Heatmap(
        x=data["x"], y=data["y"], z=data["values"], connectgaps=False,
        colorbar={"title": data["unit"]},
        meta=_trace_meta(
            x_axis=data["x_axis"], y_axis=data["y_axis"],
            source={"samples": [
                {"source_index": source, "parameters_sha256": identity, "status": status}
                for source, identity, status in zip(
                    data["source_indices"], data["identities"], data["statuses"], strict=True
                )
            ]},
        ),
        customdata=data["customdata"],
        hovertemplate=(
            f"{data['x_label']}=%{{x}}<br>{data['y_label']}=%{{y}}"
            f"<br>value=%{{z}} {data['unit']}<br>status=%{{customdata[0]}}"
            "<br>source=%{customdata[1]}<br>parameters=%{customdata[2]}"
            "<br>%{customdata[3]}<extra></extra>"
        ),
    )


def _parameter_field_status_table(result: Any, data: Mapping[str, Any]) -> Any:
    go, _, _ = _plotly()
    bindings = [
        "; ".join(
            f"{parameter.definitions_id}.{parameter.id}={value}"
            for parameter, value in sample["parameters"].values.items()
        )
        for sample in result._samples
    ]
    return go.Table(
        header={"values": ["source", "status", "parameters identity", "parameters", "failure"]},
        cells={"values": [
            [escape(value) for value in data["source_indices"]],
            [escape(value) for value in data["statuses"]],
            [escape(value) for value in data["identities"]],
            [escape(value) for value in bindings],
            [escape(value) for value in data["failures"]],
        ]},
        meta={"scnsim": {"kind": "parameter_field_status", "quantity": str(result.quantity)}},
    )


def _add_parameter_failure_annotations(
    figure: Any,
    *,
    row: int,
    col: int,
    data: Mapping[str, Any],
    heatmap: bool,
) -> None:
    reference = figure._grid_ref[row - 1][col - 1][0]
    x_axis = reference.trace_kwargs.get("xaxis") or "x"
    y_axis = reference.trace_kwargs.get("yaxis") or "y"
    for index in data["failure_indices"]:
        source = escape(data["source_indices"][index])
        identity = escape(data["identities"][index])
        failure = escape(data["failures"][index])
        if heatmap:
            x_value = data["sample_x"][index]
            y_value = data["sample_y"][index]
            y_reference = y_axis
        else:
            x_value = data["sample_x"][index]
            y_value = 1.0
            y_reference = f"{y_axis} domain"
        figure.add_annotation(
            x=x_value,
            y=y_value,
            xref=x_axis,
            yref=y_reference,
            text=f"failure source={source}<br>parameters={identity}<br>{failure}",
            showarrow=False,
            font={"size": 9},
        )


def parameter_field_plot(result: Any, *, x: Any, y: Any | None = None, theme: Theme = Theme.AUTO) -> Any:
    go, _, make_subplots = _plotly()
    data = _parameter_field_data(result, x=x, y=y)
    if data["kind"] == "status":
        figure = go.Figure(data=[_parameter_field_status_table(result, data)])
    elif data["failure_indices"]:
        figure = make_subplots(
            rows=2,
            cols=1,
            specs=[[{"type": "xy"}], [{"type": "domain"}]],
            row_heights=[0.68, 0.32],
            subplot_titles=("quantity", "sample status and identity"),
        )
        figure.add_trace(_parameter_field_trace(result, data=data, y=y), row=1, col=1)
        figure.add_trace(_parameter_field_status_table(result, data), row=2, col=1)
        figure.update_xaxes(title_text=data["x_label"], row=1, col=1)
        figure.update_yaxes(
            title_text=data["unit"] if y is None else data["y_label"], row=1, col=1
        )
    else:
        figure = go.Figure(data=[_parameter_field_trace(result, data=data, y=y)])
        figure.update_xaxes(title_text=data["x_label"])
        figure.update_yaxes(title_text=data["unit"] if y is None else data["y_label"])
    figure.update_layout(meta={"scnsim": {"kind": "parameter_field", "quantity": str(result.quantity)}})
    return _style(figure, theme, title="Parameter sweep field")


def parameter_field_add_to(result: Any, figure: Any, *, row: int, col: int, x: Any, y: Any | None = None) -> Any:
    go, _, _ = _plotly()
    if not isinstance(figure, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")
    data = _parameter_field_data(result, x=x, y=y)
    expected = "domain" if data["kind"] == "status" else "xy"
    if _subplot_type(figure, row, col) != expected:
        raise ValueError(f"parameter field requires a compatible {expected} subplot")
    if data["kind"] == "status":
        figure.add_trace(_parameter_field_status_table(result, data), row=row, col=col)
        return figure
    trace = _parameter_field_trace(result, data=data, y=y)
    _require_axis_semantics(
        figure,
        row,
        col,
        x_axis=trace.meta["scnsim"]["x_axis"],
        y_axis=trace.meta["scnsim"]["y_axis"],
    )
    figure.add_trace(trace, row=row, col=col)
    _add_parameter_failure_annotations(
        figure, row=row, col=col, data=data, heatmap=y is not None
    )
    return figure


def points_plot(points: Sequence[Any], *, theme: Theme = Theme.AUTO, title: str = "Parameter sweep outcomes") -> Any:
    from html import escape

    go, _, _ = _plotly()
    source_index: list[str] = []
    status: list[str] = []
    parameters: list[str] = []
    failure: list[str] = []
    for point in points:
        source_index.append(escape(str(point.source_index)))
        status.append("success" if point.succeeded else "failure")
        parameters.append(escape(point.identity.parameters_sha256))
        failure.append("—" if point.succeeded else escape(f"{point.failure.kind}: {point.failure}"))
    figure = go.Figure(data=[go.Table(
        header={"values": ["source index", "status", "parameters", "failure"]},
        cells={"values": [source_index, status, parameters, failure]},
    )])
    figure.update_layout(meta={"scnsim": {"kind": "parameter_sweep_outcomes", "point_count": len(points)}})
    return _style(figure, theme, title=title)


def points_add_to(points: Sequence[Any], figure: Any, *, row: int, col: int) -> Any:
    go, _, _ = _plotly()
    if not isinstance(figure, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")
    if _subplot_type(figure, row, col) != "domain":
        raise ValueError("parameter outcome table requires a compatible domain subplot")
    figure.add_trace(points_plot(points).data[0], row=row, col=col)
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


def _quantity_text(value: object) -> str:
    from ._canonical import float64_from_hex

    if not isinstance(value, Mapping):
        return "—"
    encoded = value.get("si_value_f64")
    unit = value.get("si_unit")
    if not isinstance(encoded, str) or not isinstance(unit, str):
        return "—"
    return f"{float64_from_hex(encoded):.12g} {escape(unit)}"


def _ledger_rows(result: Any) -> tuple[list[int], list[float | None], list[str], list[str]]:
    from ._canonical import float64_from_hex

    ordinals: list[int] = []
    costs: list[float | None] = []
    statuses: list[str] = []
    evidence: list[str] = []
    for ledger in result.ledger:
        candidates = ledger.get("candidates") if isinstance(ledger, Mapping) else None
        if not isinstance(candidates, Sequence):
            raise ValueError("Optimization ledger candidates are malformed")
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not isinstance(candidate.get("outcome"), Mapping):
                raise ValueError("Optimization candidate evidence is malformed")
            outcome = candidate["outcome"]
            status = str(outcome.get("status"))
            ordinal = candidate.get("evaluation_ordinal")
            if not isinstance(ordinal, int):
                raise ValueError("Optimization candidate ordinal is malformed")
            cost_hex = outcome.get("cost_f64")
            cost = float64_from_hex(cost_hex) if isinstance(cost_hex, str) else None
            components = outcome.get("objective_components")
            if not isinstance(components, Sequence):
                raise ValueError("Optimization objective evidence is malformed")
            summary = "; ".join(
                escape(
                    f"{component.get('objective_id')}: {component.get('status')} "
                    f"({len(component.get('terms', ())) if isinstance(component, Mapping) else 0} terms)"
                )
                for component in components
                if isinstance(component, Mapping)
            )
            ordinals.append(ordinal)
            costs.append(cost)
            statuses.append(status)
            evidence.append(summary)
    return ordinals, costs, statuses, evidence


def _optimization_series(
    result: Any,
    *,
    kind: Literal["history", "objective", "residual", "parameter"],
    objective: str | None,
    parameter: Any | None,
) -> tuple[list[int], list[float | None], list[str], str, str]:
    from ._canonical import float64_from_hex
    from .authoring import ParameterRef

    if kind in {"objective", "residual"}:
        if not isinstance(objective, str) or not objective:
            raise ValueError(f"kind={kind!r} requires objective=<objective ID>")
        if parameter is not None:
            raise ValueError("parameter is valid only for kind='parameter'")
    elif kind == "parameter":
        if not isinstance(parameter, ParameterRef):
            raise TypeError("kind='parameter' requires parameter=ParameterRef")
        if objective is not None:
            raise ValueError("objective is valid only for objective or residual history")
        retained = next(
            (
                item
                for item in result.best.parameters.values
                if item.definitions_id == parameter.definitions_id
                and item.id == parameter.id
            ),
            None,
        )
        if retained is None:
            raise ValueError(
                "selected parameter history is absent from the Optimization Result: "
                f"{parameter.definitions_id}.{parameter.id}"
            )
        if retained._definition_record() != parameter._definition_record():
            raise ValueError(
                "selected parameter definition disagrees with the Optimization Result"
            )
        parameter = retained
    elif objective is not None or parameter is not None:
        raise ValueError("objective and parameter are invalid for total-cost history")

    ordinals: list[int] = []
    values: list[float | None] = []
    statuses: list[str] = []
    found = False
    unit = "dimensionless"
    for ledger in result.ledger:
        candidates = ledger.get("candidates") if isinstance(ledger, Mapping) else None
        if not isinstance(candidates, Sequence):
            raise ValueError("Optimization ledger candidates are malformed")
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not isinstance(candidate.get("outcome"), Mapping):
                raise ValueError("Optimization candidate evidence is malformed")
            outcome = candidate["outcome"]
            ordinal = candidate.get("evaluation_ordinal")
            if not isinstance(ordinal, int):
                raise ValueError("Optimization candidate ordinal is malformed")
            value: float | None = None
            status = str(outcome.get("status", ""))
            if kind == "history":
                encoded = outcome.get("cost_f64")
                value = float64_from_hex(encoded) if isinstance(encoded, str) else None
            elif kind in {"objective", "residual"}:
                components = outcome.get("objective_components")
                if not isinstance(components, Sequence):
                    raise ValueError("Optimization objective evidence is malformed")
                component = next(
                    (
                        item for item in components
                        if isinstance(item, Mapping) and item.get("objective_id") == objective
                    ),
                    None,
                )
                if component is not None:
                    found = True
                    status = str(component.get("status", status))
                    field = "weighted_cost_f64" if kind == "objective" else "normalized_residual_f64"
                    encoded = component.get(field)
                    value = float64_from_hex(encoded) if isinstance(encoded, str) else None
            else:
                bindings = candidate.get("parameters", {}).get("bindings") if isinstance(candidate.get("parameters"), Mapping) else None
                if not isinstance(bindings, Sequence):
                    raise ValueError("Optimization candidate parameters are malformed")
                key = {"definitions_id": parameter.definitions_id, "parameter_id": parameter.id}
                binding = next(
                    (item for item in bindings if isinstance(item, Mapping) and item.get("parameter") == key),
                    None,
                )
                if binding is not None and isinstance(binding.get("value"), Mapping):
                    found = True
                    envelope = binding["value"]
                    encoded = envelope.get("si_value_f64")
                    stored_unit = envelope.get("si_unit")
                    if isinstance(encoded, str) and isinstance(stored_unit, str):
                        quantity = Quantity(float64_from_hex(encoded), stored_unit).to(
                            parameter.spec.si_unit
                        )
                        value = float(quantity.magnitude)
                        unit = str(quantity.units)
            ordinals.append(ordinal)
            values.append(value)
            statuses.append(status)
    if kind in {"objective", "residual", "parameter"} and not found:
        selected = objective if parameter is None else f"{parameter.definitions_id}.{parameter.id}"
        raise ValueError(f"selected {kind} history is absent from the completed ledger: {selected}")
    if kind == "history":
        label = "total cost"
    elif kind == "objective":
        label = f"{objective} weighted cost"
    elif kind == "residual":
        label = f"{objective} normalized residual"
    else:
        label = f"{parameter.definitions_id}.{parameter.id}"
    return ordinals, values, statuses, label, unit


def _optimization_term_rows(result: Any) -> tuple[list[object], ...]:
    from ._canonical import float64_from_hex

    columns: tuple[list[object], ...] = tuple([] for _ in range(13))
    for ledger in result.ledger:
        candidates = ledger.get("candidates") if isinstance(ledger, Mapping) else None
        if not isinstance(candidates, Sequence):
            raise ValueError("Optimization ledger candidates are malformed")
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not isinstance(candidate.get("outcome"), Mapping):
                raise ValueError("Optimization candidate evidence is malformed")
            outcome = candidate["outcome"]
            ordinal = candidate.get("evaluation_ordinal")
            components = outcome.get("objective_components")
            if not isinstance(ordinal, int) or not isinstance(components, Sequence):
                raise ValueError("Optimization candidate evidence is malformed")
            candidate_cost = outcome.get("cost_f64")
            cost = f"{float64_from_hex(candidate_cost):.12g}" if isinstance(candidate_cost, str) else "—"
            for component in components:
                if not isinstance(component, Mapping) or not isinstance(component.get("terms"), Sequence):
                    raise ValueError("Optimization objective evidence is malformed")
                objective = escape(str(component.get("objective_id", "")))
                component_status = escape(str(component.get("status", "")))
                normalized = component.get("normalized_residual_f64")
                residual = (
                    f"{float64_from_hex(normalized):.12g}"
                    if isinstance(normalized, str) else "—"
                )
                weighted = component.get("weighted_cost_f64")
                weighted_cost = (
                    f"{float64_from_hex(weighted):.12g}"
                    if isinstance(weighted, str) else "—"
                )
                for term in component["terms"]:
                    if not isinstance(term, Mapping):
                        raise ValueError("Optimization term evidence is malformed")
                    lineage = term.get("ref_lineage")
                    lineage_id = (
                        lineage.get("lineage_sha256") if isinstance(lineage, Mapping) else None
                    )
                    failure = term.get("failure")
                    failure_text = "—"
                    if isinstance(failure, Mapping):
                        failure_text = escape(
                            f"{failure.get('kind', '')} / {failure.get('stage', '')}: "
                            f"{failure.get('message', '')}"
                        )
                    row = (
                        ordinal,
                        escape(str(outcome.get("status", ""))),
                        cost,
                        objective,
                        component_status,
                        _quantity_text(component.get("value")),
                        residual,
                        weighted_cost,
                        term.get("term_ordinal", ""),
                        escape(str(term.get("status", ""))),
                        _quantity_text(term.get("value")),
                        escape(str(lineage_id)) if lineage_id is not None else "—",
                        failure_text,
                    )
                    for column, value in zip(columns, row, strict=True):
                        column.append(value)
    return columns


def optimization_plot(
    result: Any,
    *,
    kind: Literal["history", "objective", "residual", "parameter", "table", "comparison"] = "history",
    objective: str | None = None,
    parameter: Any | None = None,
    theme: Theme = Theme.AUTO,
) -> Any:
    go, _, _ = _plotly()
    all_ordinals, _, _, evidence = _ledger_rows(result)
    saved = f"saved best cost {result.best.cost:.12g}"
    if kind in {"history", "objective", "residual", "parameter"}:
        ordinals, values, statuses, label, unit = _optimization_series(
            result, kind=kind, objective=objective, parameter=parameter
        )
        figure = go.Figure(data=[go.Scatter(
            x=ordinals,
            y=values,
            mode="lines+markers",
            name=escape(label),
            customdata=np.asarray([[status, detail] for status, detail in zip(statuses, evidence, strict=True)], dtype=object),
            connectgaps=False,
            meta=_trace_meta(
                x_axis={"quantity": "evaluation_ordinal", "unit": "ordinal"},
                y_axis={
                    "quantity": "optimization_history",
                    "kind": kind,
                    "objective": objective,
                    "parameter": (
                        None if parameter is None
                        else f"{parameter.definitions_id}.{parameter.id}"
                    ),
                    "unit": unit,
                },
                source=_presentation_source(result),
            ),
            hovertemplate="evaluation=%{x}<br>value=%{y}<br>status=%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        )])
        figure.update_xaxes(title_text="evaluation ordinal")
        figure.update_yaxes(title_text=f"{escape(label)} ({escape(unit)})")
    elif kind == "table":
        if objective is not None or parameter is not None:
            raise ValueError("objective and parameter are invalid for the ledger table")
        columns = _optimization_term_rows(result)
        figure = go.Figure(data=[go.Table(
            header={"values": [
                "evaluation", "candidate status", "cost", "objective", "objective status",
                "objective value", "normalized residual", "weighted cost", "term",
                "term status", "term value", "View lineage", "failure",
            ]},
            cells={"values": columns},
            meta={"scnsim": {"source": _presentation_source(result)}},
        )])
    elif kind == "comparison":
        if objective is not None or parameter is not None:
            raise ValueError("objective and parameter are invalid for the comparison")
        settings, comparison = optimization_comparison_dataset(result)
        figure = go.Figure(data=[
            go.Table(
                domain={"y": [0.71, 1.0]},
                header={"values": ["setting", "value"]},
                cells={"values": list(zip(*settings, strict=True))},
                meta={"scnsim": {"source": _presentation_source(result), "section": "settings"}},
            ),
            go.Table(
                domain={"y": [0.0, 0.65]},
                header={"values": ["field", "initial", "best found"]},
                cells={"values": list(zip(*comparison, strict=True))},
                meta={"scnsim": {"source": _presentation_source(result), "section": "comparison"}},
            ),
        ])
        figure.add_annotation(
            text="Settings", x=0, y=1.04, xref="paper", yref="paper",
            xanchor="left", showarrow=False,
        )
        figure.add_annotation(
            text="Initial / best-found comparison", x=0, y=0.69,
            xref="paper", yref="paper", xanchor="left", showarrow=False,
        )
    else:
        raise ValueError("kind must be 'history', 'objective', 'residual', 'parameter', 'table', or 'comparison'")
    winner_parameters = {
        f"{parameter.definitions_id}.{parameter.id}": str(value)
        for parameter, value in result.best.parameters.values.items()
    }
    figure.update_layout(meta={"scnsim": {
        "kind": "optimization",
        "candidate_count": len(all_ordinals),
        "winner_cost": result.best.cost,
        "winner_parameters": winner_parameters,
    }})
    best_bindings = "; ".join(
        f"{escape(parameter.definitions_id)}.{escape(parameter.id)} = {escape(str(value))}"
        for parameter, value in result.best.parameters.values.items()
    )
    figure.add_annotation(
        text=f"{saved}; {best_bindings}", x=0, y=1.12,
        xref="paper", yref="paper", xanchor="left", showarrow=False,
    )
    styled = _style(figure, theme, title=f"Completed optimization {kind}")
    styled.update_layout(margin={"t": 112})
    return styled


def optimization_add_to(
    result: Any,
    figure: Any,
    *,
    row: int,
    col: int,
    kind: Literal["history", "objective", "residual", "parameter", "table", "comparison"],
    objective: str | None = None,
    parameter: Any | None = None,
) -> Any:
    go, _, _ = _plotly()
    if not isinstance(figure, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")
    expected = "domain" if kind in {"table", "comparison"} else "xy"
    if _subplot_type(figure, row, col) != expected:
        raise ValueError(f"optimization {kind} requires a compatible {expected} subplot")
    if kind == "comparison":
        if objective is not None or parameter is not None:
            raise ValueError("objective and parameter are invalid for the comparison")
        settings, comparison = optimization_comparison_dataset(result)
        rows = [
            ("settings", field, value, "", "")
            for field, value in settings
        ] + [
            ("comparison", field, "", initial, best)
            for field, initial, best in comparison
        ]
        figure.add_trace(go.Table(
            header={"values": ["section", "field", "value", "initial", "best found"]},
            cells={"values": list(zip(*rows, strict=True))},
            meta={"scnsim": {"source": _presentation_source(result)}},
        ), row=row, col=col)
        return figure
    source = optimization_plot(
        result, kind=kind, objective=objective, parameter=parameter
    )
    if kind not in {"table", "comparison"}:
        trace = source.data[0]
        _require_axis_semantics(
            figure,
            row,
            col,
            x_axis=trace.meta["scnsim"]["x_axis"],
            y_axis=trace.meta["scnsim"]["y_axis"],
        )
    figure.add_trace(source.data[0], row=row, col=col)
    return figure


def optimization_comparison_dataset(
    result: Any,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Return shared settings and initial/best data without execution."""

    from hashlib import sha256

    from ._canonical import canonical_json_bytes, float64_from_hex

    def expression_text(value: object) -> str:
        if not isinstance(value, Mapping):
            raise ValueError("Optimization expression presentation is malformed")
        kind = value.get("type")
        if kind == "quantity_sum":
            return "(" + " + ".join(expression_text(item) for item in value["terms"]) + ")"
        if kind == "quantity_difference":
            return f"({expression_text(value['left'])} - {expression_text(value['right'])})"
        if kind == "quantity_absolute":
            return f"abs({expression_text(value['operand'])})"
        view = value.get("view")
        if not isinstance(view, Mapping):
            raise ValueError("Optimization selector View presentation is malformed")
        ptc = view.get("ptc")
        transforms = view.get("transforms")
        retain = view.get("retain")
        if not isinstance(transforms, Sequence) or isinstance(transforms, (str, bytes)):
            raise ValueError("Optimization selector View transforms are malformed")
        if ptc is None:
            ptc_text = "raw"
        elif isinstance(ptc, Mapping):
            ptc_text = f"PTC[{','.join(ptc.get('selected_ports', ())) }]"
        else:
            raise ValueError("Optimization selector View PTC is malformed")
        transform_text = ",".join(str(item.get("id")) for item in transforms)
        if retain is None:
            retain_text = "all"
        elif isinstance(retain, Mapping):
            retain_text = ",".join(retain.get("retained_coordinates", ()))
        else:
            raise ValueError("Optimization selector View retain is malformed")
        view_sha256 = sha256(canonical_json_bytes(dict(view))).hexdigest()
        view_text = (
            f"{ptc_text}; transform=[{transform_text}]; retain=[{retain_text}]; "
            f"view_sha256={view_sha256}"
        )
        spec = value.get("spec")
        if not isinstance(spec, Mapping):
            raise ValueError("Optimization selector specification is malformed")
        spec_type = spec.get("type")
        def spec_text(record: Mapping[str, object]) -> str:
            record_type = record.get("type")
            if record_type == "diagonal_root":
                return (
                    f"coordinate={record.get('coordinate')}; "
                    f"root_hint={_quantity_text(record.get('root_hint'))}"
                )
            if record_type == "operator_element_root":
                return (
                    f"row={record.get('row')}; column={record.get('column')}; "
                    f"root_hint={_quantity_text(record.get('root_hint'))}"
                )
            if record_type == "hybridized_pole":
                return (
                    f"coordinates={','.join(str(item) for item in record.get('coordinates', ()))}; "
                    f"anchor={_quantity_text(record.get('anchor'))}"
                )
            return str(record_type)

        if spec_type == "diagonal_root":
            selection = (
                f"coordinate={spec.get('coordinate')}; "
                f"root_hint={_quantity_text(spec.get('root_hint'))}"
            )
        elif spec_type == "operator_element_root":
            selection = (
                f"row={spec.get('row')}; column={spec.get('column')}; "
                f"root_hint={_quantity_text(spec.get('root_hint'))}"
            )
        elif spec_type == "hybridized_pole":
            selection = (
                f"coordinates={','.join(str(item) for item in spec.get('coordinates', ()))}; "
                f"anchor={_quantity_text(spec.get('anchor'))}"
            )
        elif spec_type == "transfer_zero":
            selection = (
                f"family={spec.get('family')}; input={spec.get('input_coordinate')}; "
                f"output={spec.get('output_coordinate')}; anchor={_quantity_text(spec.get('anchor'))}"
            )
        elif spec_type == "response_element":
            selection = (
                f"family={spec.get('family')}; input={spec.get('input_coordinate')}; "
                f"output={spec.get('output_coordinate')}; frequency={_quantity_text(spec.get('frequency'))}"
            )
        elif spec_type == "residue_normalized_coupling":
            branch_a, branch_b = spec.get("branch_a"), spec.get("branch_b")
            if not isinstance(branch_a, Mapping) or not isinstance(branch_b, Mapping):
                raise ValueError("Optimization residue branch presentation is malformed")
            selection = (
                f"branch_a=({spec_text(branch_a)}); branch_b=({spec_text(branch_b)}); "
                f"frequency={_quantity_text(spec.get('frequency'))}"
            )
        else:
            selection = str(spec_type)
        return f"{kind}.{value.get('projection')} ({selection}) @ {view_text}"

    def bounds_text(value: object) -> str:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
        ):
            raise ValueError("Optimization bounds presentation is malformed")
        return f"[{_quantity_text(value[0])}, {_quantity_text(value[1])}]"

    presentation = result._presentation
    initial = presentation.get("initial_candidate")
    ordinal = presentation.get("best_evaluation_ordinal")
    objectives = presentation.get("objectives")
    variables = presentation.get("variables")
    if (
        not isinstance(initial, Mapping)
        or not isinstance(ordinal, int)
        or not isinstance(objectives, Sequence)
        or not isinstance(variables, Sequence)
    ):
        raise ValueError("Optimization comparison presentation is incomplete")
    best = initial if ordinal == 0 else None
    if best is None:
        for ledger in result.ledger:
            candidates = ledger.get("candidates") if isinstance(ledger, Mapping) else None
            if isinstance(candidates, Sequence):
                best = next(
                    (
                        candidate
                        for candidate in candidates
                        if isinstance(candidate, Mapping)
                        and candidate.get("evaluation_ordinal") == ordinal
                    ),
                    None,
                )
            if best is not None:
                break
    if not isinstance(best, Mapping):
        raise ValueError("Saved best evaluation ordinal is absent from verified evidence")
    initial_outcome, best_outcome = initial.get("outcome"), best.get("outcome")
    if not isinstance(initial_outcome, Mapping) or not isinstance(best_outcome, Mapping):
        raise ValueError("Optimization comparison candidates have no outcome")

    settings: list[tuple[str, str]] = []
    comparison: list[tuple[str, str, str]] = []
    for variable in variables:
        if not isinstance(variable, Mapping):
            raise ValueError("Optimization variable presentation is malformed")
        ref = variable.get("parameter")
        if not isinstance(ref, Mapping):
            raise ValueError("Optimization variable identity is malformed")
        name = f"{ref.get('definitions_id')}.{ref.get('parameter_id')}"
        defaults = variable.get("model_default_bounds")
        override = variable.get("consumer_override_bounds")
        settings.extend((
            (f"{name} transform", str(variable.get("transform"))),
            (f"{name} model-default bounds", bounds_text(defaults)),
            (f"{name} resolved bounds", bounds_text(override if override is not None else defaults)),
        ))
    for objective in objectives:
        if not isinstance(objective, Mapping):
            raise ValueError("Optimization objective presentation is malformed")
        name = f"objective {objective.get('id')}"
        settings.extend((
            (f"{name} expression / View", expression_text(objective.get("quantity"))),
            (f"{name} comparison / target", f"{objective.get('comparison')} / {_quantity_text(objective.get('target'))}"),
            (f"{name} scale / weight", f"{_quantity_text(objective.get('resolved_scale'))} ({objective.get('scale_source')}) / {float64_from_hex(objective['weight_f64']):.12g}"),
        ))

    initial_parameters = presentation.get("initial_parameters")
    if initial_parameters is None:
        raise ValueError("Optimization initial ParameterSet is absent")
    parameter_keys = list(result.best.parameters.values)
    initial_by_key = {
        (item.definitions_id, item.id): value
        for item, value in initial_parameters.values.items()
    }
    for parameter in parameter_keys:
        key = (parameter.definitions_id, parameter.id)
        if key not in initial_by_key:
            raise ValueError("Optimization initial ParameterSet lacks a best-point parameter")
        initial_parameter = initial_by_key[key]
        best_parameter = result.best.parameters.values[parameter]
        comparison.append((
            f"{parameter.definitions_id}.{parameter.id} ({'unchanged' if str(initial_parameter) == str(best_parameter) else 'physical value changed'})",
            str(initial_parameter),
            str(best_parameter),
        ))
    comparison.append((
        "evaluation ordinal",
        str(initial.get("evaluation_ordinal")),
        str(ordinal),
    ))
    comparison.append((
        "total cost",
        f"{float64_from_hex(initial_outcome['cost_f64']):.12g}",
        f"{result.best.cost:.12g}",
    ))
    comparison.append((
        "total cost delta (best - initial)",
        "—",
        f"{result.best.cost - float64_from_hex(initial_outcome['cost_f64']):.12g}",
    ))
    initial_components = initial_outcome.get("objective_components")
    best_components = best_outcome.get("objective_components")
    if not isinstance(initial_components, Sequence) or not isinstance(best_components, Sequence):
        raise ValueError("Optimization objective comparison evidence is absent")
    for objective, left, right in zip(objectives, initial_components, best_components, strict=True):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError("Optimization objective comparison evidence is malformed")
        name = str(objective.get("id"))
        initial_value = float64_from_hex(left["value"]["si_value_f64"])
        best_value = float64_from_hex(right["value"]["si_value_f64"])
        target_value = float64_from_hex(objective["target"]["si_value_f64"])
        initial_cost = float64_from_hex(left["weighted_cost_f64"])
        best_cost = float64_from_hex(right["weighted_cost_f64"])
        comparison.extend((
            (f"{name} value", _quantity_text(left.get("value")), _quantity_text(right.get("value"))),
            (f"{name} normalized residual", f"{float64_from_hex(left['normalized_residual_f64']):.12g}", f"{float64_from_hex(right['normalized_residual_f64']):.12g}"),
            (f"{name} weighted cost", f"{initial_cost:.12g}", f"{best_cost:.12g}"),
            (f"{name} weighted-cost delta (best - initial)", "—", f"{best_cost - initial_cost:.12g}"),
        ))
        if objective.get("comparison") == "at_least":
            comparison.append((
                f"{name} at-least condition",
                "met" if initial_value >= target_value else "below target",
                "met" if best_value >= target_value else "below target",
            ))
    return settings, comparison


def optimization_comparison_rows(result: Any) -> list[tuple[str, str, str, str]]:
    """Return neutral combined rows for report and domain-table consumers."""

    settings, comparison = optimization_comparison_dataset(result)
    return [
        ("settings", field, value, "") for field, value in settings
    ] + [
        ("comparison", field, initial, best)
        for field, initial, best in comparison
    ]


def _hb_status_table(cases: Mapping[str, Any]) -> Any:
    go, _, _ = _plotly()
    identifiers: list[str] = []
    statuses: list[str] = []
    bias: list[str] = []
    pump: list[str] = []
    failures: list[str] = []
    for identifier, outcome in cases.items():
        identifiers.append(escape(identifier))
        statuses.append("success" if outcome.succeeded else "failure")
        bias.append(escape(outcome.bias_state.value) if outcome.succeeded else "—")
        pump.append(escape(outcome.pump_state.value) if outcome.succeeded else "—")
        failures.append(
            "—" if outcome.succeeded else escape(
                f"{outcome.failure.kind} / {outcome.failure.stage}: {outcome.failure}"
            )
        )
    return go.Table(
        header={"values": ["case", "status", "bias", "pump", "failure"]},
        cells={"values": [identifiers, statuses, bias, pump, failures]},
    )


def hb_case_plot(outcome: Any, **presentation: Any) -> Any:
    theme = presentation.pop("theme", Theme.AUTO)
    if not outcome.succeeded:
        if presentation:
            raise ValueError("failed HB case presentation accepts only theme")
        go, _, _ = _plotly()
        figure = go.Figure(data=[_hb_status_table({outcome.id: outcome})])
        figure.update_layout(meta={"scnsim": {"kind": "hb_case", "case_id": outcome.id, "status": "failure"}})
        return _style(figure, theme, title=f"HB case {escape(outcome.id)}")
    return matrix_plot(outcome.s, theme=theme, **presentation)


def hb_case_add_to(outcome: Any, figure: Any, *, row: int, col: int, **presentation: Any) -> Any:
    go, _, _ = _plotly()
    if not isinstance(figure, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")
    if not outcome.succeeded:
        if presentation:
            raise ValueError("failed HB case table accepts no matrix arguments")
        if _subplot_type(figure, row, col) != "domain":
            raise ValueError("failed HB case table requires a compatible domain subplot")
        figure.add_trace(_hb_status_table({outcome.id: outcome}), row=row, col=col)
        return figure
    return matrix_add_to(outcome.s, figure, row=row, col=col, **presentation)


def hb_batch_plot(
    result: Any,
    *,
    trace: str | None = None,
    input_channel: Channel | None = None,
    output_channel: Channel | None = None,
    component: Component | None = None,
    magnitude: Literal["linear", "db"] = "linear",
    theme: Theme = Theme.AUTO,
) -> Any:
    successes = tuple(outcome for outcome in result.cases.values() if outcome.succeeded)
    failures = tuple(outcome for outcome in result.cases.values() if not outcome.succeeded)
    go, _, make_subplots = _plotly()
    if not successes:
        if trace is not None or input_channel is not None or output_channel is not None or component is not None or magnitude != "linear":
            raise ValueError("all-failed HB batch accepts only its status presentation")
        figure = go.Figure(data=[_hb_status_table(result.cases)])
        figure.update_layout(meta={"scnsim": {"kind": "hb_batch", "successful_cases": [], "failed_cases": list(result.cases)}})
        return _style(figure, theme, title="HB batch outcomes")

    declared = tuple(successes[0].traces)
    if any(tuple(outcome.traces) != declared for outcome in successes):
        raise ValueError("successful HB cases have incompatible named-trace inventories")
    if trace is not None:
        if input_channel is not None or output_channel is not None:
            raise ValueError("trace and matrix-channel selection are mutually exclusive")
        if trace not in declared:
            raise ValueError(f"unknown HB trace: {trace}")
        panels: tuple[tuple[str, str | None], ...] = ((trace, trace),)
    elif input_channel is not None or output_channel is not None:
        if input_channel is None or output_channel is None:
            raise ValueError("input_channel and output_channel must be supplied together")
        panels = (("selected S channel", None),)
    elif declared:
        panels = tuple((identifier, identifier) for identifier in declared)
    else:
        panels = (("selected S channel", None),)
    components: tuple[Component, ...] = ("magnitude", "phase") if component is None else (component,)
    if component is not None:
        _component_values(np.asarray([1 + 0j]), component, family="S", magnitude=magnitude)
    xy_rows = len(panels) * len(components)
    specs = [[{"type": "xy"}] for _ in range(xy_rows)]
    titles = [f"{label} {selected}" for label, _ in panels for selected in components]
    if failures:
        specs.append([{"type": "domain"}])
        titles.append("case status")
    figure = make_subplots(rows=len(specs), cols=1, shared_xaxes=False, specs=specs, subplot_titles=titles)
    first_frequencies = (
        successes[0].traces[panels[0][1]].frequencies
        if panels[0][1] is not None
        else successes[0].s.view.frequencies
    )
    _, batch_frequency_unit = _display_axis(first_frequencies)
    row = 1
    for _, identifier in panels:
        for selected in components:
            panel_values: list[np.ndarray] = []
            for outcome in successes:
                selected_magnitude = magnitude if selected == "magnitude" else "linear"
                if identifier is not None:
                    trace_result = outcome.traces[identifier]
                    values, _ = _component_values(
                        np.asarray(trace_result.value.magnitude), selected,
                        family="S", magnitude=selected_magnitude,
                    )
                    frequencies, frequency_unit = _display_axis(
                        trace_result.frequencies, batch_frequency_unit
                    )
                    label = escape(f"{outcome.id}: {identifier}")
                    source = _presentation_source(
                        trace_result, hb_case=outcome.id, named_trace=identifier
                    )
                else:
                    view = outcome.s.view
                    input_index = _channel_index(view.input_channels, input_channel, role="input", default=True)
                    output_index = _channel_index(view.output_channels, output_channel, role="output", default=True)
                    values, _ = _component_values(
                        _matrix_values(view)[:, output_index, input_index], selected,
                        family="S", magnitude=selected_magnitude,
                    )
                    frequencies, frequency_unit = _display_axis(
                        view.frequencies, batch_frequency_unit
                    )
                    label = escape(f"{outcome.id}: ") + (
                        f"S[{_channel_label(view.output_channels[output_index])}"
                        f" <- {_channel_label(view.input_channels[input_index])}]"
                    )
                    source = _presentation_source(
                        outcome.s,
                        hb_case=outcome.id,
                        input_channel=_channel_record(view.input_channels[input_index]),
                        output_channel=_channel_record(view.output_channels[output_index]),
                    )
                unit = "dB" if selected == "magnitude" and magnitude == "db" else (
                    "degree (exact zero undefined)" if selected == "phase" else "dimensionless"
                )
                figure.add_trace(
                    go.Scatter(
                        x=np.array(frequencies, copy=True), y=np.array(values, copy=True),
                        mode="lines", name=label,
                        meta=_trace_meta(
                            x_axis=_frequency_axis(frequency_unit),
                            y_axis=_response_axis(
                                family="S",
                                component=selected,
                                magnitude=selected_magnitude,
                                unit=unit,
                            ),
                            source=source,
                        ),
                        hovertemplate=f"frequency=%{{x}} {frequency_unit}<br>value=%{{y}} {unit}<extra>%{{fullData.name}}</extra>",
                    ),
                    row=row,
                    col=1,
                )
                panel_values.append(np.asarray(values))
            figure.update_yaxes(title_text=unit, row=row, col=1)
            figure.update_xaxes(title_text=f"frequency ({frequency_unit})", row=row, col=1)
            if selected == "magnitude" and magnitude == "db":
                _set_db_viewport(figure, np.concatenate(panel_values), row=row)
            row += 1
    if failures:
        figure.add_trace(_hb_status_table(result.cases), row=row, col=1)
    figure.update_layout(meta={"scnsim": {
        "kind": "hb_batch",
        "successful_cases": [outcome.id for outcome in successes],
        "failed_cases": [outcome.id for outcome in failures],
        "named_traces": list(declared),
    }})
    return _style(figure, theme, title="HB batch selected response")


def hb_outcomes_plot(
    result: Any,
    *,
    cases: Mapping[str, Any] | None = None,
    theme: Theme = Theme.AUTO,
) -> Any:
    """Present every declared HB case without selecting scientific payloads."""

    selected = result.cases if cases is None else cases
    go, _, _ = _plotly()
    figure = go.Figure(data=[_hb_status_table(selected)])
    figure.update_layout(meta={"scnsim": {
        "kind": "hb_batch_outcomes",
        "cases": [
            {"id": outcome.id, "status": "success" if outcome.succeeded else "failure"}
            for outcome in selected.values()
        ],
        "source": _presentation_source(result),
    }})
    return _style(figure, theme, title="HB batch outcomes")


def hb_batch_add_to(
    result: Any,
    figure: Any,
    *,
    row: int,
    col: int,
    kind: Literal["trace", "status"],
    trace: str | None = None,
    input_channel: Channel | None = None,
    output_channel: Channel | None = None,
    component: Component | None = None,
    magnitude: Literal["linear", "db"] = "linear",
) -> Any:
    go, _, _ = _plotly()
    if not isinstance(figure, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")
    if kind == "status":
        if any(value is not None for value in (trace, input_channel, output_channel, component)) or magnitude != "linear":
            raise ValueError("HB status table accepts no trace presentation arguments")
        if _subplot_type(figure, row, col) != "domain":
            raise ValueError("HB status table requires a compatible domain subplot")
        figure.add_trace(_hb_status_table(result.cases), row=row, col=col)
        return figure
    if kind != "trace":
        raise ValueError("kind must be 'trace' or 'status'")
    if _subplot_type(figure, row, col) != "xy":
        raise ValueError("HB trace requires a compatible xy subplot")
    if component is None:
        raise ValueError("HB add_to trace requires one explicit component")
    selected_magnitude = magnitude if component == "magnitude" else "linear"
    _component_values(
        np.asarray([1 + 0j]), component, family="S", magnitude=selected_magnitude
    )
    unit = "dB" if component == "magnitude" and magnitude == "db" else (
        "degree (exact zero undefined)" if component == "phase" else "dimensionless"
    )
    successes = tuple(outcome for outcome in result.cases.values() if outcome.succeeded)
    if not successes:
        raise ValueError("an all-failed HB batch has no trace to add")
    if trace is not None and (input_channel is not None or output_channel is not None):
        raise ValueError("trace and matrix-channel selection are mutually exclusive")
    if trace is not None and any(trace not in outcome.traces for outcome in successes):
        raise ValueError(f"unknown HB trace: {trace}")
    if trace is not None:
        first_frequencies = successes[0].traces[trace].frequencies
    else:
        first_frequencies = successes[0].s.view.frequencies
    y_axis = _response_axis(
        family="S",
        component=component,
        magnitude=selected_magnitude,
        unit=unit,
    )
    frequency_unit = _frequency_unit_for_axis(
        figure, row, col, frequencies=first_frequencies, y_axis=y_axis
    )
    pending: list[Any] = []
    for outcome in successes:
        if trace is not None:
            trace_result = outcome.traces[trace]
            pending.append(_trace_result_trace(
                trace_result,
                component=component,
                magnitude=magnitude,
                name=f"{outcome.id}: {trace}",
                frequency_unit=frequency_unit,
                source=_presentation_source(
                    trace_result, hb_case=outcome.id, named_trace=trace
                ),
            ))
        else:
            view = outcome.s.view
            input_index = _channel_index(
                view.input_channels, input_channel, role="input", default=False
            )
            output_index = _channel_index(
                view.output_channels, output_channel, role="output", default=False
            )
            pending.append(_trace(
                view,
                family="S",
                input_index=input_index,
                output_index=output_index,
                component=component,
                magnitude=magnitude,
                name=f"{outcome.id}: S[{_channel_label(view.output_channels[output_index])}"
                f" <- {_channel_label(view.input_channels[input_index])}] {component}",
                frequency_unit=frequency_unit,
                source=_presentation_source(
                    outcome.s,
                    hb_case=outcome.id,
                    input_channel=_channel_record(view.input_channels[input_index]),
                    output_channel=_channel_record(view.output_channels[output_index]),
                ),
            ))
    for pending_trace in pending:
        figure.add_trace(pending_trace, row=row, col=col)
    return figure


def figure_fragments(figures: Sequence[Any]) -> str:
    """Serialize Figures into one self-contained HTML fragment, bundling JS once."""

    return "".join(
        figure_fragment(title, figure, include_plotlyjs=index == 0)
        for index, (title, figure) in enumerate(figures)
    )


def figure_fragment(title: str, figure: Any, *, include_plotlyjs: bool) -> str:
    """Serialize one report Figure; the caller owns document-level JS deduplication."""

    from html import escape

    _, pio, _ = _plotly()
    return (
        f"<h2>{escape(str(title))}</h2>"
        + pio.to_html(
            figure,
            include_plotlyjs=include_plotlyjs,
            full_html=False,
            auto_play=False,
            config={"responsive": True},
        )
    )
