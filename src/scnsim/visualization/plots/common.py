"""Shared lazy rendering, metadata, and display helpers for Plotly views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Any, Literal

import numpy as np
from pint import Quantity

from ..presentation import Theme, _Palette, _palette, _require_theme

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


def _quantity_text(value: object) -> str:
    from ...canonical import float64_from_hex

    if value == "complex_root_midpoint":
        return "complex_root_midpoint"
    if not isinstance(value, Mapping):
        return "—"
    encoded = value.get("si_value_f64")
    unit = value.get("si_unit")
    if not isinstance(encoded, str) or not isinstance(unit, str):
        return "—"
    return f"{float64_from_hex(encoded):.12g} {escape(unit)}"


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
