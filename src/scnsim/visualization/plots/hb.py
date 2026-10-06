"""Harmonic-balance case and outcome Plotly views."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from typing import Any, Literal

import numpy as np

from ..presentation import Theme
from ...results.selection import _channel_index
from .common import (
    Channel, Component, _channel_label, _channel_record, _component_values,
    _display_axis, _frequency_axis, _frequency_unit_for_axis, _matrix_values,
    _plotly, _presentation_source, _response_axis, _set_db_viewport, _style,
    _subplot_type, _trace, _trace_meta,
)
from .numerical import _trace_result_trace, matrix_add_to, matrix_plot

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
