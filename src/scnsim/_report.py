"""Pure report normalization and HTML composition over verified Results.

This owner receives no CircuitRun, workspace, or execution callback.  The
runtime facade validates verified Result ownership before calling it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from html import escape
from typing import Any

from . import units
from ._canonical import canonical_json_bytes, quantity_envelope
from ._numeric_presentation import (
    figure_fragment,
    hb_outcomes_plot,
    report_palette,
)
from .presentation import Theme, _report_html
from .results import (
    AnalysisResult,
    DirectQuantityResult,
    DirectSolveResult,
    HBBatchResult,
    OperatorResult,
    OptimizationResult,
    ParameterPointIdentity,
    ParameterSweepResult,
    ReportResult,
    _is_verified_analysis_result,
    _verified_result,
)
from .specs import ReportPanel, ReportSpec


def _identity_record(result: AnalysisResult) -> dict[str, object]:
    identity = result.identity
    if isinstance(identity, ParameterPointIdentity):
        return {
            "plan_sha256": identity.batch.plan_sha256,
            "request_sha256": identity.batch.request_sha256,
            "attempt_sha256": identity.batch.attempt_sha256,
            "result_sha256": identity.batch.result_sha256,
            "source_index": identity.source_index,
            "parameters_sha256": identity.parameters_sha256,
        }
    return {
        name: getattr(identity, name)
        for name in ("plan_sha256", "request_sha256", "attempt_sha256", "result_sha256")
    }


def _channel_record(channel: object) -> object:
    if isinstance(channel, tuple):
        return {"coordinate": channel[0], "mode": list(channel[1])}
    return channel


def _panel_record(panel: ReportPanel, input_index: int) -> dict[str, object]:
    frequency = panel.frequency
    return {
        "input_index": input_index,
        "kind": panel.kind,
        "family": panel.family,
        "case": panel.case,
        "trace": panel.trace,
        "input_channel": _channel_record(panel.input_channel),
        "output_channel": _channel_record(panel.output_channel),
        "frequency": None if frequency is None else quantity_envelope(
            frequency, si_unit="hertz", registry=units.registry
        ),
        "matrix_style": panel.matrix_style,
        "component": panel.component,
        "magnitude": panel.magnitude,
        "history_metric": panel.history_metric,
        "objective": panel.objective,
        "parameter": None if panel.parameter is None else {
            "definitions_id": panel.parameter.definitions_id,
            "id": panel.parameter.id,
        },
    }


def _table(title: str, rows: Sequence[tuple[object, object]]) -> str:
    body = "".join(
        f"<tr><th>{escape(str(name))}</th><td>{escape(str(value))}</td></tr>"
        for name, value in rows
    )
    return f"<h2>{escape(title)}</h2><table><tbody>{body}</tbody></table>"


def _identity_table(inputs: Sequence[AnalysisResult]) -> str:
    rows = "".join(
        "<tr>"
        + f"<td>{index}</td><td>{escape(type(result).__name__)}</td>"
        + "".join(f"<td>{escape(str(record[name]))}</td>" for name in (
            "plan_sha256", "request_sha256", "attempt_sha256", "result_sha256"
        ))
        + f"<td>{escape(str(record.get('source_index', 'n/a')))}</td>"
        + f"<td>{escape(str(record.get('parameters_sha256', 'n/a')))}</td>"
        + "</tr>"
        for index, result in enumerate(inputs)
        for record in (_identity_record(result),)
    )
    return (
        "<h1>SCNSim report</h1><h2>Source Results</h2>"
        "<table><thead><tr><th>input</th><th>type</th><th>Plan</th><th>Request</th>"
        "<th>Attempt</th><th>Result</th><th>source index</th>"
        f"<th>parameters SHA-256</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _failed_hb_selection(panel: ReportPanel, result: HBBatchResult) -> str:
    selected = (
        result.cases
        if panel.case is None
        else {panel.case: result.cases[panel.case]}
    )
    requested: object
    if panel.trace is not None:
        declarations = result._presentation["declared_traces"]
        declaration = declarations[panel.trace]
        input_channel = declaration["input_channel"]
        output_channel = declaration["output_channel"]
        requested = (
            f"trace {panel.trace}: "
            f"{output_channel['coordinate']} mode={tuple(output_channel['mode'])} <- "
            f"{input_channel['coordinate']} mode={tuple(input_channel['mode'])}"
        )
    elif panel.input_channel is not None:
        requested = f"{panel.output_channel} <- {panel.input_channel}"
    else:
        requested = "case outcomes"
    rows: list[tuple[object, object]] = [
        ("requested family", panel.family),
        ("requested response", requested),
        ("requested component", panel.component),
        ("requested magnitude", panel.magnitude),
    ]
    rows.extend(
        (
            f"case {identifier}",
            "unavailable: "
            f"{outcome.failure.kind} / {outcome.failure.stage}: {outcome.failure}",
        )
        for identifier, outcome in selected.items()
    )
    return _table("HB requested response and typed outcomes", rows)


def _summary(result: AnalysisResult) -> str:
    rows: list[tuple[object, object]] = [("Result type", type(result).__name__)]
    if isinstance(result, DirectSolveResult):
        view = result.s.view
        rows.extend((
            ("frequency points", len(view.frequencies)),
            ("coordinates", ", ".join(view.coordinates)),
            ("input channels", ", ".join(str(item) for item in view.input_channels)),
            ("output channels", ", ".join(str(item) for item in view.output_channels)),
            ("named traces", ", ".join(result.traces) or "none"),
        ))
    elif isinstance(result, DirectQuantityResult):
        for name in (
            "root", "frequency", "linewidth", "slope", "value", "magnitude",
            "real", "imag", "zero", "numerator_slope", "denominator", "coupling",
            "branch_a_residue", "branch_b_residue", "family",
        ):
            value = getattr(result, name, None)
            if value is not None:
                rows.append((name, value))
    elif isinstance(result, OperatorResult):
        rows.extend((
            ("materialized frequencies", ", ".join(str(point.frequency) for point in result.points)),
            ("coordinates", ", ".join(result.points[0].coordinates) if result.points else "none"),
            ("matrix shapes", ", ".join(str(point.matrix.shape) for point in result.points)),
        ))
    elif isinstance(result, HBBatchResult):
        rows.extend((
            ("declared cases", ", ".join(result.cases)),
            ("successful cases", ", ".join(key for key, value in result.cases.items() if value.succeeded) or "none"),
            ("failed cases", ", ".join(key for key, value in result.cases.items() if not value.succeeded) or "none"),
        ))
    elif isinstance(result, ParameterSweepResult):
        rows.append(("declared points", len(result.points)))
    elif isinstance(result, OptimizationResult):
        rows.extend((
            ("winner cost", result.best.cost),
            ("winner parameters", "; ".join(
                f"{parameter.definitions_id}.{parameter.id}={value}"
                for parameter, value in result.best.parameters.values.items()
            )),
            ("completed generation ledgers", len(result.ledger)),
        ))
    return _table("Result summary", rows)


def _panel_content(panel: ReportPanel, *, theme: Theme) -> tuple[str, object | None]:
    result = panel.result
    if panel.kind == "summary":
        return _summary(result), None
    if panel.kind == "outcomes":
        if isinstance(result, HBBatchResult):
            return "HB outcomes", hb_outcomes_plot(result, theme=theme)
        assert isinstance(result, ParameterSweepResult)
        return "Parameter sweep outcomes", result.plot(theme=theme)
    if panel.kind == "history":
        assert isinstance(result, OptimizationResult)
        kind = "history" if panel.history_metric == "cost" else panel.history_metric
        return f"Optimization {panel.history_metric} history", result.plot(
            kind=kind, objective=panel.objective, parameter=panel.parameter, theme=theme
        )
    if panel.kind == "matrix":
        assert panel.frequency is not None and panel.matrix_style is not None
        if isinstance(result, DirectSolveResult):
            assert panel.family is not None
            return f"{panel.family} matrix {panel.matrix_style}", result.plot(
                family=panel.family,
                kind=panel.matrix_style,
                frequency=panel.frequency,
                component=panel.component,
                magnitude=panel.magnitude,
                theme=theme,
            )
        assert isinstance(result, OperatorResult)
        return f"Operator matrix {panel.matrix_style}", result.plot(
            frequency=panel.frequency,
            kind=panel.matrix_style,
            component=panel.component,
            theme=theme,
        )
    assert panel.kind == "response"
    assert panel.family is not None and panel.component is not None and panel.magnitude is not None
    if isinstance(result, DirectSolveResult):
        return f"{panel.family} response", result.plot(
            family=panel.family,
            input_channel=panel.input_channel,
            output_channel=panel.output_channel,
            component=panel.component,
            magnitude=panel.magnitude,
            theme=theme,
        )
    assert isinstance(result, HBBatchResult)
    if panel.case is None:
        if not any(outcome.succeeded for outcome in result.cases.values()):
            return _failed_hb_selection(panel, result), None
        return "HB response — all declared cases", result.plot(
            trace=panel.trace,
            input_channel=panel.input_channel,
            output_channel=panel.output_channel,
            component=panel.component,
            magnitude=panel.magnitude,
            theme=theme,
        )
    outcome = result.cases[panel.case]
    if not outcome.succeeded:
        return _failed_hb_selection(panel, result), None
    if panel.trace is not None:
        trace_result = outcome.traces[panel.trace]
        source = getattr(trace_result, "_presentation", {})
        input_channel = source.get("input_channel") if isinstance(source, Mapping) else None
        output_channel = source.get("output_channel") if isinstance(source, Mapping) else None
        title = f"HB case {panel.case}: {panel.trace} {input_channel} -> {output_channel}"
        return title, trace_result.plot(
            component=panel.component, magnitude=panel.magnitude, theme=theme
        )
    return f"HB case {panel.case} S response", outcome.s.plot(
        input_channel=panel.input_channel,
        output_channel=panel.output_channel,
        component=panel.component,
        magnitude=panel.magnitude,
        theme=theme,
    )


def _automatic_panels(result: AnalysisResult) -> tuple[ReportPanel, ...]:
    if isinstance(result, DirectSolveResult):
        view = result.s.view
        kind = "response" if len(view.input_channels) == len(view.output_channels) == 1 else "summary"
        if kind == "response":
            return (
                ReportPanel(result=result, kind="summary"),
                ReportPanel(
                    result=result, kind="response", family="S",
                    component="magnitude", magnitude="db",
                ),
            )
        return (ReportPanel(result=result, kind="summary"),)
    if isinstance(result, (HBBatchResult, ParameterSweepResult)):
        return (ReportPanel(result=result, kind="outcomes"),)
    if isinstance(result, OptimizationResult):
        return (
            ReportPanel(result=result, kind="history", history_metric="cost"),
            ReportPanel(result=result, kind="summary"),
        )
    return (ReportPanel(result=result, kind="summary"),)


def build_report(spec: ReportSpec) -> ReportResult:
    """Compose one self-contained ReportResult without execution authority."""

    if not isinstance(spec, ReportSpec) or not all(
        _is_verified_analysis_result(result) for result in spec.inputs
    ):
        raise TypeError("build_report requires verified ReportSpec inputs")
    selected: dict[int, list[ReportPanel]] = {}
    for panel in spec.panels:
        index = next((i for i, result in enumerate(spec.inputs) if panel.result is result), None)
        if index is None:
            raise ValueError("ReportPanel.result is not the exact ReportSpec input")
        selected.setdefault(index, []).append(panel)

    palette, color_scheme = report_palette(spec.theme)
    panel_records: list[dict[str, object]] = []
    parts = [_identity_table(spec.inputs)]
    included_plotly = False
    for index, result in enumerate(spec.inputs):
        panels = tuple(selected.get(index, ())) or _automatic_panels(result)
        parts.append(f"<h2>Input {index}: {escape(type(result).__name__)}</h2>")
        for panel in panels:
            panel_records.append(_panel_record(panel, index))
            title_or_html, figure = _panel_content(panel, theme=spec.theme)
            if figure is None:
                parts.append(title_or_html)
            else:
                parts.append(figure_fragment(
                    title_or_html, figure, include_plotlyjs=not included_plotly
                ))
                included_plotly = True
    identity_record = {
        "kind": "scnsim_report_presentation",
        "inputs": [_identity_record(result) for result in spec.inputs],
        "panels": panel_records,
        "theme": {
            "requested": spec.theme.value,
            "color_scheme": color_scheme,
            "background": palette.background,
            "foreground": palette.foreground,
            "secondary": palette.secondary,
            "grid": palette.grid,
            "accent": palette.accent,
            "cycle": list(palette.cycle),
        },
    }
    presentation_sha256 = sha256(canonical_json_bytes(identity_record)).hexdigest()
    parts.insert(
        1,
        "<p><strong>Presentation identity</strong> "
        f"<code>{presentation_sha256}</code></p>",
    )
    html = _report_html(
        "".join(parts), spec.theme, palette=palette, color_scheme=color_scheme
    )
    return _verified_result(
        ReportResult,
        html=html,
        inputs=spec.inputs,
        presentation_sha256=presentation_sha256,
    )
