"""Pure report normalization and HTML composition over verified Results.

This owner receives no CircuitRun, workspace, or execution callback.  The
runtime facade validates verified Result ownership before calling it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from html import escape
from typing import Any

from .. import units
from ..canonical import canonical_json_bytes, quantity_envelope
from .plots.common import _quantity_text, figure_fragment, report_palette
from .plots.optimization import optimization_comparison_dataset
from .presentation import Theme, _report_html
from ..results import (
    DirectQuantityResult,
    DirectSolveResult,
    InMemoryAnalysisResult,
    InMemoryDirectSolveResult,
    InMemoryOptimizationResult,
    InMemoryParameterSweepResult,
    InMemorySensitivityResult,
    OperatorResult,
    OptimizationResult,
    ParameterPointIdentity,
    ParameterSweepResult,
    ReportResult,
    _is_checked_report_input,
    _is_verified_analysis_result,
    _verified_result,
)
from ..specs import ReportPanel, ReportSpec


def _is_memory_analysis(result: object) -> bool:
    return type(result) is InMemoryAnalysisResult


def _is_memory_direct(result: object) -> bool:
    return type(result) is InMemoryDirectSolveResult or (
        type(result) is InMemoryAnalysisResult and result.result_kind == "direct_solve"
    )


def _identity_record(result: object) -> dict[str, object]:
    if _is_checked_report_input(result) and not _is_verified_analysis_result(result):
        return {
            "persistence": "memory",
            "plan_sha256": result.plan_sha256,
            "request_sha256": result.request_sha256,
            "execution_id": result.execution_id,
        }
    identity = result.identity
    if isinstance(identity, ParameterPointIdentity):
        return {
            "persistence": "saved",
            "plan_sha256": identity.batch.plan_sha256,
            "request_sha256": identity.batch.request_sha256,
            "attempt_sha256": identity.batch.attempt_sha256,
            "result_sha256": identity.batch.result_sha256,
            "source_index": identity.source_index,
            "parameters_sha256": identity.parameters_sha256,
        }
    return {
        "persistence": "saved",
        **{
            name: getattr(identity, name)
            for name in ("plan_sha256", "request_sha256", "attempt_sha256", "result_sha256")
        },
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


def _identity_table(inputs: Sequence[object]) -> str:
    rows = "".join(
        "<tr>"
        + f"<td>{index}</td><td>{escape(type(result).__name__)}</td>"
        + f"<td>{escape(str(record.get('persistence', 'saved')))}</td>"
        + "".join(f"<td>{escape(str(record.get(name, 'not persisted')))}</td>" for name in (
            "plan_sha256", "request_sha256", "attempt_sha256", "result_sha256"
        ))
        + f"<td>{escape(str(record.get('execution_id', 'n/a')))}</td>"
        + f"<td>{escape(str(record.get('source_index', 'n/a')))}</td>"
        + f"<td>{escape(str(record.get('parameters_sha256', 'n/a')))}</td>"
        + "</tr>"
        for index, result in enumerate(inputs)
        for record in (_identity_record(result),)
    )
    return (
        "<h1>SCNSim report</h1><h2>Source Results</h2>"
        "<table><thead><tr><th>input</th><th>type</th><th>persistence</th><th>Plan</th><th>Request</th>"
        "<th>Attempt</th><th>Result</th><th>Execution</th><th>source index</th>"
        f"<th>parameters SHA-256</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _summary(result: object) -> str:
    rows: list[tuple[object, object]] = [("Result type", type(result).__name__)]
    if _is_memory_direct(result) or isinstance(result, DirectSolveResult):
        if _is_memory_analysis(result) and result.status == "failure":
            failure = result.failure
            rows.extend((
                ("status", result.status),
                ("failure", "unavailable" if failure is None else f"{type(failure).__name__}: {failure}"),
            ))
        else:
            view = result.s.view
            rows.extend((
                ("frequency points", len(view.frequencies)),
                ("coordinates", ", ".join(view.coordinates)),
                ("input channels", ", ".join(str(item) for item in view.input_channels)),
                ("output channels", ", ".join(str(item) for item in view.output_channels)),
                ("named traces", ", ".join(result.traces) or "none"),
            ))
    elif isinstance(result, DirectQuantityResult) or (
        _is_memory_analysis(result) and result.result_kind not in {"direct_solve", "operator"}
    ):
        presentation = result._presentation
        spec = presentation.get("spec") if isinstance(presentation, Mapping) else None
        lineage = presentation.get("ref_lineage") if isinstance(presentation, Mapping) else None
        if isinstance(spec, Mapping) and spec.get("type") in {"diagonal_root", "operator_element_root"}:
            basis = lineage.get("terminal_coordinates") if isinstance(lineage, Mapping) else None
            row = spec.get("coordinate") if spec["type"] == "diagonal_root" else spec.get("row")
            column = spec.get("coordinate") if spec["type"] == "diagonal_root" else spec.get("column")
            rows.extend((("final View basis", ", ".join(str(item) for item in basis) if isinstance(basis, (list, tuple)) else "unavailable"),
                         ("root equation", f"F_View[{row}, {column}](omega) = 0"),
                         ("root hint", _quantity_text(spec.get("root_hint"))),
                         ("frequency", "Re(omega) / (2 pi); slope = dF_View[row,column] / d omega")))
        for name in (
            "root", "frequency", "linewidth", "slope", "value", "magnitude",
            "real", "imag", "zero", "numerator_slope", "denominator", "coupling",
            "branch_a_residue", "branch_b_residue", "family",
        ):
            value = getattr(result, name, None)
            if value is not None:
                rows.append((name, value))
        if _is_memory_analysis(result):
            rows.append(("status", result.status))
            if result.status == "failure":
                failure = result.failure
                rows.append((
                    "failure",
                    "unavailable" if failure is None else f"{type(failure).__name__}: {failure}",
                ))
    elif isinstance(result, OperatorResult) or (
        _is_memory_analysis(result) and result.result_kind == "operator"
    ):
        points = result.points if isinstance(result, OperatorResult) else result.operator_points
        rows.extend((
            ("materialized frequencies", ", ".join(str(point.frequency) for point in points)),
            ("coordinates", ", ".join(points[0].coordinates) if points else "none"),
            ("matrix shapes", ", ".join(str(point.matrix.shape) for point in points)),
        ))
    elif isinstance(result, (ParameterSweepResult, InMemoryParameterSweepResult)):
        rows.append(("declared points", len(result.points)))
        if isinstance(result, InMemoryParameterSweepResult):
            successes = sum(point.succeeded for point in result.points)
            rows.extend((
                ("successful points", successes),
                ("failed points", len(result.points) - successes),
                ("persistence", "memory; no saved attempt identity"),
            ))
    elif isinstance(result, (OptimizationResult, InMemoryOptimizationResult)):
        if result.best is None:
            rows.extend((
                ("comparison-model winner", "unavailable"),
                ("comparison-model identity", result.comparison_model_identity or "unavailable"),
            ))
        else:
            rows.extend((
                ("comparison-model best cost", result.best.cost),
                ("comparison-model best parameters", "; ".join(
                    f"{parameter.definitions_id}.{parameter.id}={value}"
                    for parameter, value in result.best.parameters.values.items()
                )),
            ))
        if result.stages:
            rows.extend(
                (f"stage {stage.stage_id}",
                 f"{stage.strategy}; {stage.status}; evaluations={stage.evaluation_count}; "
                 f"iterations={stage.iteration_count}; model={stage.model_identity}")
                for stage in result.stages
            )
        else:
            rows.append(("completed generation ledgers", len(result.ledger)))
        settings, comparison = optimization_comparison_dataset(result)
        return (
            _table("Optimization Result summary", rows)
            + _table("Optimization settings", settings)
            + _table(
                "Initial / best-found comparison",
                [
                    (field, f"initial {initial}; best found {best}")
                    for field, initial, best in comparison
                ],
            )
        )
    elif isinstance(result, InMemorySensitivityResult):
        rows.extend((
            ("persistence", "memory; no saved attempt identity"),
            ("execution ID", result.execution_id),
            ("status", result.status),
            ("quantities", ", ".join(quantity.id for quantity in result.quantities)),
            ("derivative operations", "; ".join(
                f"{kind}: {operation.status}"
                for kind, operation in result.operations.items()
            )),
        ))
    elif _is_memory_analysis(result):
        rows.extend((
            ("persistence", "memory; no saved attempt identity"),
            ("status", result.status),
            ("program SHA-256", result.program_sha256),
            ("cache hit", result.cache_hit),
        ))
    return _table("Result summary", rows)


def _panel_content(panel: ReportPanel, *, theme: Theme) -> tuple[str, object | None]:
    result = panel.result
    if panel.kind == "summary":
        return _summary(result), None
    if panel.kind == "outcomes":
        assert isinstance(result, (ParameterSweepResult, InMemoryParameterSweepResult))
        return "Parameter sweep outcomes", result.plot(theme=theme)
    if panel.kind == "history":
        assert isinstance(result, (OptimizationResult, InMemoryOptimizationResult))
        kind = "history" if panel.history_metric == "cost" else panel.history_metric
        return f"Optimization {panel.history_metric} history", result.plot(
            kind=kind, objective=panel.objective, parameter=panel.parameter, theme=theme
        )
    if panel.kind == "matrix":
        assert panel.frequency is not None and panel.matrix_style is not None
        if _is_memory_analysis(result) and result.status == "failure":
            return _summary(result), None
        if _is_memory_direct(result) or isinstance(result, DirectSolveResult):
            assert panel.family is not None
            return f"{panel.family} matrix {panel.matrix_style}", result.plot(
                family=panel.family,
                kind=panel.matrix_style,
                frequency=panel.frequency,
                component=panel.component,
                magnitude=panel.magnitude,
                theme=theme,
            )
        assert isinstance(result, (OperatorResult, InMemoryAnalysisResult))
        return f"Operator matrix {panel.matrix_style}", result.plot(
            frequency=panel.frequency,
            kind=panel.matrix_style,
            component=panel.component,
            theme=theme,
        )
    assert panel.kind == "response"
    assert panel.family is not None and panel.component is not None and panel.magnitude is not None
    if _is_memory_analysis(result) and result.status == "failure":
        return _summary(result), None
    if _is_memory_direct(result) or isinstance(result, DirectSolveResult):
        return f"{panel.family} response", result.plot(
            family=panel.family,
            input_channel=panel.input_channel,
            output_channel=panel.output_channel,
            component=panel.component,
            magnitude=panel.magnitude,
            theme=theme,
        )

def _automatic_panels(result: object) -> tuple[ReportPanel, ...]:
    if _is_memory_direct(result) or isinstance(result, DirectSolveResult):
        if _is_memory_analysis(result) and result.status == "failure":
            return (ReportPanel(result=result, kind="summary"),)
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
    if isinstance(result, (ParameterSweepResult, InMemoryParameterSweepResult)):
        return (ReportPanel(result=result, kind="outcomes"),)
    if isinstance(result, (OptimizationResult, InMemoryOptimizationResult)):
        return (
            ReportPanel(result=result, kind="history", history_metric="cost"),
            ReportPanel(result=result, kind="summary"),
        )
    return (ReportPanel(result=result, kind="summary"),)


def build_report(spec: ReportSpec) -> ReportResult:
    """Compose one self-contained ReportResult without execution authority."""

    if not isinstance(spec, ReportSpec) or not all(
        _is_checked_report_input(result) for result in spec.inputs
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
