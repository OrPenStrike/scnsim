"""Optimization ledger and comparison Plotly views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape, unescape
import textwrap
from typing import Any, Literal

import numpy as np

from ..presentation import Theme
from ...results.selection import _optimization_series
from .common import _plotly, _presentation_source, _quantity_text, _require_axis_semantics, _style, _subplot_type, _trace_meta

def _complex_quantity_text(value: object) -> str:
    from ...canonical import float64_from_hex

    if not isinstance(value, Mapping):
        return "—"
    real = value.get("real_si_f64")
    imag = value.get("imag_si_f64")
    unit = value.get("si_unit")
    if not all(isinstance(item, str) for item in (real, imag, unit)):
        return "—"
    real_value = float64_from_hex(real)
    imag_value = float64_from_hex(imag)
    return f"{real_value:.12g} {imag_value:+.12g}j {escape(unit)}"


def _optimization_term_rows(result: Any) -> tuple[list[object], ...]:
    from ...canonical import float64_from_hex

    objectives = result._presentation["objectives"]
    objective_units = {
        objective["id"]: objective["target"]["si_unit"]
        for objective in objectives
    }
    rows = []
    for candidate in result._fixed_reader.project("table"):
        cost = candidate.get("cost_f64")
        cost_text = f"{float64_from_hex(cost):.12g}" if isinstance(cost, str) else "—"
        parameters = "; ".join(
            f"{binding['parameter']['definitions_id']}.{binding['parameter']['parameter_id']} = "
            f"{_quantity_text(binding['value'])}"
            for binding in candidate["parameters"]["bindings"]
        )
        components = "; ".join(
            f"{component['objective_id']}: "
            + (f"{float64_from_hex(component['value_f64']):.12g} {escape(objective_units[component['objective_id']])}"
               if isinstance(component.get("value_f64"), str) else str(component["status"]))
            for component in candidate["components"]
        )
        rows.append((
            candidate["evaluation_ordinal"], candidate["generation"],
            escape(str(candidate["status"])), cost_text, parameters, components,
        ))
    return tuple(list(column) for column in zip(*rows, strict=True)) if rows else tuple([] for _ in range(6))


def _comparison_table_rows(
    rows: Sequence[Sequence[object]],
    *,
    column_widths: Sequence[int],
) -> tuple[list[list[str]], int]:
    """Wrap actual table content and estimate a uniform Plotly row height."""

    if not rows:
        return [[] for _ in column_widths], 80
    total_width = sum(column_widths)
    # Plotly figures commonly render at notebook-column widths. Budget a
    # conservative character count instead of assuming a full desktop canvas.
    available_chars = max(48, total_width * 0.96)
    char_widths = [max(12, round(available_chars * width / total_width)) for width in column_widths]
    wrapped_rows: list[list[str]] = []
    largest_lines = 1
    for row in rows:
        if len(row) != len(column_widths):
            raise ValueError("Optimization comparison table row has the wrong number of cells")
        wrapped: list[str] = []
        for value, width in zip(row, char_widths, strict=True):
            # The shared dataset is also used by HTML reports and contains a
            # mixture of plain and HTML-escaped text. Normalize it before
            # wrapping, then escape exactly once for Plotly's line-break markup.
            raw = unescape(str(value))
            lines = [
                segment
                for paragraph in raw.splitlines() or [""]
                for segment in (textwrap.wrap(
                    paragraph,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                ) or [""])
            ]
            largest_lines = max(largest_lines, len(lines))
            wrapped.append("<br>".join(escape(line) for line in lines))
        wrapped_rows.append(wrapped)
    # Plotly Table accepts one scalar row height. Size every row to the most
    # wrapped row so long physical values remain visible instead of clipping.
    return [list(column) for column in zip(*wrapped_rows, strict=True)], max(32, 19 * largest_lines + 10)


def _comparison_table(
    go: Any,
    *,
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
    column_widths: Sequence[int],
    section: str,
    source: Mapping[str, object],
) -> tuple[Any, int]:
    columns, cell_height = _comparison_table_rows(rows, column_widths=column_widths)
    row_count = max(1, len(rows))
    header_height = 40
    table_height = header_height + cell_height * row_count
    trace = go.Table(
        header={
            "values": [escape(value) for value in header],
            "align": "left",
            "height": header_height,
            "font": {"size": 13},
        },
        cells={"values": columns, "align": "left", "height": cell_height, "font": {"size": 12}},
        columnwidth=list(column_widths),
        meta={"scnsim": {"source": source, "section": section}},
    )
    return trace, table_height


def optimization_plot(
    result: Any,
    *,
    kind: Literal["history", "objective", "residual", "parameter", "table", "comparison"] = "history",
    objective: str | None = None,
    parameter: Any | None = None,
    theme: Theme = Theme.AUTO,
) -> Any:
    go, _, _ = _plotly()
    candidate_count = len(result.candidate_discretization)
    saved = f"saved best cost {result.best.cost:.12g}"
    if kind in {"history", "objective", "residual", "parameter"}:
        ordinals, values, statuses, label, unit, details = _optimization_series(
            result, kind=kind, objective=objective, parameter=parameter
        )
        figure = go.Figure(data=[go.Scatter(
            x=ordinals,
            y=values,
            mode="lines+markers",
            name=escape(label),
            customdata=np.asarray([
                [status, escape(detail)]
                for status, detail in zip(statuses, details, strict=True)
            ], dtype=object),
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
        rows = list(zip(*columns, strict=True)) if columns and columns[0] else []
        table, table_height = _comparison_table(
            go,
            header=("evaluation", "generation", "status", "cost", "parameters", "objectives"),
            rows=rows,
            column_widths=(10, 10, 12, 14, 32, 22),
            section="candidates",
            source=_presentation_source(result),
        )
        figure = go.Figure(data=[table])
    elif kind == "comparison":
        if objective is not None or parameter is not None:
            raise ValueError("objective and parameter are invalid for the comparison")
        settings, comparison = optimization_comparison_dataset(result)
        source = _presentation_source(result)
        settings_trace, settings_height = _comparison_table(
            go,
            header=("setting", "value"),
            rows=settings,
            column_widths=(34, 66),
            section="settings",
            source=source,
        )
        comparison_trace, comparison_height = _comparison_table(
            go,
            header=("field", "initial", "best found"),
            rows=comparison,
            column_widths=(34, 33, 33),
            section="comparison",
            source=source,
        )
        section_height = 28
        gap_height = 14
        plot_height = 2 * section_height + gap_height + settings_height + comparison_height
        settings_top = section_height / plot_height
        settings_bottom = (section_height + settings_height) / plot_height
        comparison_top = (section_height + settings_height + gap_height + section_height) / plot_height
        comparison_bottom = 1.0
        settings_trace.update(domain={"y": [1.0 - settings_bottom, 1.0 - settings_top]})
        comparison_trace.update(domain={"y": [1.0 - comparison_bottom, 1.0 - comparison_top]})
        figure = go.Figure(data=[
            settings_trace,
            comparison_trace,
        ])
        figure.add_annotation(
            text="Settings", x=0, y=1.0 - settings_top / 2,
            xref="paper", yref="paper",
            xanchor="left", showarrow=False,
        )
        figure.add_annotation(
            text="Initial / best-found comparison",
            x=0, y=1.0 - (section_height + settings_height + gap_height + section_height / 2) / plot_height,
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
        "candidate_count": candidate_count,
        "winner_cost": result.best.cost,
        "winner_parameters": winner_parameters,
    }})
    if kind != "comparison":
        best_bindings = "; ".join(
            f"{escape(parameter.definitions_id)}.{escape(parameter.id)} = {escape(str(value))}"
            for parameter, value in result.best.parameters.values.items()
        )
        figure.add_annotation(
            text=f"{saved}; {best_bindings}", x=0, y=1.12,
            xref="paper", yref="paper", xanchor="left", showarrow=False,
        )
    styled = _style(figure, theme, title=f"Completed optimization {kind}")
    if kind == "comparison":
        margins = styled.layout.margin
        top_margin = int(margins.t or 0)
        bottom_margin = int(margins.b or 0)
        top_margin = max(top_margin, 88)
        styled.update_layout(
            height=max(420, top_margin + bottom_margin + plot_height),
            margin={"t": top_margin, "b": bottom_margin},
        )
    elif kind == "table":
        margins = styled.layout.margin
        top_margin = max(int(margins.t or 0), 88)
        bottom_margin = int(margins.b or 0)
        styled.update_layout(
            height=max(420, top_margin + bottom_margin + table_height),
            margin={"t": top_margin, "b": bottom_margin},
        )
    else:
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

    from ...canonical import canonical_json_bytes, float64_from_hex

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
    best = presentation.get("best_candidate")
    ordinal = presentation.get("best_evaluation_ordinal")
    objectives = presentation.get("objectives")
    variables = presentation.get("variables")
    if (
        not isinstance(initial, Mapping)
        or not isinstance(best, Mapping)
        or not isinstance(ordinal, int)
        or not isinstance(objectives, Sequence)
        or not isinstance(variables, Sequence)
    ):
        raise ValueError("Optimization comparison presentation is incomplete")
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
        if "domain" in variable:
            settings.extend((
                (f"{name} domain / transform", f"{variable['domain']} / {variable['transform']}"),
                (f"{name} mapping", (f"x = x0 + {_quantity_text(variable['scale'])} · z"
                    if variable["transform"] == "linear" else "x = x0 · exp(z)")),
            ))
        else:
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
        left_terms, right_terms = left.get("terms"), right.get("terms")
        if isinstance(left_terms, Sequence) and isinstance(right_terms, Sequence):
            for left_term, right_term in zip(left_terms, right_terms, strict=True):
                if not isinstance(left_term, Mapping) or not isinstance(right_term, Mapping):
                    raise ValueError("Optimization term comparison evidence is malformed")
                selector = left_term.get("selector")
                if not isinstance(selector, Mapping) or selector.get("type") != "residue_coupling_projection":
                    continue
                left_evidence = left_term.get("residue_coupling_evidence")
                right_evidence = right_term.get("residue_coupling_evidence")
                if not isinstance(left_evidence, Mapping) or not isinstance(right_evidence, Mapping):
                    raise ValueError("Optimization coupling evidence is absent")
                ordinal = left_term.get("term_ordinal")
                label = f"{name} term {ordinal}"
                comparison.append((
                    f"{label} evaluation omega",
                    _complex_quantity_text(left_evidence.get("evaluation_omega")),
                    _complex_quantity_text(right_evidence.get("evaluation_omega")),
                ))
                for component, title in (("real", "Re J"), ("imag", "Im J"), ("magnitude", "abs J"), ("abs_real", "abs Re J")):
                    def projected(evidence: Mapping[str, object]) -> str:
                        coupling = evidence.get("coupling")
                        if not isinstance(coupling, Mapping):
                            raise ValueError("Optimization complex coupling evidence is malformed")
                        from ...canonical import float64_from_hex
                        real = float64_from_hex(coupling["real_si_f64"])
                        imag = float64_from_hex(coupling["imag_si_f64"])
                        value = {
                            "real": real,
                            "imag": imag,
                            "magnitude": abs(complex(real, imag)),
                            "abs_real": abs(real),
                        }[component]
                        return f"{value:.12g} radian / second"
                    comparison.append((
                        f"{label} {title}", projected(left_evidence), projected(right_evidence)
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
