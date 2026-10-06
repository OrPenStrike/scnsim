"""Optimization ledger and comparison Plotly views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from html import escape
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


def _ledger_rows(result: Any) -> tuple[list[int], list[float | None], list[str], list[str]]:
    from ...canonical import float64_from_hex

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


def _optimization_term_rows(result: Any) -> tuple[list[object], ...]:
    from ...canonical import float64_from_hex

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
