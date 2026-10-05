"""Parameter-space field and sampled-point Plotly views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

import numpy as np
from pint import Quantity

from ..presentation import Theme
from .common import _plotly, _require_axis_semantics, _style, _subplot_type, _trace_meta

def _parameter_field_data(result: Any, *, x: Any, y: Any | None) -> dict[str, Any]:
    from ...authoring import ParameterRef, ParameterSet

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
    from ...results import _parameter_value_bytes

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
