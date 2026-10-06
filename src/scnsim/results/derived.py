from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from html import escape
from os import O_RDONLY, PathLike, fsync, link, open as os_open
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal

from pint import Quantity

from ..construction import unavailable
from ..visualization.presentation import Theme
from .base import (
    AnalysisResult, HtmlPresentation, LineDiscretization, ParameterPointIdentity, Result, ResultIdentity,
    _fresh_quantity_attribute,
)

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

@dataclass(frozen=True, slots=True)
class TraceResult(Result):
    frequencies: Quantity
    value: Quantity
    _parent_identity: ResultIdentity | ParameterPointIdentity | None = field(
        default=None, repr=False, compare=False
    )
    _presentation: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)
    _quantity_fields = frozenset({"frequencies", "value"})

    def __init__(self) -> None:
        unavailable("TraceResult construction")

    def __getattribute__(self, name: str) -> object:
        return _fresh_quantity_attribute(self, name)

    def plot(
        self,
        *,
        component: Literal["magnitude", "phase", "real", "imag"] | None = None,
        magnitude: Literal["linear", "db"] = "linear",
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        from ..visualization.plots.numerical import trace_plot

        return trace_plot(self, component=component, magnitude=magnitude, theme=theme)

    def show(self, **presentation: object) -> None:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        row: int,
        col: int,
        component: Literal["magnitude", "phase", "real", "imag"],
        magnitude: Literal["linear", "db"] = "linear",
        name: str | None = None,
    ) -> Figure:
        from ..visualization.plots.numerical import trace_add_to

        return trace_add_to(
            self, fig, row=row, col=col, component=component, magnitude=magnitude, name=name
        )

@dataclass(frozen=True, slots=True)
class ExplanationResult(Result):
    evidence: Mapping[str, object]

    def __init__(self) -> None:
        unavailable("ExplanationResult construction")

    @property
    def discretization(self) -> tuple[LineDiscretization, ...]:
        from .discretization import _decode_discretization

        compiled = self.evidence.get("compiled", {})
        if not isinstance(compiled, Mapping):
            raise ValueError("explanation has no compiler evidence")
        return _decode_discretization(compiled.get("discretization")) or ()

    def show(self, **presentation: object) -> HtmlPresentation:
        def table(title: str, headers: tuple[str, ...], rows: object) -> str:
            body = "".join(
                "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
                for row in rows
            )
            head = "".join(f"<th>{escape(header)}</th>" for header in headers)
            return f"<h3>{escape(title)}</h3><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

        evidence = self.evidence
        compiled = evidence.get("compiled", {})
        lineage = evidence.get("ref_lineage", evidence.get("view", {}))
        hierarchy = evidence.get("component_hierarchy", ())
        parameter_source = evidence.get("parameter_source", {})
        parameters_record = evidence.get("parameters", {})
        if (
            isinstance(parameter_source, Mapping)
            and parameter_source.get("kind") == "point"
        ):
            parameters_record = parameter_source.get("parameters", {})
        parameters = parameters_record.get("bindings", ()) if isinstance(parameters_record, Mapping) else ()
        html = table(
            "Identity",
            ("field", "value"),
            ((name, evidence.get(name)) for name in ("plan_sha256", "request_sha256", "runtime_semantic", "spec")),
        )
        html += table(
            "View lineage",
            ("step", "evidence"),
            ((name, lineage.get(name)) for name in ("original", "ptc", "transforms", "retain", "terminal_coordinates", "port_realizable")),
        )
        html += table(
            "Components and parameters",
            ("kind", "identity", "declaration"),
            tuple(("component", item.get("component_path"), item) for item in hierarchy)
            + tuple(("parameter", item.get("parameter"), item.get("value")) for item in parameters),
        )
        spec = evidence.get("spec")
        if isinstance(spec, Mapping) and spec.get("type") == "optimization":
            html += table(
                "Resolved optimization declaration",
                ("kind", "identity", "declaration"),
                tuple(("variable", item.get("parameter"), item) for item in spec.get("variables", ()))
                + tuple(("objective", item.get("id"), item) for item in spec.get("objectives", ()))
                + (("optimizer", "controls", spec.get("optimizer")),),
            )
        if isinstance(compiled, Mapping):
            html += table(
                "Resolved transmission-line grids",
                ("component", "method", "length", "modal velocities", "hmax", "N", "dx"),
                (
                    (
                        row.component_path, row.kind, row.length,
                        row.modal_velocities, row.hmax,
                        row.n_sections, row.dx,
                    )
                    for row in self.discretization
                ),
            )
            html += table(
                "Compiler and capability",
                ("field", "value"),
                (
                    ("node_order", compiled.get("node_order")),
                    ("C shape", compiled.get("c_matrix", {}).get("shape")),
                    ("K shape", compiled.get("k_matrix", {}).get("shape")),
                    ("G shape", compiled.get("g_matrix", {}).get("shape")),
                    ("ports", compiled.get("ports")),
                    ("root", compiled.get("root_preflight")),
                    ("optimization", compiled.get("optimization_preflight")),
                    ("Direct / HB", compiled.get("direct_hb_capability")),
                ),
            )
            rows = compiled.get("expanded_branch_rows", ())
            line_rows = tuple(
                row for row in rows
                if isinstance(row, Mapping) and row.get("kind") == "transmission_line_audit"
            )
            if line_rows:
                html += table(
                    "Transmission-line expansion",
                    ("component", "conductors/reference", "sections", "length / dx", "orientation", "stations", "source"),
                    (
                        (
                            row.get("component_path"),
                            (row.get("conductors"), row.get("reference_conductor")),
                            row.get("n_sections"),
                            (row.get("length"), row.get("dx")),
                            row.get("orientation"), row.get("stations"), row.get("rlgc_source"),
                        )
                        for row in line_rows
                    ),
                )
            html += table(
                "Expanded branch rows",
                ("component", "kind", "section", "station/end", "row", "column", "value", "omitted"),
                (
                    (
                        row.get("component_path"), row.get("kind"), row.get("section"),
                        (row.get("station"), row.get("end")), row.get("row_conductor"),
                        row.get("column_conductor"), row.get("value"), row.get("omitted_as_zero"),
                    )
                    for row in rows
                    if isinstance(row, Mapping)
                ),
            )
        return HtmlPresentation(html)

@dataclass(frozen=True, slots=True)
class InventoryResult(Result):
    """Pure read-only evidence inventory; it never selects a result for resolve."""

    requests: tuple[Mapping[str, object], ...]
    maintenance: tuple[Mapping[str, object], ...]

    def __init__(self) -> None:
        unavailable("InventoryResult construction")

@dataclass(frozen=True, slots=True)
class ReportResult(Result):
    html: str
    inputs: tuple[AnalysisResult, ...] = ()
    presentation_sha256: str = ""

    def __init__(self) -> None:
        unavailable("ReportResult construction")

    def show(self, **presentation: object) -> HtmlPresentation:
        return HtmlPresentation(self.html)

    def save(self, path: str | PathLike[str]) -> Path:
        target = Path(path)
        if target.suffix != ".html":
            raise ValueError("report path must end in .html")
        if not target.parent.is_dir():
            raise FileNotFoundError(target.parent)
        if target.exists():
            raise FileExistsError(target)
        with NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as temporary:
            temporary.write(self.html)
            temporary.flush()
            fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            link(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
        directory_fd = os_open(target.parent, O_RDONLY)
        try:
            fsync(directory_fd)
        finally:
            from os import close
            close(directory_fd)
        return target
