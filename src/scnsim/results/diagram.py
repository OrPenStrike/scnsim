from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
import json
from typing import TYPE_CHECKING, Literal
from collections.abc import Mapping

from ..canonical import float64_from_hex
from ..construction import unavailable
from .base import HtmlPresentation, Result, _freeze

if TYPE_CHECKING:
    import schemdraw
    from ..visualization.composition.api import SchematicCompositionSnapshot

@dataclass(frozen=True, slots=True)
class CircuitDiagramAudit:
    """Read-only certificate for one frozen, independently checked scene."""

    _data: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        unavailable("CircuitDiagramAudit construction")

    @classmethod
    def _from_data(cls, data: object) -> "CircuitDiagramAudit":
        from ..visualization.diagram.audit import DiagramAuditData

        if not isinstance(data, DiagramAuditData):
            raise TypeError("CircuitDiagramAudit requires DiagramAuditData")
        data = DiagramAuditData(
            representation=data.representation,
            plan_id=data.plan_id,
            plan_sha256=data.plan_sha256,
            connectivity_sha256=data.connectivity_sha256,
            semantic_sha256=data.semantic_sha256,
            compiled_graph_sha256=data.compiled_graph_sha256,
            expanded_graph_sha256=data.expanded_graph_sha256,
            presentation_sha256=data.presentation_sha256,
            observed_electrical=_freeze(data.observed_electrical),
            observed_semantic=_freeze(data.observed_semantic),
            observed_rows=_freeze(data.observed_rows),
            verified_rows=_freeze(data.verified_rows),
        )
        result = object.__new__(cls)
        object.__setattr__(result, "_data", data)
        return result

    @property
    def representation(self) -> Literal["authoring", "compiled"]:
        return self._data.representation

    @property
    def plan_id(self) -> str:
        return self._data.plan_id

    @property
    def plan_sha256(self) -> str:
        return self._data.plan_sha256

    @property
    def connectivity_sha256(self) -> str:
        return self._data.connectivity_sha256

    @property
    def semantic_sha256(self) -> str:
        return self._data.semantic_sha256

    @property
    def compiled_graph_sha256(self) -> str | None:
        return self._data.compiled_graph_sha256

    @property
    def expanded_graph_sha256(self) -> str | None:
        return self._data.expanded_graph_sha256

    @property
    def presentation_sha256(self) -> str | None:
        return self._data.presentation_sha256

    def show(self) -> "HtmlPresentation":
        """Present observed scene rows separately from verified snapshot facts."""

        from ..canonical import float64_from_hex

        def text(value: object) -> str:
            """Format certified records without exposing encoder JSON as UI."""

            if value is None:
                return "—"
            if isinstance(value, str):
                if value.startswith("{") and value.endswith("}"):
                    try:
                        decoded = json.loads(value)
                    except json.JSONDecodeError:
                        return value
                    if isinstance(decoded, Mapping):
                        return text(decoded)
                return value
            if isinstance(value, (int, float, bool)):
                return str(value)
            if isinstance(value, Mapping):
                if value.get("type") == "quantity_f64":
                    encoded, unit = value.get("si_value_f64"), value.get("si_unit")
                    if not isinstance(encoded, str) or not isinstance(unit, str):
                        raise ValueError("certificate contains a malformed canonical quantity")
                    try:
                        return f"{float64_from_hex(encoded)!r} {unit}"
                    except ValueError as exc:
                        raise ValueError("certificate contains an invalid canonical quantity") from exc
                return "; ".join(
                    f"{key.replace('_', ' ')}={text(item)}" for key, item in value.items()
                ) or "—"
            if isinstance(value, (tuple, list)):
                return ", ".join(text(item) for item in value) or "—"
            return type(value).__name__

        def table(title: str, rows: object) -> str:
            entries = tuple(row for row in rows if isinstance(row, Mapping)) if isinstance(rows, (tuple, list)) else ()
            columns = tuple(dict.fromkeys(key for row in entries for key in row))
            if not columns:
                return f"<h4>{escape(title)}</h4><p>None</p>"
            body = "".join(
                "<tr>" + "".join(
                    f"<td>{escape(text(row.get(column)))}</td>"
                    for column in columns
                ) + "</tr>"
                for row in entries if isinstance(row, Mapping)
            )
            if not body:
                body = f"<tr><td colspan=\"{len(columns)}\">None</td></tr>"
            headings = "".join(f"<th>{escape(column.replace('_', ' '))}</th>" for column in columns)
            return f"<h4>{escape(title)}</h4><table><tr>{headings}</tr>{body}</table>"

        def values_by_identity(value: object) -> tuple[Mapping[str, object], ...]:
            if not isinstance(value, Mapping):
                return ()
            return tuple({"identity": identity, "value": item} for identity, item in value.items())

        def detail_rows(kind: str) -> tuple[Mapping[str, object], ...]:
            return tuple(
                row for row in self._data.observed_rows
                if isinstance(row, Mapping) and row.get("kind") == kind
            )

        def verified_rows(kind: str) -> tuple[Mapping[str, object], ...]:
            return tuple(
                row
                for row in self._data.verified_rows
                if isinstance(row, Mapping) and row.get("kind") == kind
            )

        def verified_record_rows(kind: str) -> tuple[Mapping[str, object], ...]:
            """Expose complete captured source records without calling them ink."""

            rows: list[Mapping[str, object]] = []
            for row in verified_rows(kind):
                record = row.get("record")
                rows.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"category", "kind"}
                    }
                    if isinstance(record, Mapping)
                    else dict(row)
                )
            return tuple(rows)

        electrical = self._data.observed_electrical
        semantic = self._data.observed_semantic
        identity_row = next(iter(verified_rows("identity")), {})
        identity = dict(identity_row)
        identity["presentation_sha256"] = self.presentation_sha256
        point_values = next(iter(verified_rows("canonical_point_values")), {})
        compiled_expansion = next(iter(verified_rows("compiled_expansion")), {})
        return HtmlPresentation(
            "<section><h3>Observed electrical reconstruction (audit A)</h3>"
            + table("Nets and exact contacts", electrical.get("nets", ()) if isinstance(electrical, Mapping) else ())
            + table("Native physical bodies: visible contacts, references, and ownership", electrical.get("bodies", ()) if isinstance(electrical, Mapping) else ())
            + table("Visible conductive junctions", electrical.get("junctions", ()) if isinstance(electrical, Mapping) else ())
            + table("Ports: role, raw Z0, orientation, and contacts", electrical.get("ports", ()) if isinstance(electrical, Mapping) else ())
            + table("Local physical ground glyphs and returns", electrical.get("grounds", ()) if isinstance(electrical, Mapping) else ())
            + table("Transmission lines: visible CPW endpoints and ordered MTL conductors", electrical.get("transmission_lines", ()) if isinstance(electrical, Mapping) else ())
            + table("Observed mutual couplings", electrical.get("couplings", ()) if isinstance(electrical, Mapping) else ())
            + table("Displayed coupling coefficients and visible polarity", detail_rows("mutual_coupling"))
            + table("Exact-zero omission evidence", electrical.get("omissions", ()) if isinstance(electrical, Mapping) else ())
            + "<h3>Observed visible structure (audit B)</h3>"
            + table("Visible regions, headers, and containment", semantic.get("regions", ()) if isinstance(semantic, Mapping) else ())
            + table("Visible leaf ownership", semantic.get("leaf_ownership", ()) if isinstance(semantic, Mapping) else ())
            + table("Visible Port ownership", semantic.get("port_ownership", ()) if isinstance(semantic, Mapping) else ())
            + table("Visible cross-boundary electrical incidence", semantic.get("boundary_incidence", ()) if isinstance(semantic, Mapping) else ())
            + "<h3>Visible public-analysis-label observations</h3>"
            + table("Exact text, owner scope, and attached electrical net", semantic.get("public_analysis_labels", ()) if isinstance(semantic, Mapping) else ())
            + "<h3>Observed displayed-point evidence</h3>"
            + table("Displayed selected physical values", detail_rows("displayed_parameter_value"))
            + table("Displayed Port reference impedances", detail_rows("port_impedance"))
            + table("Displayed baseline values retained by a compiled ledger", detail_rows("displayed_baseline_value"))
            + table("Full compiled rows", detail_rows("compiled_matrix_row"))
            + table("All observed reconstruction rows", self._data.observed_rows)
            + "<h3>Verified captured point evidence (separate from audit A/B)</h3>"
            + table("Certificate identities, including complete effective parameters", (identity,))
            + table("Verified selected-point physical values", values_by_identity(point_values.get("values") if isinstance(point_values, Mapping) else None))
            + "<h3>Verified captured source records (not inferred from the drawing)</h3>"
            + table("Authored operators", verified_record_rows("authored_operator"))
            + table("Authored buses and taps", verified_record_rows("authored_bus"))
            + table("Authored public exposures", verified_record_rows("authored_exposure"))
            + table("Authored node aliases", verified_record_rows("authored_node_alias"))
            + table("Parameter definitions", verified_record_rows("parameter_definition"))
            + table("Physical parameter field bindings", verified_record_rows("parameter_field_binding"))
            + table("Complete effective parameter point", verified_record_rows("effective_parameter_point"))
            + table("Source-unit records", verified_record_rows("source_unit"))
            + table("Ground-call records", verified_record_rows("ground_call_group"))
            + table("Verified compiled expansion bindings", (compiled_expansion,))
            + "</section>"
        )

@dataclass(frozen=True, slots=True)
class CircuitDiagramResult(Result):
    drawing: schemdraw.Drawing
    audit: CircuitDiagramAudit
    composition: SchematicCompositionSnapshot | None = None

    def __init__(self) -> None:
        unavailable("CircuitDiagramResult construction")

    def show(self) -> schemdraw.Drawing:
        return self.drawing
