"""Detached Results for all-storage-off point and parameter-space analysis.

The live execution owner returns typed numerical outcomes and their actual
branch dependencies.  This module owns only public value projections: it
retains no Run, Workspace, backend, receipt identity, or lazy solver callback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import UUID

import numpy as np
from pint import Quantity

from .. import units
from ..authoring import ParameterRef, ParameterSet
from ..canonical import canonical_json_bytes, quantity_from_envelope
from ..construction import unavailable
from ..diagnostics.memory_operation import InMemoryBenchmarkResult
from ..errors import SCNSimStateError
from ..numerics.evidence import numerical_error
from ..numerics.models import EvaluationResult
from ..numeric_encoding import record_document, decode_record
from ..value_storage import immutable_array
from ..visualization.presentation import Theme
from .base import LineDiscretization, MatrixFamilyResult, MatrixView, Result, _fresh_quantity_attribute, _sha256
from .derived import TraceResult
from .factory import _MEMORY_RESULT_TOKEN, _verified_result
from .matrix import OperatorPointResult, ScatteringMatrixResult, _unsupported_static_presentation
from .memory_optimization import _parameter_set

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


def _discretization(rows: object) -> tuple[LineDiscretization, ...] | None:
    if rows is None:
        return None
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("in-memory analysis discretization must be an ordered sequence")
    result: list[LineDiscretization] = []
    for row in rows:
        if isinstance(row, LineDiscretization):
            result.append(row)
            continue
        if not isinstance(row, Mapping):
            raise TypeError("in-memory analysis discretization row is malformed")
        result.append(LineDiscretization(
            component_path=tuple(row["component_path"]),
            kind=row["kind"],
            length=units.registry.Quantity(row["length_m"], "meter"),
            n_sections=row["n_sections"],
            dx=units.registry.Quantity(row["dx_m"], "meter"),
            modal_velocities=tuple(
                units.registry.Quantity(value, "meter / second")
                for value in row.get("modal_velocities_m_s", ())
            ),
            hmax=(None if "hmax_m" not in row else units.registry.Quantity(row["hmax_m"], "meter")),
            policy=row.get("policy"),
        ))
    return tuple(result)


def _parameter_ref(
    record: object,
    parameter_lookup: Mapping[tuple[str, str], ParameterRef],
) -> ParameterRef:
    if not isinstance(record, Mapping):
        raise TypeError("in-memory analysis parameter reference is malformed")
    key = (record["definitions_id"], record["parameter_id"])
    return parameter_lookup[key]


@dataclass(frozen=True, slots=True)
class InMemoryAnalysisDependency(Result):
    """One actual dependency edge and its detached numerical outcome."""

    locator: object
    result: EvaluationResult

    def __init__(self) -> None:
        unavailable("InMemoryAnalysisDependency construction")


@dataclass(frozen=True, slots=True)
class InMemoryAnalysisResult(Result):
    """One detached evaluate result with execution-local provenance only."""

    plan_sha256: str
    request_sha256: str
    execution_id: str
    persistence: Literal["memory"]
    operation: str
    result_kind: str
    spec: Mapping[str, object]
    parameters: ParameterSet
    source_index: int | tuple[int, ...] | None
    origin: Literal["baseline", "point"]
    status: Literal["success", "failure"]
    evaluation: EvaluationResult
    dependencies: tuple[InMemoryAnalysisDependency, ...]
    context: Mapping[str, object]
    program_sha256: str
    cache_hit: bool
    failure: object | None
    discretization: tuple[LineDiscretization, ...] | None
    root: Quantity | None = None
    frequency: Quantity | None = None
    linewidth: Quantity | None = None
    slope: Quantity | None = None
    value: Quantity | None = None
    magnitude: Quantity | None = None
    real: Quantity | None = None
    imag: Quantity | None = None
    zero: Quantity | None = None
    numerator_slope: Quantity | None = None
    denominator: Quantity | None = None
    coupling: Quantity | None = None
    branch_a_residue: Quantity | None = None
    branch_b_residue: Quantity | None = None
    evaluation_omega: Quantity | None = None
    family: Literal["S", "Y", "Z"] | None = None
    evidence: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)
    _presentation: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)
    frequencies: Quantity | None = None
    s: ScatteringMatrixResult | None = None
    y: MatrixFamilyResult | None = None
    z: MatrixFamilyResult | None = None
    traces: Mapping[str, TraceResult] = field(default_factory=dict)
    operator_points: tuple[OperatorPointResult, ...] = ()
    _benchmark: InMemoryBenchmarkResult | None = field(default=None, repr=False, compare=False)
    _failure_error: BaseException | None = field(default=None, init=False, repr=False, compare=False)
    _memory_report_token: object = field(default=None, init=False, repr=False, compare=False)
    _quantity_fields = frozenset({
        "root", "frequency", "linewidth", "slope", "value", "magnitude", "real", "imag",
        "zero", "numerator_slope", "denominator", "coupling", "branch_a_residue",
        "branch_b_residue", "evaluation_omega", "frequencies",
    })

    def __init__(self) -> None:
        unavailable("InMemoryAnalysisResult construction")

    def __getattribute__(self, name: str) -> object:
        if name == "benchmark":
            benchmark = object.__getattribute__(self, "_benchmark")
            if benchmark is None:
                raise RuntimeError("in-memory analysis result finalization is incomplete")
            return benchmark
        if name == "failure" and object.__getattribute__(self, "status") == "failure":
            failure = object.__getattribute__(self, "_failure_error")
            if failure is not None:
                return failure
        return _fresh_quantity_attribute(self, name)

    @property
    def benchmark(self) -> InMemoryBenchmarkResult:
        return object.__getattribute__(self, "_benchmark")  # type: ignore[return-value]

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def persistence_kind(self) -> Literal["memory"]:
        return "memory"

    @property
    def operator_result_kind(self) -> bool:
        return self.result_kind == "operator"

    def at(self, frequency: Quantity) -> OperatorPointResult:
        if self.result_kind != "operator":
            raise TypeError("at() is available only for an in-memory operator result")
        if not isinstance(frequency, Quantity) or np.asarray(frequency.magnitude).ndim != 0:
            raise TypeError("frequency must be one scalar Quantity")
        wanted = float(frequency.to("hertz").magnitude)
        matches = [
            point for point in self.operator_points
            if float(point.frequency.to("hertz").magnitude) == wanted
        ]
        if len(matches) != 1:
            raise KeyError("frequency was not materialized exactly once")
        return matches[0]

    def _family(self, family: Literal["S", "Y", "Z"]) -> MatrixFamilyResult:
        if family not in {"S", "Y", "Z"}:
            raise ValueError("family must be 'S', 'Y', or 'Z'")
        selected = getattr(self, family.lower())
        if selected is None:
            raise SCNSimStateError(
                "matrix family is not present in this in-memory analysis",
                stage="result_access",
                evidence={"result_kind": self.result_kind, "family": family},
            )
        return selected

    def plot(self, *, family: Literal["S", "Y", "Z"] = "S", **presentation: object) -> Figure:
        if self.result_kind == "operator":
            from ..visualization.plots.numerical import operator_plot

            return operator_plot(self, **presentation)
        if self.result_kind == "direct_solve":
            return self._family(family).plot(**presentation)
        from ..visualization.plots.numerical import scalar_plot

        return scalar_plot(self, **presentation)

    def show(self, **presentation: object) -> object:
        if self.result_kind not in {"direct_solve", "operator"}:
            from ..visualization.quantity_html import scalar_show

            return scalar_show(self, **presentation)
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(self, fig: Figure, *, row: int, col: int, **presentation: object) -> Figure:
        if self.result_kind == "operator":
            from ..visualization.plots.numerical import operator_add_to

            return operator_add_to(self, fig, row=row, col=col, **presentation)
        if self.result_kind == "direct_solve":
            return self._family(presentation.pop("family", "S")).add_to(
                fig, row=row, col=col, **presentation
            )
        from ..visualization.plots.numerical import scalar_add_to

        return scalar_add_to(self, fig, row=row, col=col)


@dataclass(frozen=True, slots=True)
class InMemoryAnalysisPointOutcome(Result):
    """One eager ordered sweep occurrence; no synthetic point receipt exists."""

    parameters: ParameterSet
    source_index: int | tuple[int, ...] | None
    origin: Literal["baseline", "point"]
    status: Literal["success", "failure"]
    analysis: InMemoryAnalysisResult

    def __init__(self) -> None:
        unavailable("InMemoryAnalysisPointOutcome construction")

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def failure(self) -> object | None:
        return self.analysis.failure

    @property
    def result(self) -> InMemoryAnalysisResult:
        if not self.succeeded:
            failure = self.analysis.failure
            if failure is None:
                raise RuntimeError("failed in-memory analysis point has no failure")
            raise failure
        return self.analysis


class InMemoryAnalysisPoints(Sequence[InMemoryAnalysisPointOutcome]):
    """Eager point sequence with request-owned grid/list indexing semantics."""

    __slots__ = ("_points", "_kind", "_shape", "_axis_parameters", "_positional")

    def __init__(self) -> None:
        unavailable("InMemoryAnalysisPoints construction")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("in-memory analysis point sequences are immutable")

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self):
        return iter(self._points)

    def __getitem__(self, index: object) -> InMemoryAnalysisPointOutcome:
        if self._positional:
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("selected point indices must be integers")
            ordinal = index + len(self._points) if index < 0 else index
            if ordinal < 0 or ordinal >= len(self._points):
                raise IndexError("selected parameter point index is out of range")
            return self._points[ordinal]
        if self._kind == "grid":
            if len(self._shape) == 1 and isinstance(index, int) and not isinstance(index, bool):
                coordinates = (index,)
            elif isinstance(index, tuple):
                coordinates = index
            else:
                raise TypeError("grid point indices must match the Cartesian rank")
            if len(coordinates) != len(self._shape) or any(
                not isinstance(item, int) or isinstance(item, bool) for item in coordinates
            ):
                raise TypeError("grid point indices must be integer tuples")
            ordinal = 0
            for item, size in zip(coordinates, self._shape, strict=True):
                current = item + size if item < 0 else item
                if current < 0 or current >= size:
                    raise IndexError("parameter grid point index is out of range")
                ordinal = ordinal * size + current
        else:
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("listed point indices must be integers")
            ordinal = index + len(self._points) if index < 0 else index
            if ordinal < 0 or ordinal >= len(self._points):
                raise IndexError("parameter point index is out of range")
        return self._points[ordinal]

    def _plot_rows(self):
        return tuple({
            "source_index": point.source_index,
            "status": point.status,
            "parameters": point.parameters,
            "failure": None if point.succeeded else {
                "kind": getattr(point.failure, "kind", type(point.failure).__name__),
                "stage": getattr(point.failure, "stage", None),
                "message": str(point.failure),
            },
            "execution_id": point.analysis.execution_id,
        } for point in self._points)


@dataclass(frozen=True, slots=True)
class InMemoryParameterField(Result):
    """Detached scalar projection over materialized sweep occurrences."""

    _samples: tuple[Mapping[str, object], ...]
    _kind: str
    _shape: tuple[int, ...]
    _axis_parameters: tuple[ParameterRef, ...]
    quantity: object

    def __init__(self) -> None:
        unavailable("InMemoryParameterField construction")

    @property
    def samples(self) -> tuple[Mapping[str, object], ...]:
        return self._samples

    def plot(self, *, x: ParameterRef, y: ParameterRef | None = None, theme: Theme = Theme.AUTO) -> Figure:
        from ..visualization.plots.parameters import parameter_field_plot

        return parameter_field_plot(self, x=x, y=y, theme=theme)

    def show(self, **presentation: object) -> object:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(self, fig: Figure, *, row: int, col: int, x: ParameterRef, y: ParameterRef | None = None) -> Figure:
        from ..visualization.plots.parameters import parameter_field_add_to

        return parameter_field_add_to(self, fig, row=row, col=col, x=x, y=y)


@dataclass(frozen=True, slots=True)
class InMemoryParameterSweepSelection(Result):
    _parent: InMemoryParameterSweepResult
    points: InMemoryAnalysisPoints

    def __init__(self) -> None:
        unavailable("InMemoryParameterSweepSelection construction")

    def collect(self, *, quantity: object) -> InMemoryParameterField:
        return self._parent._collect(quantity, tuple(self.points))

    def plot(self, *, quantity: object | None = None, x: ParameterRef | None = None,
             y: ParameterRef | None = None, theme: Theme = Theme.AUTO) -> Figure:
        if quantity is None:
            if x is not None or y is not None:
                raise ValueError("x and y require an explicitly collected quantity")
            from ..visualization.plots.parameters import points_plot

            return points_plot(self.points, theme=theme, title="Selected in-memory parameter outcomes")
        if x is None:
            raise ValueError("x is required when plotting a collected quantity")
        return self.collect(quantity=quantity).plot(x=x, y=y, theme=theme)

    def show(self, **presentation: object) -> object:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(self, fig: Figure, *, row: int, col: int, quantity: object | None = None,
               x: ParameterRef | None = None, y: ParameterRef | None = None) -> Figure:
        if quantity is None:
            if x is not None or y is not None:
                raise ValueError("x and y require an explicitly collected quantity")
            from ..visualization.plots.parameters import points_add_to

            return points_add_to(self.points, fig, row=row, col=col)
        if x is None:
            raise ValueError("x is required when adding a collected quantity")
        return self.collect(quantity=quantity).add_to(fig, row=row, col=col, x=x, y=y)


@dataclass(frozen=True, slots=True)
class InMemoryParameterSweepResult(Result):
    """Ordered, fully materialized ParameterSpace result with no receipt IDs."""

    plan_sha256: str
    request_sha256: str
    execution_id: str
    persistence: Literal["memory"]
    operation: str
    result_kind: str
    points: InMemoryAnalysisPoints
    baseline: InMemoryAnalysisPointOutcome | None
    source_kind: Literal["grid", "points"]
    shape: tuple[int, ...]
    axis_parameters: tuple[ParameterRef, ...]
    metadata: Mapping[str, object]
    summary: Mapping[str, object]
    _selector_binding: object = field(repr=False, compare=False)
    _allowed_selectors: frozenset[bytes] = field(repr=False, compare=False)
    _benchmark: InMemoryBenchmarkResult | None = field(default=None, repr=False, compare=False)
    _memory_report_token: object = field(default=None, init=False, repr=False, compare=False)

    def __init__(self) -> None:
        unavailable("InMemoryParameterSweepResult construction")

    @property
    def benchmark(self) -> InMemoryBenchmarkResult:
        benchmark = object.__getattribute__(self, "_benchmark")
        if benchmark is None:
            raise RuntimeError("in-memory sweep result finalization is incomplete")
        return benchmark

    @property
    def persistence_kind(self) -> Literal["memory"]:
        return "memory"

    def collect(self, *, quantity: object) -> InMemoryParameterField:
        return self._collect(quantity, self.points)

    def select(self, *, parameters: ParameterSet) -> InMemoryParameterSweepSelection:
        if not isinstance(parameters, ParameterSet):
            raise TypeError("parameters must be a ParameterSet")
        if parameters.values:
            if not self.points:
                raise ValueError("selection contains a parameter outside this sweep")
            available = next(iter(self.points)).parameters.values
            for parameter in parameters.values:
                match = next((current for current in available if current == parameter), None)
                if match is None or match._definition_record() != parameter._definition_record():
                    raise ValueError("selection contains a parameter outside this sweep")
        selected = []
        for point in self.points:
            for parameter, wanted in parameters.values.items():
                actual = next((current for current in point.parameters.values if current == parameter), None)
                if actual is None or actual._definition_record() != parameter._definition_record():
                    raise TypeError("in-memory sweep points have inconsistent parameter definitions")
                from .sweep import _parameter_value_bytes

                if _parameter_value_bytes(point.parameters.values[actual], actual) != _parameter_value_bytes(wanted, parameter):
                    break
            else:
                selected.append(point)
        points = _make_points(
            tuple(selected), self.source_kind, self.shape, self.axis_parameters,
            positional=True,
        )
        return _verified_result(InMemoryParameterSweepSelection, _parent=self, points=points)

    def _collect(self, quantity: object, points: Sequence[InMemoryAnalysisPointOutcome]) -> InMemoryParameterField:
        if getattr(quantity, "_view", None) is not None:
            raise ValueError("View-bound selectors are supported only by optimization")
        from ..specs import QuantitySelector

        if not isinstance(quantity, QuantitySelector):
            raise TypeError("quantity must be a QuantitySelector")
        selector_binding = object.__getattribute__(self, "_selector_binding")
        selector_key = selector_binding.encode_selector(quantity)
        if selector_key not in object.__getattribute__(self, "_allowed_selectors"):
            raise ValueError("quantity was not requested by this sweep")
        samples = []
        for point in points:
            value = None if not point.succeeded else getattr(point.analysis, quantity.projection, None)
            if point.succeeded and not isinstance(value, Quantity):
                raise ValueError("requested quantity is absent from the point Result family")
            samples.append({
                "parameters": point.parameters,
                "source_index": point.source_index,
                "value": value,
                "failure": point.failure,
                "execution_id": point.analysis.execution_id,
            })
        return _verified_result(
            InMemoryParameterField,
            _samples=tuple(samples),
            _kind=self.source_kind,
            _shape=self.shape,
            _axis_parameters=self.axis_parameters,
            quantity=quantity,
        )

    def plot(self, *, quantity: object | None = None, x: ParameterRef | None = None,
             y: ParameterRef | None = None, theme: Theme = Theme.AUTO) -> Figure:
        if quantity is None:
            if x is not None or y is not None:
                raise ValueError("x and y require an explicitly collected quantity")
            from ..visualization.plots.parameters import points_plot

            return points_plot(self.points, theme=theme)
        if x is None:
            raise ValueError("x is required when plotting a collected quantity")
        return self._collect(quantity, tuple(self.points)).plot(x=x, y=y, theme=theme)

    def show(self, **presentation: object) -> object:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(self, fig: Figure, *, row: int, col: int, quantity: object | None = None,
               x: ParameterRef | None = None, y: ParameterRef | None = None) -> Figure:
        if quantity is None:
            if x is not None or y is not None:
                raise ValueError("x and y require an explicitly collected quantity")
            from ..visualization.plots.parameters import points_add_to

            return points_add_to(self.points, fig, row=row, col=col)
        if x is None:
            raise ValueError("x is required when adding a collected quantity")
        return self._collect(quantity, tuple(self.points)).add_to(fig, row=row, col=col, x=x, y=y)


@dataclass(frozen=True, slots=True)
class InMemoryStaticCapacitanceResult(InMemoryAnalysisResult):
    """Detached static matrix; no Workspace identity or live numerical frame."""

    matrix: Quantity | None = None
    row_coordinates: tuple[str, ...] = ()
    column_coordinates: tuple[str, ...] = ()
    zero_charge: tuple[str, ...] = ()
    _quantity_fields = InMemoryAnalysisResult._quantity_fields | frozenset({"matrix"})
    plot = show = add_to = _unsupported_static_presentation

    def __init__(self) -> None:
        unavailable("InMemoryStaticCapacitanceResult construction")


@dataclass(frozen=True, slots=True)
class InMemoryStaticCapacitanceElementResult(InMemoryAnalysisResult):
    """Detached scalar static coefficient with its actual coordinate pair."""

    plot = show = add_to = _unsupported_static_presentation
    row: str = ""
    column: str = ""
    zero_charge: tuple[str, ...] = ()

    def __init__(self) -> None:
        unavailable("InMemoryStaticCapacitanceElementResult construction")


def _allowed_projections(kind: str) -> tuple[str, ...]:
    return {
        "diagonal_root": ("frequency", "linewidth"),
        "operator_element_root": ("frequency",),
        "hybridized_pole": ("frequency", "linewidth"),
        "transfer_zero": ("frequency",),
        "residue_normalized_coupling": ("real", "imag", "magnitude"),
        "response_element": ("magnitude", "real", "imag"),
    }.get(kind, ())


def _make_points(points, kind, shape, axes, *, positional=False):
    sequence = object.__new__(InMemoryAnalysisPoints)
    object.__setattr__(sequence, "_points", tuple(points))
    object.__setattr__(sequence, "_kind", kind)
    object.__setattr__(sequence, "_shape", tuple(shape))
    object.__setattr__(sequence, "_axis_parameters", tuple(axes))
    object.__setattr__(sequence, "_positional", positional)
    return sequence


def _quantity_from_spec(record: object) -> Quantity:
    if isinstance(record, Quantity):
        return record
    if not isinstance(record, Mapping):
        raise TypeError("in-memory analysis frequency declaration is malformed")
    return quantity_from_envelope(record, registry=units.registry)


def _evaluation_result(
    *, payload: Mapping[str, object], point: Mapping[str, object], request: Mapping[str, object],
    plan_sha256: str, request_sha256: str, execution_id: str,
    parameter_lookup: Mapping[tuple[str, str], ParameterRef],
) -> InMemoryAnalysisResult:
    raw = point["result"]
    if not isinstance(raw, EvaluationResult):
        raise TypeError("in-memory analysis point has no typed EvaluationResult")
    context = point["context"]
    if not isinstance(context, Mapping):
        raise TypeError("in-memory analysis point context is malformed")
    operation = payload["operation"]
    result_kind = payload["result_kind"]
    spec = request["spec"]
    if not isinstance(spec, Mapping):
        raise TypeError("in-memory analysis request Spec is malformed")
    if raw.failure is None:
        status: Literal["success", "failure"] = "success"
    else:
        status = "failure"
    kind = spec["type"]
    grid = _discretization(context.get("discretization"))
    view = context.get("view")
    lineage = context.get("lineage")
    presentation = {
        "spec": spec, "view": view, "ref_lineage": lineage,
        "persistence": "memory", "plan_sha256": plan_sha256,
        "request_sha256": request_sha256, "execution_id": execution_id,
    }
    fields: dict[str, object] = {
        "plan_sha256": plan_sha256,
        "request_sha256": request_sha256,
        "execution_id": execution_id,
        "persistence": "memory",
        "operation": operation,
        "result_kind": result_kind,
        "spec": spec,
        "parameters": _parameter_set(point["parameters"], parameter_lookup),
        "source_index": point["source_index"],
        "origin": point["origin"],
        "status": status,
        "evaluation": raw,
        "dependencies": tuple(
            _verified_result(InMemoryAnalysisDependency, locator=locator, result=dependency)
            for locator, dependency in point["dependencies"]
        ),
        "context": context,
        "program_sha256": point["program_sha256"],
        "cache_hit": point["cache_hit"],
        "failure": raw.failure,
        "discretization": grid,
        "evidence": {},
        "_presentation": presentation,
        "frequencies": None,
        "s": None, "y": None, "z": None, "traces": {}, "operator_points": (),
        "_benchmark": None,
    }
    affine_rows = record_document(raw.evidence_bytes).get("affine_evidence")
    if affine_rows:
        fields["evidence"] = {"affine_evidence": decode_record(affine_rows)}
    if raw.failure is None:
        if kind in {"static_capacitance", "static_capacitance_element"}:
            evidence = decode_record(record_document(raw.evidence_bytes))
            fields.update(evidence=evidence, zero_charge=tuple(evidence["zero_charge"]))
            value = units.registry.Quantity(raw.capacitance_values, "farad")
            if kind == "static_capacitance":
                if np.asarray(value.magnitude).shape != (len(evidence["row_coordinates"]), len(evidence["column_coordinates"])):
                    raise ValueError("Static capacitance shape differs from its compiled coordinate axes")
                fields.update(matrix=value, row_coordinates=tuple(evidence["row_coordinates"]),
                              column_coordinates=tuple(evidence["column_coordinates"]))
            else:
                if np.asarray(value.magnitude).shape != ():
                    raise ValueError("Static capacitance element is not scalar")
                fields.update(value=value, row=spec["row"], column=spec["column"])
        elif kind == "response_element":
            family = spec["family"]
            unit = {"S": "dimensionless", "Y": "siemens", "Z": "ohm"}[family]
            value = raw.response_value
            fields.update(
                family=family,
                value=units.registry.Quantity(value, unit),
                magnitude=units.registry.Quantity(abs(value), unit),
                real=units.registry.Quantity(complex(value).real, unit),
                imag=units.registry.Quantity(complex(value).imag, unit),
            )
        elif kind in {"diagonal_root", "operator_element_root", "hybridized_pole", "transfer_zero"}:
            root_name = "root_omega_rad_s" if kind != "transfer_zero" else "root_omega_rad_s"
            root_value = getattr(raw, root_name)
            root = units.registry.Quantity(root_value, "radian / second")
            from ..execution.quantities import root_frequency_linewidth

            frequency, linewidth = root_frequency_linewidth(root_value)
            fields["_presentation"] = dict(presentation, ref_lineage=lineage)
            if kind == "transfer_zero":
                fields.update(
                    zero=root,
                    frequency=units.registry.Quantity(frequency, "hertz"),
                    numerator_slope=units.registry.Quantity(raw.numerator_slope, "dimensionless"),
                    denominator=units.registry.Quantity(raw.denominator, "dimensionless"),
                    family=spec["family"],
                )
            else:
                fields.update(
                    root=root,
                    frequency=units.registry.Quantity(frequency, "hertz"),
                    linewidth=(units.registry.Quantity(linewidth, "hertz") if kind in {"diagonal_root", "hybridized_pole"} else None),
                    slope=units.registry.Quantity(raw.root_slope, "siemens"),
                )
        elif kind == "residue_normalized_coupling":
            coupling = units.registry.Quantity(raw.coupling_rad_s, "radian / second")
            fields.update(
                coupling=coupling, value=coupling, magnitude=units.registry.Quantity(abs(raw.coupling_rad_s), "radian / second"),
                real=units.registry.Quantity(complex(raw.coupling_rad_s).real, "radian / second"),
                imag=units.registry.Quantity(complex(raw.coupling_rad_s).imag, "radian / second"),
                branch_a_residue=units.registry.Quantity(raw.residue_a, "ohm"),
                branch_b_residue=units.registry.Quantity(raw.residue_b, "ohm"),
                evaluation_omega=units.registry.Quantity(raw.evaluation_omega_rad_s, "radian / second"),
            )
        elif kind == "operator":
            frequencies = tuple(_quantity_from_spec(item) for item in spec["frequencies"])
            values = np.asarray(raw.operator_values)
            if values.shape[0] != len(frequencies):
                raise ValueError("in-memory operator outcomes differ from requested frequencies")
            coordinates = tuple(context["terminal_ids"])
            points = tuple(
                _verified_result(
                    OperatorPointResult,
                    frequency=frequency,
                    matrix=units.registry.Quantity(matrix, "siemens / second"),
                    coordinates=coordinates,
                )
                for frequency, matrix in zip(frequencies, values, strict=True)
            )
            fields.update(operator_points=points)
        elif kind == "direct_solve":
            frequencies = tuple(_quantity_from_spec(item) for item in spec["frequencies"])
            metadata = context["metadata"]
            family_units = metadata["family_units"]
            coordinates = tuple(metadata["terminal_ids"])
            input_channels = tuple((row[0], tuple(row[1])) for row in metadata["input_channels"])
            output_channels = tuple((row[0], tuple(row[1])) for row in metadata["output_channels"])
            matrices = {"S": raw.S, "Y": raw.Y, "Z": raw.Z}
            frequency_axis = units.registry.Quantity(
                np.asarray([float(frequency.to("hertz").magnitude) for frequency in frequencies]),
                "hertz",
            )
            families = {}
            for family, result_type in (("S", ScatteringMatrixResult), ("Y", MatrixFamilyResult), ("Z", MatrixFamilyResult)):
                matrix = np.asarray(matrices[family])
                if len(frequencies) == 1 and matrix.ndim == 2:
                    matrix = matrix[np.newaxis, ...]
                expected_shape = (
                    len(frequencies), len(output_channels), len(input_channels),
                )
                if matrix.shape != expected_shape:
                    raise ValueError(
                        f"in-memory Direct sweep {family} family shape differs from its declared axes"
                    )
                view_result = _verified_result(
                    MatrixView,
                    matrix=units.registry.Quantity(matrix, family_units[family]),
                    frequencies=frequency_axis,
                    coordinates=coordinates,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    probe_loads=metadata["probe_loads"],
                )
                families[family] = _verified_result(
                    result_type, view=view_result, _parent_identity=None,
                    _presentation=dict(presentation, family=family),
                )
            s_values = np.asarray(families["S"].view.matrix.magnitude)
            traces = {}
            for declaration in spec["traces"]:
                out = coordinates.index(declaration["output_port"])
                inp = coordinates.index(declaration["input_port"])
                traces[declaration["id"]] = _verified_result(
                    TraceResult, frequencies=frequency_axis,
                    value=units.registry.Quantity(s_values[:, out, inp], "dimensionless"),
                    _parent_identity=None,
                    _presentation=dict(presentation, id=declaration["id"], family="S"),
                )
            fields.update(frequencies=frequency_axis, s=families["S"], y=families["Y"], z=families["Z"], traces=traces)
        else:
            raise ValueError(f"unsupported in-memory analysis Spec type: {kind}")
    result_type = {"static_capacitance": InMemoryStaticCapacitanceResult,
                   "static_capacitance_element": InMemoryStaticCapacitanceElementResult}.get(kind, InMemoryAnalysisResult)
    if raw.failure is not None and kind in {"static_capacitance", "static_capacitance_element"}:
        # Failure facts remain readable; no failed numerical placeholder is usable.
        fields.update(evidence=decode_record(record_document(raw.failure.evidence_bytes)),
                      zero_charge=tuple(context["zero_charge"]))
        if kind == "static_capacitance":
            fields.update(row_coordinates=tuple(context["row_coordinates"]),
                          column_coordinates=tuple(context["column_coordinates"]))
        if kind == "static_capacitance_element":
            fields.update(row=spec["row"], column=spec["column"])
    result = _verified_result(result_type, **fields)
    object.__setattr__(result, "_failure_error", None if raw.failure is None else numerical_error(raw.failure))
    object.__setattr__(result, "_memory_report_token", _MEMORY_RESULT_TOKEN)
    return result


def build_memory_analysis_result(
    *, payload: Mapping[str, object], request: Mapping[str, object],
    plan_sha256: str, request_sha256: str, execution_id: str,
    parameter_lookup: Mapping[tuple[str, str], ParameterRef],
    selector_binding: object,
    benchmark: InMemoryBenchmarkResult | None = None,
) -> InMemoryAnalysisResult | InMemoryParameterSweepResult:
    if not isinstance(payload, Mapping) or payload.get("persistence") != "memory":
        raise TypeError("in-memory analysis payload has no memory persistence authority")
    if payload.get("execution_id") != execution_id:
        raise ValueError("in-memory analysis payload belongs to another execution")
    if payload.get("operation") != request.get("operation"):
        raise ValueError("in-memory analysis payload operation differs from its request")
    _sha256(plan_sha256, name="plan_sha256")
    _sha256(request_sha256, name="request_sha256")
    UUID(execution_id)
    if not isinstance(parameter_lookup, Mapping):
        raise TypeError("in-memory analysis requires the captured parameter lookup")
    from ..execution.prepared import SelectorBinding

    if not isinstance(selector_binding, SelectorBinding):
        raise TypeError("in-memory analysis requires its captured selector binding")
    request_view = request.get("view")
    if not isinstance(request_view, Mapping) or selector_binding.view_bytes != canonical_json_bytes(request_view):
        raise ValueError("in-memory selector binding belongs to another prepared View")
    source = request["parameter_source"]
    if not isinstance(source, Mapping):
        raise TypeError("in-memory analysis request has no parameter source")
    if source["kind"] == "point":
        rows = payload["points"]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != 1:
            raise ValueError("point analysis payload does not contain its one requested outcome")
        point = rows[0]
        result = _evaluation_result(
            payload=payload, point=point, request=request, plan_sha256=plan_sha256,
            request_sha256=request_sha256, execution_id=execution_id,
            parameter_lookup=parameter_lookup,
        )
        return result
    if source["kind"] not in {"grid", "points"}:
        raise ValueError("in-memory analysis parameter source has an unsupported kind")
    shape = tuple(source["shape"]) if source["kind"] == "grid" else ()
    axes = tuple(_parameter_ref(axis["parameter"], parameter_lookup) for axis in source["axes"]) if source["kind"] == "grid" else ()
    baseline_payload = payload["baseline"]
    baseline = None if baseline_payload is None else _point_outcome(
        payload, baseline_payload, request, plan_sha256, request_sha256,
        execution_id, parameter_lookup,
    )
    raw_points = payload["points"]
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        raise TypeError("in-memory analysis points must be an ordered sequence")
    points = tuple(
        _point_outcome(payload, point, request, plan_sha256, request_sha256,
                       execution_id, parameter_lookup)
        for point in raw_points
    )
    point_sequence = _make_points(points, source["kind"], shape, axes)
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("in-memory analysis metadata is malformed")
    request_spec = request.get("spec")
    if not isinstance(request_spec, Mapping):
        raise TypeError("in-memory analysis request Spec is malformed")
    selector_kind = {
        "diagonal_root": "diagonal_root_projection",
        "operator_element_root": "operator_element_root_projection",
        "hybridized_pole": "hybridized_pole_projection",
        "transfer_zero": "transfer_zero_projection",
        "residue_normalized_coupling": "residue_coupling_projection",
        "response_element": "response_element_projection",
    }.get(request_spec.get("type"))
    projections = _allowed_projections(str(request_spec.get("type")))
    allowed_selectors = frozenset(
        canonical_json_bytes({
            "type": selector_kind,
            "spec": request_spec,
            "projection": projection,
        })
        for projection in projections
    )
    result = _verified_result(
        InMemoryParameterSweepResult,
        plan_sha256=plan_sha256,
        request_sha256=request_sha256,
        execution_id=execution_id,
        persistence="memory",
        operation=payload["operation"],
        result_kind=payload["result_kind"],
        points=point_sequence,
        baseline=baseline,
        source_kind=source["kind"],
        shape=shape,
        axis_parameters=axes,
        metadata=metadata,
        summary=payload["summary"],
        _selector_binding=selector_binding,
        _allowed_selectors=allowed_selectors,
        _benchmark=benchmark,
    )
    object.__setattr__(result, "_memory_report_token", _MEMORY_RESULT_TOKEN)
    return result


def _point_outcome(payload, point, request, plan_sha256, request_sha256, execution_id, parameter_lookup):
    analysis = _evaluation_result(
        payload=payload, point=point, request=request, plan_sha256=plan_sha256,
        request_sha256=request_sha256, execution_id=execution_id,
        parameter_lookup=parameter_lookup,
    )
    return _verified_result(
        InMemoryAnalysisPointOutcome,
        parameters=analysis.parameters,
        source_index=point["source_index"],
        origin=point["origin"],
        status=analysis.status,
        analysis=analysis,
    )


def attach_memory_analysis_benchmark(result, benchmark: InMemoryBenchmarkResult):
    if type(result) not in {
        InMemoryAnalysisResult, InMemoryParameterSweepResult,
        InMemoryStaticCapacitanceResult, InMemoryStaticCapacitanceElementResult,
    }:
        raise TypeError("benchmark can be attached only to an in-memory analysis Result")
    if not isinstance(benchmark, InMemoryBenchmarkResult):
        raise TypeError("in-memory analysis requires its operation benchmark")
    if object.__getattribute__(result, "_benchmark") is not None:
        raise RuntimeError("in-memory analysis benchmark was already finalized")
    object.__setattr__(result, "_benchmark", benchmark)
    return result


__all__ = [
    "InMemoryAnalysisDependency", "InMemoryAnalysisPointOutcome", "InMemoryAnalysisResult",
    "InMemoryAnalysisPoints", "InMemoryParameterField", "InMemoryParameterSweepResult",
    "InMemoryParameterSweepSelection", "attach_memory_analysis_benchmark",
    "build_memory_analysis_result",
]
