"""Receipt-backed results and downstream presentation values.

Public constructors deliberately fail: a user cannot manufacture a result that
looks like verified workspace evidence.  ``_verified_result`` is the narrow
decoder hook used after workspace receipt/artifact verification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, FrozenInstanceError, dataclass, field, fields
from enum import Enum
from html import escape
from os import O_RDONLY, PathLike, fsync, link, open as os_open
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import TypeVar, Literal

import numpy as np
from pint import Quantity

from . import units
from ._scaffold import unavailable
from .authoring import ParameterSet
from .errors import HBCaseFailure


T = TypeVar("T")


def _freeze(value: object) -> object:
    """Detach mutable decoder payloads before exposing a Result surface."""

    if isinstance(value, np.ndarray):
        copy = np.array(value, copy=True)
        copy.setflags(write=False)
        return copy
    if isinstance(value, Quantity):
        if value._REGISTRY is not units.registry:
            raise TypeError("result quantities must use scnsim.units")
        magnitude = value.magnitude
        if isinstance(magnitude, np.ndarray):
            magnitude = _freeze(magnitude)
        return units.registry.Quantity(magnitude, value.units)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _verified_result(cls: type[T], /, **values: object) -> T:
    """Private verified-decoder hook; never call this on unverified evidence.

    The caller supplies exactly the public dataclass fields for ``cls``.  The
    hook validates receipt identity fields and recursively detaches mappings,
    sequences, and NumPy/Pint arrays before the value becomes user-visible.
    """

    if not isinstance(cls, type) or not issubclass(cls, (Result, MatrixView, ResultIdentity, ReconciliationEvidence, OptimizationBest, OperatorPointResult)):
        raise TypeError("_verified_result only constructs SCNSim result values")
    if cls is HBCaseOutcome:
        expected = {
            "id", "failure", "effective_sources", "operating_point_closure", "bias_state", "pump_state", "s", "y", "z", "traces", "states", "state_node_map",
        }
        if set(values) != expected:
            raise TypeError("verified HBCaseOutcome fields mismatch")
        failure = values["failure"]
        success = failure is None
        if not isinstance(values["id"], str) or not values["id"]:
            raise ValueError("HB case id must be nonempty")
        if failure is not None and not isinstance(failure, HBCaseFailure):
            raise TypeError("HB failure must be HBCaseFailure")
        required = ("operating_point_closure", "bias_state", "pump_state", "s", "y", "z", "traces", "states", "state_node_map")
        if success != all(values[name] is not None for name in required):
            raise ValueError("HB success must provide every success-only surface")
        if not success and any(values[name] is not None for name in required):
            raise ValueError("HB failure must not retain success-only surfaces")
        if not isinstance(values["effective_sources"], (tuple, list)):
            raise TypeError("HB outcome effective_sources must be an ordered sequence")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_id", values["id"])
        object.__setattr__(instance, "_failure", failure)
        for name in ("effective_sources", "operating_point_closure", "bias_state", "pump_state", "s", "y", "z", "traces", "states", "state_node_map"):
            object.__setattr__(instance, f"_{name}", _freeze(values[name]))
        return instance
    if cls is HBBatchResult:
        if set(values) != {"identity", "cases", "topology_evidence"}:
            raise TypeError("verified HBBatchResult fields mismatch")
        identity, cases = values["identity"], values["cases"]
        if not _is_verified_identity(identity) or not isinstance(cases, Mapping) or not cases or not isinstance(values["topology_evidence"], Mapping):
            raise TypeError("verified HBBatchResult requires identity and nonempty cases")
        materialized = dict(cases)
        if any(
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(outcome, HBCaseOutcome)
            or outcome.id != identifier
            for identifier, outcome in materialized.items()
        ):
            raise TypeError("verified HBBatchResult cases are malformed")
        instance = object.__new__(cls)
        object.__setattr__(instance, "identity", identity)
        object.__setattr__(instance, "cases", MappingProxyType(materialized))
        object.__setattr__(instance, "topology_evidence", _freeze(values["topology_evidence"]))
        object.__setattr__(instance, "_verified_result_token", _VERIFIED_TOKEN)
        return instance
    expected = {item.name: item for item in fields(cls) if item.init}
    missing = set(expected) - set(values)
    extra = set(values) - set(expected)
    required_missing = {
        name for name in missing
        if expected[name].default is MISSING and expected[name].default_factory is MISSING
    }
    if required_missing or extra:
        raise TypeError(f"verified {cls.__name__} fields mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for name in missing:
        descriptor = expected[name]
        values[name] = descriptor.default_factory() if descriptor.default_factory is not MISSING else descriptor.default
    if cls is ResultIdentity:
        for name, value in values.items():
            _sha256(value, name=name)
    if issubclass(cls, AnalysisResult) and not _is_verified_identity(values.get("identity")):
        raise TypeError("analysis results require a verified ResultIdentity")
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, _freeze(value))
    if cls is ResultIdentity:
        object.__setattr__(instance, "_verified_identity_token", _VERIFIED_TOKEN)
    if issubclass(cls, AnalysisResult):
        object.__setattr__(instance, "_verified_result_token", _VERIFIED_TOKEN)
    return instance


_VERIFIED_TOKEN = object()


def _is_verified_identity(value: object) -> bool:
    return type(value) is ResultIdentity and getattr(value, "_verified_identity_token", None) is _VERIFIED_TOKEN


def _is_verified_analysis_result(value: object) -> bool:
    return (
        type(value) in (
            DirectSolveResult,
            DiagonalRootResult,
            DirectQuantityResult,
            OperatorResult,
            OptimizationResult,
            HBBatchResult,
        )
        and getattr(value, "_verified_result_token", None) is _VERIFIED_TOKEN
        and _is_verified_identity(getattr(value, "identity", None))
    )


@dataclass(frozen=True, slots=True)
class HtmlPresentation:
    """Small self-contained HTML display object for notebook and headless use."""

    html: str

    def _repr_html_(self) -> str:
        return self.html

    def __str__(self) -> str:
        return self.html


class BiasState(Enum):
    OFF = "off"
    ON = "on"


class PumpState(Enum):
    OFF = "off"
    ON = "on"


@dataclass(frozen=True, slots=True)
class Result:
    """Base role shared by immutable, already-materialized SCNSim values."""

    def __init__(self) -> None:
        unavailable(f"{type(self).__name__} construction")

    def show(self, **presentation: object) -> object:
        return HtmlPresentation(f"<pre>{escape(repr(self))}</pre>")


@dataclass(frozen=True, slots=True)
class ResultIdentity:
    """Immutable Plan/request/attempt/result hashes from one verified receipt."""

    plan_sha256: str
    request_sha256: str
    attempt_sha256: str
    result_sha256: str
    _verified_identity_token: object = field(init=False, repr=False, compare=False)

    def __init__(self) -> None:
        unavailable("ResultIdentity construction")


@dataclass(frozen=True, slots=True)
class AnalysisResult(Result):
    """Receipt-backed terminal Result returned by solve, evaluate, or optimize."""

    identity: ResultIdentity
    _verified_result_token: object = field(init=False, repr=False, compare=False)

    def __init__(self) -> None:
        unavailable(f"{type(self).__name__} construction")


@dataclass(frozen=True, slots=True)
class MatrixView:
    """Labeled selected-network matrix data with immutable array payloads."""

    matrix: Quantity
    frequencies: Quantity
    coordinates: tuple[str, ...]
    input_channels: tuple[tuple[str, tuple[int, ...]], ...] = ()
    output_channels: tuple[tuple[str, tuple[int, ...]], ...] = ()
    probe_loads: Mapping[str, Literal["raw", "compensated"]] = field(default_factory=dict)

    def __init__(self) -> None:
        unavailable("MatrixView construction")


@dataclass(frozen=True, slots=True)
class MatrixFamilyResult(Result):
    """One typed matrix family on an immutable selected-network View."""

    view: MatrixView

    def __init__(self) -> None:
        unavailable(f"{type(self).__name__} construction")


@dataclass(frozen=True, slots=True)
class ScatteringMatrixResult(MatrixFamilyResult):
    """Selected-view generalized power-wave S matrices and presentation."""

    def __init__(self) -> None:
        unavailable("ScatteringMatrixResult construction")

    def show(self, *, magnitude: Literal["linear", "db"] = "linear") -> object:
        if magnitude not in {"linear", "db"}:
            raise ValueError("magnitude must be 'linear' or 'db'")
        import matplotlib.pyplot as plt

        matrix = np.asarray(getattr(self.view.matrix, "magnitude", self.view.matrix))
        if matrix.ndim != 3:
            raise ValueError("S matrix must have [frequency, output, input] axes")
        values = matrix[:, 0, 0]
        shown_magnitude = np.abs(values)
        if magnitude == "db":
            shown_magnitude = 20.0 * np.log10(shown_magnitude)
        frequencies = np.asarray(self.view.frequencies.magnitude)
        figure, (upper, lower) = plt.subplots(2, 1, sharex=True)
        upper.plot(frequencies, shown_magnitude)
        phase = np.where(np.abs(values) == 0.0, np.nan, np.angle(values, deg=True))
        lower.plot(frequencies, phase)
        upper.set_ylabel("|S| (dB)" if magnitude == "db" else "|S|")
        lower.set_ylabel("phase (deg; exact zero undefined)")
        lower.set_xlabel("frequency")
        return figure


@dataclass(frozen=True, slots=True)
class ReconciliationEvidence:
    comparable: bool
    reason: str | None
    last_comparable_ancestor: str
    residual: float | None
    evidence_sha256: str

    def __init__(self) -> None:
        unavailable("ReconciliationEvidence construction")


@dataclass(frozen=True, slots=True)
class HBScatteringMatrixResult(ScatteringMatrixResult):
    backend_native: MatrixView | None = None
    reconciliation: ReconciliationEvidence | None = None

    def __init__(self) -> None:
        unavailable("HBScatteringMatrixResult construction")


@dataclass(frozen=True, slots=True)
class DirectSolveResult(AnalysisResult):
    """Complete finite Direct S/Y/Z response from one verified receipt."""

    frequencies: Quantity
    s: ScatteringMatrixResult
    y: MatrixFamilyResult
    z: MatrixFamilyResult
    traces: Mapping[str, TraceResult] = field(default_factory=dict)

    def __init__(self) -> None:
        unavailable("DirectSolveResult construction")


@dataclass(frozen=True, slots=True)
class DirectQuantityResult(AnalysisResult):
    """One verified scalar Direct quantity and its contract-defined evidence."""

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
    family: Literal["S", "Y", "Z"] | None = None

    def __init__(self) -> None:
        unavailable(f"{type(self).__name__} construction")


@dataclass(frozen=True, slots=True)
class DiagonalRootResult(DirectQuantityResult):
    """Loaded root and local slope evidence from one diagonal-root request."""

    root: Quantity
    frequency: Quantity
    linewidth: Quantity
    slope: Quantity

    def __init__(self) -> None:
        unavailable("DiagonalRootResult construction")


@dataclass(frozen=True, slots=True)
class OperatorPointResult:
    """One verified labeled selected-network operator at one frequency."""

    frequency: Quantity
    matrix: Quantity
    coordinates: tuple[str, ...]

    def __init__(self) -> None:
        unavailable("OperatorPointResult construction")


@dataclass(frozen=True, slots=True)
class OperatorResult(AnalysisResult):
    """Verified selected-network operator points in declared frequency order."""

    points: tuple[OperatorPointResult, ...]

    def __init__(self) -> None:
        unavailable("OperatorResult construction")

    def at(self, frequency: Quantity) -> OperatorPointResult:
        """Return the already materialized point at exactly ``frequency``."""

        for point in self.points:
            if point.frequency == frequency:
                return point
        raise KeyError("frequency was not materialized")


@dataclass(frozen=True, slots=True)
class OptimizationBest:
    """Lowest finite-cost baseline or population candidate in ledger order."""

    parameters: ParameterSet
    cost: float

    def __init__(self) -> None:
        unavailable("OptimizationBest construction")


@dataclass(frozen=True, slots=True)
class OptimizationResult(AnalysisResult):
    """Verified CMA winner and immutable completed-generation ledgers."""

    best: OptimizationBest
    ledger: tuple[Mapping[str, object], ...] = ()

    def __init__(self) -> None:
        unavailable("OptimizationResult construction")


class HBCaseOutcome(Result):
    """One named success or receipt-backed numerical HB failure.

    ``states`` and ``state_node_map`` are operating-point evidence only, never
    View coordinates or current-drive targets.
    """

    __slots__ = (
        "_id", "_failure", "_effective_sources", "_operating_point_closure", "_bias_state", "_pump_state", "_s", "_y", "_z",
        "_traces", "_states", "_state_node_map",
    )

    def __init__(self) -> None:
        unavailable("HBCaseOutcome construction")

    def __setattr__(self, name: str, value: object) -> None:
        raise FrozenInstanceError("cannot assign to field of immutable HBCaseOutcome")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError("cannot delete field of immutable HBCaseOutcome")

    @property
    def id(self) -> str:
        return self._id

    @property
    def succeeded(self) -> bool:
        return self._failure is None

    @property
    def failure(self) -> HBCaseFailure | None:
        return self._failure

    @property
    def effective_sources(self) -> tuple[Mapping[str, object], ...]:
        """Return the exact case drive evidence, including on failed cases."""

        return tuple(_freeze(source) for source in self._effective_sources)

    @property
    def operating_point_closure(self) -> Mapping[str, object]:
        """Return the verified nonlinear closure evidence for a successful case."""

        return self._success(_freeze(self._operating_point_closure))  # type: ignore[return-value]

    def _success(self, value: object) -> object:
        if self._failure is not None:
            raise self._failure
        return value

    @property
    def bias_state(self) -> BiasState:
        return self._success(self._bias_state)  # type: ignore[return-value]

    @property
    def pump_state(self) -> PumpState:
        return self._success(self._pump_state)  # type: ignore[return-value]

    @property
    def s(self) -> HBScatteringMatrixResult:
        return self._success(self._s)  # type: ignore[return-value]

    @property
    def y(self) -> MatrixFamilyResult:
        return self._success(self._y)  # type: ignore[return-value]

    @property
    def z(self) -> MatrixFamilyResult:
        return self._success(self._z)  # type: ignore[return-value]

    @property
    def traces(self) -> Mapping[str, TraceResult]:
        return self._success(self._traces)  # type: ignore[return-value]

    @property
    def states(self) -> Quantity:
        return self._success(self._states)  # type: ignore[return-value]

    @property
    def state_node_map(self) -> tuple[Mapping[str, object], ...]:
        return self._success(self._state_node_map)  # type: ignore[return-value]

    def show(self, *, magnitude: Literal["linear", "db"] = "linear") -> object:
        if self._failure is not None:
            failure = self._failure
            return HtmlPresentation(
                f"<h3>HB case {escape(self.id)}</h3><p>failure: "
                f"kind={escape(failure.kind)}; stage={escape(failure.stage)}; "
                f"message={escape(str(failure))}</p>"
            )
        return self.s.show(magnitude=magnitude)


@dataclass(frozen=True, slots=True)
class HBBatchResult(AnalysisResult):
    cases: Mapping[str, HBCaseOutcome]
    topology_evidence: Mapping[str, object]

    def __init__(self) -> None:
        unavailable("HBBatchResult construction")

    def show(self, *, magnitude: Literal["linear", "db"] = "linear") -> object:
        if magnitude not in {"linear", "db"}:
            raise ValueError("magnitude must be 'linear' or 'db'")
        successes = tuple(outcome for outcome in self.cases.values() if outcome.succeeded)
        failures = tuple(outcome for outcome in self.cases.values() if not outcome.succeeded)
        if not successes:
            rows = "".join(
                f"<li>{escape(outcome.id)}: kind={escape(outcome.failure.kind)}; "
                f"stage={escape(outcome.failure.stage)}; "
                f"message={escape(str(outcome.failure))}</li>"
                for outcome in failures
            )
            return HtmlPresentation(f"<h3>HB cases</h3><ul>{rows}</ul>")

        import matplotlib.pyplot as plt

        trace_ids = tuple(successes[0].traces)
        if trace_ids:
            panels: tuple[tuple[str, object], ...] = tuple((identifier, identifier) for identifier in trace_ids)
        else:
            view = successes[0].s.view
            panels = tuple(
                (
                    f"S[{output_coordinate},{output_mode} <- {input_coordinate},{input_mode}]",
                    (output_index, input_index),
                )
                for output_index, (output_coordinate, output_mode) in enumerate(view.output_channels)
                for input_index, (input_coordinate, input_mode) in enumerate(view.input_channels)
            )
        figure, axes = plt.subplots(len(panels), 2, squeeze=False, sharex=True)
        for (magnitude_axis, phase_axis), (panel_label, selector) in zip(axes, panels):
            for outcome in successes:
                if trace_ids:
                    trace = outcome.traces[selector]  # type: ignore[index]
                    frequency = np.asarray(trace.frequencies.magnitude)
                    values = np.asarray(trace.value.magnitude)
                else:
                    frequency = np.asarray(outcome.s.view.frequencies.magnitude)
                    output_index, input_index = selector  # type: ignore[misc]
                    values = np.asarray(outcome.s.view.matrix.magnitude)[:, output_index, input_index]
                shown = np.abs(values)
                if magnitude == "db":
                    shown = 20.0 * np.log10(shown)
                phase = np.where(np.abs(values) == 0.0, np.nan, np.angle(values, deg=True))
                magnitude_axis.plot(frequency, shown, label=outcome.id)
                phase_axis.plot(frequency, phase, label=outcome.id)
            magnitude_axis.set_ylabel(f"{panel_label} (dB)" if magnitude == "db" else panel_label)
            phase_axis.set_ylabel("phase (deg; exact zero undefined)")
            magnitude_axis.legend()
            phase_axis.legend()
        axes[-1, 0].set_xlabel("frequency")
        axes[-1, 1].set_xlabel("frequency")
        if failures:
            figure.suptitle(
                "failures: " + "; ".join(
                    f"{outcome.id}: kind={outcome.failure.kind}, "
                    f"stage={outcome.failure.stage}, message={outcome.failure}"
                    for outcome in failures
                )
            )
        return figure


@dataclass(frozen=True, slots=True)
class TraceResult(Result):
    frequencies: Quantity
    value: Quantity

    def __init__(self) -> None:
        unavailable("TraceResult construction")

    def show(self, *, magnitude: Literal["linear", "db"] = "linear") -> object:
        if magnitude not in {"linear", "db"}:
            raise ValueError("magnitude must be 'linear' or 'db'")
        import matplotlib.pyplot as plt

        values = np.abs(np.asarray(self.value.magnitude))
        if magnitude == "db":
            values = 20.0 * np.log10(values)
        figure, axis = plt.subplots()
        axis.plot(np.asarray(self.frequencies.magnitude), values)
        return figure


@dataclass(frozen=True, slots=True)
class ExplanationResult(Result):
    evidence: Mapping[str, object]

    def __init__(self) -> None:
        unavailable("ExplanationResult construction")

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
        lineage = evidence.get("ref_lineage", {})
        hierarchy = evidence.get("component_hierarchy", ())
        parameters = evidence.get("parameters", {}).get("bindings", ()) if isinstance(evidence.get("parameters"), Mapping) else ()
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
        if isinstance(compiled, Mapping):
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

    def __init__(self) -> None:
        unavailable("InventoryResult construction")


@dataclass(frozen=True, slots=True)
class ReportResult(Result):
    html: str
    inputs: tuple[AnalysisResult, ...] = ()

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


@dataclass(frozen=True, slots=True)
class CircuitDiagramResult(Result):
    drawing: object
    representation: Literal["authoring", "compiled"] = "authoring"

    def __init__(self) -> None:
        unavailable("CircuitDiagramResult construction")

    def show(self, **presentation: object) -> object:
        return self.drawing
