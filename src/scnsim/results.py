"""Typed result-surface declarations for the SCNSim V1 scaffold.

Results are immutable, already-materialized values. Analysis Results additionally
bind verified receipts and artifacts. They never become mutable state on a
``CircuitRun`` or ``NetworkViewRef``. The properties below document the intended
Notebook discovery surface; every access fails until its implementation exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Literal

from ._scaffold import unavailable
from .authoring import ParameterSet


class BiasState(Enum):
    """Whether a bound HB case has a nonzero effective all-zero-mode drive."""

    OFF = "off"
    ON = "on"


class PumpState(Enum):
    """Whether a bound HB case has a nonzero effective nonzero-mode drive."""

    OFF = "off"
    ON = "on"


class Result:
    """Base role shared by immutable, already-materialized SCNSim values.

    Reading or presenting a Result never triggers another computation. Derived
    Results need not pretend to own an analysis request or receipt.
    """

    def __init__(self) -> None:
        unavailable(f"{type(self).__name__} construction")

    def show(self, **presentation: object) -> object:
        """Present already-materialized data without solving or interpolation."""

        unavailable(f"{type(self).__name__}.show")


class ResultIdentity:
    """Immutable hashes for one receipt-backed terminal analysis Result."""

    def __init__(self) -> None:
        unavailable("ResultIdentity construction")

    @property
    def plan_sha256(self) -> str:
        unavailable("ResultIdentity.plan_sha256")

    @property
    def request_sha256(self) -> str:
        unavailable("ResultIdentity.request_sha256")

    @property
    def attempt_sha256(self) -> str:
        unavailable("ResultIdentity.attempt_sha256")

    @property
    def result_sha256(self) -> str:
        unavailable("ResultIdentity.result_sha256")


class AnalysisResult(Result):
    """Receipt-backed terminal Result returned by solve/evaluate/optimize."""

    @property
    def identity(self) -> ResultIdentity:
        """Exact Plan, request, attempt, and Result hashes."""

        unavailable(f"{type(self).__name__}.identity")


class MatrixView:
    """Labeled data for one de-embedded selected SCNSim network.

    ``matrix`` is the complex array (Quantity-valued for dimensionful
    families); ``frequencies`` and ``coordinates`` keep its axes explicit.
    Source-loaded/raw operators remain diagnostics rather than this public
    S/Y/Z surface.  This data object never performs a solve or interpolation.
    """

    def __init__(self) -> None:
        unavailable("MatrixView construction")

    @property
    def matrix(self) -> object:
        """Complex matrix or matrix stack in declared coordinate order."""

        unavailable("MatrixView.matrix")

    @property
    def frequencies(self) -> object:
        """Exact Quantity-valued frequency grid carried by the matrix stack."""

        unavailable("MatrixView.frequencies")

    @property
    def coordinates(self) -> tuple[str, ...]:
        """Ordered selected-view coordinate IDs labeling both matrix axes."""

        unavailable("MatrixView.coordinates")

    @property
    def probe_loads(self) -> Mapping[str, Literal["raw", "compensated"]]:
        """Read-only logical-Port IDs and each probe load's exact view state."""

        unavailable("MatrixView.probe_loads")


class MatrixFamilyResult(Result):
    """One selected-view matrix family, such as Y or Z, over a solved grid."""

    @property
    def view(self) -> MatrixView:
        """Return the labeled selected-view matrix data."""

        unavailable(f"{type(self).__name__}.view")


class ScatteringMatrixResult(MatrixFamilyResult):
    """Selected-view generalized power-wave S matrices and presentation."""

    def show(
        self,
        *,
        magnitude: Literal["linear", "db"] = "linear",
    ) -> object:
        """Display magnitude and wrapped phase without changing result identity."""

        unavailable(f"{type(self).__name__}.show")


class ReconciliationEvidence:
    """Typed comparison between selected-view and backend-native S evidence."""

    def __init__(self) -> None:
        unavailable("ReconciliationEvidence construction")

    @property
    def comparable(self) -> bool:
        """Whether topology, boundary, reference, frequency, and basis align."""

        unavailable("ReconciliationEvidence.comparable")

    @property
    def reason(self) -> str | None:
        """Exact mismatch reason when ``comparable`` is false."""

        unavailable("ReconciliationEvidence.reason")

    @property
    def last_comparable_ancestor(self) -> object:
        """Evidence identity for the last lineage point that remained comparable."""

        unavailable("ReconciliationEvidence.last_comparable_ancestor")


class HBScatteringMatrixResult(ScatteringMatrixResult):
    """HB S matrices with backend-native and reconciliation evidence."""

    @property
    def backend_native(self) -> MatrixView:
        """Backend-returned matrix with its native basis and normalization."""

        unavailable("HBScatteringMatrixResult.backend_native")

    @property
    def reconciliation(self) -> ReconciliationEvidence:
        """Return typed comparability and conversion/residual evidence."""

        unavailable("HBScatteringMatrixResult.reconciliation")


class DirectSolveResult(AnalysisResult):
    """Linear Direct response over the requested grid and selected view.

    ``s``, ``y``, and ``z`` expose parallel labeled matrix-family surfaces.
    The Result has no fake HB case layer.
    """

    @property
    def frequencies(self) -> object:
        """Exact Quantity-valued grid used by this Direct solve."""

        unavailable("DirectSolveResult.frequencies")

    @property
    def s(self) -> ScatteringMatrixResult:
        """Selected-view S matrices and presentation helpers."""

        unavailable("DirectSolveResult.s")

    @property
    def y(self) -> MatrixFamilyResult:
        """Selected-view Quantity-valued admittance matrices."""

        unavailable("DirectSolveResult.y")

    @property
    def z(self) -> MatrixFamilyResult:
        """Selected-view Quantity-valued impedance matrices."""

        unavailable("DirectSolveResult.z")

    @property
    def traces(self) -> Mapping[str, TraceResult]:
        """Ordered read-only mapping of declared Direct trace ID to its result."""

        unavailable("DirectSolveResult.traces")


class DirectQuantityResult(AnalysisResult):
    """One evaluated Direct root, pole, zero, coupling, or response scalar.

    The exact public properties depend on the originating Spec.  Root-like
    results expose ``root``, ``frequency``, ``linewidth``, and applicable slope
    evidence; response/coupling results expose their declared complex scalar
    and typed projections.
    """

    @property
    def root(self) -> object:
        """Complex angular-frequency Quantity when the originating Spec has one."""

        unavailable("DirectQuantityResult.root")

    @property
    def frequency(self) -> object:
        """Physical frequency Quantity derived from the selected complex root."""

        unavailable("DirectQuantityResult.frequency")

    @property
    def linewidth(self) -> object:
        """Positive passive linewidth Quantity when defined by the Spec."""

        unavailable("DirectQuantityResult.linewidth")

    @property
    def slope(self) -> object:
        """Local nonzero-derivative evidence for the machine-resolved root."""

        unavailable("DirectQuantityResult.slope")

    @property
    def value(self) -> object:
        """Declared complex response or coupling Quantity when applicable."""

        unavailable("DirectQuantityResult.value")

    @property
    def magnitude(self) -> object:
        """Magnitude projection of the declared complex scalar when applicable."""

        unavailable("DirectQuantityResult.magnitude")

    @property
    def real(self) -> object:
        """Real projection of the declared complex scalar when applicable."""

        unavailable("DirectQuantityResult.real")

    @property
    def imag(self) -> object:
        """Imaginary projection of the declared complex scalar when applicable."""

        unavailable("DirectQuantityResult.imag")


class DiagonalRootResult(DirectQuantityResult):
    """Loaded root, frequency, linewidth, and local slope from one root request."""

    @property
    def root(self) -> object:
        """Machine-resolved complex angular-frequency Quantity."""

        unavailable("DiagonalRootResult.root")

    @property
    def frequency(self) -> object:
        """Physical frequency Quantity derived from the resolved root."""

        unavailable("DiagonalRootResult.frequency")

    @property
    def linewidth(self) -> object:
        """Positive passive linewidth Quantity from the same root."""

        unavailable("DiagonalRootResult.linewidth")

    @property
    def slope(self) -> object:
        """Local nonzero complex slope evidence at the resolved root."""

        unavailable("DiagonalRootResult.slope")


class OperatorPointResult:
    """One exact frequency slice of a labeled Direct dynamic operator."""

    def __init__(self) -> None:
        unavailable("OperatorPointResult construction")

    @property
    def frequency(self) -> object:
        """Exact Quantity-valued requested frequency; never interpolated."""

        unavailable("OperatorPointResult.frequency")

    @property
    def matrix(self) -> object:
        """Coordinate- and unit-bearing complex dynamic-operator matrix."""

        unavailable("OperatorPointResult.matrix")

    @property
    def coordinates(self) -> tuple[str, ...]:
        """Ordered coordinates labeling the operator rows and columns."""

        unavailable("OperatorPointResult.coordinates")


class OperatorResult(AnalysisResult):
    """Labeled, unit-bearing Direct dynamic operator materialized on a grid."""

    def at(self, frequency: object) -> OperatorPointResult:
        """Select an exact requested grid point; never interpolate."""

        unavailable("OperatorResult.at")


class OptimizationBest:
    """Best evaluated candidate and its immutable physical ``ParameterSet``."""

    def __init__(self) -> None:
        unavailable("OptimizationBest construction")

    @property
    def parameters(self) -> ParameterSet:
        """Values that may be passed explicitly to another Direct or HB request."""

        unavailable("OptimizationBest.parameters")


class OptimizationResult(AnalysisResult):
    """Auditable Direct-only search result with candidate and failure ledgers."""

    @property
    def best(self) -> OptimizationBest:
        """Return the best successfully evaluated candidate, not stale fallback."""

        unavailable("OptimizationResult.best")


class HBCaseResult(Result):
    """One named HB operating condition from a shared ``HBBatchResult``.

    It exposes selected-view S/Y/Z matrices, named trace projections, and
    derived Bias/Pump classifications.  Its case ID is the only lookup key.
    """

    @property
    def bias_state(self) -> BiasState:
        """Derived DC-bias classification after source-vector summation."""

        unavailable("HBCaseResult.bias_state")

    @property
    def pump_state(self) -> PumpState:
        """Derived nonzero-mode classification after source-vector summation."""

        unavailable("HBCaseResult.pump_state")

    @property
    def s(self) -> HBScatteringMatrixResult:
        """Selected-view, backend-native, and reconciliation S evidence."""

        unavailable("HBCaseResult.s")

    @property
    def y(self) -> MatrixFamilyResult:
        """Selected-view Quantity-valued admittance matrices."""

        unavailable("HBCaseResult.y")

    @property
    def z(self) -> MatrixFamilyResult:
        """Selected-view Quantity-valued impedance matrices."""

        unavailable("HBCaseResult.z")

    @property
    def traces(self) -> Mapping[str, TraceResult]:
        """Ordered read-only mapping of declared trace ID to materialized trace."""

        unavailable("HBCaseResult.traces")


class HBBatchResult(AnalysisResult):
    """Ordered collection of user-named HB cases sharing one basis and request.

    Use ``hb.cases[id]``.  The batch deliberately has no ``hb[id]`` shortcut
    and no special ``pump_on`` collection because many independently named
    driven conditions may all derive ``PumpState.ON``.
    """

    @property
    def cases(self) -> Mapping[str, HBCaseResult]:
        """Ordered read-only mapping keyed only by declared case ID."""

        unavailable("HBBatchResult.cases")

    def show(
        self,
        *,
        magnitude: Literal["linear", "db"] = "linear",
    ) -> object:
        """Overlay declared cases using named traces or the full matrix fallback."""

        unavailable("HBBatchResult.show")


class TraceResult(Result):
    """One named complex S projection over the parent solve's frequency grid."""

    @property
    def frequencies(self) -> object:
        """Exact Quantity-valued grid inherited from the parent solve."""

        unavailable("TraceResult.frequencies")

    @property
    def value(self) -> object:
        """Complex trace values in declared frequency order."""

        unavailable("TraceResult.value")

    def show(
        self,
        *,
        magnitude: Literal["linear", "db"] = "linear",
    ) -> object:
        """Display magnitude and wrapped phase without recomputation."""

        unavailable("TraceResult.show")


class ExplanationResult(Result):
    """Deterministic compile preflight with no solver or workspace write."""


class InventoryResult(Result):
    """Listing of exact request identities known to a workspace, never latest."""


class ReportResult(Result):
    """Pure derived report assembled from explicit ``AnalysisResult`` inputs."""

    def save(self, path: str | PathLike[str]) -> Path:
        """Write one new self-contained HTML file atomically.

        The parent must exist, the suffix must be ``.html``, and an existing
        target is never overwritten. The returned ``Path`` names the new file.
        Invalid suffix, missing parent, and existing target use ``ValueError``,
        ``FileNotFoundError``, and ``FileExistsError`` respectively.
        """

        unavailable("ReportResult.save")


class CircuitDiagramResult(Result):
    """Materialized authoring or compiler-expanded Plan diagram.

    A compiled result records Plan, compiler, and expanded-graph identities.
    Neither representation contains a reduced equivalent or backend row and
    neither becomes topology or numerical-result authority.
    """

    def show(self) -> object:
        """Present this materialized diagram without compiling or analyzing."""

        unavailable("CircuitDiagramResult.show")
