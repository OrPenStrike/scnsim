"""SCNSim — Superconducting Circuit Network Simulation.

Numerical execution remains separately accepted where documented.  Structured
authoring, parameter closure, and semantic diagram capture are an actively
converging V1 candidate.

Start with :class:`CircuitPlan` if you develop reusable circuit models.  Start
with :class:`CircuitRun` plus a model package supplied by your team if you only
consume an existing model.
"""

from importlib import import_module
from importlib.metadata import version as metadata_version
from typing import TYPE_CHECKING

from . import units
from .authoring import (
    RLGC,
    AffineMap,
    CircuitPlan,
    ComponentInstance,
    CompositePlan,
    CoordinateRef,
    ElectricNodeRef,
    InductiveBranchRef,
    Library,
    ParameterDefinitions,
    ParameterRef,
    ParameterSet,
    ParameterSpace,
    ParameterSpec,
    RLGCParameterSpec,
    BusRef,
    TapRef,
    TwoTerminalUse,
    GroundRef,
    SeriesRef,
    ParallelRef,
    BranchRef,
    LinkRef,
    CouplingRef,
    PinRef,
    PortRef,
    SubsystemPlan,
    components,
)
from .errors import (
    BackendProtocolError,
    CompilerInvariantError,
    DirectResponseFormationError,
    EliminatedBlockSolveFailure,
    EvidenceIntegrityError,
    HBCaseFailure,
    InvalidCandidatePhysicalParameter,
    InvalidDiagonalRootHint,
    InvalidOptimizationSpec,
    NumericalResolutionUnresolved,
    PlanSealedError,
    PortRealizabilityError,
    ResultUnavailableError,
    RootSlopeUnresolved,
    RuntimePreparationError,
    ScaffoldUnavailableError,
    SCNSimCapabilityError,
    SCNSimError,
    SCNSimEvidenceError,
    SCNSimExecutionError,
    SCNSimStateError,
    SCNSimValidationError,
    UnsupportedEvidenceVersionError,
    UnsupportedRuntimePlatformError,
    UnsupportedSingularCapacitanceForDiagonalRootV1,
    WorkspacePlanReplacedError,
    WorkspaceCommitIndeterminateError,
    WorkspaceVersioningDowngradeForbidden,
)
from .io import load_q2d_rlgc
from .presentation import Theme
from .results import (
    AnalysisResult,
    BiasState,
    DiagonalRootResult,
    DirectQuantityResult,
    DirectSolveResult,
    ExplanationResult,
    HBBatchResult,
    HBCaseOutcome,
    HBScatteringMatrixResult,
    InventoryResult,
    MatrixFamilyResult,
    MatrixView,
    OperatorPointResult,
    OperatorResult,
    OptimizationBest,
    OptimizationResult,
    ParameterField,
    ParameterPointAccessor,
    ParameterPointIdentity,
    ParameterPointOutcome,
    ParameterSweepResult,
    ParameterSweepSelection,
    PumpState,
    ReconciliationEvidence,
    ReportResult,
    Result,
    ResultIdentity,
    ScatteringMatrixResult,
    TraceResult,
)
from .runtime import CircuitRun, NetworkViewRef, ReductionPipeline
from .specs import (
    CMAESSpec,
    CostObjective,
    CurrentDrive,
    DiagonalRootSpec,
    DirectSolveSpec,
    HBCaseSpec,
    HBSolveSpec,
    HBTruncation,
    HybridizedPoleSpec,
    OperatorSpec,
    OptimizationSpec,
    OptimizationVariable,
    PumpAxis,
    QuantitySum,
    ReportSpec,
    ResidueNormalizedCouplingSpec,
    ResponseElementSpec,
    SParameterTrace,
    TransferZeroSpec,
)


# Diagram declarations remain lazy at runtime so numerical consumers do not
# import rendering modules.  These explicit type-only reexports keep the
# documented package names visible to static ``py.typed`` consumers.
if TYPE_CHECKING:
    from ._diagram_spec import CircuitDiagramSpec
    from .composition import SchematicComposition, SchematicCompositionSnapshot
    from .results import CircuitDiagramAudit, CircuitDiagramResult
    from .schematic import DiagramAxis, SchematicLayout
    from .specs import DiagramSide


_LAZY_DIAGRAM_EXPORTS = {
    "CircuitDiagramAudit": (".results", "CircuitDiagramAudit"),
    "CircuitDiagramResult": (".results", "CircuitDiagramResult"),
    "CircuitDiagramSpec": (".specs", "CircuitDiagramSpec"),
    "DiagramAxis": (".schematic", "DiagramAxis"),
    "DiagramSide": (".specs", "DiagramSide"),
    "SchematicComposition": (".composition", "SchematicComposition"),
    "SchematicCompositionSnapshot": (".composition", "SchematicCompositionSnapshot"),
    "SchematicLayout": (".schematic", "SchematicLayout"),
}


def __getattr__(name: str) -> object:
    """Resolve diagram presentation names only when a caller asks for one."""

    target = _LAZY_DIAGRAM_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__version__ = metadata_version("scnsim")

__all__ = [
    "RLGC",
    "AffineMap",
    "AnalysisResult",
    "BackendProtocolError",
    "BiasState",
    "BranchRef",
    "BusRef",
    "CMAESSpec",
    "CircuitDiagramAudit",
    "CircuitDiagramResult",
    "CircuitDiagramSpec",
    "CircuitPlan",
    "CircuitRun",
    "CompilerInvariantError",
    "ComponentInstance",
    "CompositePlan",
    "CoordinateRef",
    "CouplingRef",
    "CostObjective",
    "CurrentDrive",
    "DiagonalRootResult",
    "DiagonalRootSpec",
    "DirectQuantityResult",
    "DirectResponseFormationError",
    "DirectSolveResult",
    "DirectSolveSpec",
    "DiagramSide",
    "ElectricNodeRef",
    "EliminatedBlockSolveFailure",
    "EvidenceIntegrityError",
    "ExplanationResult",
    "GroundRef",
    "HBBatchResult",
    "HBCaseFailure",
    "HBCaseOutcome",
    "HBCaseSpec",
    "HBScatteringMatrixResult",
    "HBSolveSpec",
    "HBTruncation",
    "HybridizedPoleSpec",
    "InductiveBranchRef",
    "InvalidCandidatePhysicalParameter",
    "InvalidDiagonalRootHint",
    "InvalidOptimizationSpec",
    "InventoryResult",
    "Library",
    "MatrixFamilyResult",
    "MatrixView",
    "NetworkViewRef",
    "NumericalResolutionUnresolved",
    "OperatorPointResult",
    "OperatorResult",
    "OperatorSpec",
    "OptimizationBest",
    "OptimizationResult",
    "OptimizationSpec",
    "OptimizationVariable",
    "ParameterField",
    "ParameterRef",
    "ParameterDefinitions",
    "ParameterPointAccessor",
    "ParameterPointIdentity",
    "ParameterPointOutcome",
    "ParameterSet",
    "ParameterSpace",
    "ParameterSpec",
    "ParameterSweepResult",
    "ParameterSweepSelection",
    "ParallelRef",
    "PinRef",
    "PlanSealedError",
    "PortRealizabilityError",
    "PortRef",
    "PumpAxis",
    "PumpState",
    "QuantitySum",
    "ReconciliationEvidence",
    "ReductionPipeline",
    "RLGCParameterSpec",
    "ReportResult",
    "ReportSpec",
    "ResidueNormalizedCouplingSpec",
    "ResponseElementSpec",
    "Result",
    "ResultIdentity",
    "ResultUnavailableError",
    "RootSlopeUnresolved",
    "RuntimePreparationError",
    "SCNSimCapabilityError",
    "SCNSimError",
    "SCNSimEvidenceError",
    "SCNSimExecutionError",
    "SCNSimStateError",
    "SCNSimValidationError",
    "SParameterTrace",
    "DiagramAxis",
    "SchematicLayout",
    "SchematicComposition",
    "SchematicCompositionSnapshot",
    "SeriesRef",
    "ScaffoldUnavailableError",
    "ScatteringMatrixResult",
    "TraceResult",
    "Theme",
    "SubsystemPlan",
    "TapRef",
    "TwoTerminalUse",
    "TransferZeroSpec",
    "UnsupportedEvidenceVersionError",
    "UnsupportedRuntimePlatformError",
    "UnsupportedSingularCapacitanceForDiagonalRootV1",
    "WorkspacePlanReplacedError",
    "WorkspaceCommitIndeterminateError",
    "WorkspaceVersioningDowngradeForbidden",
    "components",
    "load_q2d_rlgc",
    "units",
]
