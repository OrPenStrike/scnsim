"""Public Plan/Ref/Run execution declarations.

The runtime separates three beginner-facing ideas:

``CircuitPlan``
    The model developer's one physical circuit definition.
``NetworkViewRef``
    A lazy immutable description of which topology compensation, coordinates,
    and external boundary an analysis should use.
``CircuitRun``
    The only object allowed to compile, solve, evaluate, optimize, resolve, or
    assemble durable evidence.

Nothing in this scaffold executes.  Every action fails loudly so an importable
package cannot be mistaken for a working simulator.
"""

from __future__ import annotations

from os import PathLike
from typing import overload

from ._scaffold import unavailable
from .authoring import (
    CircuitPlan,
    CoordinateRef,
    ElectricNodeRef,
    ParameterSet,
    PortRef,
)
from .results import (
    DirectQuantityResult,
    DiagonalRootResult,
    DirectSolveResult,
    ExplanationResult,
    HBBatchResult,
    InventoryResult,
    OperatorResult,
    OptimizationResult,
    ReportResult,
)
from .specs import (
    DiagonalRootSpec,
    DirectSolveSpec,
    HBSolveSpec,
    HybridizedPoleSpec,
    OptimizationSpec,
    OperatorSpec,
    ReportSpec,
    ResidueNormalizedCouplingSpec,
    ResponseElementSpec,
    TransferZeroSpec,
)


class ReductionPipeline:
    """Declarative ordered steps used to derive a lazy ``NetworkViewRef``.

    ``ptc()`` changes effective analysis topology by compensating evidenced
    nonloading-probe Port loads.  ``transform_pair()`` changes one ordered pair
    of power-conjugate node coordinates.  ``retain()`` selects an ordered node
    view and eliminates its complement at zero external injection.  The same
    lineage is backend-neutral; a terminal operation checks whether its
    selected coordinates are realizable by Direct or HB.  Declaring steps
    never compiles, solves, or writes a receipt.
    """

    def __init__(self) -> None:
        unavailable("ReductionPipeline construction")

    def ptc(self, *ports: PortRef) -> ReductionPipeline:
        """Compensate one or more evidenced ``nonloading_probe`` port shunts."""

        unavailable("ReductionPipeline.ptc")

    def transform_pair(
        self,
        node_a: str | ElectricNodeRef | CoordinateRef,
        node_b: str | ElectricNodeRef | CoordinateRef,
        *,
        id: str,
    ) -> ReductionPipeline:
        """Create automatic common/differential coordinates for one node pair."""

        unavailable("ReductionPipeline.transform_pair")

    def retain(
        self,
        *coordinates: str | ElectricNodeRef | CoordinateRef,
    ) -> ReductionPipeline:
        """Retain ordered node coordinates; eliminate the complement at I=0."""

        unavailable("ReductionPipeline.retain")


class NetworkViewRef:
    """Immutable lazy reference to one Plan plus one ordered reduction lineage.

    A Ref answers "which analysis view?" It owns neither a compiled operator nor
    a result.  Reuse and branch Refs freely, then pass the chosen Ref to one
    terminal ``CircuitRun`` operation.
    """

    def __init__(self) -> None:
        unavailable("NetworkViewRef construction")

    def reduce(self, pipeline: ReductionPipeline) -> NetworkViewRef:
        """Return a child Ref without executing or mutating this lineage."""

        unavailable("NetworkViewRef.reduce")

class CircuitRun:
    """Execution namespace for one sealed Plan and one evidence workspace.

    A circuit-model developer supplies the Plan; a model user chooses a Ref,
    a Spec, and optional ``ParameterSet``.  ``solve()``, ``evaluate()``, and
    ``optimize()`` execute.  ``resolve()`` only verifies and loads the exact
    prior receipt.  Results are returned directly and are never stored as a
    mutable current result on the Run or Ref.

    ``versioned=False`` uses the replaceable Notebook-workbench policy.
    ``versioned=True`` preserves distinct Plan identities in iteration
    subworkspaces.  Construction remains unavailable in the current scaffold.
    """

    def __init__(
        self,
        *,
        plan: CircuitPlan,
        workspace: str | PathLike[str],
        versioned: bool = False,
    ) -> None:
        unavailable("CircuitRun construction")

    @property
    def original(self) -> NetworkViewRef:
        """Zero-reduction root Ref for the sealed physical Plan."""

        unavailable("CircuitRun.original")

    @overload
    def solve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> DirectSolveResult: ...

    @overload
    def solve(
        self,
        ref: NetworkViewRef,
        spec: HBSolveSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> HBBatchResult: ...

    def solve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> DirectSolveResult | HBBatchResult:
        """Execute or reuse one sealed solve on a Port-realizable selected view."""

        unavailable("CircuitRun.solve")

    @overload
    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: DiagonalRootSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> DiagonalRootResult: ...

    @overload
    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: OperatorSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> OperatorResult: ...

    @overload
    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: (
            HybridizedPoleSpec
            | TransferZeroSpec
            | ResidueNormalizedCouplingSpec
            | ResponseElementSpec
        ),
        *,
        parameters: ParameterSet | None = None,
    ) -> DirectQuantityResult: ...

    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: (
            DiagonalRootSpec
            | HybridizedPoleSpec
            | TransferZeroSpec
            | ResidueNormalizedCouplingSpec
            | ResponseElementSpec
            | OperatorSpec
        ),
        *,
        parameters: ParameterSet | None = None,
    ) -> DiagonalRootResult | DirectQuantityResult | OperatorResult:
        """Evaluate one typed Direct quantity or materialize its full operator.

        This public method does not imply per-candidate Python execution:
        ``optimize()`` evaluates the same quantity semantics inside one Julia
        process.
        """

        unavailable("CircuitRun.evaluate")

    def optimize(
        self,
        ref: NetworkViewRef,
        spec: OptimizationSpec,
    ) -> OptimizationResult:
        """Execute one Direct-only multi-variable, multi-objective search."""

        unavailable("CircuitRun.optimize")

    @overload
    def resolve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> DirectSolveResult: ...

    @overload
    def resolve(
        self,
        ref: NetworkViewRef,
        spec: DiagonalRootSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> DiagonalRootResult: ...

    @overload
    def resolve(
        self,
        ref: NetworkViewRef,
        spec: OptimizationSpec,
    ) -> OptimizationResult: ...

    @overload
    def resolve(
        self,
        ref: NetworkViewRef,
        spec: HBSolveSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> HBBatchResult: ...

    @overload
    def resolve(
        self,
        ref: NetworkViewRef,
        spec: OperatorSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> OperatorResult: ...

    @overload
    def resolve(
        self,
        ref: NetworkViewRef,
        spec: (
            HybridizedPoleSpec
            | TransferZeroSpec
            | ResidueNormalizedCouplingSpec
            | ResponseElementSpec
        ),
        *,
        parameters: ParameterSet | None = None,
    ) -> DirectQuantityResult: ...

    def resolve(
        self,
        ref: NetworkViewRef,
        spec: (
            DirectSolveSpec
            | HBSolveSpec
            | DiagonalRootSpec
            | HybridizedPoleSpec
            | TransferZeroSpec
            | ResidueNormalizedCouplingSpec
            | ResponseElementSpec
            | OperatorSpec
            | OptimizationSpec
        ),
        *,
        parameters: ParameterSet | None = None,
    ) -> (
        DirectSolveResult
        | HBBatchResult
        | DiagonalRootResult
        | DirectQuantityResult
        | OperatorResult
        | OptimizationResult
    ):
        """Verify and load the exact prior result without executing or finding latest.

        ``parameters`` must be omitted for ``OptimizationSpec`` because
        ``optimize()`` always starts from the sealed Plan baseline.
        """

        unavailable("CircuitRun.resolve")

    @overload
    def explain(
        self,
        ref: NetworkViewRef,
        spec: OptimizationSpec,
    ) -> ExplanationResult: ...

    @overload
    def explain(
        self,
        ref: NetworkViewRef,
        spec: (
            DirectSolveSpec
            | HBSolveSpec
            | DiagonalRootSpec
            | HybridizedPoleSpec
            | TransferZeroSpec
            | ResidueNormalizedCouplingSpec
            | ResponseElementSpec
            | OperatorSpec
        ),
        *,
        parameters: ParameterSet | None = None,
    ) -> ExplanationResult: ...

    def explain(
        self,
        ref: NetworkViewRef,
        spec: (
            DirectSolveSpec
            | HBSolveSpec
            | DiagonalRootSpec
            | HybridizedPoleSpec
            | TransferZeroSpec
            | ResidueNormalizedCouplingSpec
            | ResponseElementSpec
            | OperatorSpec
            | OptimizationSpec
        ),
        *,
        parameters: ParameterSet | None = None,
    ) -> ExplanationResult:
        """Compile deterministic preflight evidence without solver/workspace writes.

        ``parameters`` must be omitted for ``OptimizationSpec`` because the
        optimization baseline is the sealed Plan baseline.
        """

        unavailable("CircuitRun.explain")

    def inventory(self) -> InventoryResult:
        """List exact request identities in the workspace without selecting one."""

        unavailable("CircuitRun.inventory")

    def build_report(self, spec: ReportSpec) -> ReportResult:
        """Assemble explicitly supplied existing Results; never run an analysis."""

        unavailable("CircuitRun.build_report")
