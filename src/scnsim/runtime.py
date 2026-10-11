"""Plan-bound execution, exact request identity, and typed Result reconstruction.

One captured authoring snapshot and its resolved parameter points feed the
declarative Direct, HB, and optimization request boundary. Workspace receipts
and artifact manifests remain the only authority for reconstructing results.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import overload
from time import perf_counter_ns
from weakref import ref as weakref
from uuid import uuid4

from .workspace.operation_lease import operation_lease

from .execution.run_lifecycle import RunLifecycle, run_activity
from .execution.run_binding import capture_run_binding, compatible_parameter, _parameter_key
from .execution.request_binding import _view_declaration, _quantity_selectors, source_units, prepare_request
from .execution.view_binding import ViewDeclaration, derive_view, coordinate_id, _coordinate_id, _original_lineage_document


from . import units
from .execution.compilation import _run_preflight
from .execution.identity import _runtime_identity_base
from .execution.prepared import (
    BoundOptimization,
    BoundOptimizationLeaf,
    PreparedAnalysis,
    _coordinate_binding_key,
    _encode_direct_quantity,
    _encode_spec,
    _quantity_coordinates,
)
from .canonical import (
    _identifier as _canonical_identifier,
    canonical_json_bytes,
    sha256_hex,
)
from .workspace.artifacts import _VerifiedEvidenceLease, _verified_evidence_lease
from .authoring.resolution import resolve_parameter_point
from .authoring.physical_values import RLGCParameterSpec
from .construction import unavailable
from .workspace import VerifiedSuccess, bind_workspace
from .authoring import (
    CircuitPlan,
    CoordinateRef,
    ElectricNodeRef,
    ParameterRef,
    ParameterSet,
    ParameterSpace,
    PortRef,
)
from .errors import (
    CompilerInvariantError,
    EvidenceIntegrityError,
    InvalidOptimizationSpec,
    PortRealizabilityError,
    SCNSimValidationError,
    RuntimePreparationError,
)
from .results import (
    DiagonalRootResult,
    OperatorElementRootResult,
    DirectQuantityResult,
    DirectSolveResult,
    ExplanationResult,
    HBBatchResult,
    InventoryResult,
    OperatorResult,
    OptimizationResult,
    ParameterSweepResult,
    ReportResult,
    _is_verified_analysis_result,
    _verified_result,
)
from .specs import (
    DiagonalRootSpec,
    OperatorElementRootSpec,
    DirectSolveSpec,
    HBSolveSpec,
    HybridizedPoleSpec,
    OperatorSpec,
    OptimizationSpec,
    OptimizationProgress,
    QuantityAbsolute,
    QuantityDifference,
    QuantitySelector,
    QuantitySum,
    ReportSpec,
    ResidueNormalizedCouplingSpec,
    ResponseElementSpec,
    TransferZeroSpec,
    _selector_unit,
)


# Omission selects the backend default; explicit None remains invalid.
_OMITTED_COMMIT_EVERY_GENERATIONS = object()







def _parameter_value_record(parameter: ParameterRef, value: object) -> Mapping[str, object]:
    return ParameterSet({parameter: value})._record()["bindings"][0]["value"]


def _uses_baseline_root(spec: object) -> bool:
    """Return whether a Spec owns baseline-root continuation."""

    if isinstance(spec, (DiagonalRootSpec, OperatorElementRootSpec, HybridizedPoleSpec, TransferZeroSpec)):
        return True
    if isinstance(spec, ResidueNormalizedCouplingSpec):
        return True
    if isinstance(spec, OptimizationSpec):
        return any(
            _uses_baseline_root(selector.spec)
            for objective in spec.objectives
            for selector in _quantity_selectors(objective.quantity)
        )
    return False










class ReductionPipeline:
    """An immutable declaration of the shared Direct view grammar."""

    __slots__ = ("_ptc", "_transforms", "_retained")

    def __init__(self) -> None:
        self._ptc: tuple[PortRef, ...] | None = None
        self._transforms: tuple[tuple[str | ElectricNodeRef | CoordinateRef, str | ElectricNodeRef | CoordinateRef, str], ...] = ()
        self._retained: tuple[str | ElectricNodeRef | CoordinateRef, ...] | None = None

    def _copy(self) -> ReductionPipeline:
        child = ReductionPipeline()
        child._ptc = self._ptc
        child._transforms = self._transforms
        child._retained = self._retained
        return child

    def ptc(self, *ports: PortRef) -> ReductionPipeline:
        """Declare the one optional, first compensation step."""

        if self._ptc is not None:
            raise ValueError("ptc() may appear at most once")
        if self._transforms or self._retained is not None:
            raise ValueError("ptc() must precede transform_pair() and retain()")
        if not ports or any(not isinstance(port, PortRef) for port in ports) or len(set(ports)) != len(ports):
            raise ValueError("ptc() requires unique PortRef values")
        child = self._copy()
        child._ptc = tuple(ports)
        return child

    def transform_pair(
        self,
        node_a: str | ElectricNodeRef | CoordinateRef,
        node_b: str | ElectricNodeRef | CoordinateRef,
        *,
        id: str,
    ) -> ReductionPipeline:
        """Declare one ordered automatic pair transform."""

        if self._retained is not None:
            raise ValueError("transform_pair() must precede retain()")
        id = _canonical_identifier(id, field="transform_pair id")
        if any(existing[2] == id for existing in self._transforms):
            raise ValueError("transform_pair IDs must be unique")
        child = self._copy()
        child._transforms = (*self._transforms, (node_a, node_b, id))
        return child

    def retain(
        self,
        *coordinates: str | ElectricNodeRef | CoordinateRef,
    ) -> ReductionPipeline:
        """Return a new pipeline with a terminal retained analysis boundary."""

        if self._retained is not None:
            raise ValueError("retain() is terminal and may appear at most once")
        if not coordinates:
            raise ValueError("retain() requires at least one coordinate")
        child = self._copy()
        child._retained = tuple(coordinates)
        return child


class NetworkViewRef:
    """Immutable lazy reference to one Plan and one reduction lineage."""

    __slots__ = (
        "__weakref__",
        "_run",
        "_lineage",
        "_retained",
        "_available_coordinates",
        "_port_coordinates",
        "_coordinate_load_states",
    )

    def __init__(self) -> None:
        unavailable("NetworkViewRef construction")

    @classmethod
    def _create(
        cls,
        run: CircuitRun,
        lineage: Mapping[str, object],
        retained: tuple[str, ...] = (),
        available_coordinates: Sequence[str] = (),
        port_coordinates: Mapping[str, str] | None = None,
        coordinate_load_states: Mapping[str, str] | None = None,
    ) -> NetworkViewRef:
        ref = object.__new__(cls)
        ref._run = run
        ref._lineage = MappingProxyType(dict(lineage))
        ref._retained = retained
        ref._available_coordinates = tuple(available_coordinates)
        ref._port_coordinates = MappingProxyType(dict(port_coordinates or {}))
        ref._coordinate_load_states = MappingProxyType(dict(coordinate_load_states or {}))
        return ref

    def reduce(self, pipeline: ReductionPipeline) -> NetworkViewRef:
        """Derive an immutable lazy child View without compiling or solving."""

        if not isinstance(pipeline, ReductionPipeline):
            raise TypeError("reduce() requires a ReductionPipeline")
        if self._retained:
            raise ValueError("a terminal retained View cannot be reduced again")
        return self._run._derive_view(self, pipeline)


class CircuitRun:
    """Execution namespace for one permanently sealed Plan and workspace leaf."""

    __slots__ = (
        "_backend",
        "_precision",
        "_timing",
        "_plan",
        "_snapshot",
        "_baseline_point",
        "_plan_document",
        "_plan_bytes",
        "_plan_sha256",
        "_binding",
        "_original",
        "_original_declaration",
        "_lifecycle",
        "_runtime_base",
        "_parameter_lookup",
        "_public_coordinates",
        "_coordinate_lookup",
        "_affine_plan",
        "_source_provenance",
        "_warm_executables",
    )

    def __init__(
        self,
        *,
        plan: CircuitPlan,
        workspace: str | PathLike[str],
        versioned: bool = False,
        backend: str = "jax",
        precision: str = "float64",
        timing: str = "aggregate",
    ) -> None:
        if not isinstance(plan, CircuitPlan):
            raise TypeError("plan must be a CircuitPlan")
        if not isinstance(versioned, bool):
            raise TypeError("versioned must be bool")
        self._backend, self._precision = self._backend_options(backend, precision)
        self._timing = self._timing_options(timing)
        self._lifecycle = RunLifecycle()
        # Run-owned executable entries exclude Plan/coefficient/result bodies.
        self._warm_executables = {}
        with plan._run_seal_preparation() as seal_token:
            self._prepare_run(
                plan=plan,
                workspace=workspace,
                versioned=versioned,
                seal_token=seal_token,
            )

    def _prepare_run(
        self,
        *,
        plan: CircuitPlan,
        workspace: str | PathLike[str],
        versioned: bool,
        seal_token: object | None,
    ) -> None:
        captured = capture_run_binding(plan)
        snapshot = captured.snapshot
        self._plan = plan
        self._snapshot = snapshot
        self._baseline_point = captured.baseline_point
        self._plan_document = captured.plan_document
        self._plan_bytes = captured.plan_bytes
        self._plan_sha256 = captured.plan_sha256
        self._runtime_base = _runtime_identity_base()
        self._parameter_lookup = captured.parameter_lookup
        self._source_provenance = snapshot.source_provenance
        self._public_coordinates = captured.public_coordinates
        self._coordinate_lookup = captured.coordinate_lookup
        self._affine_plan = captured.affine_plan
        original = self._original_lineage(coordinate_order=captured.coordinate_order)
        nodes_by_net = {
            str(node["final_net"]): str(node["compiler_node_id"])
            for node in self._plan_document["connectivity"]["node_coordinates"]
        }
        port_coordinates = {
            nodes_by_net[str(port["net"])]: str(port["id"])
            for port in self._plan_document["connectivity"]["ports"]
        }
        self._original_declaration = ViewDeclaration.create(
            lineage=original, available_coordinates=tuple(sorted(self._public_coordinates)),
            port_coordinates=port_coordinates,
            coordinate_load_states={coordinate: "raw" for coordinate in port_coordinates},
        )
        self._original = None
        self._binding = bind_workspace(
            workspace,
            plan_sha256=self._plan_sha256,
            plan_bytes=self._plan_bytes,
            versioned=versioned,
            commit=lambda: plan._seal_validated(snapshot, seal_token),
        )
        from .workspace.evidence import recover_operation_workspace

        with self._binding.writer(cleanup_staging=False):
            recover_operation_workspace(self._binding)

    @property
    @run_activity
    def original(self) -> NetworkViewRef:
        """The sealed Plan's immutable zero-reduction root View."""

        existing = None if self._original is None else self._original()
        if existing is None:
            declaration = self._original_declaration
            existing = NetworkViewRef._create(
                self, declaration.lineage(),
                available_coordinates=declaration.available_coordinates,
                port_coordinates=dict(declaration.port_coordinates),
                coordinate_load_states=dict(declaration.coordinate_load_states),
            )
            self._original = weakref(existing)
        return existing

    def close(self) -> None:
        """Release this Run's warm holders; active calls reject close, not cancel."""
        def release():
            self._warm_executables.clear()
            self._original = None
        self._lifecycle.close(release)

    def _original_lineage(self, *, coordinate_order: tuple[str, ...]) -> dict[str, object]:
        return _original_lineage_document(
            self._plan_document, self._plan_sha256, self._runtime_base,
            coordinate_order=coordinate_order,
        )

    @run_activity
    def _derive_view(self, parent: NetworkViewRef, pipeline: ReductionPipeline) -> NetworkViewRef:
        declaration = derive_view(plan=self._plan, coordinate_lookup=self._coordinate_lookup,
                                  parent=parent, pipeline=pipeline)
        return NetworkViewRef._create(
            self, declaration.lineage(), declaration.retained,
            available_coordinates=declaration.available_coordinates,
            port_coordinates=dict(declaration.port_coordinates),
            coordinate_load_states=dict(declaration.coordinate_load_states),
        )

    @staticmethod
    def _backend_options(backend: str, precision: str) -> tuple[str, str]:
        if backend not in ("jax", "julia"):
            raise ValueError("backend must be jax or julia")
        if precision not in ("float64", "float32"):
            raise ValueError("precision must be float64 or float32")
        return backend, precision

    def _selected_backend(self, backend, precision) -> tuple[str, str]:
        return self._backend_options(
            self._backend if backend is None else backend,
            self._precision if precision is None else precision,
        )

    @staticmethod
    def _timing_options(timing: str) -> str:
        if timing not in ("aggregate", "detailed"):
            raise ValueError("timing must be aggregate or detailed")
        return timing

    def _selected_timing(self, timing: str | None) -> str:
        return self._timing_options(self._timing if timing is None else timing)

    @staticmethod
    def _require_backend_spec(backend, precision, spec) -> None:
        if backend == "julia":
            if precision != "float64":
                raise RuntimePreparationError(
                    "The Julia backend requires precision='float64'.", stage="runtime_prepare")
            return
        supported = (
            DirectSolveSpec, DiagonalRootSpec, OperatorElementRootSpec,
            HybridizedPoleSpec, TransferZeroSpec, ResidueNormalizedCouplingSpec,
            ResponseElementSpec, OperatorSpec,
        )
        quantities = (selector.spec for objective in spec.objectives
                      for selector in _quantity_selectors(objective.quantity)) if isinstance(spec, OptimizationSpec) else (spec,)
        for quantity in quantities:
            if not isinstance(quantity, supported):
                raise RuntimePreparationError(
                    f"{type(quantity).__name__} requires backend='julia', scnsim[julia] "
                    "and manually provided Julia 1.12.6; JAX has no fallback.", stage="runtime_prepare")

    @contextmanager
    def _execution_scope(self, method, backend, precision, start_tick_ns, timing=None):
        from .diagnostics.operations import OperationRecorder
        backend, precision = self._selected_backend(backend, precision)
        recorder = OperationRecorder(self._binding, kind=method, backend=backend,
                                     precision=precision, start_tick_ns=start_tick_ns,
                                     timing=self._selected_timing(timing))
        with operation_lease(self._binding, recorder.operation_id) as lease:
            with ExitStack() as resources:
                native_supervisor = None
                if backend == "julia":
                    from .execution.native_supervisor import NativeSupervisor
                    native_supervisor = resources.enter_context(NativeSupervisor(
                        operation_id=recorder.operation_id, lease_fds=lease.descriptors,
                    ))
                with recorder as trace:
                    yield backend, precision, trace, lease, native_supervisor

    @overload
    def solve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec,
        *,
        parameters: ParameterSet | None = None,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> DirectSolveResult: ...

    @overload
    def solve(
        self,
        ref: NetworkViewRef,
        spec: HBSolveSpec,
        *,
        parameters: ParameterSet | None = None,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> HBBatchResult: ...

    @overload
    def solve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec,
        *,
        parameters: ParameterSpace,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> ParameterSweepResult: ...

    @run_activity
    def solve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> DirectSolveResult | HBBatchResult | ParameterSweepResult:
        """Execute the selected Direct response or one shared-basis HB batch."""

        started_ns = perf_counter_ns()
        with self._execution_scope("solve", backend, precision, started_ns, timing) as (backend, precision, trace, lease, native_supervisor):
            self._require_ref(ref)
            self._require_backend_spec(backend, precision, spec)
            operation = "solve_hb" if isinstance(spec, HBSolveSpec) else "solve_direct"
            with trace.span("preparation"):
                prepared = self._prepare_analysis(operation, ref, spec, parameters,
                                                  backend=backend, precision=precision)
            return self._execute(prepared, bound_spec=spec, trace=trace, operation_lease=lease,
                                 native_supervisor=native_supervisor)

    @overload
    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: DiagonalRootSpec,
        *,
        parameters: ParameterSet | None = None,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> DiagonalRootResult: ...

    @overload
    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: OperatorElementRootSpec,
        *,
        parameters: ParameterSet | None = None,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> OperatorElementRootResult: ...

    @overload
    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: OperatorSpec,
        *,
        parameters: ParameterSet | None = None,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> OperatorResult: ...

    @overload
    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec,
        *,
        parameters: ParameterSet | None = None,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> DirectQuantityResult: ...

    @overload
    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: DiagonalRootSpec | OperatorElementRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec,
        *,
        parameters: ParameterSpace,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> ParameterSweepResult: ...

    @run_activity
    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: DiagonalRootSpec
        | OperatorElementRootSpec
        | HybridizedPoleSpec
        | TransferZeroSpec
        | ResidueNormalizedCouplingSpec
        | ResponseElementSpec
        | OperatorSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> (
        DiagonalRootResult
        | OperatorElementRootResult
        | DirectQuantityResult
        | OperatorResult
        | ParameterSweepResult
    ):
        """Evaluate one typed Direct quantity without an unrelated sweep."""

        started_ns = perf_counter_ns()
        with self._execution_scope("evaluate", backend, precision, started_ns, timing) as (backend, precision, trace, lease, native_supervisor):
            self._require_ref(ref)
            self._require_backend_spec(backend, precision, spec)
            with trace.span("preparation"):
                prepared = self._prepare_analysis("evaluate_direct", ref, spec, parameters,
                                                  backend=backend, precision=precision)
            return self._execute(prepared, bound_spec=spec, trace=trace, operation_lease=lease,
                                 native_supervisor=native_supervisor)

    @run_activity
    def warmup(
        self,
        view: NetworkViewRef,
        spec: object,
        *,
        parameters: ParameterSet | None = None,
    ):
        """Prepare exact JAX signatures without solving or storing success."""
        from .execution.warmup import prepare_warmup

        if self._backend != "jax":
            raise RuntimePreparationError('CircuitRun.warmup requires backend="jax"', stage="warmup")
        self._require_ref(view)
        self._require_backend_spec("jax", self._precision, spec)
        operation = ("optimize_direct" if isinstance(spec, OptimizationSpec) else
                     "solve_direct" if isinstance(spec, DirectSolveSpec) else "evaluate_direct")
        prepared = self._prepare_analysis(operation, view, spec, parameters,
                                          backend="jax", precision=self._precision)
        with operation_lease(self._binding, str(uuid4())):
            return prepare_warmup(
                plan_document=self._plan_document, analysis=prepared,
                precision=self._precision, executable_owner=self._warm_executables,
                baseline_parameters=self._baseline_point.parameter_record,
            )

    @run_activity
    def benchmark(
        self, *, operations=None, method=None, backend=None, precision=None,
        status=None,
    ):
        """Read recorded operation traces from this Plan leaf without execution."""
        from .benchmark.api import benchmark
        return benchmark(self._binding, operations=operations, method=method,
                         backend=backend, precision=precision, status=status)

    @run_activity
    def recover_workspace(self) -> None:
        """Explicitly recover only this Run's bound Plan-leaf operation store."""
        from .workspace.evidence import recover_operation_workspace

        with self._binding.writer(cleanup_staging=False):
            recover_operation_workspace(self._binding)

    @overload
    def optimize(
        self,
        spec: OptimizationSpec,
        *,
        parameters: ParameterSet | None = None,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
        progress: bool = True,
        on_progress: Callable[[OptimizationProgress], object] | None = None,
        checkpoint: str = "generation",
        resume_from: Mapping[str, object] | None = None,
        commit_every_generations: int | object = _OMITTED_COMMIT_EVERY_GENERATIONS,
    ) -> OptimizationResult: ...

    @overload
    def optimize(
        self,
        ref: NetworkViewRef,
        spec: OptimizationSpec,
        *,
        parameters: ParameterSet | None = None,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
        progress: bool = True,
        on_progress: Callable[[OptimizationProgress], object] | None = None,
        checkpoint: str = "generation",
        resume_from: Mapping[str, object] | None = None,
        commit_every_generations: int | object = _OMITTED_COMMIT_EVERY_GENERATIONS,
    ) -> OptimizationResult: ...

    @run_activity
    def optimize(
        self,
        ref_or_spec: NetworkViewRef | OptimizationSpec,
        spec: OptimizationSpec | None = None,
        *,
        parameters: ParameterSet | None = None,
        backend: str | None = None,
        precision: str | None = None,
        timing: str | None = None,
        progress: bool = True,
        on_progress: Callable[[OptimizationProgress], object] | None = None,
        checkpoint: str = "generation",
        resume_from: Mapping[str, object] | None = None,
        commit_every_generations: int | object = _OMITTED_COMMIT_EVERY_GENERATIONS,
    ) -> OptimizationResult:
        """Run one pinned Direct CMA-ES request and return its exact winner."""

        started_ns = perf_counter_ns()
        with self._execution_scope("optimize", backend, precision, started_ns, timing) as (backend, precision, trace, lease, native_supervisor):
            if commit_every_generations is _OMITTED_COMMIT_EVERY_GENERATIONS:
                commit_every_generations = 10 if backend == "jax" else 1
            if isinstance(commit_every_generations, bool) or not isinstance(commit_every_generations, int):
                raise TypeError("commit_every_generations must be a positive integer")
            if commit_every_generations <= 0:
                raise ValueError("commit_every_generations must be a positive integer")
            if backend == "julia" and commit_every_generations != 1:
                raise RuntimePreparationError(
                    "commit_every_generations overrides require the JAX backend.", stage="runtime_prepare"
                )
            if parameters is not None and not isinstance(parameters, ParameterSet):
                raise TypeError(
                    "OptimizationSpec parameters must be a ParameterSet or None"
                )
            if not isinstance(progress, bool):
                raise TypeError("progress must be bool")
            if on_progress is not None and not callable(on_progress):
                raise TypeError("on_progress must be callable or None")
            default_ref, optimization_spec = self._optimization_arguments(ref_or_spec, spec)
            ref, selector_views = self._optimization_views(
                optimization_spec, default_ref=default_ref
            )
            self._require_backend_spec(backend, precision, optimization_spec)
            if checkpoint not in ("generation", "off"):
                raise ValueError("checkpoint must be generation or off")
            if backend == "julia" and (checkpoint != "generation" or resume_from is not None):
                raise RuntimePreparationError("Explicit checkpoint overrides require the JAX backend.", stage="runtime_prepare")
            display = None
            if progress:
                from .execution.progress import OptimizationProgressDisplay

                display = OptimizationProgressDisplay(enabled=True)
                display.start()
            try:
                try:
                    with trace.span("preparation"):
                        prepared = self._prepare_analysis(
                            "optimize_direct",
                            ref,
                            optimization_spec,
                            parameters,
                            selector_views=selector_views,
                            backend=backend, precision=precision,
                        )
                except (Exception, KeyboardInterrupt) as error:
                    if display is not None:
                        try:
                            display.update(
                                phase="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                completed_generations=0, total_generations=0,
                                evaluated_count=0, best_cost=None,
                                committed_generations=0,
                            )
                        except Exception as display_error:
                            error.add_note(
                                "Progress display failure during preparation error: "
                                f"{type(display_error).__name__}"
                            )
                    raise
                return self._execute(
                    prepared, bound_spec=optimization_spec, on_progress=on_progress,
                    progress_observer=display, checkpoint_policy=checkpoint,
                    resume_from=resume_from,
                    commit_every_generations=commit_every_generations, trace=trace,
                    operation_lease=lease,
                    native_supervisor=native_supervisor,
                )
            finally:
                if display is not None:
                    display.close()

    @overload
    def resolve(self, ref: NetworkViewRef, spec: DirectSolveSpec, *, parameters: ParameterSet | ParameterSpace | None = None, backend: str | None = None, precision: str | None = None) -> DirectSolveResult | ParameterSweepResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: DiagonalRootSpec, *, parameters: ParameterSet | ParameterSpace | None = None, backend: str | None = None, precision: str | None = None) -> DiagonalRootResult | ParameterSweepResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: OperatorElementRootSpec, *, parameters: ParameterSet | ParameterSpace | None = None, backend: str | None = None, precision: str | None = None) -> OperatorElementRootResult | ParameterSweepResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: OptimizationSpec, *, parameters: ParameterSet | None = None, backend: str | None = None, precision: str | None = None) -> OptimizationResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: HBSolveSpec, *, parameters: ParameterSet | ParameterSpace | None = None, backend: str | None = None, precision: str | None = None) -> HBBatchResult | ParameterSweepResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: OperatorSpec, *, parameters: ParameterSet | ParameterSpace | None = None, backend: str | None = None, precision: str | None = None) -> OperatorResult | ParameterSweepResult: ...

    @overload
    def resolve(
        self,
        ref: NetworkViewRef,
        spec: HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
        backend: str | None = None,
        precision: str | None = None,
    ) -> DirectQuantityResult | ParameterSweepResult: ...

    @run_activity
    def resolve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec
        | HBSolveSpec
        | DiagonalRootSpec
        | OperatorElementRootSpec
        | HybridizedPoleSpec
        | TransferZeroSpec
        | ResidueNormalizedCouplingSpec
        | ResponseElementSpec
        | OperatorSpec
        | OptimizationSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
        backend: str | None = None,
        precision: str | None = None,
    ) -> (
        DirectSolveResult
        | HBBatchResult
        | DiagonalRootResult
        | OperatorElementRootResult
        | DirectQuantityResult
        | OperatorResult
        | OptimizationResult
        | ParameterSweepResult
    ):
        """Verify and load the success for this exact request without retrying."""

        backend, precision = self._selected_backend(backend, precision)
        self._require_backend_spec(backend, precision, spec)
        self._require_ref(ref)
        if isinstance(spec, DirectSolveSpec):
            operation = "solve_direct"
        elif isinstance(spec, HBSolveSpec):
            operation = "solve_hb"
        elif isinstance(
            spec,
            (
                DiagonalRootSpec,
                OperatorElementRootSpec,
                HybridizedPoleSpec,
                TransferZeroSpec,
                ResidueNormalizedCouplingSpec,
                ResponseElementSpec,
                OperatorSpec,
            ),
        ):
            operation = "evaluate_direct"
        elif isinstance(spec, OptimizationSpec):
            if parameters is not None and not isinstance(parameters, ParameterSet):
                raise TypeError(
                    "OptimizationSpec parameters must be a ParameterSet or None"
                )
            operation = "optimize_direct"
        else:
            unavailable(f"CircuitRun.resolve({type(spec).__name__})")
        selector_views = None
        if isinstance(spec, OptimizationSpec):
            ref, selector_views = self._optimization_views(spec, default_ref=ref)
        prepared = self._prepare_analysis(
            operation,
            ref,
            spec,
            parameters,
            selector_views=selector_views,
            backend=backend, precision=precision,
        )
        if backend == "jax":
            from .execution.jax_operation import resolve_jax_operation
            return resolve_jax_operation(binding=self._binding, prepared_analysis=prepared,
                                         decoder=self._result_decoder(), bound_spec=spec)
        with self._binding.reader():
            success = self._binding.resolve_success(prepared.request_sha256)
            evidence_lease = _verified_evidence_lease(self._binding, success)
            return self._decode_success(
                success,
                bound_spec=spec,
                evidence_lease=evidence_lease,
            )

    @run_activity
    def explain(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec
        | HBSolveSpec
        | DiagonalRootSpec
        | OperatorElementRootSpec
        | HybridizedPoleSpec
        | TransferZeroSpec
        | ResidueNormalizedCouplingSpec
        | ResponseElementSpec
        | OperatorSpec
        | OptimizationSpec,
        *,
        parameters: ParameterSet | None = None,
        backend: str | None = None,
        precision: str | None = None,
    ) -> ExplanationResult:
        """Explain the selected backend compiler without creating an attempt or solve."""

        backend, precision = self._selected_backend(backend, precision)
        self._require_ref(ref)
        if not isinstance(
            spec,
            (
                DirectSolveSpec,
                HBSolveSpec,
                DiagonalRootSpec,
                OperatorElementRootSpec,
                HybridizedPoleSpec,
                TransferZeroSpec,
                ResidueNormalizedCouplingSpec,
                ResponseElementSpec,
                OperatorSpec,
                OptimizationSpec,
            ),
        ):
            unavailable(f"CircuitRun.explain({type(spec).__name__})")
        operation = (
            "solve_hb"
            if isinstance(spec, HBSolveSpec)
            else "solve_direct"
            if isinstance(spec, DirectSolveSpec)
            else "evaluate_direct"
            if isinstance(
                spec,
                (
                    DiagonalRootSpec,
                    OperatorElementRootSpec,
                    HybridizedPoleSpec,
                    TransferZeroSpec,
                    ResidueNormalizedCouplingSpec,
                    ResponseElementSpec,
                    OperatorSpec,
                ),
            )
            else "optimize_direct"
        )
        self._require_backend_spec(backend, precision, spec)
        selector_views = None
        if isinstance(spec, OptimizationSpec):
            ref, selector_views = self._optimization_views(spec, default_ref=ref)
        prepared = self._prepare_analysis(
            operation,
            ref,
            spec,
            parameters,
            selector_views=selector_views,
            backend=backend, precision=precision,
        )
        request = prepared.request()
        operation_id = str(uuid4())
        with operation_lease(self._binding, operation_id) as lease:
            if backend == "jax":
                from .execution.compilation import _run_jax_preflight
                compiled = _run_jax_preflight(self._plan_document, request)
            else:
                from .execution.native_supervisor import NativeSupervisor
                with NativeSupervisor(operation_id=operation_id,
                                      lease_fds=lease.descriptors) as native_supervisor:
                    compiled = self._preflight(request, native_supervisor=native_supervisor)
        return _verified_result(
            ExplanationResult,
            evidence={
                "plan_sha256": self._plan_sha256,
                "request_sha256": sha256_hex(canonical_json_bytes(request)),
                "runtime_semantic": request["runtime_semantic"],
                "view": request["view"],
                "parameter_source": request["parameter_source"],
                "spec": request["spec"],
                "scope_hierarchy": self._plan_document["scope_hierarchy"],
                "occurrences": self._plan_document["occurrences"],
                "physical_leaves": self._plan_document["physical_leaves"],
                "connectivity": self._plan_document["connectivity"],
                "compiled": compiled,
            },
        )

    @run_activity
    def inventory(self) -> InventoryResult:
        """Inspect this Run's exact workspace leaf without selecting a latest result."""

        with self._binding.reader():
            inventory = self._binding.inventory_document()
        requests = inventory.get("requests")
        maintenance = inventory.get("maintenance")
        if (
            inventory.get("schema") != "scnsim.inventory"
            or inventory.get("schema_version") != 3
            or inventory.get("plan_sha256") != self._plan_sha256
            or not isinstance(requests, list)
            or any(not isinstance(row, Mapping) for row in requests)
            or not isinstance(maintenance, list)
            or len(maintenance) > 1
            or any(not isinstance(row, Mapping) for row in maintenance)
        ):
            raise EvidenceIntegrityError("workspace inventory is malformed", stage="inventory")
        return _verified_result(
            InventoryResult,
            requests=tuple(dict(row) for row in requests),
            maintenance=tuple(dict(row) for row in maintenance),
        )

    @run_activity
    def build_report(self, spec: ReportSpec) -> ReportResult:
        """Derive a self-contained report from explicit receipt-backed Results."""

        if (
            not isinstance(spec, ReportSpec)
            or not spec.inputs
            or not all(_is_verified_analysis_result(result) for result in spec.inputs)
        ):
            raise TypeError("build_report() requires ReportSpec")
        from .visualization.report import build_report

        return build_report(spec)

    def _require_ref(self, ref: NetworkViewRef) -> None:
        if not isinstance(ref, NetworkViewRef) or ref._run is not self:
            raise ValueError("NetworkViewRef belongs to another CircuitRun")

    def _optimization_arguments(
        self,
        ref_or_spec: NetworkViewRef | OptimizationSpec,
        spec: OptimizationSpec | None,
    ) -> tuple[NetworkViewRef | None, OptimizationSpec]:
        if isinstance(ref_or_spec, OptimizationSpec):
            if spec is not None:
                raise TypeError("optimize() received two OptimizationSpec values")
            return None, ref_or_spec
        if not isinstance(ref_or_spec, NetworkViewRef) or not isinstance(spec, OptimizationSpec):
            raise TypeError("optimize() requires OptimizationSpec or NetworkViewRef, OptimizationSpec")
        self._require_ref(ref_or_spec)
        return ref_or_spec, spec

    def _optimization_views(
        self,
        spec: OptimizationSpec,
        *,
        default_ref: NetworkViewRef | None,
    ) -> tuple[NetworkViewRef, Mapping[int, NetworkViewRef]]:
        bindings: dict[int, NetworkViewRef] = {}
        ordered: list[NetworkViewRef] = []
        for objective in spec.objectives:
            for selector in _quantity_selectors(objective.quantity):
                selected = selector._view if selector._view is not None else default_ref
                if not isinstance(selected, NetworkViewRef) or selected._run is not self:
                    raise InvalidOptimizationSpec(
                        "every optimization selector must bind a View from this CircuitRun",
                        stage="spec_validation",
                    )
                bindings[id(selector)] = selected
                ordered.append(selected)
        if not ordered:
            raise InvalidOptimizationSpec(
                "optimization requires at least one bound selector",
                stage="spec_validation",
            )
        return ordered[0], MappingProxyType(bindings)

    def _coordinate_id(self, value):
        return coordinate_id(self._plan, self._coordinate_lookup, value)

    def _view_coordinate_id(
        self,
        ref: NetworkViewRef,
        value: str | ElectricNodeRef | CoordinateRef,
    ) -> str:
        if isinstance(value, str) and (
            value in ref._available_coordinates
            or value in ref._lineage["terminal_coordinates"]
        ):
            return value
        return self._coordinate_id(value)

    def _validate_direct_request(
        self,
        operation: str,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | DiagonalRootSpec | OperatorElementRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        *,
        selector_views: Mapping[int, NetworkViewRef] | None = None,
    ) -> None:
        if isinstance(spec, DirectSolveSpec):
            if ref._lineage["port_realizable"] is not True:
                raise PortRealizabilityError(
                    "Direct response requires a port-realizable View",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": operation, "context_kind": "direct_response"},
                )
            channels = frozenset(ref._lineage["terminal_coordinates"])
            for trace in spec.traces:
                try:
                    input_channel = self._trace_request_channel(ref, trace.input_port)
                    output_channel = self._trace_request_channel(ref, trace.output_port)
                except ValueError:
                    input_channel = output_channel = None
                if input_channel not in channels or output_channel not in channels:
                    raise PortRealizabilityError(
                        "Direct trace names a channel outside the selected View",
                        stage="preflight",
                        evidence={"type": "failure_evidence", "operation": operation, "context_kind": "direct_response"},
                    )
                if trace.input_mode or trace.output_mode:
                    raise ValueError("Direct traces require empty mode tuples")
            return
        if isinstance(spec, (DiagonalRootSpec, OperatorElementRootSpec, HybridizedPoleSpec, TransferZeroSpec, ResidueNormalizedCouplingSpec, ResponseElementSpec, OperatorSpec)):
            self._validate_direct_quantity_spec(operation, ref, spec)
            return
        active = {
            _parameter_key(variable.parameter): variable.parameter
            for variable in spec.variables
        }
        for key, parameter in active.items():
            current = self._parameter_lookup.get(key)
            if current is None or current._definition_record() != parameter._definition_record():
                raise InvalidOptimizationSpec(
                    "optimization variable belongs to another Plan",
                    stage="spec_validation",
                )
        for parameter in spec.allow_extrapolation:
            key = _parameter_key(parameter)
            selected = active.get(key)
            if selected is None or selected._definition_record() != parameter._definition_record():
                raise InvalidOptimizationSpec(
                    "optimization extrapolation authorization must name an active Plan parameter",
                    stage="spec_validation",
                )
        for objective in spec.objectives:
            for selector in _quantity_selectors(objective.quantity):
                selected = None if selector_views is None else selector_views.get(id(selector))
                if selected is None:
                    raise InvalidOptimizationSpec(
                        "optimization selector View normalization is incomplete",
                        stage="spec_validation",
                    )
                self._validate_direct_quantity_spec(operation, selected, selector.spec)

    def _validate_direct_quantity_spec(
        self,
        operation: str,
        ref: NetworkViewRef,
        spec: DiagonalRootSpec | OperatorElementRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec,
        *,
        residue_branch: bool = False,
    ) -> None:
        """Apply the selected-View contract shared by evaluate and CMA selectors."""

        if isinstance(spec, DiagonalRootSpec):
            coordinate = self._view_coordinate_id(ref, spec.coordinate)
            channels = tuple(ref._lineage["terminal_coordinates"])
            invalid = coordinate not in channels or (residue_branch and len(channels) < 2)
            if invalid:
                raise SCNSimValidationError(
                    "DiagonalRootSpec coordinate is absent from the final View basis",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": operation, "context_kind": "direct_quantity"},
                )
            return
        if isinstance(spec, OperatorElementRootSpec):
            channels = frozenset(ref._lineage["terminal_coordinates"])
            if self._view_coordinate_id(ref, spec.row) not in channels or self._view_coordinate_id(ref, spec.column) not in channels:
                raise SCNSimValidationError(
                    "OperatorElementRootSpec row and column must belong to the final View basis",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": operation, "context_kind": "direct_quantity"},
                )
            return
        if isinstance(spec, HybridizedPoleSpec):
            coordinates = tuple(self._view_coordinate_id(ref, value) for value in spec.coordinates)
            if not ref._retained or coordinates != ref._retained:
                raise SCNSimValidationError(
                    "HybridizedPoleSpec coordinates must equal the retained View order",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": operation, "context_kind": "direct_quantity"},
                )
            return
        if isinstance(spec, (TransferZeroSpec, ResponseElementSpec)):
            channels = set(ref._lineage["terminal_coordinates"])
            if self._view_coordinate_id(ref, spec.input_coordinate) not in channels or self._view_coordinate_id(ref, spec.output_coordinate) not in channels:
                raise PortRealizabilityError(
                    "Direct element Spec coordinates must belong to the selected View",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": operation, "context_kind": "direct_quantity"},
                )
            if spec.family == "S" and ref._lineage["port_realizable"] is not True:
                raise PortRealizabilityError(
                    "S-family Direct elements require a port-realizable View",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": operation, "context_kind": "direct_quantity"},
                )
            return
        if isinstance(spec, ResidueNormalizedCouplingSpec):
            self._validate_direct_quantity_spec(operation, ref, spec.branch_a, residue_branch=True)
            self._validate_direct_quantity_spec(operation, ref, spec.branch_b, residue_branch=True)
            return
        if isinstance(spec, OperatorSpec):
            return
        raise InvalidOptimizationSpec("optimization selector is outside the Direct quantity catalog", stage="spec_validation")

    def _validate_hb_request(self, ref: NetworkViewRef, spec: HBSolveSpec) -> None:
        """Bind public HB declarations to this sealed Plan and selected View."""

        if ref._lineage.get("port_realizable") is not True:
            raise PortRealizabilityError(
                "HB solve requires a Port-realizable final View",
                stage="preflight",
                evidence={"type": "failure_evidence", "operation": "solve_hb", "context_kind": "runtime"},
            )
        plan_ports = tuple(self._plan.ports)
        for drive in spec.drives:
            if not any(port is drive.at for port in plan_ports):
                raise SCNSimValidationError(
                    "HB CurrentDrive belongs to another Plan",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": "solve_hb", "context_kind": "runtime"},
                )
        channels = frozenset(ref._lineage["terminal_coordinates"])
        for trace in spec.traces:
            try:
                input_channel = self._trace_request_channel(ref, trace.input_port)
                output_channel = self._trace_request_channel(ref, trace.output_port)
            except ValueError:
                input_channel = output_channel = None
            if input_channel not in channels or output_channel not in channels:
                raise PortRealizabilityError(
                    "HB trace names a channel outside the selected View",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": "solve_hb", "context_kind": "runtime"},
                )
        # Driven-PTC authorization is decided by Julia preflight after exact
        # oriented source-vector accumulation.  Comparing declaration scalars
        # here would reject valid cancellation across distinct logical Ports.

    def _trace_request_channel(self, ref: NetworkViewRef, value: str) -> str:
        """Normalize one trace name into the exact final View namespace."""

        if not ref._retained:
            return value
        if value in ref._available_coordinates:
            return value
        matches = {
            str(node["compiler_node_id"])
            for node in self._plan_document["connectivity"]["node_coordinates"]
            for alias in node["public_aliases"]
            if alias.get("kind") != "port" and alias.get("id") == value
        }
        if len(matches) != 1:
            raise ValueError("trace channel is not a unique public coordinate")
        return matches.pop()

    def _complete_parameters(self, supplied: ParameterSet | None):
        if supplied is not None and not isinstance(supplied, ParameterSet):
            raise TypeError("parameters must be ParameterSet or None")
        return resolve_parameter_point(self._snapshot, supplied)

    def _compatible_parameter(self, parameter: ParameterRef) -> ParameterRef:
        return compatible_parameter(self._parameter_lookup, parameter)

    def _parameter_source(
        self,
        parameters: ParameterSet | ParameterSpace | None,
    ) -> tuple[Mapping[str, object], object]:
        if parameters is None or isinstance(parameters, ParameterSet):
            point = self._complete_parameters(parameters)
            return {"kind": "point", "parameters": point.parameter_record}, point
        if not isinstance(parameters, ParameterSpace):
            raise TypeError("parameters must be ParameterSet, ParameterSpace, or None")
        baseline = self._complete_parameters(None)
        if parameters.kind == "grid":
            base = self._complete_parameters(parameters.fixed)
            axes: list[dict[str, object]] = []
            for parameter, values in parameters.axes:
                current = self._compatible_parameter(parameter)
                for value in values:
                    merged = dict(parameters.fixed.values)
                    merged[current] = value
                    self._complete_parameters(
                        ParameterSet(merged, allow_extrapolation=parameters.fixed.allow_extrapolation)
                    )
                axes.append({
                    "parameter": current._key_record(),
                    "values": [_parameter_value_record(current, value) for value in values],
                })
            return {
                "kind": "grid",
                "base_parameters": base.parameter_record,
                "axes": axes,
                "shape": [len(values) for _, values in parameters.axes],
            }, base
        if parameters.kind != "points":
            raise CompilerInvariantError("ParameterSpace kind is invalid", stage="request_encode")
        points: list[Mapping[str, object]] = []
        for supplied in parameters._points:
            normalized_values = {
                self._compatible_parameter(parameter): value
                for parameter, value in supplied.values.items()
            }
            normalized_authorizations = tuple(
                self._compatible_parameter(parameter)
                for parameter in supplied.allow_extrapolation
            )
            normalized = ParameterSet(
                normalized_values,
                allow_extrapolation=normalized_authorizations,
            )
            self._complete_parameters(normalized)
            points.append(normalized._record())
        return {
            "kind": "points",
            "baseline_parameters": baseline.parameter_record,
            "points": points,
        }, baseline

    def _validate_root_parameter_source(
        self,
        spec: object,
        parameter_source: Mapping[str, object],
    ) -> None:
        if not _uses_baseline_root(spec):
            return
        rlgc_keys = {
            _parameter_key(parameter)
            for parameter in self._parameter_lookup.values()
            if isinstance(parameter.spec, RLGCParameterSpec)
        }
        if not rlgc_keys:
            return

        def values(record: Mapping[str, object]) -> dict[tuple[str, str], object]:
            return {
                (binding["parameter"]["definitions_id"], binding["parameter"]["parameter_id"]): binding["value"]
                for binding in record["bindings"]
            }

        baseline = values(self._baseline_point.parameter_record)
        kind = parameter_source["kind"]
        records: list[Mapping[str, object]] = []
        if kind == "point":
            records.append(parameter_source["parameters"])
        elif kind == "grid":
            records.append(parameter_source["base_parameters"])
            for axis in parameter_source["axes"]:
                key = (axis["parameter"]["definitions_id"], axis["parameter"]["parameter_id"])
                if key in rlgc_keys and any(value != baseline[key] for value in axis["values"]):
                    raise SCNSimValidationError(
                        "baseline-root calculations do not support changing RLGC values",
                        stage="preflight",
                    )
        elif kind == "points":
            records.append(parameter_source["baseline_parameters"])
            records.extend(parameter_source["points"])
        for record in records:
            selected = values(record)
            if any(key in selected and selected[key] != baseline[key] for key in rlgc_keys):
                raise SCNSimValidationError(
                    "baseline-root calculations do not support changing RLGC values",
                    stage="preflight",
                )


    def _bind_optimization(
        self,
        spec: OptimizationSpec,
        parameters: ParameterSet,
        *,
        selector_views: Mapping[int, NetworkViewRef],
    ) -> BoundOptimization:
        """Bind ordered selector leaves without retaining live View objects."""

        expressions: list[Mapping[str, object]] = []
        leaves: list[BoundOptimizationLeaf] = []
        for objective_ordinal, objective in enumerate(spec.objectives):
            term_ordinal = 0

            def bind_expression(value: object) -> Mapping[str, object]:
                nonlocal term_ordinal
                if isinstance(value, QuantitySum):
                    return {
                        "type": "quantity_sum",
                        "terms": [bind_expression(term) for term in value.terms],
                    }
                if isinstance(value, QuantityDifference):
                    return {
                        "type": "quantity_difference",
                        "left": bind_expression(value.left),
                        "right": bind_expression(value.right),
                    }
                if isinstance(value, QuantityAbsolute):
                    return {
                        "type": "quantity_absolute",
                        "operand": bind_expression(value.operand),
                    }
                selector = value
                if not isinstance(selector, QuantitySelector):
                    raise InvalidOptimizationSpec(
                        "optimization objective contains a non-selector leaf",
                        stage="spec_validation",
                    )
                selected = selector_views.get(id(selector))
                if selected is None:
                    raise InvalidOptimizationSpec(
                        "optimization selector View normalization is incomplete",
                        stage="spec_validation",
                    )
                declaration = {
                    "type": selector.type,
                    "spec": _encode_direct_quantity(
                        selector.spec,
                        coordinate_bindings={
                            _coordinate_binding_key(
                                coordinate
                            ): self._view_coordinate_id(selected, coordinate)
                            for coordinate in _quantity_coordinates(selector.spec)
                        },
                    ),
                    "projection": selector.projection,
                    "view": _view_declaration(selected._lineage),
                }
                leaves.append(
                    BoundOptimizationLeaf.create(
                        objective_id=objective.id,
                        objective_ordinal=objective_ordinal,
                        term_ordinal=term_ordinal,
                        declaration=declaration,
                    )
                )
                term_ordinal += 1
                return declaration

            expressions.append(bind_expression(objective.quantity))
        encoded = _encode_spec(
            spec,
            parameters,
            coordinate_bindings={},
            trace_channels={},
            optimization_quantities=expressions,
        )
        return BoundOptimization.create(spec=encoded, leaves=leaves)

    def _prepare_analysis(
        self,
        operation: str,
        ref: NetworkViewRef,
        spec: DirectSolveSpec
        | HBSolveSpec
        | DiagonalRootSpec
        | OperatorElementRootSpec
        | HybridizedPoleSpec
        | TransferZeroSpec
        | ResidueNormalizedCouplingSpec
        | ResponseElementSpec
        | OperatorSpec
        | OptimizationSpec,
        parameters: ParameterSet | ParameterSpace | None,
        *,
        selector_views: Mapping[int, NetworkViewRef] | None = None,
        backend: str = "julia",
        precision: str = "float64",
    ) -> PreparedAnalysis:
        """Normalize one non-executable analysis declaration exactly once."""

        if isinstance(spec, OptimizationSpec) and selector_views is None:
            ref, selector_views = self._optimization_views(spec, default_ref=ref)
        if isinstance(spec, HBSolveSpec):
            if operation != "solve_hb":
                raise CompilerInvariantError(
                    "HB Spec has a non-HB operation", stage="request_encode"
                )
            self._validate_hb_request(ref, spec)
        else:
            self._validate_direct_request(
                operation,
                ref,
                spec,
                selector_views=selector_views,
            )
        parameter_source, resolved = self._parameter_source(parameters)
        if isinstance(spec, OptimizationSpec):
            active = {_parameter_key(variable.parameter) for variable in spec.variables}
            if parameters is not None and any(
                _parameter_key(parameter) in active for parameter in parameters.values
            ):
                raise InvalidOptimizationSpec(
                    "optimization fixed parameters overlap active variables",
                    stage="spec_validation",
                )
            # Request-level authorization is consumed only by compiler
            # baseline lowering. CMA candidate and winner
            # ParameterSets remain authorization-free and ledger-owned.
            authorized = ParameterSet(
                resolved.effective_parameters.values,
                allow_extrapolation=tuple(
                    {
                        *resolved.effective_parameters.allow_extrapolation,
                        *spec.allow_extrapolation,
                    }
                ),
            )
            resolved = self._complete_parameters(authorized)
            parameter_source = {
                "kind": "point",
                "parameters": resolved.parameter_record,
            }
        self._validate_root_parameter_source(spec, parameter_source)
        effective = resolved.effective_parameters
        bound_optimization: BoundOptimization | None = None
        try:
            if isinstance(spec, OptimizationSpec):
                if selector_views is None:
                    raise InvalidOptimizationSpec(
                        "optimization selector View normalization is incomplete",
                        stage="spec_validation",
                    )
                bound_optimization = self._bind_optimization(
                    spec, effective, selector_views=selector_views
                )
                encoded_spec = bound_optimization.spec()
            else:
                encoded_spec = _encode_spec(
                    spec,
                    effective,
                    coordinate_bindings={
                        _coordinate_binding_key(coordinate): self._view_coordinate_id(
                            ref, coordinate
                        )
                        for coordinate in _quantity_coordinates(spec)
                    }
                    if not isinstance(spec, (DirectSolveSpec, HBSolveSpec))
                    else {},
                    trace_channels={
                        channel: self._trace_request_channel(ref, channel)
                        for trace in getattr(spec, "traces", ())
                        for channel in (trace.input_port, trace.output_port)
                    },
                )
            source_units = self._source_units(
                spec, effective, parameter_space=parameters
            )
        except InvalidOptimizationSpec:
            raise
        except (SCNSimValidationError, TypeError, ValueError, AttributeError) as error:
            if not isinstance(spec, OptimizationSpec):
                raise
            raise InvalidOptimizationSpec(
                "optimization declaration is not valid for this Plan",
                stage="spec_validation",
            ) from error
        return prepare_request(
            plan_sha256=self._plan_sha256, operation=operation,
            view=_view_declaration(ref._lineage), encoded_spec=encoded_spec,
            parameter_source=parameter_source, runtime_base=self._runtime_base,
            source_units=source_units, backend=backend, precision=precision,
        )

    def _preflight(self, request: Mapping[str, object], *, native_supervisor) -> Mapping[str, object]:
        """Run the compiler-only realization boundary without allocating work."""

        return _run_preflight(self._plan_document, self._plan_bytes, request,
                              native_supervisor=native_supervisor)

    def _execute(
        self,
        prepared_analysis: PreparedAnalysis,
        *,
        trace,
        operation_lease,
        native_supervisor,
        bound_spec: object | None = None,
        on_progress: Callable[[OptimizationProgress], object] | None = None,
        progress_observer: object | None = None,
        checkpoint_policy: str = "generation",
        resume_from: Mapping[str, object] | None = None,
        commit_every_generations: int = 1,
    ):
        if prepared_analysis.request()["runtime_semantic"].get("backend") == "jax":
            from .execution.jax_operation import execute_jax_operation
            return execute_jax_operation(
                binding=self._binding, plan_document=self._plan_document,
                prepared_analysis=prepared_analysis, decoder=self._result_decoder(),
                bound_spec=bound_spec, on_progress=on_progress,
                progress_observer=progress_observer,
                checkpoint_policy=checkpoint_policy, resume_from=resume_from,
                commit_every_generations=commit_every_generations, trace=trace,
            )
        from .execution.coordinator import execute_prepared

        request = prepared_analysis.request()
        # Native evidence has request/attempt/receipt authorities, rather than a
        # benchmark task or environment record. Keep unavailable IDs null.
        trace.bind(operation=request["operation"], request_sha256=prepared_analysis.request_sha256,
                   task_id=None, environment_sha256=None, attempt_id=None,
                   details={"native_clock": "unavailable", "runtime_semantic": request["runtime_semantic"]})
        native_on_progress = on_progress
        last_native_progress = None
        if progress_observer is not None:
            def native_on_progress(event: OptimizationProgress) -> None:
                nonlocal last_native_progress
                last_native_progress = event
                # Native frames report staged progress, not publication ACKs.
                progress_observer.update(
                    phase="baseline" if event.phase == "initial" else event.phase,
                    completed_generations=event.completed_generations,
                    total_generations=event.total_generations,
                    evaluated_count=event.evaluated_count,
                    best_cost=event.best_cost,
                    committed_generations=None,
                    reused=event.reused,
                )
                if on_progress is not None:
                    on_progress(event)

        native_success = None
        with trace.span("julia_execution", details={"request_sha256": prepared_analysis.request_sha256}):
            try:
                with execute_prepared(
                    binding=self._binding,
                    prepared_analysis=prepared_analysis,
                    operation_id=trace.operation_id,
                    operation_lease=operation_lease,
                    native_supervisor=native_supervisor,
                    on_progress=native_on_progress,
                    _timing_observer=lambda stage, start, end, details: trace.measure(
                        stage, start_tick_ns=start, end_tick_ns=end, details=dict(details)),
                ) as success:
                    native_success = success
                    evidence_lease = _verified_evidence_lease(self._binding, success)
                    with trace.span("result_decode"):
                        result = self._decode_success(
                            success, bound_spec=bound_spec, evidence_lease=evidence_lease,
                        )
            finally:
                if native_success is not None:
                    # execute_prepared has released its reader/writer before
                    # trace.bind starts its own short Plan transaction.
                    from sys import exception
                    primary = exception()
                    try:
                        attempt = native_success.attempt
                        attempt_sha = sha256_hex(canonical_json_bytes(attempt))
                        receipt_sha = sha256_hex(canonical_json_bytes(native_success.receipt))
                        environment = {
                            key: attempt[key] for key in (
                                "julia_executable_sha256", "os", "architecture", "cpu",
                                "julia_threads", "blas_threads", "blas_vendor",
                            ) if key in attempt
                        }
                        trace.bind(
                            operation=request["operation"], request_sha256=prepared_analysis.request_sha256,
                            task_id=None, environment_sha256=None, attempt_id=attempt_sha,
                            details={"native_clock": "unavailable", "native_environment": environment,
                                     "runtime_semantic": request["runtime_semantic"],
                                     "native_attempt_directory": attempt["directory"],
                                     "native_identity": {"attempt_sha256": attempt_sha,
                                                         "attempt_id_kind": "canonical_attempt_sha256",
                                                         "receipt_sha256": receipt_sha}},
                        )
                        trace.add_numerical_ref({
                            "role": "native_verified_success",
                            "request_sha256": prepared_analysis.request_sha256,
                            "attempt_sha256": attempt_sha,
                            "result_sha256": native_success.receipt["result_sha256"],
                            "receipt_sha256": receipt_sha,
                            "attempt_directory": attempt["directory"],
                        })
                    except BaseException as trace_error:
                        if primary is None:
                            raise
                        primary.add_note(
                            f"Native operation trace binding also failed: {type(trace_error).__name__}: {trace_error}"
                        )
        if progress_observer is not None and last_native_progress is not None:
            # Only the verified terminal success establishes saved generations.
            event = last_native_progress
            progress_observer.update(
                phase="result_reuse" if event.reused else "complete",
                completed_generations=event.completed_generations,
                total_generations=event.total_generations,
                evaluated_count=event.evaluated_count,
                best_cost=event.best_cost,
                committed_generations=event.completed_generations,
                reused=event.reused,
            )
        return result

    def _source_units(self, spec, parameters, *, parameter_space):
        return source_units(self._source_provenance, self._parameter_lookup,
                            spec, parameters, parameter_space=parameter_space)

    def _result_decoder(self):
        from .results.decode import VerifiedResultDecoder
        return VerifiedResultDecoder(
            plan_sha256=self._plan_sha256, plan_owner_id=id(self._plan),
            binding_identity=MappingProxyType({
                "root": self._binding.root,
                "leaf": self._binding.leaf,
                "plan_sha256": self._binding.plan_sha256,
                "workspace_instance_id": self._binding.workspace_instance_id,
            }),
            parameter_lookup=self._parameter_lookup,
            coordinate_lookup=self._coordinate_lookup,
        )

    def _decode_success(
        self,
        success: VerifiedSuccess,
        *,
        bound_spec: object | None = None,
        evidence_lease: _VerifiedEvidenceLease,
    ):
        decoder = self._result_decoder()
        return decoder._decode_success(
            success,
            bound_spec=bound_spec,
            evidence_lease=evidence_lease,
        )


