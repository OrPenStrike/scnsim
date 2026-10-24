"""Plan-bound execution, exact request identity, and typed Result reconstruction.

One captured authoring snapshot and its resolved parameter points feed the
declarative Direct and optimization request boundary. Workspace receipts
and artifact manifests remain the authority for saved reconstruction. A memory
Run instead owns no Workspace and returns detached Direct or Optimization results with a compact
in-memory operation Benchmark; it cannot resolve persisted evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from os import PathLike
from types import MappingProxyType
from typing import overload
from time import perf_counter_ns
from weakref import ref as weakref
from uuid import uuid4

from .execution.prepared_function import PreparedFunction
from .execution.policy import ExecutionPolicy
from .execution.compile_owner import CompileOwner
from .results.memory_direct import InMemoryDirectSolveResult
from .results.memory_optimization import InMemoryOptimizationResult
from .results.memory_analysis import InMemoryAnalysisResult, InMemoryParameterSweepResult
from .results.memory_sensitivity import InMemorySensitivityResult
from .workspace.operation_lease import operation_lease

from .execution.run_lifecycle import RunLifecycle, run_activity
from .execution.run_binding import capture_run_binding, compatible_parameter, _parameter_key
from .execution.request_binding import _view_declaration, _quantity_selectors, source_units, prepare_request
from .execution.view_binding import ViewDeclaration, derive_view, coordinate_id, _original_lineage_document


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
from .authoring.resolution import resolve_parameter_point
from .authoring.physical_values import RLGCParameterSpec
from .construction import unavailable
from .workspace import bind_workspace
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
    ResultUnavailableError,
)
from .results import (
    DiagonalRootResult,
    OperatorElementRootResult,
    DirectQuantityResult,
    DirectSolveResult,
    ExplanationResult,
    OperatorResult,
    StaticCapacitanceResult,
    StaticCapacitanceElementResult,
    OptimizationResult,
    ParameterSweepResult,
    SensitivityResult,
    ReportResult,
    _is_verified_analysis_result,
    _verified_result,
)
from .specs import (
    DiagonalRootSpec,
    OperatorElementRootSpec,
    DirectSolveSpec,
    HybridizedPoleSpec,
    OperatorSpec,
    StaticCapacitanceSpec,
    StaticCapacitanceElementSpec,
    OptimizationSpec,
    CMAESSpec,
    SensitivitySpec,
    FixedMeshSpec,
    AcceptedIterateSelection,
    LBFGSBSpec,
    ProjectedLBFGSSpec,
    HybridSpec,
    OptimizationProgress,
    QuantityAbsolute,
    QuantityDifference,
    QuantitySelector,
    QuantitySum,
    ReportSpec,
    ResidueNormalizedCouplingSpec,
    ResponseElementSpec,
    TransferZeroSpec,
)


# Omission selects the existing cadence default; explicit None remains invalid.
_OMITTED_COMMIT_EVERY_GENERATIONS = object()
_OMITTED_CHECKPOINT = object()


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
    """Sealed Plan execution with saved Workspace or detached checked JAX memory Results."""

    __slots__ = (
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
        "_execution_policy",
        "_storage",
        "_last_memory_benchmark",
    )

    def __init__(
        self,
        *,
        plan: CircuitPlan,
        workspace: str | PathLike[str] | None = None,
        versioned: bool = False,
        storage: str = "workspace",
        precision: str = "float64",
        timing: str = "aggregate",
        execution_policy: ExecutionPolicy | None = None,
    ) -> None:
        if not isinstance(plan, CircuitPlan):
            raise TypeError("plan must be a CircuitPlan")
        if not isinstance(versioned, bool):
            raise TypeError("versioned must be bool")
        if storage not in {"workspace", "memory"}:
            raise ValueError("storage must be 'workspace' or 'memory'")
        if storage == "workspace" and workspace is None:
            raise TypeError("storage='workspace' requires workspace")
        if storage == "memory" and (workspace is not None or versioned):
            raise ValueError("storage='memory' does not accept workspace or versioned=True")
        self._storage = storage
        self._last_memory_benchmark = None
        self._precision = self._precision_options(precision)
        self._timing = self._timing_options(timing)
        if storage == "memory" and self._timing != "aggregate":
            raise ValueError("storage='memory' supports timing='aggregate' only")
        self._lifecycle = RunLifecycle()
        if execution_policy is not None and not isinstance(execution_policy, ExecutionPolicy):
            raise TypeError("execution_policy must be ExecutionPolicy or None")
        self._execution_policy = ExecutionPolicy() if execution_policy is None else execution_policy
        # Run-owned executable entries exclude Plan/coefficient/result bodies.
        self._warm_executables = CompileOwner(policy=self._execution_policy)
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
        workspace: str | PathLike[str] | None,
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
        if self._storage == "memory":
            plan._seal_validated(snapshot, seal_token)
            self._binding = None
            return
        self._binding = bind_workspace(
            workspace,
            plan_sha256=self._plan_sha256,
            plan_bytes=self._plan_bytes,
            versioned=versioned,
            commit=lambda: plan._seal_validated(snapshot, seal_token),
        )
        from .workspace.evidence import recover_operation_workspace

        with self._binding.writer():
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
    def _precision_options(precision: str) -> str:
        if precision not in ("float64", "float32"):
            raise ValueError("precision must be float64 or float32")
        return precision

    def _selected_runtime(self, precision) -> tuple[str, str]:
        return "jax", self._precision_options(
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
    def _require_spec(spec) -> None:
        supported = (
            DirectSolveSpec, DiagonalRootSpec, OperatorElementRootSpec,
            HybridizedPoleSpec, TransferZeroSpec, ResidueNormalizedCouplingSpec,
            ResponseElementSpec, OperatorSpec, StaticCapacitanceSpec, StaticCapacitanceElementSpec,
        )
        quantities = (selector.spec for objective in spec.objectives
                      for selector in _quantity_selectors(objective.quantity)) if isinstance(spec, OptimizationSpec) else (spec,)
        for quantity in quantities:
            if not isinstance(quantity, supported):
                raise RuntimePreparationError(
                    f"{type(quantity).__name__} is not supported by the JAX runtime.", stage="runtime_prepare")

    @contextmanager
    def _execution_scope(self, method, precision, start_tick_ns, timing=None):
        from .diagnostics.operations import OperationRecorder
        backend, precision = self._selected_runtime(precision)
        recorder = OperationRecorder(self._binding, kind=method, backend=backend,
                                     precision=precision, start_tick_ns=start_tick_ns,
                                     timing=self._selected_timing(timing))
        with operation_lease(self._binding, recorder.operation_id) as lease:
            with recorder as trace:
                yield backend, precision, trace, lease

    @overload
    def solve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec,
        *,
        parameters: ParameterSet | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> DirectSolveResult | InMemoryDirectSolveResult: ...


    @overload
    def solve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec,
        *,
        parameters: ParameterSpace,
        precision: str | None = None,
        timing: str | None = None,
    ) -> ParameterSweepResult: ...

    @run_activity
    def solve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> DirectSolveResult | InMemoryDirectSolveResult | ParameterSweepResult | InMemoryParameterSweepResult:
        """Execute the selected Direct response."""

        started_ns = perf_counter_ns()
        if self._storage == "memory":
            if isinstance(parameters, ParameterSpace):
                return self._analyze_memory("solve_direct", ref, spec, parameters=parameters,
                                            precision=precision,
                                            timing=timing, started_ns=started_ns)
            return self._solve_memory(ref, spec, parameters=parameters,
                                      precision=precision, timing=timing, started_ns=started_ns)
        prepared: PreparedAnalysis
        with self._execution_scope("solve", precision, started_ns, timing) as (backend, precision, trace, lease):
            self._require_ref(ref)
            self._require_spec(spec)
            operation = "solve_direct"
            with trace.span("preparation"):
                prepared = self._prepare_analysis(operation, ref, spec, parameters,
                                                  precision=precision)
            return self._execute(prepared, bound_spec=spec, trace=trace)

    @overload
    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: DiagonalRootSpec,
        *,
        parameters: ParameterSet | None = None,
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
        precision: str | None = None,
        timing: str | None = None,
    ) -> ParameterSweepResult: ...

    @overload
    def evaluate(self, ref: NetworkViewRef, spec: StaticCapacitanceSpec, *,
                 parameters: ParameterSet | None = None,
                 precision: str | None = None, timing: str | None = None) -> StaticCapacitanceResult: ...

    @overload
    def evaluate(self, ref: NetworkViewRef, spec: StaticCapacitanceElementSpec, *,
                 parameters: ParameterSet | None = None,
                 precision: str | None = None, timing: str | None = None) -> StaticCapacitanceElementResult: ...

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
        | OperatorSpec
        | StaticCapacitanceSpec
        | StaticCapacitanceElementSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
        precision: str | None = None,
        timing: str | None = None,
    ) -> (
        DiagonalRootResult
        | OperatorElementRootResult
        | DirectQuantityResult
        | OperatorResult
        | StaticCapacitanceResult
        | StaticCapacitanceElementResult
        | ParameterSweepResult
        | InMemoryAnalysisResult
        | InMemoryParameterSweepResult
    ):
        """Evaluate one typed Direct quantity without an unrelated sweep."""
        if isinstance(spec, (StaticCapacitanceSpec, StaticCapacitanceElementSpec)) and isinstance(parameters, ParameterSpace):
            raise RuntimePreparationError("Static capacitance currently supports a single ParameterSet", stage="preflight")
        started_ns = perf_counter_ns()
        if self._storage == "memory":
            return self._analyze_memory("evaluate_direct", ref, spec, parameters=parameters,
                                        precision=precision,
                                        timing=timing, started_ns=started_ns)
        with self._execution_scope("evaluate", precision, started_ns, timing) as (backend, precision, trace, lease):
            self._require_ref(ref)
            self._require_spec(spec)
            with trace.span("preparation"):
                prepared = self._prepare_analysis("evaluate_direct", ref, spec, parameters,
                                                  precision=precision)
            return self._execute(prepared, bound_spec=spec, trace=trace)

    def _sensitivity_quantities(self, quantities, *, encoded: bool):
        """Bind every expression leaf separately without retaining View closures."""
        from .specs import _sensitivity_quantity_record

        def bind(value):
            if isinstance(value, QuantitySum):
                return {"type": "quantity_sum", "terms": tuple(bind(term) for term in value.terms)}
            if isinstance(value, QuantityDifference):
                return {"type": "quantity_difference", "left": bind(value.left), "right": bind(value.right)}
            if isinstance(value, QuantityAbsolute):
                return {"type": "quantity_absolute", "operand": bind(value.operand)}
            if not isinstance(value, QuantitySelector):
                raise TypeError("quantities must contain bound physical selectors")
            selected = value._view
            self._require_ref(selected)
            if not encoded:
                return _sensitivity_quantity_record(value)
            selected_spec = value.spec
            if not isinstance(selected_spec, DirectSolveSpec):
                self._validate_direct_quantity_spec("sensitivity", selected, selected_spec)
            spec_record = _encode_spec(
                selected_spec, self._baseline_point.effective_parameters,
                coordinate_bindings={_coordinate_binding_key(coordinate): (
                    self._static_coordinate_id(selected, coordinate)
                    if isinstance(selected_spec, (StaticCapacitanceSpec, StaticCapacitanceElementSpec))
                    else self._view_coordinate_id(selected, coordinate))
                    for coordinate in _quantity_coordinates(selected_spec)}
                if not isinstance(selected_spec, DirectSolveSpec) else {},
                trace_channels={channel: self._trace_request_channel(selected, channel)
                                for trace in getattr(selected_spec, "traces", ())
                                for channel in (trace.input_port, trace.output_port)},
            )
            return {"type": value.type, "projection": value.projection, "spec": spec_record,
                    "view": {**_view_declaration(selected._lineage), "plan_sha256": self._plan_sha256}}

        return tuple({"id": label, "quantity": bind(quantity)} for label, quantity in quantities.items())

    @run_activity
    def prepare(
        self, *, quantities: Mapping[str, object],
        parameters: ParameterSet | None = None,
        wrt: Sequence[ParameterRef] | None = None,
        discretization: FixedMeshSpec | None = None,
        precision: str | None = None,
    ) -> PreparedFunction:
        """Acquire a detached physical callable; do not solve or publish Results.

        Compilation expands ordered structured continuous coordinates. Omitted
        wrt selects its canonical captured continuous order. Remaining effective
        point values are constants, without any Run/Workspace closure capture.
        """
        from .compilation.prepared_function import compile_prepared_function

        backend, selected_precision = self._selected_runtime(precision)
        from ._native import require_compensated_kernel
        require_compensated_kernel()
        declarations = self._sensitivity_quantities(quantities, encoded=True)
        point = self._complete_parameters(parameters)
        active = None if wrt is None else tuple(
            _parameter_key(self._compatible_parameter(parameter)) for parameter in wrt
        )
        if discretization is not None and not isinstance(discretization, FixedMeshSpec):
            raise TypeError("discretization must be FixedMeshSpec or None")
        return compile_prepared_function(
            plan_document=self._plan_document,
            quantity_declarations=declarations,
            parameter_record=point.parameter_record,
            wrt=active,
            mesh=None if discretization is None else discretization._canonical_record(),
            precision=selected_precision,
            program_owner=self._warm_executables,
        )

    @run_activity
    def dependencies(self, *, quantities: Mapping[str, object]):
        """Read declared physical consumers without mesh or device preparation."""
        from .compilation.dependencies import describe_dependencies
        declarations = self._sensitivity_quantities(quantities, encoded=True)
        return describe_dependencies(self._plan_document, declarations,
                                     self._baseline_point.parameter_record)

    def _encoded_sensitivity_spec(self, spec: SensitivitySpec):
        if not isinstance(spec, SensitivitySpec):
            raise TypeError("spec must be SensitivitySpec")
        for parameter in spec.wrt:
            self._compatible_parameter(parameter)
        declaration = dict(spec._canonical_record())
        declaration["quantities"] = self._sensitivity_quantities(spec.quantities, encoded=True)
        return declaration

    def _prepare_sensitivity(self, spec: SensitivitySpec, parameters, *, precision):
        from .execution.sensitivity_request import prepare_sensitivity
        declaration = self._encoded_sensitivity_spec(spec)
        point = self._complete_parameters(parameters)
        return prepare_sensitivity(
            plan_sha256=self._plan_sha256, encoded_spec=declaration,
            parameter_record=point.parameter_record, runtime_base=self._runtime_base,
            precision=precision,
            source_units=self._source_units(spec, point.effective_parameters, parameter_space=parameters),
        )

    @run_activity
    def sensitivity(self, spec: SensitivitySpec, *, parameters: ParameterSet | None = None,
                    precision: str | None = None,
                    timing: str | None = None) -> SensitivityResult | InMemorySensitivityResult:
        """Execute the requested checked derivative of the same physical program."""
        started_ns = perf_counter_ns()
        if self._storage == "memory":
            return self._sensitivity_memory(spec, parameters=parameters,
                                             precision=precision, timing=timing, started_ns=started_ns)
        backend, precision = self._selected_runtime(precision)
        started_ns = perf_counter_ns()
        with self._execution_scope("sensitivity", precision, started_ns, timing) as (backend, precision, trace, lease):
            with trace.span("preparation"):
                prepared = self._prepare_sensitivity(spec, parameters, precision=precision)
            from .execution.jax_operation import execute_sensitivity_operation
            return execute_sensitivity_operation(
                binding=self._binding, plan_document=self._plan_document,
                prepared_analysis=prepared, decoder=self._result_decoder(),
                bound_spec=spec, trace=trace,
                precision=precision, backend=backend,
                program_owner=self._warm_executables,
                execution_policy=self._execution_policy,
            )

    @run_activity
    def warmup(
        self,
        view: NetworkViewRef | PreparedFunction,
        spec: object | None = None,
        *,
        parameters: ParameterSet | None = None,
        transforms: Sequence[str] = ("value",),
        batch_shapes: Sequence[tuple[int, ...]] = (),
    ):
        """Compile declared signatures from detached or view/spec acquisition.

        batch_shapes declare leading theta point-batch dimensions. View/spec
        warming plans the actual checked operation, including ordered Objective
        loss and each declared Hybrid stage. Prepared transforms remain raw,
        caller-owned signatures. Runtime coefficients do not change static
        variant identity within fixed operand modes; actual-frame consumers
        remain explicitly lazy when their exact producer tree is unavailable.
        Compilation performs no solve or successful evidence publication.
        """
        from .execution.warmup import prepare_warmup, warm_prepared

        from ._native import require_compensated_kernel
        require_compensated_kernel()
        if isinstance(view, PreparedFunction):
            if spec is not None or parameters is not None:
                raise TypeError("prepared warmup does not take a spec or parameters")
            with (nullcontext() if self._storage == "memory" else operation_lease(self._binding, str(uuid4()))):
                return warm_prepared(view, executable_owner=self._warm_executables,
                                     transforms=transforms, batch_shapes=batch_shapes)
        self._require_ref(view)
        if isinstance(spec, SensitivitySpec):
            # Each finalized expression leaf owns its exact View; this argument
            # verifies Run ownership rather than imposing a single leaf View.
            prepared = self._prepare_sensitivity(spec, parameters, precision=self._precision)
        else:
            self._require_spec(spec)
            operation = ("optimize_direct" if isinstance(spec, OptimizationSpec) else
                         "solve_direct" if isinstance(spec, DirectSolveSpec) else "evaluate_direct")
            prepared = self._prepare_analysis(operation, view, spec, parameters,
                                              precision=self._precision)
        with (nullcontext() if self._storage == "memory" else operation_lease(self._binding, str(uuid4()))):
            return prepare_warmup(
                plan_document=self._plan_document, analysis=prepared,
                precision=self._precision, executable_owner=self._warm_executables,
                baseline_parameters=self._baseline_point.parameter_record,
                transforms=transforms, batch_shapes=batch_shapes,
            )

    @run_activity
    def benchmark(
        self, *, operations=None, method=None, precision=None,
        status=None,
    ):
        """Read recorded operation traces from this Plan leaf without execution."""
        if self._storage == "memory":
            if any(value is not None for value in (operations, method, precision, status)):
                raise TypeError("Memory benchmark returns only the last operation and accepts no filters")
            if self._last_memory_benchmark is None:
                raise ResultUnavailableError("No in-memory operation has completed", stage="memory_benchmark")
            return self._last_memory_benchmark
        from .benchmark.api import benchmark
        return benchmark(self._binding, operations=operations, method=method,
                         precision=precision, status=status)

    @run_activity
    def recover_workspace(self) -> None:
        """Explicitly recover only this Run's bound Plan-leaf operation store."""
        self._require_workspace_storage("recover_workspace")
        from .workspace.evidence import recover_operation_workspace

        with self._binding.writer():
            recover_operation_workspace(self._binding)

    @overload
    def optimize(
        self,
        spec: OptimizationSpec,
        *,
        parameters: ParameterSet | None = None,
        precision: str | None = None,
        timing: str | None = None,
        progress: bool = True,
        on_progress: Callable[[OptimizationProgress], object] | None = None,
        checkpoint: str | object = _OMITTED_CHECKPOINT,
        resume_from: Mapping[str, object] | None = None,
        restart_from: AcceptedIterateSelection | None = None,
        commit_every_generations: int | object = _OMITTED_COMMIT_EVERY_GENERATIONS,
    ) -> OptimizationResult | InMemoryOptimizationResult: ...

    @overload
    def optimize(
        self,
        ref: NetworkViewRef,
        spec: OptimizationSpec,
        *,
        parameters: ParameterSet | None = None,
        precision: str | None = None,
        timing: str | None = None,
        progress: bool = True,
        on_progress: Callable[[OptimizationProgress], object] | None = None,
        checkpoint: str | object = _OMITTED_CHECKPOINT,
        resume_from: Mapping[str, object] | None = None,
        restart_from: AcceptedIterateSelection | None = None,
        commit_every_generations: int | object = _OMITTED_COMMIT_EVERY_GENERATIONS,
    ) -> OptimizationResult | InMemoryOptimizationResult: ...

    @run_activity
    def optimize(
        self,
        ref_or_spec: NetworkViewRef | OptimizationSpec,
        spec: OptimizationSpec | None = None,
        *,
        parameters: ParameterSet | None = None,
        precision: str | None = None,
        timing: str | None = None,
        progress: bool = True,
        on_progress: Callable[[OptimizationProgress], object] | None = None,
        checkpoint: str | object = _OMITTED_CHECKPOINT,
        resume_from: Mapping[str, object] | None = None,
        restart_from: AcceptedIterateSelection | None = None,
        commit_every_generations: int | object = _OMITTED_COMMIT_EVERY_GENERATIONS,
    ) -> OptimizationResult | InMemoryOptimizationResult:
        """Run the declared physical objective strategy with explicit persistence."""
        if checkpoint is _OMITTED_CHECKPOINT:
            checkpoint = "off" if self._storage == "memory" else "generation"
        if self._storage == "memory":
            if checkpoint != "off" or resume_from is not None or restart_from is not None:
                raise RuntimePreparationError(
                    "Memory optimization has no durable checkpoint, resume or saved accepted-iterate restart",
                    stage="memory_capability",
                )

        started_ns = perf_counter_ns()
        with self._optimization_scope(precision, started_ns, timing) as (backend, precision, trace, lease):
            generation_cadence_supplied = commit_every_generations is not _OMITTED_COMMIT_EVERY_GENERATIONS
            if commit_every_generations is _OMITTED_COMMIT_EVERY_GENERATIONS:
                commit_every_generations = 10
            if isinstance(commit_every_generations, bool) or not isinstance(commit_every_generations, int):
                raise TypeError("commit_every_generations must be a positive integer")
            if commit_every_generations <= 0:
                raise ValueError("commit_every_generations must be a positive integer")
            if parameters is not None and not isinstance(parameters, ParameterSet):
                raise TypeError(
                    "OptimizationSpec parameters must be a ParameterSet or None"
                )
            if not isinstance(progress, bool):
                raise TypeError("progress must be bool")
            if on_progress is not None and not callable(on_progress):
                raise TypeError("on_progress must be callable or None")
            default_ref, optimization_spec = self._optimization_arguments(ref_or_spec, spec)
            has_cma = isinstance(optimization_spec.optimizer, CMAESSpec) or (
                isinstance(optimization_spec.optimizer, HybridSpec)
                and any(isinstance(stage.optimizer, CMAESSpec) for stage in optimization_spec.optimizer.stages)
            )
            if generation_cadence_supplied and not has_cma:
                raise TypeError("commit_every_generations applies to CMA stages; local stages use commit_every_iterations")
            if restart_from is not None:
                if not isinstance(restart_from, AcceptedIterateSelection):
                    raise TypeError("restart_from must be AcceptedIterateSelection")
                if resume_from is not None:
                    raise ValueError("restart_from and exact resume_from are distinct operations")
                if isinstance(optimization_spec.optimizer, CMAESSpec):
                    raise RuntimePreparationError("accepted-iterate restart requires local or Hybrid JAX optimization", stage="runtime_prepare")
            ref, selector_views = self._optimization_views(
                optimization_spec, default_ref=default_ref
            )
            self._require_spec(optimization_spec)
            if checkpoint not in ("generation", "off"):
                raise ValueError("checkpoint must be generation or off")
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
                            precision=precision,
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
                if self._storage == "memory":
                    from .execution.memory_optimization import execute_memory_optimization
                    from .results.memory_optimization import build_memory_optimization_result
                    trace.bind(operation="optimize_direct", request_sha256=prepared.request_sha256)
                    payload = execute_memory_optimization(
                        plan_document=self._plan_document, prepared_analysis=prepared,
                        trace=trace, program_owner=self._warm_executables,
                        execution_policy=self._execution_policy,
                        commit_every_generations=commit_every_generations,
                        on_progress=on_progress, progress_observer=display,
                    )
                    with trace.span("result_assembly"):
                        result = build_memory_optimization_result(
                            payload=payload, request=prepared.request(),
                            plan_sha256=self._plan_sha256, request_sha256=prepared.request_sha256,
                            execution_id=trace.execution_id,
                            parameter_lookup=MappingProxyType(dict(self._parameter_lookup)),
                        )
                else:
                    result = self._execute(
                        prepared, bound_spec=optimization_spec, on_progress=on_progress,
                        progress_observer=display, checkpoint_policy=checkpoint,
                        resume_from=resume_from,
                        restart_from=None if restart_from is None else restart_from._canonical_record(),
                        commit_every_generations=commit_every_generations, trace=trace,
                    )
            finally:
                if display is not None:
                    display.close()
        if self._storage == "memory":
            from .results.memory_optimization import attach_memory_optimization_benchmark
            return attach_memory_optimization_benchmark(result, trace.snapshot())
        return result

    @overload
    def resolve(self, ref: NetworkViewRef, spec: DirectSolveSpec, *, parameters: ParameterSet | ParameterSpace | None = None, precision: str | None = None, attempt: object | None = None) -> DirectSolveResult | ParameterSweepResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: DiagonalRootSpec, *, parameters: ParameterSet | ParameterSpace | None = None, precision: str | None = None, attempt: object | None = None) -> DiagonalRootResult | ParameterSweepResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: OperatorElementRootSpec, *, parameters: ParameterSet | ParameterSpace | None = None, precision: str | None = None, attempt: object | None = None) -> OperatorElementRootResult | ParameterSweepResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: OptimizationSpec, *, parameters: ParameterSet | None = None, precision: str | None = None, attempt: object | None = None) -> OptimizationResult: ...


    @overload
    def resolve(self, ref: NetworkViewRef, spec: OperatorSpec, *, parameters: ParameterSet | ParameterSpace | None = None, precision: str | None = None, attempt: object | None = None) -> OperatorResult | ParameterSweepResult: ...

    @overload
    def resolve(
        self,
        ref: NetworkViewRef,
        spec: HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
        precision: str | None = None,
    ) -> DirectQuantityResult | ParameterSweepResult: ...

    @staticmethod
    def _terminal_attempt_id(attempt):
        if attempt is None:
            return None
        if isinstance(attempt, str):
            identity = attempt
        elif isinstance(attempt, Mapping):
            identity = attempt.get("attempt_id")
        else:
            identity = getattr(attempt, "attempt_id", None)
        if not isinstance(identity, str) or not identity:
            raise TypeError("attempt must supply an exact terminal attempt_id")
        return identity

    @overload
    def resolve(self, spec: SensitivitySpec, *, parameters: ParameterSet | None = None,
                precision: str | None = None,
                attempt: object | None = None) -> SensitivityResult | None: ...

    @run_activity
    def resolve(
        self,
        ref: NetworkViewRef | SensitivitySpec,
        spec: DirectSolveSpec
        | DiagonalRootSpec
        | OperatorElementRootSpec
        | HybridizedPoleSpec
        | TransferZeroSpec
        | ResidueNormalizedCouplingSpec
        | ResponseElementSpec
        | OperatorSpec
        | StaticCapacitanceSpec
        | StaticCapacitanceElementSpec
        | OptimizationSpec
        | None = None,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
        precision: str | None = None,
        attempt: object | None = None,
    ) -> (
        DirectSolveResult

        | DiagonalRootResult
        | OperatorElementRootResult
        | DirectQuantityResult
        | OperatorResult
        | StaticCapacitanceResult
        | StaticCapacitanceElementResult
        | OptimizationResult
        | ParameterSweepResult
        | SensitivityResult
        | None
    ):
        """Read exact success, or an explicitly selected sealed terminal."""
        self._require_workspace_storage("resolve")

        requested_precision = precision
        backend, precision = self._selected_runtime(precision)
        attempt_id = self._terminal_attempt_id(attempt)
        if isinstance(ref, SensitivitySpec):
            if spec is not None:
                raise TypeError("resolve received both a SensitivitySpec and a second Spec")
            from .execution.jax_operation import resolve_jax_operation
            if attempt_id is not None and parameters is None:
                from .execution.sensitivity_request import finalize_sensitivity_spec
                expected_spec = finalize_sensitivity_spec(self._encoded_sensitivity_spec(ref))
                return resolve_jax_operation(
                    binding=self._binding, prepared_analysis=None,
                    decoder=self._result_decoder(), bound_spec=ref, attempt_id=attempt_id,
                    expected_spec=expected_spec, expected_backend="jax",
                    expected_precision=requested_precision,
                )
            prepared = self._prepare_sensitivity(ref, parameters, precision=precision)
            return resolve_jax_operation(binding=self._binding, prepared_analysis=prepared,
                                         decoder=self._result_decoder(), bound_spec=ref,
                                         attempt_id=attempt_id)
        self._require_spec(spec)
        self._require_ref(ref)
        if isinstance(spec, DirectSolveSpec):
            operation = "solve_direct"
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
                StaticCapacitanceSpec,
                StaticCapacitanceElementSpec,
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
            precision=precision,
        )
        from .execution.jax_operation import resolve_jax_operation
        return resolve_jax_operation(binding=self._binding, prepared_analysis=prepared,
                                     decoder=self._result_decoder(), bound_spec=spec, attempt_id=attempt_id)

    @run_activity
    def explain(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec
        | DiagonalRootSpec
        | OperatorElementRootSpec
        | HybridizedPoleSpec
        | TransferZeroSpec
        | ResidueNormalizedCouplingSpec
        | ResponseElementSpec
        | OperatorSpec
        | StaticCapacitanceSpec
        | StaticCapacitanceElementSpec
        | OptimizationSpec,
        *,
        parameters: ParameterSet | None = None,
        precision: str | None = None,
    ) -> ExplanationResult:
        """Explain the JAX compiler without creating an attempt or solve."""
        self._require_workspace_storage("explain")

        backend, precision = self._selected_runtime(precision)
        self._require_ref(ref)
        if not isinstance(
            spec,
            (
                DirectSolveSpec,
                DiagonalRootSpec,
                OperatorElementRootSpec,
                HybridizedPoleSpec,
                TransferZeroSpec,
                ResidueNormalizedCouplingSpec,
                ResponseElementSpec,
                OperatorSpec,
                StaticCapacitanceSpec,
                StaticCapacitanceElementSpec,
                OptimizationSpec,
            ),
        ):
            unavailable(f"CircuitRun.explain({type(spec).__name__})")
        operation = (
            "solve_direct"
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
                    StaticCapacitanceSpec,
                    StaticCapacitanceElementSpec,
                ),
            )
            else "optimize_direct"
        )
        self._require_spec(spec)
        selector_views = None
        if isinstance(spec, OptimizationSpec):
            ref, selector_views = self._optimization_views(spec, default_ref=ref)
        prepared = self._prepare_analysis(
            operation,
            ref,
            spec,
            parameters,
            selector_views=selector_views,
            precision=precision,
        )
        request = prepared.request()
        operation_id = str(uuid4())
        with operation_lease(self._binding, operation_id) as lease:
            from .execution.compilation import _run_jax_preflight
            compiled = _run_jax_preflight(self._plan_document, request)
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
    def build_report(self, spec: ReportSpec) -> ReportResult:
        """Derive a pure report from checked saved or detached numerical Results."""
        from .results import _is_checked_report_input

        if (
            not isinstance(spec, ReportSpec)
            or not spec.inputs
            or not all(_is_checked_report_input(result) for result in spec.inputs)
        ):
            raise TypeError("build_report() requires ReportSpec")
        from .visualization.report import build_report

        return build_report(spec)

    def _solve_memory(self, ref, spec, *, parameters, precision, timing, started_ns):
        """Own the full Direct call without entering any saved-operation scope."""
        backend, precision = self._selected_runtime(precision)
        if not isinstance(spec, DirectSolveSpec) or isinstance(parameters, ParameterSpace):
            raise RuntimePreparationError(
                "Memory solve supports one JAX DirectSolveSpec point only",
                stage="memory_capability",
            )
        if self._selected_timing(timing) != "aggregate":
            raise ValueError("Memory solve supports timing='aggregate' only")
        self._require_ref(ref)
        from .diagnostics.memory_operation import MemoryOperationRecorder
        from .execution.memory_direct import execute_memory_direct
        from .results.memory_direct import build_memory_direct_result, attach_memory_benchmark

        execution_id = str(uuid4())
        trace = MemoryOperationRecorder(execution_id=execution_id, kind="solve",
                                        backend=backend, precision=precision,
                                        start_tick_ns=started_ns)
        primary = None
        try:
            with trace.span("preparation"):
                prepared = self._prepare_analysis("solve_direct", ref, spec, parameters,
                                                  precision=precision)
                trace.bind(operation="solve_direct", request_sha256=prepared.request_sha256)
            # The execution owner drains and closes its operation resources
            # before returning arrays; this facade also times typed assembly.
            payload = execute_memory_direct(
                plan_document=self._plan_document, prepared_analysis=prepared,
                trace=trace, program_owner=self._warm_executables,
                execution_policy=self._execution_policy,
            )
            with trace.span("result_assembly"):
                result = build_memory_direct_result(
                    payload=payload, request=prepared.request(),
                    plan_sha256=self._plan_sha256, request_sha256=prepared.request_sha256,
                    execution_id=execution_id,
                )
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                self._last_memory_benchmark = trace.finish(error=primary)
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    f"In-memory timing finalization also failed: {type(cleanup_error).__name__}: {cleanup_error}"
                )
        return attach_memory_benchmark(result, trace.snapshot())

    @contextmanager
    def _optimization_scope(self, precision, started_ns, timing):
        """Select the actual storage owner without changing optimization math."""
        if self._storage != "memory":
            with self._execution_scope("optimize", precision, started_ns, timing) as scope:
                yield scope
            return
        with self._memory_scope("optimize", precision, started_ns, timing) as scope:
            yield (*scope, None)

    @contextmanager
    def _memory_scope(self, kind, precision, started_ns, timing):
        """Own one RAM operation clock through producer drain and projection."""
        backend, precision = self._selected_runtime(precision)
        if self._selected_timing(timing) != "aggregate":
            raise ValueError("Memory execution supports timing='aggregate' only")
        from .diagnostics.memory_operation import MemoryOperationRecorder
        trace = MemoryOperationRecorder(execution_id=str(uuid4()), kind=kind,
                                        backend=backend, precision=precision,
                                        start_tick_ns=started_ns)
        primary = None
        try:
            yield backend, precision, trace
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                self._last_memory_benchmark = trace.finish(error=primary)
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    f"In-memory timing finalization also failed: {type(cleanup_error).__name__}: {cleanup_error}"
                )

    def _analyze_memory(self, operation, ref, spec, *, parameters,
                        precision, timing, started_ns):
        """Bind ordinary point/Sweep declarations to the sole RAM producer."""
        from .execution.memory_analysis import execute_memory_analysis
        from .results.memory_analysis import (
            build_memory_analysis_result, attach_memory_analysis_benchmark,
        )
        kind = "solve" if operation == "solve_direct" else "evaluate"
        with self._memory_scope(kind, precision, started_ns, timing) as (backend, precision, trace):
            self._require_ref(ref)
            self._require_spec(spec)
            with trace.span("preparation"):
                prepared = self._prepare_analysis(operation, ref, spec, parameters,
                                                  precision=precision)
                trace.bind(operation=operation, request_sha256=prepared.request_sha256)
                from .execution.prepared import capture_selector_binding
                selector_binding = capture_selector_binding(
                    plan_owner_id=id(self._plan), coordinate_lookup=self._coordinate_lookup,
                    view=prepared.request()["view"],
                )
            payload = execute_memory_analysis(
                plan_document=self._plan_document, prepared_analysis=prepared,
                trace=trace, program_owner=self._warm_executables,
                execution_policy=self._execution_policy,
            )
            with trace.span("result_assembly"):
                result = build_memory_analysis_result(
                    payload=payload, request=prepared.request(),
                    plan_sha256=self._plan_sha256, request_sha256=prepared.request_sha256,
                    execution_id=trace.execution_id,
                    parameter_lookup=MappingProxyType(dict(self._parameter_lookup)),
                    selector_binding=selector_binding,
                )
        return attach_memory_analysis_benchmark(result, trace.snapshot())

    def _sensitivity_memory(self, spec, *, parameters, precision,
                            timing, started_ns):
        """Keep checked derivative payload and Result ownership separate."""
        from .execution.memory_analysis import execute_memory_sensitivity
        from .results.memory_sensitivity import (
            build_memory_sensitivity_result, attach_memory_sensitivity_benchmark,
        )
        with self._memory_scope("sensitivity", precision, started_ns, timing) as (backend, precision, trace):
            with trace.span("preparation"):
                prepared = self._prepare_sensitivity(spec, parameters, precision=precision)
                trace.bind(operation="sensitivity", request_sha256=prepared.request_sha256)
            payload = execute_memory_sensitivity(
                plan_document=self._plan_document, prepared_analysis=prepared,
                trace=trace, program_owner=self._warm_executables,
                execution_policy=self._execution_policy,
            )
            with trace.span("result_assembly"):
                result = build_memory_sensitivity_result(
                    payload=payload, request=prepared.request(),
                    plan_sha256=self._plan_sha256, request_sha256=prepared.request_sha256,
                    execution_id=trace.execution_id,
                    parameter_lookup=MappingProxyType(dict(self._parameter_lookup)),
                )
        return attach_memory_sensitivity_benchmark(result, trace.snapshot())

    def _require_workspace_storage(self, method: str) -> None:
        if self._storage == "memory":
            raise RuntimePreparationError(
                f"CircuitRun.{method} is unavailable with storage='memory'; memory execution supports checked JAX analysis, Sensitivity and Optimization",
                stage="memory_capability",
            )

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

    def _static_coordinate_id(self, ref: NetworkViewRef, value: CoordinateRef | ElectricNodeRef | str) -> str:
        """Bind public handles/aliases or retain a canonical static output-axis ID.

        Full finite-mesh node membership belongs to the same-point compiler.
        This does not expose internal nodes to RF View/retain declarations.
        """
        if isinstance(value, str) and value not in ref._available_coordinates and value not in self._coordinate_lookup:
            return value
        return self._view_coordinate_id(ref, value)

    def _validate_direct_request(
        self,
        operation: str,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | DiagonalRootSpec | OperatorElementRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | StaticCapacitanceSpec | StaticCapacitanceElementSpec | OptimizationSpec,
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
        if isinstance(spec, (DiagonalRootSpec, OperatorElementRootSpec, HybridizedPoleSpec, TransferZeroSpec, ResidueNormalizedCouplingSpec, ResponseElementSpec, OperatorSpec, StaticCapacitanceSpec, StaticCapacitanceElementSpec)):
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
        spec: DiagonalRootSpec | OperatorElementRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | StaticCapacitanceSpec | StaticCapacitanceElementSpec,
        *,
        residue_branch: bool = False,
    ) -> None:
        """Apply the selected-View contract shared by evaluate and CMA selectors."""

        if isinstance(spec, (StaticCapacitanceSpec, StaticCapacitanceElementSpec)):
            # View binding owns public-coordinate retention and PTC lineage.
            # An element binds after the same compiled static partition; full
            # View elements may still address returned finite-mesh axis IDs.
            if isinstance(spec, StaticCapacitanceElementSpec):
                for value in (spec.row, spec.column):
                    self._static_coordinate_id(ref, value)
            return
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

        stages = spec.optimizer.stages if isinstance(spec.optimizer, HybridSpec) else ()
        controls = tuple(stage.optimizer for stage in stages) or (spec.optimizer,)
        for control in controls:
            if isinstance(control, (LBFGSBSpec, ProjectedLBFGSSpec)) and control.coordinate_scales is not None:
                for parameter in control.coordinate_scales:
                    self._compatible_parameter(parameter)
        for stage in stages:
            if stage.seed is not None:
                for parameter in stage.seed.values:
                    self._compatible_parameter(parameter)

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
                if isinstance(selector.spec, (StaticCapacitanceSpec, StaticCapacitanceElementSpec)):
                    raise RuntimePreparationError("Static capacitance optimization is not supported", stage="preflight")
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
        | DiagonalRootSpec
        | OperatorElementRootSpec
        | HybridizedPoleSpec
        | TransferZeroSpec
        | ResidueNormalizedCouplingSpec
        | ResponseElementSpec
        | OperatorSpec
        | StaticCapacitanceSpec
        | StaticCapacitanceElementSpec
        | OptimizationSpec,
        parameters: ParameterSet | ParameterSpace | None,
        *,
        selector_views: Mapping[int, NetworkViewRef] | None = None,
        precision: str = "float64",
    ) -> PreparedAnalysis:
        """Normalize one non-executable analysis declaration exactly once."""

        if isinstance(spec, (StaticCapacitanceSpec, StaticCapacitanceElementSpec)) and isinstance(parameters, ParameterSpace):
            raise RuntimePreparationError("Static capacitance currently supports a single ParameterSet", stage="preflight")
        if isinstance(spec, OptimizationSpec) and selector_views is None:
            ref, selector_views = self._optimization_views(spec, default_ref=ref)
        self._validate_direct_request(
            operation,
            ref,
            spec,
            selector_views=selector_views,
        )
        parameter_source, resolved = self._parameter_source(parameters)
        parameter_authorizations = tuple(ref._key_record()
            for ref in resolved.effective_parameters.allow_extrapolation)
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
                        _coordinate_binding_key(coordinate): (
                            self._static_coordinate_id(ref, coordinate)
                            if isinstance(spec, (StaticCapacitanceSpec, StaticCapacitanceElementSpec))
                            else self._view_coordinate_id(ref, coordinate))
                        for coordinate in _quantity_coordinates(spec)
                    }
                    if not isinstance(spec, DirectSolveSpec)
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
            source_units=source_units, precision=precision,
            parameter_authorizations=parameter_authorizations,
        )


    def _execute(
        self,
        prepared_analysis: PreparedAnalysis,
        *,
        trace,
        bound_spec: object | None = None,
        on_progress: Callable[[OptimizationProgress], object] | None = None,
        progress_observer: object | None = None,
        checkpoint_policy: str = "generation",
        resume_from: Mapping[str, object] | None = None,
        restart_from: Mapping[str, object] | None = None,
        commit_every_generations: int = 1,
    ):
        from .execution.jax_operation import execute_jax_operation
        return execute_jax_operation(
            binding=self._binding, plan_document=self._plan_document,
            prepared_analysis=prepared_analysis, decoder=self._result_decoder(),
            bound_spec=bound_spec, on_progress=on_progress,
            progress_observer=progress_observer,
            checkpoint_policy=checkpoint_policy, resume_from=resume_from, restart_from=restart_from,
            commit_every_generations=commit_every_generations, trace=trace,
            program_owner=self._warm_executables,
            execution_policy=self._execution_policy,
        )

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
