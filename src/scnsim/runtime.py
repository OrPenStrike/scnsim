"""Plan-bound execution, exact request identity, and typed Result reconstruction.

One captured authoring snapshot and its resolved parameter points feed the
declarative Direct, HB, and optimization request boundary. Workspace receipts
and artifact manifests remain the only authority for reconstructing results.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from hashlib import sha256
from html import escape
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import overload

import numpy as np

from . import units
from ._analysis import (
    BoundOptimization,
    BoundOptimizationLeaf,
    PreparedAnalysis,
    _coordinate_binding_key,
    _encode_direct_quantity,
    _encode_spec,
    _quantity_coordinates,
)
from ._authoring_snapshot import ResolvedPlanPoint, freeze
from ._backend import (
    prepare_runtime,
    run_compiler_audit,
    run_preflight,
)
from ._canonical import (
    _identifier as _canonical_identifier,
    canonical_expanded_graph_sha256,
    canonical_json_bytes,
    canonical_plan_snapshot,
    canonical_resolved_plan_point,
    complex_quantity_envelope,
    quantity_envelope,
    sha256_hex,
)
from ._evidence import (
    _VerifiedEvidenceLease,
    _error_from_record,
    _validated_failure_record,
    _verified_evidence_lease,
)
from ._parameter_resolution import resolve_parameter_point
from ._physical_values import RLGC, RLGCParameterSpec
from ._scaffold import unavailable
from ._workspace import (
    VerifiedSuccess,
    _plan_coordinates,
    bind_workspace,
)
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
    BackendProtocolError,
    CompilerInvariantError,
    EvidenceIntegrityError,
    InvalidOptimizationSpec,
    PortRealizabilityError,
    SCNSimValidationError,
)
from .presentation import _report_html
from .results import (
    DiagonalRootResult,
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
    DirectSolveSpec,
    HBSolveSpec,
    HybridizedPoleSpec,
    OperatorSpec,
    OptimizationSpec,
    QuantitySelector,
    QuantitySum,
    ReportSpec,
    ResidueNormalizedCouplingSpec,
    ResponseElementSpec,
    TransferZeroSpec,
    _selector_unit,
)


def _coordinate_id(value: str | ElectricNodeRef | CoordinateRef) -> str:
    if isinstance(value, str):
        if not value:
            raise ValueError("coordinate IDs must not be empty")
        return value
    identifier = getattr(value, "id", None)
    if isinstance(identifier, str) and identifier:
        return identifier
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    raise TypeError("coordinate must be a public SCNSim coordinate handle or ID")


def _parameter_key(parameter: ParameterRef) -> tuple[str, str]:
    """Return one independent definitions-collection/local identity."""

    if not isinstance(parameter, ParameterRef):
        raise TypeError("parameter must be ParameterRef")
    definitions_id = getattr(parameter, "definitions_id", None)
    identifier = getattr(parameter, "id", None)
    if not isinstance(definitions_id, str) or not definitions_id or not isinstance(identifier, str) or not identifier:
        raise TypeError("ParameterRef has no canonical SCNSim parameter identity")
    return definitions_id, identifier


def _view_declaration(lineage: Mapping[str, object]) -> dict[str, object]:
    """Project a lazy Python View to the request's declarative-only record."""

    ptc = lineage.get("ptc")
    transforms = lineage.get("transforms")
    retain = lineage.get("retain")
    if transforms is None or not isinstance(transforms, Sequence) or isinstance(transforms, (str, bytes)):
        raise CompilerInvariantError("View transform declaration is malformed", stage="request_encode")
    return {
        "type": "network_view",
        "ptc": None if ptc is None else {"selected_ports": list(ptc["selected_ports"])},
        "transforms": [
            {
                "id": item["id"],
                "input_coordinates": list(item["input_coordinates"]),
                "output_coordinates": list(item["output_coordinates"]),
            }
            for item in transforms
        ],
        "retain": None if retain is None else {
            "retained_coordinates": list(retain["retained_coordinates"]),
        },
    }


def _parameter_value_record(parameter: ParameterRef, value: object) -> Mapping[str, object]:
    record = ParameterSet({parameter: value})._record()
    bindings = record["bindings"]
    if len(bindings) != 1:
        raise CompilerInvariantError("parameter value did not encode uniquely", stage="request_encode")
    return bindings[0]["value"]


def _uses_baseline_root(spec: object) -> bool:
    """Return whether a Spec owns baseline-root continuation."""

    if isinstance(spec, (DiagonalRootSpec, HybridizedPoleSpec, TransferZeroSpec)):
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


def _plan_has_affine_binding(value: object) -> bool:
    """Recognize affine expansion without interpreting its unimplemented support."""

    if isinstance(value, Mapping):
        return value.get("kind") == "affine" or any(
            _plan_has_affine_binding(item) for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_plan_has_affine_binding(item) for item in value)
    return False


def _plan_public_coordinates(plan: Mapping[str, object]) -> tuple[str, ...]:
    """Collect selectable opaque compiler coordinates from the sealed snapshot."""

    connectivity = plan.get("connectivity")
    nodes = connectivity.get("node_coordinates") if isinstance(connectivity, Mapping) else None
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        raise CompilerInvariantError("Plan node-coordinate table is malformed", stage="plan_seal")
    coordinates = tuple(
        str(node["compiler_node_id"])
        for node in nodes
        if isinstance(node, Mapping) and node.get("visibility") == "public"
    )
    if not coordinates or len(set(coordinates)) != len(coordinates):
        raise CompilerInvariantError("Plan public coordinate table is malformed", stage="plan_seal")
    return coordinates


def _raw_view_lineage(lineage: Mapping[str, object]) -> bool:
    """Recognize the immutable original View that needs no realization pass."""

    return (
        lineage.get("ptc") is None
        and lineage.get("transforms") == []
        and lineage.get("retain") is None
    )


def _source_unit_identity(
    *,
    scope: str,
    component_path: Sequence[str] = (),
    parameter_id: str,
    field: str,
) -> str:
    """Encode one source-unit authority without flattening path segments."""

    if (
        not isinstance(scope, str)
        or not scope
        or not isinstance(parameter_id, str)
        or not parameter_id
        or not isinstance(field, str)
        or not field
        or any(not isinstance(segment, str) or not segment for segment in component_path)
    ):
        raise CompilerInvariantError("source-unit provenance identity is malformed", stage="request_encode")
    return canonical_json_bytes(
        {
            "scope": scope,
            "component_path": list(component_path),
            "parameter_id": parameter_id,
            "field": field,
        }
    ).decode("utf-8")


def _quantity_selectors(value: object) -> tuple[QuantitySelector, ...]:
    if isinstance(value, QuantitySelector):
        return (value,)
    if isinstance(value, QuantitySum):
        selectors = tuple(
            selector
            for term in value.terms
            for selector in _quantity_selectors(term)
        )
        if selectors:
            return selectors
    raise InvalidOptimizationSpec(
        "objectives require a Direct quantity selector or its QuantitySum",
        stage="spec_validation",
    )


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
        "_plan",
        "_snapshot",
        "_baseline_point",
        "_plan_document",
        "_plan_bytes",
        "_plan_sha256",
        "_binding",
        "_original",
        "_runtime_base",
        "_parameter_lookup",
        "_public_coordinates",
        "_coordinate_lookup",
        "_affine_plan",
        "_source_provenance",
    )

    def __init__(
        self,
        *,
        plan: CircuitPlan,
        workspace: str | PathLike[str],
        versioned: bool = False,
    ) -> None:
        if not isinstance(plan, CircuitPlan):
            raise TypeError("plan must be a CircuitPlan")
        if not isinstance(versioned, bool):
            raise TypeError("versioned must be bool")
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
        snapshot = plan._capture_authoring_snapshot()
        baseline_point = resolve_parameter_point(snapshot)
        self._plan = plan
        self._snapshot = snapshot
        self._baseline_point = baseline_point
        self._plan_document = canonical_plan_snapshot(snapshot)
        self._plan_bytes = canonical_json_bytes(self._plan_document)
        self._plan_sha256 = sha256_hex(self._plan_bytes)
        self._runtime_base = _runtime_identity_base()
        self._parameter_lookup = {
            _parameter_key(parameter): parameter
            for parameter in baseline_point.effective_parameters.values
        }
        self._source_provenance = snapshot.source_provenance
        self._public_coordinates = frozenset(_plan_public_coordinates(self._plan_document))
        lookup: dict[str, str | None] = {}
        for node in self._plan_document["connectivity"]["node_coordinates"]:
            compiler_id = str(node["compiler_node_id"])
            lookup[compiler_id] = compiler_id
            for alias in node["public_aliases"]:
                identifier = alias.get("id")
                if isinstance(identifier, str):
                    if identifier not in lookup:
                        lookup[identifier] = compiler_id
                    elif lookup[identifier] != compiler_id:
                        lookup[identifier] = None
                if alias.get("kind") == "exposed_coordinate":
                    scope = alias.get("scope")
                    if isinstance(scope, list) and all(isinstance(item, str) for item in scope) and isinstance(identifier, str):
                        scoped = canonical_json_bytes({"scope": scope, "id": identifier}).decode("utf-8")
                        previous = lookup.setdefault(scoped, compiler_id)
                        if previous != compiler_id:
                            raise CompilerInvariantError(
                                "typed public coordinate resolves to multiple physical nodes",
                                stage="plan_seal",
                            )
        self._coordinate_lookup = MappingProxyType(lookup)
        self._affine_plan = _plan_has_affine_binding(self._plan_document)
        original = self._original_lineage()
        nodes_by_net = {
            str(node["final_net"]): str(node["compiler_node_id"])
            for node in self._plan_document["connectivity"]["node_coordinates"]
        }
        port_coordinates = {
            nodes_by_net[str(port["net"])]: str(port["id"])
            for port in self._plan_document["connectivity"]["ports"]
        }
        self._original = NetworkViewRef._create(
            self,
            original,
            available_coordinates=tuple(sorted(self._public_coordinates)),
            port_coordinates=port_coordinates,
            coordinate_load_states={coordinate: "raw" for coordinate in port_coordinates},
        )
        self._binding = bind_workspace(
            workspace,
            plan_sha256=self._plan_sha256,
            plan_bytes=self._plan_bytes,
            versioned=versioned,
            commit=lambda: plan._seal_validated(snapshot, seal_token),
        )

    @property
    def original(self) -> NetworkViewRef:
        """The sealed Plan's immutable zero-reduction root View."""

        return self._original

    def _original_lineage(self) -> dict[str, object]:
        return _original_lineage_document(
            self._plan_document, self._plan_sha256, self._runtime_base
        )

    def _derive_view(self, parent: NetworkViewRef, pipeline: ReductionPipeline) -> NetworkViewRef:
        """Apply one immutable dev5 grammar suffix without executing it.

        Candidate-dependent transform weights and B/R/M realization remain a
        preflight responsibility; this Ref records only exact declarations and
        current coordinate identities.
        """

        if pipeline._retained is not None and parent._retained:
            raise ValueError("retain() is terminal and cannot be added to a retained View")
        if pipeline._ptc is not None and (
            parent._lineage["ptc"] is not None or parent._lineage["transforms"]
        ):
            raise ValueError("ptc() must be the first reduction in a View lineage")
        available = list(parent._available_coordinates)
        port_coordinates = dict(parent._port_coordinates)
        load_states = dict(parent._coordinate_load_states)
        ptc = parent._lineage["ptc"]
        if pipeline._ptc is not None:
            port_by_id = {port.id: port for port in self._plan.ports}
            requested: set[str] = set()
            for port in pipeline._ptc:
                if port.plan is not self._plan or port.id not in port_by_id or port_by_id[port.id] is not port:
                    raise ValueError("ptc() PortRef belongs to another Plan")
                if port.role != "nonloading_probe":
                    raise ValueError("ptc() accepts only nonloading_probe Ports")
                if port.id in requested:
                    raise ValueError("ptc() Ports must be unique")
                requested.add(port.id)
            ptc = {
                "type": "ptc",
                "selected_ports": [port.id for port in self._plan.ports if port.id in requested],
            }
            selected_ports = set(ptc["selected_ports"])
            for coordinate, port_id in port_coordinates.items():
                load_states[coordinate] = (
                    "compensated" if port_id in selected_ports else "raw"
                )
        transforms = [dict(value) for value in parent._lineage["transforms"]]

        def resolve_coordinate(value: str | ElectricNodeRef | CoordinateRef) -> str:
            # Derived coordinate IDs are already in the current basis. Every
            # other spelling must resolve through the snapshot's typed/public
            # alias table to one opaque compiler node ID.
            if isinstance(value, str) and value in available:
                return value
            return self._coordinate_id(value)

        for raw_left, raw_right, identifier in pipeline._transforms:
            left, right = resolve_coordinate(raw_left), resolve_coordinate(raw_right)
            if isinstance(raw_left, ElectricNodeRef) and raw_left.plan is not self._plan:
                raise ValueError("transform_pair node belongs to another Plan")
            if isinstance(raw_right, ElectricNodeRef) and raw_right.plan is not self._plan:
                raise ValueError("transform_pair node belongs to another Plan")
            if left == right or left not in available or right not in available:
                raise ValueError("transform_pair() requires two distinct current Public coordinates")
            common, differential = f"{identifier}.common", f"{identifier}.differential"
            if (
                common in available
                or differential in available
                or common in self._coordinate_lookup
                or differential in self._coordinate_lookup
                or common == differential
            ):
                raise ValueError("transform_pair generated coordinate collides with the current basis")
            left_state = load_states.get(left, "not-port")
            right_state = load_states.get(right, "not-port")
            if (
                left_state != right_state
                and left_state != "not-port"
                and right_state != "not-port"
            ):
                raise ValueError("transform_pair Port inputs must share one PTC load state")
            generated_port = (
                identifier
                if left_state == right_state and left_state != "not-port"
                else None
            )
            left_index, right_index = available.index(left), available.index(right)
            insert_at = min(left_index, right_index)
            available = [value for value in available if value not in {left, right}]
            available[insert_at:insert_at] = [common, differential]
            port_coordinates.pop(left, None)
            port_coordinates.pop(right, None)
            load_states.pop(left, None)
            load_states.pop(right, None)
            if generated_port is not None:
                port_coordinates[common] = common
                port_coordinates[differential] = differential
                load_states[common] = left_state
                load_states[differential] = left_state
            else:
                load_states[common] = "not-port"
                load_states[differential] = "not-port"
            transforms.append(
                {
                    "type": "transform_pair",
                    "id": identifier,
                    "input_coordinates": [left, right],
                    "output_coordinates": [common, differential],
                }
            )
        retained: tuple[str, ...] = parent._retained
        retain_record = parent._lineage["retain"]
        if pipeline._retained is not None:
            resolved = tuple(resolve_coordinate(value) for value in pipeline._retained)
            if any(isinstance(value, ElectricNodeRef) and value.plan is not self._plan for value in pipeline._retained):
                raise ValueError("retained node belongs to another Plan")
            if len(set(resolved)) != len(resolved) or not resolved or any(value not in available for value in resolved):
                raise ValueError("retain() accepts only unique current Public coordinates")
            retained = resolved
            # Candidate-dependent B/R/M matrices are resolved by the Julia
            # preflight from this exact declarative lineage.
            retain_record = {
                "type": "retain",
                "retained_coordinates": list(resolved),
                "eliminated_coordinates": [value for value in available if value not in resolved],
                "output_coordinate_order": list(resolved),
            }
        terminal = list(retained) if retained else [port.id for port in self._plan.ports]
        # A transform without retain() changes the compiled physical basis but
        # not the raw public Direct boundary: logical Plan Ports remain the
        # terminal channels in their declaration order.
        port_realizable = (
            bool(terminal)
            if not retained
            else bool(terminal) and all(value in port_coordinates for value in terminal)
        )
        record: dict[str, object] = {
            "type": "network_view_lineage",
            "original": dict(parent._lineage["original"]),
            "ptc": ptc,
            "transforms": transforms,
            "retain": retain_record,
            "terminal_coordinates": terminal,
            "port_realizable": port_realizable,
        }
        record["lineage_sha256"] = sha256_hex(record)
        return NetworkViewRef._create(
            self,
            record,
            retained,
            available_coordinates=tuple(available),
            port_coordinates=port_coordinates,
            coordinate_load_states=load_states,
        )

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

    @overload
    def solve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec,
        *,
        parameters: ParameterSpace,
    ) -> ParameterSweepResult: ...

    def solve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
    ) -> DirectSolveResult | HBBatchResult | ParameterSweepResult:
        """Execute the selected Direct response or one shared-basis HB batch."""

        self._require_ref(ref)
        operation = "solve_hb" if isinstance(spec, HBSolveSpec) else "solve_direct"
        prepared = self._prepare_analysis(operation, ref, spec, parameters)
        return self._execute(prepared, bound_spec=spec)

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
        spec: HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> DirectQuantityResult: ...

    @overload
    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec,
        *,
        parameters: ParameterSpace,
    ) -> ParameterSweepResult: ...

    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: DiagonalRootSpec
        | HybridizedPoleSpec
        | TransferZeroSpec
        | ResidueNormalizedCouplingSpec
        | ResponseElementSpec
        | OperatorSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
    ) -> (
        DiagonalRootResult
        | DirectQuantityResult
        | OperatorResult
        | ParameterSweepResult
    ):
        """Evaluate one typed Direct quantity without an unrelated sweep."""

        self._require_ref(ref)
        prepared = self._prepare_analysis("evaluate_direct", ref, spec, parameters)
        return self._execute(prepared, bound_spec=spec)

    @overload
    def optimize(
        self,
        spec: OptimizationSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> OptimizationResult: ...

    @overload
    def optimize(
        self,
        ref: NetworkViewRef,
        spec: OptimizationSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> OptimizationResult: ...

    def optimize(
        self,
        ref_or_spec: NetworkViewRef | OptimizationSpec,
        spec: OptimizationSpec | None = None,
        *,
        parameters: ParameterSet | None = None,
    ) -> OptimizationResult:
        """Run one pinned Direct CMA-ES request and return its exact winner."""

        if parameters is not None and not isinstance(parameters, ParameterSet):
            raise TypeError(
                "OptimizationSpec parameters must be a ParameterSet or None"
            )
        default_ref, optimization_spec = self._optimization_arguments(ref_or_spec, spec)
        ref, selector_views = self._optimization_views(
            optimization_spec, default_ref=default_ref
        )
        prepared = self._prepare_analysis(
            "optimize_direct",
            ref,
            optimization_spec,
            parameters,
            selector_views=selector_views,
        )
        return self._execute(prepared, bound_spec=optimization_spec)

    @overload
    def resolve(self, ref: NetworkViewRef, spec: DirectSolveSpec, *, parameters: ParameterSet | ParameterSpace | None = None) -> DirectSolveResult | ParameterSweepResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: DiagonalRootSpec, *, parameters: ParameterSet | ParameterSpace | None = None) -> DiagonalRootResult | ParameterSweepResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: OptimizationSpec, *, parameters: ParameterSet | None = None) -> OptimizationResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: HBSolveSpec, *, parameters: ParameterSet | ParameterSpace | None = None) -> HBBatchResult | ParameterSweepResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: OperatorSpec, *, parameters: ParameterSet | ParameterSpace | None = None) -> OperatorResult | ParameterSweepResult: ...

    @overload
    def resolve(
        self,
        ref: NetworkViewRef,
        spec: HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
    ) -> DirectQuantityResult | ParameterSweepResult: ...

    def resolve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec
        | HBSolveSpec
        | DiagonalRootSpec
        | HybridizedPoleSpec
        | TransferZeroSpec
        | ResidueNormalizedCouplingSpec
        | ResponseElementSpec
        | OperatorSpec
        | OptimizationSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
    ) -> (
        DirectSolveResult
        | HBBatchResult
        | DiagonalRootResult
        | DirectQuantityResult
        | OperatorResult
        | OptimizationResult
        | ParameterSweepResult
    ):
        """Verify and load the success for this exact request without retrying."""

        self._require_ref(ref)
        if isinstance(spec, DirectSolveSpec):
            operation = "solve_direct"
        elif isinstance(spec, HBSolveSpec):
            operation = "solve_hb"
        elif isinstance(
            spec,
            (
                DiagonalRootSpec,
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
        )
        with self._binding.reader():
            success = self._binding.resolve_success(prepared.request_sha256)
            evidence_lease = _verified_evidence_lease(self._binding, success)
            return self._decode_success(
                success,
                bound_spec=spec,
                evidence_lease=evidence_lease,
            )

    def explain(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec
        | HBSolveSpec
        | DiagonalRootSpec
        | HybridizedPoleSpec
        | TransferZeroSpec
        | ResidueNormalizedCouplingSpec
        | ResponseElementSpec
        | OperatorSpec
        | OptimizationSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> ExplanationResult:
        """Compile and present request evidence without creating an attempt."""

        self._require_ref(ref)
        if not isinstance(
            spec,
            (
                DirectSolveSpec,
                HBSolveSpec,
                DiagonalRootSpec,
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
                    HybridizedPoleSpec,
                    TransferZeroSpec,
                    ResidueNormalizedCouplingSpec,
                    ResponseElementSpec,
                    OperatorSpec,
                ),
            )
            else "optimize_direct"
        )
        selector_views = None
        if isinstance(spec, OptimizationSpec):
            ref, selector_views = self._optimization_views(spec, default_ref=ref)
        prepared = self._prepare_analysis(
            operation,
            ref,
            spec,
            parameters,
            selector_views=selector_views,
        )
        request = prepared.request()
        compiled = self._preflight(request)
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

    def inventory(self) -> InventoryResult:
        """Inspect this Run's exact workspace leaf without selecting a latest result."""

        with self._binding.reader():
            inventory = self._binding.inventory_document()
        requests = inventory.get("requests")
        maintenance = inventory.get("maintenance")
        if (
            inventory.get("schema") != "scnsim.inventory"
            or inventory.get("schema_version") != 2
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

    def build_report(self, spec: ReportSpec) -> ReportResult:
        """Derive a self-contained report from explicit receipt-backed Results."""

        if (
            not isinstance(spec, ReportSpec)
            or not spec.inputs
            or not all(_is_verified_analysis_result(result) for result in spec.inputs)
        ):
            raise TypeError("build_report() requires ReportSpec")
        rows = "".join(
            "<tr>" + "".join(f"<td>{escape(getattr(result.identity, field))}</td>" for field in ("plan_sha256", "request_sha256", "attempt_sha256", "result_sha256")) + "</tr>"
            for result in spec.inputs
        )
        figures: list[tuple[str, object]] = []
        for result in spec.inputs:
            if isinstance(result, DirectSolveResult):
                figures.append(("Direct response", result.s.plot(magnitude="db", theme=spec.theme)))
            elif isinstance(result, HBBatchResult):
                hb_presentation: dict[str, object] = {"theme": spec.theme}
                if any(outcome.succeeded for outcome in result.cases.values()):
                    hb_presentation["magnitude"] = "db"
                figures.append(("HB batch", result.plot(**hb_presentation)))
            elif isinstance(result, DirectQuantityResult):
                figures.append(("Direct scalar quantity", result.plot(theme=spec.theme)))
            elif isinstance(result, OptimizationResult):
                figures.extend((
                    ("Optimization history", result.plot(kind="history", theme=spec.theme)),
                    ("Optimization term evidence", result.plot(kind="table", theme=spec.theme)),
                ))
            elif isinstance(result, OperatorResult):
                figures.extend(
                    (f"Operator at {point.frequency}", result.plot(frequency=point.frequency, theme=spec.theme))
                    for point in result.points
                )
            elif isinstance(result, ParameterSweepResult):
                figures.append(("Parameter sweep outcomes", result.plot(theme=spec.theme)))
        from ._numeric_presentation import figure_fragments, report_palette

        embedded = figure_fragments(figures)
        body = (
            "<h1>SCNSim report</h1><table><thead><tr><th>Plan</th><th>Request</th><th>Attempt</th><th>Result</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>{embedded}"
        )
        palette, color_scheme = report_palette(spec.theme)
        html = _report_html(
            body, spec.theme, palette=palette, color_scheme=color_scheme
        )
        return _verified_result(ReportResult, html=html, inputs=spec.inputs)

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

    def _coordinate_id(self, value: str | ElectricNodeRef | CoordinateRef) -> str:
        """Resolve a public alias to the snapshot's canonical compiler node."""

        if isinstance(value, ElectricNodeRef) and value.plan is not self._plan:
            raise ValueError("coordinate belongs to another Plan")
        if isinstance(value, CoordinateRef):
            if value.scope.root is not self._plan:
                raise ValueError("coordinate belongs to another Plan")
            key = canonical_json_bytes({"scope": list(value.scope.path()), "id": value.id}).decode("utf-8")
            resolved = self._coordinate_lookup.get(key)
        else:
            resolved = self._coordinate_lookup.get(_coordinate_id(value))
        if resolved is None:
            raise ValueError("coordinate is not a public alias in this Plan")
        return resolved

    def _view_coordinate_id(
        self,
        ref: NetworkViewRef,
        value: str | ElectricNodeRef | CoordinateRef,
    ) -> str:
        if isinstance(value, str) and value in ref._available_coordinates:
            return value
        return self._coordinate_id(value)

    def _validate_direct_request(
        self,
        operation: str,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
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
        if isinstance(spec, (DiagonalRootSpec, HybridizedPoleSpec, TransferZeroSpec, ResidueNormalizedCouplingSpec, ResponseElementSpec, OperatorSpec)):
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
        spec: DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec,
        *,
        residue_branch: bool = False,
    ) -> None:
        """Apply the selected-View contract shared by evaluate and CMA selectors."""

        if isinstance(spec, DiagonalRootSpec):
            coordinate = self._view_coordinate_id(ref, spec.coordinate)
            invalid = (
                coordinate not in ref._retained or len(ref._retained) < 2
                if residue_branch
                else len(ref._retained) != 1 or ref._retained[0] != coordinate
            )
            if invalid:
                raise SCNSimValidationError(
                    "DiagonalRootSpec coordinate is incompatible with the retained View",
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
        current = self._parameter_lookup.get(_parameter_key(parameter))
        if current is None or current._definition_record() != parameter._definition_record():
            raise SCNSimValidationError(
                "parameter is not a compatible consumed definition in this Plan",
                stage="preflight",
            )
        return current

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
            selectors = (
                objective.quantity.terms
                if isinstance(objective.quantity, QuantitySum)
                else (objective.quantity,)
            )
            terms: list[dict[str, object]] = []
            for term_ordinal, selector in enumerate(selectors):
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
                terms.append(declaration)
                leaves.append(
                    BoundOptimizationLeaf.create(
                        objective_id=objective.id,
                        objective_ordinal=objective_ordinal,
                        term_ordinal=term_ordinal,
                        declaration=declaration,
                    )
                )
            expressions.append(
                {"type": "quantity_sum", "terms": terms}
                if isinstance(objective.quantity, QuantitySum)
                else terms[0]
            )
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
        | HybridizedPoleSpec
        | TransferZeroSpec
        | ResidueNormalizedCouplingSpec
        | ResponseElementSpec
        | OperatorSpec
        | OptimizationSpec,
        parameters: ParameterSet | ParameterSpace | None,
        *,
        selector_views: Mapping[int, NetworkViewRef] | None = None,
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
        semantic = dict(self._runtime_base)
        if operation == "solve_direct":
            semantic["algorithm_id"] = "scnsim.direct_response.v1"
        elif operation == "solve_hb":
            semantic["algorithm_id"] = "scnsim.hb_response.josephsoncircuits.v1"
        elif operation == "evaluate_direct":
            semantic["algorithm_id"] = {
                "diagonal_root": "scnsim.diagonal_root.newton32.v1",
                "hybridized_pole": "scnsim.hybridized_pole.newton32.v1",
                "transfer_zero": "scnsim.transfer_zero.newton32.v1",
                "residue_normalized_coupling": "scnsim.residue_normalized_coupling.v1",
                "response_element": "scnsim.response_element.v1",
                "operator": "scnsim.direct_operator.v1",
            }[encoded_spec["type"]]
        elif operation == "optimize_direct":
            semantic["algorithm_id"] = (
                "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v3"
            )
        else:
            raise CompilerInvariantError(
                "operation is outside the runtime", stage="request_encode"
            )
        return PreparedAnalysis.create(
            plan_sha256=self._plan_sha256,
            operation=operation,
            view=_view_declaration(ref._lineage),
            spec=encoded_spec,
            parameter_source=parameter_source,
            runtime_semantic=semantic,
            source_units=source_units,
            bound_optimization=bound_optimization,
        )

    def _preflight(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Run the compiler-only realization boundary without allocating work."""

        return _run_preflight(self._plan_bytes, request)

    def _execute(
        self,
        prepared_analysis: PreparedAnalysis,
        *,
        bound_spec: object | None = None,
    ):
        from ._execution import execute_prepared

        with execute_prepared(
            binding=self._binding,
            plan_document=self._plan_document,
            prepared_analysis=prepared_analysis,
        ) as success:
            evidence_lease = _verified_evidence_lease(self._binding, success)
            return self._decode_success(
                success,
                bound_spec=bound_spec,
                evidence_lease=evidence_lease,
            )

    def _source_units(
        self,
        spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        parameters: ParameterSet,
        *,
        parameter_space: ParameterSet | ParameterSpace | None,
    ) -> list[dict[str, object]]:
        captured = self._source_provenance.get("source_units")
        if not isinstance(captured, Sequence) or isinstance(captured, (str, bytes)):
            raise CompilerInvariantError(
                "snapshot source-unit provenance is missing",
                stage="request_encode",
            )
        evidence: list[dict[str, object]] = []
        identities: set[str] = set()
        for raw in captured:
            if not isinstance(raw, Mapping) or set(raw) != {
                "identity", "source_unit", "canonical_si_unit", "canonical_dimensionality",
            }:
                raise CompilerInvariantError(
                    "snapshot source-unit provenance is malformed",
                    stage="request_encode",
                )
            row = dict(raw)
            identity = row.get("identity")
            if not isinstance(identity, str) or not identity or identity in identities:
                raise CompilerInvariantError(
                    "snapshot source-unit provenance identities are invalid",
                    stage="request_encode",
                )
            identities.add(identity)
            evidence.append(row)

        def add(identity: str, value: object, si_unit: str) -> None:
            if identity in identities:
                raise CompilerInvariantError(
                    "source-unit provenance has duplicate parameter authority",
                    stage="request_encode",
                    evidence={"identity": identity},
                )
            identities.add(identity)
            magnitude = np.asarray(value.magnitude)
            probe = (
                value
                if magnitude.ndim == 0
                else units.registry.Quantity(float(magnitude.flat[0]), value.units)
            )
            source_magnitude = getattr(probe, "magnitude", None)
            encoded = (
                complex_quantity_envelope(probe, si_unit=si_unit, registry=units.registry)
                if isinstance(source_magnitude, complex) or getattr(getattr(source_magnitude, "dtype", None), "kind", None) == "c"
                else quantity_envelope(probe, si_unit=si_unit, registry=units.registry)
            )
            evidence.append(
                {
                    "identity": identity,
                    "source_unit": str(value.units),
                    "canonical_si_unit": encoded["si_unit"],
                    "canonical_dimensionality": encoded["dimensionality"],
                }
            )

        for parameter, value in parameters.values.items():
            definitions_id, identifier = _parameter_key(parameter)
            if isinstance(parameter.spec, RLGCParameterSpec):
                if not isinstance(value, RLGC):
                    raise CompilerInvariantError("resolved RLGC parameter is malformed", stage="request_encode")
                units_by_field = {
                    "resistance_per_length": "ohm / meter",
                    "inductance_per_length": "henry / meter",
                    "conductance_per_length": "siemens / meter",
                    "capacitance_per_length": "farad / meter",
                    "extraction_frequency": "hertz",
                }
                for field, quantity in value._source_quantities.items():
                    add(
                        _source_unit_identity(
                            scope="request_parameter_rlgc",
                            component_path=(definitions_id,),
                            parameter_id=identifier,
                            field=field,
                        ),
                        quantity,
                        units_by_field[field],
                    )
            else:
                source_unit = parameters._source_units.get(parameter)
                source_value = value if source_unit is None else value.to(source_unit)
                add(
                    _source_unit_identity(
                        scope="request_parameter",
                        component_path=(definitions_id,),
                        parameter_id=identifier,
                        field="value",
                    ),
                    source_value,
                    parameter.spec.si_unit,
                )
        if isinstance(parameter_space, ParameterSpace) and parameter_space.kind == "grid":
            for axis_index, ((parameter, values), source_units) in enumerate(
                zip(parameter_space.axes, parameter_space._axis_source_units)
            ):
                current = self._compatible_parameter(parameter)
                if isinstance(current.spec, RLGCParameterSpec):
                    units_by_field = {
                        "resistance_per_length": "ohm / meter",
                        "inductance_per_length": "henry / meter",
                        "conductance_per_length": "siemens / meter",
                        "capacitance_per_length": "farad / meter",
                        "extraction_frequency": "hertz",
                    }
                    for value_index, value in enumerate(values):
                        if not isinstance(value, RLGC):
                            raise CompilerInvariantError(
                                "grid RLGC parameter is malformed",
                                stage="request_encode",
                            )
                        for field, quantity in value._source_quantities.items():
                            add(
                                _source_unit_identity(
                                    scope="request_grid_axis_rlgc",
                                    component_path=(current.definitions_id,),
                                    parameter_id=current.id,
                                    field=f"{axis_index}:{value_index}:{field}",
                                ),
                                quantity,
                                units_by_field[field],
                            )
                    continue
                for value_index, (value, source_unit) in enumerate(zip(values, source_units)):
                    source_value = value if source_unit is None else value.to(source_unit)
                    add(
                        _source_unit_identity(
                            scope="request_grid_axis",
                            component_path=(current.definitions_id,),
                            parameter_id=current.id,
                            field=f"{axis_index}:{value_index}",
                        ),
                        source_value,
                        current.spec.si_unit,
                    )
        elif isinstance(parameter_space, ParameterSpace) and parameter_space.kind == "points":
            for point_index, point in enumerate(parameter_space._points):
                for parameter, value in point.values.items():
                    current = self._compatible_parameter(parameter)
                    if isinstance(current.spec, RLGCParameterSpec):
                        if not isinstance(value, RLGC):
                            raise CompilerInvariantError(
                                "listed RLGC parameter is malformed",
                                stage="request_encode",
                            )
                        units_by_field = {
                            "resistance_per_length": "ohm / meter",
                            "inductance_per_length": "henry / meter",
                            "conductance_per_length": "siemens / meter",
                            "capacitance_per_length": "farad / meter",
                            "extraction_frequency": "hertz",
                        }
                        for field, quantity in value._source_quantities.items():
                            add(
                                _source_unit_identity(
                                    scope="request_listed_point_rlgc",
                                    component_path=(current.definitions_id,),
                                    parameter_id=current.id,
                                    field=f"{point_index}:{field}",
                                ),
                                quantity,
                                units_by_field[field],
                            )
                        continue
                    source_unit = point._source_units.get(parameter)
                    if source_unit is None:
                        continue
                    add(
                        _source_unit_identity(
                            scope="request_listed_point",
                            component_path=(current.definitions_id,),
                            parameter_id=current.id,
                            field=f"{point_index}",
                        ),
                        value.to(source_unit),
                        current.spec.si_unit,
                    )
        if isinstance(spec, DirectSolveSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="frequencies", field="value"), spec.frequencies, "hertz")
        elif isinstance(spec, HBSolveSpec):
            add(_source_unit_identity(scope="request_hb", parameter_id="frequencies", field="value"), spec.frequencies, "hertz")
            for axis in spec.pump_axes:
                add(_source_unit_identity(scope="request_hb", parameter_id=axis.id, field="pump_frequency"), axis.frequency, "hertz")
            for case in spec.cases:
                for drive in spec.drives:
                    if drive in case.currents:
                        add(
                            _source_unit_identity(
                                scope="request_hb",
                                parameter_id=case.id,
                                field=f"drive:{drive.id}:coefficient",
                            ),
                            case.currents[drive],
                            "ampere",
                        )
        elif isinstance(spec, DiagonalRootSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="root_hint", field="value"), spec.root_hint, "hertz")
        elif isinstance(spec, HybridizedPoleSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="hybridized_pole", field="anchor"), spec.anchor, "hertz")
        elif isinstance(spec, TransferZeroSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="transfer_zero", field="anchor"), spec.anchor, "hertz")
        elif isinstance(spec, ResponseElementSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="response_element", field="frequency"), spec.frequency, "hertz")
        elif isinstance(spec, OperatorSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="operator", field="frequencies"), spec.frequencies, "hertz")
        elif isinstance(spec, ResidueNormalizedCouplingSpec):
            add(_source_unit_identity(scope="request_spec", parameter_id="residue_normalized_coupling", field="frequency"), spec.frequency, "hertz")
            for branch_name, branch in (("branch_a", spec.branch_a), ("branch_b", spec.branch_b)):
                if isinstance(branch, DiagonalRootSpec):
                    add(_source_unit_identity(scope="request_spec", parameter_id="residue_normalized_coupling", field=f"{branch_name}:root_hint"), branch.root_hint, "hertz")
                else:
                    add(_source_unit_identity(scope="request_spec", parameter_id="residue_normalized_coupling", field=f"{branch_name}:anchor"), branch.anchor, "hertz")
        else:
            for index, variable in enumerate(spec.variables):
                parameter = variable.parameter
                definitions_id, identifier = _parameter_key(parameter)
                for role, bounds in (
                    ("model_default", variable.model_default_bounds),
                    ("consumer_override", variable.consumer_override_bounds),
                ):
                    if bounds is None:
                        continue
                    add(
                        _source_unit_identity(
                            scope="request_optimization_variable",
                            component_path=(definitions_id,),
                            parameter_id=identifier,
                            field=f"{index}:{role}:lower",
                        ),
                        bounds[0],
                        parameter.spec.si_unit,
                    )
                    add(
                        _source_unit_identity(
                            scope="request_optimization_variable",
                            component_path=(definitions_id,),
                            parameter_id=identifier,
                            field=f"{index}:{role}:upper",
                        ),
                        bounds[1],
                        parameter.spec.si_unit,
                    )
            for index, objective in enumerate(spec.objectives):
                parameter_id = f"objective:{index}"
                selectors = objective.quantity.terms if isinstance(objective.quantity, QuantitySum) else (objective.quantity,)
                selector = selectors[0]
                objective_unit = _selector_unit(selector)
                if objective_unit is None:
                    raise InvalidOptimizationSpec(
                        "optimization objective has no scalar quantity unit",
                        stage="spec_validation",
                    )
                add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field="target"), objective.target, objective_unit)
                add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field="weight"), objective.weight, "dimensionless")
                if objective.scale is not None:
                    add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field="scale"), objective.scale, objective_unit)
                for term_index, term in enumerate(selectors):
                    selected_spec = term.spec
                    prefix = f"selector:{term_index}"
                    if isinstance(selected_spec, DiagonalRootSpec):
                        add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:root_hint"), selected_spec.root_hint, "hertz")
                    elif isinstance(selected_spec, HybridizedPoleSpec):
                        add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:anchor"), selected_spec.anchor, "hertz")
                    elif isinstance(selected_spec, TransferZeroSpec):
                        add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:anchor"), selected_spec.anchor, "hertz")
                    elif isinstance(selected_spec, ResponseElementSpec):
                        add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:frequency"), selected_spec.frequency, "hertz")
                    elif isinstance(selected_spec, ResidueNormalizedCouplingSpec):
                        add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:frequency"), selected_spec.frequency, "hertz")
                        for branch_name, branch in (("branch_a", selected_spec.branch_a), ("branch_b", selected_spec.branch_b)):
                            field = "root_hint" if isinstance(branch, DiagonalRootSpec) else "anchor"
                            add(_source_unit_identity(scope="request_optimization_objective", parameter_id=parameter_id, field=f"{prefix}:{branch_name}:{field}"), getattr(branch, field), "hertz")
        return sorted(evidence, key=lambda item: str(item["identity"]))

    def _decode_success(
        self,
        success: VerifiedSuccess,
        *,
        bound_spec: object | None = None,
        evidence_lease: _VerifiedEvidenceLease,
    ):
        from ._result_decode import VerifiedResultDecoder

        decoder = VerifiedResultDecoder(
            plan_sha256=self._plan_sha256,
            plan=self._plan,
            parameter_lookup=self._parameter_lookup,
            coordinate_lookup=self._coordinate_lookup,
        )
        return decoder._decode_success(
            success,
            bound_spec=bound_spec,
            evidence_lease=evidence_lease,
        )


def _original_lineage_document(
    plan: Mapping[str, object],
    plan_sha256: str,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    node_order, _ = _plan_coordinates(plan)
    connectivity = plan.get("connectivity")
    ports = connectivity.get("ports") if isinstance(connectivity, Mapping) else None
    if not isinstance(ports, Sequence) or isinstance(ports, (str, bytes)):
        raise CompilerInvariantError("Plan Port inventory is malformed", stage="plan_seal")
    port_order = [port["id"] for port in ports]
    original = {
        "type": "original",
        "compiled_graph_sha256": sha256_hex(
            {
                "schema": "scnsim.compiled_graph_identity",
                "schema_version": 1,
                "plan_sha256": plan_sha256,
                "julia_source_sha256": runtime["julia_source_sha256"],
            }
        ),
        "coordinate_order": node_order,
        "port_order": port_order,
        "port_realizable": bool(port_order),
    }
    record: dict[str, object] = {
        "type": "network_view_lineage",
        "original": original,
        "ptc": None,
        "transforms": [],
        "retain": None,
        "terminal_coordinates": port_order,
        "port_realizable": bool(port_order),
    }
    record["lineage_sha256"] = sha256_hex(record)
    return record


def _run_preflight(
    plan_bytes: bytes,
    request: Mapping[str, object],
) -> Mapping[str, object]:
    prepared = prepare_runtime()
    with tempfile.TemporaryDirectory(prefix="scnsim-preflight-") as temporary:
        plan_path = Path(temporary) / "plan.json"
        request_path = Path(temporary) / "request.json"
        plan_path.write_bytes(plan_bytes)
        request_path.write_bytes(canonical_json_bytes(request))
        compiled = run_preflight(
            prepared,
            plan_path=plan_path.resolve(),
            request_path=request_path.resolve(),
        )
    if compiled.get("schema") == "scnsim.preflight_failure":
        raise _error_from_record(
            _validated_failure_record(compiled.get("failure"), request["operation"])
        )
    return compiled


def _compiled_schematic_evidence(point: ResolvedPlanPoint) -> Mapping[str, object]:
    """Compile one immutable point without a Run, View, or analysis workspace."""

    if not isinstance(point, ResolvedPlanPoint):
        raise TypeError("_compiled_schematic_evidence() requires ResolvedPlanPoint")
    plan_document = canonical_plan_snapshot(point.snapshot)
    plan_bytes = canonical_json_bytes(plan_document)
    plan_sha = sha256_hex(plan_bytes)
    point_document = canonical_resolved_plan_point(point, plan_sha256=plan_sha)
    point_bytes = canonical_json_bytes(point_document)
    prepared = prepare_runtime()
    with tempfile.TemporaryDirectory(prefix="scnsim-compiler-audit-") as temporary:
        plan_path = Path(temporary) / "plan.json"
        point_path = Path(temporary) / "point.json"
        plan_path.write_bytes(plan_bytes)
        point_path.write_bytes(point_bytes)
        compiled = dict(
            run_compiler_audit(
                prepared,
                plan_path=plan_path.resolve(),
                point_path=point_path.resolve(),
            )
        )
    required = {
        "schema", "schema_version", "plan_sha256", "parameters_sha256",
        "node_order", "matrix_order", "resolved_bindings",
        "expanded_branch_rows", "c_matrix", "k_matrix", "g_matrix", "ports",
    }
    if (
        set(compiled) != required
        or compiled.get("schema") != "scnsim.compiler_audit"
        or compiled.get("schema_version") != 2
        or compiled.get("plan_sha256") != plan_sha
        or compiled.get("parameters_sha256") != point_document["parameters_sha256"]
        or compiled.get("matrix_order") != "canonical_node_id"
        or not isinstance(compiled.get("node_order"), list)
        or not compiled["node_order"]
        or len(set(compiled["node_order"])) != len(compiled["node_order"])
        or any(not isinstance(compiled.get(field), list) for field in ("resolved_bindings", "expanded_branch_rows"))
        or any(not isinstance(compiled.get(field), Mapping) for field in ("c_matrix", "k_matrix", "g_matrix", "ports"))
    ):
        raise BackendProtocolError(
            "compiler-audit evidence does not bind the resolved point",
            stage="compiler_audit",
        )
    runtime = _runtime_identity_base()
    compiled["compiled_graph_sha256"] = sha256_hex({
        "schema": "scnsim.compiled_graph_identity",
        "schema_version": 1,
        "plan_sha256": plan_sha,
        "julia_source_sha256": runtime["julia_source_sha256"],
    })
    compiled["expanded_graph_sha256"] = canonical_expanded_graph_sha256(
        plan_sha256=plan_sha,
        node_order=compiled["node_order"],
        resolved_bindings=compiled["resolved_bindings"],
        expanded_branch_rows=compiled["expanded_branch_rows"],
    )
    return freeze(compiled)


def _runtime_identity_base() -> dict[str, object]:
    package = Path(__file__).resolve().parent

    def manifest(paths: Sequence[Path]) -> str:
        rows = [
            {"path": path.relative_to(package).as_posix(), "mode": "100644", "sha256": sha256(path.read_bytes()).hexdigest()}
            for path in sorted(paths)
        ]
        return sha256_hex({"schema": "scnsim.source_manifest", "schema_version": 1, "files": rows})

    python_files = [*package.glob("*.py"), *package.glob("_schemas/*.json"), package / "_julia" / "runtime.json"]
    julia_files = list((package / "_julia").rglob("*.jl"))
    project = package / "_julia" / "Project.toml"
    julia_manifest = package / "_julia" / "Manifest.toml"
    if not all(path.is_file() for path in (*python_files, *julia_files, project, julia_manifest)):
        raise RuntimeError("SCNSim packaged runtime resources are incomplete")
    runtime = json.loads((package / "_julia" / "runtime.json").read_text(encoding="utf-8"))
    return {
        "python_source_sha256": manifest(python_files),
        "julia_source_sha256": manifest(julia_files),
        "julia_version": runtime["julia_version"],
        "project_sha256": sha256(project.read_bytes()).hexdigest(),
        "manifest_sha256": sha256(julia_manifest.read_bytes()).hexdigest(),
    }
