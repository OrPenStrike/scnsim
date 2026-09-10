"""Plan-bound execution, exact request identity, and typed Result reconstruction.

One captured authoring snapshot and its resolved parameter points feed the
declarative Direct, HB, and optimization request boundary. Workspace receipts
and artifact manifests remain the only authority for reconstructing results.
"""

from __future__ import annotations

import json
import math
import platform
import signal
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import overload

import numpy as np

from . import units
from ._backend import (
    BootstrapReady,
    prepare_runtime,
    run_compiler_audit,
    run_preflight,
    run_terminal,
)
from ._authoring_snapshot import ResolvedPlanPoint, freeze
from ._parameter_resolution import resolve_parameter_point
from ._canonical import (
    _identifier as _canonical_identifier,
    canonical_json_bytes,
    canonical_expanded_graph_sha256,
    canonical_plan_snapshot,
    canonical_parameters_sha256,
    canonical_resolved_plan_point,
    canonical_receipt_document,
    canonical_request_document,
    complex_quantity_envelope,
    complex_quantity_from_envelope,
    float64_from_hex,
    float64_hex,
    quantity_envelope,
    quantity_from_envelope,
    sha256_hex,
    zarr_array_metadata_bytes,
    zarr_artifact_manifest,
)
from ._scaffold import unavailable
from ._physical_values import RLGC, RLGCParameterSpec
from ._workspace import (
    AttemptAllocation,
    VerifiedSuccess,
    _inside,
    _verify_artifact_inventory,
    _verify_generation_artifacts,
    _verify_result_document,
    _verify_v1_lineage,
    _required_extrapolation_rows,
    _plan_coordinates,
    bind_workspace,
    verified_generation_links,
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
    DirectResponseFormationError,
    EliminatedBlockSolveFailure,
    EvidenceIntegrityError,
    HBCaseFailure,
    InvalidDiagonalRootHint,
    InvalidCandidatePhysicalParameter,
    InvalidOptimizationSpec,
    NumericalResolutionUnresolved,
    PortRealizabilityError,
    RootSlopeUnresolved,
    SCNSimError,
    SCNSimValidationError,
    ScaffoldUnavailableError,
    UnsupportedSingularCapacitanceForDiagonalRootV1,
)
from .presentation import _report_html
from .results import (
    BiasState,
    DirectQuantityResult,
    DiagonalRootResult,
    DirectSolveResult,
    ExplanationResult,
    HBBatchResult,
    HBCaseOutcome,
    HBScatteringMatrixResult,
    InventoryResult,
    MatrixFamilyResult,
    MatrixView,
    OptimizationBest,
    OptimizationResult,
    OperatorPointResult,
    OperatorResult,
    ParameterPointIdentity,
    ParameterPointOutcome,
    ParameterSweepResult,
    PumpState,
    ReconciliationEvidence,
    ReportResult,
    ResultIdentity,
    ScatteringMatrixResult,
    TraceResult,
    _is_verified_analysis_result,
    _parameter_sweep_result,
    _point_accessor,
    _point_outcome,
    _verified_result,
)
from .specs import (
    DiagonalRootSpec,
    DirectSolveSpec,
    HBSolveSpec,
    HybridizedPoleSpec,
    OptimizationSpec,
    OperatorSpec,
    QuantitySelector,
    QuantitySum,
    ReportSpec,
    ResidueNormalizedCouplingSpec,
    ResponseElementSpec,
    TransferZeroSpec,
    _selector_unit,
)

_FAILURES: dict[str, type[SCNSimError]] = {
    "backend_protocol": BackendProtocolError,
    "compiler_invariant": CompilerInvariantError,
    "direct_response_formation": DirectResponseFormationError,
    "eliminated_block_solve_failure": EliminatedBlockSolveFailure,
    "evidence_integrity": EvidenceIntegrityError,
    "invalid_diagonal_root_hint": InvalidDiagonalRootHint,
    "invalid_candidate_physical_parameter": InvalidCandidatePhysicalParameter,
    "invalid_optimization_spec": InvalidOptimizationSpec,
    "numerical_resolution_unresolved": NumericalResolutionUnresolved,
    "port_realizability": PortRealizabilityError,
    "root_slope_unresolved": RootSlopeUnresolved,
    "scaffold_unavailable": ScaffoldUnavailableError,
    "unsupported_singular_capacitance_for_diagonal_root_v1": UnsupportedSingularCapacitanceForDiagonalRootV1,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


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
        snapshot = plan._capture_authoring_snapshot()
        baseline_point = resolve_parameter_point(snapshot)
        self._plan = plan._seal()
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
        self._binding = bind_workspace(
            workspace,
            plan_sha256=self._plan_sha256,
            plan_bytes=self._plan_bytes,
            versioned=versioned,
        )
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
        request, source_units, _ = self._materialized_request(operation, ref, spec, parameters)
        return self._execute(request, source_units, bound_spec=spec)

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

    def evaluate(
        self,
        ref: NetworkViewRef,
        spec: DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
    ) -> DiagonalRootResult | DirectQuantityResult | OperatorResult | ParameterSweepResult:
        """Evaluate one typed Direct quantity without an unrelated sweep."""

        self._require_ref(ref)
        request, source_units, _ = self._materialized_request("evaluate_direct", ref, spec, parameters)
        return self._execute(request, source_units, bound_spec=spec)

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
            raise TypeError("OptimizationSpec parameters must be a ParameterSet or None")
        default_ref, optimization_spec = self._optimization_arguments(ref_or_spec, spec)
        ref, selector_views = self._optimization_views(optimization_spec, default_ref=default_ref)
        request, source_units, _ = self._materialized_request(
            "optimize_direct", ref, optimization_spec, parameters,
            selector_views=selector_views,
        )
        return self._execute(request, source_units, bound_spec=optimization_spec)

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
        spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        *,
        parameters: ParameterSet | ParameterSpace | None = None,
    ) -> DirectSolveResult | HBBatchResult | DiagonalRootResult | DirectQuantityResult | OperatorResult | OptimizationResult | ParameterSweepResult:
        """Verify and load the success for this exact request without retrying."""

        self._require_ref(ref)
        if isinstance(spec, DirectSolveSpec):
            operation = "solve_direct"
        elif isinstance(spec, HBSolveSpec):
            operation = "solve_hb"
        elif isinstance(spec, (DiagonalRootSpec, HybridizedPoleSpec, TransferZeroSpec, ResidueNormalizedCouplingSpec, ResponseElementSpec, OperatorSpec)):
            operation = "evaluate_direct"
        elif isinstance(spec, OptimizationSpec):
            if parameters is not None and not isinstance(parameters, ParameterSet):
                raise TypeError("OptimizationSpec parameters must be a ParameterSet or None")
            operation = "optimize_direct"
        else:
            unavailable(f"CircuitRun.resolve({type(spec).__name__})")
        selector_views = None
        if isinstance(spec, OptimizationSpec):
            ref, selector_views = self._optimization_views(spec, default_ref=ref)
        declaration, _ = self._request_declaration(
            operation, ref, spec, parameters, selector_views=selector_views,
        )
        request_sha256 = sha256_hex(canonical_json_bytes(declaration))
        with self._binding.reader():
            success = self._binding.resolve_success(request_sha256)
        return self._decode_success(success, bound_spec=spec)

    def explain(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> ExplanationResult:
        """Compile and present request evidence without creating an attempt."""

        self._require_ref(ref)
        if not isinstance(spec, (DirectSolveSpec, HBSolveSpec, DiagonalRootSpec, HybridizedPoleSpec, TransferZeroSpec, ResidueNormalizedCouplingSpec, ResponseElementSpec, OperatorSpec, OptimizationSpec)):
            unavailable(f"CircuitRun.explain({type(spec).__name__})")
        operation = (
            "solve_hb" if isinstance(spec, HBSolveSpec)
            else "solve_direct" if isinstance(spec, DirectSolveSpec)
            else "evaluate_direct" if isinstance(spec, (DiagonalRootSpec, HybridizedPoleSpec, TransferZeroSpec, ResidueNormalizedCouplingSpec, ResponseElementSpec, OperatorSpec))
            else "optimize_direct"
        )
        selector_views = None
        if isinstance(spec, OptimizationSpec):
            ref, selector_views = self._optimization_views(spec, default_ref=ref)
        request, _, compiled = self._materialized_request(
            operation, ref, spec, parameters, selector_views=selector_views,
        )
        if compiled is None:
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
        if (
            inventory.get("schema") != "scnsim.inventory"
            or inventory.get("schema_version") != 1
            or inventory.get("plan_sha256") != self._plan_sha256
            or not isinstance(requests, list)
            or any(not isinstance(row, Mapping) for row in requests)
        ):
            raise EvidenceIntegrityError("workspace inventory is malformed", stage="inventory")
        return _verified_result(InventoryResult, requests=tuple(dict(row) for row in requests))

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

    def _materialized_request(
        self,
        operation: str,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        parameters: ParameterSet | ParameterSpace | None,
        *,
        selector_views: Mapping[int, NetworkViewRef] | None = None,
    ) -> tuple[dict[str, object], list[dict[str, object]], Mapping[str, object] | None]:
        """Build the closed declarative request without launching Julia."""

        request, source_units = self._request_declaration(
            operation, ref, spec, parameters, selector_views=selector_views,
        )
        return request, source_units, None

    def _request_declaration(
        self,
        operation: str,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        parameters: ParameterSet | ParameterSpace | None,
        *,
        selector_views: Mapping[int, NetworkViewRef] | None = None,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        """Encode a read-only lazy View declaration without invoking Julia."""

        if isinstance(spec, OptimizationSpec) and selector_views is None:
            ref, selector_views = self._optimization_views(spec, default_ref=ref)
        if isinstance(spec, HBSolveSpec):
            if operation != "solve_hb":
                raise CompilerInvariantError("HB Spec has a non-HB operation", stage="request_encode")
            self._validate_hb_request(ref, spec)
        else:
            self._validate_direct_request(
                operation, ref, spec, selector_views=selector_views,
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
                allow_extrapolation=tuple({
                    *resolved.effective_parameters.allow_extrapolation,
                    *spec.allow_extrapolation,
                }),
            )
            resolved = self._complete_parameters(authorized)
            parameter_source = {"kind": "point", "parameters": resolved.parameter_record}
        self._validate_root_parameter_source(spec, parameter_source)
        effective = resolved.effective_parameters
        try:
            encoded_spec = _encode_spec(
                spec,
                effective,
                coordinate_id=lambda value: self._view_coordinate_id(ref, value),
                trace_channel_id=lambda value: self._trace_request_channel(ref, value),
                selector_view=(
                    None if selector_views is None
                    else lambda selector: selector_views[id(selector)]
                ),
                view_declaration=lambda selected: _view_declaration(selected._lineage),
                coordinate_id_for_view=lambda selected, value: self._view_coordinate_id(selected, value),
            )
            source_units = self._source_units(spec, effective, parameter_space=parameters)
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
            semantic["algorithm_id"] = "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v3"
        else:
            raise CompilerInvariantError("operation is outside the runtime", stage="request_encode")
        preliminary = canonical_request_document(
            plan_sha256=self._plan_sha256,
            operation=operation,
            view=_view_declaration(ref._lineage),
            spec=encoded_spec,
            parameter_source=parameter_source,
            runtime_semantic=semantic,
        )
        return preliminary, source_units

    def _preflight(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Run the compiler-only realization boundary without allocating work."""

        return _run_preflight(self._plan_bytes, request)

    def _execute(
        self,
        request: Mapping[str, object],
        source_units: Sequence[Mapping[str, object]],
        *,
        bound_spec: object | None = None,
    ):
        request_bytes = canonical_json_bytes(request)
        request_sha = sha256_hex(request_bytes)
        with self._binding.reader():
            success = self._binding.find_success(request_sha)
            if success is not None:
                return self._decode_success(success, bound_spec=bound_spec)
        prepared = prepare_runtime()
        executable_sha = sha256(prepared.executable.read_bytes()).hexdigest()
        started = _utc_now()
        with self._binding.writer():
            success = self._binding.find_success(request_sha)
            if success is not None:
                return self._decode_success(success, bound_spec=bound_spec)
            request_directory = self._binding.ensure_request(request_sha, request_bytes)
            resume_ledger_sha = self._binding.resume_ledger_sha256(request_sha)
            allocation = self._binding.allocate_attempt(request_sha)
            attempt_sha: str | None = None

            def seal_protocol_failure(
                error: BackendProtocolError,
                *,
                stdout: Sequence[str] = (),
                stderr: Sequence[str] = (),
            ) -> None:
                nonlocal attempt_sha
                if attempt_sha is None:
                    attempt_sha = self._binding.seal_attempt(
                        allocation,
                        _attempt_document(
                            allocation,
                            started=started,
                            executable_sha=executable_sha,
                            state="allocated",
                            resume_ledger_sha=resume_ledger_sha,
                        ),
                    )
                _write_logs(allocation.staging_directory, stdout, (*stderr, str(error)))
                _discard_untrusted_outputs(allocation.staging_directory)
                receipt = _receipt(
                    request=request,
                    plan_document=self._plan_document,
                    request_sha=request_sha,
                    attempt_sha=attempt_sha,
                    outcome="failure",
                    artifacts=[],
                    source_units=source_units,
                    failure=_failure_record(error, request["operation"], request_sha, attempt_sha),
                )
                promote(receipt)

            def promote(receipt: Mapping[str, object]) -> None:
                previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
                try:
                    self._binding.promote_attempt(allocation, receipt)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous)

            def seal_interruption(error: KeyboardInterrupt) -> None:
                nonlocal attempt_sha
                if allocation.final_directory.exists():
                    return
                if attempt_sha is None:
                    attempt_sha = self._binding.seal_attempt(
                        allocation,
                        _attempt_document(
                            allocation,
                            started=started,
                            executable_sha=executable_sha,
                            state="allocated",
                            resume_ledger_sha=resume_ledger_sha,
                        ),
                    )
                _write_logs(allocation.staging_directory, (), ())
                artifacts = verified_generation_links(
                    allocation.staging_directory,
                    request_sha256=request_sha,
                    attempt_sha256=attempt_sha,
                    allow_other_artifacts=True,
                )
                _discard_untrusted_outputs(allocation.staging_directory, keep_ledgers=True)
                receipt = _receipt(
                    request=request,
                    plan_document=self._plan_document,
                    request_sha=request_sha,
                    attempt_sha=attempt_sha,
                    outcome="interrupted",
                    artifacts=artifacts,
                    source_units=source_units,
                    interruption={"kind": "keyboard_interrupt", "termination": getattr(error, "termination", "terminated"), "interrupted_at_utc": _utc_now()},
                )
                promote(receipt)

            def authorize(ready: BootstrapReady) -> str:
                nonlocal attempt_sha
                attempt = _attempt_document(
                    allocation,
                    started=started,
                    executable_sha=executable_sha,
                    state="launched",
                    ready=ready,
                    resume_ledger_sha=resume_ledger_sha,
                )
                attempt_sha = self._binding.seal_attempt(allocation, attempt)
                return attempt_sha

            try:
                terminal = run_terminal(
                    prepared,
                    request_path=(request_directory / "request.json").resolve(),
                    staging_directory=allocation.staging_directory.resolve(),
                    request_sha256=request_sha,
                    attempt_ordinal=allocation.ordinal,
                    authorize=authorize,
                )
            except KeyboardInterrupt as error:
                seal_interruption(error)
                raise
            except BackendProtocolError as error:
                seal_protocol_failure(error)
                raise

            assert attempt_sha is not None
            try:
                _write_logs(allocation.staging_directory, terminal.stdout_log, terminal.stderr_log)
                outcome_path = allocation.staging_directory / "outcome.json"
                if (
                    allocation.staging_directory.is_symlink()
                    or not allocation.staging_directory.is_dir()
                    or outcome_path.is_symlink()
                    or not outcome_path.is_file()
                ):
                    raise BackendProtocolError("outcome.json is not a regular file", stage="outcome")
                outcome_raw = outcome_path.read_bytes()
                outcome = terminal.outcome
                if canonical_json_bytes(outcome) != outcome_raw:
                    raise BackendProtocolError("outcome.json is not canonical", stage="outcome")
                if (
                    outcome.get("runtime_semantic") != request.get("runtime_semantic")
                    or outcome.get("request_sha256") != request_sha
                    or outcome.get("attempt_sha256") != attempt_sha
                    or outcome.get("status") not in {"success", "failure"}
                    or not isinstance(outcome.get("artifacts"), list)
                ):
                    raise BackendProtocolError("outcome envelope does not bind this execution", stage="outcome")
                expected_outcome_fields = {
                    "schema", "schema_version", "request_sha256", "attempt_sha256",
                    "runtime_semantic", "status", "artifacts",
                    "result_sha256" if outcome["status"] == "success" else "failure",
                }
                if set(outcome) != expected_outcome_fields:
                    raise BackendProtocolError("outcome envelope has unsupported fields", stage="outcome")
                _validate_terminal_staging_layout(
                    allocation.staging_directory,
                    success=outcome["status"] == "success",
                )
                artifacts = list(outcome["artifacts"])
                outcome_sha = sha256_hex(outcome_raw)
                if outcome["status"] == "success":
                    _validate_success_staging(
                        allocation.staging_directory, outcome, request, self._plan_document
                    )
                    receipt = _receipt(
                        request=request,
                        plan_document=self._plan_document,
                        request_sha=request_sha,
                        attempt_sha=attempt_sha,
                        outcome="success",
                        artifacts=artifacts,
                        source_units=source_units,
                        outcome_sha=outcome_sha,
                        result_sha=outcome["result_sha256"],
                    )
                    failure = None
                else:
                    failure = _validated_failure_record(outcome.get("failure"), request["operation"])
                    result_path = allocation.staging_directory / "result.json"
                    if result_path.exists() or result_path.is_symlink():
                        raise BackendProtocolError("failure outcome must not publish result.json", stage="outcome")
                    verified_links = verified_generation_links(
                        allocation.staging_directory,
                        request_sha256=request_sha,
                        attempt_sha256=attempt_sha,
                    )
                    if artifacts != verified_links:
                        raise BackendProtocolError(
                            "failure outcome does not exactly bind completed generation ledgers",
                            stage="outcome",
                        )
                    receipt = _receipt(
                        request=request,
                        plan_document=self._plan_document,
                        request_sha=request_sha,
                        attempt_sha=attempt_sha,
                        outcome="failure",
                        artifacts=artifacts,
                        source_units=source_units,
                        outcome_sha=outcome_sha,
                        failure=failure,
                    )
            except BackendProtocolError as error:
                seal_protocol_failure(error, stdout=terminal.stdout_log, stderr=terminal.stderr_log)
                raise
            except KeyboardInterrupt as error:
                seal_interruption(error)
                raise
            except Exception as error:
                protocol = BackendProtocolError(
                    "Julia terminal evidence failed closed validation",
                    stage="outcome",
                    evidence={"error": str(error)},
                )
                seal_protocol_failure(protocol, stdout=terminal.stdout_log, stderr=terminal.stderr_log)
                raise protocol from error
            try:
                promote(receipt)
            except KeyboardInterrupt:
                # SIGINT may be delivered immediately after the atomic rename;
                # the finalized receipt remains authoritative.
                raise
            if failure is None:
                return self._decode_success(
                    self._binding.resolve_success(request_sha), bound_spec=bound_spec
                )
            raise _error_from_record(failure)

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
    ):
        attempt_sha = sha256_hex(canonical_json_bytes(success.attempt))
        result_sha = success.receipt["result_sha256"]
        identity = _verified_result(
            ResultIdentity,
            plan_sha256=self._plan_sha256,
            request_sha256=str(success.attempt["request_sha256"]),
            attempt_sha256=attempt_sha,
            result_sha256=result_sha,
        )
        result = success.result
        kind = result["result_kind"]
        if kind == "parameter_sweep":
            return self._decode_parameter_sweep(
                identity,
                result,
                success.request,
                success.directory,
                bound_spec=bound_spec,
            )
        return self._decode_result(
            identity,
            result,
            success.request,
            success.directory,
        )

    def _decode_result(
        self,
        identity: ResultIdentity | ParameterPointIdentity,
        result: Mapping[str, object],
        request: Mapping[str, object],
        directory: Path,
    ):
        """Decode one already-verified ordinary scientific payload."""

        kind = result["result_kind"]
        if kind == "hb_batch":
            return self._decode_hb_batch(identity, result, request, directory)
        if kind == "direct_response":
            arrays = result["array_catalog"]
            frequency = _read_zarr(directory, arrays["frequencies"], complex_values=False)
            s = _read_zarr(directory, arrays["s"], complex_values=True)
            y = _read_zarr(directory, arrays["y"], complex_values=True)
            z = _read_zarr(directory, arrays["z"], complex_values=True)
            _validate_direct_values(
                frequency,
                s,
                y,
                z,
                expected_frequency=_direct_request_frequencies(request),
                stage="result_decode",
            )
            frequencies = units.registry.Quantity(frequency, "hertz")
            coordinates = tuple(arrays["s"]["coordinate_ids"])
            expected_shape = (frequency.size, len(coordinates), len(coordinates))
            if (
                not coordinates
                or len(set(coordinates)) != len(coordinates)
                or any(not isinstance(coordinate, str) or not coordinate for coordinate in coordinates)
                or any(values.shape != expected_shape for values in (s, y, z))
                or tuple(arrays["y"].get("coordinate_ids", ())) != coordinates
                or tuple(arrays["z"].get("coordinate_ids", ())) != coordinates
            ):
                raise EvidenceIntegrityError("Direct arrays disagree with the selected N-port basis", stage="result_decode")
            channels = tuple((coordinate, ()) for coordinate in coordinates)
            loads = {item["port_id"]: item["state"] for item in arrays["s"]["probe_load_state"]}

            def view(values: np.ndarray, unit: str) -> MatrixView:
                return _verified_result(
                    MatrixView,
                    matrix=units.registry.Quantity(values, unit),
                    frequencies=frequencies,
                    coordinates=coordinates,
                    input_channels=channels,
                    output_channels=channels,
                    probe_loads=loads,
                )

            trace_spec = request.get("spec")
            declared_traces = trace_spec.get("traces") if isinstance(trace_spec, Mapping) else None
            if not isinstance(declared_traces, list):
                raise EvidenceIntegrityError("Direct request trace declarations are malformed", stage="result_decode")
            trace_results: dict[str, TraceResult] = {}
            for trace in declared_traces:
                if not isinstance(trace, Mapping):
                    raise EvidenceIntegrityError("Direct trace declaration is malformed", stage="result_decode")
                identifier = trace.get("id")
                input_coordinate = trace.get("input_port")
                output_coordinate = trace.get("output_port")
                input_mode = trace.get("input_mode")
                output_mode = trace.get("output_mode")
                if (
                    not isinstance(identifier, str)
                    or not isinstance(input_coordinate, str)
                    or not isinstance(output_coordinate, str)
                    or input_mode != []
                    or output_mode != []
                    or identifier in trace_results
                    or input_coordinate not in coordinates
                    or output_coordinate not in coordinates
                ):
                    raise EvidenceIntegrityError("Direct trace does not bind the selected S basis", stage="result_decode")
                trace_results[identifier] = _verified_result(
                    TraceResult,
                    frequencies=frequencies,
                    value=units.registry.Quantity(
                        s[:, coordinates.index(output_coordinate), coordinates.index(input_coordinate)],
                        "dimensionless",
                    ),
                    _parent_identity=identity,
                    _presentation={
                        "id": identifier,
                        "family": "S",
                        "input_channel": {"coordinate": input_coordinate, "mode": []},
                        "output_channel": {"coordinate": output_coordinate, "mode": []},
                        "view": request.get("view"),
                        "ref_lineage": result.get("ref_lineage"),
                    },
                )

            return _verified_result(
                DirectSolveResult,
                identity=identity,
                frequencies=frequencies,
                s=_verified_result(
                    ScatteringMatrixResult,
                    view=view(s, "dimensionless"),
                    _parent_identity=identity,
                    _presentation={
                        "family": "S",
                        "view": request.get("view"),
                        "ref_lineage": result.get("ref_lineage"),
                    },
                ),
                y=_verified_result(
                    MatrixFamilyResult,
                    view=view(y, "siemens"),
                    _parent_identity=identity,
                    _presentation={
                        "family": "Y",
                        "view": request.get("view"),
                        "ref_lineage": result.get("ref_lineage"),
                    },
                ),
                z=_verified_result(
                    MatrixFamilyResult,
                    view=view(z, "ohm"),
                    _parent_identity=identity,
                    _presentation={
                        "family": "Z",
                        "view": request.get("view"),
                        "ref_lineage": result.get("ref_lineage"),
                    },
                ),
                traces=trace_results,
            )
        if kind == "diagonal_root":
            scalars = result["scalar_catalog"]
            return _verified_result(
                DiagonalRootResult,
                identity=identity,
                root=complex_quantity_from_envelope(scalars["root"], registry=units.registry),
                frequency=quantity_from_envelope(scalars["frequency"], registry=units.registry),
                linewidth=quantity_from_envelope(scalars["linewidth"], registry=units.registry),
                slope=complex_quantity_from_envelope(scalars["slope"], registry=units.registry),
                value=None,
                magnitude=None,
                real=None,
                imag=None,
                _presentation={
                    "view": request.get("view"),
                        "ref_lineage": result.get("ref_lineage"),
                    "spec": request.get("spec"),
                },
            )
        if kind == "hybridized_pole":
            scalars = result["scalar_catalog"]
            return _verified_result(
                DirectQuantityResult,
                identity=identity,
                root=complex_quantity_from_envelope(scalars["root"], registry=units.registry),
                frequency=quantity_from_envelope(scalars["frequency"], registry=units.registry),
                linewidth=quantity_from_envelope(scalars["linewidth"], registry=units.registry),
                slope=complex_quantity_from_envelope(scalars["slope"], registry=units.registry),
                _presentation={
                    "view": request.get("view"),
                    "ref_lineage": result.get("ref_lineage"),
                    "spec": request.get("spec"),
                },
            )
        if kind == "transfer_zero":
            scalars = result["scalar_catalog"]
            return _verified_result(
                DirectQuantityResult,
                identity=identity,
                zero=complex_quantity_from_envelope(scalars["zero"], registry=units.registry),
                frequency=quantity_from_envelope(scalars["frequency"], registry=units.registry),
                numerator_slope=complex_quantity_from_envelope(scalars["numerator_slope"], registry=units.registry),
                denominator=complex_quantity_from_envelope(scalars["denominator"], registry=units.registry),
                _presentation={
                    "view": request.get("view"),
                    "ref_lineage": result.get("ref_lineage"),
                    "spec": request.get("spec"),
                },
            )
        if kind == "residue_normalized_coupling":
            scalars = result["scalar_catalog"]
            return _verified_result(
                DirectQuantityResult,
                identity=identity,
                coupling=complex_quantity_from_envelope(scalars["coupling"], registry=units.registry),
                magnitude=quantity_from_envelope(scalars["magnitude"], registry=units.registry),
                branch_a_residue=complex_quantity_from_envelope(scalars["branch_a_residue"], registry=units.registry),
                branch_b_residue=complex_quantity_from_envelope(scalars["branch_b_residue"], registry=units.registry),
                _presentation={
                    "view": request.get("view"),
                    "ref_lineage": result.get("ref_lineage"),
                    "spec": request.get("spec"),
                },
            )
        if kind == "response_element":
            scalars = result["scalar_catalog"]
            return _verified_result(
                DirectQuantityResult,
                identity=identity,
                family=scalars["family"],
                value=complex_quantity_from_envelope(scalars["value"], registry=units.registry),
                magnitude=quantity_from_envelope(scalars["magnitude"], registry=units.registry),
                real=quantity_from_envelope(scalars["real"], registry=units.registry),
                imag=quantity_from_envelope(scalars["imag"], registry=units.registry),
                _presentation={
                    "view": request.get("view"),
                    "ref_lineage": result.get("ref_lineage"),
                    "spec": request.get("spec"),
                },
            )
        if kind == "operator":
            arrays = result["array_catalog"]
            frequency = _read_zarr(directory, arrays["frequencies"], complex_values=False)
            matrix = _read_zarr(directory, arrays["operator"], complex_values=True)
            coordinates = tuple(arrays["operator"].get("coordinate_ids", ()))
            expected = _operator_request_frequencies(request)
            if (
                frequency.shape != expected.shape
                or not np.array_equal(frequency.view(np.uint64), expected.view(np.uint64))
                or matrix.shape != (frequency.size, len(coordinates), len(coordinates))
                or not coordinates
                or len(set(coordinates)) != len(coordinates)
                or any(not isinstance(value, str) or not value for value in coordinates)
                or not np.all(np.isfinite(matrix))
            ):
                raise EvidenceIntegrityError("operator artifacts disagree with the request basis", stage="result_decode")
            frequencies = units.registry.Quantity(frequency, "hertz")
            points = tuple(
                _verified_result(
                    OperatorPointResult,
                    frequency=units.registry.Quantity(float(value), "hertz"),
                    matrix=units.registry.Quantity(matrix[index], "siemens / second"),
                    coordinates=coordinates,
                )
                for index, value in enumerate(frequency)
            )
            return _verified_result(OperatorResult, identity=identity, points=points)
        if kind == "optimization":
            best = result["best"]
            parameters = self._decode_parameter_set(best["parameters"])
            ledger = tuple(_read_json_artifact(directory, artifact) for artifact in result["ledger_artifacts"])
            return _verified_result(
                OptimizationResult,
                identity=identity,
                best=_verified_result(
                    OptimizationBest,
                    parameters=parameters,
                    cost=float64_from_hex(best["cost_f64"]),
                ),
                ledger=ledger,
            )
        raise EvidenceIntegrityError("verified Result kind is outside the runtime", stage="result_decode", evidence={"result_kind": kind})

    def _decode_parameter_sweep(
        self,
        identity: ResultIdentity,
        result: Mapping[str, object],
        request: Mapping[str, object],
        directory: Path,
        *,
        bound_spec: object | None,
    ) -> ParameterSweepResult:
        """Expose verified point metadata lazily and defer scientific payload I/O."""

        del bound_spec  # The canonical request, not a live Spec, owns selector identity.
        source = request.get("parameter_source")
        if not isinstance(source, Mapping) or source.get("kind") not in {"grid", "points"}:
            raise EvidenceIntegrityError(
                "parameter sweep has no ordered parameter source",
                stage="result_decode",
            )
        manifest_link = result.get("manifest")
        if not isinstance(manifest_link, Mapping):
            raise EvidenceIntegrityError(
                "parameter sweep manifest link is malformed",
                stage="result_decode",
            )
        manifest = _read_canonical_artifact_json(
            directory,
            manifest_link.get("path"),
            manifest_link.get("sha256"),
            stage="result_decode",
        )
        rows = manifest.get("files")
        if not isinstance(rows, list):
            raise EvidenceIntegrityError(
                "parameter sweep manifest file catalog is malformed",
                stage="result_decode",
            )
        file_hashes = {
            f"artifacts/parameter_points/{row['path']}": row["sha256"]
            for row in rows
            if isinstance(row, Mapping)
            and isinstance(row.get("path"), str)
            and isinstance(row.get("sha256"), str)
        }
        if len(file_hashes) != len(rows):
            raise EvidenceIntegrityError(
                "parameter sweep manifest file identities are malformed",
                stage="result_decode",
            )

        raw_chunks = result.get("chunks")
        if not isinstance(raw_chunks, list):
            raise EvidenceIntegrityError(
                "parameter sweep chunk catalog is malformed",
                stage="result_decode",
            )
        chunks = tuple(raw_chunks)
        chunk_cache: dict[int, Mapping[str, object]] = {}
        outcome_cache: dict[int, ParameterPointOutcome] = {}

        def chunk_for(ordinal: int) -> Mapping[str, object]:
            chunk_ordinal = ordinal // 64
            if chunk_ordinal < 0 or chunk_ordinal >= len(chunks):
                raise IndexError("parameter point index is out of range")
            cached = chunk_cache.get(chunk_ordinal)
            if cached is not None:
                return cached
            link = chunks[chunk_ordinal]
            if not isinstance(link, Mapping):
                raise EvidenceIntegrityError(
                    "parameter sweep chunk link is malformed",
                    stage="result_decode",
                )
            path = link.get("path")
            expected_sha = link.get("sha256")
            if file_hashes.get(path) != expected_sha:
                raise EvidenceIntegrityError(
                    "parameter sweep chunk is not bound by its manifest",
                    stage="result_decode",
                )
            chunk = _read_canonical_artifact_json(
                directory, path, expected_sha, stage="result_decode"
            )
            if (
                chunk.get("chunk_ordinal") != chunk_ordinal
                or chunk.get("first_point") != link.get("first_point")
                or not isinstance(chunk.get("points"), list)
                or len(chunk["points"]) != link.get("point_count")
            ):
                raise EvidenceIntegrityError(
                    "parameter sweep chunk identity is malformed",
                    stage="result_decode",
                )
            chunk_cache[chunk_ordinal] = chunk
            return chunk

        def load_point(ordinal: int) -> ParameterPointOutcome:
            cached = outcome_cache.get(ordinal)
            if cached is not None:
                return cached
            chunk = chunk_for(ordinal)
            offset = ordinal - int(chunk["first_point"])
            points = chunk["points"]
            if offset < 0 or offset >= len(points):
                raise EvidenceIntegrityError(
                    "parameter sweep chunk does not contain its declared point",
                    stage="result_decode",
                )
            point = points[offset]
            if not isinstance(point, Mapping) or point.get("ordinal") != ordinal:
                raise EvidenceIntegrityError(
                    "parameter sweep point identity is malformed",
                    stage="result_decode",
                )
            raw_source_index = point.get("source_index")
            source_index = (
                tuple(raw_source_index)
                if isinstance(raw_source_index, list)
                else raw_source_index
            )
            parameters_record = point.get("parameters")
            if not isinstance(parameters_record, Mapping):
                raise EvidenceIntegrityError(
                    "parameter sweep point parameters are malformed",
                    stage="result_decode",
                )
            parameters = self._decode_parameter_set(parameters_record)
            parameters_sha256 = point.get("parameters_sha256")
            if parameters_sha256 != canonical_parameters_sha256(parameters_record):
                raise EvidenceIntegrityError(
                    "parameter sweep point parameter identity is malformed",
                    stage="result_decode",
                )
            point_identity = _verified_result(
                ParameterPointIdentity,
                batch=identity,
                source_index=source_index,
                parameters_sha256=parameters_sha256,
            )
            if point.get("status") == "failure":
                failure_record = point.get("failure")
                if not isinstance(failure_record, Mapping):
                    raise EvidenceIntegrityError(
                        "parameter sweep point failure is malformed",
                        stage="result_decode",
                    )
                outcome = _point_outcome(
                    parameters=parameters,
                    source_index=source_index,
                    identity=point_identity,
                    result=None,
                    failure=_error_from_record(failure_record),
                )
            elif point.get("status") == "success":
                payload_path = point.get("payload_path")
                payload_sha = file_hashes.get(payload_path)
                if not isinstance(payload_path, str) or payload_sha is None:
                    raise EvidenceIntegrityError(
                        "parameter sweep point payload is not bound by its manifest",
                        stage="result_decode",
                    )
                decoded: dict[str, object] = {}

                def load_result() -> object:
                    existing = decoded.get("result")
                    if existing is not None:
                        return existing
                    payload = _read_canonical_artifact_json(
                        directory,
                        payload_path,
                        payload_sha,
                        stage="result_decode",
                    )
                    if (
                        payload.get("schema") != "scnsim.parameter_point_payload"
                        or payload.get("schema_version") != 2
                    ):
                        raise EvidenceIntegrityError(
                            "parameter sweep point payload is malformed",
                            stage="result_decode",
                        )
                    value = self._decode_result(
                        point_identity,
                        payload,
                        request,
                        directory,
                    )
                    decoded["result"] = value
                    return value

                outcome = _point_outcome(
                    parameters=parameters,
                    source_index=source_index,
                    identity=point_identity,
                    result=load_result,
                    failure=None,
                )
            else:
                raise EvidenceIntegrityError(
                    "parameter sweep point status is malformed",
                    stage="result_decode",
                )
            outcome_cache[ordinal] = outcome
            return outcome

        kind = str(source["kind"])
        shape = tuple(source["shape"]) if kind == "grid" else ()
        axis_parameters = (
            tuple(self._decode_parameter_ref(axis["parameter"]) for axis in source["axes"])
            if kind == "grid"
            else ()
        )
        count = result.get("point_count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise EvidenceIntegrityError(
                "parameter sweep point count is malformed",
                stage="result_decode",
            )
        points = _point_accessor(
            load_point,
            count,
            kind,
            shape,
            axis_parameters,
        )

        request_spec = request.get("spec")
        if not isinstance(request_spec, Mapping):
            raise EvidenceIntegrityError(
                "parameter sweep Spec is malformed",
                stage="result_decode",
            )
        selector_kind = {
            "diagonal_root": "diagonal_root_projection",
            "hybridized_pole": "hybridized_pole_projection",
            "transfer_zero": "transfer_zero_projection",
            "residue_normalized_coupling": "residue_coupling_projection",
            "response_element": "response_element_projection",
        }.get(request_spec.get("type"))
        projections = {
            "diagonal_root": ("frequency", "linewidth"),
            "hybridized_pole": ("frequency", "linewidth"),
            "transfer_zero": ("frequency",),
            "residue_normalized_coupling": ("magnitude",),
            "response_element": ("magnitude", "real", "imag"),
        }.get(request_spec.get("type"), ())
        allowed = (
            tuple(
                canonical_json_bytes(
                    {
                        "type": selector_kind,
                        "spec": request_spec,
                        "projection": projection,
                    }
                )
                for projection in projections
            )
            if selector_kind is not None
            else ()
        )
        derived_coordinates = {
            coordinate
            for transform in request.get("view", {}).get("transforms", ())
            if isinstance(transform, Mapping)
            for coordinate in transform.get("output_coordinates", ())
            if isinstance(coordinate, str)
        }

        def selector_encoder(value: object) -> bytes:
            if not isinstance(value, QuantitySelector):
                raise TypeError("quantity must be a QuantitySelector")

            def coordinate_id(coordinate: object) -> str:
                if isinstance(coordinate, str) and coordinate in derived_coordinates:
                    return coordinate
                return self._coordinate_id(coordinate)  # type: ignore[arg-type]

            return canonical_json_bytes(
                _encode_scalar_expression(value, coordinate_id=coordinate_id)
            )

        return _parameter_sweep_result(
            identity=identity,
            points=points,
            selector_encoder=selector_encoder,
            allowed_selectors=allowed,
        )

    def _decode_hb_batch(
        self,
        identity: ResultIdentity | ParameterPointIdentity,
        result: Mapping[str, object],
        request: Mapping[str, object],
        directory: Path,
    ) -> HBBatchResult:
        """Reconstruct one fully verified ordered HB case batch."""

        raw_cases = result.get("cases")
        topology_evidence = result.get("topology_evidence")
        request_spec = request.get("spec")
        declared = request_spec.get("cases") if isinstance(request_spec, Mapping) else None
        if not isinstance(raw_cases, list) or not isinstance(declared, list) or len(raw_cases) != len(declared) or not isinstance(topology_evidence, Mapping):
            raise EvidenceIntegrityError("HB batch cases disagree with the request", stage="result_decode")
        outcomes: dict[str, HBCaseOutcome] = {}
        for ordinal, (raw, declaration) in enumerate(zip(raw_cases, declared), 1):
            if not isinstance(raw, Mapping) or not isinstance(declaration, Mapping):
                raise EvidenceIntegrityError("HB case outcome is malformed", stage="result_decode")
            case_id = declaration.get("id")
            if raw.get("case_ordinal") != ordinal or raw.get("case_id") != case_id or not isinstance(case_id, str) or case_id in outcomes:
                raise EvidenceIntegrityError("HB case ordering or identity is malformed", stage="result_decode")
            effective_sources = _decode_hb_effective_sources(raw.get("effective_sources"))
            status = raw.get("status")
            if status == "failure":
                failure = raw.get("failure")
                if (
                    not isinstance(failure, Mapping)
                    or set(failure) != {"kind", "stage", "message", "evidence_sha256"}
                    or failure.get("kind") != "hb_case_failure"
                    or failure.get("stage") not in {"operating_point", "linearization", "response_formation"}
                    or not isinstance(failure.get("message"), str)
                    or not failure["message"]
                    or not _is_sha256_text(failure.get("evidence_sha256"))
                ):
                    raise EvidenceIntegrityError("HB case failure is malformed", stage="result_decode")
                outcome_failure = HBCaseFailure(
                    failure["message"],
                    stage=failure["stage"],
                    evidence={"evidence_sha256": failure["evidence_sha256"]},
                )
                outcomes[case_id] = _verified_result(
                    HBCaseOutcome,
                    id=case_id,
                    failure=outcome_failure,
                    effective_sources=effective_sources,
                    operating_point_closure=None,
                    bias_state=None,
                    pump_state=None,
                    s=None,
                    y=None,
                    z=None,
                    traces=None,
                    states=None,
                    state_node_map=None,
                )
                continue
            if status != "success":
                raise EvidenceIntegrityError("HB case status is malformed", stage="result_decode")
            artifacts = raw.get("artifacts")
            trace_artifacts = raw.get("traces")
            reconciliation = raw.get("reconciliation")
            state_node_map = raw.get("state_node_map")
            operating_point_closure = raw.get("operating_point_closure")
            if (
                not isinstance(artifacts, Mapping)
                or set(artifacts) != {"s", "y", "z", "backend_native_s", "backend_native_z", "states", "effective_source_vectors"}
                or not isinstance(trace_artifacts, list)
                or not isinstance(reconciliation, Mapping)
                or not isinstance(operating_point_closure, Mapping)
                or not isinstance(state_node_map, list)
                or not state_node_map
            ):
                raise EvidenceIntegrityError("successful HB case evidence is malformed", stage="result_decode")

            frequency = _direct_request_frequencies(request)
            frequencies = units.registry.Quantity(frequency, "hertz")
            decoded_arrays = {
                name: _read_zarr(directory, artifacts[name], complex_values=True)
                for name in ("s", "y", "z", "backend_native_s", "backend_native_z", "states", "effective_source_vectors")
            }
            if any(not np.all(np.isfinite(values)) for values in decoded_arrays.values()):
                raise EvidenceIntegrityError("HB artifacts contain non-finite values", stage="result_decode")

            def matrix_view(name: str, unit: str) -> MatrixView:
                artifact = artifacts[name]
                if not isinstance(artifact, Mapping):
                    raise EvidenceIntegrityError("HB matrix catalog is malformed", stage="result_decode")
                output_channels = _decode_hb_channel_axis(artifact, index=1, kind="output_channel")
                input_channels = _decode_hb_channel_axis(artifact, index=2, kind="input_channel")
                values = decoded_arrays[name]
                if values.shape != (frequency.size, len(output_channels), len(input_channels)):
                    raise EvidenceIntegrityError("HB matrix shape disagrees with its channel axes", stage="result_decode")
                coordinates = tuple(artifact.get("coordinate_ids", ()))
                if (
                    not coordinates
                    or any(not isinstance(item, str) or not item for item in coordinates)
                    or len(set(coordinates)) != len(coordinates)
                ):
                    raise EvidenceIntegrityError("HB matrix coordinate identity is malformed", stage="result_decode")
                loads = artifact.get("probe_load_state")
                if not isinstance(loads, list):
                    raise EvidenceIntegrityError("HB probe-load evidence is malformed", stage="result_decode")
                probe_loads: dict[str, str] = {}
                for item in loads:
                    if not isinstance(item, Mapping) or set(item) != {"port_id", "state"} or item.get("state") not in {"raw", "compensated"}:
                        raise EvidenceIntegrityError("HB probe-load evidence is malformed", stage="result_decode")
                    port_id = item.get("port_id")
                    if not isinstance(port_id, str) or not port_id or port_id in probe_loads:
                        raise EvidenceIntegrityError("HB probe-load identity is malformed", stage="result_decode")
                    probe_loads[port_id] = item["state"]
                return _verified_result(
                    MatrixView,
                    matrix=units.registry.Quantity(values, unit),
                    frequencies=frequencies,
                    coordinates=coordinates,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    probe_loads=probe_loads,
                )

            selected_s = matrix_view("s", "dimensionless")
            selected_y = matrix_view("y", "siemens")
            selected_z = matrix_view("z", "ohm")
            native_s = matrix_view("backend_native_s", "dimensionless")
            # Native Z is durable evidence even though the public S surface owns
            # only the native scattering view.
            matrix_view("backend_native_z", "ohm")
            recon = _decode_hb_reconciliation(reconciliation)
            _verify_hb_reconciliation_projection(
                reconciliation,
                selected=np.asarray(selected_s.matrix.magnitude),
                native=np.asarray(native_s.matrix.magnitude),
            )
            traces: dict[str, TraceResult] = {}
            trace_declarations = request_spec.get("traces") if isinstance(request_spec, Mapping) else None
            if not isinstance(trace_declarations, list) or len(trace_artifacts) != len(trace_declarations):
                raise EvidenceIntegrityError("HB trace catalog disagrees with its request", stage="result_decode")
            selected_matrix = np.asarray(selected_s.matrix.magnitude)
            for artifact, declaration in zip(trace_artifacts, trace_declarations):
                if not isinstance(artifact, Mapping):
                    raise EvidenceIntegrityError("HB trace catalog is malformed", stage="result_decode")
                identifier = artifact.get("id")
                values = _read_zarr(directory, artifact, complex_values=True)
                if (
                    not isinstance(declaration, Mapping)
                    or not isinstance(identifier, str)
                    or not identifier
                    or identifier != declaration.get("id")
                    or identifier in traces
                    or values.shape != (frequency.size,)
                    or not np.all(np.isfinite(values))
                ):
                    raise EvidenceIntegrityError("HB trace artifact is malformed", stage="result_decode")
                input_channel = (declaration.get("input_port"), tuple(declaration.get("input_mode", ())))
                output_channel = (declaration.get("output_port"), tuple(declaration.get("output_mode", ())))
                try:
                    input_index = selected_s.input_channels.index(input_channel)
                    output_index = selected_s.output_channels.index(output_channel)
                except ValueError as error:
                    raise EvidenceIntegrityError(
                        "HB trace declaration is absent from the selected S basis",
                        stage="result_decode",
                    ) from error
                projected = selected_matrix[:, output_index, input_index]
                values_bits = np.ascontiguousarray(values).view(np.uint64)
                projected_bits = np.ascontiguousarray(projected).view(np.uint64)
                if not np.array_equal(values_bits, projected_bits):
                    raise EvidenceIntegrityError(
                        "HB trace artifact is not the bit-exact declared projection of selected S",
                        stage="result_decode",
                    )
                traces[identifier] = _verified_result(
                    TraceResult,
                    frequencies=frequencies,
                    value=units.registry.Quantity(values, "dimensionless"),
                    _parent_identity=identity,
                    _presentation={
                        "id": identifier,
                        "family": "S",
                        "case_id": case_id,
                        "input_channel": {"coordinate": input_channel[0], "mode": list(input_channel[1])},
                        "output_channel": {"coordinate": output_channel[0], "mode": list(output_channel[1])},
                        "view": request.get("view"),
                        "ref_lineage": result.get("ref_lineage"),
                    },
                )
            states = decoded_arrays["states"]
            if states.ndim != 2 or states.shape[1] != len(state_node_map):
                raise EvidenceIntegrityError("HB state evidence disagrees with state_node_map", stage="result_decode")
            source_modes = _decode_hb_mode_axis(artifacts["effective_source_vectors"], kind="pump_mode")
            source_vectors = decoded_arrays["effective_source_vectors"]
            if source_vectors.ndim != 2 or source_vectors.shape[0] != len(source_modes):
                raise EvidenceIntegrityError("HB effective-source vectors disagree with their mode axis", stage="result_decode")
            active_rows = np.any(source_vectors != 0.0, axis=1)
            derived_bias = any(active and not any(mode) for active, mode in zip(active_rows, source_modes))
            derived_pump = any(active and any(mode) for active, mode in zip(active_rows, source_modes))
            if raw.get("bias_state") != ("on" if derived_bias else "off") or raw.get("pump_state") != ("on" if derived_pump else "off"):
                raise EvidenceIntegrityError("HB BiasState/PumpState disagrees with effective source vectors", stage="result_decode")
            outcomes[case_id] = _verified_result(
                HBCaseOutcome,
                id=case_id,
                failure=None,
                effective_sources=effective_sources,
                operating_point_closure=operating_point_closure,
                bias_state=BiasState(raw["bias_state"]),
                pump_state=PumpState(raw["pump_state"]),
                s=_verified_result(
                    HBScatteringMatrixResult,
                    view=selected_s,
                    backend_native=native_s,
                    reconciliation=recon,
                    _parent_identity=identity,
                    _presentation={
                        "family": "S",
                        "case_id": case_id,
                        "view": request.get("view"),
                        "ref_lineage": result.get("ref_lineage"),
                    },
                ),
                y=_verified_result(
                    MatrixFamilyResult,
                    view=selected_y,
                    _parent_identity=identity,
                    _presentation={
                        "family": "Y",
                        "case_id": case_id,
                        "view": request.get("view"),
                        "ref_lineage": result.get("ref_lineage"),
                    },
                ),
                z=_verified_result(
                    MatrixFamilyResult,
                    view=selected_z,
                    _parent_identity=identity,
                    _presentation={
                        "family": "Z",
                        "case_id": case_id,
                        "view": request.get("view"),
                        "ref_lineage": result.get("ref_lineage"),
                    },
                ),
                traces=traces,
                states=units.registry.Quantity(states, "weber"),
                state_node_map=tuple(state_node_map),
            )
        return _verified_result(
            HBBatchResult,
            identity=identity,
            cases=outcomes,
            topology_evidence=topology_evidence,
        )

    def _decode_parameter_ref(self, record: object) -> ParameterRef:
        if not isinstance(record, Mapping) or set(record) != {
            "definitions_id", "parameter_id"
        }:
            raise EvidenceIntegrityError(
                "parameter identity is malformed",
                stage="result_decode",
            )
        parameter = self._parameter_lookup.get(
            (record["definitions_id"], record["parameter_id"])
        )
        if parameter is None:
            raise EvidenceIntegrityError(
                "parameter is absent from sealed Plan",
                stage="result_decode",
            )
        return parameter

    def _decode_parameter_value(
        self,
        parameter: ParameterRef,
        record: object,
    ) -> object:
        if not isinstance(record, Mapping):
            raise EvidenceIntegrityError(
                "parameter value is malformed",
                stage="result_decode",
            )
        if not isinstance(parameter.spec, RLGCParameterSpec):
            return quantity_from_envelope(record, registry=units.registry)
        if record.get("type") != "rlgc":
            raise EvidenceIntegrityError(
                "RLGC parameter value is malformed",
                stage="result_decode",
            )

        def matrix(name: str, unit: str) -> object:
            value = record.get(name)
            if not isinstance(value, Mapping):
                raise EvidenceIntegrityError(
                    "RLGC parameter matrix is malformed",
                    stage="result_decode",
                )
            shape = value.get("shape")
            values = value.get("values_f64")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or not all(isinstance(item, int) and not isinstance(item, bool) for item in shape)
                or not isinstance(values, list)
            ):
                raise EvidenceIntegrityError(
                    "RLGC parameter matrix is malformed",
                    stage="result_decode",
                )
            try:
                decoded = np.asarray(
                    [float64_from_hex(item) for item in values], dtype=np.float64
                ).reshape(tuple(shape))
            except (TypeError, ValueError) as error:
                raise EvidenceIntegrityError(
                    "RLGC parameter matrix is malformed",
                    stage="result_decode",
                ) from error
            return units.registry.Quantity(decoded, unit)

        extraction = record.get("extraction_frequency")
        source = record.get("source")
        if not isinstance(source, Mapping):
            raise EvidenceIntegrityError(
                "RLGC parameter source is malformed",
                stage="result_decode",
            )
        value = RLGC._from_source(
            conductors=tuple(record.get("conductors", ())),
            reference_conductor=record.get("reference_conductor"),
            resistance_per_length=matrix("resistance_per_length", "ohm / meter"),
            inductance_per_length=matrix("inductance_per_length", "henry / meter"),
            conductance_per_length=matrix("conductance_per_length", "siemens / meter"),
            capacitance_per_length=matrix("capacitance_per_length", "farad / meter"),
            extraction_frequency=(
                None
                if extraction is None
                else quantity_from_envelope(extraction, registry=units.registry)
            ),
            source=source,
        )
        if value._record() != record:
            raise EvidenceIntegrityError(
                "decoded RLGC parameter differs from its verified record",
                stage="result_decode",
            )
        return value

    def _decode_parameter_set(self, record: Mapping[str, object]) -> ParameterSet:
        if set(record) != {"type", "bindings", "allow_extrapolation"}:
            raise EvidenceIntegrityError(
                "parameter set is malformed",
                stage="result_decode",
            )
        bindings = record.get("bindings")
        authorizations = record.get("allow_extrapolation")
        if not isinstance(bindings, list) or not isinstance(authorizations, list):
            raise EvidenceIntegrityError(
                "parameter set bindings are malformed",
                stage="result_decode",
            )
        values: dict[ParameterRef, object] = {}
        for binding in bindings:
            if not isinstance(binding, Mapping) or set(binding) != {"parameter", "value"}:
                raise EvidenceIntegrityError(
                    "parameter binding is malformed",
                    stage="result_decode",
                )
            parameter = self._decode_parameter_ref(binding["parameter"])
            values[parameter] = self._decode_parameter_value(parameter, binding["value"])
        allowed = tuple(self._decode_parameter_ref(value) for value in authorizations)
        parameters = ParameterSet(values, allow_extrapolation=allowed)
        if parameters._record() != record:
            raise EvidenceIntegrityError(
                "decoded parameter set differs from its verified record",
                stage="result_decode",
            )
        return parameters


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


def _frequency_grid(value: object) -> list[dict[str, str]]:
    converted = value.to("hertz")
    magnitudes = np.asarray(converted.magnitude, dtype=np.float64)
    return [quantity_envelope(units.registry.Quantity(float(item), "hertz"), si_unit="hertz", registry=units.registry) for item in magnitudes]


def _is_sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _decode_hb_effective_sources(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise EvidenceIntegrityError("HB effective-source evidence is malformed", stage="result_decode")
    decoded: list[Mapping[str, object]] = []
    identities: set[tuple[str, tuple[int, ...]]] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "drive_id", "mode", "coefficient", "generated_conjugate",
            "backend_binding", "injection_map_sha256",
        }:
            raise EvidenceIntegrityError("HB effective-source evidence is malformed", stage="result_decode")
        drive_id = item.get("drive_id")
        mode = item.get("mode")
        conjugate = item.get("generated_conjugate")
        backend = item.get("backend_binding")
        if (
            not isinstance(drive_id, str)
            or not drive_id
            or not isinstance(mode, list)
            or any(not isinstance(entry, int) or isinstance(entry, bool) for entry in mode)
            or not isinstance(conjugate, Mapping)
            or set(conjugate) != {"mode", "coefficient"}
            or not isinstance(conjugate.get("mode"), list)
            or any(not isinstance(entry, int) or isinstance(entry, bool) for entry in conjugate["mode"])
            or not isinstance(backend, Mapping)
            or set(backend) != {"representative_mode", "representative_index", "coefficient", "coefficient_convention"}
            or not isinstance(backend.get("representative_mode"), list)
            or any(not isinstance(entry, int) or isinstance(entry, bool) for entry in backend["representative_mode"])
            or not isinstance(backend.get("representative_index"), int)
            or isinstance(backend.get("representative_index"), bool)
            or backend["representative_index"] < 0
            or backend.get("coefficient_convention") != "exp_plus_i_m_dot_omega_t_josephsoncircuits_source"
            or not _is_sha256_text(item.get("injection_map_sha256"))
        ):
            raise EvidenceIntegrityError("HB effective-source identity is malformed", stage="result_decode")
        key = (drive_id, tuple(mode))
        if key in identities:
            raise EvidenceIntegrityError("HB effective-source identity is duplicated", stage="result_decode")
        identities.add(key)
        decoded.append(
            {
                "drive_id": drive_id,
                "mode": tuple(mode),
                "coefficient": complex_quantity_from_envelope(item["coefficient"], registry=units.registry),
                "generated_conjugate": {
                    "mode": tuple(conjugate["mode"]),
                    "coefficient": complex_quantity_from_envelope(conjugate["coefficient"], registry=units.registry),
                },
                "backend_binding": {
                    "representative_mode": tuple(backend["representative_mode"]),
                    "representative_index": backend["representative_index"],
                    "coefficient": complex_quantity_from_envelope(backend["coefficient"], registry=units.registry),
                    "coefficient_convention": backend["coefficient_convention"],
                },
                "injection_map_sha256": item["injection_map_sha256"],
            }
        )
    return tuple(decoded)


def _decode_hb_channel_axis(
    artifact: Mapping[str, object],
    *,
    index: int,
    kind: str,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    axes = artifact.get("axes")
    if not isinstance(axes, list) or len(axes) != 3 or not isinstance(axes[index], Mapping):
        raise EvidenceIntegrityError("HB matrix axes are malformed", stage="result_decode")
    axis = axes[index]
    values = axis.get("values")
    if axis.get("kind") != kind or not isinstance(values, list) or not values:
        raise EvidenceIntegrityError("HB matrix channel axis is malformed", stage="result_decode")
    channels: list[tuple[str, tuple[int, ...]]] = []
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {"coordinate", "mode"}:
            raise EvidenceIntegrityError("HB matrix channel label is malformed", stage="result_decode")
        coordinate = value.get("coordinate")
        mode = value.get("mode")
        if (
            not isinstance(coordinate, str)
            or not coordinate
            or not isinstance(mode, list)
            or any(not isinstance(entry, int) or isinstance(entry, bool) for entry in mode)
        ):
            raise EvidenceIntegrityError("HB matrix channel label is malformed", stage="result_decode")
        channels.append((coordinate, tuple(mode)))
    if len(set(channels)) != len(channels):
        raise EvidenceIntegrityError("HB matrix channel labels are duplicated", stage="result_decode")
    return tuple(channels)


def _decode_hb_mode_axis(artifact: object, *, kind: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(artifact, Mapping):
        raise EvidenceIntegrityError("HB mode artifact is malformed", stage="result_decode")
    axes = artifact.get("axes")
    if not isinstance(axes, list) or not axes or not isinstance(axes[0], Mapping):
        raise EvidenceIntegrityError("HB mode axis is malformed", stage="result_decode")
    axis = axes[0]
    values = axis.get("values")
    if (
        axis.get("kind") != kind
        or not isinstance(values, list)
        or (not values and kind != "pump_mode")
    ):
        raise EvidenceIntegrityError("HB mode axis is malformed", stage="result_decode")
    modes: list[tuple[int, ...]] = []
    for value in values:
        if not isinstance(value, list) or any(not isinstance(entry, int) or isinstance(entry, bool) for entry in value):
            raise EvidenceIntegrityError("HB mode-axis tuple is malformed", stage="result_decode")
        modes.append(tuple(value))
    if len(set(modes)) != len(modes):
        raise EvidenceIntegrityError("HB mode axis repeats a tuple", stage="result_decode")
    return tuple(modes)


def _decode_hb_reconciliation(value: Mapping[str, object]) -> ReconciliationEvidence:
    comparable = value.get("comparable")
    expected = {
        "comparable", "reason", "last_comparable_ancestor", "normalization", "evidence_sha256",
        *(("residual_f64", "coordinate_projection") if comparable is True else ()),
    }
    if (
        isinstance(comparable, bool)
        and set(value) == expected
        and value.get("normalization") == "backend_photon_flux_to_scnsim_power_wave"
        and _is_sha256_text(value.get("last_comparable_ancestor"))
        and _is_sha256_text(value.get("evidence_sha256"))
        and ((comparable and value.get("reason") is None) or (not comparable and value.get("reason") in {
            "topology", "load_or_ptc", "reference_plane", "reference_matrix",
            "signed_frequency_grid", "channel_basis", "normalization",
        }))
    ):
        try:
            residual = float64_from_hex(value["residual_f64"]) if comparable else None
        except (TypeError, ValueError) as error:
            raise EvidenceIntegrityError("HB reconciliation residual is malformed", stage="result_decode") from error
        if residual is None or (math.isfinite(residual) and residual >= 0.0):
            return _verified_result(
                ReconciliationEvidence,
                comparable=comparable,
                reason=value.get("reason"),
                last_comparable_ancestor=value["last_comparable_ancestor"],
                residual=residual,
                evidence_sha256=value["evidence_sha256"],
            )
    raise EvidenceIntegrityError("HB reconciliation evidence is malformed", stage="result_decode")


def _verify_hb_reconciliation_projection(
    evidence: Mapping[str, object],
    *,
    selected: np.ndarray,
    native: np.ndarray,
) -> None:
    """Reproduce the comparable HB projection with a fixed scalar order."""

    if evidence.get("comparable") is not True:
        return
    projection = evidence.get("coordinate_projection")
    if not isinstance(projection, Mapping) or set(projection) != {"shape", "values_f64"}:
        raise EvidenceIntegrityError("HB reconciliation coordinate projection is malformed", stage="result_decode")
    shape = projection.get("shape")
    values = projection.get("values_f64")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in shape)
        or not isinstance(values, list)
        or len(values) != shape[0] * shape[1]
    ):
        raise EvidenceIntegrityError("HB reconciliation coordinate projection is malformed", stage="result_decode")
    try:
        q = np.asarray([float64_from_hex(value) for value in values], dtype=np.float64).reshape(tuple(shape))
    except (TypeError, ValueError) as error:
        raise EvidenceIntegrityError("HB reconciliation coordinate projection is malformed", stage="result_decode") from error
    rows, columns = shape
    if (
        selected.ndim != 3
        or native.ndim != 3
        or selected.shape[0] != native.shape[0]
        or selected.shape[1] != selected.shape[2]
        or native.shape[1] != native.shape[2]
        or selected.shape[1] % rows != 0
        or native.shape[1] % columns != 0
        or selected.shape[1] // rows != native.shape[1] // columns
    ):
        raise EvidenceIntegrityError("HB reconciliation matrices disagree with their coordinate projection", stage="result_decode")
    mode_count = selected.shape[1] // rows
    residuals: list[float] = []
    for frequency in range(selected.shape[0]):
        projected = np.empty_like(selected[frequency])
        for output_coordinate in range(rows):
            for output_mode in range(mode_count):
                output = output_coordinate * mode_count + output_mode
                for input_coordinate in range(rows):
                    for input_mode in range(mode_count):
                        input_ = input_coordinate * mode_count + input_mode
                        value = 0.0 + 0.0j
                        for native_output in range(columns):
                            for native_input in range(columns):
                                value += (
                                    q[output_coordinate, native_output]
                                    * native[frequency, native_output * mode_count + output_mode, native_input * mode_count + input_mode]
                                    * q[input_coordinate, native_input]
                                )
                        projected[output, input_] = value
        numerator = max(
            sum(abs(selected[frequency, row, column] - projected[row, column]) for column in range(projected.shape[1]))
            for row in range(projected.shape[0])
        )
        denominator = max(
            sum(abs(selected[frequency, row, column]) + abs(projected[row, column]) for column in range(projected.shape[1]))
            for row in range(projected.shape[0])
        )
        residuals.append(
            0.0
            if denominator == 0.0 and numerator == 0.0
            else math.inf
            if denominator == 0.0
            else numerator / denominator
        )
    residual = max(residuals)
    if not math.isfinite(residual) or float64_hex(residual) != evidence.get("residual_f64"):
        raise EvidenceIntegrityError("HB reconciliation residual does not reproduce selected S from backend-native S", stage="result_decode")


def _frequency_anchor_envelope(value: object) -> dict[str, str]:
    """Preserve an authored complex seed, including an explicit ``x + 0j``."""

    magnitude = getattr(value, "magnitude", None)
    authored_complex = isinstance(magnitude, complex) or getattr(getattr(magnitude, "dtype", None), "kind", None) == "c"
    encoder = complex_quantity_envelope if authored_complex else quantity_envelope
    return encoder(value, si_unit="hertz", registry=units.registry)


def _encode_root(
    spec: DiagonalRootSpec,
    *,
    coordinate_id: Callable[[str | ElectricNodeRef | CoordinateRef], str] = _coordinate_id,
) -> dict[str, object]:
    return {
        "type": "diagonal_root",
        "coordinate": coordinate_id(spec.coordinate),
        "root_hint": quantity_envelope(spec.root_hint, si_unit="hertz", registry=units.registry),
    }


def _encode_scalar_expression(
    value: object,
    *,
    coordinate_id: Callable[[str | ElectricNodeRef | CoordinateRef], str] = _coordinate_id,
    selector_view: Callable[[QuantitySelector], NetworkViewRef] | None = None,
    view_declaration: Callable[[NetworkViewRef], Mapping[str, object]] | None = None,
    coordinate_id_for_view: Callable[[NetworkViewRef, str | ElectricNodeRef | CoordinateRef], str] | None = None,
) -> dict[str, object]:
    if isinstance(value, QuantitySelector):
        if selector_view is None and view_declaration is None and coordinate_id_for_view is None:
            return {
                "type": value.type,
                "spec": _encode_direct_quantity(value.spec, coordinate_id=coordinate_id),
                "projection": value.projection,
            }
        if selector_view is None or view_declaration is None or coordinate_id_for_view is None:
            raise CompilerInvariantError(
                "optimization selector View encoder is incomplete",
                stage="request_encode",
            )
        selected = selector_view(value)
        return {
            "type": value.type,
            "spec": _encode_direct_quantity(
                value.spec,
                coordinate_id=lambda coordinate: coordinate_id_for_view(selected, coordinate),
            ),
            "projection": value.projection,
            "view": dict(view_declaration(selected)),
        }
    if isinstance(value, QuantitySum):
        return {
            "type": "quantity_sum",
            "terms": [
                _encode_scalar_expression(
                    term,
                    coordinate_id=coordinate_id,
                    selector_view=selector_view,
                    view_declaration=view_declaration,
                    coordinate_id_for_view=coordinate_id_for_view,
                )
                for term in value.terms
            ],
        }
    raise InvalidOptimizationSpec(
        "objective quantity must be a supported scalar expression",
        stage="spec_validation",
    )


def _encode_direct_quantity(
    spec: DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec,
    *,
    coordinate_id: Callable[[str | ElectricNodeRef | CoordinateRef], str] = _coordinate_id,
) -> dict[str, object]:
    if isinstance(spec, DiagonalRootSpec):
        return _encode_root(spec, coordinate_id=coordinate_id)
    if isinstance(spec, HybridizedPoleSpec):
        return {
            "type": "hybridized_pole",
            "coordinates": [coordinate_id(value) for value in spec.coordinates],
            "anchor": _frequency_anchor_envelope(spec.anchor),
        }
    if isinstance(spec, TransferZeroSpec):
        return {
            "type": "transfer_zero",
            "anchor": _frequency_anchor_envelope(spec.anchor),
            "family": spec.family,
            "input_coordinate": coordinate_id(spec.input_coordinate),
            "output_coordinate": coordinate_id(spec.output_coordinate),
        }
    if isinstance(spec, ResidueNormalizedCouplingSpec):
        return {
            "type": "residue_normalized_coupling",
            "branch_a": _encode_direct_quantity(spec.branch_a, coordinate_id=coordinate_id),
            "branch_b": _encode_direct_quantity(spec.branch_b, coordinate_id=coordinate_id),
            "frequency": quantity_envelope(spec.frequency, si_unit="hertz", registry=units.registry),
        }
    if isinstance(spec, ResponseElementSpec):
        return {
            "type": "response_element",
            "family": spec.family,
            "input_coordinate": coordinate_id(spec.input_coordinate),
            "output_coordinate": coordinate_id(spec.output_coordinate),
            "frequency": quantity_envelope(spec.frequency, si_unit="hertz", registry=units.registry),
        }
    if isinstance(spec, OperatorSpec):
        return {"type": "operator", "frequencies": _frequency_grid(spec.frequencies)}
    raise TypeError("unsupported Direct quantity Spec")


def _encode_spec(
    spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
    parameters: ParameterSet,
    *,
    coordinate_id: Callable[[str | ElectricNodeRef | CoordinateRef], str] = _coordinate_id,
    trace_channel_id: Callable[[str], str] = lambda value: value,
    selector_view: Callable[[QuantitySelector], NetworkViewRef] | None = None,
    view_declaration: Callable[[NetworkViewRef], Mapping[str, object]] | None = None,
    coordinate_id_for_view: Callable[[NetworkViewRef, str | ElectricNodeRef | CoordinateRef], str] | None = None,
) -> dict[str, object]:
    def trace_record(trace: SParameterTrace) -> dict[str, object]:
        record = dict(trace._canonical_record())
        record["input_port"] = trace_channel_id(trace.input_port)
        record["output_port"] = trace_channel_id(trace.output_port)
        return record

    if isinstance(spec, DirectSolveSpec):
        return {
            "type": "direct_solve",
            "frequencies": _frequency_grid(spec.frequencies),
            "traces": [trace_record(trace) for trace in spec.traces],
        }
    if isinstance(spec, HBSolveSpec):
        return {
            "type": "hb_solve",
            "pump_axes": [
                {
                    "id": axis.id,
                    "frequency": quantity_envelope(axis.frequency, si_unit="hertz", registry=units.registry),
                }
                for axis in spec.pump_axes
            ],
            "drives": [
                {
                    "id": drive.id,
                    "port_id": drive.at.id,
                    "mode": list(drive.mode),
                    "orientation": "port_node_to_reference",
                }
                for drive in spec.drives
            ],
            "frequencies": _frequency_grid(spec.frequencies),
            "cases": [
                {
                    "id": case.id,
                    "currents": [
                        {
                            "drive_id": drive.id,
                            "coefficient": complex_quantity_envelope(
                                case.currents[drive], si_unit="ampere", registry=units.registry,
                            ),
                            "coefficient_convention": "exp_minus_i_m_dot_omega_t_fourier_coefficient",
                        }
                        for drive in spec.drives
                        if drive in case.currents
                    ],
                }
                for case in spec.cases
            ],
            "truncation": {
                "pump_harmonics": list(spec.truncation.pump_harmonics),
                "modulation_harmonics": list(spec.truncation.modulation_harmonics),
                "max_intermodulation_order": spec.truncation.max_intermodulation_order,
                "three_wave_mixing": spec.truncation.three_wave_mixing,
                "four_wave_mixing": spec.truncation.four_wave_mixing,
            },
            "traces": [trace_record(trace) for trace in spec.traces],
            "allow_driven_ptc": spec.allow_driven_ptc,
        }
    if isinstance(spec, (DiagonalRootSpec, HybridizedPoleSpec, TransferZeroSpec, ResidueNormalizedCouplingSpec, ResponseElementSpec, OperatorSpec)):
        return _encode_direct_quantity(spec, coordinate_id=coordinate_id)
    variables: list[dict[str, object]] = []
    for variable in spec.variables:
        parameter = variable.parameter
        if isinstance(parameter.spec, RLGCParameterSpec):
            raise InvalidOptimizationSpec(
                "RLGC parameters cannot be continuous optimization variables",
                stage="spec_validation",
            )
        parameter_unit = parameter.spec.si_unit
        lower, upper = variable.bounds
        low = quantity_envelope(lower, si_unit=parameter_unit, registry=units.registry)
        high = quantity_envelope(upper, si_unit=parameter_unit, registry=units.registry)
        low_value = float(lower.to(parameter_unit).magnitude)
        high_value = float(upper.to(parameter_unit).magnitude)
        baseline_value = float(parameters.values[parameter].to(parameter_unit).magnitude)
        if low_value >= high_value:
            raise InvalidOptimizationSpec(
                "optimization lower bound must be below upper bound",
                stage="spec_validation",
            )
        if not low_value <= baseline_value <= high_value:
            raise InvalidOptimizationSpec(
                "sealed baseline must lie within resolved variable bounds",
                stage="spec_validation",
            )
        if variable.transform == "log" and low_value <= 0.0:
            raise InvalidOptimizationSpec(
                "log optimization bounds must be strictly positive",
                stage="spec_validation",
            )
        default = [quantity_envelope(item, si_unit=parameter_unit, registry=units.registry) for item in variable.model_default_bounds]
        override = None if variable.consumer_override_bounds is None else [quantity_envelope(item, si_unit=parameter_unit, registry=units.registry) for item in variable.consumer_override_bounds]
        variables.append(
            {
                "parameter": parameter._key_record(),
                "model_default_bounds": default,
                "consumer_override_bounds": override,
                "lower": low,
                "upper": high,
                "transform": variable.transform,
            }
        )
    n = len(variables)
    population = spec.optimizer.population_size or (4 + math.floor(3 * math.log(n)))
    generations = (spec.optimizer.max_evaluations - 1) // population
    if generations < 1:
        raise ValueError("CMA-ES budget must fit the baseline and one complete generation")
    unused = spec.optimizer.max_evaluations - (1 + generations * population)
    objectives: list[dict[str, object]] = []
    for objective in spec.objectives:
        selector = objective.quantity.terms[0] if isinstance(objective.quantity, QuantitySum) else objective.quantity
        objective_unit = _selector_unit(selector)
        if objective_unit is None:
            raise InvalidOptimizationSpec(
                "optimization objective has no scalar quantity unit",
                stage="spec_validation",
            )
        target = quantity_envelope(objective.target, si_unit=objective_unit, registry=units.registry)
        target_value = abs(float(objective.target.to(objective_unit).magnitude))
        if objective.scale is None:
            if target_value == 0.0:
                if objective.target.dimensionless:
                    scale_value = units.registry.Quantity(1.0, "dimensionless")
                    scale_source = "dimensionless_unity"
                else:
                    raise ValueError("a dimensional zero target requires an explicit objective scale")
            else:
                scale_value = units.registry.Quantity(target_value, objective_unit)
                scale_source = "relative_target"
        else:
            scale_value = objective.scale
            scale_source = "explicit"
        scale_magnitude = float(scale_value.to(objective_unit).magnitude)
        if not math.isfinite(scale_magnitude) or scale_magnitude <= 0.0:
            raise InvalidOptimizationSpec(
                "objective scale must be finite and strictly positive",
                stage="spec_validation",
            )
        weight = float(objective.weight.to("dimensionless").magnitude)
        if not math.isfinite(weight) or weight <= 0.0:
            raise InvalidOptimizationSpec(
                "objective weight must be finite and strictly positive",
                stage="spec_validation",
            )
        objectives.append(
            {
                "id": objective.id,
                "quantity": _encode_scalar_expression(
                    objective.quantity,
                    coordinate_id=coordinate_id,
                    selector_view=selector_view,
                    view_declaration=view_declaration,
                    coordinate_id_for_view=coordinate_id_for_view,
                ),
                "target": target,
                "weight_f64": float64_hex(weight),
                "resolved_scale": quantity_envelope(scale_value, si_unit=objective_unit, registry=units.registry),
                "scale_source": scale_source,
            }
        )
    return {
        "type": "optimization",
        "variables": variables,
        "objectives": objectives,
        "optimizer": {
            "type": "cma_es",
            "seed": spec.optimizer.seed,
            "max_evaluations": spec.optimizer.max_evaluations,
            "population_size": spec.optimizer.population_size,
            "resolved_population_size": population,
            "initial_sigma_f64": float64_hex(spec.optimizer.initial_sigma),
            "box_transform_id": "cmaes-jl-0.2.6-linquad-unit-box.v1",
            "complete_generations": generations,
            "unused_evaluations": unused,
            "hidden_stops": "disabled",
        },
        "allow_extrapolation": [parameter._key_record() for parameter in spec.allow_extrapolation],
    }


def _attempt_document(
    allocation: AttemptAllocation,
    *,
    started: str,
    executable_sha: str,
    state: str,
    ready: BootstrapReady | None = None,
    resume_ledger_sha: str | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": "scnsim.attempt",
        "schema_version": 1,
        "request_sha256": allocation.request_sha256,
        "ordinal": allocation.ordinal,
        "ordinal_text": allocation.ordinal_text,
        "directory": allocation.attempt_directory_text,
        "staging_directory": allocation.staging_directory_text,
        "attempt_state": state,
        "started_at_utc": started,
        "julia_executable_sha256": executable_sha,
        "os": platform.system(),
        "architecture": platform.machine() or "unknown",
        "cpu": platform.processor() or "unknown",
    }
    if ready is not None:
        document.update({"julia_threads": ready.julia_threads, "blas_threads": ready.blas_threads, "blas_vendor": ready.blas_vendor})
        fftw_threads = getattr(ready, "fftw_threads", None)
        if fftw_threads is not None:
            document["fftw_threads"] = fftw_threads
    if resume_ledger_sha is not None:
        document["resume_ledger_sha256"] = resume_ledger_sha
    return document


def _failure_record(error: SCNSimError, operation: object, request_sha: str, attempt_sha: str) -> dict[str, object]:
    return {
        "category": error.category,
        "kind": error.kind,
        "stage": error.stage,
        "message": str(error),
        "evidence": {
            "type": "failure_evidence",
            "operation": operation if operation in {"solve_direct", "solve_hb", "evaluate_direct", "optimize_direct"} else "backend_protocol",
            "context_kind": "protocol",
            "request_sha256": request_sha,
            "attempt_sha256": attempt_sha,
        },
    }


def _receipt(
    *,
    request: Mapping[str, object],
    plan_document: Mapping[str, object],
    request_sha: str,
    attempt_sha: str,
    outcome: str,
    artifacts: Sequence[object],
    source_units: Sequence[Mapping[str, object]],
    outcome_sha: str | None = None,
    result_sha: object | None = None,
    failure: Mapping[str, object] | None = None,
    interruption: Mapping[str, object] | None = None,
) -> dict[str, object]:
    runtime_sha = sha256_hex(request["runtime_semantic"])
    provenance = sha256_hex({"schema": "scnsim.receipt_provenance", "source_units": list(source_units)})
    evidence: dict[str, object] = {
        "runtime_semantic_sha256": runtime_sha,
        "source_units": list(source_units),
        "extrapolation_evidence": _receipt_extrapolation_evidence(
            request, plan_document, require_authorized=outcome == "success"
        ),
        "provenance_sha256": provenance,
    }
    evidence["evidence_sha256"] = sha256_hex(evidence)
    document: dict[str, object] = {
        "request_sha256": request_sha,
        "attempt_sha256": attempt_sha,
        "outcome": outcome,
        "artifacts": list(artifacts),
        "evidence": evidence,
        "sealed_at_utc": _utc_now(),
    }
    if outcome_sha is not None:
        document["outcome_sha256"] = outcome_sha
    if result_sha is not None:
        document["result_sha256"] = result_sha
    if failure is not None:
        document["failure"] = dict(failure)
    if interruption is not None:
        document["interruption"] = dict(interruption)
    return canonical_receipt_document(document)


def _receipt_extrapolation_evidence(
    request: Mapping[str, object],
    plan_document: Mapping[str, object],
    *,
    require_authorized: bool,
) -> list[dict[str, object]]:
    """Project receipt evidence through the workspace's closed fan-out verifier."""
    if request.get("operation") == "optimize_direct":
        return []
    source = request.get("parameter_source")
    if isinstance(source, Mapping) and source.get("kind") in {"grid", "points"}:
        # Point-local authorization is verified against every chunk entry;
        # the request receipt must not pretend an ordered space is one point.
        return []
    parameters = source.get("parameters") if isinstance(source, Mapping) else None
    if not isinstance(parameters, Mapping):
        raise CompilerInvariantError("receipt request has no ParameterSet", stage="receipt")
    return _required_extrapolation_rows(
        plan_document, parameters, authorization_source="parameter_set",
        require_authorized=require_authorized,
    )


def _require_staging_directory(staging: Path) -> None:
    if staging.parent.is_symlink() or staging.is_symlink() or not staging.is_dir():
        raise EvidenceIntegrityError(
            "attempt staging is not a regular directory",
            stage="workspace",
            evidence={"path": str(staging)},
        )


def _remove_untrusted(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _write_logs(staging: Path, stdout: Sequence[str], stderr: Sequence[str]) -> None:
    _require_staging_directory(staging)
    directory = staging / "logs"
    if directory.exists() or directory.is_symlink():
        _remove_untrusted(directory)
    if not stdout and not stderr:
        return
    directory.mkdir()
    if stdout:
        path = directory / "stdout.log"
        if path.is_symlink():
            path.unlink()
        path.write_text("".join(stdout), encoding="utf-8")
    if stderr:
        path = directory / "stderr.log"
        if path.is_symlink():
            path.unlink()
        path.write_text("".join(stderr), encoding="utf-8")


def _discard_untrusted_outputs(staging: Path, *, keep_ledgers: bool = False) -> None:
    _require_staging_directory(staging)
    outcome = staging / "outcome.json"
    if outcome.exists() or outcome.is_symlink():
        logs = staging / "logs"
        if logs.is_symlink() or logs.exists() and not logs.is_dir():
            _remove_untrusted(logs)
        logs.mkdir(exist_ok=True)
        destination = logs / "untrusted-outcome.json"
        if destination.exists() or destination.is_symlink():
            _remove_untrusted(destination)
        if outcome.is_symlink():
            outcome.unlink()
        elif outcome.is_file():
            shutil.move(outcome, destination)
        else:
            _remove_untrusted(outcome)
    result = staging / "result.json"
    if result.exists() or result.is_symlink():
        _remove_untrusted(result)
    artifacts = staging / "artifacts"
    if artifacts.exists() or artifacts.is_symlink():
        if artifacts.is_symlink():
            artifacts.unlink()
        else:
            generations = artifacts / "generations"
            if keep_ledgers and not generations.is_symlink() and generations.is_dir():
                for child in artifacts.iterdir():
                    if child != generations:
                        _remove_untrusted(child)
                if not any(generations.iterdir()):
                    generations.rmdir()
                    artifacts.rmdir()
            else:
                _remove_untrusted(artifacts)
    allowed = {"attempt.json", "logs"}
    if keep_ledgers and (staging / "artifacts").is_dir():
        allowed.add("artifacts")
    for child in staging.iterdir():
        if child.name not in allowed:
            _remove_untrusted(child)


def _validate_terminal_staging_layout(staging: Path, *, success: bool) -> None:
    _require_staging_directory(staging)
    allowed = {"attempt.json", "logs", "outcome.json", "artifacts"}
    if success:
        allowed.add("result.json")
    unexpected = sorted(child.name for child in staging.iterdir() if child.name not in allowed)
    if unexpected:
        raise BackendProtocolError(
            "terminal staging contains unsupported entries",
            stage="outcome",
            evidence={"entries": unexpected},
        )


def _validate_success_staging(
    staging: Path,
    outcome: Mapping[str, object],
    request: Mapping[str, object],
    plan: Mapping[str, object],
) -> None:
    if outcome.get("runtime_semantic") != request.get("runtime_semantic"):
        raise BackendProtocolError("outcome runtime identity does not match the request", stage="outcome")
    result_path = _inside(staging, "result.json")
    if result_path.is_symlink() or not result_path.is_file() or sha256(result_path.read_bytes()).hexdigest() != outcome.get("result_sha256"):
        raise BackendProtocolError("success outcome does not bind result.json", stage="outcome")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if canonical_json_bytes(result) != result_path.read_bytes():
        raise BackendProtocolError("result.json is not canonical", stage="outcome")
    parameter_source = request.get("parameter_source")
    is_parameter_sweep = isinstance(parameter_source, Mapping) and parameter_source.get("kind") in {"grid", "points"}
    expected_kind = (
        "parameter_sweep" if is_parameter_sweep
        else "direct_response" if request.get("operation") == "solve_direct"
        else "hb_batch" if request.get("operation") == "solve_hb"
        else "optimization" if request.get("operation") == "optimize_direct"
        else request.get("spec", {}).get("type")
        if request.get("operation") == "evaluate_direct" and isinstance(request.get("spec"), Mapping)
        else None
    )
    expected_result_fields = (
        {
            "schema", "schema_version", "result_kind", "request_sha256",
            "attempt_sha256", "parameter_source_sha256", "point_count",
            "chunk_size", "manifest", "chunks",
        }
        if expected_kind == "parameter_sweep"
        else
        {
            "schema", "schema_version", "result_kind", "request_sha256",
            "attempt_sha256", "parameters", "parameters_sha256", "ref_lineage",
            "scalar_catalog", "array_catalog",
        }
        if expected_kind in {
            "direct_response", "diagonal_root", "hybridized_pole", "transfer_zero",
            "residue_normalized_coupling", "response_element", "operator",
        }
        else {
            "schema", "schema_version", "result_kind", "request_sha256",
            "attempt_sha256", "parameters", "parameters_sha256", "ref_lineage",
            "baseline", "best", "completed_generations",
            "unused_evaluations", "ledger_artifacts",
        }
        if expected_kind == "optimization"
        else {
            "schema", "schema_version", "result_kind", "request_sha256",
            "attempt_sha256", "parameters", "parameters_sha256", "ref_lineage",
            "lattice", "truncation", "topology_evidence", "cases",
        }
        if expected_kind == "hb_batch"
        else None
    )
    if (
        expected_result_fields is None
        or set(result) != expected_result_fields
        or result.get("schema") != "scnsim.result"
        or result.get("schema_version") != 2
        or result.get("result_kind") != expected_kind
        or result.get("request_sha256") != outcome.get("request_sha256")
        or result.get("attempt_sha256") != outcome.get("attempt_sha256")
    ):
        raise BackendProtocolError("result envelope does not match its request and operation", stage="outcome")
    _verify_result_document(
        result,
        request,
        str(outcome.get("request_sha256")),
        str(outcome.get("attempt_sha256")),
        plan,
    )
    catalogs: list[Mapping[str, object]] = []
    if expected_kind == "parameter_sweep":
        catalogs.append(result["manifest"])
        expected_links = [dict(result["manifest"])]
    elif expected_kind == "hb_batch":
        expected_links: list[dict[str, object]] = []
        cases = result.get("cases")
        if not isinstance(cases, list):
            raise BackendProtocolError("HB result has no ordered case catalog", stage="outcome")
        for case in cases:
            if not isinstance(case, Mapping):
                raise BackendProtocolError("HB result case catalog is malformed", stage="outcome")
            if case.get("status") == "failure":
                continue
            artifacts = case.get("artifacts")
            traces = case.get("traces")
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not isinstance(artifacts, Mapping) or not isinstance(traces, list):
                raise BackendProtocolError("HB success artifact catalog is malformed", stage="outcome")
            ordered = [artifacts[name] for name in ("s", "y", "z", "backend_native_s", "backend_native_z", "states", "effective_source_vectors")]
            ordered.extend(traces)
            for artifact in ordered:
                if not isinstance(artifact, Mapping):
                    raise BackendProtocolError("HB success artifact catalog is malformed", stage="outcome")
                catalogs.append(artifact)
                expected_links.append(
                    {
                        "case_id": case_id,
                        "id": artifact.get("id"),
                        "path": artifact.get("path"),
                        "sha256": artifact.get("sha256"),
                    }
                )
        if outcome.get("artifacts") != expected_links:
            raise BackendProtocolError("HB outcome artifact inventory does not match result.json", stage="outcome")
        _verify_artifact_inventory(staging, result, {"artifacts": expected_links})
    elif expected_kind != "optimization":
        catalogs.extend(result["array_catalog"].values())
    else:
        catalogs.extend(result["ledger_artifacts"])
    if expected_kind == "parameter_sweep":
        if outcome.get("artifacts") != expected_links:
            raise BackendProtocolError("batch outcome does not bind its manifest", stage="outcome")
        _verify_artifact_inventory(staging, result, {"artifacts": expected_links})
    elif expected_kind != "hb_batch":
        expected_links = [{"id": artifact["id"], "sha256": artifact["sha256"]} for artifact in catalogs]
        if outcome.get("artifacts") != expected_links or len({item["id"] for item in expected_links}) != len(expected_links):
            raise BackendProtocolError("outcome artifact inventory does not match result.json", stage="outcome")
        _verify_artifact_inventory(staging, result, {"artifacts": expected_links})
    if expected_kind == "optimization":
        _verify_generation_artifacts(
            staging,
            expected_links,
            request_sha256=str(outcome["request_sha256"]),
            attempt_sha256=str(outcome["attempt_sha256"]),
        )
    for artifact in catalogs:
        path = _inside(staging, str(artifact["path"]))
        if artifact.get("media_type") == "application/vnd+zarr-v2":
            manifest_path = _inside(staging, str(artifact["file_manifest"]))
            rebuilt = zarr_artifact_manifest(artifact_directory=path, artifact_id=artifact["id"], artifact_path=artifact["path"])
            if manifest_path.is_symlink() or not manifest_path.is_file() or canonical_json_bytes(rebuilt) != manifest_path.read_bytes() or sha256(manifest_path.read_bytes()).hexdigest() != artifact["sha256"]:
                raise EvidenceIntegrityError("Zarr manifest does not match exact artifact bytes", stage="artifact_validation")
            _verify_zarr_catalog_metadata(path, artifact, stage="artifact_validation")
        else:
            if path.is_symlink() or not path.is_file() or path.stat().st_size != artifact["byte_length"] or sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                raise EvidenceIntegrityError("file artifact does not match its catalog", stage="artifact_validation")
    if expected_kind == "direct_response":
        arrays = result["array_catalog"]
        _validate_direct_values(
            _read_zarr(staging, arrays["frequencies"], complex_values=False),
            _read_zarr(staging, arrays["s"], complex_values=True),
            _read_zarr(staging, arrays["y"], complex_values=True),
            _read_zarr(staging, arrays["z"], complex_values=True),
            expected_frequency=_direct_request_frequencies(request),
            stage="artifact_validation",
        )


def _read_canonical_artifact_json(
    attempt: Path,
    path_value: object,
    digest_value: object,
    *,
    stage: str,
) -> Mapping[str, object]:
    if (
        not isinstance(path_value, str)
        or not path_value
        or not _is_sha256_text(digest_value)
    ):
        raise EvidenceIntegrityError(
            "JSON artifact link is malformed",
            stage=stage,
        )
    path = _inside(attempt, path_value)
    if path.is_symlink() or not path.is_file():
        raise EvidenceIntegrityError(
            "JSON artifact is not a regular file",
            stage=stage,
        )
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest_value:
        raise EvidenceIntegrityError(
            "JSON artifact failed exact hash verification",
            stage=stage,
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceIntegrityError(
            "JSON artifact is malformed",
            stage=stage,
        ) from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        raise EvidenceIntegrityError(
            "JSON artifact is not a canonical object",
            stage=stage,
        )
    return value


def _read_zarr(attempt: Path, artifact: Mapping[str, object], *, complex_values: bool) -> np.ndarray:
    root = _inside(attempt, str(artifact["path"]))
    manifest_path = _inside(attempt, str(artifact["file_manifest"]))
    rebuilt = zarr_artifact_manifest(artifact_directory=root, artifact_id=artifact["id"], artifact_path=artifact["path"])
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise EvidenceIntegrityError("stored Zarr manifest is not a regular file", stage="result_decode")
    manifest_bytes = manifest_path.read_bytes()
    if canonical_json_bytes(rebuilt) != manifest_bytes or sha256(manifest_bytes).hexdigest() != artifact["sha256"]:
        raise EvidenceIntegrityError("stored Zarr manifest failed exact verification", stage="result_decode")
    _verify_zarr_catalog_metadata(root, artifact, stage="result_decode")
    import zarr

    group = zarr.open_group(root, mode="r")
    if complex_values:
        return np.asarray(group["real"][:], dtype=np.float64) + 1j * np.asarray(group["imag"][:], dtype=np.float64)
    return np.asarray(group["values"][:], dtype=np.float64)


def _read_json_artifact(attempt: Path, artifact: Mapping[str, object]) -> Mapping[str, object]:
    path = _inside(attempt, str(artifact["path"]))
    if path.is_symlink() or not path.is_file():
        raise EvidenceIntegrityError("optimization ledger is not a regular file", stage="result_decode")
    raw = path.read_bytes()
    if len(raw) != artifact["byte_length"] or sha256(raw).hexdigest() != artifact["sha256"]:
        raise EvidenceIntegrityError("optimization ledger artifact failed verification", stage="result_decode")
    value = json.loads(raw)
    if canonical_json_bytes(value) != raw:
        raise EvidenceIntegrityError("optimization ledger is not canonical", stage="result_decode")
    return value


def _validate_direct_values(
    frequency: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    expected_frequency: np.ndarray,
    stage: str,
) -> None:
    if (
        not np.all(np.isfinite(frequency))
        or np.any(frequency <= 0.0)
        or np.any(np.diff(frequency) <= 0.0)
        or frequency.shape != expected_frequency.shape
        or not np.array_equal(frequency.view(np.uint64), expected_frequency.view(np.uint64))
        or any(not np.all(np.isfinite(values)) for values in (s, y, z))
    ):
        raise EvidenceIntegrityError(
            "Direct artifacts contain non-finite values or an invalid frequency grid",
            stage=stage,
        )


def _direct_request_frequencies(request: Mapping[str, object]) -> np.ndarray:
    spec = request.get("spec")
    values = spec.get("frequencies") if isinstance(spec, Mapping) else None
    if not isinstance(values, list):
        raise EvidenceIntegrityError("Direct request frequency grid is malformed", stage="request_decode")
    try:
        return np.asarray([float64_from_hex(item["si_value_f64"]) for item in values], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceIntegrityError("Direct request frequency grid is malformed", stage="request_decode") from error


def _operator_request_frequencies(request: Mapping[str, object]) -> np.ndarray:
    spec = request.get("spec")
    values = spec.get("frequencies") if isinstance(spec, Mapping) and spec.get("type") == "operator" else None
    if not isinstance(values, list):
        raise EvidenceIntegrityError("Operator request frequency grid is malformed", stage="request_decode")
    try:
        result = np.asarray([float64_from_hex(item["si_value_f64"]) for item in values], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceIntegrityError("Operator request frequency grid is malformed", stage="request_decode") from error
    if not result.size or not np.all(np.isfinite(result)) or np.any(result <= 0.0) or np.any(np.diff(result) <= 0.0):
        raise EvidenceIntegrityError("Operator request frequency grid is malformed", stage="request_decode")
    return result


def _verify_zarr_catalog_metadata(root: Path, artifact: Mapping[str, object], *, stage: str) -> None:
    datasets = artifact.get("datasets")
    if not isinstance(datasets, list):
        raise EvidenceIntegrityError("Zarr catalog datasets are malformed", stage=stage)
    for dataset in datasets:
        if not isinstance(dataset, Mapping) or not isinstance(dataset.get("path"), str):
            raise EvidenceIntegrityError("Zarr catalog dataset is malformed", stage=stage)
        metadata = dataset.get("metadata")
        path = root / str(dataset["path"]) / ".zarray"
        if (
            not isinstance(metadata, Mapping)
            or path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != zarr_array_metadata_bytes(
                shape=metadata.get("shape", ()),
                chunks=metadata.get("chunks", ()),
            )
        ):
            raise EvidenceIntegrityError("Zarr catalog metadata disagrees with exact artifact bytes", stage=stage)


def _validated_failure_record(value: object, operation: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"category", "kind", "stage", "message", "evidence"}:
        raise BackendProtocolError("failure outcome lacks a closed typed failure", stage="outcome")
    kind = value.get("kind")
    cls = _FAILURES.get(kind) if isinstance(kind, str) else None
    evidence = value.get("evidence")
    if (
        cls is None
        or value.get("category") != cls.category
        or not isinstance(value.get("stage"), str)
        or not value["stage"]
        or not isinstance(value.get("message"), str)
        or not value["message"]
        or not isinstance(evidence, Mapping)
        or evidence.get("type") != "failure_evidence"
        or evidence.get("operation") != operation
        or not isinstance(evidence.get("context_kind"), str)
        or not evidence["context_kind"]
    ):
        raise BackendProtocolError("failure outcome discriminator or evidence is invalid", stage="outcome")
    return dict(value)


def _error_from_record(record: Mapping[str, object]) -> SCNSimError:
    cls = _FAILURES.get(str(record.get("kind")))
    if cls is None or record.get("category") != cls.category:
        raise EvidenceIntegrityError(
            "stored failure discriminator is unknown or inconsistent",
            stage="failure_decode",
            evidence={"kind": record.get("kind"), "category": record.get("category")},
        )
    evidence = record.get("evidence")
    return cls(str(record.get("message", "SCNSim backend failure")), stage=str(record.get("stage", "backend")), evidence=evidence if isinstance(evidence, Mapping) else None)
