"""Plan-bound execution, exact request identity, and typed Result reconstruction.

The dev5 candidate extends the accepted Composite path through RLGC, selected
N-port Views, expanded Direct quantities, extrapolation, and inventory. The
dev6 candidate adds the public HB request and dispatch boundary.
"""

from __future__ import annotations

import base64
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
from ._backend import BootstrapReady, prepare_runtime, run_preflight, run_terminal
from ._canonical import (
    _identifier as _canonical_identifier,
    canonical_json_bytes,
    canonical_plan_document,
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
    PumpState,
    ReconciliationEvidence,
    ReportResult,
    ResultIdentity,
    ScatteringMatrixResult,
    TraceResult,
    _is_verified_analysis_result,
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


def _parameter_key(parameter: ParameterRef) -> tuple[tuple[str, ...], str]:
    """Return the canonical key shared by primitive and Composite parameters."""

    reference = parameter._canonical_ref()
    path = reference.get("component_path")
    identifier = reference.get("parameter_id")
    if (
        not isinstance(path, Sequence)
        or isinstance(path, (str, bytes))
        or not path
        or not all(isinstance(part, str) and part for part in path)
        or not isinstance(identifier, str)
        or not identifier
    ):
        raise TypeError("ParameterRef has no canonical SCNSim parameter identity")
    return tuple(path), identifier


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
    """Collect only directly selectable Plan and top-level Composite coordinates."""

    return tuple(sorted(_plan_coordinates(plan)[1]))


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
        if _coordinate_id(node_a) == _coordinate_id(node_b):
            raise ValueError("transform_pair() requires two distinct coordinates")
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
        identifiers = tuple(_coordinate_id(value) for value in coordinates)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("retained coordinates must be unique")
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
        "_plan_document",
        "_plan_bytes",
        "_plan_sha256",
        "_binding",
        "_original",
        "_runtime_base",
        "_parameter_lookup",
        "_public_coordinates",
        "_affine_plan",
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
        self._plan = plan._seal()
        self._plan_document = canonical_plan_document(self._plan._canonical_snapshot())
        self._plan_bytes = canonical_json_bytes(self._plan_document)
        self._plan_sha256 = sha256_hex(self._plan_bytes)
        self._runtime_base = _runtime_identity_base()
        self._parameter_lookup = {
            _parameter_key(parameter): parameter
            for component in self._plan.components
            for parameter in component._parameters.values()
        }
        self._public_coordinates = frozenset(_plan_public_coordinates(self._plan_document))
        self._affine_plan = _plan_has_affine_binding(self._plan_document)
        original = self._original_lineage()
        self._binding = bind_workspace(
            workspace,
            plan_sha256=self._plan_sha256,
            plan_bytes=self._plan_bytes,
            versioned=versioned,
        )
        port_coordinates = {
            str(port["node_id"]): str(port["port_id"])
            for port in self._plan_document["ports"]
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

    def _derive_legacy_view(self, pipeline: ReductionPipeline) -> NetworkViewRef:
        if pipeline._retained is None:
            raise ValueError("a derived View requires terminal retain()")
        if len(pipeline._retained) != 1:
            unavailable("multi-coordinate ReductionPipeline.retain")
        value = pipeline._retained[0]
        coordinate = self._coordinate_id(value)
        if isinstance(value, ElectricNodeRef) and value._plan is not self._plan:
            raise ValueError("retained node belongs to another Plan")
        if coordinate not in self._public_coordinates:
            raise ValueError("retain() accepts only a Public Plan coordinate")
        nodes = list(self._original._lineage["original"]["coordinate_order"])
        eliminated = [node for node in nodes if node != coordinate]
        port_records = list(self._plan_document["ports"])
        ports = [port["port_id"] for port in port_records]
        n, p = len(nodes), len(ports)
        matching_ports = [index for index, port in enumerate(port_records) if port["node_id"] == coordinate]
        port_realizable = len(matching_ports) == 1

        def evidence(
            label: str,
            values: Sequence[Sequence[float]],
            *,
            applicability: str,
        ) -> dict[str, object]:
            rows = len(values)
            columns = len(values[0]) if rows else 0
            if any(len(row) != columns for row in values):
                raise CompilerInvariantError("lineage matrix is ragged", stage="view_lineage")
            return {
                "rows": rows,
                "columns": columns,
                "sha256": sha256_hex(
                    {
                        "schema": "scnsim.lineage_matrix",
                        "schema_version": 1,
                        "label": label,
                        "applicability": applicability,
                        "shape": [rows, columns],
                        "row_major_f64": [float64_hex(value) for row in values for value in row],
                    }
                ),
            }

        matrix_labels = (
            "a", "b", "r", "d", "q", "selected_projector",
            "omitted_projector", "omitted_matched_loads",
        )
        if port_realizable:
            impedances = np.asarray([
                float(quantity_from_envelope(port["reference_impedance"], registry=units.registry).to("ohm").magnitude)
                for port in port_records
            ])
            b_p = np.asarray([
                [1.0 if node == port["node_id"] else 0.0 for port in port_records]
                for node in nodes
            ])
            a = np.zeros((1, p), dtype=np.float64)
            a[0, matching_ports[0]] = 1.0
            r_p = np.diag(impedances)
            d_p = np.diag(np.sqrt(impedances))
            b = b_p @ a.T
            r = a @ r_p @ a.T
            d = np.asarray([[math.sqrt(float(r[0, 0]))]])
            q = np.linalg.solve(d, a @ d_p)
            selected_projector = q.T @ q
            omitted_projector = np.eye(p) - selected_projector
            omitted_matched_loads = (
                np.linalg.solve(d_p, omitted_projector)
                @ omitted_projector
                @ np.linalg.solve(d_p, np.eye(p))
            )
            matrices = {
                "a": evidence("a", a.tolist(), applicability="port_realizable"),
                "b": evidence("b", b.tolist(), applicability="port_realizable"),
                "r": evidence("r", r.tolist(), applicability="port_realizable"),
                "d": evidence("d", d.tolist(), applicability="port_realizable"),
                "q": evidence("q", q.tolist(), applicability="port_realizable"),
                "selected_projector": evidence("selected_projector", selected_projector.tolist(), applicability="port_realizable"),
                "omitted_projector": evidence("omitted_projector", omitted_projector.tolist(), applicability="port_realizable"),
                "omitted_matched_loads": evidence("omitted_matched_loads", omitted_matched_loads.tolist(), applicability="port_realizable"),
            }
            source_boundary = {
                "schema": "scnsim.source_boundary", "schema_version": 1,
                "applicability": "port_realizable", "b": matrices["b"], "r": matrices["r"],
            }
            deembedding = {
                "schema": "scnsim.deembedding", "schema_version": 1,
                "applicability": "port_realizable", "d": matrices["d"], "q": matrices["q"],
            }
        else:
            matrices = {
                label: evidence(label, [], applicability="not_port_realizable")
                for label in matrix_labels
            }
            source_boundary = {
                "schema": "scnsim.source_boundary", "schema_version": 1,
                "applicability": "not_port_realizable",
            }
            deembedding = {
                "schema": "scnsim.deembedding", "schema_version": 1,
                "applicability": "not_port_realizable",
            }

        retain = {
            "type": "retain",
            "retained_coordinates": [coordinate],
            "eliminated_coordinates": eliminated,
            "output_coordinate_order": [coordinate],
            "a_matrix": matrices["a"],
            "b_matrix": matrices["b"],
            "r_matrix": matrices["r"],
            "d_matrix": matrices["d"],
            "q_matrix": matrices["q"],
            "selected_projector": matrices["selected_projector"],
            "omitted_projector": matrices["omitted_projector"],
            "omitted_matched_loads": matrices["omitted_matched_loads"],
            "source_boundary_sha256": sha256_hex(source_boundary),
            "deembedding_evidence_sha256": sha256_hex(deembedding),
        }
        record: dict[str, object] = {
            "type": "network_view_lineage",
            "original": dict(self._original._lineage["original"]),
            "ptc": None,
            "transforms": [],
            "retain": retain,
            "terminal_coordinates": [coordinate],
            "port_realizable": port_realizable,
        }
        record["lineage_sha256"] = sha256_hex(record)
        return NetworkViewRef._create(self, record, (coordinate,))

    def _derive_view(self, parent: NetworkViewRef, pipeline: ReductionPipeline) -> NetworkViewRef:
        """Apply one immutable dev5 grammar suffix without executing it.

        Candidate-dependent transform weights and B/R/M realization remain a
        preflight responsibility; this Ref records only exact declarations and
        current coordinate identities.
        """

        if (
            parent is self._original
            and pipeline._ptc is None
            and not pipeline._transforms
            and pipeline._retained is not None
            and len(pipeline._retained) == 1
            and len(self._plan.ports) == 1
        ):
            return self._derive_legacy_view(pipeline)
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
                if port._plan is not self._plan or port.id not in port_by_id or port_by_id[port.id] is not port:
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
        for raw_left, raw_right, identifier in pipeline._transforms:
            left, right = self._coordinate_id(raw_left), self._coordinate_id(raw_right)
            if isinstance(raw_left, ElectricNodeRef) and raw_left._plan is not self._plan:
                raise ValueError("transform_pair node belongs to another Plan")
            if isinstance(raw_right, ElectricNodeRef) and raw_right._plan is not self._plan:
                raise ValueError("transform_pair node belongs to another Plan")
            if left == right or left not in available or right not in available:
                raise ValueError("transform_pair() requires two distinct current Public coordinates")
            common, differential = f"{identifier}.common", f"{identifier}.differential"
            if common in available or differential in available or common == differential:
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
            resolved = tuple(self._coordinate_id(value) for value in pipeline._retained)
            if any(isinstance(value, ElectricNodeRef) and value._plan is not self._plan for value in pipeline._retained):
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
        parameters: ParameterSet | None = None,
    ) -> DirectSolveResult | HBBatchResult:
        """Execute the selected Direct response or one shared-basis HB batch."""

        self._require_ref(ref)
        operation = "solve_hb" if isinstance(spec, HBSolveSpec) else "solve_direct"
        request, source_units, _ = self._materialized_request(operation, ref, spec, parameters)
        return self._execute(request, source_units)

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
        parameters: ParameterSet | None = None,
    ) -> DiagonalRootResult | DirectQuantityResult | OperatorResult:
        """Evaluate one typed Direct quantity without an unrelated sweep."""

        self._require_ref(ref)
        request, source_units, _ = self._materialized_request("evaluate_direct", ref, spec, parameters)
        return self._execute(request, source_units)

    def optimize(self, ref: NetworkViewRef, spec: OptimizationSpec) -> OptimizationResult:
        """Run one pinned Direct CMA-ES request and return its exact winner."""

        self._require_ref(ref)
        request, source_units, _ = self._materialized_request("optimize_direct", ref, spec, None)
        return self._execute(request, source_units)

    @overload
    def resolve(self, ref: NetworkViewRef, spec: DirectSolveSpec, *, parameters: ParameterSet | None = None) -> DirectSolveResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: DiagonalRootSpec, *, parameters: ParameterSet | None = None) -> DiagonalRootResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: OptimizationSpec) -> OptimizationResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: HBSolveSpec, *, parameters: ParameterSet | None = None) -> HBBatchResult: ...

    @overload
    def resolve(self, ref: NetworkViewRef, spec: OperatorSpec, *, parameters: ParameterSet | None = None) -> OperatorResult: ...

    @overload
    def resolve(
        self,
        ref: NetworkViewRef,
        spec: HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> DirectQuantityResult: ...

    def resolve(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> DirectSolveResult | HBBatchResult | DiagonalRootResult | DirectQuantityResult | OperatorResult | OptimizationResult:
        """Verify and load the success for this exact request without retrying."""

        self._require_ref(ref)
        if isinstance(spec, DirectSolveSpec):
            operation = "solve_direct"
        elif isinstance(spec, HBSolveSpec):
            operation = "solve_hb"
        elif isinstance(spec, (DiagonalRootSpec, HybridizedPoleSpec, TransferZeroSpec, ResidueNormalizedCouplingSpec, ResponseElementSpec, OperatorSpec)):
            operation = "evaluate_direct"
        elif isinstance(spec, OptimizationSpec):
            if parameters is not None:
                raise TypeError("parameters must be omitted for OptimizationSpec")
            operation = "optimize_direct"
        else:
            unavailable(f"CircuitRun.resolve({type(spec).__name__})")
        declaration, _ = self._request_declaration(operation, ref, spec, parameters)
        return self._decode_success(
            self._binding.resolve_matching_success(
                operation=operation,
                spec=declaration["spec"],
                parameters=declaration["parameters"],
                runtime_semantic=declaration["runtime_semantic"],
                lazy_lineage=ref._lineage,
            )
        )

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
        if isinstance(spec, OptimizationSpec) and parameters is not None:
            raise TypeError("parameters must be omitted for OptimizationSpec")
        operation = (
            "solve_hb" if isinstance(spec, HBSolveSpec)
            else "solve_direct" if isinstance(spec, DirectSolveSpec)
            else "evaluate_direct" if isinstance(spec, (DiagonalRootSpec, HybridizedPoleSpec, TransferZeroSpec, ResidueNormalizedCouplingSpec, ResponseElementSpec, OperatorSpec))
            else "optimize_direct"
        )
        request, _, compiled = self._materialized_request(operation, ref, spec, parameters)
        if compiled is None:
            compiled = self._preflight(request)
        return _verified_result(
            ExplanationResult,
            evidence={
                "plan_sha256": self._plan_sha256,
                "request_sha256": sha256_hex(canonical_json_bytes(request)),
                "runtime_semantic": request["runtime_semantic"],
                "ref_lineage": request["ref_lineage"],
                "parameters": request["parameters"],
                "spec": request["spec"],
                "component_hierarchy": self._plan_document["components"],
                "plan_nodes": self._plan_document["nodes"],
                "grounded_endpoints": self._plan_document["grounded_endpoints"],
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
        sections: list[str] = []
        for result in spec.inputs:
            if isinstance(result, DirectSolveResult):
                figure = result.s.show(magnitude="db")
                from io import BytesIO
                import matplotlib.pyplot as plt

                output = BytesIO()
                try:
                    import matplotlib as mpl

                    with mpl.rc_context({"svg.hashsalt": "scnsim.report.v1"}):
                        figure.savefig(
                            output,
                            format="svg",
                            bbox_inches="tight",
                            metadata={"Date": None},
                        )
                finally:
                    plt.close(figure)
                encoded = base64.b64encode(output.getvalue()).decode("ascii")
                sections.append(f'<h2>Direct response</h2><img alt="Direct S magnitude and phase" src="data:image/svg+xml;base64,{encoded}">')
            elif isinstance(result, DiagonalRootResult):
                sections.append(
                    "<h2>Loaded root</h2><table><tbody>"
                    f"<tr><th>Frequency</th><td>{escape(str(result.frequency))}</td></tr>"
                    f"<tr><th>Linewidth</th><td>{escape(str(result.linewidth))}</td></tr>"
                    "</tbody></table>"
                )
            elif isinstance(result, OptimizationResult):
                bindings = "".join(
                    f"<li>{escape(parameter.component_id)}.{escape(parameter.id)} = {escape(str(value))}</li>"
                    for parameter, value in result.best.parameters.values.items()
                )
                sections.append(
                    "<h2>Optimization winner</h2>"
                    f"<p>Cost: {escape(str(result.best.cost))}</p><ul>{bindings}</ul>"
                )
        embedded = "".join(sections)
        html = (
            "<!doctype html><html><head><meta charset=\"utf-8\"><title>SCNSim report</title></head><body>"
            "<h1>SCNSim report</h1><table><thead><tr><th>Plan</th><th>Request</th><th>Attempt</th><th>Result</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>{embedded}</body></html>"
        )
        return _verified_result(ReportResult, html=html, inputs=spec.inputs)

    def _require_ref(self, ref: NetworkViewRef) -> None:
        if not isinstance(ref, NetworkViewRef) or ref._run is not self:
            raise ValueError("NetworkViewRef belongs to another CircuitRun")

    def _coordinate_id(self, value: str | ElectricNodeRef | CoordinateRef) -> str:
        """Resolve a Composite handle to its sealed physical Plan coordinate."""

        if isinstance(value, CoordinateRef):
            return self._plan._resolve_coordinate(value)
        return _coordinate_id(value)

    def _validate_direct_request(
        self,
        operation: str,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
    ) -> None:
        if isinstance(spec, DirectSolveSpec):
            if ref._lineage["port_realizable"] is not True:
                raise PortRealizabilityError(
                    "Direct response requires a port-realizable View",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": operation, "context_kind": "direct_response"},
                )
            channels = tuple(ref._lineage["terminal_coordinates"])
            for trace in spec.traces:
                if trace.input_port not in channels or trace.output_port not in channels:
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
            if self._parameter_lookup.get(key) is not parameter:
                raise InvalidOptimizationSpec(
                    "optimization variable belongs to another Plan",
                    stage="spec_validation",
                )
        for parameter in spec.allow_extrapolation:
            key = _parameter_key(parameter)
            if active.get(key) is not parameter:
                raise InvalidOptimizationSpec(
                    "optimization extrapolation authorization must name an active Plan parameter",
                    stage="spec_validation",
                )
        for objective in spec.objectives:
            for selector in _quantity_selectors(objective.quantity):
                self._validate_direct_quantity_spec(operation, ref, selector.spec)

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
            coordinate = self._coordinate_id(spec.coordinate)
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
            coordinates = tuple(self._coordinate_id(value) for value in spec.coordinates)
            if not ref._retained or coordinates != ref._retained:
                raise SCNSimValidationError(
                    "HybridizedPoleSpec coordinates must equal the retained View order",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": operation, "context_kind": "direct_quantity"},
                )
            return
        if isinstance(spec, (TransferZeroSpec, ResponseElementSpec)):
            channels = set(ref._lineage["terminal_coordinates"])
            if self._coordinate_id(spec.input_coordinate) not in channels or self._coordinate_id(spec.output_coordinate) not in channels:
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
        channels = set(ref._lineage["terminal_coordinates"])
        for trace in spec.traces:
            if trace.input_port not in channels or trace.output_port not in channels:
                raise PortRealizabilityError(
                    "HB trace names a channel outside the selected View",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": "solve_hb", "context_kind": "runtime"},
                )
        if ref._lineage.get("ptc") is not None and not spec.allow_driven_ptc:
            effective: dict[tuple[str, tuple[int, ...]], complex] = {}
            for case in spec.cases:
                effective.clear()
                for drive in spec.drives:
                    current = case.currents.get(drive)
                    if current is not None:
                        key = (drive.at.id, drive.mode)
                        effective[key] = effective.get(key, 0j) + complex(current.to("ampere").magnitude)
                if any(value != 0j for value in effective.values()):
                    raise SCNSimValidationError(
                        "a driven HB request through PTC requires allow_driven_ptc=True",
                        stage="preflight",
                        evidence={"type": "failure_evidence", "operation": "solve_hb", "context_kind": "runtime"},
                    )

    def _complete_parameters(self, supplied: ParameterSet | None) -> ParameterSet:
        baselines = {parameter: parameter.baseline for parameter in self._parameter_lookup.values()}
        if supplied is None:
            return ParameterSet(baselines)
        if not isinstance(supplied, ParameterSet):
            raise TypeError("parameters must be ParameterSet or None")
        for parameter, value in supplied.values.items():
            key = _parameter_key(parameter)
            current = self._parameter_lookup.get(key)
            if current is not parameter:
                raise ValueError("ParameterSet contains a parameter from another Plan")
            baselines[current] = value
        for parameter in supplied.allow_extrapolation:
            key = _parameter_key(parameter)
            if self._parameter_lookup.get(key) is not parameter:
                raise ValueError("ParameterSet extrapolation authorization belongs to another Plan")
        return ParameterSet(baselines, allow_extrapolation=supplied.allow_extrapolation)

    def _request(
        self,
        operation: str,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        parameters: ParameterSet | None,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        """Compatibility request encoder for the already-complete raw View."""

        request, source_units, _ = self._materialized_request(operation, ref, spec, parameters)
        return request, source_units

    def _materialized_request(
        self,
        operation: str,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        parameters: ParameterSet | None,
    ) -> tuple[dict[str, object], list[dict[str, object]], Mapping[str, object] | None]:
        """Materialize one bound request before it can enter the workspace.

        A ``NetworkViewRef`` keeps only immutable user declarations.  The
        compiler owns the candidate-dependent PTC, transform, and retain
        evidence, so it realizes a temporary request first; only the returned
        closed lineage is incorporated into the durable request identity.
        """
        preliminary, source_units = self._request_declaration(operation, ref, spec, parameters)
        if operation != "solve_hb" and _raw_view_lineage(ref._lineage):
            # Raw 0/1/N-port Direct has no candidate-dependent reduction
            # evidence.  Its sealed original lineage is already final and
            # preserves the accepted dev3/dev4 request path (including
            # resolve without a Julia preparation).
            return preliminary, source_units, None
        compiled = self._preflight(preliminary)
        realized = compiled.get("ref_lineage")
        try:
            _verify_v1_lineage(realized, self._plan_document)
        except EvidenceIntegrityError as error:
            raise BackendProtocolError(
                "Julia preflight returned an invalid realized View lineage",
                stage="preflight",
                evidence={"error": str(error)},
            ) from error
        request = canonical_request_document(
            plan_sha256=self._plan_sha256,
            operation=operation,
            ref_lineage=realized,
            spec=preliminary["spec"],
            parameters=preliminary["parameters"],
            runtime_semantic=preliminary["runtime_semantic"],
        )
        return request, source_units, compiled

    def _request_declaration(
        self,
        operation: str,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        parameters: ParameterSet | None,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        """Encode a read-only lazy View declaration without invoking Julia."""

        if isinstance(spec, HBSolveSpec):
            if operation != "solve_hb":
                raise CompilerInvariantError("HB Spec has a non-HB operation", stage="request_encode")
            self._validate_hb_request(ref, spec)
        else:
            self._validate_direct_request(operation, ref, spec)
        resolved = self._complete_parameters(parameters)
        if isinstance(spec, OptimizationSpec):
            # Request-level authorization is consumed only by compiler
            # baseline/preflight lowering. CMA candidate and winner
            # ParameterSets remain authorization-free and ledger-owned.
            resolved = ParameterSet(
                resolved.values,
                allow_extrapolation=spec.allow_extrapolation,
            )
        try:
            encoded_spec = _encode_spec(spec, resolved, coordinate_id=self._coordinate_id)
            source_units = self._source_units(spec, resolved)
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
            semantic["algorithm_id"] = "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v2"
        else:
            raise CompilerInvariantError("operation is outside the runtime", stage="request_encode")
        preliminary = canonical_request_document(
            plan_sha256=self._plan_sha256,
            operation=operation,
            ref_lineage=ref._lineage,
            spec=encoded_spec,
            parameters=resolved._canonical_record(),
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
    ):
        request_bytes = canonical_json_bytes(request)
        request_sha = sha256_hex(request_bytes)
        with self._binding.reader():
            success = self._binding.find_success(request_sha)
            if success is not None:
                return self._decode_success(success)
        prepared = prepare_runtime()
        executable_sha = sha256(prepared.executable.read_bytes()).hexdigest()
        started = _utc_now()
        with self._binding.writer():
            success = self._binding.find_success(request_sha)
            if success is not None:
                return self._decode_success(success)
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
                return self._decode_success(self._binding.resolve_success(request_sha))
            raise _error_from_record(failure)

    def _source_units(
        self,
        spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        parameters: ParameterSet,
    ) -> list[dict[str, object]]:
        evidence: list[dict[str, object]] = []
        identities: set[str] = set()

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

        def add_component(component: object, path: tuple[str, ...]) -> None:
            parameters_by_id = getattr(component, "_parameters", None)
            realization = getattr(component, "_realization", None)
            affine_sources = getattr(component, "_affine_sources", None)
            rlgc_source = getattr(component, "_rlgc_source", None)
            if (
                not isinstance(parameters_by_id, Mapping)
                or not isinstance(realization, Mapping)
                or not isinstance(affine_sources, Mapping)
            ):
                raise CompilerInvariantError("sealed component source provenance is malformed", stage="request_encode")
            for parameter in component._parameters.values():
                add(
                    _source_unit_identity(
                        scope="plan_parameter",
                        component_path=path,
                        parameter_id=parameter.id,
                        field="baseline",
                    ),
                    parameter.baseline,
                    parameter.unit,
                )
            if rlgc_source is not None:
                if not isinstance(rlgc_source, Mapping):
                    raise CompilerInvariantError("sealed RLGC source provenance is malformed", stage="request_encode")
                units_by_field = {
                    "resistance_per_length": "ohm / meter",
                    "inductance_per_length": "henry / meter",
                    "conductance_per_length": "siemens / meter",
                    "capacitance_per_length": "farad / meter",
                    "extraction_frequency": "hertz",
                }
                if not set(rlgc_source) <= set(units_by_field):
                    raise CompilerInvariantError("sealed RLGC source provenance is malformed", stage="request_encode")
                for field, value in rlgc_source.items():
                    add(
                        _source_unit_identity(
                            scope="plan_rlgc",
                            component_path=path,
                            parameter_id="rlgc",
                            field=field,
                        ),
                        value,
                        units_by_field[field],
                    )
            bindings = realization.get("bindings")
            if not isinstance(bindings, Mapping):
                raise CompilerInvariantError("sealed component binding provenance is malformed", stage="request_encode")
            for parameter_id, source in affine_sources.items():
                binding = bindings.get(parameter_id)
                if (
                    not isinstance(parameter_id, str)
                    or not parameter_id
                    or not isinstance(source, Mapping)
                    or set(source) != {"slope", "intercept", "support"}
                    or not isinstance(binding, Mapping)
                    or binding.get("kind") != "affine"
                ):
                    raise CompilerInvariantError("sealed AffineMap source provenance is malformed", stage="request_encode")

                def source_unit(field: str, index: int | None = None) -> str:
                    envelope = binding.get(field)
                    if index is not None:
                        if not isinstance(envelope, Sequence) or isinstance(envelope, (str, bytes)) or len(envelope) != 2:
                            raise CompilerInvariantError("sealed AffineMap support provenance is malformed", stage="request_encode")
                        envelope = envelope[index]
                    if not isinstance(envelope, Mapping) or not isinstance(envelope.get("si_unit"), str) or not envelope["si_unit"]:
                        raise CompilerInvariantError("sealed AffineMap quantity provenance is malformed", stage="request_encode")
                    return envelope["si_unit"]

                support = source["support"]
                if not isinstance(support, tuple) or len(support) != 2:
                    raise CompilerInvariantError("sealed AffineMap support provenance is malformed", stage="request_encode")
                for field, value, unit in (
                    ("slope", source["slope"], source_unit("slope")),
                    ("intercept", source["intercept"], source_unit("intercept")),
                    ("support_lower", support[0], source_unit("support", 0)),
                    ("support_upper", support[1], source_unit("support", 1)),
                ):
                    add(
                        _source_unit_identity(
                            scope="plan_affine",
                            component_path=path,
                            parameter_id=parameter_id,
                            field=field,
                        ),
                        value,
                        unit,
                    )
            children = realization.get("children", ())
            if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
                raise CompilerInvariantError("sealed Composite child provenance is malformed", stage="request_encode")
            for child in children:
                child_id = getattr(child, "id", None)
                if not isinstance(child_id, str) or not child_id:
                    raise CompilerInvariantError("sealed Composite child provenance is malformed", stage="request_encode")
                add_component(child, (*path, child_id))

        for component in self._plan.components:
            add_component(component, (component.id,))
        for port in self._plan.ports:
            add(
                _source_unit_identity(
                    scope="plan_port",
                    parameter_id=port.id,
                    field="reference_impedance",
                ),
                port.reference_impedance,
                "ohm",
            )
        for parameter, value in parameters.values.items():
            path, identifier = _parameter_key(parameter)
            add(
                _source_unit_identity(
                    scope="request_parameter",
                    component_path=path,
                    parameter_id=identifier,
                    field="value",
                ),
                value,
                parameter.unit,
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
                path, identifier = _parameter_key(parameter)
                for role, bounds in (
                    ("model_default", variable.model_default_bounds),
                    ("consumer_override", variable.consumer_override_bounds),
                ):
                    if bounds is None:
                        continue
                    add(
                        _source_unit_identity(
                            scope="request_optimization_variable",
                            component_path=path,
                            parameter_id=identifier,
                            field=f"{index}:{role}:lower",
                        ),
                        bounds[0],
                        parameter.unit,
                    )
                    add(
                        _source_unit_identity(
                            scope="request_optimization_variable",
                            component_path=path,
                            parameter_id=identifier,
                            field=f"{index}:{role}:upper",
                        ),
                        bounds[1],
                        parameter.unit,
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

    def _decode_success(self, success: VerifiedSuccess):
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
        if kind == "hb_batch":
            return self._decode_hb_batch(identity, result, success.request, success.directory)
        if kind == "direct_response":
            arrays = result["array_catalog"]
            frequency = _read_zarr(success.directory, arrays["frequencies"], complex_values=False)
            s = _read_zarr(success.directory, arrays["s"], complex_values=True)
            y = _read_zarr(success.directory, arrays["y"], complex_values=True)
            z = _read_zarr(success.directory, arrays["z"], complex_values=True)
            _validate_direct_values(
                frequency,
                s,
                y,
                z,
                expected_frequency=_direct_request_frequencies(success.request),
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

            trace_spec = success.request.get("spec")
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
                )

            return _verified_result(
                DirectSolveResult,
                identity=identity,
                frequencies=frequencies,
                s=_verified_result(ScatteringMatrixResult, view=view(s, "dimensionless")),
                y=_verified_result(MatrixFamilyResult, view=view(y, "siemens")),
                z=_verified_result(MatrixFamilyResult, view=view(z, "ohm")),
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
            )
        if kind == "operator":
            arrays = result["array_catalog"]
            frequency = _read_zarr(success.directory, arrays["frequencies"], complex_values=False)
            matrix = _read_zarr(success.directory, arrays["operator"], complex_values=True)
            coordinates = tuple(arrays["operator"].get("coordinate_ids", ()))
            expected = _operator_request_frequencies(success.request)
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
            ledger = tuple(_read_json_artifact(success.directory, artifact) for artifact in result["ledger_artifacts"])
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

    def _decode_hb_batch(
        self,
        identity: ResultIdentity,
        result: Mapping[str, object],
        request: Mapping[str, object],
        directory: Path,
    ) -> HBBatchResult:
        """Reconstruct one fully verified ordered HB case batch."""

        raw_cases = result.get("cases")
        request_spec = request.get("spec")
        declared = request_spec.get("cases") if isinstance(request_spec, Mapping) else None
        if not isinstance(raw_cases, list) or not isinstance(declared, list) or len(raw_cases) != len(declared):
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
            if (
                not isinstance(artifacts, Mapping)
                or set(artifacts) != {"s", "y", "z", "backend_native_s", "backend_native_z", "states", "effective_source_vectors"}
                or not isinstance(trace_artifacts, list)
                or not isinstance(reconciliation, Mapping)
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
                if not np.array_equal(values.view(np.uint64), projected.view(np.uint64)):
                    raise EvidenceIntegrityError(
                        "HB trace artifact is not the bit-exact declared projection of selected S",
                        stage="result_decode",
                    )
                traces[identifier] = _verified_result(
                    TraceResult,
                    frequencies=frequencies,
                    value=units.registry.Quantity(values, "dimensionless"),
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
                bias_state=BiasState(raw["bias_state"]),
                pump_state=PumpState(raw["pump_state"]),
                s=_verified_result(
                    HBScatteringMatrixResult,
                    view=selected_s,
                    backend_native=native_s,
                    reconciliation=recon,
                ),
                y=_verified_result(MatrixFamilyResult, view=selected_y),
                z=_verified_result(MatrixFamilyResult, view=selected_z),
                traces=traces,
                states=units.registry.Quantity(states, "weber"),
                state_node_map=tuple(state_node_map),
            )
        return _verified_result(HBBatchResult, identity=identity, cases=outcomes)

    def _decode_parameter_set(self, record: Mapping[str, object]) -> ParameterSet:
        values: dict[ParameterRef, object] = {}
        for binding in record["bindings"]:
            reference = binding["parameter"]
            path = reference["component_path"]
            if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
                raise EvidenceIntegrityError("winner parameter path is malformed", stage="result_decode")
            parameter = self._parameter_lookup.get((tuple(path), reference["parameter_id"]))
            if parameter is None:
                raise EvidenceIntegrityError("winner parameter is absent from sealed Plan", stage="result_decode")
            values[parameter] = quantity_from_envelope(binding["value"], registry=units.registry)
        return ParameterSet(values)


def _original_lineage_document(
    plan: Mapping[str, object],
    plan_sha256: str,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    node_order, _ = _plan_coordinates(plan)
    port_order = [port["port_id"] for port in plan["ports"]]
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


def _compiled_schematic_evidence(plan: CircuitPlan) -> Mapping[str, object]:
    """Compile a sealed baseline declaration without a workspace or solver."""

    node_state = tuple((node, node.id, node.visibility) for node in plan._nodes)
    coordinate_resolution = plan._coordinate_resolution
    try:
        plan_document = canonical_plan_document(plan._canonical_snapshot())
    finally:
        for node, identifier, visibility in node_state:
            node.id, node.visibility = identifier, visibility
        plan._coordinate_resolution = coordinate_resolution
    plan_bytes = canonical_json_bytes(plan_document)
    plan_sha = sha256_hex(plan_bytes)
    runtime = _runtime_identity_base()
    parameters = ParameterSet(
        {
            parameter: parameter.baseline
            for component in plan.components
            for parameter in component._parameters.values()
        }
    )._canonical_record()
    semantic = dict(runtime)
    semantic["algorithm_id"] = "scnsim.direct_response.v1"
    request = canonical_request_document(
        plan_sha256=plan_sha,
        operation="solve_direct",
        ref_lineage=_original_lineage_document(plan_document, plan_sha, runtime),
        spec={
            "type": "direct_solve",
            "frequencies": [
                quantity_envelope(
                    units.registry.Quantity(1.0, "hertz"),
                    si_unit="hertz",
                    registry=units.registry,
                )
            ],
            "traces": [],
        },
        parameters=parameters,
        runtime_semantic=semantic,
    )
    compiled = dict(_run_preflight(plan_bytes, request))
    compiled["expanded_graph_sha256"] = sha256_hex(
        {
            "schema": "scnsim.expanded_graph_identity",
            "schema_version": 1,
            "plan_sha256": compiled["plan_sha256"],
            "node_order": compiled["node_order"],
            "resolved_bindings": compiled["resolved_bindings"],
            "expanded_branch_rows": compiled["expanded_branch_rows"],
        }
    )
    return compiled


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
        if not isinstance(item, Mapping) or set(item) != {"drive_id", "mode", "coefficient", "injection_map_sha256"}:
            raise EvidenceIntegrityError("HB effective-source evidence is malformed", stage="result_decode")
        drive_id = item.get("drive_id")
        mode = item.get("mode")
        if (
            not isinstance(drive_id, str)
            or not drive_id
            or not isinstance(mode, list)
            or any(not isinstance(entry, int) or isinstance(entry, bool) for entry in mode)
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
    if axis.get("kind") != kind or not isinstance(values, list) or not values:
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
        *(('residual_f64',) if comparable is True else ()),
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
) -> dict[str, object]:
    if isinstance(value, QuantitySelector):
        return {
            "type": value.type,
            "spec": _encode_direct_quantity(value.spec, coordinate_id=coordinate_id),
            "projection": value.projection,
        }
    if isinstance(value, QuantitySum):
        return {
            "type": "quantity_sum",
            "terms": [
                _encode_scalar_expression(term, coordinate_id=coordinate_id)
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
) -> dict[str, object]:
    if isinstance(spec, DirectSolveSpec):
        return {
            "type": "direct_solve",
            "frequencies": _frequency_grid(spec.frequencies),
            "traces": [dict(trace._canonical_record()) for trace in spec.traces],
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
            "traces": [dict(trace._canonical_record()) for trace in spec.traces],
            "allow_driven_ptc": spec.allow_driven_ptc,
        }
    if isinstance(spec, (DiagonalRootSpec, HybridizedPoleSpec, TransferZeroSpec, ResidueNormalizedCouplingSpec, ResponseElementSpec, OperatorSpec)):
        return _encode_direct_quantity(spec, coordinate_id=coordinate_id)
    variables: list[dict[str, object]] = []
    for variable in spec.variables:
        parameter = variable.parameter
        lower, upper = variable.bounds
        low = quantity_envelope(lower, si_unit=parameter.unit, registry=units.registry)
        high = quantity_envelope(upper, si_unit=parameter.unit, registry=units.registry)
        low_value = float(lower.to(parameter.unit).magnitude)
        high_value = float(upper.to(parameter.unit).magnitude)
        baseline_value = float(parameter.baseline.to(parameter.unit).magnitude)
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
        default = [quantity_envelope(item, si_unit=parameter.unit, registry=units.registry) for item in variable.model_default_bounds]
        override = None if variable.consumer_override_bounds is None else [quantity_envelope(item, si_unit=parameter.unit, registry=units.registry) for item in variable.consumer_override_bounds]
        variables.append(
            {
                "parameter": parameter._canonical_ref(),
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
        "allow_extrapolation": [parameter._canonical_ref() for parameter in spec.allow_extrapolation],
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
    parameters = request.get("parameters")
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
    expected_kind = (
        "direct_response" if request.get("operation") == "solve_direct"
        else "hb_batch" if request.get("operation") == "solve_hb"
        else "optimization" if request.get("operation") == "optimize_direct"
        else request.get("spec", {}).get("type")
        if request.get("operation") == "evaluate_direct" and isinstance(request.get("spec"), Mapping)
        else None
    )
    expected_result_fields = (
        {
            "schema", "schema_version", "result_kind", "request_sha256",
            "attempt_sha256", "scalar_catalog", "array_catalog",
        }
        if expected_kind in {
            "direct_response", "diagonal_root", "hybridized_pole", "transfer_zero",
            "residue_normalized_coupling", "response_element", "operator",
        }
        else {
            "schema", "schema_version", "result_kind", "request_sha256",
            "attempt_sha256", "baseline", "best", "completed_generations",
            "unused_evaluations", "ledger_artifacts",
        }
        if expected_kind == "optimization"
        else {
            "schema", "schema_version", "result_kind", "request_sha256",
            "attempt_sha256", "lattice", "truncation", "cases",
        }
        if expected_kind == "hb_batch"
        else None
    )
    if (
        expected_result_fields is None
        or set(result) != expected_result_fields
        or result.get("schema") != "scnsim.result"
        or result.get("schema_version") != 1
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
    if expected_kind == "hb_batch":
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
    if expected_kind != "hb_batch":
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
