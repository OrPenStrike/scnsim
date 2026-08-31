"""Plan-bound execution, exact request identity, and typed Result reconstruction.

The dev4 candidate extends the one-Port Lessons 1--5 path through sealed
Composite public parameters and coordinates.  Every later V1 surface remains
importable but fails explicitly instead of returning a partial result.
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
    canonical_json_bytes,
    canonical_plan_document,
    canonical_receipt_document,
    canonical_request_document,
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
    DirectQuantityResult,
    DiagonalRootResult,
    DirectSolveResult,
    ExplanationResult,
    HBBatchResult,
    InventoryResult,
    MatrixFamilyResult,
    MatrixView,
    OptimizationBest,
    OptimizationResult,
    OperatorResult,
    ReportResult,
    ResultIdentity,
    ScatteringMatrixResult,
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
    """Collect public node and Composite-coordinate identities from one seal."""

    coordinates = {
        node["node_id"]
        for node in plan["nodes"]
        if node["visibility"] in {"public", "port_promoted"}
    }

    def visit(component: Mapping[str, object]) -> None:
        realization = component.get("realization")
        if not isinstance(realization, Mapping) or realization.get("kind") != "composite":
            return
        for record in realization.get("public_coordinate_map", ()):
            if not isinstance(record, Mapping) or not isinstance(record.get("public_id"), str):
                raise CompilerInvariantError(
                    "Composite public-coordinate map is malformed",
                    stage="plan_seal",
                )
            coordinates.add(record["public_id"])
    for component in plan["components"]:
        if not isinstance(component, Mapping):
            raise CompilerInvariantError("Plan component is malformed", stage="plan_seal")
        visit(component)
    return tuple(sorted(coordinates))


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
        "dev4 objectives require diagonal-root selectors or their QuantitySum",
        stage="spec_validation",
    )


class ReductionPipeline:
    """An immutable ordered declaration of analysis-view reductions.

    dev4 executes the terminal single-coordinate ``retain()`` step.  PTC,
    paired transforms, and multi-coordinate retain stay explicit later-slice
    capabilities.
    """

    __slots__ = ("_retained",)

    def __init__(self) -> None:
        self._retained: tuple[str | ElectricNodeRef | CoordinateRef, ...] | None = None

    def ptc(self, *ports: PortRef) -> ReductionPipeline:
        """Declare probe-load compensation; available in the dev5 slice."""

        unavailable("ReductionPipeline.ptc")

    def transform_pair(
        self,
        node_a: str | ElectricNodeRef | CoordinateRef,
        node_b: str | ElectricNodeRef | CoordinateRef,
        *,
        id: str,
    ) -> ReductionPipeline:
        """Declare one paired-coordinate basis transform; available in dev5."""

        unavailable("ReductionPipeline.transform_pair")

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
        child = ReductionPipeline()
        child._retained = tuple(coordinates)
        return child


class NetworkViewRef:
    """Immutable lazy reference to one Plan and one reduction lineage."""

    __slots__ = ("_run", "_lineage", "_retained")

    def __init__(self) -> None:
        unavailable("NetworkViewRef construction")

    @classmethod
    def _create(
        cls,
        run: CircuitRun,
        lineage: Mapping[str, object],
        retained: tuple[str, ...] = (),
    ) -> NetworkViewRef:
        ref = object.__new__(cls)
        ref._run = run
        ref._lineage = MappingProxyType(dict(lineage))
        ref._retained = retained
        return ref

    def reduce(self, pipeline: ReductionPipeline) -> NetworkViewRef:
        """Derive an immutable lazy child View without compiling or solving."""

        if not isinstance(pipeline, ReductionPipeline):
            raise TypeError("reduce() requires a ReductionPipeline")
        if self._retained:
            raise ValueError("a terminal retained View cannot be reduced again")
        return self._run._derive_view(pipeline)


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
        self._original = NetworkViewRef._create(self, original)

    @property
    def original(self) -> NetworkViewRef:
        """The sealed Plan's immutable zero-reduction root View."""

        return self._original

    def _original_lineage(self) -> dict[str, object]:
        node_order = list(dict.fromkeys(
            [node["node_id"] for node in self._plan_document["nodes"]]
            + sorted(self._public_coordinates)
        ))
        port_order = [port["port_id"] for port in self._plan_document["ports"]]
        if len(port_order) != 1:
            raise PortRealizabilityError(
                "dev4 CircuitRun requires exactly one logical Port",
                stage="plan_seal",
                evidence={"type": "failure_evidence", "operation": "workspace_bind", "context_kind": "workspace"},
            )
        compiled_graph = sha256_hex(
            {
                "schema": "scnsim.compiled_graph_identity",
                "schema_version": 1,
                "plan_sha256": self._plan_sha256,
                "julia_source_sha256": self._runtime_base["julia_source_sha256"],
            }
        )
        original = {
            "type": "original",
            "compiled_graph_sha256": compiled_graph,
            "coordinate_order": node_order,
            "port_order": port_order,
            "port_realizable": len(port_order) == 1,
        }
        record: dict[str, object] = {
            "type": "network_view_lineage",
            "original": original,
            "ptc": None,
            "transforms": [],
            "retain": None,
            "terminal_coordinates": port_order,
            "port_realizable": len(port_order) == 1,
        }
        record["lineage_sha256"] = sha256_hex(record)
        return record

    def _derive_view(self, pipeline: ReductionPipeline) -> NetworkViewRef:
        if pipeline._retained is None:
            raise ValueError("a dev4 derived View requires terminal retain()")
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
        """Execute one-Port Direct in dev4; HB remains an explicit dev6 fail-fast surface."""

        if isinstance(spec, HBSolveSpec):
            unavailable("CircuitRun.solve(HBSolveSpec)")
        self._require_ref(ref)
        request, source_units = self._request("solve_direct", ref, spec, parameters)
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
        if not isinstance(spec, DiagonalRootSpec):
            unavailable(f"CircuitRun.evaluate({type(spec).__name__})")
        request, source_units = self._request("evaluate_direct", ref, spec, parameters)
        return self._execute(request, source_units)

    def optimize(self, ref: NetworkViewRef, spec: OptimizationSpec) -> OptimizationResult:
        """Run one pinned Direct CMA-ES request and return its exact winner."""

        self._require_ref(ref)
        request, source_units = self._request("optimize_direct", ref, spec, None)
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
        elif isinstance(spec, DiagonalRootSpec):
            operation = "evaluate_direct"
        elif isinstance(spec, OptimizationSpec):
            if parameters is not None:
                raise TypeError("parameters must be omitted for OptimizationSpec")
            operation = "optimize_direct"
        else:
            unavailable(f"CircuitRun.resolve({type(spec).__name__})")
        request, _ = self._request(operation, ref, spec, parameters)
        request_sha = sha256_hex(canonical_json_bytes(request))
        with self._binding.reader():
            return self._decode_success(self._binding.resolve_success(request_sha))

    def explain(
        self,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | HBSolveSpec | DiagonalRootSpec | HybridizedPoleSpec | TransferZeroSpec | ResidueNormalizedCouplingSpec | ResponseElementSpec | OperatorSpec | OptimizationSpec,
        *,
        parameters: ParameterSet | None = None,
    ) -> ExplanationResult:
        """Compile and present request evidence without creating an attempt."""

        self._require_ref(ref)
        if isinstance(spec, HBSolveSpec) or not isinstance(spec, (DirectSolveSpec, DiagonalRootSpec, OptimizationSpec)):
            unavailable(f"CircuitRun.explain({type(spec).__name__})")
        if isinstance(spec, OptimizationSpec) and parameters is not None:
            raise TypeError("parameters must be omitted for OptimizationSpec")
        operation = "solve_direct" if isinstance(spec, DirectSolveSpec) else "evaluate_direct" if isinstance(spec, DiagonalRootSpec) else "optimize_direct"
        request, _ = self._request(operation, ref, spec, parameters)
        prepared = prepare_runtime()
        with tempfile.TemporaryDirectory(prefix="scnsim-explain-") as temporary:
            plan_path = Path(temporary) / "plan.json"
            request_path = Path(temporary) / "request.json"
            plan_path.write_bytes(self._plan_bytes)
            request_path.write_bytes(canonical_json_bytes(request))
            compiled = run_preflight(
                prepared,
                plan_path=plan_path.resolve(),
                request_path=request_path.resolve(),
            )
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

        unavailable("CircuitRun.inventory")

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

    def _validate_dev4_request(
        self,
        operation: str,
        ref: NetworkViewRef,
        spec: DirectSolveSpec | DiagonalRootSpec | OptimizationSpec,
    ) -> None:
        if isinstance(spec, DirectSolveSpec):
            if spec.traces:
                unavailable("DirectSolveSpec.traces")
            if ref._lineage["port_realizable"] is not True:
                raise PortRealizabilityError(
                    "Direct response requires a port-realizable View",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": operation, "context_kind": "direct_response"},
                )
            if ref is not self._original:
                unavailable("dev4 Direct solve on a derived Port-realizable View")
            if len(self._plan.ports) != 1:
                unavailable("N-port Direct response")
            return
        if isinstance(spec, DiagonalRootSpec):
            if len(ref._retained) != 1 or ref._retained[0] != self._coordinate_id(spec.coordinate):
                raise PortRealizabilityError(
                    "DiagonalRootSpec coordinate must equal the retained View coordinate",
                    stage="preflight",
                    evidence={"type": "failure_evidence", "operation": operation, "context_kind": "direct_quantity"},
                )
            return
        if len(ref._retained) != 1:
            unavailable("CircuitRun.optimize on non-single-retained View")
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
                if not isinstance(selector.spec, DiagonalRootSpec):
                    unavailable("dev4 optimization selector")
                if self._coordinate_id(selector.spec.coordinate) != ref._retained[0]:
                    raise InvalidOptimizationSpec(
                        "optimization selector coordinate must equal the retained View",
                        stage="spec_validation",
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
        spec: DirectSolveSpec | DiagonalRootSpec | OptimizationSpec,
        parameters: ParameterSet | None,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        self._validate_dev4_request(operation, ref, spec)
        resolved = self._complete_parameters(parameters)
        if self._affine_plan and (
            resolved.allow_extrapolation
            or isinstance(spec, OptimizationSpec) and spec.allow_extrapolation
        ):
            unavailable("AffineMap allow_extrapolation")
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
        semantic["algorithm_id"] = {
            "solve_direct": "scnsim.direct_response.v1",
            "evaluate_direct": "scnsim.diagonal_root.newton32.v1",
            "optimize_direct": "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v2",
        }[operation]
        request = canonical_request_document(
            plan_sha256=self._plan_sha256,
            operation=operation,
            ref_lineage=ref._lineage,
            spec=encoded_spec,
            parameters=resolved._canonical_record(),
            runtime_semantic=semantic,
        )
        return request, source_units

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
                    _validate_success_staging(allocation.staging_directory, outcome, request)
                    receipt = _receipt(
                        request=request,
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
        spec: DirectSolveSpec | DiagonalRootSpec | OptimizationSpec,
        parameters: ParameterSet,
    ) -> list[dict[str, object]]:
        evidence: list[dict[str, object]] = []

        def add(identity: str, value: object, si_unit: str) -> None:
            magnitude = np.asarray(value.magnitude)
            probe = (
                value
                if magnitude.ndim == 0
                else units.registry.Quantity(float(magnitude.flat[0]), value.units)
            )
            encoded = quantity_envelope(probe, si_unit=si_unit, registry=units.registry)
            evidence.append(
                {
                    "identity": identity,
                    "source_unit": str(value.units),
                    "canonical_si_unit": encoded["si_unit"],
                    "canonical_dimensionality": encoded["dimensionality"],
                }
            )

        for component in self._plan.components:
            for parameter in component._parameters.values():
                path, identifier = _parameter_key(parameter)
                add(f"plan.{'.'.join(path)}.{identifier}", parameter.baseline, parameter.unit)
        for port in self._plan.ports:
            add(f"plan.port.{port.id}.reference_impedance", port.reference_impedance, "ohm")
        for parameter, value in parameters.values.items():
            path, identifier = _parameter_key(parameter)
            add(
                f"request.parameter.{'.'.join(path)}.{identifier}",
                value,
                parameter.unit,
            )
        if isinstance(spec, DirectSolveSpec):
            add("request.spec.frequencies", spec.frequencies, "hertz")
        elif isinstance(spec, DiagonalRootSpec):
            add("request.spec.root_hint", spec.root_hint, "hertz")
        else:
            for index, variable in enumerate(spec.variables):
                parameter = variable.parameter
                for role, bounds in (
                    ("model_default", variable.model_default_bounds),
                    ("consumer_override", variable.consumer_override_bounds),
                ):
                    if bounds is None:
                        continue
                    add(f"request.spec.variable.{index}.{role}.lower", bounds[0], parameter.unit)
                    add(f"request.spec.variable.{index}.{role}.upper", bounds[1], parameter.unit)
            for index, objective in enumerate(spec.objectives):
                add(f"request.spec.objective.{index}.target", objective.target, "hertz")
                add(f"request.spec.objective.{index}.weight", objective.weight, "dimensionless")
                if objective.scale is not None:
                    add(f"request.spec.objective.{index}.scale", objective.scale, "hertz")
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

            return _verified_result(
                DirectSolveResult,
                identity=identity,
                frequencies=frequencies,
                s=_verified_result(ScatteringMatrixResult, view=view(s, "dimensionless")),
                y=_verified_result(MatrixFamilyResult, view=view(y, "siemens")),
                z=_verified_result(MatrixFamilyResult, view=view(z, "ohm")),
                traces={},
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
        raise EvidenceIntegrityError("verified Result kind is outside dev4", stage="result_decode", evidence={"result_kind": kind})

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
        if not isinstance(value.spec, DiagonalRootSpec):
            unavailable("dev4 CostObjective quantity")
        return {
            "type": value.type,
            "spec": _encode_root(value.spec, coordinate_id=coordinate_id),
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


def _encode_spec(
    spec: DirectSolveSpec | DiagonalRootSpec | OptimizationSpec,
    parameters: ParameterSet,
    *,
    coordinate_id: Callable[[str | ElectricNodeRef | CoordinateRef], str] = _coordinate_id,
) -> dict[str, object]:
    if isinstance(spec, DirectSolveSpec):
        return {"type": "direct_solve", "frequencies": _frequency_grid(spec.frequencies), "traces": []}
    if isinstance(spec, DiagonalRootSpec):
        return _encode_root(spec, coordinate_id=coordinate_id)
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
        target = quantity_envelope(objective.target, si_unit="hertz", registry=units.registry)
        target_value = abs(float(objective.target.to("hertz").magnitude))
        if objective.scale is None:
            if target_value == 0.0:
                raise ValueError("a dimensional zero target requires an explicit objective scale")
            scale_value = units.registry.Quantity(target_value, "hertz")
            scale_source = "relative_target"
        else:
            scale_value = objective.scale
            scale_source = "explicit"
        scale_magnitude = float(scale_value.to("hertz").magnitude)
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
                "resolved_scale": quantity_envelope(scale_value, si_unit="hertz", registry=units.registry),
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
            "operation": operation if operation in {"solve_direct", "evaluate_direct", "optimize_direct"} else "backend_protocol",
            "context_kind": "protocol",
            "request_sha256": request_sha,
            "attempt_sha256": attempt_sha,
        },
    }


def _receipt(
    *,
    request: Mapping[str, object],
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
        "extrapolation_evidence": [],
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
) -> None:
    if outcome.get("runtime_semantic") != request.get("runtime_semantic"):
        raise BackendProtocolError("outcome runtime identity does not match the request", stage="outcome")
    result_path = _inside(staging, "result.json")
    if result_path.is_symlink() or not result_path.is_file() or sha256(result_path.read_bytes()).hexdigest() != outcome.get("result_sha256"):
        raise BackendProtocolError("success outcome does not bind result.json", stage="outcome")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if canonical_json_bytes(result) != result_path.read_bytes():
        raise BackendProtocolError("result.json is not canonical", stage="outcome")
    expected_kind = {
        "solve_direct": "direct_response",
        "evaluate_direct": "diagonal_root",
        "optimize_direct": "optimization",
    }.get(request.get("operation"))
    expected_result_fields = {
        "direct_response": {
            "schema", "schema_version", "result_kind", "request_sha256",
            "attempt_sha256", "scalar_catalog", "array_catalog",
        },
        "diagonal_root": {
            "schema", "schema_version", "result_kind", "request_sha256",
            "attempt_sha256", "scalar_catalog", "array_catalog",
        },
        "optimization": {
            "schema", "schema_version", "result_kind", "request_sha256",
            "attempt_sha256", "baseline", "best", "completed_generations",
            "unused_evaluations", "ledger_artifacts",
        },
    }.get(expected_kind)
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
    )
    catalogs: list[Mapping[str, object]] = []
    if expected_kind == "direct_response":
        catalogs.extend(result["array_catalog"].values())
    elif expected_kind == "optimization":
        catalogs.extend(result["ledger_artifacts"])
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
