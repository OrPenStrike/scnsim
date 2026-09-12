"""Decode typed Results only from independently verified workspace evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

from . import units
from ._analysis import (
    _coordinate_binding_key,
    _encode_scalar_expression,
    _quantity_coordinates,
)
from ._canonical import (
    canonical_json_bytes,
    canonical_parameters_sha256,
    complex_quantity_from_envelope,
    float64_from_hex,
    float64_hex,
    quantity_from_envelope,
    sha256_hex,
)
from ._evidence import (
    _VerifiedEvidenceLease,
    _direct_request_frequencies,
    _error_from_record,
    _is_sha256_text,
    _operator_request_frequencies,
    _read_canonical_artifact_json,
    _read_json_artifact,
    _read_zarr,
    _validate_direct_values,
)
from ._physical_values import RLGC, RLGCParameterSpec
from ._workspace import VerifiedSuccess
from .authoring import (
    CircuitPlan,
    CoordinateRef,
    ElectricNodeRef,
    ParameterRef,
    ParameterSet,
)
from .errors import EvidenceIntegrityError, HBCaseFailure
from .results import (
    BiasState,
    DiagonalRootResult,
    DirectQuantityResult,
    DirectSolveResult,
    HBBatchResult,
    HBCaseOutcome,
    HBScatteringMatrixResult,
    MatrixFamilyResult,
    MatrixView,
    OperatorPointResult,
    OperatorResult,
    OptimizationBest,
    OptimizationResult,
    ParameterPointIdentity,
    ParameterPointOutcome,
    ParameterSweepResult,
    PumpState,
    ReconciliationEvidence,
    ResultIdentity,
    ScatteringMatrixResult,
    TraceResult,
    _parameter_sweep_result,
    _point_accessor,
    _point_outcome,
    _verified_result,
)
from .specs import QuantitySelector


class VerifiedResultDecoder:
    """Captured identity lookups for decoding one verified success chain."""

    __slots__ = (
        "_plan_sha256",
        "_plan",
        "_parameter_lookup",
        "_coordinate_lookup",
    )

    def __init__(
        self,
        *,
        plan_sha256: str,
        plan: CircuitPlan,
        parameter_lookup: Mapping[tuple[str, str], ParameterRef],
        coordinate_lookup: Mapping[str, str | None],
    ) -> None:
        self._plan_sha256 = plan_sha256
        self._plan = plan
        self._parameter_lookup = MappingProxyType(dict(parameter_lookup))
        self._coordinate_lookup = MappingProxyType(dict(coordinate_lookup))

    def _coordinate_id(
        self,
        value: str | ElectricNodeRef | CoordinateRef,
    ) -> str:
        if isinstance(value, ElectricNodeRef) and value.plan is not self._plan:
            raise ValueError("coordinate belongs to another Plan")
        if isinstance(value, CoordinateRef):
            if value.scope.root is not self._plan:
                raise ValueError("coordinate belongs to another Plan")
            key = canonical_json_bytes(
                {"scope": list(value.scope.path()), "id": value.id}
            ).decode("utf-8")
            resolved = self._coordinate_lookup.get(key)
        else:
            identifier = value if isinstance(value, str) else getattr(value, "id", None)
            resolved = self._coordinate_lookup.get(identifier)
        if resolved is None:
            raise ValueError("coordinate is not a public alias in this Plan")
        return resolved

    def _decode_success(
        self,
        success: VerifiedSuccess,
        *,
        bound_spec: object | None = None,
        evidence_lease: _VerifiedEvidenceLease,
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
                evidence_lease=evidence_lease,
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
            frequency = _read_zarr(
                directory, arrays["frequencies"], complex_values=False
            )
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
                or any(
                    not isinstance(coordinate, str) or not coordinate
                    for coordinate in coordinates
                )
                or any(values.shape != expected_shape for values in (s, y, z))
                or tuple(arrays["y"].get("coordinate_ids", ())) != coordinates
                or tuple(arrays["z"].get("coordinate_ids", ())) != coordinates
            ):
                raise EvidenceIntegrityError(
                    "Direct arrays disagree with the selected N-port basis",
                    stage="result_decode",
                )
            channels = tuple((coordinate, ()) for coordinate in coordinates)
            loads = {
                item["port_id"]: item["state"]
                for item in arrays["s"]["probe_load_state"]
            }

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
            declared_traces = (
                trace_spec.get("traces") if isinstance(trace_spec, Mapping) else None
            )
            if not isinstance(declared_traces, list):
                raise EvidenceIntegrityError(
                    "Direct request trace declarations are malformed",
                    stage="result_decode",
                )
            trace_results: dict[str, TraceResult] = {}
            for trace in declared_traces:
                if not isinstance(trace, Mapping):
                    raise EvidenceIntegrityError(
                        "Direct trace declaration is malformed", stage="result_decode"
                    )
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
                    raise EvidenceIntegrityError(
                        "Direct trace does not bind the selected S basis",
                        stage="result_decode",
                    )
                trace_results[identifier] = _verified_result(
                    TraceResult,
                    frequencies=frequencies,
                    value=units.registry.Quantity(
                        s[
                            :,
                            coordinates.index(output_coordinate),
                            coordinates.index(input_coordinate),
                        ],
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
                root=complex_quantity_from_envelope(
                    scalars["root"], registry=units.registry
                ),
                frequency=quantity_from_envelope(
                    scalars["frequency"], registry=units.registry
                ),
                linewidth=quantity_from_envelope(
                    scalars["linewidth"], registry=units.registry
                ),
                slope=complex_quantity_from_envelope(
                    scalars["slope"], registry=units.registry
                ),
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
                root=complex_quantity_from_envelope(
                    scalars["root"], registry=units.registry
                ),
                frequency=quantity_from_envelope(
                    scalars["frequency"], registry=units.registry
                ),
                linewidth=quantity_from_envelope(
                    scalars["linewidth"], registry=units.registry
                ),
                slope=complex_quantity_from_envelope(
                    scalars["slope"], registry=units.registry
                ),
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
                zero=complex_quantity_from_envelope(
                    scalars["zero"], registry=units.registry
                ),
                frequency=quantity_from_envelope(
                    scalars["frequency"], registry=units.registry
                ),
                numerator_slope=complex_quantity_from_envelope(
                    scalars["numerator_slope"], registry=units.registry
                ),
                denominator=complex_quantity_from_envelope(
                    scalars["denominator"], registry=units.registry
                ),
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
                coupling=complex_quantity_from_envelope(
                    scalars["coupling"], registry=units.registry
                ),
                magnitude=quantity_from_envelope(
                    scalars["magnitude"], registry=units.registry
                ),
                branch_a_residue=complex_quantity_from_envelope(
                    scalars["branch_a_residue"], registry=units.registry
                ),
                branch_b_residue=complex_quantity_from_envelope(
                    scalars["branch_b_residue"], registry=units.registry
                ),
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
                value=complex_quantity_from_envelope(
                    scalars["value"], registry=units.registry
                ),
                magnitude=quantity_from_envelope(
                    scalars["magnitude"], registry=units.registry
                ),
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
            frequency = _read_zarr(
                directory, arrays["frequencies"], complex_values=False
            )
            matrix = _read_zarr(directory, arrays["operator"], complex_values=True)
            coordinates = tuple(arrays["operator"].get("coordinate_ids", ()))
            expected = _operator_request_frequencies(request)
            if (
                frequency.shape != expected.shape
                or not np.array_equal(
                    frequency.view(np.uint64), expected.view(np.uint64)
                )
                or matrix.shape != (frequency.size, len(coordinates), len(coordinates))
                or not coordinates
                or len(set(coordinates)) != len(coordinates)
                or any(not isinstance(value, str) or not value for value in coordinates)
                or not np.all(np.isfinite(matrix))
            ):
                raise EvidenceIntegrityError(
                    "operator artifacts disagree with the request basis",
                    stage="result_decode",
                )
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
            ledger = tuple(
                _read_json_artifact(directory, artifact)
                for artifact in result["ledger_artifacts"]
            )
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
        raise EvidenceIntegrityError(
            "verified Result kind is outside the runtime",
            stage="result_decode",
            evidence={"result_kind": kind},
        )

    def _decode_parameter_sweep(
        self,
        identity: ResultIdentity,
        result: Mapping[str, object],
        request: Mapping[str, object],
        directory: Path,
        *,
        bound_spec: object | None,
        evidence_lease: _VerifiedEvidenceLease,
    ) -> ParameterSweepResult:
        """Expose verified point metadata lazily and defer scientific payload I/O."""

        del (
            bound_spec
        )  # The canonical request, not a live Spec, owns selector identity.
        source = request.get("parameter_source")
        if not isinstance(source, Mapping) or source.get("kind") not in {
            "grid",
            "points",
        }:
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
            with evidence_lease.reader() as current_directory:
                chunk = _read_canonical_artifact_json(
                    current_directory, path, expected_sha, stage="result_decode"
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
                    with evidence_lease.reader() as current_directory:
                        payload = _read_canonical_artifact_json(
                            current_directory,
                            payload_path,
                            payload_sha,
                            stage="result_decode",
                        )
                        if (
                            payload.get("schema")
                            != "scnsim.parameter_point_payload"
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
                            current_directory,
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
            tuple(
                self._decode_parameter_ref(axis["parameter"]) for axis in source["axes"]
            )
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
            return canonical_json_bytes(
                _encode_scalar_expression(
                    value,
                    coordinate_bindings={
                        _coordinate_binding_key(coordinate): (
                            coordinate
                            if isinstance(coordinate, str)
                            and coordinate in derived_coordinates
                            else self._coordinate_id(coordinate)
                        )
                        for coordinate in _quantity_coordinates(value.spec)
                    },
                )
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
        declared = (
            request_spec.get("cases") if isinstance(request_spec, Mapping) else None
        )
        trace_declarations = (
            request_spec.get("traces") if isinstance(request_spec, Mapping) else None
        )
        if (
            not isinstance(raw_cases, list)
            or not isinstance(declared, list)
            or len(raw_cases) != len(declared)
            or not isinstance(trace_declarations, list)
            or not isinstance(topology_evidence, Mapping)
        ):
            raise EvidenceIntegrityError(
                "HB batch cases disagree with the request", stage="result_decode"
            )
        declared_traces: dict[str, dict[str, object]] = {}
        for declaration in trace_declarations:
            if not isinstance(declaration, Mapping):
                raise EvidenceIntegrityError(
                    "HB request trace declaration is malformed", stage="result_decode"
                )
            identifier = declaration.get("id")
            input_port = declaration.get("input_port")
            output_port = declaration.get("output_port")
            input_mode = declaration.get("input_mode")
            output_mode = declaration.get("output_mode")
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in declared_traces
                or not isinstance(input_port, str)
                or not input_port
                or not isinstance(output_port, str)
                or not output_port
                or not isinstance(input_mode, list)
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in input_mode
                )
                or not isinstance(output_mode, list)
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in output_mode
                )
            ):
                raise EvidenceIntegrityError(
                    "HB request trace declaration is malformed", stage="result_decode"
                )
            declared_traces[identifier] = {
                "input_channel": {
                    "coordinate": input_port,
                    "mode": list(input_mode),
                },
                "output_channel": {
                    "coordinate": output_port,
                    "mode": list(output_mode),
                },
            }
        outcomes: dict[str, HBCaseOutcome] = {}
        for ordinal, (raw, declaration) in enumerate(zip(raw_cases, declared), 1):
            if not isinstance(raw, Mapping) or not isinstance(declaration, Mapping):
                raise EvidenceIntegrityError(
                    "HB case outcome is malformed", stage="result_decode"
                )
            case_id = declaration.get("id")
            if (
                raw.get("case_ordinal") != ordinal
                or raw.get("case_id") != case_id
                or not isinstance(case_id, str)
                or case_id in outcomes
            ):
                raise EvidenceIntegrityError(
                    "HB case ordering or identity is malformed", stage="result_decode"
                )
            effective_sources = _decode_hb_effective_sources(
                raw.get("effective_sources")
            )
            status = raw.get("status")
            if status == "failure":
                failure = raw.get("failure")
                if (
                    not isinstance(failure, Mapping)
                    or set(failure) != {"kind", "stage", "message", "evidence_sha256"}
                    or failure.get("kind") != "hb_case_failure"
                    or failure.get("stage")
                    not in {"operating_point", "linearization", "response_formation"}
                    or not isinstance(failure.get("message"), str)
                    or not failure["message"]
                    or not _is_sha256_text(failure.get("evidence_sha256"))
                ):
                    raise EvidenceIntegrityError(
                        "HB case failure is malformed", stage="result_decode"
                    )
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
                raise EvidenceIntegrityError(
                    "HB case status is malformed", stage="result_decode"
                )
            artifacts = raw.get("artifacts")
            trace_artifacts = raw.get("traces")
            reconciliation = raw.get("reconciliation")
            state_node_map = raw.get("state_node_map")
            operating_point_closure = raw.get("operating_point_closure")
            if (
                not isinstance(artifacts, Mapping)
                or set(artifacts)
                != {
                    "s",
                    "y",
                    "z",
                    "backend_native_s",
                    "backend_native_z",
                    "states",
                    "effective_source_vectors",
                }
                or not isinstance(trace_artifacts, list)
                or not isinstance(reconciliation, Mapping)
                or not isinstance(operating_point_closure, Mapping)
                or not isinstance(state_node_map, list)
                or not state_node_map
            ):
                raise EvidenceIntegrityError(
                    "successful HB case evidence is malformed", stage="result_decode"
                )

            frequency = _direct_request_frequencies(request)
            frequencies = units.registry.Quantity(frequency, "hertz")
            decoded_arrays = {
                name: _read_zarr(directory, artifacts[name], complex_values=True)
                for name in (
                    "s",
                    "y",
                    "z",
                    "backend_native_s",
                    "backend_native_z",
                    "states",
                    "effective_source_vectors",
                )
            }
            if any(
                not np.all(np.isfinite(values)) for values in decoded_arrays.values()
            ):
                raise EvidenceIntegrityError(
                    "HB artifacts contain non-finite values", stage="result_decode"
                )

            def matrix_view(name: str, unit: str) -> MatrixView:
                artifact = artifacts[name]
                if not isinstance(artifact, Mapping):
                    raise EvidenceIntegrityError(
                        "HB matrix catalog is malformed", stage="result_decode"
                    )
                output_channels = _decode_hb_channel_axis(
                    artifact, index=1, kind="output_channel"
                )
                input_channels = _decode_hb_channel_axis(
                    artifact, index=2, kind="input_channel"
                )
                values = decoded_arrays[name]
                if values.shape != (
                    frequency.size,
                    len(output_channels),
                    len(input_channels),
                ):
                    raise EvidenceIntegrityError(
                        "HB matrix shape disagrees with its channel axes",
                        stage="result_decode",
                    )
                coordinates = tuple(artifact.get("coordinate_ids", ()))
                if (
                    not coordinates
                    or any(
                        not isinstance(item, str) or not item for item in coordinates
                    )
                    or len(set(coordinates)) != len(coordinates)
                ):
                    raise EvidenceIntegrityError(
                        "HB matrix coordinate identity is malformed",
                        stage="result_decode",
                    )
                loads = artifact.get("probe_load_state")
                if not isinstance(loads, list):
                    raise EvidenceIntegrityError(
                        "HB probe-load evidence is malformed", stage="result_decode"
                    )
                probe_loads: dict[str, str] = {}
                for item in loads:
                    if (
                        not isinstance(item, Mapping)
                        or set(item) != {"port_id", "state"}
                        or item.get("state") not in {"raw", "compensated"}
                    ):
                        raise EvidenceIntegrityError(
                            "HB probe-load evidence is malformed", stage="result_decode"
                        )
                    port_id = item.get("port_id")
                    if (
                        not isinstance(port_id, str)
                        or not port_id
                        or port_id in probe_loads
                    ):
                        raise EvidenceIntegrityError(
                            "HB probe-load identity is malformed", stage="result_decode"
                        )
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
            if len(trace_artifacts) != len(trace_declarations):
                raise EvidenceIntegrityError(
                    "HB trace catalog disagrees with its request", stage="result_decode"
                )
            selected_matrix = np.asarray(selected_s.matrix.magnitude)
            for artifact, declaration in zip(trace_artifacts, trace_declarations):
                if not isinstance(artifact, Mapping):
                    raise EvidenceIntegrityError(
                        "HB trace catalog is malformed", stage="result_decode"
                    )
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
                    raise EvidenceIntegrityError(
                        "HB trace artifact is malformed", stage="result_decode"
                    )
                input_channel = (
                    declaration.get("input_port"),
                    tuple(declaration.get("input_mode", ())),
                )
                output_channel = (
                    declaration.get("output_port"),
                    tuple(declaration.get("output_mode", ())),
                )
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
                        "input_channel": {
                            "coordinate": input_channel[0],
                            "mode": list(input_channel[1]),
                        },
                        "output_channel": {
                            "coordinate": output_channel[0],
                            "mode": list(output_channel[1]),
                        },
                        "view": request.get("view"),
                        "ref_lineage": result.get("ref_lineage"),
                    },
                )
            states = decoded_arrays["states"]
            if states.ndim != 2 or states.shape[1] != len(state_node_map):
                raise EvidenceIntegrityError(
                    "HB state evidence disagrees with state_node_map",
                    stage="result_decode",
                )
            source_modes = _decode_hb_mode_axis(
                artifacts["effective_source_vectors"], kind="pump_mode"
            )
            source_vectors = decoded_arrays["effective_source_vectors"]
            if source_vectors.ndim != 2 or source_vectors.shape[0] != len(source_modes):
                raise EvidenceIntegrityError(
                    "HB effective-source vectors disagree with their mode axis",
                    stage="result_decode",
                )
            active_rows = np.any(source_vectors != 0.0, axis=1)
            derived_bias = any(
                active and not any(mode)
                for active, mode in zip(active_rows, source_modes)
            )
            derived_pump = any(
                active and any(mode) for active, mode in zip(active_rows, source_modes)
            )
            if raw.get("bias_state") != ("on" if derived_bias else "off") or raw.get(
                "pump_state"
            ) != ("on" if derived_pump else "off"):
                raise EvidenceIntegrityError(
                    "HB BiasState/PumpState disagrees with effective source vectors",
                    stage="result_decode",
                )
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
            _presentation={"declared_traces": declared_traces},
        )

    def _decode_parameter_ref(self, record: object) -> ParameterRef:
        if not isinstance(record, Mapping) or set(record) != {
            "definitions_id",
            "parameter_id",
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
                or not all(
                    isinstance(item, int) and not isinstance(item, bool)
                    for item in shape
                )
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
            if not isinstance(binding, Mapping) or set(binding) != {
                "parameter",
                "value",
            }:
                raise EvidenceIntegrityError(
                    "parameter binding is malformed",
                    stage="result_decode",
                )
            parameter = self._decode_parameter_ref(binding["parameter"])
            values[parameter] = self._decode_parameter_value(
                parameter, binding["value"]
            )
        allowed = tuple(self._decode_parameter_ref(value) for value in authorizations)
        parameters = ParameterSet(values, allow_extrapolation=allowed)
        if parameters._record() != record:
            raise EvidenceIntegrityError(
                "decoded parameter set differs from its verified record",
                stage="result_decode",
            )
        return parameters


def _decode_hb_effective_sources(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise EvidenceIntegrityError(
            "HB effective-source evidence is malformed", stage="result_decode"
        )
    decoded: list[Mapping[str, object]] = []
    identities: set[tuple[str, tuple[int, ...]]] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "drive_id",
            "mode",
            "coefficient",
            "generated_conjugate",
            "backend_binding",
            "injection_map_sha256",
        }:
            raise EvidenceIntegrityError(
                "HB effective-source evidence is malformed", stage="result_decode"
            )
        drive_id = item.get("drive_id")
        mode = item.get("mode")
        conjugate = item.get("generated_conjugate")
        backend = item.get("backend_binding")
        if (
            not isinstance(drive_id, str)
            or not drive_id
            or not isinstance(mode, list)
            or any(
                not isinstance(entry, int) or isinstance(entry, bool) for entry in mode
            )
            or not isinstance(conjugate, Mapping)
            or set(conjugate) != {"mode", "coefficient"}
            or not isinstance(conjugate.get("mode"), list)
            or any(
                not isinstance(entry, int) or isinstance(entry, bool)
                for entry in conjugate["mode"]
            )
            or not isinstance(backend, Mapping)
            or set(backend)
            != {
                "representative_mode",
                "representative_index",
                "coefficient",
                "coefficient_convention",
            }
            or not isinstance(backend.get("representative_mode"), list)
            or any(
                not isinstance(entry, int) or isinstance(entry, bool)
                for entry in backend["representative_mode"]
            )
            or not isinstance(backend.get("representative_index"), int)
            or isinstance(backend.get("representative_index"), bool)
            or backend["representative_index"] < 0
            or backend.get("coefficient_convention")
            != "exp_plus_i_m_dot_omega_t_josephsoncircuits_source"
            or not _is_sha256_text(item.get("injection_map_sha256"))
        ):
            raise EvidenceIntegrityError(
                "HB effective-source identity is malformed", stage="result_decode"
            )
        key = (drive_id, tuple(mode))
        if key in identities:
            raise EvidenceIntegrityError(
                "HB effective-source identity is duplicated", stage="result_decode"
            )
        identities.add(key)
        decoded.append(
            {
                "drive_id": drive_id,
                "mode": tuple(mode),
                "coefficient": complex_quantity_from_envelope(
                    item["coefficient"], registry=units.registry
                ),
                "generated_conjugate": {
                    "mode": tuple(conjugate["mode"]),
                    "coefficient": complex_quantity_from_envelope(
                        conjugate["coefficient"], registry=units.registry
                    ),
                },
                "backend_binding": {
                    "representative_mode": tuple(backend["representative_mode"]),
                    "representative_index": backend["representative_index"],
                    "coefficient": complex_quantity_from_envelope(
                        backend["coefficient"], registry=units.registry
                    ),
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
    if (
        not isinstance(axes, list)
        or len(axes) != 3
        or not isinstance(axes[index], Mapping)
    ):
        raise EvidenceIntegrityError(
            "HB matrix axes are malformed", stage="result_decode"
        )
    axis = axes[index]
    values = axis.get("values")
    if axis.get("kind") != kind or not isinstance(values, list) or not values:
        raise EvidenceIntegrityError(
            "HB matrix channel axis is malformed", stage="result_decode"
        )
    channels: list[tuple[str, tuple[int, ...]]] = []
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {"coordinate", "mode"}:
            raise EvidenceIntegrityError(
                "HB matrix channel label is malformed", stage="result_decode"
            )
        coordinate = value.get("coordinate")
        mode = value.get("mode")
        if (
            not isinstance(coordinate, str)
            or not coordinate
            or not isinstance(mode, list)
            or any(
                not isinstance(entry, int) or isinstance(entry, bool) for entry in mode
            )
        ):
            raise EvidenceIntegrityError(
                "HB matrix channel label is malformed", stage="result_decode"
            )
        channels.append((coordinate, tuple(mode)))
    if len(set(channels)) != len(channels):
        raise EvidenceIntegrityError(
            "HB matrix channel labels are duplicated", stage="result_decode"
        )
    return tuple(channels)


def _decode_hb_mode_axis(artifact: object, *, kind: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(artifact, Mapping):
        raise EvidenceIntegrityError(
            "HB mode artifact is malformed", stage="result_decode"
        )
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
        if not isinstance(value, list) or any(
            not isinstance(entry, int) or isinstance(entry, bool) for entry in value
        ):
            raise EvidenceIntegrityError(
                "HB mode-axis tuple is malformed", stage="result_decode"
            )
        modes.append(tuple(value))
    if len(set(modes)) != len(modes):
        raise EvidenceIntegrityError(
            "HB mode axis repeats a tuple", stage="result_decode"
        )
    return tuple(modes)


def _decode_hb_reconciliation(value: Mapping[str, object]) -> ReconciliationEvidence:
    comparable = value.get("comparable")
    expected = {
        "comparable",
        "reason",
        "last_comparable_ancestor",
        "normalization",
        "evidence_sha256",
        *(("residual_f64", "coordinate_projection") if comparable is True else ()),
    }
    if (
        isinstance(comparable, bool)
        and set(value) == expected
        and value.get("normalization") == "backend_photon_flux_to_scnsim_power_wave"
        and _is_sha256_text(value.get("last_comparable_ancestor"))
        and _is_sha256_text(value.get("evidence_sha256"))
        and (
            (comparable and value.get("reason") is None)
            or (
                not comparable
                and value.get("reason")
                in {
                    "topology",
                    "load_or_ptc",
                    "reference_plane",
                    "reference_matrix",
                    "signed_frequency_grid",
                    "channel_basis",
                    "normalization",
                }
            )
        )
    ):
        try:
            residual = float64_from_hex(value["residual_f64"]) if comparable else None
        except (TypeError, ValueError) as error:
            raise EvidenceIntegrityError(
                "HB reconciliation residual is malformed", stage="result_decode"
            ) from error
        if residual is None or (math.isfinite(residual) and residual >= 0.0):
            return _verified_result(
                ReconciliationEvidence,
                comparable=comparable,
                reason=value.get("reason"),
                last_comparable_ancestor=value["last_comparable_ancestor"],
                residual=residual,
                evidence_sha256=value["evidence_sha256"],
            )
    raise EvidenceIntegrityError(
        "HB reconciliation evidence is malformed", stage="result_decode"
    )


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
    if not isinstance(projection, Mapping) or set(projection) != {
        "shape",
        "values_f64",
    }:
        raise EvidenceIntegrityError(
            "HB reconciliation coordinate projection is malformed",
            stage="result_decode",
        )
    shape = projection.get("shape")
    values = projection.get("values_f64")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in shape
        )
        or not isinstance(values, list)
        or len(values) != shape[0] * shape[1]
    ):
        raise EvidenceIntegrityError(
            "HB reconciliation coordinate projection is malformed",
            stage="result_decode",
        )
    try:
        q = np.asarray(
            [float64_from_hex(value) for value in values], dtype=np.float64
        ).reshape(tuple(shape))
    except (TypeError, ValueError) as error:
        raise EvidenceIntegrityError(
            "HB reconciliation coordinate projection is malformed",
            stage="result_decode",
        ) from error
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
        raise EvidenceIntegrityError(
            "HB reconciliation matrices disagree with their coordinate projection",
            stage="result_decode",
        )
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
                                    * native[
                                        frequency,
                                        native_output * mode_count + output_mode,
                                        native_input * mode_count + input_mode,
                                    ]
                                    * q[input_coordinate, native_input]
                                )
                        projected[output, input_] = value
        numerator = max(
            sum(
                abs(selected[frequency, row, column] - projected[row, column])
                for column in range(projected.shape[1])
            )
            for row in range(projected.shape[0])
        )
        denominator = max(
            sum(
                abs(selected[frequency, row, column]) + abs(projected[row, column])
                for column in range(projected.shape[1])
            )
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
    if not math.isfinite(residual) or float64_hex(residual) != evidence.get(
        "residual_f64"
    ):
        raise EvidenceIntegrityError(
            "HB reconciliation residual does not reproduce selected S from backend-native S",
            stage="result_decode",
        )
