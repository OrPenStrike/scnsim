"""Materialize lazy parameter-sweep Result views from exact artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..canonical import canonical_json_bytes
from ..authoring.identity import canonical_parameters_sha256
from ..execution.prepared import _coordinate_binding_key, _encode_scalar_expression, _quantity_coordinates
from ..errors import EvidenceIntegrityError
from ..specs import QuantitySelector
from ..workspace.artifacts import (
    _VerifiedEvidenceLease, _error_from_record, _read_canonical_artifact_json,
)
from .base import ParameterPointIdentity, ResultIdentity
from .factory import _verified_result
from .sweep import (
    ParameterPointOutcome, ParameterSweepResult, _parameter_sweep_result,
    _point_accessor, _point_outcome,
)

def _decode_parameter_sweep_operation(
    decoder,
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
        parameters = decoder._decode_parameter_set(parameters_record)
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
                    value = decoder._decode_result(
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
            decoder._decode_parameter_ref(axis["parameter"]) for axis in source["axes"]
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
        "operator_element_root": "operator_element_root_projection",
        "hybridized_pole": "hybridized_pole_projection",
        "transfer_zero": "transfer_zero_projection",
        "residue_normalized_coupling": "residue_coupling_projection",
        "response_element": "response_element_projection",
    }.get(request_spec.get("type"))
    projections = {
        "diagonal_root": ("frequency", "linewidth"),
        "operator_element_root": ("frequency",),
        "hybridized_pole": ("frequency", "linewidth"),
        "transfer_zero": ("frequency",),
        "residue_normalized_coupling": ("real", "imag", "magnitude"),
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
                        else decoder._coordinate_id(coordinate)
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
