"""Result envelope dispatcher and operation-specific payload validation."""

from __future__ import annotations

import math
from collections.abc import Mapping

from ..records import BaselineCheckpoint
from .common import (
    _integrity,
    _valid_sha,
    _verify_discretization,
    _verify_parameter_set_document,
)
from .optimization import _verify_checkpoint_primary_lineage
from .requests import _verify_v1_lineage
from .result_direct import _verify_direct_response_result, _verify_operator_result
from .result_hb import _verify_hb_batch_result
from .result_optimization import _verify_optimization_result
from .result_scalars import _verify_scalar_result_payload

def _verify_result_document(
    result: Mapping[str, object],
    request: Mapping[str, object],
    request_sha256: str,
    attempt_sha256: str,
    plan: Mapping[str, object],
    *,
    optimization_checkpoint: BaselineCheckpoint | None = None,
) -> None:
    """Close one schema-version 2 single or batch result."""

    if result.get("result_kind") == "parameter_sweep":
        _verify_parameter_sweep_result(result, request, request_sha256, attempt_sha256)
        return
    common = {
        "schema", "schema_version", "result_kind", "request_sha256", "attempt_sha256",
        "parameters", "parameters_sha256", "ref_lineage",
    }
    if (
        result.get("schema") != "scnsim.result"
        or result.get("schema_version") != 2
        or result.get("request_sha256") != request_sha256
        or result.get("attempt_sha256") != attempt_sha256
        or not common <= set(result)
    ):
        raise _integrity("Result envelope does not bind its request and attempt.")
    parameters = result.get("parameters")
    _verify_parameter_set_document(parameters)
    from ...authoring.identity import canonical_parameters_sha256

    if result.get("parameters_sha256") != canonical_parameters_sha256(parameters):
        raise _integrity("Result parameter identity is malformed.")
    source = request.get("parameter_source")
    if not isinstance(source, Mapping) or source.get("kind") != "point" or parameters != source.get("parameters"):
        raise _integrity("Single-point Result does not bind its requested point.")
    _verify_v1_lineage(result.get("ref_lineage"), plan)
    _verify_discretization(result.get("discretization"), plan)
    if result.get("result_kind") == "optimization":
        if optimization_checkpoint is None:
            raise _integrity("Optimization Result lacks its verified baseline checkpoint.")
        if result.get("baseline") != optimization_checkpoint.checkpoint.get("baseline"):
            raise _integrity("Optimization Result baseline differs from its verified checkpoint.")
        primary_lineage = _verify_checkpoint_primary_lineage(
            optimization_checkpoint.checkpoint.get("baseline"),
            request_view=request.get("view"),
            plan=plan,
        )
        if result.get("ref_lineage") != primary_lineage:
            raise _integrity(
                "Optimization Result primary lineage differs from its verified checkpoint."
            )
    scientific_result = dict(result)
    for field in ("parameters", "parameters_sha256", "ref_lineage", "discretization"):
        scientific_result.pop(field)
    scientific_result["schema_version"] = 1
    scientific_request = dict(request)
    scientific_request["schema_version"] = 1
    scientific_request["ref_lineage"] = result["ref_lineage"]
    scientific_request["parameters"] = parameters
    scientific_request.pop("view", None)
    scientific_request.pop("parameter_source", None)
    _verify_single_result_document(
        scientific_result, scientific_request, request_sha256, attempt_sha256, plan,
        discretization=result.get("discretization"),
    )

def _verify_parameter_sweep_result(
    result: Mapping[str, object],
    request: Mapping[str, object],
    request_sha256: str,
    attempt_sha256: str,
) -> None:
    expected = {
        "schema", "schema_version", "result_kind", "request_sha256", "attempt_sha256",
        "parameter_source_sha256", "point_count", "chunk_size", "manifest", "chunks",
    }
    source = request.get("parameter_source")
    from ...canonical import sha256_hex

    if (
        set(result) != expected
        or result.get("schema") != "scnsim.result"
        or result.get("schema_version") != 2
        or result.get("result_kind") != "parameter_sweep"
        or result.get("request_sha256") != request_sha256
        or result.get("attempt_sha256") != attempt_sha256
        or not isinstance(source, Mapping)
        or source.get("kind") not in {"grid", "points"}
        or result.get("parameter_source_sha256") != sha256_hex(source)
        or result.get("chunk_size") != 64
        or not isinstance(result.get("point_count"), int)
        or isinstance(result.get("point_count"), bool)
        or result["point_count"] < 1
        or not isinstance(result.get("manifest"), Mapping)
        or not isinstance(result.get("chunks"), list)
    ):
        raise _integrity("Parameter-sweep Result envelope is malformed.")
    expected_count = (
        math.prod(source["shape"])
        if source["kind"] == "grid"
        else len(source["points"])
    )
    if result["point_count"] != expected_count:
        raise _integrity("Parameter-sweep point count disagrees with its source.")
    manifest = result["manifest"]
    if (
        set(manifest) != {"id", "path", "sha256", "media_type", "byte_length"}
        or manifest.get("id") != "parameter_points"
        or manifest.get("path") != "artifacts/parameter_points.manifest.json"
        or manifest.get("media_type") != "application/json"
        or not isinstance(manifest.get("byte_length"), int)
        or isinstance(manifest.get("byte_length"), bool)
        or manifest["byte_length"] < 1
    ):
        raise _integrity("Parameter-sweep manifest link is malformed.")
    _valid_sha(manifest.get("sha256"))
    chunks = result["chunks"]
    expected_chunks = (expected_count + 63) // 64
    if len(chunks) != expected_chunks:
        raise _integrity("Parameter-sweep chunk count is malformed.")
    for index, chunk in enumerate(chunks):
        first = index * 64
        count = min(64, expected_count - first)
        if (
            not isinstance(chunk, Mapping)
            or set(chunk) != {"chunk_ordinal", "first_point", "point_count", "path", "sha256"}
            or chunk.get("chunk_ordinal") != index
            or chunk.get("first_point") != first
            or chunk.get("point_count") != count
            or chunk.get("path") != f"artifacts/parameter_points/chunks/{index:06d}.json"
        ):
            raise _integrity("Parameter-sweep chunk link is malformed.")
        _valid_sha(chunk.get("sha256"))

def _verify_single_result_document(
    result: Mapping[str, object],
    request: Mapping[str, object],
    request_sha256: str,
    attempt_sha256: str,
    plan: Mapping[str, object],
    *,
    discretization: object = None,
) -> None:
    """Verify the unchanged inner scientific result records."""
    spec = request.get("spec")
    kind = (
        "direct_response" if request.get("operation") == "solve_direct"
        else "hb_batch" if request.get("operation") == "solve_hb"
        else "optimization" if request.get("operation") == "optimize_direct"
        else spec.get("type") if request.get("operation") == "evaluate_direct" and isinstance(spec, dict)
        else None
    )
    common = {"schema", "schema_version", "result_kind", "request_sha256", "attempt_sha256"}
    if (
        result.get("schema") != "scnsim.result"
        or result.get("schema_version") != 1
        or result.get("result_kind") != kind
        or result.get("request_sha256") != request_sha256
        or result.get("attempt_sha256") != attempt_sha256
    ):
        raise _integrity("Result envelope does not match its request and attempt.")
    if kind == "hb_batch":
        _verify_hb_batch_result(result, request, plan, discretization=discretization)
    elif kind == "direct_response":
        _verify_direct_response_result(result, request, plan, common)
    elif kind == "operator":
        _verify_operator_result(result, request, plan, common)
    elif kind == "optimization":
        _verify_optimization_result(result, request, plan, common)
    else:
        _verify_scalar_result_payload(result, request, plan, kind, common)
