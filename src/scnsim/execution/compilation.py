"""Compiler-only realization of immutable captured Plan declarations.

Temporary documents feed the real Julia compiler without allocating an attempt
or mutating an analysis workspace. Returned evidence binds the resolved point."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from ..authoring.snapshot import ResolvedPlanPoint, freeze
from ..canonical import canonical_json_bytes, sha256_hex
from ..authoring.identity import (
    canonical_expanded_graph_sha256,
    canonical_plan_snapshot,
    canonical_resolved_plan_point,
)
from ..workspace.artifacts import _error_from_record, _validated_failure_record
from ..errors import BackendProtocolError
from .identity import _runtime_identity_base
from .preparation import prepare_runtime
from .process import run_compiler_audit, run_preflight


def _run_preflight(
    plan_document: Mapping[str, object],
    plan_bytes: bytes,
    request: Mapping[str, object],
) -> Mapping[str, object]:
    prepared = prepare_runtime(feature=f"CircuitRun.explain compiler preflight ({request['operation']})")
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
            _validated_failure_record(
                compiled.get("failure"), request["operation"],
                request=request, plan=plan_document,
            )
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
    prepared = prepare_runtime(feature="compiled schematic")
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
        "expanded_branch_rows", "discretization", "c_matrix", "k_matrix", "g_matrix", "ports",
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
        or any(not isinstance(compiled.get(field), list) for field in ("resolved_bindings", "expanded_branch_rows", "discretization"))
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
