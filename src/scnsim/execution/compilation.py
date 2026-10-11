"""Compiler-only realization of immutable captured Plan declarations.

JAX explanations consume sparse Python lowering/View realization without a
numerical backend. Explicit Julia preflight and compiled schematics retain their
native compiler. Neither route allocates an attempt or mutates analysis state."""

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
from .preparation import prepare_runtime, _native_supervisor_scope
from .process import run_compiler_audit, run_preflight


def _run_preflight(
    plan_document: Mapping[str, object],
    plan_bytes: bytes,
    request: Mapping[str, object],
    *, native_supervisor=None,
) -> Mapping[str, object]:
    with _native_supervisor_scope(native_supervisor) as supervisor:
        prepared = prepare_runtime(feature=f"CircuitRun.explain compiler preflight ({request['operation']})", native_supervisor=supervisor)
        with tempfile.TemporaryDirectory(prefix="scnsim-preflight-") as temporary:
            plan_path = Path(temporary) / "plan.json"
            request_path = Path(temporary) / "request.json"
            plan_path.write_bytes(plan_bytes)
            request_path.write_bytes(canonical_json_bytes(request))
            compiled = run_preflight(
                prepared,
                plan_path=plan_path.resolve(),
                request_path=request_path.resolve(),
                native_supervisor=supervisor,
            )
        if compiled.get("schema") == "scnsim.preflight_failure":
            raise _error_from_record(
                _validated_failure_record(
                    compiled.get("failure"), request["operation"],
                    request=request, plan=plan_document,
                )
            )
        return compiled


def _run_jax_preflight(
    plan_document: Mapping[str, object],
    request: Mapping[str, object],
) -> Mapping[str, object]:
    """Expose actual sparse compiler evidence, without JAX initialization/solving.

    Lower once at the effective request point; realize each distinct objective
    View once while retaining objective/term order in its references. This is
    preparation evidence, never a fabricated numerical or native Julia receipt.
    """
    from ..compilation.compiler import compile_model, parameter_key, parameter_values
    from ..numeric_encoding import array_record, record_document
    from ..compilation.views import realize_view
    from ..canonical import float64_hex
    from .quantities import expression_leaves

    source = request["parameter_source"]
    point = source["parameters"]
    spec = request["spec"]
    authorized = {parameter_key(row) for row in point.get("allow_extrapolation", ())}
    if request["operation"] == "optimize_direct":
        authorized.update(parameter_key(row) for row in spec.get("allow_extrapolation", ()))
    raw = compile_model(plan_document, parameter_values(point), authorized=authorized,
                        preparation_cache={})
    raw_evidence = record_document(raw.evidence_bytes)

    def sparse(matrix, unit="dimensionless"):
        return {"format": "coo", "unit": unit, "shape": list(matrix.shape),
                "rows": matrix.rows.tolist(), "cols": matrix.cols.tolist(),
                "values": array_record(matrix.values)}

    views = {}

    def realize(declaration):
        key = sha256_hex(canonical_json_bytes(declaration))
        if key not in views:
            view = realize_view(raw, declaration)
            model = view.model
            selected = set(view.selected_indices)
            views[key] = {
                "declaration": declaration,
                "lineage": record_document(view.lineage_bytes),
                "node_order": list(model.node_ids),
                "original_node_order": list(view.original_node_ids),
                "coordinates": list(view.coordinates),
                "terminal_ids": list(view.terminal_ids),
                "selected_indices": list(view.selected_indices),
                "eliminated_indices": [i for i in range(len(model.node_ids)) if i not in selected],
                "port_realizable": view.port_realizable,
                "c_matrix": sparse(model.C, "farad"), "k_matrix": sparse(model.K, "1 / henry"),
                "g_matrix": sparse(model.G, "siemens"),
                "ports": {"ids": list(model.port_ids), "selector": sparse(model.B),
                          "reference_matrix": {**array_record(model.R), "unit": "ohm"},
                          "load_mask": array_record(model.M)},
                "series_rl": [{"incidence": sparse(block.incidence),
                               "resistance": {**array_record(block.resistance), "unit": "ohm"},
                               "inductance": {**array_record(block.inductance), "unit": "henry"}}
                              for block in model.series_rl],
                "coordinate_port_map": array_record(view.coordinate_port_map),
                "selected_boundary": {
                    name: sparse(value) if name == "Bk" else {
                        **array_record(value), "unit": {"selected_map": "dimensionless",
                        "Rk": "ohm", "Dk": "ohm ** 0.5", "Go": "siemens"}[name]}
                    for name in ("selected_map", "Bk", "Rk", "Dk", "Go")
                    if (value := getattr(view, name)) is not None
                },
                "leaf_evidence": record_document(model.evidence_bytes),
            }
        return key

    primary_key = realize(request["view"])
    objective_views = []
    if request["operation"] == "optimize_direct":
        for objective_ordinal, objective in enumerate(spec["objectives"]):
            for term_ordinal, selector in enumerate(expression_leaves(objective["quantity"])):
                objective_views.append({"objective_ordinal": objective_ordinal,
                                        "term_ordinal": term_ordinal,
                                        "selector": selector,
                                        "view_id": realize(selector["view"])})

    def quantity(value, unit, dimension):
        return {"type": "quantity_f64", "si_unit": unit, "dimensionality": dimension,
                "si_value_f64": float64_hex(value)}

    grids = []
    for grid in raw_evidence["discretization"]:
        row = {"component_path": grid["component_path"], "kind": grid["kind"],
               "n_sections": grid["n_sections"],
               "length": quantity(grid["length_m"], "meter", "length"),
               "dx": quantity(grid["dx_m"], "meter", "length")}
        if "hmax_m" in grid:
            row["hmax"] = quantity(grid["hmax_m"], "meter", "length")
        if "modal_velocities_m_s" in grid:
            row["modal_velocities"] = [quantity(v, "meter / second", "velocity")
                                       for v in grid["modal_velocities_m_s"]]
        if "policy" in grid:
            row["policy"] = grid["policy"]
        grids.append(row)
    primary = views[primary_key]
    return {
        "schema": "scnsim.jax_preflight", "schema_version": 1,
        "plan_sha256": request["plan_sha256"],
        "request_sha256": sha256_hex(canonical_json_bytes(request)),
        "runtime_semantic": request["runtime_semantic"],
        "lowering_precision": "float64", "matrix_order": "realized_view",
        "node_order": primary["node_order"],
        "c_matrix": primary["c_matrix"], "k_matrix": primary["k_matrix"],
        "g_matrix": primary["g_matrix"], "ports": primary["ports"],
        "discretization": grids, "resolved_bindings": raw_evidence["resolved_fields"],
        "expanded_branch_rows": raw_evidence["branches"],
        "primary_view_id": primary_key, "views": views,
        "root_preflight": {"spec": spec, "view_id": primary_key,
                           "algorithm_id": request["runtime_semantic"]["algorithm_id"]},
        "optimization_preflight": {"objective_views": objective_views},
        "direct_hb_capability": {"backend": "jax", "direct": "sparse_cpu_superlu",
                                 "hb": "requires_explicit_julia"},
    }


def _compiled_schematic_evidence(point: ResolvedPlanPoint, *, native_supervisor=None) -> Mapping[str, object]:
    """Compile one immutable point without a Run, View, or analysis workspace."""
    with _native_supervisor_scope(native_supervisor) as supervisor:

        if not isinstance(point, ResolvedPlanPoint):
            raise TypeError("_compiled_schematic_evidence() requires ResolvedPlanPoint")
        plan_document = canonical_plan_snapshot(point.snapshot)
        plan_bytes = canonical_json_bytes(plan_document)
        plan_sha = sha256_hex(plan_bytes)
        point_document = canonical_resolved_plan_point(point, plan_sha256=plan_sha)
        point_bytes = canonical_json_bytes(point_document)
        prepared = prepare_runtime(feature="compiled schematic", native_supervisor=supervisor)
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
                    native_supervisor=supervisor,
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
