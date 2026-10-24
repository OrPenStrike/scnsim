"""Compiler-only realization of immutable captured Plan declarations.

JAX explanations consume sparse Python lowering/View realization without a
numerical backend. This route allocates no attempt and mutates no analysis state."""

from __future__ import annotations

from collections.abc import Mapping
from ..canonical import canonical_json_bytes, sha256_hex


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
        "direct_hb_capability": {"backend": "jax", "direct": "sparse_cpu_superlu"},
    }
