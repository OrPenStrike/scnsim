"""Candidate-specific local sparse congruence and selected Port boundary.

Pair declarations consume the current coalesced C. Only the local inverse and
Port boundary are dense; no global transformation/incidence is materialized.
"""

from __future__ import annotations

from dataclasses import replace
import json
import numpy as np

from ..canonical import canonical_json_bytes
from ..errors import CompilerInvariantError, DirectResponseFormationError, PortRealizabilityError
from .compiler import backward_residual
from .models import CompiledModel, RealizedView, SeriesRL
from .prepared import record_bytes, record_document


def _view_template_key(model: CompiledModel, declaration: dict, template_cache: dict | None):
    if template_cache is None:
        return None
    structure = template_cache.get("_active_compiled_model_structure")
    if structure is None:
        return None
    declaration_bytes = canonical_json_bytes(declaration)
    return ("view_topology", structure, declaration_bytes.decode("utf-8"))


def _transform_structure(node_ids: tuple[str, ...], declaration_transform: dict, *,
                         i: int, j: int, common: str, differential: str) -> dict:
    """Build only the immutable routing template for one reached transform."""
    left, right = declaration_transform["input_coordinates"]
    next_nodes = tuple(name for name in node_ids if name not in (left, right)) + (common, differential)
    next_indices = {name: index for index, name in enumerate(next_nodes)}
    routes = []
    for row, name in enumerate(node_ids):
        if row == i:
            routes.append(("left", next_indices[common], next_indices[differential]))
        elif row == j:
            routes.append(("right", next_indices[common], next_indices[differential]))
        else:
            routes.append(("single", next_indices[name]))
    return {"left": left, "right": right, "common": common,
            "differential": differential, "i": i, "j": j,
            "next_nodes": next_nodes, "routes": tuple(routes)}


def _weighted_mapping(template: dict, alpha: float, beta: float) -> tuple:
    mapping = []
    for route in template["routes"]:
        if route[0] == "left":
            mapping.append(((route[1], 1.0), (route[2], beta)))
        elif route[0] == "right":
            mapping.append(((route[1], 1.0), (route[2], -alpha)))
        else:
            mapping.append(((route[1], 1.0),))
    return tuple(mapping)


def spd_root(matrix: np.ndarray) -> np.ndarray:
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise PortRealizabilityError("reference matrix must be positive definite", stage="reference_matrix") from error
    values, vectors = np.linalg.eigh(matrix)
    if not np.all(np.isfinite(values)) or not np.all(values > 0):
        raise PortRealizabilityError("reference matrix has no finite positive principal spectrum", stage="reference_matrix")
    root = (vectors * np.sqrt(values)) @ vectors.T
    residual = backward_residual(root, root, matrix)
    if not np.isfinite(residual) or residual > 256 * (len(matrix) + 1) * np.finfo(np.float64).eps:
        raise DirectResponseFormationError("reference square-root reconstruction exceeds the normalized residual contract", stage="reference_matrix")
    return (root + root.T) / 2


def realize_view(model: CompiledModel, declaration: dict, *, template_cache: dict | None = None) -> RealizedView:
    original_coordinates = json.loads(model.evidence_bytes)["original_coordinates"]
    p = len(model.port_ids)
    mask = model.M.copy()
    ptc = declaration.get("ptc")
    if ptc:
        for port in ptc["selected_ports"]:
            mask[model.port_ids.index(port)] = 0
    working = replace(model, M=mask)
    template_key = _view_template_key(model, declaration, template_cache)
    view_template = None if template_key is None else template_cache.get(template_key)
    if view_template is None:
        view_template = {"transforms": {}}
        if template_key is not None:
            template_cache[template_key] = view_template
    names = list(original_coordinates)
    maps = {name: model.B.row(model.node_ids.index(name)) for name in names}
    logical = {port: np.eye(p)[i] for i, port in enumerate(model.port_ids)}
    coordinate_map = np.zeros((len(model.node_ids), p))
    for column in range(p):
        entries = model.B.column_rows(column)
        if len(entries) != 1:
            raise PortRealizabilityError("logical Port must select one physical coordinate", stage="selected_network")
        coordinate_map[entries[0], column] = 1
    transforms = []
    for transform_index, declaration_transform in enumerate(declaration.get("transforms", [])):
        left, right = declaration_transform["input_coordinates"]
        outputs = declaration_transform.get("output_coordinates")
        common, differential = outputs if outputs is not None else (declaration_transform["common_id"], declaration_transform["differential_id"])
        i, j = working.node_ids.index(left), working.node_ids.index(right)
        cl, cr = working.C.entry(i, i) + working.C.entry(i, j), working.C.entry(j, j) + working.C.entry(i, j)
        if not np.isfinite(cl) or not np.isfinite(cr) or cl < 0 or cr < 0 or cl + cr <= 0:
            raise PortRealizabilityError("external capacitance cut does not define pair weights", stage="transform_pair")
        alpha, beta = cl / (cl + cr), cr / (cl + cr)
        transform_template = view_template["transforms"].get(transform_index)
        if transform_template is None:
            transform_template = _transform_structure(
                working.node_ids, declaration_transform, i=i, j=j,
                common=common, differential=differential,
            )
            view_template["transforms"][transform_index] = transform_template
        # Refresh candidate weights over the immutable row-routing template.
        mapping = _weighted_mapping(transform_template, alpha, beta)
        next_nodes = transform_template["next_nodes"]
        local_A = np.asarray([[alpha, beta], [1.0, -1.0]])
        local_T = np.asarray([[1.0, beta], [1.0, -alpha]])
        reconstruction = backward_residual(local_A, local_T, np.eye(2))
        if not np.isfinite(reconstruction) or reconstruction > 256 * (len(working.node_ids) + 1) * np.finfo(np.float64).eps:
            raise CompilerInvariantError("floating-pair reconstruction exceeds the normalized residual contract", stage="transform_pair")
        transformed_blocks = tuple(SeriesRL(block.incidence.transform_rows(mapping, len(next_nodes)),
                                            block.resistance, block.inductance) for block in working.series_rl)
        evidence = record_document(working.evidence_bytes)
        for row in evidence["branches"]:
            for key in ("positive", "negative"):
                if key in row:
                    old = np.asarray(row[key])
                    transformed = np.zeros(len(next_nodes))
                    for source, targets in enumerate(mapping):
                        for target, weight in targets:
                            transformed[target] += weight * old[source]
                    row[key] = transformed.tolist()
        working = replace(working, node_ids=tuple(next_nodes),
                          C=working.C.congruence(mapping, len(next_nodes)),
                          K=working.K.congruence(mapping, len(next_nodes)),
                          G=working.G.congruence(mapping, len(next_nodes)),
                          B=working.B.transform_rows(mapping, len(next_nodes)), series_rl=transformed_blocks,
                          evidence_bytes=record_bytes(evidence))
        maps[common] = alpha * maps[left] + beta * maps[right]
        maps[differential] = maps[left] - maps[right]
        del maps[left], maps[right]
        names = [name for name in names if name not in (left, right)] + [common, differential]
        transforms.append({"input_coordinates": [left, right], "output_coordinates": [common, differential],
                           "weights": [float(alpha), float(beta)]})
    retain = declaration.get("retain")
    terminals = tuple(retain["retained_coordinates"] if retain else model.port_ids)
    terminal_maps = maps if retain else logical
    selected_map = np.vstack([terminal_maps[name] for name in terminals]) if terminals else np.zeros((0, p))
    selected_indices = tuple(working.node_ids.index(name) for name in terminals) if retain else tuple(
        int(model.B.column_rows(model.port_ids.index(port))[0]) for port in terminals) if not transforms else ()
    realizable = bool(terminals) and len(terminals) <= p and np.linalg.matrix_rank(selected_map) == len(terminals)
    boundary = {}
    if realizable:
        Dp = spd_root(working.R)
        Rk = selected_map @ working.R @ selected_map.T
        Dk = spd_root(Rk)
        Qk = np.linalg.solve(Dk, selected_map @ Dp)
        Po = np.eye(p) - Qk.T @ Qk
        Dp_inv = np.linalg.solve(Dp, np.eye(p))
        boundary = dict(selected_map=selected_map, Bk=working.B.right_multiply(selected_map.T),
                        Rk=Rk, Dk=Dk, Go=Dp_inv @ Po @ np.diag(working.M) @ Po @ Dp_inv)
    lineage = {"declaration": declaration, "transforms": transforms, "port_realizable": bool(realizable)}
    grids = json.loads(model.evidence_bytes)["discretization"]
    signature = (working.node_ids, working.port_ids, terminals, selected_indices,
                 tuple(block.inductance.shape for block in working.series_rl),
                 tuple((tuple(grid["component_path"]), grid["n_sections"]) for grid in grids))
    return RealizedView(working, tuple(names), terminals, coordinate_map, selected_indices,
                        bool(realizable), lineage_bytes=record_bytes(lineage), signature=signature,
                        original_node_ids=model.node_ids, **boundary)
