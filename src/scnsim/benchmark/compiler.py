"""Independent Plan-v2 physical lowering; no Julia discovery or preflight.

Connectivity supplies structural slots, including zero-valued coefficients.
Only local conductor/coupling support is dense; no nodal dense forms are built.
"""

from __future__ import annotations

import math
import numpy as np

from ..canonical import canonical_json_bytes, sha256_hex
from ..errors import CompilerInvariantError, InvalidCandidatePhysicalParameter
from .mesh import prepare_rlgc, quantity, realize_line
from .models import CompiledModel, MeshSpec, SeriesRL, SparseMatrix
from .prepared import record_bytes


def backward_residual(A: np.ndarray, X: np.ndarray, B: np.ndarray) -> float:
    """Existing SCNSim componentwise normalized solve residual."""
    numerator = np.max(np.abs(A @ X - B))
    denominator = np.max(np.abs(A) @ np.abs(X) + np.abs(B))
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.inf
    return 0 if numerator == denominator == 0 else math.inf if denominator == 0 else float(numerator / denominator)


def parameter_key(record: dict) -> tuple[str, str]:
    return record["definitions_id"], record["parameter_id"]


def parameter_values(record: dict) -> dict:
    return {parameter_key(b["parameter"]): b["value"] if b["value"]["type"] == "rlgc" else quantity(b["value"])
            for b in record["bindings"]}


def resolve_fields(plan: dict, values: dict, authorized: set = frozenset()) -> tuple[dict, list]:
    fields, evidence = {}, []
    for leaf in plan["physical_leaves"]:
        for item in leaf["fields"]:
            binding = item["binding"]
            kind = binding["kind"]
            if kind == "constant":
                source = binding["value"]
                value = source if source["type"] == "rlgc" else quantity(source)
            elif kind == "ref":
                value = values[parameter_key(binding["parameter"])]
            elif kind == "affine":
                key = parameter_key(binding["input"])
                value = values[key]
                lower, upper = map(quantity, binding["support"])
                if not lower <= value <= upper:
                    evidence.append({"parameter": binding["input"], "consumer_target": {"path": leaf["path"], "field": item["id"]},
                                     "input_si": value, "support_si": [lower, upper], "authorized": key in authorized})
                    if key not in authorized:
                        raise InvalidCandidatePhysicalParameter("affine input is outside declared support", stage="affine_support", evidence=evidence[-1])
                value = quantity(binding["slope"]) * value + quantity(binding["intercept"])
            else:
                raise CompilerInvariantError("unknown physical binding", stage="compile")
            fields[tuple(leaf["path"]), item["id"]] = value
    return fields, evidence


def internal_node(leaf: dict, sections: int, station: int, conductor: str) -> str:
    record = {"schema": "scnsim.line_station", "schema_version": 1,
              "component_path": leaf["path"], "station": station, "conductor": conductor}
    if "discretization" in leaf["model_metadata"]:
        record["n_sections"] = sections
    return "internal-" + sha256_hex(record)


def compile_model(plan: dict, values: dict, *, mesh: MeshSpec = MeshSpec(), authorized: set = frozenset(), preparation_cache: dict | None = None) -> CompiledModel:
    fields, extrapolation = resolve_fields(plan, values, authorized)
    connectivity = plan["connectivity"]
    ground = connectivity["canonical_ground"]
    endpoints = {(tuple(row["path"]), row["pin"]): row["net"] for row in connectivity["physical_endpoints"]}
    nodes = [row["compiler_node_id"] for row in connectivity["node_coordinates"]]
    grids = {}
    for leaf in plan["physical_leaves"]:
        if leaf["model"] == "transmission_line":
            path = tuple(leaf["path"])
            rlgc = fields[path, "rlgc"]
            grids[path] = realize_line(leaf, fields[path, "length"], rlgc, values, mesh, preparation_cache)
            for station in range(1, grids[path]["n_sections"]):
                nodes.extend(internal_node(leaf, grids[path]["n_sections"], station, c) for c in rlgc["conductors"])
    indices = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    C, K, G = (dict(rows=[], cols=[], values=[]) for _ in range(3))

    def stamp(target: dict, support, local: np.ndarray) -> None:
        support = tuple(support)
        target["rows"].extend(np.repeat(support, len(support)))
        target["cols"].extend(np.tile(support, len(support)))
        target["values"].extend(local.reshape(-1))

    def outer_stamp(target: dict, b: dict, scale: float) -> None:
        support = tuple(b)
        weights = np.asarray([b[row] for row in support])
        stamp(target, support, scale * np.outer(weights, weights))

    def difference(positive: dict, negative: dict) -> dict:
        result = dict(positive)
        for row, value in negative.items():
            result[row] = result.get(row, 0) - value
        return result

    def evidence_vector(b: dict) -> list:
        vector = np.zeros(n)
        for row, value in b.items():
            vector[row] = value
        return vector.tolist()
    blocks, branches, rows = [], [], []

    def incidence(net: str) -> dict:
        return {} if net == ground else {indices[net]: 1.0}

    for leaf in plan["physical_leaves"]:
        path = tuple(leaf["path"])
        model = leaf["model"]
        field = lambda name: fields[path, name]
        if model == "transmission_line":
            rlgc, grid = field("rlgc"), grids[path]
            sections, dx = grid["n_sections"], grid["dx_m"]
            conductors = rlgc["conductors"]

            def station_map(station: int) -> SparseMatrix:
                rr, cc, vv = [], [], []
                for col, conductor in enumerate(conductors):
                    net = (endpoints[path, ("head." if station == 0 else "tail.") + conductor]
                           if station in (0, sections) else internal_node(leaf, sections, station, conductor))
                    for row, value in incidence(net).items():
                        rr.append(row); cc.append(col); vv.append(value)
                return SparseMatrix.from_entries((n, len(conductors)), rr, cc, vv)

            Rline, Lline, Gline, Cline = (value * dx for value in prepare_rlgc(rlgc, preparation_cache)[:4])
            for section in range(sections):
                left, right = station_map(section), station_map(section + 1)
                binding = SparseMatrix.from_entries(left.shape, np.r_[left.rows, right.rows],
                            np.r_[left.cols, right.cols], np.r_[left.values, -right.values])
                blocks.append(SeriesRL(binding, Rline, Lline))
                for station in (left, right):
                    active = np.unique(station.rows)
                    local_indices = {row: index for index, row in enumerate(active)}
                    local = np.zeros((len(active), len(conductors)))
                    for row, col, value in zip(station.rows, station.cols, station.values, strict=True):
                        local[local_indices[row], col] += value
                    stamp(C, active, local @ (Cline / 2) @ local.T)
                    stamp(G, active, local @ (Gline / 2) @ local.T)
                rows.append({"component_path": leaf["path"], "kind": "pi_section", "section": section + 1,
                             "dx_m": dx, "conductors": conductors})
            continue
        positive, negative = (incidence(endpoints[path, pin]) for pin in leaf["pin_order"])
        b = difference(positive, negative)
        if model in ("capacitor", "resistor"):
            value = field("capacitance" if model == "capacitor" else "resistance")
            if not math.isfinite(value) or value <= 0:
                raise InvalidCandidatePhysicalParameter("primitive R/C must be positive", stage="physical_validation")
            if model == "capacitor":
                outer_stamp(C, b, value)
            else:
                outer_stamp(G, b, 1 / value)
        elif model in ("inductor", "josephson_junction"):
            if model == "josephson_junction":
                capacitance = field("junction_capacitance")
                if not math.isfinite(capacitance) or capacitance < 0:
                    raise InvalidCandidatePhysicalParameter("Cj must be finite and nonnegative", stage="physical_validation")
                outer_stamp(C, b, capacitance)
            for branch in leaf["oriented_branches"]:
                value = field(branch["value_field"])
                if not math.isfinite(value) or value <= 0:
                    raise InvalidCandidatePhysicalParameter("inductance must be positive", stage="physical_validation")
                branch_b = difference(incidence(endpoints[path, branch["positive_pin"]]), incidence(endpoints[path, branch["negative_pin"]]))
                branches.append((path, branch["id"], branch_b, value))
        else:
            raise CompilerInvariantError(f"unsupported physical model {model}", stage="compile")
        rows.append({"component_path": leaf["path"], "kind": model, "positive": evidence_vector(positive), "negative": evidence_vector(negative)})
    if branches:
        branch_indices = {(branch[0], branch[1]): i for i, branch in enumerate(branches)}
        neighbors = [[] for _ in branches]
        edges = []
        for coupling in connectivity["couplings"]:
            a, b = (branch_indices[tuple(coupling[key]["path"]), coupling[key]["branch_id"]] for key in ("inductor_a", "inductor_b"))
            coefficient = quantity(coupling["coupling_coefficient"])
            if not math.isfinite(coefficient) or abs(coefficient) >= 1:
                raise InvalidCandidatePhysicalParameter("mutual coupling coefficient must satisfy abs(k) < 1", stage="physical_validation")
            mutual = coefficient * np.sqrt(branches[a][3] * branches[b][3])
            edges.append((a, b, mutual))
            neighbors[a].append(b)
            neighbors[b].append(a)
        visited = set()
        for start in range(len(branches)):
            if start in visited:
                continue
            pending, group = [start], []
            visited.add(start)
            while pending:
                index = pending.pop()
                group.append(index)
                for neighbor in neighbors[index]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        pending.append(neighbor)
            if len(group) == 1:
                branch = branches[group[0]]
                outer_stamp(K, branch[2], 1 / branch[3])
                continue
            local = {index: ordinal for ordinal, index in enumerate(group)}
            L = np.diag([branches[index][3] for index in group])
            for a, b, mutual in edges:
                if a in local and b in local:
                    L[local[a], local[b]] = L[local[b], local[a]] = mutual
            try:
                np.linalg.cholesky(L)
            except np.linalg.LinAlgError as error:
                raise InvalidCandidatePhysicalParameter("complete reciprocal inductance matrix is not positive definite", stage="physical_validation") from error
            support = tuple(sorted({row for index in group for row in branches[index][2]}))
            support_indices = {row: index for index, row in enumerate(support)}
            binding = np.zeros((len(support), len(group)))
            for col, index in enumerate(group):
                for row, value in branches[index][2].items():
                    binding[support_indices[row], col] = value
            reciprocal = np.linalg.solve(L, binding.T)
            residual = backward_residual(L, reciprocal, binding.T) if support else 0.0
            if not math.isfinite(residual) or residual > 256 * (len(group) + 1) * np.finfo(np.float64).eps:
                raise InvalidCandidatePhysicalParameter("reciprocal inductance solve exceeded normalized residual contract", stage="physical_validation")
            stamp(K, support, binding @ reciprocal)
    ports = connectivity["ports"]
    port_rows, port_cols, port_values = [], [], []
    for col, port in enumerate(ports):
        for row, value in incidence(port["net"]).items():
            port_rows.append(row); port_cols.append(col); port_values.append(value)
    B = SparseMatrix.from_entries((n, len(ports)), port_rows, port_cols, port_values)
    C, K, G = (SparseMatrix.from_entries((n, n), **target) for target in (C, K, G))
    R = np.diag([quantity(port["reference_impedance"]) for port in ports])
    evidence = {"original_coordinates": [row["compiler_node_id"] for row in connectivity["node_coordinates"]],
                "discretization": list(grids.values()), "branches": rows, "extrapolation": extrapolation,
                "resolved_fields": [{"component_path": list(path), "field": name, "value_si": value}
                                    for (path, name), value in fields.items()]}
    return CompiledModel(tuple(nodes), tuple(port["id"] for port in ports), C, K, G, B, R,
                         np.ones(len(ports)), tuple(blocks), record_bytes(evidence))
