"""Request-owned section realization; never modifies physical Plan metadata."""

from __future__ import annotations

import math
import numpy as np

from ..canonical import canonical_json_bytes, float64_from_hex
from ..errors import CompilerInvariantError, InvalidCandidatePhysicalParameter
from .models import MeshGroup, MeshSpec, immutable_array


def quantity(record: dict) -> float:
    return float64_from_hex(record["si_value_f64"])


def matrix(record: dict) -> np.ndarray:
    return np.array([float64_from_hex(v) for v in record["values_f64"]]).reshape(record["shape"])


def prepare_rlgc(rlgc: dict, cache: dict | None = None):
    """Reuse only immutable material identity, independently of length/mesh."""
    key = canonical_json_bytes(rlgc)
    if cache is not None and key in cache:
        return cache[key]
    R, L, G, C = (matrix(rlgc[name + "_per_length"])
                  for name in ("resistance", "inductance", "conductance", "capacitance"))
    if any(value.shape != (len(rlgc["conductors"]),) * 2 for value in (R, L, G, C)):
        raise CompilerInvariantError("RLGC matrix dimension disagrees with line conductors", stage="compile")
    try:
        factor = np.linalg.cholesky(L)
        np.linalg.cholesky(C)
        if np.linalg.eigvalsh(R)[0] < 0 or np.linalg.eigvalsh(G)[0] < 0:
            raise np.linalg.LinAlgError("RLGC R/G must be positive semidefinite")
    except np.linalg.LinAlgError as error:
        raise InvalidCandidatePhysicalParameter("RLGC physical matrix validation failed", stage="physical_validation") from error
    result = tuple(immutable_array(value) for value in (R, L, G, C, factor))
    if cache is not None:
        cache[key] = result
    return result


def realize_line(leaf: dict, length: float, rlgc: dict, values: dict, mesh: MeshSpec,
                 preparation_cache: dict | None = None) -> dict:
    if not math.isfinite(length) or length <= 0:
        raise InvalidCandidatePhysicalParameter("line length must be finite and positive", stage="physical_validation")
    path = tuple(leaf["path"])
    metadata = leaf["model_metadata"]
    R, L, G, C, factor = prepare_rlgc(rlgc, preparation_cache)
    record = {"component_path": list(path), "length_m": length, "strategy": mesh.kind}
    if "n_sections" in metadata:
        sections = metadata["n_sections"]
        record["kind"] = "fixed_count"
    else:
        policy = metadata["discretization"]
        if np.any(R) or np.any(G):
            raise InvalidCandidatePhysicalParameter("ElectricalResolution requires zero R and G", stage="physical_validation")
        delay_key = (canonical_json_bytes(rlgc), "modal_delays")
        delays = preparation_cache.get(delay_key) if preparation_cache is not None else None
        if delays is None:
            try:
                eigenvalues = np.linalg.eigvalsh(factor.T @ C @ factor)
            except np.linalg.LinAlgError as error:
                raise InvalidCandidatePhysicalParameter("lossless modal decomposition failed", stage="physical_validation") from error
            if not np.all(np.isfinite(eigenvalues)) or not np.all(eigenvalues > 0):
                raise InvalidCandidatePhysicalParameter("lossless modal delays are not finite and positive", stage="physical_validation")
            delays = immutable_array(np.sqrt(eigenvalues))
            if preparation_cache is not None:
                preparation_cache[delay_key] = delays
        hmax = 1.0 / (policy["sections_per_wavelength"] * quantity(policy["max_frequency"]) * max(delays))
        ratio = length / hmax
        if (not np.all(np.isfinite(1 / delays)) or not math.isfinite(hmax) or hmax <= 0 or
                not math.isfinite(ratio) or ratio > np.iinfo(np.int64).max):
            raise InvalidCandidatePhysicalParameter("electrical resolution is not representable", stage="physical_validation")
        sections = max(1, math.ceil(ratio))
        record.update(kind="electrical_resolution", policy=policy, hmax_m=float(hmax), modal_velocities_m_s=(1 / delays).tolist())
    if mesh.kind == "fixed" and "discretization" in metadata:
        sections = dict(mesh.sections)[path]
        record["override_applied"] = True
    elif mesh.kind == "grouped" and "discretization" in metadata:
        position = values[mesh.parameter_key]
        selected = next(group for group in mesh.groups if
                        (position >= group.lower if group.lower_inclusive else position > group.lower)
                        and (position <= group.upper if group.upper_inclusive else position < group.upper))
        sections = dict(selected.sections)[path]
        record["group"] = mesh.groups.index(selected)
        record["override_applied"] = True
    elif mesh.kind not in ("dynamic", "fixed", "grouped"):
        raise ValueError(f"unsupported mesh strategy {mesh.kind!r}")
    if not isinstance(sections, int) or isinstance(sections, bool) or sections <= 0:
        raise CompilerInvariantError("benchmark mesh section count is invalid", stage="compile")
    dx = length / sections
    with np.errstate(over="ignore", under="ignore"):
        scaled = tuple(value * dx for value in (R, L, G, C))
    if not math.isfinite(dx) or dx <= 0 or any(not np.all(np.isfinite(value)) for value in scaled):
        raise InvalidCandidatePhysicalParameter("line section realization is not finite", stage="physical_validation")
    try:
        np.linalg.cholesky(scaled[1])
        np.linalg.cholesky(scaled[3])
    except np.linalg.LinAlgError as error:
        raise InvalidCandidatePhysicalParameter("RLGC section physical matrix validation failed", stage="physical_validation") from error
    record.update(n_sections=sections, dx_m=dx)
    return record


def mesh_from_record(record: dict) -> MeshSpec:
    sections = lambda items: tuple((tuple(path), count) for path, count in items)
    return MeshSpec(
        kind=record["kind"], sections=sections(record["sections"]),
        parameter_key=tuple(record["parameter_key"]) if record["parameter_key"] is not None else None,
        groups=tuple(MeshGroup(float64_from_hex(g["lower_f64"]), float64_from_hex(g["upper_f64"]), g["lower_inclusive"], g["upper_inclusive"], sections(g["sections"]))
                     for g in record["groups"]),
        derivation_bytes=canonical_json_bytes(record["derivation"]),
    )

