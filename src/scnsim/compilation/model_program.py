"""Static sparse stamps with runtime physical fields and fixed line sections.

The existing compiler supplies only initial topology/physical validation here.
Every candidate coefficient is rebuilt by the same traced field/stamp program;
no host-realized candidate coefficients or mutable compiler cache are captured.
Global matrices remain sparse slots. Dense operands are local conductor groups.
Consecutive equal-shape station arithmetic is batched per exact line material;
compiled scans preserve the original sparse addition order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple
import json
import math
import warnings
import numpy as np

from .compiler import compile_model, parameter_values, resolve_fields
from .mesh import matrix, prepare_rlgc, quantity, realize_line
from .models import MeshSpec
from .program import FieldProgram, MeshPlan, SparsePattern, compile_fields, _indices
from ..canonical import canonical_json_bytes
from ..numeric_encoding import record_data, record_document


class PhysicalChecks(NamedTuple):
    status: Any
    failed_field: Any


class ModelPayload(NamedTuple):
    C: Any
    K: Any
    G: Any
    B: Any
    R: Any
    M: Any
    series: tuple
    status: Any
    failed_site: Any


@dataclass(frozen=True, slots=True, eq=False)
class Stamp:
    target: str
    slots: Any
    weights: Any
    field: int
    reciprocal: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class LineStamp:
    field: int
    sections: int
    blocks: tuple
    material: int
    stations: tuple
    station_groups: tuple
    declared_lossless: bool
    electrical_resolution: bool
    resolution_policy: tuple[int, float] | None


class PackedSeries(NamedTuple):
    """Runtime banks and incidence values; static section order stays in layout."""
    R: Any
    L: Any
    material_indices: Any
    weights: Any


@dataclass(frozen=True, slots=True, eq=False)
class SeriesLayout:
    width: int
    line_indices: tuple[int, ...]
    material_indices: Any
    weights: Any


@dataclass(frozen=True, slots=True, eq=False)
class StationGroup:
    """Consecutive same-shape stations for ONE material/length dependency.

    Group order and the leading station axis preserve compiler stamping order;
    only local matrix arithmetic is batched, never conflicting sparse sums.
    """
    C_slots: Any
    G_slots: Any
    local: Any


@dataclass(frozen=True, slots=True, eq=False)
class InductanceGroup:
    fields: tuple[int, ...]
    couplings: tuple[tuple[int, int, float], ...]
    binding: Any
    slots: Any


@dataclass(frozen=True, slots=True, eq=False)
class ModelProgram:
    fields: FieldProgram
    mesh: MeshPlan
    node_ids: tuple[str, ...]
    port_ids: tuple[str, ...]
    patterns: tuple[SparsePattern, SparsePattern, SparsePattern, SparsePattern]
    stamps: tuple[Stamp, ...]
    lines: tuple[LineStamp, ...]
    series_layouts: tuple[SeriesLayout, ...]
    inductors: tuple[InductanceGroup, ...]
    B_values: Any
    port_resistance: Any
    scalar_domains: tuple[tuple[int, bool], ...]
    evidence_bytes: bytes

    def physical_fields(self, theta, *, affine_evidence=None):
        """Validate the original binary64 point before numerical dtype casting.

        This shares field expressions and fixed stamps with the traced path;
        selected-dtype underflow does not create a different physical domain.
        """
        from ..errors import InvalidCandidatePhysicalParameter
        if affine_evidence is not None:
            affine_evidence.rows.clear()
        fields = self.fields.apply(np.asarray(theta, dtype=np.float64), xp=np,
                                   affine_evidence=affine_evidence)
        if fields.status:
            raise InvalidCandidatePhysicalParameter("affine input is outside declared support", stage="affine_support",
                **({"evidence": {"affine_evidence": affine_evidence.rows}}
                   if affine_evidence is not None and affine_evidence.rows else {}))
        for index, nonnegative in self.scalar_domains:
            value = fields.scalars[index]
            if not math.isfinite(value) or (value < 0 if nonnegative else value <= 0):
                raise InvalidCandidatePhysicalParameter("physical scalar must be finite and nonnegative" if nonnegative else
                                                        "physical scalar must be finite and positive", stage="physical_validation")
        for line in self.lines:
            R, L, G, C = fields.materials[line.material]
            try:
                factor = np.linalg.cholesky(L)
                np.linalg.cholesky(C)
                if np.linalg.eigvalsh(R)[0] < 0 or np.linalg.eigvalsh(G)[0] < 0:
                    raise np.linalg.LinAlgError("RLGC R/G must be positive semidefinite")
            except np.linalg.LinAlgError as error:
                raise InvalidCandidatePhysicalParameter("RLGC physical matrix validation failed", stage="physical_validation") from error
            if line.electrical_resolution:
                if np.any(R) or np.any(G):
                    raise InvalidCandidatePhysicalParameter("ElectricalResolution requires zero R and G", stage="physical_validation")
                try:
                    eigenvalues = np.linalg.eigvalsh(factor.T @ C @ factor)
                except np.linalg.LinAlgError as error:
                    raise InvalidCandidatePhysicalParameter("lossless modal decomposition failed", stage="physical_validation") from error
                if not np.all(np.isfinite(eigenvalues)) or not np.all(eigenvalues > 0):
                    raise InvalidCandidatePhysicalParameter("lossless modal delays are not finite and positive", stage="physical_validation")
                delays = np.sqrt(eigenvalues)
                samples, maximum_frequency = line.resolution_policy
                hmax = 1.0 / (samples * maximum_frequency * max(delays))
                ratio = fields.scalars[line.field] / hmax
                if (not np.all(np.isfinite(1 / delays)) or not math.isfinite(hmax) or hmax <= 0 or
                        not math.isfinite(ratio) or ratio > np.iinfo(np.int64).max):
                    raise InvalidCandidatePhysicalParameter("electrical resolution is not representable", stage="physical_validation")
            dx = fields.scalars[line.field] / line.sections
            scaled = tuple(value * dx for value in (R, L, G, C))
            if not math.isfinite(dx) or dx <= 0 or any(not np.all(np.isfinite(value)) for value in scaled):
                raise InvalidCandidatePhysicalParameter("line section realization is not finite", stage="physical_validation")
            try:
                np.linalg.cholesky(scaled[1])
                np.linalg.cholesky(scaled[3])
            except np.linalg.LinAlgError as error:
                raise InvalidCandidatePhysicalParameter("RLGC section physical matrix validation failed", stage="physical_validation") from error
        for group in self.inductors:
            diagonal = fields.scalars[np.asarray(group.fields)]
            L = np.diag(diagonal)
            for a, b, k in group.couplings:
                L[a, b] = L[b, a] = k * np.sqrt(diagonal[a] * diagonal[b])
            try:
                np.linalg.cholesky(L)
            except np.linalg.LinAlgError as error:
                raise InvalidCandidatePhysicalParameter("complete reciprocal inductance matrix is not positive definite", stage="physical_validation") from error
        return fields

    def physical_checks(self, theta):
        """Array-only form of the existing ORIGINAL binary64 physical domain.

        Preparation chooses a finite route partition; this callable reports a
        future runtime point outside that domain without host bool conversion.
        Numerical selected-dtype certificates are a separate responsibility.
        """
        import jax
        import jax.numpy as jnp
        with jax.enable_x64(True):
            fields = self.fields.apply(jnp.asarray(theta, dtype=jnp.float64), xp=jnp, check_support=True)
            status = jnp.where(fields.status != 0, jnp.int32(1), jnp.int32(0))
            failed = fields.failed_field
            def check(bad, field):
                nonlocal status, failed
                owns = (status == 0) & bad
                failed = jnp.where(owns, jnp.int32(field), failed)
                status = jnp.where(owns, jnp.int32(2), status)
            for index, nonnegative in self.scalar_domains:
                value = fields.scalars[index]
                check((~jnp.isfinite(value)) | ((value < 0) if nonnegative else (value <= 0)), index)
            for line in self.lines:
                R, L, G, C = fields.materials[line.material]
                factor = jnp.linalg.cholesky(L)
                capacitance_factor = jnp.linalg.cholesky(C)
                check((~jnp.all(jnp.isfinite(factor))) | (~jnp.all(jnp.isfinite(capacitance_factor))) |
                      (jnp.linalg.eigvalsh(R)[0] < 0) | (jnp.linalg.eigvalsh(G)[0] < 0), line.field)
                if line.electrical_resolution:
                    check(jnp.any(R != 0) | jnp.any(G != 0), line.field)
                    eigenvalues = jnp.linalg.eigvalsh(factor.T @ C @ factor)
                    check((~jnp.all(jnp.isfinite(eigenvalues))) | (~jnp.all(eigenvalues > 0)), line.field)
                    delays = jnp.sqrt(eigenvalues)
                    samples, maximum_frequency = line.resolution_policy
                    hmax = 1 / (samples * maximum_frequency * jnp.max(delays))
                    ratio = fields.scalars[line.field] / hmax
                    check((~jnp.all(jnp.isfinite(1 / delays))) | (~jnp.isfinite(hmax)) | (hmax <= 0) |
                          (~jnp.isfinite(ratio)) | (ratio > jnp.iinfo(jnp.int64).max), line.field)
                dx = fields.scalars[line.field] / line.sections
                scaled = tuple(material * dx for material in (R, L, G, C))
                check((~jnp.isfinite(dx)) | (dx <= 0) |
                      ~jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in scaled))), line.field)
                check((~jnp.all(jnp.isfinite(jnp.linalg.cholesky(scaled[1])))) |
                      (~jnp.all(jnp.isfinite(jnp.linalg.cholesky(scaled[3])))), line.field)
            for group in self.inductors:
                diagonal = fields.scalars[jnp.asarray(group.fields)]
                L = jnp.diag(diagonal)
                for a, b, k in group.couplings:
                    mutual = k * jnp.sqrt(diagonal[a] * diagonal[b])
                    L = L.at[a, b].set(mutual).at[b, a].set(mutual)
                check(~jnp.all(jnp.isfinite(jnp.linalg.cholesky(L))), group.fields[0])
            return PhysicalChecks(status, failed)

    def point_discretization(self, fields):
        """Refresh measured grid metadata without planning a different mesh.

        Fixed section counts belong to this program. Length and modal metrics
        belong to the effective binary64 point, never the compilation baseline.
        """
        rows = record_document(self.evidence_bytes)['discretization']
        by_path = {tuple(row['component_path']): row for row in rows}
        for line in self.lines:
            path, _ = self.fields.keys[line.field]
            row = by_path[path]
            length = float(fields.scalars[line.field])
            row.update(length_m=length, n_sections=line.sections, dx_m=length / line.sections)
            if line.electrical_resolution:
                _, L, _, C = fields.materials[line.material]
                factor = np.linalg.cholesky(L)
                delays = np.sqrt(np.linalg.eigvalsh(factor.T @ C @ factor))
                samples, maximum_frequency = line.resolution_policy
                row.update(hmax_m=float(1.0 / (samples * maximum_frequency * max(delays))),
                           modal_velocities_m_s=(1.0 / delays).tolist())
        return rows

    def physical_capacitance(self, fields):
        """Original-dtype capacitance slots for the physical View rank rule."""
        result = np.zeros(len(self.patterns[0].rows), dtype=np.float64)
        for stamp in self.stamps:
            if stamp.target == 'C':
                value = fields.scalars[stamp.field]
                np.add.at(result, stamp.slots, stamp.weights * (1 / value if stamp.reciprocal else value))
        for line in self.lines:
            C = fields.materials[line.material][3]
            dx = fields.scalars[line.field] / line.sections
            for slots, local in line.stations:
                np.add.at(result, slots[0], (local @ (C * 0.5) @ local.T).ravel() * dx)
        return result

    def __call__(self, theta):
        import jax
        import jax.numpy as jnp
        fields = self.fields(theta)
        coefficients = {name: jnp.zeros((len(pattern.rows),), dtype=theta.dtype)
                        for name, pattern in zip(('C', 'K', 'G'), self.patterns[:3], strict=True)}
        status, site = fields.status, fields.failed_field
        for stamp in self.stamps:
            value = fields.scalars[stamp.field]
            value = 1 / value if stamp.reciprocal else value
            coefficients[stamp.target] = coefficients[stamp.target].at[jnp.asarray(stamp.slots)].add(
                jnp.asarray(stamp.weights, dtype=theta.dtype) * value)
        line_materials = []
        for line in self.lines:
            dx = fields.scalars[line.field] / jnp.asarray(line.sections, dtype=theta.dtype)
            R, L, G, C = fields.materials[line.material]
            half = jnp.asarray(0.5, dtype=theta.dtype)
            for group in line.station_groups:
                local = jnp.asarray(group.local, dtype=theta.dtype)
                # vmap keeps the same local dot contraction per station. The
                # sparse accumulator is an ordered loop, not an unordered
                # batched scatter/reduction across overlapping stations.
                def local_value(binding, material):
                    return (binding @ (material * half) @ binding.T).ravel() * dx
                c_values = jax.vmap(local_value, in_axes=(0, None))(local, C)
                g_values = jax.vmap(local_value, in_axes=(0, None))(local, G)
                def add_station(carry, station):
                    c, g = carry
                    c_slots, g_slots, c_value, g_value = station
                    return (c.at[c_slots].add(c_value), g.at[g_slots].add(g_value)), None
                (coefficients['C'], coefficients['G']), _ = jax.lax.scan(
                    add_station, (coefficients['C'], coefficients['G']),
                    (jnp.asarray(group.C_slots), jnp.asarray(group.G_slots), c_values, g_values))
            section_R, section_L = R * dx, L * dx
            line_materials.append((section_R, section_L))
        series = tuple(PackedSeries(
            jnp.stack(tuple(line_materials[index][0] for index in group.line_indices)),
            jnp.stack(tuple(line_materials[index][1] for index in group.line_indices)),
            jnp.asarray(group.material_indices), jnp.asarray(group.weights, dtype=theta.dtype))
            for group in self.series_layouts)
        for group in self.inductors:
            diagonal = fields.scalars[jnp.asarray(group.fields)]
            L = jnp.diag(diagonal)
            for a, b, k in group.couplings:
                mutual = jnp.asarray(k, dtype=theta.dtype) * jnp.sqrt(diagonal[a] * diagonal[b])
                L = L.at[a, b].set(mutual).at[b, a].set(mutual)
            binding = jnp.asarray(group.binding, dtype=theta.dtype)
            x = jnp.linalg.solve(L, binding.T)
            num = jnp.max(jnp.abs(L @ x - binding.T), initial=0)
            den = jnp.max(jnp.abs(L) @ jnp.abs(x) + jnp.abs(binding.T), initial=0)
            eta = jnp.where(jnp.isfinite(num) & jnp.isfinite(den), jnp.where(den == 0, jnp.where(num == 0, 0, jnp.inf), num / den), jnp.inf)
            threshold = jnp.asarray(256 * (len(group.fields) + 1), dtype=theta.dtype) * jnp.finfo(theta.dtype).eps
            bad = (~jnp.isfinite(eta)) | (eta > threshold)
            first = (status == 0) & bad
            site = jnp.where(first, jnp.int32(group.fields[0]), site)
            status = jnp.where(first, jnp.int32(2), status)
            coefficients['K'] = coefficients['K'].at[jnp.asarray(group.slots)].add((binding @ x).ravel())
        return ModelPayload(coefficients['C'], coefficients['K'], coefficients['G'],
                            jnp.asarray(self.B_values, dtype=theta.dtype),
                            jnp.asarray(self.port_resistance, dtype=theta.dtype),
                            jnp.ones((len(self.port_ids),), dtype=theta.dtype), tuple(series), status, site)


def _immutable(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)




def _series_layouts(lines):
    """One realized material bank per line, sections remain ordered gathers."""
    groups = {}
    for line_index, line in enumerate(lines):
        width = line.blocks[0][3][1]
        groups.setdefault(width, []).append(line_index)
    result = []
    for width, line_indices in groups.items():
        material_indices, weights = [], []
        for bank_index, line_index in enumerate(line_indices):
            for incidence in lines[line_index].blocks:
                material_indices.append(bank_index)
                weights.extend(incidence[2])
        result.append(SeriesLayout(width, tuple(line_indices), _indices(material_indices),
                                   _immutable(np.asarray(weights, dtype=np.float64))))
    return tuple(result)


def _station_groups(stations):
    """Freeze contiguous shape runs without changing station/material order."""
    groups = []
    start = 0
    while start < len(stations):
        stop = start + 1
        shape = stations[start][1].shape
        while stop < len(stations) and stations[stop][1].shape == shape:
            stop += 1
        run = stations[start:stop]
        groups.append(StationGroup(
            _immutable(np.stack(tuple(slots[0] for slots, _ in run))),
            _immutable(np.stack(tuple(slots[1] for slots, _ in run))),
            _immutable(np.stack(tuple(local for _, local in run)))))
        start = stop
    return tuple(groups)


def _material_envelope(binding, bounds):
    """Absolute entry enclosure from DECLARED scalar coefficient bounds only."""
    if binding['kind'] == 'constant':
        return tuple(np.abs(matrix(binding['value'][name + '_per_length']))
                     for name in ('inductance', 'capacitance'))
    if binding['kind'] == 'ref':
        # A whole-material point is a runtime operand. Scalar search bounds do
        # not enclose a matrix; no hidden element ranges are manufactured.
        return None
    if binding['kind'] != 'scalar_rlgc':
        raise TypeError('unsupported structured material binding')
    from .compiler import parameter_key
    names = ('inductance', 'capacitance')
    envelope = [np.abs(matrix(binding['base'][name + '_per_length'])) for name in names]
    for term in binding['terms']:
        if all(term[name] is None or not np.any(matrix(term[name])) for name in names):
            continue
        source = term['source']
        ref = source['parameter'] if source['kind'] == 'ref' else source['input']
        interval = bounds.get(parameter_key(ref))
        if interval is None or not all(math.isfinite(item) for item in interval):
            return None
        if source['kind'] == 'affine':
            interval = tuple(quantity(source['slope']) * item + quantity(source['intercept']) for item in interval)
        magnitude = max(map(abs, interval))
        for index, name in enumerate(names):
            if term[name] is not None:
                envelope[index] = envelope[index] + magnitude * np.abs(matrix(term[name]))
    return tuple(envelope)


def plan_mesh(plan, parameters, *, bounds=None, authorized=frozenset(), resolution=None) -> MeshPlan:
    """Plan fixed sections from actual declared physical enclosures.

    Total matrices are validated by the current lossless ElectricalResolution
    domain before planning. Missing/unavailable enclosures warn and bind the
    baseline realized stage mesh, not an invented range or adaptive mesh.
    """
    values = parameter_values(parameters)
    fields, _ = resolve_fields(plan, values, authorized)
    sections, derivations, notes = [], [], []
    bounds = {} if bounds is None else bounds
    for leaf in plan['physical_leaves']:
        if leaf['model'] != 'transmission_line':
            continue
        path = tuple(leaf['path'])
        length, rlgc = fields[path, 'length'], fields[path, 'rlgc']
        if resolution is not None:
            leaf = dict(leaf, model_metadata=dict(leaf['model_metadata'], discretization=resolution))
        grid = realize_line(leaf, length, rlgc, values, MeshSpec(), {})
        count = grid['n_sections']
        declarations = {row['id']: row['binding'] for row in leaf['fields']}
        binding = declarations['length']
        upper, delay_bound = None, None
        if 'discretization' in leaf['model_metadata']:
            if binding['kind'] == 'constant':
                upper = length
            else:
                from .compiler import parameter_key
                key = parameter_key(binding['parameter'] if binding['kind'] == 'ref' else binding['input'])
                interval = bounds.get(key)
                if interval is not None and all(math.isfinite(item) for item in interval):
                    ends = interval
                    if binding['kind'] == 'affine':
                        ends = tuple(quantity(binding['slope']) * item + quantity(binding['intercept']) for item in ends)
                    upper = max(ends)
            envelope = _material_envelope(declarations['rlgc'], bounds)
            if envelope is not None:
                # For symmetric physical L/C, spectral norm is bounded by the
                # maximum absolute row sum. eig(LC) <= ||L||2 ||C||2, so this
                # enclosure safely bounds all declared modal delays.
                norms = tuple(float(np.max(np.sum(part, axis=1))) for part in envelope)
                delay_bound = math.sqrt(norms[0]) * math.sqrt(norms[1])
            if upper is None or delay_bound is None or not math.isfinite(delay_bound):
                note = f'{"/".join(path)}: unavailable finite declared length/material enclosure; fixed baseline-realized mesh'
                notes.append(note)
                warnings.warn(note, RuntimeWarning, stacklevel=2)
            else:
                policy = leaf['model_metadata']['discretization']
                ratio = upper * policy['sections_per_wavelength'] * quantity(policy['max_frequency']) * delay_bound
                if not math.isfinite(ratio):
                    note = f'{"/".join(path)}: declared enclosure mesh is not representable; fixed baseline-realized mesh'
                    notes.append(note)
                    warnings.warn(note, RuntimeWarning, stacklevel=2)
                else:
                    count = max(1, math.ceil(ratio))
        sections.append((path, count))
        derivations.append(dict(path=list(path), baseline_sections=grid['n_sections'], planned_sections=count,
                                maximum_length_m=upper, modal_delay_bound_s_per_m=delay_bound,
                                physical_policy=leaf['model_metadata']))
    # Preserve represented physical derivation scalars with the shared exact
    # binary64 record codec before canonical JSON rejects untagged floats.
    return MeshPlan(tuple(sections), 'declared_bounds_fixed',
                    canonical_json_bytes(record_data(derivations)), tuple(notes))


def compile_model_program(plan, parameters, *, mesh=None, bounds=None,
                          active_keys=None, authorized=frozenset()) -> ModelProgram:
    mesh = plan_mesh(plan, parameters, bounds=bounds, authorized=authorized) if mesh is None else mesh
    fields = compile_fields(plan, parameters, active_keys=active_keys, authorized=authorized)
    values = parameter_values(parameters)
    # Physical validation still precedes the fixed override, including original
    # lossless ElectricalResolution requirements. Fixed-count lossy lines keep
    # their current capability and generic resistance dependence.
    templates = {}
    raw = compile_model(plan, values, mesh=MeshSpec(kind='fixed', sections=mesh.sections,
                                                  derivation_bytes=mesh.derivation_bytes),
                        authorized=authorized, template_cache=templates)
    template = next(value for key, value in templates.items() if key[0] == 'compiler_topology')
    matrices = (raw.C, raw.K, raw.G, raw.B)
    patterns, raw_maps = zip(*(SparsePattern.from_coo(m.shape, m.rows, m.cols) for m in matrices), strict=True)
    locations = [{(int(row), int(col)): index for index, (row, col) in enumerate(zip(p.rows, p.cols, strict=True))}
                 for p in patterns]
    field_index = {key: index for index, key in enumerate(fields.keys)}
    endpoints, indices = template['endpoints'], template['indices']
    ground = plan['connectivity']['canonical_ground']
    def incidence(net):
        return {} if net == ground else {indices[net]: 1.0}
    def difference(a, b):
        result = dict(a)
        for row, value in b.items(): result[row] = result.get(row, 0) - value
        return result
    def slots(name, support):
        lookup = locations[('C', 'K', 'G').index(name)]
        return _indices([lookup[row, col] for row in support for col in support])
    stamps, lines, branches, domains = [], [], [], []
    resolved, _ = resolve_fields(plan, values, authorized)
    for leaf in plan['physical_leaves']:
        path, kind = tuple(leaf['path']), leaf['model']
        if kind == 'transmission_line':
            source = field_index[path, 'length']; domains.append((source, False))
            count = dict(mesh.sections)[path]
            R, L, G, C = prepare_rlgc(resolved[path, 'rlgc'])[:4]
            blocks, stations = [], []
            for section in range(count):
                binding = template['section_bindings'][path, section]
                bp, bm = SparsePattern.from_coo(binding.shape, binding.rows, binding.cols)
                bv = np.zeros(len(bp.rows))
                np.add.at(bv, bm, binding.values)
                blocks.append((tuple(bp.rows), tuple(bp.cols), tuple(bv), binding.shape))
                for station in (section, section + 1):
                    station_map = template['station_maps'][path, station]
                    support = tuple(np.unique(station_map.rows))
                    local_indices = {row: index for index, row in enumerate(support)}
                    local = np.zeros((len(support), len(R)))
                    for row, col, value in zip(station_map.rows, station_map.cols, station_map.values, strict=True):
                        local[local_indices[row], col] += value
                    stations.append(((slots('C', support), slots('G', support)), _immutable(local)))
            lines.append(LineStamp(source, count, tuple(blocks),
                                   next(index for index, material in enumerate(fields.structured) if material.key == (path, 'rlgc')),
                                   tuple(stations), _station_groups(stations),
                                   next(material.declared_lossless for material in fields.structured if material.key == (path, 'rlgc')),
                                   'discretization' in leaf['model_metadata'],
                                   (leaf['model_metadata']['discretization']['sections_per_wavelength'],
                                    quantity(leaf['model_metadata']['discretization']['max_frequency']))
                                   if 'discretization' in leaf['model_metadata'] else None))
            continue
        positive, negative = (incidence(endpoints[path, pin]) for pin in leaf['pin_order'])
        b = difference(positive, negative)
        if kind in ('capacitor', 'resistor', 'josephson_junction'):
            name = 'capacitance' if kind == 'capacitor' else 'resistance' if kind == 'resistor' else 'junction_capacitance'
            source = field_index[path, name]
            domains.append((source, kind == 'josephson_junction'))
            target = 'G' if kind == 'resistor' else 'C'
            stamps.append(Stamp(target, slots(target, tuple(b)), _immutable(np.outer(tuple(b.values()), tuple(b.values())).ravel()),
                                source, kind == 'resistor'))
        for branch in leaf.get('oriented_branches', ()):
            source = field_index[path, branch['value_field']]
            domains.append((source, False))
            branch_b = difference(incidence(endpoints[path, branch['positive_pin']]), incidence(endpoints[path, branch['negative_pin']]))
            branches.append((path, branch['id'], source, branch_b))
    lookup = {(path, key): index for index, (path, key, _, _) in enumerate(branches)}
    edges, neighbors = [], [[] for _ in branches]
    for coupling in plan['connectivity']['couplings']:
        a, b = (lookup[tuple(coupling[name]['path']), coupling[name]['branch_id']] for name in ('inductor_a', 'inductor_b'))
        edges.append((a, b, quantity(coupling['coupling_coefficient'])))
        neighbors[a].append(b); neighbors[b].append(a)
    groups, visited = [], set()
    for start in range(len(branches)):
        if start in visited: continue
        visited.add(start); group, pending = [], [start]
        while pending:
            index = pending.pop(); group.append(index)
            for neighbor in neighbors[index]:
                if neighbor not in visited: visited.add(neighbor); pending.append(neighbor)
        support = tuple(sorted({row for index in group for row in branches[index][3]}))
        route = {row: index for index, row in enumerate(support)}
        binding = np.zeros((len(support), len(group)))
        local = {index: position for position, index in enumerate(group)}
        for column, index in enumerate(group):
            for row, weight in branches[index][3].items(): binding[route[row], column] = weight
        local_edges = tuple((local[a], local[b], k) for a, b, k in edges if a in local and b in local)
        groups.append(InductanceGroup(tuple(branches[index][2] for index in group), local_edges,
                                      _immutable(binding), slots('K', support)))
    B = np.zeros(len(patterns[3].rows))
    np.add.at(B, raw_maps[3], raw.B.values)
    return ModelProgram(fields, mesh, raw.node_ids, raw.port_ids, tuple(patterns), tuple(stamps),
                         tuple(lines), _series_layouts(lines), tuple(groups), _immutable(B), _immutable(raw.R),
                         tuple(domains), canonical_json_bytes({
                             'original_coordinates': [row['compiler_node_id'] for row in plan['connectivity']['node_coordinates']],
                             # Keep the actual fixed compilation's grid records,
                             # including exact numeric tags, as metadata seeds.
                             # Runtime lengths/modal metrics are refreshed above.
                             'discretization': json.loads(raw.evidence_bytes)['discretization'],
                             'mesh_derivation': json.loads(mesh.derivation_bytes),
                             'physical_declaration_sha256': __import__('hashlib').sha256(canonical_json_bytes(plan)).hexdigest()}))
