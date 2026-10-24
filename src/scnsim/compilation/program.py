"""Immutable physical-program layouts and traced scalar field declarations.

Compilation binds topology, mesh and identities before runtime initialization.
Runtime parameters are selected-dtype SI operands, never JIT key constants.
This module owns no Run, Workspace, executable, native factor or AD residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple
import numpy as np
import struct

from ..canonical import canonical_json_bytes
from .compiler import parameter_key
from .mesh import quantity, matrix


class _AffineEvidence:
    """One host physical-check observation; never retained by a compiled core."""

    def __init__(self, *, parameter_set=(), optimization_spec=()):
        self.sources = tuple((name, frozenset(parameter_key(ref) for ref in refs))
                             for name, refs in (("parameter_set", parameter_set),
                                                ("optimization_spec", optimization_spec)))
        self.rows = []

    def observe(self, axis, target, argument, support, authorized, *, term_ordinal=None):
        lower, upper = support
        # Called only for an actually observed out-of-support host operand.
        side = "lower" if argument < lower else "upper" if argument > upper else None
        value = lambda x: {"si_unit": axis.si_unit, "si_value_f64": struct.pack(">d", float(x)).hex()}
        # Evidence encoding adds no finite-value admission rule ahead of the
        # existing physical validator. Unordered inputs have no invented side.
        distance = (lower - argument if side == "lower" else
                    argument - upper if side == "upper" else None)
        consumer = {"path": list(target[0]), "field": target[1]}
        if term_ordinal is not None:
            consumer["term_ordinal"] = term_ordinal
        self.rows.append({
            "parameter": {"definitions_id": axis.key[0], "parameter_id": axis.key[1]},
            "consumer_target": consumer, "input_value": value(argument),
            "support": [value(lower), value(upper)], "side": side,
            "distance": None if distance is None else value(distance),
            "authorized": bool(authorized),
            "authorization_sources": [name for name, keys in self.sources if axis.key in keys],
        })


def _indices(value):
    array = np.asarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class ParameterAxis:
    key: tuple[str, str]
    name: str
    si_unit: str
    matrix_field: str | None = None
    matrix_entry: int | None = None


@dataclass(frozen=True, slots=True)
class ParameterLayout:
    axes: tuple[ParameterAxis, ...]

    def pack(self, parameter_set: dict, *, dtype):
        bindings = {parameter_key(row['parameter']): row['value']
                    for row in parameter_set['bindings']}
        return np.asarray([quantity(bindings[axis.key]) if axis.matrix_field is None else
                           matrix(bindings[axis.key][axis.matrix_field]).ravel()[axis.matrix_entry]
                           for axis in self.axes], dtype=dtype)

    def input_indices(self, keys):
        # Requested derivative order is independent of the complete operand
        # layout; structured operands keep their own row-major slot order.
        by_key = {}
        for index, axis in enumerate(self.axes):
            by_key.setdefault(axis.key, []).append(index)
        return tuple(index for key in keys for index in by_key[key])


@dataclass(frozen=True, slots=True)
class MeshPlan:
    sections: tuple[tuple[tuple[str, ...], int], ...]
    strategy: str
    derivation_bytes: bytes
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class SparsePattern:
    """Canonical CSC slots; transpose routing does not infer reciprocity."""
    shape: tuple[int, int]
    rows: Any
    cols: Any
    indptr: Any

    def __post_init__(self):
        for name in ('rows', 'cols', 'indptr'):
            object.__setattr__(self, name, _indices(getattr(self, name)))

    @classmethod
    def from_coo(cls, shape, rows, cols):
        rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        order = np.lexsort((rows, cols))
        rows, cols = rows[order], cols[order]
        keys = cols * shape[0] + rows
        starts = np.r_[True, keys[1:] != keys[:-1]] if len(keys) else np.zeros(0, dtype=bool)
        unique_rows, unique_cols = rows[starts], cols[starts]
        slots = np.empty(len(order), dtype=np.int64)
        slots[order] = np.cumsum(starts) - 1
        counts = np.bincount(unique_cols, minlength=shape[1])
        return cls(tuple(shape), unique_rows, unique_cols, np.r_[0, np.cumsum(counts)]), _indices(slots)


@dataclass(frozen=True, slots=True)
class OutputAxis:
    quantity_key: str
    component: str
    si_unit: str
    coordinates: tuple[str, ...]
    shape: tuple[int, ...]


class FieldPayload(NamedTuple):
    scalars: Any
    materials: tuple
    status: Any
    failed_field: Any


@dataclass(frozen=True, slots=True, eq=False)
class MaterialTerm:
    source: int
    slope: float
    intercept: float
    support: tuple[float, float] | None
    authorized: bool
    bases: tuple[Any, ...]


@dataclass(frozen=True, slots=True, eq=False)
class MaterialField:
    key: tuple[tuple[str, ...], str]
    base: tuple[Any, ...]
    operand_indices: tuple[tuple[int, ...], ...]
    terms: tuple[MaterialTerm, ...]
    declared_lossless: bool

    def __call__(self, theta):
        import jax.numpy as jnp
        return self.apply(theta, xp=jnp, check_support=False)

    def apply(self, theta, *, xp, check_support=True, affine_evidence=None, layout=None):
        """One field expression in the physical or selected numerical namespace."""
        jnp = xp
        result = [jnp.asarray(value, dtype=theta.dtype) for value in self.base]
        for index, operands in enumerate(self.operand_indices):
            if operands:
                result[index] = theta[jnp.asarray(operands)].reshape(result[index].shape)
        status = jnp.int32(0)
        for term_ordinal, term in enumerate(self.terms):
            argument = theta[term.source]
            if check_support and term.support is not None and (not term.authorized or affine_evidence is not None):
                outside = ~((argument >= jnp.asarray(term.support[0], dtype=theta.dtype)) & (
                    argument <= jnp.asarray(term.support[1], dtype=theta.dtype)))
                if affine_evidence is not None and bool(outside):
                    affine_evidence.observe(layout.axes[term.source], self.key, argument,
                                            term.support, term.authorized, term_ordinal=term_ordinal)
                if not term.authorized:
                    status = jnp.where((status == 0) & outside, jnp.int32(1), status)
            value = argument * jnp.asarray(term.slope, dtype=theta.dtype) + jnp.asarray(term.intercept, dtype=theta.dtype)
            result = [a + value * jnp.asarray(b, dtype=theta.dtype)
                      for a, b in zip(result, term.bases, strict=True)]
        return tuple(result), status


@dataclass(frozen=True, slots=True)
class FieldProgram:
    """Complete runtime point, separate from the requested derivative axes.

    Whole RLGC refs use private row-major operand slots under their ONE public
    ref identity. Declared constant matrices alone are embedded coefficients.
    """
    layout: ParameterLayout
    keys: tuple[tuple[tuple[str, ...], str], ...]
    parameter_indices: tuple[int, ...]
    slopes: tuple[float, ...]
    intercepts: tuple[float, ...]
    supports: tuple[tuple[float, float] | None, ...]
    authorized: tuple[bool, ...]
    structured: tuple[MaterialField, ...]

    def __call__(self, theta):
        import jax.numpy as jnp
        return self.apply(theta, xp=jnp, check_support=False)

    def apply(self, theta, *, xp, check_support=True, affine_evidence=None):
        jnp = xp
        values = []
        status, failed = jnp.int32(0), jnp.int32(-1)
        for index, (source, slope, intercept, support, authorized) in enumerate(zip(
                self.parameter_indices, self.slopes, self.intercepts,
                self.supports, self.authorized, strict=True)):
            value = jnp.asarray(intercept, dtype=theta.dtype)
            if source >= 0:
                argument = theta[source]
                value = jnp.asarray(slope, dtype=theta.dtype) * argument + value
                if check_support and support is not None and (not authorized or affine_evidence is not None):
                    outside = ~((argument >= jnp.asarray(support[0], dtype=theta.dtype)) & (
                        argument <= jnp.asarray(support[1], dtype=theta.dtype)))
                    if affine_evidence is not None and bool(outside):
                        affine_evidence.observe(self.layout.axes[source], self.keys[index],
                                                argument, support, authorized)
                    if not authorized:
                        first = (status == 0) & outside
                        failed = jnp.where(first, jnp.int32(index), failed)
                        status = jnp.where(first, jnp.int32(1), status)
            values.append(value)
        materials = []
        for index, material in enumerate(self.structured):
            result, code = material.apply(theta, xp=xp, check_support=check_support,
                                          affine_evidence=affine_evidence, layout=self.layout)
            first = (status == 0) & (code != 0)
            failed = jnp.where(first, jnp.int32(len(self.keys) + index), failed)
            status = jnp.where(first, code, status)
            materials.append(result)
        return FieldPayload(jnp.stack(values) if values else jnp.zeros((0,), dtype=theta.dtype),
                            tuple(materials), status, failed)


_MATERIAL_NAMES = ('resistance', 'inductance', 'conductance', 'capacitance')


def _fixed(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def compile_fields(plan: dict, parameters: dict, *, active_keys=None,
                   authorized=frozenset()) -> FieldProgram:
    # active_keys selects differentiation at the entrypoint, never physical
    # constants or a narrower effective point. Every binding remains runtime.
    records = {parameter_key(row['parameter']): row for row in parameters['bindings']}
    axes = []
    for key, row in records.items():
        ref, value = row['parameter'], row['value']
        name = ref.get('name', ref['parameter_id'])
        if value['type'] == 'rlgc':
            for material_name in _MATERIAL_NAMES:
                field = material_name + '_per_length'
                for entry in range(int(np.prod(value[field]['shape']))):
                    axes.append(ParameterAxis(key, name, value[field]['si_unit'], field, entry))
        else:
            axes.append(ParameterAxis(key, name, value['si_unit']))
    layout = ParameterLayout(tuple(axes))
    positions = {axis.key: index for index, axis in enumerate(layout.axes) if axis.matrix_field is None}
    structured_positions = {(key, name + '_per_length'): tuple(index for index, axis in enumerate(layout.axes)
                              if axis.key == key and axis.matrix_field == name + '_per_length')
                            for key in records for name in _MATERIAL_NAMES}
    keys, sources, slopes, intercepts, supports, allowed, structured = ([] for _ in range(7))
    for leaf in plan['physical_leaves']:
        for field in leaf['fields']:
            key = (tuple(leaf['path']), field['id'])
            binding, kind = field['binding'], field['binding']['kind']
            if kind == 'scalar_rlgc':
                base = tuple(_fixed(matrix(binding['base'][name + '_per_length'])) for name in _MATERIAL_NAMES)
                terms = []
                for term in binding['terms']:
                    source = term['source']
                    ref = source['parameter'] if source['kind'] == 'ref' else source['input']
                    refkey = parameter_key(ref)
                    slope, intercept = (1.0, 0.0) if source['kind'] == 'ref' else (quantity(source['slope']), quantity(source['intercept']))
                    support = None if source['kind'] == 'ref' else tuple(quantity(item) for item in source['support'])
                    bases = tuple(_fixed(np.zeros_like(a) if term[name] is None else matrix(term[name]))
                                  for name, a in zip(_MATERIAL_NAMES, base, strict=True))
                    terms.append(MaterialTerm(positions[refkey], slope, intercept, support, refkey in authorized, bases))
                pure = not np.any(base[0]) and all(not np.any(term.bases[0]) for term in terms)
                structured.append(MaterialField(key, base, ((),) * 4, tuple(terms), pure))
                continue
            if kind == 'constant':
                value, source_key = binding['value'], None
            elif kind in ('ref', 'affine'):
                source_key = parameter_key(binding['parameter'] if kind == 'ref' else binding['input'])
                value = records[source_key]['value']
            else:
                raise TypeError(f'unsupported physical binding {kind}')
            if value['type'] == 'rlgc':
                if kind == 'affine':
                    raise TypeError('scalar AffineMap cannot implicitly map a structured RLGC')
                base = tuple(_fixed(matrix(value[name + '_per_length'])) for name in _MATERIAL_NAMES)
                operands = ((),) * 4 if source_key is None else tuple(structured_positions[source_key, name + '_per_length'] for name in _MATERIAL_NAMES)
                # A runtime whole-material ref is never specialized from its
                # present R=0; its active resistance tangent remains available.
                pure = source_key is None and not np.any(base[0])
                coefficient_base = base if source_key is None else tuple(_fixed(np.zeros_like(part)) for part in base)
                structured.append(MaterialField(key, coefficient_base, operands, (), pure))
                continue
            source = positions.get(source_key, -1)
            if kind == 'affine':
                slope, intercept = quantity(binding['slope']), quantity(binding['intercept'])
                support = tuple(quantity(item) for item in binding['support'])
            else:
                slope, intercept = (1.0, 0.0) if source >= 0 else (0.0, quantity(value))
                support = None
            keys.append(key); sources.append(source); slopes.append(slope)
            intercepts.append(intercept); supports.append(support)
            allowed.append(source_key in authorized)
    return FieldProgram(layout, tuple(keys), tuple(sources), tuple(slopes),
                        tuple(intercepts), tuple(supports), tuple(allowed), tuple(structured))
