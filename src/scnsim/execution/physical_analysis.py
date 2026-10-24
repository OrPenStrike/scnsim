"""Ordinary physical-program execution and detached numerical body projection.

The caller owns request lookup, operation leases, persistence and final decoding.
This adapter owns requested-point occurrences and the existing baseline-anchored
continuation policy. Every value/branch uses one compiled QuantityProgram; it
never constructs a second physical model or runs an optimizer.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace

import numpy as np

from ..canonical import canonical_json_bytes
from ..authoring.identity import canonical_parameters_sha256
from ..compilation.compiler import parameter_key, parameter_values
from ..compilation.mesh import quantity
from ..compilation.quantity_declarations import quantity_declarations
from ..compilation.quantity_program import compile_point_quantity_program
from ..errors import SCNSimError
from ..numeric_encoding import array_from_record, array_record, bits, record_bytes, record_document
from ..numerics.evidence import evidence_bytes, numerical_error
from ..numerics.models import EvaluationResult, NumericalFailure
from .quantities import (
    EvaluationFailure, continue_root, quantity_body_id, quantity_record,
)
from .quantity_policy import diagonal_passive_policy
from .policy import ExecutionPolicy


def _auxiliary_by_ordinal(rows):
    """Index synchronized auxiliary rows by their host-side occurrence ID."""
    return {int(np.asarray(ordinal).item()): value for ordinal, value in rows}


def execute_native_branch_packet(native, candidate, start_index, seed, *, backend,
                                 program_owner, operation_resources=None, compact_evidence=False):
    from ..numerics.native_sparse import SparseEquationCheck, checked_native
    from ..numerics.evidence import native_primal_exception, project_native_root

    import jax.numpy as jnp

    # Construct selected complex operands under the same x64/device scope
    # used by execute_native. JAX otherwise truncates complex128 inputs at
    # creation time when the ambient x64 setting is disabled.
    with backend.jax.enable_x64(True), backend.jax.default_device(backend.device):
        seed = jnp.asarray(seed, dtype=backend.complex_dtype)
    operands = (candidate, seed)
    branch = checked_native(native.branch_function(start_index))
    compile_key = native.branch_variant_key(
        start_index, device=backend.device, operands=operands,
        transform='checked_value',
    )
    candidate_scope = (nullcontext() if operation_resources is None else
                       operation_resources.candidate_compute())
    with candidate_scope:
        error, packet = backend.execute_native(
            branch, *operands, precision=backend.precision,
            compile_key=compile_key, owner=program_owner,
            phase='native_root_branch', details={'start_index': start_index},
        )
    exception = native_primal_exception(error)
    if exception is not None and not isinstance(exception, SparseEquationCheck):
        error.throw()
    row = next(row for row in native.root_layout if row['start_index'] == start_index)
    result = project_native_root(row['declaration'], packet, compact_evidence=compact_evidence)
    return result, packet, error



def _native_chunk_sparse_exception(exception):
    """Recognize only owned equation leaves; keep mixed batch errors throwable."""
    from jax._src.checkify import BatchedError
    from jax.tree_util import PyTreeDef
    from ..numerics.native_sparse import ArchivedPrimalCheck, SparseEquationCheck

    if isinstance(exception, SparseEquationCheck):
        return True
    if type(exception) is BatchedError:
        rows = exception.error_mapping
        return (type(rows) is dict and bool(rows)
                and all(type(index) is tuple and all(type(i) is int and i >= 0 for i in index)
                        and _native_chunk_sparse_exception(value)
                        for index, value in rows.items()))
    if type(exception) is ArchivedPrimalCheck:
        if not isinstance(exception.original_tree, PyTreeDef) or len(exception.payload) < 2:
            return False
        try:
            original = exception.original_tree.unflatten(exception.payload[:-2])
        except (TypeError, ValueError):
            return False
        return _native_chunk_sparse_exception(original)
    return False


def _raise_unowned_native_chunk_error(error):
    """Leave owned lane failures to the original ordered status reducers."""
    from ..numerics.evidence import native_error_may_raise

    if native_error_may_raise(error):
        exception = error.get_exception()
        if exception is not None and not _native_chunk_sparse_exception(exception):
            error.throw()


def acquire_native_root(program, native, candidate, row, seed, *, backend,
                        program_owner, operation_resources=None, compact_evidence=False,
                        discover=True):
    """Discover once at the initial point; later points use owned anchors."""
    from ..errors import RuntimePreparationError
    from ..numerics.root_search import complex_frequency, select_declared_root
    from ..numerics.native_polynomial import transmission_zero_spectrum
    from ..numerics.native_sparse import checked_native
    spec = row['spec']
    search = spec.get('search')
    def correct(omega):
        result,packet,error = execute_native_branch_packet(
            native,candidate,row['start_index'],omega,backend=backend,
            program_owner=program_owner,operation_resources=operation_resources,
            compact_evidence=compact_evidence)
        actual_errors[id(packet)] = error
        return result,packet
    actual_errors = {}
    if search is None or not discover:
        result,packet = correct(seed)
        return result,packet,actual_errors[id(packet)],None
    hint = complex_frequency(spec['root_hint'] if 'root_hint' in spec else spec['anchor'])
    catalogue = None
    if search['method'] == 'polynomial_transfer_zero':
        if row['kind'] != 'transfer_zero' or row['declaration'].family != 'Z':
            raise RuntimePreparationError(
                'Polynomial discovery supports intrinsic Z transmission zeros only',
                stage='root_search_prepare')
        view_index = row['view_index']
        function = checked_native(native.polynomial_coefficient_function(view_index))
        operands = (candidate,)
        error, coefficients = backend.execute_native(
            function, *operands, precision=backend.precision,
            compile_key=native.variant_key('polynomial_coefficients',
                device=backend.device, operands=operands, view_index=view_index,
                transform='checked_coefficients'), owner=program_owner,
            phase='native_root_search_coefficients')
        error.throw()
        if int(np.asarray(coefficients[3])) != 0:
            from ..numerics.jax_core import FAILURES
            status = int(np.asarray(coefficients[3]))
            kind,stage = FAILURES[status]
            failure = NumericalFailure(kind,stage,
                'Original assembly coefficient formation failed',
                evidence_bytes({'status':status,'phase':'root_search_coefficients'}))
            return EvaluationResult(row['declaration'].quantity_key,failure=failure),None,None,None
        frequency = native.core.frequency_programs[view_index]
        scope = (nullcontext() if operation_resources is None else
                 operation_resources.exclusive_assembly())
        with scope:
            spectrum = transmission_zero_spectrum(frequency.assembly.pattern,
                coefficients[:3], frequency.view_program.selected,
                input_index=row['declaration'].input_index,
                output_index=row['declaration'].output_index,
                reference_omega_rad_s=abs(hint.real)*2*np.pi)
        catalogue = spectrum._asdict()
        candidates = tuple(complex(value) if defined else None for value,defined
                           in zip(spectrum.omega_rad_s,spectrum.defined,strict=True))
    elif search['method'] == 'finite_seeds':
        candidates = tuple(complex_frequency(value)*2*np.pi for value in search['seeds'])
    else:
        raise RuntimePreparationError('Unsupported root discovery method',stage='root_search_prepare')
    result,packet,facts = select_declared_root(search,hint,candidates,
        correct,
        catalogue=catalogue)
    return result,packet,(None if packet is None else actual_errors[id(packet)]),facts


def acquire_completed_root_frame(program, inputs, *, backend, program_owner,
                                 operation_resources=None, compact_evidence=False,
                                 discover=False, affine_evidence=None):
    """Acquire current-point packets/archives once for value and local AD.

    Only original-point acquisition enables declared discovery. Later points
    use supplied branch anchors; each owns its own checked packets and Errors.
    The caller releases this working bundle after synchronous consumption.
    """
    from ..numerics.evidence import require_native_candidate
    native = program.native_program(inputs,program_owner=program_owner,admit_core=True,
                                    affine_evidence=affine_evidence)
    borrowed = (nullcontext() if program_owner is None else
                program_owner.hold_exact_core(native.core))
    completed,errors,results,searches = [],[],[],[]
    with borrowed,backend.jax.enable_x64(True),backend.jax.default_device(backend.device):
        theta = backend.jax.numpy.asarray(inputs.theta,dtype=backend.jax.numpy.float64)
        operands = (theta,)
        candidate = backend.execute_native(native.candidate_function,*operands,
            precision=backend.precision,compile_key=native.variant_key('candidate',
                device=backend.device,operands=operands),owner=program_owner,
            phase='native_completed_root_candidate')
        require_native_candidate(candidate)
        for row in native.root_layout:
            start = row['start_index']
            result,packet,error,facts = acquire_native_root(program,native,candidate,row,
                inputs.branch_starts_rad_s[start],backend=backend,program_owner=program_owner,
                operation_resources=operation_resources,compact_evidence=compact_evidence,
                discover=discover)
            results.append((start,result))
            if facts is not None:
                searches.append((start,facts))
            if result.failure is not None:
                raise numerical_error(result.failure)
            completed.append((start,packet));errors.append(error)
    return native,(tuple(completed),tuple(errors)),tuple(results),tuple(searches)


class ChunkedResultSink:
    """Bounded operation-local staging for one Direct or Sweep completion.

    Numerical rows are stored as checked immutable SQLite blocks. Only
    immutable byte-range references remain resident while computation runs;
    publication reads one Direct chunk or one budget-sized Sweep group at a
    time. The sink has no Workspace transaction or success-selection authority.
    """

    def __init__(self, spool, *, domain, scratch_budget_bytes, sweep_grid_shape=None):
        if domain not in {"direct", "sweep"}:
            raise ValueError(f"unsupported chunked result domain {domain!r}")
        self._spool = spool
        self._domain = domain
        self._scratch_budget_bytes = scratch_budget_bytes
        self._sweep_grid_shape = (
            None if sweep_grid_shape is None else tuple(sweep_grid_shape)
        )
        self._references = []
        self._native_work_references = {}
        self._point_count = 0
        self._summary_reference = None
        self._continuations = {}
        self._metadata = None

    @property
    def domain(self):
        return self._domain

    @property
    def metadata(self):
        if self._metadata is None:
            raise RuntimeError("chunked result metadata was not supplied by its compiled program")
        return self._metadata

    def set_metadata(self, metadata):
        if not isinstance(metadata, Mapping):
            raise TypeError("chunked result metadata must be a mapping")
        self._metadata = dict(metadata)

    def set_affine_evidence(self, rows):
        """Retain the actual Direct check's conditional facts in its fixed metadata."""
        self._metadata['affine_evidence'] = rows

    @property
    def point_count(self):
        return self._point_count

    def set_summary(self, summary):
        if not isinstance(summary, Mapping):
            raise TypeError("chunked terminal summary must be a mapping")
        self._summary_reference = self._spool.put_record(dict(summary))

    def merge_summary(self, summary):
        if not isinstance(summary, Mapping):
            raise TypeError("chunked terminal summary must be a mapping")
        current = self.summary()
        current.update(summary)
        self.set_summary(current)

    def summary(self):
        if self._summary_reference is None:
            return {}
        return self._spool.get_record(self._summary_reference)

    def append_direct_chunk(self, *, frequencies_hz, families, evidence):
        if self._domain != "direct":
            raise TypeError("Direct data cannot be appended to a Sweep sink")
        if set(families) != {"S", "Y", "Z"}:
            raise ValueError("a Direct chunk must contain its declared S, Y, and Z families")
        frequencies = np.asarray(frequencies_hz)
        family_records = {name: np.asarray(families[name]) for name in ("S", "Y", "Z")}
        count = len(np.asarray(frequencies_hz))
        source_chunk = {
            "point_count": count,
            "payload": {"frequencies_hz": frequencies, "families": family_records},
            "summary_json": {"evidence": dict(evidence)},
        }
        self._references.append(self._spool.put_record(source_chunk))
        self._point_count += count

    def append_direct_record(self, chunk):
        """Stage one already canonical Direct chunk from the native provider."""
        if self._domain != "direct":
            raise TypeError("Direct data cannot be appended to a Sweep sink")
        if not isinstance(chunk, Mapping):
            raise TypeError("Direct chunk must be a mapping")
        point_count = chunk.get("point_count")
        payload = chunk.get("payload")
        summary_json = chunk.get("summary_json")
        if (not isinstance(point_count, int) or isinstance(point_count, bool)
                or point_count <= 0 or not isinstance(payload, Mapping)
                or not isinstance(summary_json, Mapping)):
            raise TypeError("Direct chunk has incomplete source fields")
        self._references.append(self._spool.put_record(dict(chunk)))
        self._point_count += point_count

    def stage_direct_work(self, work_index, record):
        """Keep one synchronized native work row in scratch, not in RAM history."""
        if self._domain != "direct":
            raise TypeError("native frequency work belongs to a Direct sink")
        self.stage_native_work(None, work_index, record)

    def stage_native_work(self, namespace, work_index, record):
        """Stage one completed native frequency row under its candidate scope."""
        if type(work_index) is not int or work_index < 0:
            raise TypeError("native work index must be a non-negative integer")
        key = work_index if namespace is None else (namespace, work_index)
        if key in self._native_work_references:
            raise RuntimeError("native work index was staged more than once in its scope")
        if not isinstance(record, Mapping):
            raise TypeError("native work row must be a mapping")
        self._native_work_references[key] = self._spool.put_record(dict(record))

    def native_work(self, namespace, work_index):
        """Read one synchronized native row from the active candidate scope."""
        key = work_index if namespace is None else (namespace, work_index)
        try:
            reference = self._native_work_references[key]
        except KeyError as error:
            raise RuntimeError("native work row was not staged") from error
        return self._spool.get_record(reference)

    def release_native_work(self, namespace):
        """Drop only references for a completed candidate; scratch owns the bytes."""
        if namespace is None:
            return
        for key in tuple(self._native_work_references):
            if isinstance(key, tuple) and key[0] == namespace:
                del self._native_work_references[key]

    def direct_work(self, work_index):
        """Read a single staged native work row while assembling output chunks."""
        if self._domain != "direct":
            raise TypeError("native frequency work belongs to a Direct sink")
        return self.native_work(None, work_index)

    def append_sweep_point(self, point):
        if self._domain != "sweep":
            raise TypeError("Sweep point cannot be appended to a Direct sink")
        if not isinstance(point, Mapping):
            raise TypeError("Sweep occurrence must be a mapping")
        ordinal = self._point_count
        if "ordinal" in point and point["ordinal"] != ordinal:
            raise ValueError("Sweep occurrence ordinal differs from its source order")
        body = dict(point)
        body["ordinal"] = ordinal
        if self._sweep_grid_shape is not None:
            # Publication uses Cartesian request coordinates; scheduling stays flat.
            remaining = ordinal
            indices = []
            for size in reversed(self._sweep_grid_shape):
                remaining, index = divmod(remaining, size)
                indices.append(index)
            body["source_index"] = list(reversed(indices))
        continuation_refs = self._continuations.pop(ordinal, ())
        if continuation_refs:
            body["continuations"] = [
                self._spool.get_record(reference) for reference in continuation_refs
            ]
        reference = self._spool.put_record(body)
        self._references.append(reference)
        self._point_count += 1
        return reference

    def sweep_point(self, reference):
        """Read one occurrence from scratch for exact-value cache reuse."""
        if self._domain != "sweep":
            raise TypeError("Sweep point cannot be read from a Direct sink")
        return self._spool.get_record(reference)

    def cache_sweep_point(self, point):
        """Keep a single immutable point record available for exact cache hits."""
        if self._domain != "sweep":
            raise TypeError("Sweep cache cannot be used with a Direct sink")
        return self._spool.put_record(dict(point))

    def append_sweep_continuation(self, ordinal, record):
        if self._domain != "sweep":
            raise TypeError("continuation rows belong to Sweep results")
        if type(ordinal) is not int or ordinal < 0:
            raise TypeError("Sweep continuation ordinal must be a non-negative integer")
        if ordinal < self._point_count:
            raise RuntimeError("Sweep continuation arrived after its requested occurrence was staged")
        if not isinstance(record, Mapping):
            raise TypeError("Sweep continuation evidence must be a mapping")
        reference = self._spool.put_record(dict(record))
        self._continuations.setdefault(ordinal, []).append(reference)

    def chunks(self, snapshot=None):
        """Yield the canonical chunk records required by the atomic publisher."""
        if self._continuations:
            raise RuntimeError(
                "Sweep continuation evidence has no matching requested occurrence"
            )
        if self._domain == "direct":
            for reference in self._references:
                yield self._spool.get_record(reference, snapshot=snapshot)
            return

        pending = []
        pending_bytes = 0
        for reference in self._references:
            if pending and pending_bytes + reference.byte_length > self._scratch_budget_bytes:
                yield {"point_count": len(pending), "payload": {"points": pending},
                       "summary_json": {}}
                pending = []
                pending_bytes = 0
            pending.append(self._spool.get_record(reference, snapshot=snapshot))
            pending_bytes += reference.byte_length
        if pending:
            yield {"point_count": len(pending), "payload": {"points": pending},
                   "summary_json": {}}

_ROOT_TYPES = frozenset(('diagonal_root', 'operator_element_root', 'hybridized_pole',
                         'transfer_zero', 'residue_normalized_coupling'))
_NUMERICAL_FAILURES = frozenset(('direct_response_formation', 'port_realizability',
                                'eliminated_block_solve_failure', 'root_slope_unresolved',
                                'numerical_resolution_unresolved',
                                'unsupported_singular_capacitance_for_diagonal_root_v1'))


def point_identity(point):
    return canonical_json_bytes({
        'bindings': sorted(point['bindings'], key=lambda row: parameter_key(row['parameter'])),
        'allow_extrapolation': sorted(point['allow_extrapolation'], key=parameter_key),
    })


def _point_at(template, values):
    return dict(template, bindings=[
        dict(row, value=values[parameter_key(row['parameter'])]
             if isinstance(values[parameter_key(row['parameter'])], dict)
             else dict(row['value'], si_value_f64=bits(values[parameter_key(row['parameter'])])))
        for row in template['bindings']
    ])


def program_evidence(outcome, context):
    """Detach the actual same-frame primal certificates, never derivative claims."""
    facts = {
        'certificates': tuple({field.name: getattr(row, field.name) for field in fields(row)}
                              for row in outcome.aux.certificates),
        'realized_capabilities': outcome.aux.realized_capabilities,
        'terminal_ids':context['terminal_ids'],
        'axes':{key:context[key] for key in ('probe_loads','reference_coordinate_ids',
                                           'reference_impedances_ohm','load_mask')},
    }
    if 'row_coordinates' in context:
        facts.update({key: context[key] for key in ('row_coordinates', 'column_coordinates',
                      'zero_charge', 'coordinate_order', 'original_coordinate_order', 'finite_mesh')})
        locator = context['coordinate_transform_locator']
        weights = _auxiliary_by_ordinal(outcome.aux.static_facts)[locator['observable_index']]
        facts['coordinate_transform'] = tuple(tuple(np.asarray(value) for value in pair) for pair in weights)
    return evidence_bytes(facts)


def point_contexts(program, inputs):
    """Detach actual binary64 View lineage and realized sections per observable."""
    metadata = program.point_metadata(inputs)
    views = {row['view_index']: row for row in metadata['views']}
    observables = {row['observable_index']: row for row in program.observable_metadata()}
    return {row['observable_index']: {
        'lineage': views[row['view_index']]['lineage'],
        **{key:views[row['view_index']][key] for key in ('terminal_ids','probe_loads',
              'reference_coordinate_ids','reference_impedances_ohm','load_mask')},
        'discretization': metadata['discretization'],
        'root_search_requests': metadata['root_search_requests'],
        **({key: observables[row['observable_index']][key] for key in ('row_coordinates', 'column_coordinates', 'zero_charge',
             'coordinate_order', 'original_coordinate_order', 'coordinate_transform_locator', 'finite_mesh')}
           if observables[row['observable_index']]['kind'].startswith('static_capacitance') else {}),
    } for row in metadata['observables']}


def project_primal_bodies(program, outcome, inputs, *, ephemeral=False):
    """Numerical body encoding authority shared by ordinary and optimizer sinks.

    Locators refer exclusively to the supplied synchronized graph outcome. No
    selector evaluation, sparse solve, objective reduction or View lowering runs.
    """
    metadata = program.observable_metadata()
    contexts = point_contexts(program, inputs)
    branches = _auxiliary_by_ordinal(outcome.aux.branch_results)
    quantities = _auxiliary_by_ordinal(outcome.aux.quantity_results)
    bodies = {}
    for row in metadata:
        locator, index = row['primal_locator'], row['observable_index']
        identity = f"{row['quantity_key']}:observable:{index}"
        if locator['kind'] == 'branch_results':
            result = replace(branches[locator['start_index']], id=identity)
        elif locator['kind'] == 'quantity_results':
            result = replace(quantities[locator['observable_index']], id=identity)
        else:
            raw = np.asarray(outcome.aux.observable_raw_values[locator['observable_index']])
            field = (row['kind'] if row['kind'] in ('S', 'Y', 'Z') else
                     'operator_values' if row['kind'] == 'operator' else
                     'capacitance_values' if row['kind'].startswith('static_capacitance') else 'response_value')
            result = EvaluationResult(identity, **{field: raw if raw.shape else raw[()]},
                                      evidence_bytes=program_evidence(outcome, contexts[index]))
        bodies[index] = result
    if ephemeral:
        # RAM owners retain the same typed bodies and occurrence metadata;
        # durable branch body identifiers are not fabricated for this execution.
        return bodies
    for row in metadata:
        if row['kind'] != 'residue_coupling':
            continue
        dependencies = []
        for role, spec in (('a', row['spec']['branch_a']), ('b', row['spec']['branch_b'])):
            branch = next(item for item in metadata if item['start_index'] is not None
                          and canonical_json_bytes(item['spec']) == canonical_json_bytes(spec)
                          and canonical_json_bytes(item['view']) == canonical_json_bytes(row['view']))
            dependencies.append({'role': role, 'binding': {'kind': spec['type'], 'spec': dict(spec), 'view': dict(row['view'])},
                                 'body_id': quantity_body_id(bodies[branch['observable_index']])[0]})
        result = bodies[row['observable_index']]
        evidence = record_document(result.evidence_bytes)
        evidence['branches'] = dependencies
        bodies[row['observable_index']] = replace(result, evidence_bytes=record_bytes(evidence))
    return bodies


class PhysicalAnalysis:
    """One request-local static program; points and outcomes are never retained in Run."""

    def __init__(self, plan, analysis, backend, trace, emit, points, *, program_owner=None,
                 execution_policy=None, result_sink=None, operation_resources=None,
                 first_point=None):
        self.analysis, self.backend, self.trace, self.emit = analysis, backend, trace, emit
        self.execution_policy = (ExecutionPolicy() if execution_policy is None
                                 else execution_policy)
        self.result_sink = result_sink
        self.ephemeral = bool(getattr(result_sink, "in_memory_analysis", False))
        self.operation_resources = operation_resources
        self._native_slot_admission = None
        self._affine_point_key = None
        self._affine_evidence = None
        self.spec = analysis['spec']
        self.baseline = {
            'type': 'parameter_set_v2', 'allow_extrapolation': [],
            'bindings': [{'parameter': {'definitions_id': row['definitions_id'],
                                       'parameter_id': row['parameter_id']},
                          'value': row['baseline']}
                         for row in plan['parameter_closure']['definitions']],
        }
        self.baseline_values = parameter_values(self.baseline)
        self.anchors = {}
        self._root_search_metadata = {}
        self._plan = plan
        self._program_owner = program_owner
        def has_search(value):
            if isinstance(value, Mapping):
                return 'search' in value or any(has_search(item) for item in value.values())
            return isinstance(value, (tuple,list)) and any(has_search(item) for item in value)
        if (self.analysis['parameter_source']['kind'] == 'point'
                and first_point is not None and has_search(self.spec)):
            # An explicit search initializes the branch at the requested
            # original point, not a different Plan-default circuit.
            self.baseline = first_point
            self.baseline_values = parameter_values(first_point)
        initial_point = (self.baseline if self.spec['type'] in _ROOT_TYPES
                         or first_point is None else first_point)
        self._initial_point_key = point_identity(initial_point)
        self._active_point_key = None
        self._active_point_program = None
        self.program = self.program_at(initial_point)
        self.metadata = self.program.observable_metadata()
        self.roots = tuple(row for row in self.metadata if row['start_index'] is not None)
        if self.result_sink is not None and self.result_sink.domain == 'direct':
            self.result_sink.set_metadata(self._direct_metadata(initial_point))
        elif self.result_sink is not None and self.result_sink.domain == 'sweep':
            self.result_sink.set_metadata(self._sweep_metadata(initial_point))

    def _store_array(self, value):
        if self.ephemeral:
            from ..numerics.evidence import _detach_native
            return _detach_native(np.asarray(value))
        return array_record(np.asarray(value))

    def _load_array(self, value):
        return np.asarray(value) if self.ephemeral else array_from_record(value)

    def _anchor_references(self):
        if self.ephemeral:
            return {key: value.root_omega_rad_s for key, value in self.anchors.items()}
        return {key: quantity_body_id(value)[0] for key, value in self.anchors.items()}

    def _direct_metadata(self, point):
        """Bind the one-point Direct index to its exact compiled declaration."""
        if self.spec['type'] != 'direct_solve':
            raise TypeError('Direct result sink requires a direct_solve declaration')
        program = self.program_at(point)
        observable = program.observable_metadata()[0]
        context = point_contexts(program, program.inputs(point))[observable['observable_index']]
        descriptors = tuple(program.program.observables)
        first_view = descriptors[observable['observable_index']].view_index
        if any(row.view_index != first_view for row in descriptors):
            raise TypeError('Direct matrix families must share their declared View')
        return {
            'result_kind': self.spec['type'],
            'program_sha256': program.program_sha256,
            'backend_identity': self.backend.native_program_identity('direct'),
            'precision': self.backend.precision,
            'view': self.analysis['view'],
            'lineage': context['lineage'],
            'discretization': context['discretization'],
            'terminal_ids': tuple(context['terminal_ids']),
            'probe_loads': dict(context['probe_loads']),
        }

    def _sweep_metadata(self, point):
        """Bind Sweep compilation identity to each actual point occurrence."""
        observable = next((row for row in self.metadata if not row['hidden']), None)
        if observable is None:
            raise TypeError('Sweep result sink requires a declared public observable')
        contexts = point_contexts(self.program, self.program.inputs(point))
        context = contexts[observable['observable_index']]
        return {
            'result_kind': self.spec['type'],
            'program_identity_scope': 'per_point',
            'program_sha256': None,
            'backend_identity': self.backend.native_program_identity(
                'direct' if self.spec['type'] == 'direct_solve' else 'evaluate',
            ),
            'precision': self.backend.precision,
            'view': self.analysis['view'],
            'terminal_ids': tuple(context['terminal_ids']),
            'probe_loads': dict(context['probe_loads']),
        }

    def program_at(self, point):
        key = point_identity(point)
        if key != self._affine_point_key:
            from ..compilation.program import _AffineEvidence
            self._affine_evidence = _AffineEvidence(parameter_set=point['allow_extrapolation'])
            self._affine_point_key = key
        if key == self._active_point_key:
            return self._active_point_program
        if key == self._initial_point_key and hasattr(self, 'program'):
            return self.program
        with self.trace.span('physical_program_preparation'):
            program = compile_point_quantity_program(
                self._plan, quantity_declarations(self.spec, self.analysis['view']), point,
                default_view=self.analysis['view'], mesh=self.spec.get('discretization'),
                precision=self.backend.precision,
                authorized=frozenset(parameter_key(ref) for ref in point['allow_extrapolation']),
                program_variants={}, program_owner=self._program_owner,
            )
        # A parameter source can realize a different line mesh at every point.
        # Keep only the current point program here; stable admitted declarations
        # remain owned by CompileOwner and numeric values remain runtime inputs.
        self._active_point_key = key
        self._active_point_program = program
        return program

    def context(self, point):
        program = self.program_at(point)
        context = point_contexts(program, program.inputs(point))[0]
        if self._root_search_metadata:
            context = dict(context, root_search=tuple(self._root_search_metadata.values()))
        return context

    def _search_native_branch(self, program, native, candidate, row, seed):
        key = self._dependency_key(row)
        result,packet,error,facts = acquire_native_root(program,native,candidate,row,seed,
            backend=self.backend,program_owner=self._program_owner,
            operation_resources=self.operation_resources,compact_evidence=self.ephemeral,
            discover=key not in self.anchors)
        if facts is not None:
            self._root_search_metadata[key] = facts
        return result,packet,error

    @staticmethod
    def _dependency_key(row):
        return canonical_json_bytes({'spec': row['spec'], 'view': row['view'],
                                     'type': row['spec']['type']}).decode()

    def _dependencies(self, results):
        if self.ephemeral:
            return {('root_start', index): result for index, result in results.items()}
        return dict(quantity_body_id(result) for result in results.values())

    @staticmethod
    def _bind_affine_evidence(result, observation):
        """Detach this actual check's facts before the result enters a cache."""
        if observation is None or not observation.rows:
            return result
        return replace(result, evidence_bytes=record_bytes({
            **record_document(result.evidence_bytes),
            'affine_evidence': observation.rows,
        }))

    def evaluate(self, point, identity, *, baseline=False):
        branch_results = {}
        observation = None
        try:
            program = self.program_at(point)
            observation = self._affine_evidence
            if baseline or not self.roots:
                # The complete baseline/healthy value frame already returns
                # its actual root packets. Do not solve those roots in a
                # separate prepass and then repeat them in the aggregate.
                inputs = program.inputs(point)
                if any(row['spec'].get('search') for row in program.observable_metadata()
                       if row['start_index'] is not None):
                    native = program.native_program(inputs,program_owner=self._program_owner,admit_core=True,
                                                    affine_evidence=self._affine_evidence)
                    candidate = self._execute_native_candidate(native,inputs,operation='evaluate')
                    completed,completed_errors = [],[]
                    for row in native.root_layout:
                        result,packet,error = self._search_native_branch(program,native,candidate,row,
                            inputs.branch_starts_rad_s[row['start_index']])
                        branch_results[row['start_index']] = result
                        if result.failure is not None:
                            raise EvaluationFailure(result.failure,result=result,
                                dependencies=self._dependencies(branch_results))
                        completed.append((row['start_index'],packet))
                        completed_errors.append(error)
                    outcome = self._execute_native_completed_value(program,point,tuple(completed),
                        completed_branch_errors=tuple(completed_errors),operation='evaluate')
                else:
                    outcome = self._execute_native_value(program, inputs, operation='evaluate')
            else:
                # First evaluate the complete candidate once, using the
                # baseline roots only as Newton starts. Healthy roots and all
                # requested values therefore share this actual graph frame.
                starts = self._native_branch_starts(point, baseline=False)
                inputs = program.inputs(point, branch_starts_rad_s=starts)
                outcome, native, raw_frame = self._execute_native_value(
                    program, inputs, operation='evaluate', return_frame=True,
                )
                raw_result,current_error = raw_frame
                raw_aux = raw_result[1]
                packet_by_start = _auxiliary_by_ordinal(raw_aux.branch_packets)
                projected_by_start = _auxiliary_by_ordinal(outcome.aux.branch_results)
                packets,packet_errors = {},{}
                needs_completed_frame = False
                branch_packets_by_id = {}
                for row in native.root_layout:
                    start_index = row['start_index']
                    key = self._dependency_key(row)
                    initial = projected_by_start[start_index]
                    packet = packet_by_start[start_index]
                    from ..numerics.native_sparse import root_error_frame
                    declaration = row['declaration']
                    dependency = ('root_occurrence',row['view_index'],start_index,
                        (declaration.kind,declaration.coordinate_index,declaration.row_index,
                         declaration.column_index,declaration.family,declaration.output_index,
                         declaration.input_index))
                    packet_error = root_error_frame(current_error,dependency)
                    if initial.failure is not None:
                        if initial.failure.kind != 'numerical_resolution_unresolved':
                            branch_results[start_index] = initial
                            raise EvaluationFailure(
                                initial.failure, result=initial,
                                dependencies=self._dependencies(branch_results),
                            )

                        needs_completed_frame = True

                        def make_job(job_id, actual, start):
                            return SimpleNamespace(id=job_id, point=actual, omega=start)

                        def evaluate(jobs):
                            values_at = []
                            for job in jobs:
                                result_at, packet_at, error_at = self._native_branch_at(
                                    job.point, start_index, job.omega,
                                    operation='evaluate',
                                )
                                result_at = replace(result_at, id=job.id)
                                branch_packets_by_id[job.id] = (packet_at,error_at)
                                values_at.append(result_at)
                            return tuple(values_at)

                        def observe(values_at, actual, result_at, t):
                            record = quantity_record(result_at)
                            record.update(parameters=actual, origin='root_continuation',
                                          source_index=int(identity),
                                          continuation_t_f64=bits(t),
                                          **self.context(actual))
                            if self.result_sink is not None and self.result_sink.domain == 'sweep':
                                record['program_sha256'] = self.program_at(actual).program_sha256
                            self.emit('evaluation', record)

                        initial = continue_root(
                            baseline=self.baseline_values,
                            values=parameter_values(point),
                            baseline_root=self.anchors[key].root_omega_rad_s,
                            endpoint_view=point,
                            initial_result=initial,
                            candidate_view=lambda values_at: _point_at(point, values_at),
                            make_job=make_job, evaluate=evaluate,
                            identity=str(identity), observe=observe,
                        )
                        packet,packet_error = branch_packets_by_id[initial.id]
                    if row['kind'] == 'diagonal_root':
                        initial = diagonal_passive_policy(initial,
                            node_count=len(program.program.views[row['view_index']].node_ids),
                            precision=program.program.precision)
                    if initial.failure is not None:
                        branch_results[start_index] = initial
                        raise EvaluationFailure(
                            initial.failure, result=initial,
                            dependencies=self._dependencies(branch_results),
                        )
                    branch_results[start_index] = initial
                    packets[start_index] = packet
                    packet_errors[start_index] = packet_error
                if needs_completed_frame:
                    completed_branches = tuple(
                        (row['start_index'], packets[row['start_index']])
                        for row in native.root_layout
                    )
                    outcome = self._execute_native_completed_value(
                        program, point, completed_branches,
                        completed_branch_errors=tuple(packet_errors[row['start_index']] for row in native.root_layout),
                        operation='evaluate',
                    )
            program.require_primal(outcome)
            bodies = project_primal_bodies(program, outcome, inputs, ephemeral=self.ephemeral)
            branches = _auxiliary_by_ordinal(outcome.aux.branch_results)
            for row in self.roots:
                index = row['start_index']
                actual = branches[index]
                if row['kind'] == 'diagonal_root' and not (baseline and self.spec['type'] == 'diagonal_root'):
                    actual = diagonal_passive_policy(actual,
                        node_count=len(program.program.views[
                            program.program.observables[row['observable_index']].view_index].node_ids),
                        precision=program.program.precision)
                branches[index] = actual
                if actual.failure is not None:
                    raise EvaluationFailure(actual.failure, result=actual,
                                            dependencies=self._dependencies(branches))
            visible = [row for row in program.observable_metadata() if not row['hidden']]
            if self.spec['type'] == 'direct_solve':
                values = {row['kind']: getattr(bodies[row['observable_index']], row['kind']) for row in visible}
                result = EvaluationResult(identity, **values, evidence_bytes=program_evidence(
                    outcome, point_contexts(program, inputs)[visible[0]['observable_index']]))
            else:
                result = replace(bodies[visible[0]['observable_index']], id=identity)
            dependencies = {quantity_body_id(bodies[row['observable_index']])[0]:
                            quantity_body_id(bodies[row['observable_index']])[1]
                            for row in program.observable_metadata() if row['hidden']}
            if baseline:
                for row in self.roots:
                    anchor = bodies[row['observable_index']]
                    if not row['hidden']:
                        anchor = result
                    self.anchors[self._dependency_key(row)] = anchor
            return self._bind_affine_evidence(result, observation), dependencies
        except EvaluationFailure as error:
            result = error.result if error.result is not None else EvaluationResult(identity, failure=error.failure)
            result = replace(result, id=identity)
            dependencies = dict(error.dependencies)
            dependencies.pop(quantity_body_id(result)[0], None)
            return self._bind_affine_evidence(result, observation), dependencies
        except SCNSimError as error:
            if error.kind not in _NUMERICAL_FAILURES:
                raise
            failure = NumericalFailure(error.kind, error.stage, str(error), record_bytes(error.evidence))
            return self._bind_affine_evidence(
                EvaluationResult(identity, failure=failure), observation
            ), self._dependencies(branch_results)


    def record(self, point, result, dependencies, **occurrence):
        if self.ephemeral:
            context = self.context(point)
            program = self.program_at(point)
            if self.spec['type'] == 'direct_solve':
                from .memory_direct import direct_result_metadata
                context = dict(context, metadata=direct_result_metadata(
                    self._direct_metadata(point), self.spec,
                    self._store_array(np.asarray(
                        [quantity(value) for value in self.spec['frequencies']],
                        dtype=np.float64,
                    )),
                ))
            return dict(id=result.id, parameters=point, result=result,
                status='failure' if result.failure is not None else 'success',
                dependencies=tuple(dependencies.items()), context=context,
                program_sha256=program.program_sha256,
                **{'source_index':None,'cache_hit':False,**occurrence})
        record = quantity_record(result)
        context = self.context(point)
        record.update(parameters=point, dependencies=dependencies, **context, **occurrence)
        if self.result_sink is not None and self.result_sink.domain == 'sweep':
            record['program_sha256'] = self.program_at(point).program_sha256
            record['parameters_sha256'] = canonical_parameters_sha256(point)
        if self.spec['type'] in ('direct_solve', 'operator'):
            record['frequencies_hz'] = array_record(np.asarray(
                [quantity(value) for value in self.spec['frequencies']], dtype=np.float64))
        elif self.spec['type'] == 'response_element':
            record['frequencies_hz'] = array_record(np.asarray([quantity(self.spec['frequency'])], dtype=np.float64))
        return record

    def _native_work_batches(self, native):
        """Yield policy-sized batches followed by ordered power-of-two tails."""
        rows = tuple(native.work_layout)
        if not rows:
            raise RuntimeError('native program has no requested frequency work')
        group_key = None
        group = []
        for row in rows:
            key = (row['view_index'], tuple(row['demand']))
            if group and key != group_key:
                yield from self._native_batches_for_group(group)
                group = []
            group_key = key
            group.append(row)
        if group:
            yield from self._native_batches_for_group(group)

    def _native_batches_for_group(self, rows):
        """Yield full policy buckets followed by ordered power-of-two tails."""
        main_bucket = self.execution_policy.frequency_chunk_size
        offset = 0
        while len(rows) - offset >= main_bucket:
            yield tuple(rows[offset:offset + main_bucket])
            offset += main_bucket
        while offset < len(rows):
            remaining = len(rows) - offset
            bucket = 1 << (remaining.bit_length() - 1)
            yield tuple(rows[offset:offset + bucket])
            offset += bucket

    @staticmethod
    def _native_certificate_row(certificate, lane):
        values = {}
        for name in ('status', 'eta', 'residual_numerator', 'residual_denominator',
                     'performed', 'present'):
            values[name] = array_record(np.asarray(getattr(certificate, name))[lane])
        return {
            'site': certificate.site, 'role': certificate.role,
            'stage': certificate.stage, 'derivative_order': certificate.derivative_order,
            **values,
        }

    @staticmethod
    def _native_certificate_lane(certificate, lane):
        """Detach one failed lane while leaving successful certificates unencoded."""
        return replace(
            certificate,
            status=np.asarray(certificate.status)[lane],
            eta=np.asarray(certificate.eta)[lane],
            residual_numerator=np.asarray(certificate.residual_numerator)[lane],
            residual_denominator=np.asarray(certificate.residual_denominator)[lane],
            performed=np.asarray(certificate.performed)[lane],
            present=np.asarray(certificate.present)[lane],
        )

    def _pin_native_core(self, native, stack):
        if self._program_owner is not None:
            stack.enter_context(self._program_owner.hold_exact_core(native.core))

    def _execute_native_candidate(self, native, inputs, *, operation):
        from ..numerics.evidence import require_native_candidate

        operands = (inputs.theta,)
        candidate_key = native.variant_key(
            'candidate', device=self.backend.device, operands=operands,
        )
        candidate_scope = (nullcontext() if self.operation_resources is None else
                           self.operation_resources.candidate_compute())
        with candidate_scope:
            candidate = self.backend.execute_native(
                native.candidate_function, *operands,
                precision=self.backend.precision, compile_key=candidate_key,
                owner=self._program_owner, phase='native_candidate',
                details={'operation': operation},
            )
        require_native_candidate(candidate)
        return candidate

    def _execute_native_value(self, program, inputs, *, operation, return_frame=False):
        """Execute one checked whole-value frame through the native provider."""
        import jax.numpy as jnp
        from ..numerics.native_sparse import checked_native
        from ..numerics.evidence import project_native_outcome

        native = program.native_program(
            inputs, program_owner=self._program_owner, admit_core=True,
            affine_evidence=self._affine_evidence,
        )
        source_indices = tuple(native.input_layout['omega']['source_indices'])
        with self.backend.jax.enable_x64(True), self.backend.jax.default_device(self.backend.device):
            theta = jnp.asarray(inputs.theta, dtype=np.float64)
            omega_source = jnp.asarray(inputs.omega_rad_s, dtype=self.backend.complex_dtype)
            omega = omega_source[jnp.asarray(source_indices, dtype=np.int32)]
            seeds = jnp.asarray(inputs.branch_starts_rad_s, dtype=self.backend.complex_dtype)
        operands = (theta, omega, seeds)
        function = checked_native(native.numeric_function)
        compile_key = native.variant_key(
            'value', device=self.backend.device, operands=operands,
            transform='checked_value',
        )
        candidate_scope = (nullcontext() if self.operation_resources is None else
                           self.operation_resources.candidate_compute())
        with ExitStack() as stack:
            self._pin_native_core(native, stack)
            with candidate_scope:
                checked_error, result = self.backend.execute_native(
                    function, *operands, precision=self.backend.precision,
                    compile_key=compile_key, owner=self._program_owner,
                    phase='native_physical_value', details={'operation': operation},
                )
            outcome = project_native_outcome(native, result, checked_error, compact_evidence=self.ephemeral)
        if return_frame:
            # The detached branch values explain typed outcomes; this short
            # lived raw frame is needed only when an unresolved endpoint must
            # continue from its actual final root packet.
            return outcome, native, (result,checked_error)
        return outcome

    def _execute_native_completed_value(self, program, point, completed_branches, *, completed_branch_errors, operation):
        """Finish one endpoint frame from actual continuation packets, without re-rooting."""
        import jax.numpy as jnp
        from ..numerics.native_sparse import checked_native
        from ..numerics.evidence import project_native_outcome

        starts = self._native_branch_starts(point, baseline=not self.anchors)
        for start_index, packet in completed_branches:
            starts[start_index] = np.asarray(packet.omega).item()
        inputs = program.inputs(point, branch_starts_rad_s=starts)
        native = program.native_program(
            inputs, program_owner=self._program_owner, admit_core=True,
            affine_evidence=self._affine_evidence,
        )
        branch_starts = tuple(start for start, _ in completed_branches)
        packets = tuple(packet for _, packet in completed_branches)
        source_indices = tuple(native.input_layout['omega']['source_indices'])
        with self.backend.jax.enable_x64(True), self.backend.jax.default_device(self.backend.device):
            omega_source = jnp.asarray(inputs.omega_rad_s, dtype=self.backend.complex_dtype)
            omega = omega_source[jnp.asarray(source_indices, dtype=jnp.int32)]
        candidate = self._execute_native_candidate(native, inputs, operation=operation)

        errors = completed_branch_errors
        def complete(candidate_state, frequencies, actual_packets, actual_errors):
            ordered = tuple(zip(branch_starts, actual_packets, strict=True))
            return native.completed_primal_function(candidate_state, frequencies, ordered, actual_errors)

        operands = (candidate, omega, packets, errors)
        function = checked_native(complete)
        compile_key = native.variant_key(
            'value', device=self.backend.device, operands=operands,
            transform=('checked_completed_value',str(self.backend.jax.tree.structure(errors))),
        )
        candidate_scope = (nullcontext() if self.operation_resources is None else
                           self.operation_resources.candidate_compute())
        with ExitStack() as stack:
            self._pin_native_core(native, stack)
            with candidate_scope:
                error, result = self.backend.execute_native(
                    function, *operands, precision=self.backend.precision,
                    compile_key=compile_key, owner=self._program_owner,
                    phase='native_completed_physical_value',
                    details={'operation': operation},
                )
            return project_native_outcome(native, result, error, compact_evidence=self.ephemeral)

    def _execute_native_chunk(self, native, candidate, rows, omega_source, *,
                              prior_check_status, prior_certificate_status,
                              operation):
        """Run one exact frequency bucket and return its synchronized outcome."""
        jnp = self.backend.jax.numpy
        view_index = rows[0]['view_index']
        demand = tuple(rows[0]['demand'])
        bucket = len(rows)
        with self.backend.jax.enable_x64(True), self.backend.jax.default_device(self.backend.device):
            # Gather exact work order on the host before one typed device conversion.
            omega_indices = np.fromiter(
                (row['omega_index'] for row in rows), dtype=np.intp, count=bucket,
            )
            omega = jnp.asarray(
                omega_source[omega_indices],
                dtype=self.backend.complex_dtype,
            )
            prior_check = jnp.asarray(prior_check_status, dtype=np.int32)
            prior_certificate = jnp.asarray(prior_certificate_status, dtype=np.int32)
        operands = (candidate, omega, prior_check, prior_certificate)
        transform = ('ordered_tail'
                     if bucket < self.execution_policy.frequency_chunk_size else 'value')
        function = native.chunk_function(view_index, demand, bucket, transform=transform)
        compile_key = native.variant_key(
            'chunk', device=self.backend.device, operands=operands,
            view_index=view_index, demand=demand, bucket=bucket, transform=transform,
        )
        candidate_scope = (nullcontext() if self.operation_resources is None else
                           self.operation_resources.candidate_compute())
        with candidate_scope:
            outcome = self.backend.execute_native(
                function, *operands, precision=self.backend.precision,
                compile_key=compile_key, owner=self._program_owner,
                phase='native_frequency_chunk',
                details={'operation': operation, 'view_index': view_index,
                         'demand': demand, 'bucket': bucket,
                         'first_source_index': rows[0]['source_index'],
                         **(self._native_slot_admission or {})},
            )
        _raise_unowned_native_chunk_error(outcome.errors)
        return outcome

    def _native_frequency_slots(self, native):
        """Choose Direct slots from a truthful per-chunk scratch estimate.

        The native callback's sparse-factor fill and backend temporary working
        set are not exposed as statically sized buffers. The returned matrix
        arrays provide an exact lower bound, but not a total scratch estimate;
        an unknown total therefore schedules one chunk at a time. A single
        legal chunk is still run if even its known output lower bound exceeds
        the configured budget.
        """
        capacity = (1 if self.operation_resources is None else
                    self.operation_resources.worker_capacity)
        requested = self.execution_policy.max_in_flight
        requested_limit = capacity if requested is None else min(capacity, requested)
        matrix_lower_bound = 0
        over_budget_lower_bound = False
        complex_bytes = np.dtype(self.backend.complex_dtype).itemsize
        for rows in self._native_work_batches(native):
            view_index = rows[0]['view_index']
            frequency = native.core.frequency_programs[view_index]
            q = len(frequency.view_program.terminal_ids)
            # S/Y/Z have the selected terminal shape. The operator result is
            # either its declared selected Schur shape or a typed empty matrix
            # with the terminal shape when it is not requested.
            demand = tuple(rows[0]['demand'])
            operator_extent = (len(frequency.view_program.selected) ** 2
                               if 'operator' in demand else q * q)
            known = len(rows) * (3 * q * q + operator_extent) * complex_bytes
            matrix_lower_bound = max(matrix_lower_bound, known)
            over_budget_lower_bound |= known > self.execution_policy.scratch_budget_bytes

        # Sparse LU fill and native execution temporaries are call-local and
        # depend on runtime numeric factorization. No honest total estimate is
        # available from the immutable sparse pattern alone, so this is the
        # specified conservative one-slot case rather than a guessed n^2 cap.
        slots = 1
        admission = {
            'frequency_slots': slots,
            'frequency_slot_limit': requested_limit,
            'scratch_budget_bytes': self.execution_policy.scratch_budget_bytes,
            'scratch_estimate_bytes': None,
            'scratch_estimate_unknown': 'native_sparse_factor_fill_and_backend_working_set',
            'known_matrix_output_bytes_lower_bound': matrix_lower_bound,
            'known_single_chunk_exceeds_scratch_budget': bool(over_budget_lower_bound),
        }
        self._native_slot_admission = admission
        if self.operation_resources is not None:
            self.operation_resources.statistic('native_frequency_slot_count', slots)
            self.operation_resources.statistic('native_frequency_scratch_estimate_unknown_count')
            self.operation_resources.statistic(
                'native_frequency_known_matrix_output_bytes_lower_bound', matrix_lower_bound,
            )
            if over_budget_lower_bound:
                self.operation_resources.statistic(
                    'native_frequency_known_single_chunk_over_budget_count',
                )
        return slots

    def _execute_native_work(self, point, *, operation, validate_work,
                             consume_work, failure_certificates,
                             include_roots=False, branch_starts_rad_s=None,
                             frequency_slots=1, prepare_work=None,
                             raw_failure_certificates=False):
        """Execute one candidate and its ordered native frequency work once."""
        from ..numerics.native_sparse import sparse_certificates
        from ..numerics.evidence import require_native_frequency

        program = self.program_at(point)
        inputs = program.inputs(point, branch_starts_rad_s=branch_starts_rad_s)
        native = program.native_program(
            inputs, program_owner=self._program_owner, admit_core=True,
            affine_evidence=self._affine_evidence,
        )
        work_rows = tuple(native.work_layout)
        source_indices = tuple(native.input_layout['omega']['source_indices'])
        omega_source = np.asarray(inputs.omega_rad_s)[np.asarray(source_indices, dtype=np.intp)]
        if (not work_rows and not native.root_layout) or any(
            type(row['omega_index']) is not int or row['omega_index'] < 0
            or row['omega_index'] >= len(omega_source) for row in work_rows
        ):
            raise RuntimeError('native program has no declared work or its frequency work is outside its operands')
        validate_work(program, inputs, native, work_rows)
        if prepare_work is not None:
            prepare_work(program, inputs, native, work_rows)
        if frequency_slots is None:
            frequency_slots = self._native_frequency_slots(native)

        with ExitStack() as stack:
            self._pin_native_core(native, stack)
            candidate = self._execute_native_candidate(native, inputs, operation=operation)

            check_status = 0
            certificate_status = 0
            check_position = -1
            certificate_position = -1
            executed_rows = []
            failed_rows = []
            from itertools import islice

            batches = iter(self._native_work_batches(native)) if work_rows else iter(())
            while True:
                window = tuple(islice(batches, max(1, frequency_slots)))
                if not window:
                    break
                window_check = check_status
                window_certificate = certificate_status

                def execute_item(item):
                    rows = item
                    return self._execute_native_chunk(
                        native, candidate, rows, omega_source,
                        prior_check_status=window_check,
                        prior_certificate_status=window_certificate,
                        operation=operation,
                    )

                if len(window) > 1 and self.operation_resources is not None:
                    outcomes = self.operation_resources.map_ordered(
                        execute_item, window,
                        identity=lambda rows: rows[0]['work_index'],
                    )
                else:
                    outcomes = map(execute_item, window)

                for rows, outcome in zip(window, outcomes, strict=True):
                    certificates = (None if raw_failure_certificates else
                                    sparse_certificates(outcome.errors))
                    family_arrays = {}
                    local_check = np.asarray(outcome.outcomes.check_status)
                    local_certificate = np.asarray(outcome.outcomes.certificate_status)
                    local_check_positions = np.asarray(outcome.outcomes.check_position)
                    local_certificate_positions = np.asarray(outcome.outcomes.certificate_position)
                    executed = np.asarray(outcome.executed, dtype=np.bool_)

                    def family_array(family):
                        # Materialize each requested family once, on first use.
                        if family not in family_arrays:
                            family_arrays[family] = np.asarray(
                                getattr(outcome.outcomes, family)
                            )
                        return family_arrays[family]

                    for lane, row in enumerate(rows):
                        if not executed[lane]:
                            continue
                        check_value = int(local_check[lane])
                        certificate_value = int(local_certificate[lane])
                        lane_failed = bool(check_value or certificate_value)
                        if not raw_failure_certificates:
                            executed_rows.append(row)
                        if raw_failure_certificates:
                            if lane_failed:
                                if certificates is None:
                                    certificates = sparse_certificates(outcome.errors)
                                certificate_rows = tuple(
                                    self._native_certificate_lane(certificate, lane)
                                    for certificate in certificates
                                )
                                failed_rows.append(row)
                            else:
                                certificate_rows = ()
                        else:
                            certificate_rows = tuple(
                                self._native_certificate_row(certificate, lane)
                                for certificate in certificates
                            )
                        consume_work(
                            row, lane,
                            check_value=check_value,
                            certificate_value=certificate_value,
                            check_position=int(local_check_positions[lane]),
                            certificate_position=int(local_certificate_positions[lane]),
                            certificates=certificate_rows,
                            family_array=family_array,
                        )
                        if check_position == -1 and check_value != 0:
                            check_position = row['source_index']
                        if certificate_position == -1 and certificate_value != 0:
                            certificate_position = row['source_index']
                        if check_status == 0 and check_value != 0:
                            check_status = check_value
                        if certificate_status == 0 and certificate_value != 0:
                            certificate_status = certificate_value

                if check_status or certificate_status:
                    # All already-submitted work in this window has now been
                    # gathered in semantic order. Do not submit another window.
                    break

            if check_status or certificate_status:
                require_native_frequency(
                    check_status=check_status,
                    certificate_status=certificate_status,
                    check_position=check_position,
                    certificate_position=certificate_position,
                    certificates=failure_certificates(tuple(
                        failed_rows if raw_failure_certificates else executed_rows
                    )),
                    capabilities=candidate.realized_capabilities,
                )

            root_results = {}
            root_packets = {}
            root_errors = {}
            if include_roots:
                for row in native.root_layout:
                    start_index = row['start_index']
                    result, packet, error = self._search_native_branch(
                        program, native, candidate, row,
                        inputs.branch_starts_rad_s[start_index],
                    )
                    root_results[start_index] = result
                    root_packets[start_index] = packet
                    root_errors[start_index] = error
                    if result.failure is not None:
                        break

        return program, inputs, native, candidate, root_results, root_packets, root_errors

    def _run_native_direct(self, point):
        """Run one Direct candidate, then stream ordered native output chunks."""
        if self.result_sink is None or self.result_sink.domain != 'direct':
            raise RuntimeError('native Direct execution requires its sealed chunk sink')
        in_memory = bool(getattr(self.result_sink, 'in_memory_direct', False))

        def validate_work(program, inputs, native, work_rows):
            if any(set(row['demand']) != {'S', 'Y', 'Z'} for row in work_rows):
                raise RuntimeError('native Direct work must share S, Y, and Z at each frequency')

        def consume_work(row, lane, *, check_value, certificate_value,
                         check_position, certificate_position, certificates,
                         family_array):
            if in_memory:
                record = {
                    'families': {
                        family: family_array(family)[lane]
                        for family in ('S', 'Y', 'Z')
                    },
                }
                if check_value or certificate_value:
                    record['certificates'] = certificates
                self.result_sink.stage_direct_work(row['work_index'], record)
                return
            self.result_sink.stage_direct_work(row['work_index'], {
                'work_index': row['work_index'], 'source_index': row['source_index'],
                'view_index': row['view_index'], 'demand': tuple(row['demand']),
                'check_status': check_value, 'certificate_status': certificate_value,
                'check_position': check_position,
                'certificate_position': certificate_position,
                'certificates': certificates,
                'families': {
                    family: array_record(family_array(family)[lane])
                    for family in ('S', 'Y', 'Z')
                },
            })

        def failure_certificates(work_rows):
            return tuple(
                certificate
                for row in work_rows
                for certificate in self.result_sink.direct_work(
                    row['work_index'])['certificates']
            )

        def prepare_work(program, inputs, native, work_rows):
            if not in_memory:
                return
            occurrences = tuple(native.occurrence_layout)
            if not occurrences:
                raise RuntimeError('native Direct program has no declared matrix occurrences')
            work_indices = tuple(occurrences[0]['work_indices'])
            if any(tuple(row['work_indices']) != work_indices for row in occurrences):
                raise RuntimeError('Direct matrix families do not share the declared frequency layout')
            frequencies = np.asarray([quantity(value) for value in self.spec['frequencies']],
                                     dtype=np.float64)
            if len(work_indices) != len(frequencies):
                raise RuntimeError('Direct output gather differs from the exact request frequency count')
            context = point_contexts(program, inputs)[occurrences[0]['observable_index']]
            self.result_sink.prepare_direct_output(
                frequencies_hz=frequencies,
                work_indices=work_indices,
                terminal_count=len(context['terminal_ids']),
                complex_dtype=self.backend.complex_dtype,
            )

        program, inputs, native, candidate, _, _, _ = self._execute_native_work(
            point, operation='solve_direct', validate_work=validate_work,
            consume_work=consume_work, failure_certificates=failure_certificates,
            frequency_slots=None, prepare_work=prepare_work,
            raw_failure_certificates=in_memory,
        )
        if self._affine_evidence.rows:
            self.result_sink.set_affine_evidence(self._affine_evidence.rows)
        occurrence_layouts = tuple(native.occurrence_layout)
        if not occurrence_layouts:
            raise RuntimeError('native Direct program has no declared matrix occurrences')
        if any(
            tuple(row['work_indices']) != tuple(occurrence_layouts[0]['work_indices'])
            for row in occurrence_layouts
        ):
            raise RuntimeError('Direct matrix families do not share the declared frequency layout')
        work_indices = tuple(occurrence_layouts[0]['work_indices'])
        frequencies = np.asarray([quantity(value) for value in self.spec['frequencies']],
                                 dtype=np.float64)
        if len(work_indices) != len(frequencies):
            raise RuntimeError('Direct output gather differs from the exact request frequency count')
        if in_memory:
            return {'type': 'direct_solve', 'point_count': len(frequencies),
                    'backend': self.backend.native_program_identity('direct')}
        context = point_contexts(program, inputs)[occurrence_layouts[0]['observable_index']]
        capabilities = tuple(bool(value) for value in np.asarray(candidate.realized_capabilities))
        output_chunk_size = self.execution_policy.frequency_chunk_size
        for start in range(0, len(work_indices), output_chunk_size):
            stop = min(start + output_chunk_size, len(work_indices))
            selected_work = work_indices[start:stop]
            records = tuple(self.result_sink.direct_work(index) for index in selected_work)
            families = {
                family: np.stack(tuple(array_from_record(record['families'][family])
                                       for record in records))
                for family in ('S', 'Y', 'Z')
            }
            unique_work = tuple(dict.fromkeys(selected_work))
            work_summaries = tuple({
                'work_index': record['work_index'],
                'source_index': record['source_index'],
                'check_status': record['check_status'],
                'certificate_status': record['certificate_status'],
                'check_position': record['check_position'],
                'certificate_position': record['certificate_position'],
                'certificates': record['certificates'],
            } for index in unique_work for record in (self.result_sink.direct_work(index),))
            evidence = record_document(evidence_bytes({
                'terminal_ids': context['terminal_ids'],
                'axes': {key: context[key] for key in (
                    'probe_loads', 'reference_coordinate_ids',
                    'reference_impedances_ohm', 'load_mask',
                )},
                'realized_capabilities': capabilities,
                **({'affine_evidence':self._affine_evidence.rows}
                   if self._affine_evidence.rows else {}),
                'work_evidence': work_summaries,
            }))
            self.result_sink.append_direct_chunk(
                frequencies_hz=frequencies[start:stop], families=families, evidence=evidence,
            )

        return {'type': 'direct_solve', 'point_count': len(frequencies),
                'backend': self.backend.native_program_identity('direct')}

    def _native_branch_at(self, point, start_index, omega, *, operation):
        """Evaluate one existing declared root branch on the same native core."""
        program = self.program_at(point)
        base = program.inputs(point)
        starts = np.array(base.branch_starts_rad_s, dtype=np.complex128, copy=True)
        starts[start_index] = omega
        inputs = program.inputs(point, branch_starts_rad_s=starts)
        native = program.native_program(
            inputs, program_owner=self._program_owner, admit_core=True,
            affine_evidence=self._affine_evidence,
        )
        observation = self._affine_evidence
        with ExitStack() as stack:
            self._pin_native_core(native, stack)
            candidate = self._execute_native_candidate(native, inputs, operation=operation)
            result, packet, error = execute_native_branch_packet(native,candidate,start_index,omega,
                backend=self.backend,program_owner=self._program_owner,
                operation_resources=self.operation_resources,compact_evidence=self.ephemeral)
            return self._bind_affine_evidence(result, observation), packet, error

    def _native_branch_starts(self, point, *, baseline):
        program = self.program_at(point)
        starts = np.array(program.branch_starts_rad_s, dtype=np.complex128, copy=True)
        if baseline:
            return starts
        by_start = {}
        for row in self.roots:
            start_index = row['start_index']
            anchor = self.anchors[self._dependency_key(row)]
            omega = anchor.root_omega_rad_s
            prior = by_start.get(start_index)
            if prior is not None and prior != omega:
                raise RuntimeError('one declared root branch has inconsistent baseline anchors')
            by_start[start_index] = omega
        for start_index, omega in by_start.items():
            starts[start_index] = omega
        return starts

    def _run_native_sweep_point(self, point, identity, *, baseline=False):
        """Evaluate one Sweep point with the shared native candidate/work graph."""
        if self.result_sink is None or self.result_sink.domain != 'sweep':
            raise RuntimeError('native Sweep evaluation requires its operation scratch sink')
        if self.spec['type'] not in {
            'direct_solve', 'response_element', 'operator', 'diagonal_root',
            'operator_element_root', 'hybridized_pole', 'transfer_zero',
            'residue_normalized_coupling',
        }:
            raise TypeError(
                f'native Sweep provider is not integrated for {self.spec["type"]!r}'
            )
        ordinal = None if baseline else int(identity)
        namespace = ('baseline' if baseline else ordinal)
        operation = 'solve_direct' if self.spec['type'] == 'direct_solve' else 'evaluate'

        def validate_work(program, inputs, native, work_rows):
            if not native.occurrence_layout and not native.root_layout:
                raise RuntimeError('native Sweep program has no declared observable work')
            if self.spec['type'] == 'direct_solve':
                if any(set(row['demand']) != {'S', 'Y', 'Z'} for row in work_rows):
                    raise RuntimeError(
                        'native Direct Sweep work must share S, Y, and Z at each frequency'
                    )
                return
            if any(set(row['demand']) - {'S', 'Y', 'Z', 'operator'} for row in work_rows):
                raise TypeError('native Sweep provider returned an undeclared matrix family')

        def consume_work(row, lane, *, check_value, certificate_value,
                         check_position, certificate_position, certificates,
                         family_array):
            self.result_sink.stage_native_work(namespace, row['work_index'], {
                'work_index': row['work_index'],
                'source_index': row['source_index'],
                'view_index': row['view_index'],
                'demand': tuple(row['demand']),
                'check_status': check_value,
                'certificate_status': certificate_value,
                'check_position': check_position,
                'certificate_position': certificate_position,
                'certificates': certificates,
                'families': {
                    family: self._store_array(family_array(family)[lane])
                    for family in row['demand']
                },
            })

        def failure_certificates(work_rows):
            return tuple(
                certificate
                for row in work_rows
                for certificate in self.result_sink.native_work(
                    namespace, row['work_index'])['certificates']
            )

        starts = self._native_branch_starts(point, baseline=baseline)
        observation = self._affine_evidence
        try:
            program, inputs, native, candidate, root_results, root_packets, root_errors = self._execute_native_work(
                point, operation=operation, validate_work=validate_work,
                consume_work=consume_work, failure_certificates=failure_certificates,
                include_roots=self.spec['type'] != 'direct_solve',
                branch_starts_rad_s=starts,
            )
            if self.spec['type'] == 'direct_solve':
                occurrence_layouts = tuple(native.occurrence_layout)
                if not occurrence_layouts:
                    raise RuntimeError('native Direct Sweep program has no declared matrix occurrences')
                work_indices = tuple(occurrence_layouts[0]['work_indices'])
                if any(
                    tuple(row['work_indices']) != work_indices
                    for row in occurrence_layouts
                ):
                    raise RuntimeError(
                        'Direct matrix families do not share the declared frequency layout'
                    )
                frequencies = np.asarray(
                    [quantity(value) for value in self.spec['frequencies']],
                    dtype=np.float64,
                )
                if len(work_indices) != len(frequencies):
                    raise RuntimeError(
                        'Direct output gather differs from the exact request frequency count'
                    )
                families = {
                    family: np.stack(tuple(
                        self._load_array(self.result_sink.native_work(
                            namespace, work_index,
                        )['families'][family])
                        for work_index in work_indices
                    ))
                    for family in ('S', 'Y', 'Z')
                }
                context = point_contexts(program, inputs)[
                    occurrence_layouts[0]['observable_index']
                ]
                unique_work = tuple(dict.fromkeys(work_indices))
                work_evidence = tuple({
                    'work_index': record['work_index'],
                    'source_index': record['source_index'],
                    'check_status': record['check_status'],
                    'certificate_status': record['certificate_status'],
                    'check_position': record['check_position'],
                    'certificate_position': record['certificate_position'],
                    'certificates': record['certificates'],
                } for work_index in unique_work for record in (
                    self.result_sink.native_work(namespace, work_index),
                ))
                evidence = evidence_bytes({
                    'terminal_ids': context['terminal_ids'],
                    'axes': {key: context[key] for key in (
                        'probe_loads', 'reference_coordinate_ids',
                        'reference_impedances_ohm', 'load_mask',
                    )},
                    'realized_capabilities': tuple(
                        bool(value) for value in np.asarray(candidate.realized_capabilities)
                    ),
                    'work_evidence': work_evidence,
                })
                return self._bind_affine_evidence(EvaluationResult(
                    str(identity), S=families['S'], Y=families['Y'], Z=families['Z'],
                    evidence_bytes=evidence,
                ), observation), {}
            branch_rows = {
                row['start_index']: row for row in native.root_layout
            }
            for row in self.roots:
                start_index = row['start_index']
                if start_index not in branch_rows:
                    continue
                result = root_results.get(start_index)
                if result is None:
                    raise RuntimeError('native Sweep root work omitted a declared branch result')
                key = self._dependency_key(row)
                if not baseline and result.failure is not None:
                    branch_packets_by_id = {}

                    def make_job(job_id, actual, start):
                        return SimpleNamespace(id=job_id, point=actual, omega=start)

                    def evaluate(jobs):
                        values_at = []
                        for job in jobs:
                            result_at, packet_at, error_at = self._native_branch_at(
                                job.point, start_index, job.omega,
                                operation='evaluate',
                            )
                            branch_packets_by_id[job.id] = (packet_at,error_at)
                            values_at.append(replace(result_at, id=job.id))
                        return tuple(values_at)

                    def observe(values_at, actual, attempt, t):
                        record = (self.record(actual, attempt, {}, origin='root_continuation')
                                  if self.ephemeral else quantity_record(attempt))
                        record.update(parameters=actual, origin='root_continuation',
                                      source_index=ordinal, continuation_t_f64=bits(t),
                                      **self.context(actual))
                        record['program_sha256'] = self.program_at(actual).program_sha256
                        self.emit('evaluation', record)

                    anchor = self.anchors[key]
                    result = continue_root(
                        baseline=self.baseline_values,
                        values=parameter_values(point),
                        baseline_root=anchor.root_omega_rad_s,
                        endpoint_view=point,
                        initial_result=result,
                        candidate_view=lambda values_at: _point_at(point, values_at),
                        make_job=make_job,
                        evaluate=evaluate,
                        identity=str(ordinal),
                        observe=observe,
                    )
                    root_packets[start_index],root_errors[start_index] = branch_packets_by_id[result.id]
                if row['kind'] == 'diagonal_root' and not (
                        baseline and self.spec['type'] == 'diagonal_root'):
                    result = diagonal_passive_policy(result,
                        node_count=len(program.program.views[
                            program.program.observables[row['observable_index']].view_index].node_ids),
                        precision=program.program.precision)
                result = replace(result, id='baseline' if baseline else str(identity))
                root_results[start_index] = result
                if result.failure is not None:
                    raise EvaluationFailure(
                        result.failure, result=result,
                        dependencies=self._dependencies({index:item for index,item in root_results.items()
                                                        if item.failure is None}),
                    )
                if baseline:
                    self.anchors[key] = result

            if self.spec['type'] == 'residue_normalized_coupling':
                # Consume the already selected states and actual final root
                # packets. Coupling does not locate either branch again.
                public_rows = tuple(row for row in program.observable_metadata()
                                    if not row['hidden'])
                if len(public_rows) != 1:
                    raise RuntimeError('native coupling projection requires one declared public quantity')
                public = public_rows[0]
                coupling = next(row for row in native.coupling_layout
                                if row['observable_index'] == public['observable_index'])
                branch_packets = tuple(root_packets[index]
                                       for index in coupling['branch_indices'])
                source_indices = tuple(native.input_layout['omega']['source_indices'])
                with self.backend.jax.enable_x64(True), self.backend.jax.default_device(self.backend.device):
                    omega_source = self.backend.jax.numpy.asarray(
                        inputs.omega_rad_s, dtype=self.backend.complex_dtype,
                    )
                    omega_source = omega_source[self.backend.jax.numpy.asarray(
                        source_indices, dtype=np.int32,
                    )]
                    evaluation = (
                        self.backend.jax.numpy.asarray(0, dtype=self.backend.complex_dtype)
                        if coupling['midpoint'] else omega_source[coupling['evaluation_index']]
                    )
                branch_errors = tuple(root_errors[index] for index in coupling['branch_indices'])
                operands = (candidate, branch_packets, evaluation, branch_errors)
                from ..numerics.native_sparse import (SparseEquationCheck, checked_native,
                    archive_completed_primal_errors, sparse_certificates)
                from ..numerics.evidence import (
                    native_primal_exception, project_native_coupling,
                )
                coupling_function = native.coupling_function(public['observable_index'])
                def complete_coupling(point, packets, frequency, errors):
                    archive_completed_primal_errors(errors)
                    return coupling_function(point,packets,frequency)
                checked = checked_native(complete_coupling)
                compile_key = native.coupling_variant_key(
                    public['observable_index'], device=self.backend.device,
                    operands=operands, transform=('checked_completed_value',
                        str(self.backend.jax.tree.structure(branch_errors))),
                )
                candidate_scope = (nullcontext() if self.operation_resources is None else
                                   self.operation_resources.candidate_compute())
                with ExitStack() as stack:
                    self._pin_native_core(native, stack)
                    with candidate_scope:
                        checked_error, packet = self.backend.execute_native(
                            checked, *operands, precision=self.backend.precision,
                            compile_key=compile_key, owner=self._program_owner,
                            phase='native_residue_coupling',
                            details={'operation': 'evaluate',
                                     'observable_index': public['observable_index']},
                        )
                exception = native_primal_exception(checked_error)
                if exception is not None and not isinstance(exception, SparseEquationCheck):
                    checked_error.throw()
                selected = project_native_coupling(public['quantity_key'], packet)
                from dataclasses import asdict
                facts = record_document(selected.evidence_bytes)
                facts['equation_certificates'] = tuple(asdict(value)
                    for value in sparse_certificates(checked_error))
                selected = replace(selected,evidence_bytes=evidence_bytes(facts))
                dependencies = {}
                for row in program.observable_metadata():
                    if not row['hidden']:
                        continue
                    body = replace(
                        root_results[row['start_index']],
                        id=f"{row['quantity_key']}:observable:{row['observable_index']}",
                    )
                    if body.failure is None:
                        if self.ephemeral:
                            dependencies[('observable', row['observable_index'])] = body
                        else:
                            body_id, body_bytes = quantity_body_id(body)
                            dependencies[body_id] = body_bytes
                if selected.failure is not None:
                    raise EvaluationFailure(selected.failure, result=selected,
                                            dependencies=dependencies)
                return self._bind_affine_evidence(
                    replace(selected, id=str(identity)), observation), dependencies

            matrices = []
            for occurrence in native.occurrence_layout:
                start_index = (occurrence['branch_indices'][0]
                               if occurrence['branch_indices'] else None)
                if start_index is not None:
                    result = root_results[start_index]
                    matrices.append(result.root_omega_rad_s)
                    continue
                family = occurrence['family']
                family_values = []
                for work_index in occurrence['work_indices']:
                    work_record = self.result_sink.native_work(namespace, work_index)
                    family_values.append(self._load_array(work_record['families'][family]))
                matrices.append(np.stack(tuple(family_values)))
            with self.backend.jax.enable_x64(True), self.backend.jax.default_device(self.backend.device):
                public = native.project_public_values(tuple(matrices))
                self.backend.jax.block_until_ready(public)

            public_rows = tuple(row for row in program.observable_metadata() if not row['hidden'])
            if self.spec['type'] in {
                'diagonal_root', 'operator_element_root', 'hybridized_pole', 'transfer_zero',
            }:
                row = public_rows[0]
                result = root_results[row['start_index']]
            elif self.spec['type'] == 'response_element':
                row = public_rows[0]
                raw = np.asarray(public.observable_raw_values[row['observable_index']])
                if raw.shape:
                    raw = raw[()]
                context = point_contexts(program, inputs)[row['observable_index']]
                work_evidence = self._native_sweep_work_evidence(
                    native, namespace, context, candidate,
                )
                result = EvaluationResult(
                    str(identity), response_value=raw,
                    evidence_bytes=work_evidence,
                )
            elif self.spec['type'] == 'operator':
                row = public_rows[0]
                raw = np.asarray(public.observable_raw_values[row['observable_index']])
                context = point_contexts(program, inputs)[row['observable_index']]
                work_evidence = self._native_sweep_work_evidence(
                    native, namespace, context, candidate,
                )
                result = EvaluationResult(
                    str(identity), operator_values=raw,
                    evidence_bytes=work_evidence,
                )
            else:
                raise TypeError(f'native Sweep result projection is not integrated for {self.spec["type"]!r}')

            dependencies = {}
            return self._bind_affine_evidence(result, observation), dependencies
        except EvaluationFailure as error:
            result = error.result if error.result is not None else EvaluationResult(
                str(identity), failure=error.failure,
            )
            return self._bind_affine_evidence(
                replace(result, id=str(identity)), observation), dict(error.dependencies)
        except SCNSimError as error:
            if error.kind not in _NUMERICAL_FAILURES:
                raise
            failure = NumericalFailure(error.kind, error.stage, str(error), record_bytes(error.evidence))
            return self._bind_affine_evidence(
                EvaluationResult(str(identity), failure=failure), observation), {}
        finally:
            self.result_sink.release_native_work(namespace)

    def _native_sweep_work_evidence(self, native, namespace, context, candidate):
        """Retain actual per-frequency certificates without retaining outcomes."""
        work = tuple(
            self.result_sink.native_work(namespace, row['work_index'])
            for row in native.work_layout
        )
        return evidence_bytes({
            'certificates': (),
            'realized_capabilities': tuple(
                bool(value) for value in np.asarray(candidate.realized_capabilities)
            ),
            'terminal_ids': context['terminal_ids'],
            'axes': {key: context[key] for key in (
                'probe_loads', 'reference_coordinate_ids',
                'reference_impedances_ohm', 'load_mask',
            )},
            'work_evidence': tuple({
                key: row[key] for key in (
                    'work_index', 'source_index', 'view_index', 'demand',
                    'check_status', 'certificate_status', 'check_position',
                    'certificate_position', 'certificates',
                )
            } for row in work),
        })

    def _run_native_sweep(self, points):
        """Run each declared point in order and retain only spool body refs."""
        cache = {}
        baseline_needed = self.spec['type'] in _ROOT_TYPES
        if baseline_needed:
            result, dependencies = self._run_native_sweep_point(
                self.baseline, 'baseline', baseline=True,
            )
            baseline_record = self.record(
                self.baseline, result, dependencies,
                origin='baseline', cache_hit=False,
            )
            self.emit('evaluation', baseline_record)
            if result.failure is not None:
                raise numerical_error(result.failure)
            self.emit('baseline_ready', {
                'schema': 'scnsim.benchmark_root_anchor', 'schema_version': 1,
                'baseline': baseline_record,
                'anchors': self._anchor_references(),
                'resume_state': None,
            })
            key = point_identity(self.baseline)
            cache[key] = self.result_sink.cache_sweep_point(baseline_record)

        point_count = 0
        for ordinal, point in enumerate(points):
            point_count += 1
            key = point_identity(point)
            if key in cache:
                original = self.result_sink.sweep_point(cache[key])
                record = dict(original)
                record.pop('ordinal', None)
                record.pop('continuations', None)
                record.update(
                    parameters=point, source_index=ordinal,
                    origin='requested_point', cache_hit=True,
                    numerical_source_id=original['id'],
                )
                self.emit('evaluation', record)
            else:
                result, dependencies = self._run_native_sweep_point(
                    point, str(ordinal), baseline=False,
                )
                record = self.record(
                    point, result, dependencies,
                    source_index=ordinal, origin='requested_point',
                    cache_hit=False, numerical_source_id=result.id,
                )
                cache[key] = self.emit('evaluation', record)

        terminal = {
            'type': self.spec['type'], 'point_count': point_count,
            'backend': self.backend.native_program_identity(
                'direct' if self.spec['type'] == 'direct_solve' else 'evaluate',
            ),
        }
        if baseline_needed:
            terminal['baseline_anchor_schema'] = 'scnsim.benchmark_root_anchor'
        return terminal

    def run(self, points):
        if self.result_sink is not None and self.result_sink.domain == 'direct':
            points = iter(points)
            try:
                point = next(points)
            except StopIteration as error:
                raise RuntimeError('Direct request has no exact source point') from error
            try:
                next(points)
            except StopIteration:
                pass
            else:
                raise RuntimeError('Direct request yielded more than its one declared point')
            return self._run_native_direct(point)
        if self.result_sink is not None and self.result_sink.domain == 'sweep':
            return self._run_native_sweep(points)
        cache, seen = {}, set()
        baseline_needed = self.spec['type'] in _ROOT_TYPES
        if baseline_needed:
            result, dependencies = self.evaluate(self.baseline, 'baseline', baseline=True)
            record = self.record(self.baseline, result, dependencies, origin='baseline', cache_hit=False)
            self.emit('evaluation', record)
            if result.failure is not None:
                raise numerical_error(result.failure)
            self.emit('baseline_ready', {
                'schema': 'scnsim.benchmark_root_anchor', 'schema_version': 1,
                'baseline': record,
                'anchors': self._anchor_references(),
                'resume_state': None,
            })
            if self.spec['type'] == 'diagonal_root':
                result = diagonal_passive_policy(result,
                    node_count=len(self.program.program.views[
                        self.program.program.observables[0].view_index].node_ids),
                    precision=self.program.program.precision)
            key = point_identity(self.baseline)
            cache[key], seen = (result, dependencies), {key}
        first_failure = None
        point_count = 0
        for ordinal, point in enumerate(points):
            point_count += 1
            key = point_identity(point)
            if key not in cache:
                cache[key] = self.evaluate(point, str(ordinal))
            result, dependencies = cache[key]
            self.emit('evaluation', self.record(
                point, result, dependencies, source_index=ordinal, origin='requested_point',
                cache_hit=key in seen, numerical_source_id=result.id,
            ))
            seen.add(key)
            if first_failure is None:
                first_failure = result.failure
        if first_failure is not None and self.analysis['parameter_source']['kind'] == 'point' and not self.ephemeral:
            raise numerical_error(first_failure)
        terminal = {'type': self.spec['type'], 'point_count': point_count,
                    'backend': self.backend.native_program_identity('evaluate')}
        if baseline_needed:
            terminal['baseline_anchor_schema'] = 'scnsim.benchmark_root_anchor'
        return terminal
