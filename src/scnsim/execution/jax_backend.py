"""Shared same-process JAX assembly and worker-local CPU SuperLU evaluation.

Executable owners are a Run's explicit warm holder or one operation handle.
The process-wide index is weak: it shares live static-signature kernels without
extending their lifetime. Plans, coefficients and numerical history are not cached.
"""

from __future__ import annotations

import importlib.metadata
import os
from contextlib import contextmanager, nullcontext
from threading import RLock, local
from time import perf_counter_ns
from weakref import WeakValueDictionary
from types import SimpleNamespace

import numpy as np

from ..numerics.models import EvaluationJob, EvaluationResult, NumericalFailure
from ..numerics.evidence import evidence_bytes


# Lock lookup/compile only; execution arguments remain independent and local.
_ALGORITHM_ID = "scnsim.jax-sparse-csc-superlu-direct-quantities-newton32.v8"
_EXECUTABLES = WeakValueDictionary()
_EXECUTABLE_LOCK = RLock()


class _ExecutableEntry:
    __slots__ = ('executable', '__weakref__')

    def __init__(self, executable):
        self.executable = executable


def _span(trace, kind, **details):
    return nullcontext(None) if trace is None else trace.span(kind, details=details)


def get_jax_backend(*, precision, resources, trace, operation_resources=None, diagnostics=True):
    return JaxBackend(precision=precision, resources=resources, trace=trace,
                      operation_resources=operation_resources, diagnostics=diagnostics)




class JaxBackend:
    def __init__(self, *, precision="float64", resources=None, trace=None, device="cpu", cpu_threads=None,
                 operation_resources=None, diagnostics=True):
        if device != "cpu":
            raise ValueError("JAX execution supports only the CPU device")
        if precision not in ("float32", "float64"):
            raise ValueError("precision must be float32 or float64")
        from .config import ensure_jax_configuration, get_runtime_configuration, mark_jax_initialized
        configured = get_runtime_configuration()
        resources = configured if resources is None else resources
        if resources != configured:
            raise RuntimeError("JAX resources differ from configure_runtime; restart the Kernel to change initialized resources")
        if cpu_threads is not None and cpu_threads != resources.cpu_threads:
            raise RuntimeError("use configure_runtime(cpu_threads=...) before JAX initialization")
        self.trace = trace
        self.operation_resources = operation_resources
        self.diagnostics = diagnostics
        started = perf_counter_ns()
        with _span(trace, "numerical_backend_initialization", precision=precision):
            ensure_jax_configuration()
            import jax
            from ..numerics import jax_core
            self.jax = jax
            self.core = jax_core
            self.device = jax.devices("cpu")[0]
            mark_jax_initialized(resources)
        self.precision = precision
        self.real_dtype = np.float32 if precision == "float32" else np.float64
        self.complex_dtype = np.complex64 if precision == "float32" else np.complex128
        self.cpu_threads = resources.cpu_threads
        self._closed = False
        self._template_cache = {}
        self._executables = {}
        self._default_owner = SimpleNamespace(templates=self._template_cache, executables=self._executables)
        self._owner_local = local()
        self.initialization_ns = perf_counter_ns() - started

    def _numerical_owner(self):
        return getattr(self._owner_local, 'current', self._default_owner)

    @contextmanager
    def continuation_step_scope(self):
        """Own transient patterns/kernels through synchronized step completion.

        Broker requests carry this owner explicitly; a shared executable remains
        live while any endpoint, warm Run or concurrent step owns its entry.
        Releasing Python ownership does not promise immediate native map reuse.
        """
        previous = self._numerical_owner()
        owner = SimpleNamespace(templates={}, executables={})
        self._owner_local.current = owner
        try:
            yield
        finally:
            self._owner_local.current = previous
            owner.templates.clear()
            with _EXECUTABLE_LOCK:
                owner.executables.clear()

    def _compiled(self, signature, compile_kernel, *, owners=None):
        """Share live kernels; retain only the explicit consuming owners."""
        owners = (self._numerical_owner(),) if owners is None else owners
        with _EXECUTABLE_LOCK:
            entry = next((owner.executables[signature] for owner in owners
                          if signature in owner.executables), None)
            entry = entry or self._executables.get(signature) or _EXECUTABLES.get(signature)
            new_shape = entry is None
            if new_shape:
                entry = _ExecutableEntry(compile_kernel())
                _EXECUTABLES[signature] = entry
            for owner in owners:
                owner.executables[signature] = entry
        return entry.executable, new_shape

    def compile_warm_view(self, view, *, kind, frequency_count=1, family=None,
                          candidate_batch=False):
        """Lower static signatures without running an assembly or local solve.

        Abstract selected-dtype operands preserve the execution cache keys.
        Candidate batch widths, continuation meshes and future CSR shapes are
        unknown here and remain lazy; no fabricated numerical state is emitted.
        """
        from ..numerics.sparse_direct import System
        from scipy.sparse import coo_matrix
        system = System(view, self.real_dtype, self.complex_dtype,
                        template_cache=self._template_cache)
        observations = []
        exclusive = (nullcontext() if self.operation_resources is None
                     else self.operation_resources.exclusive_assembly())
        with exclusive, self.jax.enable_x64(self.precision == 'float64'):
            def prepare(args, kernel, controls, function):
                args = self.jax.tree.map(
                    lambda value: self.jax.ShapeDtypeStruct(value.shape, value.dtype), args)
                leaves, tree = self.jax.tree.flatten(args)
                shapes = tuple((tuple(value.shape), str(value.dtype)) for value in leaves)
                signature = (self.device.platform, self.device.id, self.cpu_threads,
                             self.precision, _ALGORITHM_ID, kernel, *controls, tree, shapes)
                compile_ns = None
                def compile_kernel():
                    nonlocal compile_ns
                    started = perf_counter_ns()
                    try:
                        return self.jax.jit(function).lower(*args).compile()
                    finally:
                        compile_ns = perf_counter_ns() - started
                started = perf_counter_ns()
                _, new_shape = self._compiled(
                    signature, compile_kernel)
                observations.append((signature, kernel, shapes, not new_shape,
                                     perf_counter_ns() - started, compile_ns))

            groups = system.payload[3]
            if any(np.any(group[6]) for group in groups):
                prepare((groups,), 'pure_inductive_coefficients', (),
                        self.core.inductive_coefficients)
            if kind in ('direct', 'response_element'):
                loaded = derivative = not view.port_realizable
            elif kind == 'transfer_zero':
                loaded, derivative = not view.port_realizable, True
            else:
                loaded = derivative = True
            compensated = kind == 'transfer_zero' and family == 'Z' and not view.port_realizable
            counts = (min(8, frequency_count),)
            if frequency_count > 8 and frequency_count % 8:
                counts += (frequency_count % 8,)
            for count in dict.fromkeys(counts):
                omega = np.empty(count, dtype=self.complex_dtype)
                core = self.core
                function = self.jax.vmap(
                    lambda data, w: core.assemble(data, w, loaded=loaded,
                                                 derivative=derivative, compensated=compensated),
                    in_axes=(None, 0))
                args, kernel = (system.payload, omega), 'sparse_assembly'
                if candidate_batch:
                    # The baseline actor is the one known ready candidate.
                    # Future concurrently ready widths remain runtime-lazy.
                    function = self.jax.vmap(function, in_axes=(0, 0))
                    args = (self.jax.tree.map(lambda value: value[None, ...], system.payload),
                            omega[None, ...])
                    kernel = 'candidate_sparse_assembly'
                prepare(args, kernel,
                        (loaded, derivative, compensated), function)
            if compensated and len(system.eliminated):
                # Slice the actual structural pattern, including its zeros.
                # Only actual-RHS-sized placeholders are dense, never global Q.
                pattern = coo_matrix((np.zeros(len(system.rows), dtype=self.complex_dtype),
                                      (system.rows, system.cols)), shape=(system.n, system.n)).tocsr()
                for rows in (system.eliminated, system.selected):
                    matrix = pattern[rows][:, system.eliminated].tocsr()
                    rhs = np.empty((len(system.eliminated), len(system.selected)),
                                   dtype=self.complex_dtype)
                    payload = (matrix.data, matrix.indices, matrix.indptr,
                               matrix.data, matrix.indices, matrix.indptr, rhs, rhs)
                    prepare((payload,), 'csr_pair_product', (), self.core.csr_pair_product)
        return tuple(observations)

    def identity(self) -> dict[str, object]:
        return {
            "backend": "jax", "algorithm_id": _ALGORITHM_ID,
            "jax": importlib.metadata.version("jax"), "jaxlib": importlib.metadata.version("jaxlib"),
            "scipy": importlib.metadata.version("scipy"),
            "threadpoolctl": importlib.metadata.version("threadpoolctl"),
            "sparse_solver": "scipy.sparse.linalg.splu", "factorization_backend": "SuperLU",
            "superlu_version": None, "superlu_version_status": "not exposed by scipy public API",
            "assembly": "jax-runtime-coo-coalesced",
            "executable_ownership": "Run_warm_or_operation_strong_process_index_weak",
            "pure_inductive_stamp": "exact-realized-zero-R-real-dtype-scaled-analytic-width1-2-L-inverse-LU-larger", "assembly_chunk_limit": 8,
            "compensated_arithmetic": {
                "scope": "transfer_zero.Z.non_port_realizable",
                "representation": "two_components_in_requested_base_dtype",
                "coverage": "indexed_assembly_compiled_csr_residual_selected_formation",
                "pair_product_kernel": "jax-runtime-csr-row-batched-high-then-low.v1",
                "correction": "one_primal_and_one_derivative_same_high_factor",
                "projection": "F_Fp_then_Y_Yp_declared_complex_dtype",
            },
            "dtype": f"{self.precision}/{np.dtype(self.complex_dtype).name}", "arithmetic_precision": self.precision, "device": str(self.device),
            "platform": self.device.platform, "device_kind": self.device.device_kind,
            "requested_cpu_threads": self.cpu_threads,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")},
            "initialization_ns": self.initialization_ns,
        }

    def evaluate_batch(self, jobs: tuple[EvaluationJob, ...]) -> tuple[EvaluationResult, ...]:
        # Public JAX context is thread-local in pinned JAX, restored on all exits.
        # No global platform/x64 option is changed by SCNSim.
        compute = nullcontext() if self.operation_resources is None else self.operation_resources.candidate_compute()
        with compute, self.jax.enable_x64(self.precision == "float64"):
            return self._evaluate_batch(jobs)

    def _prepare_inductive_coefficients(self, system):
        if system.inductive_coefficients_ready:
            return
        if not any(np.any(group[6]) for group in system.payload[3]):
            system.inductive_coefficients_ready = True
            return
        # Candidate-local values, process-wide shape-only executable; use the
        # same exclusive numerical resource boundary before the cache lock.
        exclusive = nullcontext() if self.operation_resources is None else self.operation_resources.exclusive_assembly()
        with exclusive:
            groups = system.payload[3]
            trace = self.trace if self.diagnostics else None
            context = dict(kernel='inductive_coefficients', algorithm_id=_ALGORITHM_ID,
                           arithmetic_precision=self.precision, device=str(self.device),
                           requested_cpu_threads=self.cpu_threads)
            with _span(trace, 'transfer_to_device', **context):
                args = self.jax.device_put((groups,), self.device)
                self.jax.block_until_ready(args)
            leaves, tree = self.jax.tree.flatten(args)
            signature = (self.device.platform, self.device.id, self.cpu_threads,
                         self.precision, _ALGORITHM_ID, 'pure_inductive_coefficients', tree,
                         tuple((tuple(a.shape), str(a.dtype)) for a in leaves))
            context['argument_shapes'] = [[list(a.shape), str(a.dtype)] for a in leaves]
            def compile_kernel():
                with _span(trace, 'shape_compilation', **context):
                    return self.jax.jit(self.core.inductive_coefficients).lower(*args).compile()
            executable, new_shape = self._compiled(signature, compile_kernel)
            if self.operation_resources is not None:
                self.operation_resources.record_assembly(new_shape=new_shape)
            context['executable_cache_hit'] = not new_shape
            with _span(trace, 'synchronized_compute', **context):
                output = executable(*args)
                self.jax.block_until_ready(output)
            with _span(trace, 'transfer_to_host', **context):
                coefficients = self.jax.tree.map(np.asarray, output)
            prepared = tuple(group[:7] + (inverse, codes)
                             for group, (inverse, codes) in zip(groups, coefficients))
            system.payload = system.payload[:3] + (prepared,)
            system.inductive_coefficients_ready = True

    def _assembly(self, system, omegas, *, loaded, derivative, measurements, compensated=False):
        self._prepare_inductive_coefficients(system)
        if self.operation_resources is not None and self.operation_resources.assembly_batching:
            omega_array = np.asarray(omegas, dtype=self.complex_dtype)
            leaves, tree = self.jax.tree.flatten((system.payload, omega_array))
            compatibility = (self.device.platform, self.device.id, self.precision,
                             loaded, derivative, compensated, system.n, tree,
                             tuple((tuple(value.shape), str(value.dtype)) for value in leaves))
            request = dict(compatibility=compatibility, owner=self._numerical_owner(), payload=system.payload,
                           omegas=omega_array, loaded=loaded, derivative=derivative, compensated=compensated)
            return self.operation_resources.assemble(request, self._assemble_candidates_exclusive)
        return self._assembly_single(system, omegas, loaded=loaded, derivative=derivative,
                                     measurements=measurements, compensated=compensated)

    def _assemble_candidates_exclusive(self, requests):
        # Broker owns its own dtype context. All operands, including stamp maps,
        # are paired runtime inputs; neither topology nor coefficients are constants.
        with self.jax.enable_x64(self.precision == 'float64'):
            with self.operation_resources.phase('ready_assembly_transfer_compute'):
                payloads = self.jax.tree.map(lambda *parts: np.stack(parts),
                                            *(request['payload'] for request in requests))
                omegas = np.stack([request['omegas'] for request in requests])
                args = self.jax.device_put((payloads, omegas), self.device)
                self.jax.block_until_ready(args)
                leaves, tree = self.jax.tree.flatten(args)
                loaded, derivative = requests[0]['loaded'], requests[0]['derivative']
                compensated = requests[0]['compensated']
                signature = (self.device.platform, self.device.id, self.cpu_threads,
                             self.precision, _ALGORITHM_ID, 'candidate_sparse_assembly', loaded, derivative, compensated,
                             tree, tuple((tuple(a.shape),str(a.dtype)) for a in leaves))
                def compile_kernel():
                    core = self.core
                    one_candidate = self.jax.vmap(
                        lambda data,omega: core.assemble(data,omega,loaded=loaded,derivative=derivative,compensated=compensated),
                        in_axes=(None,0))
                    function = self.jax.vmap(one_candidate,in_axes=(0,0))
                    return self.jax.jit(function).lower(*args).compile()
                executable, new_shape = self._compiled(
                    signature, compile_kernel, owners=tuple(request['owner'] for request in requests))
                self.operation_resources.record_assembly(new_shape=new_shape)
                output = executable(*args)
                self.jax.block_until_ready(output)
                output = self.jax.tree.map(np.asarray,output)
                return [tuple(part[index] for part in output) for index in range(len(requests))]

    def _assembly_single(self, system, omegas, *, loaded, derivative, measurements, compensated=False):
        """Exclusive transfer/JIT/compute/download before acquiring cache lock."""
        exclusive = nullcontext() if self.operation_resources is None else self.operation_resources.exclusive_assembly()
        with exclusive:
            return self._assemble_exclusive(system, omegas, loaded=loaded, derivative=derivative,
                                            measurements=measurements, compensated=compensated)

    def _assemble_exclusive(self, system, omegas, *, loaded, derivative, measurements, compensated=False):
        trace = self.trace if self.diagnostics else None
        context = dict(batch_size=len(omegas), arithmetic_precision=self.precision,
                       operation='sparse_assembly', algorithm_id=_ALGORITHM_ID,
                       device=str(self.device), requested_cpu_threads=self.cpu_threads,
                       static_pattern_sha256=system.pattern_sha256)
        with _span(trace, 'transfer_to_device', **context):
            args = self.jax.device_put((system.payload, np.asarray(omegas, dtype=self.complex_dtype)), self.device)
            self.jax.block_until_ready(args)
        leaves, tree = self.jax.tree.flatten(args)
        shapes = tuple((tuple(a.shape), str(a.dtype)) for a in leaves)
        context['argument_shapes'] = [[list(shape), dtype] for shape, dtype in shapes]
        signature = (self.device.platform, self.device.id, self.cpu_threads, self.precision,
                     _ALGORITHM_ID, 'sparse_assembly', loaded, derivative, compensated, tree, shapes)
        def compile_kernel():
            # Coefficients and maps remain runtime inputs; only controls are static.
            with _span(trace, 'shape_compilation', **context):
                core = self.core
                function = self.jax.vmap(lambda data, omega: core.assemble(data, omega, loaded=loaded, derivative=derivative, compensated=compensated), in_axes=(None, 0))
                return self.jax.jit(function).lower(*args).compile()
        executable, new_shape = self._compiled(signature, compile_kernel)
        if self.operation_resources is not None:
            self.operation_resources.record_assembly(new_shape=new_shape)
        context['executable_cache_hit'] = not new_shape
        with _span(trace, 'synchronized_compute', **context):
            output = executable(*args)
            self.jax.block_until_ready(output)
        with _span(trace, 'transfer_to_host', **context):
            output = self.jax.tree.map(np.asarray, output)
        return output

    def _pair_product(self, payload, measurements):
        """Execute runtime CSR pair operands under the existing assembly permit.

        Cache entries retain executables, never CSR values or candidate RHSs.
        Numerical owners consume the returned components without projection.
        """
        exclusive = nullcontext() if self.operation_resources is None else self.operation_resources.exclusive_assembly()
        with exclusive, self.jax.enable_x64(self.precision == 'float64'):
            with measurements.phase('compiled_csr_pair_product', algorithm_id=_ALGORITHM_ID,
                                    arithmetic_precision=self.precision, device=str(self.device),
                                    requested_cpu_threads=self.cpu_threads):
                with measurements.phase('kernel_transfer_to_device', kernel='csr_pair_product'):
                    args = self.jax.device_put((payload,), self.device)
                    self.jax.block_until_ready(args)
                leaves, tree = self.jax.tree.flatten(args)
                shapes = tuple((tuple(value.shape), str(value.dtype)) for value in leaves)
                signature = (self.device.platform, self.device.id, self.cpu_threads,
                             self.precision, _ALGORITHM_ID, 'csr_pair_product', tree, shapes)
                def compile_kernel():
                    with measurements.phase('kernel_shape_compilation', kernel='csr_pair_product',
                                            argument_shapes=[[list(shape), dtype] for shape, dtype in shapes]):
                        return self.jax.jit(self.core.csr_pair_product).lower(*args).compile()
                executable, new_shape = self._compiled(signature, compile_kernel)
                with measurements.phase('kernel_synchronized_compute', kernel='csr_pair_product',
                                        executable_cache_hit=not new_shape,
                                        argument_shapes=[[list(shape), dtype] for shape, dtype in shapes]):
                    output = executable(*args)
                    self.jax.block_until_ready(output)
                with measurements.phase('kernel_transfer_to_host', kernel='csr_pair_product'):
                    return self.jax.tree.map(np.asarray, output)

    def _evaluate_batch(self, jobs: tuple[EvaluationJob, ...]) -> tuple[EvaluationResult, ...]:
        from ..numerics.sparse_direct import Measurements, System, network, retained_response
        from ..numerics.sparse_root import diagonal_root
        from ..numerics.sparse_quantities import evaluate_quantity
        from ..numerics.determinants import QuantityError
        if self._closed:
            raise RuntimeError('numerical backend is closed')
        results = []
        for job in jobs:
            if job.kind not in ('direct', 'response_element', 'diagonal_root', 'operator', 'operator_element_root', 'hybridized_pole', 'transfer_zero', 'residue_normalized_coupling'):
                raise ValueError(f'unsupported JAX numerical operation: {job.kind}')
            if not job.view.port_realizable and (job.kind == 'direct' or (job.kind in ('response_element', 'transfer_zero') and job.family == 'S')):
                results.append(EvaluationResult(job.id, failure=NumericalFailure(
                    'port_realizability', 'selected_network', 'wave response requires a Port-realizable View')))
                continue
            with _span(self.trace if self.diagnostics else None, 'numerical_batch', operation=job.kind,
                       batch_size=1, arithmetic_precision=self.precision) as parent:
                measurements = Measurements(self.trace, parent, enabled=self.diagnostics, resource=self.operation_resources)
                with measurements.phase('sparse_pattern_prepare'):
                    system = System(job.view, self.real_dtype, self.complex_dtype, template_cache=self._numerical_owner().templates, resource=self.operation_resources,
                                    pair_product=lambda payload: self._pair_product(payload, measurements))
                def assemble(omega, *, loaded, derivative, compensated=False):
                    arrays = self._assembly(system, [omega], loaded=loaded, derivative=derivative,
                                            measurements=measurements, compensated=compensated)
                    return tuple(a[0] for a in arrays)
                if job.kind == 'transfer_zero' and job.family == 'Z' and not job.view.port_realizable:
                    system.compensation_evidence = dict(
                        scope='transfer_zero.Z.non_port_realizable',
                        representation='Q_hi_plus_Q_lo_and_Qp_hi_plus_Qp_lo',
                        correction_policy='not_reached')
                quantity_failure = None
                code = 0
                try:
                    if job.kind == 'diagonal_root':
                        start = (self.complex_dtype(job.omega_start_rad_s) if job.omega_start_rad_s is not None
                                 else self.complex_dtype(2 * np.pi * job.root_hint_hz))
                        omega, slope, certificate, code, steps = diagonal_root(system, start, job.coordinate_index,
                                                                               assemble, measurements)
                        values = {'root_omega_rad_s': omega, 'root_slope': slope}
                        extra = {'newton_steps': steps, 'certificate': certificate.tolist(),
                                 'certificates': dict(zip(('eliminated_residual', 'operator_residual',
                                     'element_residual', 'relative_correction', 'normalized_slope', 'slope_scale'),
                                     certificate, strict=True))}
                    elif job.kind == 'operator':
                        frequencies = np.asarray(job.frequencies_hz, dtype=np.float64)
                        omegas = np.asarray(2*np.pi*frequencies, dtype=self.complex_dtype)
                        operators = []
                        for offset in range(0, len(omegas), 8):
                            arrays = self._assembly(system, omegas[offset:offset+8], loaded=True, derivative=True,
                                                    measurements=measurements)
                            from ..numerics.sparse_direct import selected_state
                            for index in range(len(omegas[offset:offset+8])):
                                state, code = selected_state(system, tuple(a[index] for a in arrays), measurements)
                                if code:
                                    from ..numerics.sparse_quantities import numerical_code
                                    numerical_code(code)
                                operators.append(state[0])
                        values = {'operator_values': np.asarray(operators, dtype=self.complex_dtype)}
                        extra = {'frequency_count': len(frequencies)}
                    elif job.kind in ('operator_element_root', 'hybridized_pole', 'transfer_zero', 'residue_normalized_coupling'):
                        values, certificates = evaluate_quantity(system, job, assemble, measurements)
                        extra = {'certificates': certificates}
                    else:
                        family = 'all' if job.kind == 'direct' else job.family
                        frequencies = np.asarray(job.frequencies_hz, dtype=np.float64)
                        omegas = np.asarray(2 * np.pi * frequencies, dtype=self.complex_dtype)
                        outputs = []
                        codes = []
                        for offset in range(0, len(omegas), 8):
                            assembled = self._assembly(system, omegas[offset:offset+8],
                                                       loaded=not job.view.port_realizable,
                                                       derivative=not job.view.port_realizable,
                                                       measurements=measurements)
                            for index, omega_at in enumerate(omegas[offset:offset+8]):
                                state = tuple(a[index] for a in assembled)
                                if job.view.port_realizable:
                                    value, status = network(system, omega_at, state, family, measurements)
                                else:
                                    value, status = retained_response(system, omega_at, state, family, measurements)
                                outputs.append(value)
                                codes.append(status)
                        failed = next((i for i, code_at in enumerate(codes) if code_at), None)
                        code = 0 if failed is None else codes[failed]
                        extra = {'frequency_count': len(frequencies)}
                        if failed is not None:
                            extra['first_failure_frequency_index'] = failed
                            values = {}
                        elif job.kind == 'direct':
                            values = {name: np.asarray([value[i] for value in outputs], dtype=self.complex_dtype)
                                      for i, name in enumerate(('S', 'Y', 'Z'))}
                        else:
                            family_index = {'S': 0, 'Y': 1, 'Z': 2}[job.family]
                            values = {'response_value': self.complex_dtype(outputs[0][family_index][job.output_index, job.input_index])}
                except QuantityError as error:
                    values = {}
                    extra = {'certificates': error.facts}
                    quantity_failure = error
                observations = {'quantity_kind': job.kind, 'terminal_ids': list(job.view.terminal_ids),
                                'arithmetic_precision': self.precision, 'algorithm_id': _ALGORITHM_ID,
                                'factors': system.factor_facts,
                                'static_pattern_sha256': system.pattern_sha256, **extra}
                if job.kind == 'transfer_zero' and job.family == 'Z' and not job.view.port_realizable:
                    observations['arithmetic'] = dict(system.compensation_evidence)
                observations['axes'] = dict(original_node_ids=list(job.view.original_node_ids),
                    model_node_ids=list(job.view.model.node_ids), terminal_ids=list(job.view.terminal_ids),
                    port_ids=list(job.view.model.port_ids), selected_indices=list(job.view.selected_indices),
                    probe_loads={port: 'raw' if flag != 0 else 'compensated'
                                 for port, flag in zip(job.view.model.port_ids, job.view.model.M, strict=True)})
                for key, index in (('coordinate', job.coordinate_index), ('row', job.row_index),
                                   ('column', job.column_index), ('input', job.input_index), ('output', job.output_index)):
                    if index is not None:
                        observations[key] = job.view.terminal_ids[index]
                if job.family is not None:
                    observations['family'] = job.family
                failure = None
                if quantity_failure is not None:
                    failure = NumericalFailure(quantity_failure.kind, quantity_failure.stage, quantity_failure.detail,
                                               evidence_bytes(observations))
                elif code:
                    kind, stage = self.core.FAILURES[code]
                    failure = NumericalFailure(kind, stage, 'numerical operation did not satisfy its existing solve/root contract', evidence_bytes(observations))
                    values = {}
                results.append(EvaluationResult(job.id, failure=failure, evidence_bytes=evidence_bytes(observations), **values))
        return tuple(results)

    def close(self) -> None:
        self._template_cache.clear()
        self._executables.clear()
        self._closed = True
        self.trace = None
        self.operation_resources = None
