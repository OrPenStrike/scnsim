"""Shared same-process JAX assembly and worker-local CPU SuperLU evaluation.

Executable reuse is process-local and contains only static signatures and compiled
kernels. Plans, coefficients, Jobs, traces and numerical history remain call-local.
"""

from __future__ import annotations

import importlib.metadata
import os
from contextlib import nullcontext
from threading import RLock
from time import perf_counter_ns

import numpy as np

from ..models import EvaluationJob, EvaluationResult, NumericalFailure
from .base import evidence_bytes


# Cache lifetime is the Python process; closing a call-local handle never drops
# reusable kernels. Lock only lookup/compile; executions use independent arguments.
_ALGORITHM_ID = "scnsim.jax-sparse-csc-superlu-direct-quantities-newton32.v7"
_EXECUTABLES = {}
_EXECUTABLE_LOCK = RLock()


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
        from ...execution.config import ensure_jax_configuration, get_runtime_configuration, mark_jax_initialized
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
            from . import jax_core
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
        self.initialization_ns = perf_counter_ns() - started

    def identity(self) -> dict[str, object]:
        return {
            "backend": "jax", "algorithm_id": _ALGORITHM_ID,
            "jax": importlib.metadata.version("jax"), "jaxlib": importlib.metadata.version("jaxlib"),
            "scipy": importlib.metadata.version("scipy"),
            "threadpoolctl": importlib.metadata.version("threadpoolctl"),
            "sparse_solver": "scipy.sparse.linalg.splu", "factorization_backend": "SuperLU",
            "superlu_version": None, "superlu_version_status": "not exposed by scipy public API",
            "assembly": "jax-runtime-coo-coalesced",
            "pure_inductive_stamp": "exact-realized-zero-R-real-dtype-scaled-analytic-width1-2-L-inverse-LU-larger", "assembly_chunk_limit": 8,
            "compensated_arithmetic": {
                "scope": "transfer_zero.Z.non_port_realizable",
                "representation": "two_components_in_requested_base_dtype",
                "coverage": "indexed_assembly_residual_selected_formation",
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
            args = self.jax.device_put((groups,), self.device)
            leaves, tree = self.jax.tree.flatten(args)
            signature = (self.device.platform, self.device.id, self.cpu_threads,
                         self.precision, 'pure_inductive_coefficients', tree,
                         tuple((tuple(a.shape), str(a.dtype)) for a in leaves))
            with _EXECUTABLE_LOCK:
                executable = _EXECUTABLES.get(signature)
                new_shape = executable is None
                if new_shape:
                    executable = self.jax.jit(self.core.inductive_coefficients).lower(*args).compile()
                    _EXECUTABLES[signature] = executable
            if self.operation_resources is not None:
                self.operation_resources.record_assembly(new_shape=new_shape)
            output = executable(*args)
            self.jax.block_until_ready(output)
            coefficients = self.jax.tree.map(np.asarray, output)
            prepared = tuple(group[:7] + (inverse, codes)
                             for group, (inverse, codes) in zip(groups, coefficients))
            system.payload = system.payload[:3] + (prepared,)
            system.inductive_coefficients_ready = True

    def _assembly(self, system, omegas, *, loaded, derivative, measurements, batches, compensated=False):
        self._prepare_inductive_coefficients(system)
        if self.operation_resources is not None and self.operation_resources.assembly_batching:
            omega_array = np.asarray(omegas, dtype=self.complex_dtype)
            leaves, tree = self.jax.tree.flatten((system.payload, omega_array))
            compatibility = (self.device.platform, self.device.id, self.precision,
                             loaded, derivative, compensated, system.n, tree,
                             tuple((tuple(value.shape), str(value.dtype)) for value in leaves))
            request = dict(compatibility=compatibility, payload=system.payload,
                           omegas=omega_array, loaded=loaded, derivative=derivative, compensated=compensated)
            return self.operation_resources.assemble(request, self._assemble_candidates_exclusive)
        return self._assembly_single(system, omegas, loaded=loaded, derivative=derivative,
                                     measurements=measurements, batches=batches, compensated=compensated)

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
                             self.precision, 'candidate_sparse_assembly', loaded, derivative, compensated,
                             tree, tuple((tuple(a.shape),str(a.dtype)) for a in leaves))
                with _EXECUTABLE_LOCK:
                    executable = _EXECUTABLES.get(signature)
                    new_shape = executable is None
                    if new_shape:
                        core = self.core
                        one_candidate = self.jax.vmap(
                            lambda data,omega: core.assemble(data,omega,loaded=loaded,derivative=derivative,compensated=compensated),
                            in_axes=(None,0))
                        function = self.jax.vmap(one_candidate,in_axes=(0,0))
                        executable = self.jax.jit(function).lower(*args).compile()
                        _EXECUTABLES[signature] = executable
                self.operation_resources.record_assembly(new_shape=new_shape)
                output = executable(*args)
                self.jax.block_until_ready(output)
                output = self.jax.tree.map(np.asarray,output)
                return [tuple(part[index] for part in output) for index in range(len(requests))]

    def _assembly_single(self, system, omegas, *, loaded, derivative, measurements, batches, compensated=False):
        """Exclusive transfer/JIT/compute/download before acquiring cache lock."""
        exclusive = nullcontext() if self.operation_resources is None else self.operation_resources.exclusive_assembly()
        with exclusive:
            return self._assemble_exclusive(system, omegas, loaded=loaded, derivative=derivative,
                                            measurements=measurements, batches=batches, compensated=compensated)

    def _assemble_exclusive(self, system, omegas, *, loaded, derivative, measurements, batches, compensated=False):
        transfer_start = perf_counter_ns() if self.diagnostics else None
        args = self.jax.device_put((system.payload, np.asarray(omegas, dtype=self.complex_dtype)), self.device)
        self.jax.block_until_ready(args)
        transfer_ns = perf_counter_ns() - transfer_start if self.diagnostics else None
        leaves, tree = self.jax.tree.flatten(args)
        shapes = tuple((tuple(a.shape), str(a.dtype)) for a in leaves)
        signature = (self.device.platform, self.device.id, self.cpu_threads, self.precision,
                     'sparse_assembly', loaded, derivative, compensated, tree, shapes)
        compile_start, compilation_ns = None, None
        with _EXECUTABLE_LOCK:
            executable = _EXECUTABLES.get(signature)
            new_shape = executable is None
            if new_shape:
                compile_start = perf_counter_ns() if self.diagnostics else None
                # Only operation controls are constants; coefficients and maps
                # remain runtime inputs to the process-wide executable cache.
                core = self.core
                function = self.jax.vmap(lambda data, omega: core.assemble(data, omega, loaded=loaded, derivative=derivative, compensated=compensated), in_axes=(None, 0))
                executable = self.jax.jit(function).lower(*args).compile()
                compilation_ns = perf_counter_ns() - compile_start if self.diagnostics else None
                _EXECUTABLES[signature] = executable
        if self.operation_resources is not None:
            self.operation_resources.record_assembly(new_shape=new_shape)
        compute_start = perf_counter_ns() if self.diagnostics else None
        output = executable(*args)
        self.jax.block_until_ready(output)
        compute_ns = perf_counter_ns() - compute_start if self.diagnostics else None
        download_start = perf_counter_ns() if self.diagnostics else None
        output = self.jax.tree.map(np.asarray, output)
        if self.diagnostics:
            download_ns = perf_counter_ns() - download_start
            row = dict(batch_size=len(omegas), measurement_start_tick_ns=transfer_start,
                       measurement_end_tick_ns=perf_counter_ns(),
                       transfer_to_device_ns=transfer_ns, shape_compilation_ns=compilation_ns,
                       synchronized_compute_ns=compute_ns, transfer_to_host_ns=download_ns,
                       new_shape=new_shape, executable_cache_hit=not new_shape,
                       argument_shapes=[[list(shape), dtype] for shape, dtype in shapes],
                       input_bytes=sum(a.size*a.dtype.itemsize for a in leaves),
                       output_bytes=sum(a.nbytes for a in self.jax.tree.leaves(output)),
                       static_pattern_sha256=system.pattern_sha256, static_pattern_cache_hit=False)
            batches.append(row)
            for kind, start, duration in (
                ('transfer_to_device', transfer_start, transfer_ns),
                ('shape_compilation', compile_start, compilation_ns),
                ('synchronized_compute', compute_start, compute_ns),
                ('transfer_to_host', download_start, download_ns),
            ):
                if start is not None and self.trace is not None:
                    self.trace.measure(kind, start_tick_ns=start, end_tick_ns=start+duration,
                                       parent_span_id=measurements.parent,
                                       details={'batch_size': len(omegas), 'arithmetic_precision': self.precision,
                                                'operation': 'sparse_assembly'})
        return output

    def _evaluate_batch(self, jobs: tuple[EvaluationJob, ...]) -> tuple[EvaluationResult, ...]:
        from .sparse_direct import Measurements, System, network, retained_response
        from .sparse_root import diagonal_root
        from .sparse_quantities import evaluate_quantity
        from .determinants import QuantityError
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
            started = perf_counter_ns() if self.diagnostics else None
            with _span(self.trace if self.diagnostics else None, 'numerical_batch', operation=job.kind,
                       batch_size=1, arithmetic_precision=self.precision) as parent:
                measurements = Measurements(self.trace, parent, enabled=self.diagnostics, resource=self.operation_resources)
                with measurements.phase('sparse_pattern_prepare'):
                    system = System(job.view, self.real_dtype, self.complex_dtype, template_cache=self._template_cache, resource=self.operation_resources)
                batches = []
                def assemble(omega, *, loaded, derivative, compensated=False):
                    arrays = self._assembly(system, [omega], loaded=loaded, derivative=derivative,
                                            measurements=measurements, batches=batches, compensated=compensated)
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
                                                    measurements=measurements, batches=batches)
                            from .sparse_direct import selected_state
                            for index in range(len(omegas[offset:offset+8])):
                                state, code = selected_state(system, tuple(a[index] for a in arrays), measurements)
                                if code:
                                    from .sparse_quantities import numerical_code
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
                                                       measurements=measurements, batches=batches)
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
                if self.diagnostics:
                    ended = perf_counter_ns()
                    compile_times = [row['shape_compilation_ns'] for row in batches if row['shape_compilation_ns'] is not None]
                    # Job intervals contain assembly intervals; they are not additive.
                    timing = dict(batch_size=1, measurement_start_tick_ns=started, measurement_end_tick_ns=ended,
                                  numerical_inclusive_ns=ended-started, assembly_batches=batches,
                                  shape_compilation_ns=sum(compile_times) if compile_times else None,
                                  new_shape=bool(compile_times), executable_cache_hit=all(row['executable_cache_hit'] for row in batches),
                                  transfer_to_device_ns=sum(row['transfer_to_device_ns'] for row in batches),
                                  synchronized_compute_ns=sum(row['synchronized_compute_ns'] for row in batches),
                                  transfer_to_host_ns=sum(row['transfer_to_host_ns'] for row in batches))
                    observations.update(batch=timing, batch_index=0, batch_span_id=parent, sparse_phases=measurements.rows)
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
        self._closed = True
        self.trace = None
