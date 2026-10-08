"""Same-process JAX sparse assembly and serial CPU SuperLU evaluation.

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
_EXECUTABLES = {}
_EXECUTABLE_LOCK = RLock()


def _span(trace, kind, **details):
    return nullcontext(None) if trace is None else trace.span(kind, details=details)


def get_jax_backend(*, precision, resources, trace):
    return JaxBackend(precision=precision, resources=resources, trace=trace)




class JaxBackend:
    def __init__(self, *, precision="float64", resources=None, trace=None, device="cpu", cpu_threads=None):
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
        self.initialization_ns = perf_counter_ns() - started

    def identity(self) -> dict[str, object]:
        return {
            "backend": "jax", "algorithm_id": "scnsim.jax-sparse-csc-superlu-direct-quantities-newton32.v4",
            "jax": importlib.metadata.version("jax"), "jaxlib": importlib.metadata.version("jaxlib"),
            "scipy": importlib.metadata.version("scipy"),
            "sparse_solver": "scipy.sparse.linalg.splu", "factorization_backend": "SuperLU",
            "superlu_version": None, "superlu_version_status": "not exposed by scipy public API",
            "assembly": "jax-runtime-coo-coalesced", "assembly_chunk_limit": 8,
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
        with self.jax.enable_x64(self.precision == "float64"):
            return self._evaluate_batch(jobs)

    def _assembly(self, system, omegas, *, loaded, derivative, measurements, batches):
        """One bounded runtime-operand assembly; no model closure in cache."""
        transfer_start = perf_counter_ns()
        args = self.jax.device_put((system.payload, np.asarray(omegas, dtype=self.complex_dtype)), self.device)
        self.jax.block_until_ready(args)
        transfer_ns = perf_counter_ns() - transfer_start
        leaves, tree = self.jax.tree.flatten(args)
        shapes = tuple((tuple(a.shape), str(a.dtype)) for a in leaves)
        signature = (self.device.platform, self.device.id, self.cpu_threads, self.precision,
                     'sparse_assembly', loaded, derivative, tree, shapes)
        compile_start, compilation_ns = None, None
        with _EXECUTABLE_LOCK:
            executable = _EXECUTABLES.get(signature)
            if executable is None:
                compile_start = perf_counter_ns()
                # Constants are only the operation controls. Every coefficient,
                # index and coalescing map belongs to the runtime argument tree.
                core = self.core
                function = self.jax.vmap(lambda data, omega: core.assemble(data, omega, loaded=loaded, derivative=derivative), in_axes=(None, 0))
                executable = self.jax.jit(function).lower(*args).compile()
                compilation_ns = perf_counter_ns() - compile_start
                _EXECUTABLES[signature] = executable
        compute_start = perf_counter_ns()
        output = executable(*args)
        self.jax.block_until_ready(output)
        compute_ns = perf_counter_ns() - compute_start
        download_start = perf_counter_ns()
        output = self.jax.tree.map(np.asarray, output)
        download_ns = perf_counter_ns() - download_start
        row = dict(batch_size=len(omegas), measurement_start_tick_ns=transfer_start,
                   measurement_end_tick_ns=perf_counter_ns(),
                   transfer_to_device_ns=transfer_ns, shape_compilation_ns=compilation_ns,
                   synchronized_compute_ns=compute_ns, transfer_to_host_ns=download_ns,
                   new_shape=compilation_ns is not None, executable_cache_hit=compilation_ns is None,
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
            started = perf_counter_ns()
            with _span(self.trace, 'numerical_batch', operation=job.kind,
                       batch_size=1, arithmetic_precision=self.precision) as parent:
                measurements = Measurements(self.trace, parent)
                with measurements.phase('sparse_pattern_prepare'):
                    system = System(job.view, self.real_dtype, self.complex_dtype)
                batches = []
                def assemble(omega, *, loaded, derivative):
                    arrays = self._assembly(system, [omega], loaded=loaded, derivative=derivative,
                                            measurements=measurements, batches=batches)
                    return tuple(a[0] for a in arrays)
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
                ended = perf_counter_ns()
                compile_times = [row['shape_compilation_ns'] for row in batches if row['shape_compilation_ns'] is not None]
                # The outer interval is a job observation; actual JAX assembly
                # batches retain their own clocks rather than copied estimates.
                timing = dict(batch_size=1, measurement_start_tick_ns=started, measurement_end_tick_ns=ended,
                              numerical_inclusive_ns=ended-started, assembly_batches=batches,
                              shape_compilation_ns=sum(compile_times) if compile_times else None,
                              new_shape=bool(compile_times), executable_cache_hit=all(row['executable_cache_hit'] for row in batches),
                              transfer_to_device_ns=sum(row['transfer_to_device_ns'] for row in batches),
                              synchronized_compute_ns=sum(row['synchronized_compute_ns'] for row in batches),
                              transfer_to_host_ns=sum(row['transfer_to_host_ns'] for row in batches))
                observations = {'quantity_kind': job.kind, 'terminal_ids': list(job.view.terminal_ids), 'batch': timing, 'batch_index': 0, 'batch_span_id': parent,
                                'arithmetic_precision': self.precision, 'algorithm_id': self.identity()['algorithm_id'],
                                'sparse_phases': measurements.rows, 'factors': system.factor_facts,
                                'static_pattern_sha256': system.pattern_sha256, **extra}
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
        self._closed = True
        self.trace = None
