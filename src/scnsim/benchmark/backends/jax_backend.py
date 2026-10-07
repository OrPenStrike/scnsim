"""Same-process CPU JAX adapter with dtype-preserving grouped evaluation.

Executable reuse is process-local and contains only static signatures and compiled
kernels. Plans, coefficients, Jobs, traces and numerical history remain call-local.
"""

from __future__ import annotations

import importlib.metadata
import os
from contextlib import nullcontext
from functools import partial
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


def _data(job: EvaluationJob, real_dtype, complex_dtype) -> tuple:
    view, model = job.view, job.view.model
    # Every repeated pi section in one physical line shares an impedance.
    groups: dict[tuple, list] = {}
    for block in model.series_rl:
        key = (block.resistance.shape, block.resistance.tobytes(), block.inductance.tobytes())
        if key not in groups:
            groups[key] = [[], block.resistance, block.inductance]
        groups[key][0].append(block.incidence)
    packed = tuple((np.stack(rows), R, L) for rows, R, L in groups.values())
    selected = np.asarray(view.selected_indices, dtype=np.int32)
    eliminated = np.asarray([i for i in range(len(model.node_ids)) if i not in view.selected_indices], dtype=np.int32)
    # Root-only Views do not use these empty boundary arguments.
    boundary = tuple(np.asarray(a) if a is not None else np.zeros(shape, dtype=dtype)
                     for a, shape, dtype in ((view.Bk, (len(model.node_ids), 0), np.float64),
                                             (view.Rk, (0, 0), np.float64),
                                             (view.Dk, (0, 0), np.float64),
                                             (view.Go, (len(model.port_ids), len(model.port_ids)), np.complex128)))
    def cast(array):
        array = np.asarray(array)
        if np.issubdtype(array.dtype, np.integer):
            return array
        return np.asarray(array, dtype=complex_dtype if np.iscomplexobj(array) else real_dtype)
    groups = tuple(tuple(cast(a) for a in group) for group in packed)
    return *(cast(a) for a in (model.C, model.K, model.G, model.B, model.R, model.M)), groups, selected, eliminated, *(cast(a) for a in boundary)


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
            "backend": "jax", "algorithm_id": "scnsim.jax-pivoted-lu-reuse-analytic-newton32.v1",
            "jax": importlib.metadata.version("jax"), "jaxlib": importlib.metadata.version("jaxlib"),
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

    def _evaluate_batch(self, jobs: tuple[EvaluationJob, ...]) -> tuple[EvaluationResult, ...]:
        if self._closed:
            raise RuntimeError("numerical backend is closed")
        grouped = {}
        results = [None] * len(jobs)
        for ordinal, job in enumerate(jobs):
            if job.kind not in ("direct", "response_element", "diagonal_root"):
                raise ValueError(f"unsupported JAX numerical operation: {job.kind}")
            if not job.view.port_realizable and (job.kind == "direct" or (job.kind == "response_element" and job.family == "S")):
                results[ordinal] = EvaluationResult(job.id, failure=NumericalFailure(
                    "port_realizability", "selected_network", "wave response requires a Port-realizable View"))
                continue
            data = _data(job, self.real_dtype, self.complex_dtype)
            family = "all" if job.kind == "direct" else job.family
            shape = tuple((a.shape, a.dtype.str) for a in self.jax.tree.leaves(data))
            frequencies = np.asarray(job.frequencies_hz)
            key = (job.kind, family, shape, None if job.kind == "diagonal_root" else frequencies.shape, job.view.port_realizable)
            grouped.setdefault(key, []).append((ordinal, job, data))
        for key, group in grouped.items():
            with _span(self.trace, "numerical_batch", operation=key[0], family=key[1], batch_size=len(group), arithmetic_precision=self.precision) as batch_span_id:
                self._evaluate_group(key, group, results, batch_span_id)
        return tuple(results)

    def _evaluate_group(self, key, group, results, batch_span_id):
        transfer_start = perf_counter_ns()
        data = self.jax.tree.map(lambda *values: np.stack(values), *(row[2] for row in group))
        if key[0] == "diagonal_root":
            starts = np.asarray([complex(row[1].omega_start_rad_s) if row[1].omega_start_rad_s is not None
                                 else complex(2 * np.pi * row[1].root_hint_hz) for row in group], dtype=self.complex_dtype)
            coordinates = np.asarray([row[1].coordinate_index for row in group], dtype=np.int32)
            args = (data, starts, coordinates)
            function = self.jax.vmap(self.core.diagonal_root)
        else:
            frequencies = np.stack([row[1].frequencies_hz for row in group])
            args = (data, np.asarray(2 * np.pi * frequencies, dtype=self.complex_dtype))
            kernel = self.core.network if key[4] else self.core.retained_response
            function = self.jax.vmap(self.jax.vmap(partial(kernel, family=key[1]), in_axes=(None, 0)))
        args = self.jax.device_put(args, self.device)
        self.jax.block_until_ready(args)
        transfer_ns = perf_counter_ns() - transfer_start
        leaves, tree = self.jax.tree.flatten(args)
        argument_shapes = tuple((tuple(a.shape), str(a.dtype)) for a in leaves)
        signature = (self.device.platform, self.device.id, self.cpu_threads, self.precision,
                     key[0], key[1], key[4], tree, argument_shapes)
        compilation_ns = None
        compile_start = None
        with _EXECUTABLE_LOCK:
            executable = _EXECUTABLES.get(signature)
            if executable is None:
                compile_start = perf_counter_ns()
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
        timing = {"batch_size": len(group), "transfer_to_device_ns": transfer_ns,
                  "shape_compilation_ns": compilation_ns, "synchronized_compute_ns": compute_ns,
                  "transfer_to_host_ns": download_ns,
                  "input_bytes": sum(a.size * a.dtype.itemsize for a in self.jax.tree.leaves(args)),
                  "output_bytes": sum(a.nbytes for a in self.jax.tree.leaves(output)),
                  "new_shape": compilation_ns is not None, "executable_cache_hit": compilation_ns is None,
                  "argument_shapes": [[list(shape), dtype] for shape, dtype in argument_shapes],
                  "transfer_to_device_start_ns": transfer_start, "shape_compilation_start_ns": compile_start,
                  "synchronized_compute_start_ns": compute_start, "transfer_to_host_start_ns": download_start}
        if self.trace is not None:
            for kind, start, duration in (
                ("transfer_to_device", transfer_start, transfer_ns),
                ("shape_compilation", compile_start, compilation_ns),
                ("synchronized_compute", compute_start, compute_ns),
                ("transfer_to_host", download_start, download_ns),
            ):
                if start is not None:
                    self.trace.measure(kind, start_tick_ns=start, end_tick_ns=start + duration,
                                       parent_span_id=batch_span_id,
                                       details={"batch_size": len(group), "arithmetic_precision": self.precision})
        for index, (ordinal, job, _) in enumerate(group):
            observations = {"batch": timing, "batch_index": index, "batch_span_id": batch_span_id,
                            "arithmetic_precision": self.precision, "algorithm_id": self.identity()["algorithm_id"]}
            observations["axes"] = {
                "original_node_ids": list(job.view.original_node_ids),
                "model_node_ids": list(job.view.model.node_ids),
                "terminal_ids": list(job.view.terminal_ids),
                "port_ids": list(job.view.model.port_ids),
                "probe_loads": {port: "raw" if flag != 0 else "compensated"
                                for port, flag in zip(job.view.model.port_ids, job.view.model.M, strict=True)},
                "selected_indices": list(job.view.selected_indices),
            }
            if job.kind == "diagonal_root":
                omega, slope, certificate, statuses, steps = output
                code = int(statuses[index])
                observations.update(newton_steps=int(steps[index]), certificate=certificate[index].tolist())
                values = {"root_omega_rad_s": complex(omega[index]), "root_slope": complex(slope[index])}
            else:
                S, Y, Z, statuses = output
                failed = np.flatnonzero(statuses[index])
                code = int(statuses[index, failed[0]]) if len(failed) else 0
                observations["frequency_count"] = len(job.frequencies_hz)
                if len(failed):
                    observations["first_failure_frequency_index"] = int(failed[0])
                values = {"S": S[index], "Y": Y[index], "Z": Z[index]} if job.kind == "direct" else {
                    "response_value": complex({"S": S, "Y": Y, "Z": Z}[job.family][index, 0, job.output_index, job.input_index])}
            failure = None
            if code:
                kind, stage = self.core.FAILURES[code]
                failure = NumericalFailure(kind, stage, "numerical operation did not satisfy its existing solve/root contract", evidence_bytes(observations))
                values = {}
            results[ordinal] = EvaluationResult(job.id, failure=failure, evidence_bytes=evidence_bytes(observations), **values)

    def close(self) -> None:
        self._closed = True
        self.trace = None
