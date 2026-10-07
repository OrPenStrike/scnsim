"""Lazy optional JAX adapter with synchronized grouped numerical evaluation.

Shape compilation and transfer intervals are observed separately. The adapter
never starts Julia, captures candidate coefficients, or owns optimizer state.
"""

from __future__ import annotations

import importlib.metadata
import os
from time import perf_counter_ns

import numpy as np

from ..models import EvaluationJob, EvaluationResult, NumericalFailure
from .base import evidence_bytes


def _data(job: EvaluationJob) -> tuple:
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
    return model.C, model.K, model.G, model.B, model.R, model.M, packed, selected, eliminated, *boundary


class JaxBackend:
    def __init__(self, *, device: str = "cpu", cpu_threads: int = 1):
        if device != "cpu":
            raise ValueError("the experimental JAX benchmark supports only the CPU device")
        started = perf_counter_ns()
        import jax
        # Select the declared platform before discovery, including environments
        # that retain optional accelerator packages from earlier experiments.
        jax.config.update("jax_platforms", "cpu")
        jax.config.update("jax_enable_x64", True)
        from . import jax_core
        self.jax = jax
        self.core = jax_core
        self.device = jax.devices("cpu")[0]
        self.cpu_threads = cpu_threads
        self._compiled = {}
        self._closed = False
        self.initialization_ns = perf_counter_ns() - started

    def identity(self) -> dict[str, object]:
        return {
            "backend": "jax", "algorithm_id": "scnsim.experimental.jax-pivoted-lu-reuse-analytic-newton32.v1",
            "jax": importlib.metadata.version("jax"), "jaxlib": importlib.metadata.version("jaxlib"),
            "dtype": "float64/complex128", "device": str(self.device),
            "platform": self.device.platform, "device_kind": self.device.device_kind,
            "requested_cpu_threads": self.cpu_threads,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")},
            "initialization_ns": self.initialization_ns,
        }

    def evaluate_batch(self, jobs: tuple[EvaluationJob, ...]) -> tuple[EvaluationResult, ...]:
        if self._closed:
            raise RuntimeError("numerical backend is closed")
        grouped = {}
        results = [None] * len(jobs)
        for ordinal, job in enumerate(jobs):
            if job.kind not in ("direct", "response_element", "diagonal_root"):
                raise ValueError(f"unsupported benchmark numerical operation: {job.kind}")
            if not job.view.port_realizable and (job.kind == "direct" or (job.kind == "response_element" and job.family == "S")):
                results[ordinal] = EvaluationResult(job.id, failure=NumericalFailure(
                    "port_realizability", "selected_network", "wave response requires a Port-realizable View"))
                continue
            data = _data(job)
            family = "all" if job.kind == "direct" else job.family
            shape = tuple((a.shape, a.dtype.str) for a in self.jax.tree.leaves(data))
            frequencies = np.asarray(job.frequencies_hz)
            key = (job.kind, family, shape, None if job.kind == "diagonal_root" else frequencies.shape, job.view.port_realizable)
            grouped.setdefault(key, []).append((ordinal, job, data))
        for key, group in grouped.items():
            transfer_start = perf_counter_ns()
            data = self.jax.tree.map(lambda *values: np.stack(values), *(row[2] for row in group))
            if key[0] == "diagonal_root":
                starts = np.asarray([complex(row[1].omega_start_rad_s) if row[1].omega_start_rad_s is not None
                                     else complex(2 * np.pi * row[1].root_hint_hz) for row in group], dtype=np.complex128)
                coordinates = np.asarray([row[1].coordinate_index for row in group], dtype=np.int32)
                args = (data, starts, coordinates)
                function = self.jax.vmap(self.core.diagonal_root)
            else:
                frequencies = np.stack([row[1].frequencies_hz for row in group])
                args = (data, np.asarray(2 * np.pi * frequencies, dtype=np.complex128))
                kernel = self.core.network if key[4] else self.core.retained_response
                function = self.jax.vmap(self.jax.vmap(lambda d, w: kernel(d, w, key[1]), in_axes=(None, 0)))
            args = self.jax.device_put(args, self.device)
            self.jax.block_until_ready(args)
            transfer_ns = perf_counter_ns() - transfer_start
            signature = (key, len(group))
            compilation_ns = None
            compile_start = None
            if signature not in self._compiled:
                compile_start = perf_counter_ns()
                self._compiled[signature] = self.jax.jit(function).lower(*args).compile()
                compilation_ns = perf_counter_ns() - compile_start
            compute_start = perf_counter_ns()
            output = self._compiled[signature](*args)
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
                      "new_shape": compilation_ns is not None,
                      "transfer_to_device_start_ns": transfer_start, "shape_compilation_start_ns": compile_start,
                      "synchronized_compute_start_ns": compute_start, "transfer_to_host_start_ns": download_start}
            for index, (ordinal, job, _) in enumerate(group):
                observations = {"batch": timing, "batch_index": index, "algorithm_id": self.identity()["algorithm_id"]}
                observations["axes"] = {
                    "original_node_ids": list(job.view.original_node_ids),
                    "model_node_ids": list(job.view.model.node_ids),
                    "terminal_ids": list(job.view.terminal_ids),
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
        return tuple(results)

    def close(self) -> None:
        self._compiled.clear()
        self._closed = True
