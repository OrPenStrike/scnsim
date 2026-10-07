"""Persistent isolated Julia numerical comparator on common Python descriptors.

This is a benchmark worker, not the normal terminal protocol or optimizer.
Each batch retains order and exact binary64 arrays. Worker death or corrupt
frames is fatal; only declared native numerical failures become outcomes.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import threading
from time import perf_counter_ns

from ...canonical import canonical_json_bytes
from ...execution.preparation import packaged_julia_resources, prepare_runtime
from ..models import EvaluationJob, EvaluationResult, NumericalFailure
from ..prepared import array_from_record
from .base import evidence_bytes, job_record


class JuliaBackend:
    def __init__(self, *, algorithm: str = "reuse", cpu_threads: int = 1,
                 julia_threads: int | None = None, julia_blas_threads: int | None = None,
                 executable: str | Path | None = None, project: str | Path | None = None):
        if algorithm not in ("reuse", "lu"):
            raise ValueError("Julia benchmark algorithm must be reuse or lu")
        # Native worker/library counts are independent of the host affinity quota.
        julia_threads = cpu_threads if julia_threads is None else julia_threads
        julia_blas_threads = cpu_threads if julia_blas_threads is None else julia_blas_threads
        started = perf_counter_ns()
        self._invocation_start_ns = started
        self._resources = ExitStack()
        if project is None:
            project = self._resources.enter_context(packaged_julia_resources())[0]
        executable = prepare_runtime().executable if executable is None else executable
        script = Path(project) / "bin" / "scnsim_benchmark.jl"
        env = dict(os.environ)
        env["JULIA_NUM_THREADS"] = str(julia_threads)
        env.update({name: str(julia_blas_threads) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
        self._stderr = []
        self._process = subprocess.Popen(
            [str(executable), "--startup-file=no", "--history-file=no", f"--threads={julia_threads}", f"--project={project}",
             str(script), "--numerical", algorithm, str(julia_blas_threads)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", env=env, bufsize=1)
        def read_stderr():
            for line in self._process.stderr:
                self._stderr.append(line.rstrip("\n"))
        self._reader = threading.Thread(target=read_stderr, daemon=True)
        self._reader.start()
        self.algorithm = algorithm
        self.cpu_threads = cpu_threads
        self.julia_threads = julia_threads
        self.julia_blas_threads = julia_blas_threads
        self._sequence = 0
        self._closed = False
        try:
            self._ready = self._read("scnsim.benchmark_backend_ready")
            if self._ready["julia_threads"] != julia_threads or self._ready["blas_threads"] != julia_blas_threads:
                raise RuntimeError("Julia benchmark worker thread settings disagree with the requested native profile")
        except BaseException:
            self.close()
            raise
        self.initialization_ns = perf_counter_ns() - started

    def _read(self, schema):
        line = self._process.stdout.readline()
        if not line:
            raise self._worker_failure()
        frame = json.loads(line)
        if frame.get("schema") != schema or canonical_json_bytes(frame) + b"\n" != line.encode():
            raise RuntimeError("Julia benchmark worker emitted an invalid protocol frame")
        return frame

    def _worker_failure(self):
        """Retain native exit diagnostics when transport closes before a reply."""
        self._process.wait()
        self._reader.join()
        return RuntimeError(f"Julia benchmark worker exited ({self._process.returncode}): " + "\n".join(self._stderr))

    def identity(self):
        return {"backend": "julia", "algorithm_id": f"scnsim.experimental.julia-{'symmetric-bk' if self.algorithm == 'reuse' else 'pivoted-lu'}-reuse-analytic-newton32.v1",
                "dtype": "float64/complex128", "device": "cpu", "requested_cpu_threads": self.cpu_threads,
                "requested_julia_threads": self.julia_threads, "requested_julia_blas_threads": self.julia_blas_threads,
                "initialization_ns": self.initialization_ns, "native": self._ready}

    def evaluate_batch(self, jobs: tuple[EvaluationJob, ...]) -> tuple[EvaluationResult, ...]:
        if self._closed:
            raise RuntimeError("numerical backend is closed")
        self._sequence += 1
        transfer_start = perf_counter_ns()
        payload = canonical_json_bytes({"batch_id": self._sequence, "jobs": [job_record(job) for job in jobs]})
        try:
            self._process.stdin.write(payload.decode() + "\n")
            self._process.stdin.flush()
        except BrokenPipeError as error:
            raise self._worker_failure() from error
        frame = self._read("scnsim.benchmark_backend_result")
        transfer_end = perf_counter_ns()
        transfer_compute_ns = transfer_end - transfer_start
        if frame.get("batch_id") != self._sequence or len(frame["results"]) != len(jobs):
            raise RuntimeError("Julia benchmark batch identity/order mismatch")
        results = []
        for job, record in zip(jobs, frame["results"], strict=True):
            if record.get("id") != job.id:
                raise RuntimeError("Julia benchmark result order mismatch")
            observed = {"native_timing": record["timing"], "batch_transfer_and_compute_ns": transfer_compute_ns,
                        "batch_id": self._sequence, "backend_invocation_start_ns": self._invocation_start_ns,
                        "batch_transfer_and_compute_start_ns": transfer_start,
                        "batch_transfer_and_compute_end_ns": transfer_end,
                        "batch_size": len(jobs), "input_bytes": len(payload), "algorithm_id": self.identity()["algorithm_id"]}
            failure = None
            values = {}
            if "failure" in record:
                declared = record["failure"]
                failure = NumericalFailure(declared["kind"], declared["stage"], declared["detail"], evidence_bytes(observed))
            else:
                source = record["values"]
                observed.update(source.get("evidence", {}))
                for name in ("S", "Y", "Z", "response_value", "root_omega_rad_s", "root_slope"):
                    if name in source:
                        decoded = array_from_record(source[name])
                        values[name] = decoded if name in ("S", "Y", "Z") else complex(decoded.item())
            results.append(EvaluationResult(job.id, failure=failure, evidence_bytes=evidence_bytes(observed), **values))
        return tuple(results)

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        process = getattr(self, "_process", None)
        if process is not None:
            if process.poll() is None:
                try:
                    process.stdin.write('{"event":"close"}\n')
                    process.stdin.flush()
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait()
            if hasattr(self, "_reader"):
                self._reader.join()
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
        self._resources.close()
