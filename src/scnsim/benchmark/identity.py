"""Stable task and artifact identities from actual benchmark inputs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from ..canonical import canonical_json_bytes, sha256_hex


_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "XLA_FLAGS",
)


@dataclass(frozen=True, slots=True)
class _NativeThreadSpec:
    """Effective Julia and BLAS counts, separate from physical CPU affinity."""

    julia_threads: int
    blas_threads: int


def _native_thread_spec(
    cpu_threads: int,
    *,
    julia_threads: int | None = None,
    julia_blas_threads: int | None = None,
) -> _NativeThreadSpec:
    """Resolve each native count independently against the task's CPU quota."""
    return _NativeThreadSpec(
        cpu_threads if julia_threads is None else julia_threads,
        cpu_threads if julia_blas_threads is None else julia_blas_threads,
    )


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_snapshot(
    *,
    arm: str,
    device: str,
    cpu_threads: int,
    backend: Mapping[str, object] | None = None,
    native_threads: _NativeThreadSpec | None = None,
) -> dict[str, object]:
    """Capture the selected process, installed numerical packages and device."""
    executable = Path(sys.executable)
    executable_sha256 = sha256(executable.read_bytes()).hexdigest() if executable.is_file() else None
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    resource_cap = os.environ.get("SCNSIM_BENCHMARK_CPU_AFFINITY")
    snapshot = {
        "arm": arm,
        "python": {
            "executable": str(executable),
            "executable_sha256": executable_sha256,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": affinity,
            "benchmark_resource_cap": json.loads(resource_cap) if resource_cap is not None else None,
        },
        "device_requested": device,
        "cpu_threads_requested": cpu_threads,
        "thread_environment": {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT},
        "packages": {
            "scnsim": _distribution_version("scnsim"),
            "numpy": _distribution_version("numpy"),
            "jax": _distribution_version("jax"),
            "jaxlib": _distribution_version("jaxlib"),
            "cmaes": _distribution_version("cmaes"),
        },
        "backend": dict(backend or {}),
    }
    if native_threads is not None:
        snapshot["julia_threads_requested"] = native_threads.julia_threads
        snapshot["julia_blas_threads_requested"] = native_threads.blas_threads
    return snapshot


def environment_identity(snapshot: Mapping[str, object]) -> str:
    """Hash stable environment facts, excluding timing observations."""
    return sha256_hex(_stable_identity(dict(snapshot)))


def _stable_identity(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_identity(item)
            for key, item in value.items()
            if str(key) != "initialization_ns" and not str(key).endswith("_duration_ns")
            and not str(key).endswith("_elapsed_ns") and str(key) != "environment_sha256"
        }
    if isinstance(value, (list, tuple)):
        return [_stable_identity(item) for item in value]
    return value


def task_request_identity(
    *,
    benchmark_sha256: str,
    arm: str,
    environment_sha256: str,
    device: str,
    cpu_threads: int,
    backend_identity: Mapping[str, object],
    mesh_identity: Mapping[str, object] | None = None,
) -> str:
    """Bind one arm/profile request to its algorithm, environment and mesh."""
    return sha256_hex({
        "schema": "scnsim.benchmark_task_request",
        "schema_version": 1,
        "benchmark_sha256": benchmark_sha256,
        "arm": arm,
        "environment_sha256": environment_sha256,
        "device": device,
        "cpu_threads": cpu_threads,
        "backend": _stable_identity(dict(backend_identity)),
        "mesh": dict(mesh_identity or {}),
    })


def task_identifier(*, request_sha256: str, sample: int) -> str:
    """Derive a stable independent task identity for one complete sample."""
    return sha256_hex({
        "schema": "scnsim.benchmark_task",
        "schema_version": 1,
        "request_sha256": request_sha256,
        "sample": sample,
    })


def new_attempt_id() -> str:
    """Allocate an opaque attempt identifier for one actual launch."""
    return str(uuid.uuid4())


def artifact_reference(root: Path, path: Path, *, role: str) -> dict[str, object]:
    """Return content identity for a real regular artifact beneath ``root``."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"benchmark artifact is not a regular file: {path}")
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    relative = resolved_path.relative_to(resolved_root).as_posix()
    data = path.read_bytes()
    return {
        "path": relative,
        "role": role,
        "byte_length": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def checkpoint_seal(
    *,
    task_id: str,
    request_sha256: str,
    arm: str,
    sample: int,
    environment_sha256: str,
    attempt_id: str,
    checkpoint_sha256: str,
    byte_length: int,
) -> dict[str, object]:
    """Bind an immutable numeric checkpoint to its exact task and attempt."""
    body = {
        "schema": "scnsim.benchmark_checkpoint_seal",
        "schema_version": 1,
        "task_id": task_id,
        "request_sha256": request_sha256,
        "arm": arm,
        "sample": sample,
        "environment_sha256": environment_sha256,
        "attempt_id": attempt_id,
        "checkpoint_sha256": checkpoint_sha256,
        "byte_length": byte_length,
    }
    return {**body, "seal_sha256": sha256_hex(body)}
