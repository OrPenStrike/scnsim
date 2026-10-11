"""Stable operation and checkpoint identities from exact runtime inputs.

This module is deliberately independent of Workspace, execution coordinators,
and result models. It owns canonical identity construction shared by the
operation recorder and numerical execution adapters.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from ..canonical import sha256_hex


_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "XLA_FLAGS",
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
) -> dict[str, object]:
    """Capture the selected process, installed numerical packages and device."""
    executable = Path(sys.executable)
    executable_sha256 = sha256(executable.read_bytes()).hexdigest() if executable.is_file() else None
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    resource_cap = os.environ.get("SCNSIM_BENCHMARK_CPU_AFFINITY")
    return {
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
            "scipy": _distribution_version("scipy"),
            "jax": _distribution_version("jax"),
            "jaxlib": _distribution_version("jaxlib"),
            "cmaes": _distribution_version("cmaes"),
        },
        "backend": dict(backend or {}),
    }


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


def environment_identity(snapshot: Mapping[str, object]) -> str:
    """Hash stable environment facts, excluding timing observations."""
    return sha256_hex(_stable_identity(dict(snapshot)))


def operation_task_identifier(
    *,
    plan_sha256: str,
    request_sha256: str,
    method: str,
    backend: str,
    precision: str,
    resources: Mapping[str, object],
    algorithm_id: str,
    environment_sha256: str,
    checkpoint_policy: str,
    commit_every_generations: int = 1,
) -> str:
    """Bind resumable same-process work without changing numerical request identity."""
    return sha256_hex({
        "schema": "scnsim.operation_task",
        "schema_version": 1,
        "plan_sha256": plan_sha256,
        "request_sha256": request_sha256,
        "method": method,
        "backend": backend,
        "precision": precision,
        "resources": _stable_identity(dict(resources)),
        "algorithm_id": algorithm_id,
        "environment_sha256": environment_sha256,
        "checkpoint_policy": checkpoint_policy,
        "commit_every_generations": commit_every_generations,
    })


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
    """Bind immutable numeric checkpoint bytes to the exact task and attempt."""
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


__all__ = [
    "checkpoint_seal",
    "environment_identity",
    "environment_snapshot",
    "operation_task_identifier",
]
