"""Process-owned immutable JAX/task CPU declaration, independent of Plans.

PJRT_NPROC is applied before backend initialization. This module neither imports
JAX nor changes affinity, platform selection or process-global precision.
The resource owner consumes the declared Optimization capacity and manages
native library limits; None preserves serial execution with the environment.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from threading import RLock

from ..errors import RuntimePreparationError


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    cpu_threads: int | None = None

    def __post_init__(self) -> None:
        if self.cpu_threads is not None:
            if isinstance(self.cpu_threads, bool) or not isinstance(self.cpu_threads, int):
                raise TypeError("cpu_threads must be an integer or None")
            if self.cpu_threads <= 0:
                raise ValueError("cpu_threads must be positive")

    @property
    def optimization_workers(self) -> int:
        """Task capacity; an omitted declaration preserves serial execution."""
        return 1 if self.cpu_threads is None else self.cpu_threads


_configuration = RuntimeConfiguration()
_initialized_configuration: RuntimeConfiguration | None = None
_lock = RLock()
_prior_pjrt_nproc: str | None = None
_owns_pjrt_nproc = False


def _backends_initialized() -> bool:
    # Pinned JAX 0.11.2 provides this non-initializing query. Looking only at an
    # already imported module preserves lazy base/readonly operations.
    bridge = sys.modules.get("jax._src.xla_bridge")
    return bridge is not None and bridge.backends_are_initialized()


def _restart_error() -> RuntimePreparationError:
    return RuntimePreparationError(
        "JAX resources are already initialized; restart the Kernel and call "
        "configure_runtime before numerical execution to change cpu_threads.",
        stage="runtime_configuration",
    )


def configure_runtime(*, cpu_threads: int | None = None) -> RuntimeConfiguration:
    """Declare JAX pool and task budget before initialization; repeats are inert."""
    global _configuration, _prior_pjrt_nproc, _owns_pjrt_nproc
    requested = RuntimeConfiguration(cpu_threads)
    with _lock:
        if _backends_initialized():
            if _initialized_configuration is not None:
                if requested != _initialized_configuration:
                    raise _restart_error()
            elif requested.cpu_threads is not None:
                # External initialization does not reveal its effective pool.
                raise _restart_error()
            return _configuration
        if requested.cpu_threads is not None:
            if not _owns_pjrt_nproc:
                _prior_pjrt_nproc = os.environ.get("PJRT_NPROC")
                _owns_pjrt_nproc = True
            os.environ["PJRT_NPROC"] = str(requested.cpu_threads)
        elif _owns_pjrt_nproc:
            if _prior_pjrt_nproc is None:
                os.environ.pop("PJRT_NPROC", None)
            else:
                os.environ["PJRT_NPROC"] = _prior_pjrt_nproc
            _owns_pjrt_nproc = False
        _configuration = requested
        return requested


def get_runtime_configuration() -> RuntimeConfiguration:
    with _lock:
        return _configuration


def ensure_jax_configuration() -> RuntimeConfiguration:
    """Check the declared pool without initializing or discovering a device."""
    with _lock:
        if _backends_initialized():
            if _initialized_configuration is not None:
                if _configuration != _initialized_configuration:
                    raise _restart_error()
            elif _configuration.cpu_threads is not None:
                raise _restart_error()
        elif _configuration.cpu_threads is not None:
            os.environ["PJRT_NPROC"] = str(_configuration.cpu_threads)
        return _configuration


def mark_jax_initialized(configuration: RuntimeConfiguration) -> None:
    """Record the declaration only after the adapter observes initialized JAX."""
    global _initialized_configuration
    with _lock:
        if configuration != _configuration:
            raise _restart_error()
        _initialized_configuration = configuration


def runtime_resource_identity() -> dict[str, object]:
    """Bind declaration and observed environment without claiming pool discovery."""
    configuration = get_runtime_configuration()
    return {"cpu_threads": configuration.cpu_threads,
            "pjrt_nproc": os.environ.get("PJRT_NPROC"),
            "allocation_policy": "scnsim.same_generation_fair_gate_blas1.v1",
            "optimization_workers": configuration.optimization_workers}
