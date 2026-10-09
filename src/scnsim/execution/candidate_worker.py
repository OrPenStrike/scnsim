"""Lazy candidate backend construction shared by the owning actor threads.

No numerical initialization occurs on import. Candidate state and numerical
handles are created in the actor owner thread under its operation resource lease.
"""
from __future__ import annotations


def _codec():
    from ..benchmark.prepared import record_bytes, record_document
    return record_bytes, record_document



def create_backend_factory(declaration_bytes, *, operation_resources=None):
    declaration = _codec()[1](declaration_bytes)
    def factory():
        from .config import get_runtime_configuration
        from ..benchmark.backends.jax_backend import get_jax_backend
        return get_jax_backend(precision=declaration['precision'],
                               resources=get_runtime_configuration(), trace=None,
                               operation_resources=operation_resources, diagnostics=False)
    return factory

