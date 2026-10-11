"""Read-only projection of current Workspace operation evidence.

Numerical execution belongs to normal Run methods and execution.python_host.
This boundary never prepares a backend, launches a task or binds another leaf.
"""
from __future__ import annotations


def benchmark(binding, *, operations=None, method=None, backend=None,
              precision=None, status=None):
    from ..diagnostics.operations import read_operations

    return read_operations(
        binding, operations=operations, method=method, backend=backend,
        precision=precision, status=status,
    )
