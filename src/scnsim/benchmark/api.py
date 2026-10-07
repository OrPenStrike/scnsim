"""Read-only aggregation of recorded operations in a sealed Run's Plan leaf.

Numerical execution belongs to normal Run methods and execution.python_host.
This boundary never prepares a backend, launches a task or binds another leaf.
"""
from __future__ import annotations


def benchmark(run, *, operations=None, method=None, backend=None,
              precision=None, status=None):
    from .operations import read_operations

    return read_operations(
        run._binding, operations=operations, method=method, backend=backend,
        precision=precision, status=status,
    )
