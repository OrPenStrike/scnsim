"""Read-only operation reports and immutable historical Benchmark readers.

Normal CircuitRun methods own numerical execution. Importing this package does
not initialize JAX, discover Julia, launch a process or mutate a workspace.
"""
from .models import BenchmarkResult

__all__ = ["BenchmarkResult"]
