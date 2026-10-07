"""Experimental benchmark declarations; execution integration is separately owned.

Importing this package discovers neither Julia nor JAX and launches no process.
Existing CircuitRun operations keep their original Julia execution contract.
"""

from .models import BenchmarkResult, BenchmarkSpec, MeshGroup, MeshSpec

__all__ = ["BenchmarkResult", "BenchmarkSpec", "MeshGroup", "MeshSpec"]
