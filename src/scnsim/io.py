"""Public data-ingestion declarations for circuit-model authoring.

This module converts supported external solver artifacts into SCNSim authoring
values.  It does not run AEDT, import PyAEDT, or make an external artifact a
second circuit authority.  All call bodies remain fail-fast while the V1
contract is ``CONVERGING``.
"""

from __future__ import annotations

from os import PathLike

from ._scaffold import unavailable
from .authoring import ScalarRLGC


def load_q2d_scalar_rlgc(path: str | PathLike[str]) -> ScalarRLGC:
    """Load one scalar RLGC value from a solver-native AEDT Q2D CSV.

    The candidate reader requires one 1x1 signal-conductor matrix in each of
    the native capacitance, conductance, inductance, and resistance blocks.  It
    preserves content-derived provenance and the extraction frequency, but a
    raw CSV alone does not prove solver convergence or receipt verification.
    """

    unavailable("load_q2d_scalar_rlgc")


__all__ = ["load_q2d_scalar_rlgc"]
