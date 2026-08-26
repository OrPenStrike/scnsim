"""Public data-ingestion declarations for circuit-model authoring.

This module converts supported external solver artifacts into SCNSim authoring
values.  It does not run AEDT, import PyAEDT, or make an external artifact a
second circuit authority.  All call bodies remain fail-fast while the V1
contract is ``CONVERGING``.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike

from ._scaffold import unavailable
from .authoring import RLGC


def load_q2d_rlgc(
    path: str | PathLike[str],
    *,
    reference_conductor: str,
    conductor_map: Mapping[str, str] | None = None,
) -> RLGC:
    """Load frozen N-conductor RLGC matrices from an AEDT Q2D raw CSV.

    The reader requires exactly one primary capacitance, conductance,
    inductance, and resistance block with identical native order.
    ``conductor_map`` must be a complete bijective rename and cannot reorder or
    discard mutual terms.  The source must identify one consistent extractor
    +z direction, which becomes positive head-to-tail series current.  Source
    labels, units, extraction frequency, conductor order, +z direction, and
    content hash remain provenance; this helper does not run AEDT or prove
    solver convergence.
    """

    unavailable("load_q2d_rlgc")


__all__ = ["load_q2d_rlgc"]
