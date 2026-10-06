"""Public structured authoring declarations and physical value interfaces.

Mutable scope assembly, catalog provenance, and immutable capture have separate
owners. Public imports retain the existing types and one builtin catalog.
"""

from .assembly import CompositePlan, SubsystemPlan
from .catalogs import Library, components
from .components import ComponentInstance
from .handles import (
    BranchRef, BusRef, CoordinateRef, CouplingRef, ElectricNodeRef, GroundRef,
    InductiveBranchRef, LinkRef, ParallelRef, PinRef, PortRef, SeriesRef,
    TapRef, TwoTerminalUse,
)
from .parameters import ParameterDefinitions, ParameterRef, ParameterSet, ParameterSpace
from .physical_values import AffineMap, ElectricalResolution, ParameterSpec, RLGC, RLGCParameterSpec
from .plan import CircuitPlan

__all__ = [
    "RLGC",
    "AffineMap",
    "ElectricalResolution",
    "CircuitPlan",
    "ComponentInstance",
    "CompositePlan",
    "CoordinateRef",
    "ElectricNodeRef",
    "InductiveBranchRef",
    "Library",
    "ParameterDefinitions",
    "ParameterRef",
    "ParameterSet",
    "ParameterSpace",
    "ParameterSpec",
    "RLGCParameterSpec",
    "BusRef",
    "TapRef",
    "TwoTerminalUse",
    "GroundRef",
    "SeriesRef",
    "ParallelRef",
    "BranchRef",
    "LinkRef",
    "CouplingRef",
    "PinRef",
    "PortRef",
    "SubsystemPlan",
    "components",
]
