"""Public capture-bound explicit schematic models and Plan entrypoints."""

from .visualization.schematic import (
    DiagramRef, DiagramPose, DiagramEndpoint, DiagramRoute, DiagramJunction,
    DiagramJump, DiagramCaption, DiagramLeader, DiagramCoupling, SchematicScopeLayout,
    SchematicLayout, SchematicPreparation, ScopeInventory, MeasuredFragment,
    ScopeMeasurement, GeometryRealization, _prepare_schematic, _render_schematic,
)

__all__ = [
    "DiagramRef", "DiagramPose", "DiagramEndpoint", "DiagramRoute",
    "DiagramJunction", "DiagramJump", "DiagramCaption", "DiagramLeader", "DiagramCoupling",
    "SchematicScopeLayout", "SchematicLayout", "SchematicPreparation",
    "ScopeInventory", "MeasuredFragment", "ScopeMeasurement", "GeometryRealization",
]
