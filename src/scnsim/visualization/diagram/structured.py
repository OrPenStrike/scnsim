"""Explicit-layout authoring diagram realization.

All target capture and native measurement belongs to ``preparation``.  This
module only asks the preparation to realize its complete declared scope tree;
it contains no automatic placement, route selection, or missing-field repair.
"""

from __future__ import annotations

from ...errors import SCNSimValidationError
from ..schematic import SchematicLayout, SchematicPreparation
from .scene import NeutralScene


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_layout", evidence=evidence)


def realize_layout(preparation: SchematicPreparation, layout: SchematicLayout) -> NeutralScene:
    """Materialize exactly the captured geometry named by a full layout."""
    if not isinstance(preparation, SchematicPreparation):
        raise TypeError("explicit schematic realization requires SchematicPreparation")
    if not isinstance(layout, SchematicLayout):
        raise TypeError("explicit schematic realization requires SchematicLayout")
    if dict(preparation.identity) != dict(layout.preparation.identity):
        raise _fail("layout binding differs from the captured preparation")
    if layout.root.scope != preparation.root:
        raise _fail("layout root differs from the captured preparation")
    measured = preparation.measure_scope(preparation.root, layout.root)
    scene = measured.fragment
    if not isinstance(scene, NeutralScene):
        raise TypeError("scope realization did not return NeutralScene")
    return scene


__all__ = ["realize_layout"]
