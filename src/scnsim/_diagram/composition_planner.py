"""One isolated default completion and one measured authoring realization.

Explicit declarations are authoritative. Geometry never chooses a different
recipe, side, order, or pose after defaults have been completed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._authoring_snapshot import ResolvedPlanPoint
from .composition_model import CapturedComposition, CompositionIntent
from .scene import NeutralScene
from .structured import StructuredLowerer


@dataclass(frozen=True, slots=True)
class LoweredComposition:
    scene: NeutralScene
    composition: CapturedComposition


def plan_composition(
    point: ResolvedPlanPoint, intent: CompositionIntent, *, show_values: bool
) -> LoweredComposition:
    """Return the single completed recipe and its same scene."""
    from ..composition import finalize_composition

    if not isinstance(point, ResolvedPlanPoint) or not isinstance(intent, CompositionIntent):
        raise TypeError("composition planning requires a resolved point and captured intent")
    if point.snapshot != intent.inventory.snapshot:
        raise ValueError("composition intent and resolved point have different Plan identities")
    captured = intent.materialize()
    lowerer = StructuredLowerer(point, captured, show_values=show_values)
    scene = lowerer.lower()
    final = finalize_composition(captured, tap_order=lowerer.selected_tap_orders())
    return LoweredComposition(scene, final)


__all__ = ["LoweredComposition", "plan_composition"]
