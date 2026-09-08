"""Diagram-only request declaration, isolated from numerical spec imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .composition import SchematicComposition
from .errors import SCNSimValidationError
from .presentation import Theme, _require_theme
from .schematic import SchematicLayout


@dataclass(frozen=True, slots=True)
class CircuitDiagramSpec:
    """Immutable request for an authoring or compiler-evidence diagram."""

    representation: Literal["authoring", "compiled"] = "authoring"
    theme: Theme = Theme.AUTO
    show_parameter_values: bool = True
    show_provenance: bool = True
    layout: SchematicComposition | SchematicLayout | None = None

    def __init__(
        self,
        *,
        representation: Literal["authoring", "compiled"] = "authoring",
        theme: Theme = Theme.AUTO,
        show_parameter_values: bool = True,
        show_provenance: bool = True,
        layout: SchematicComposition | SchematicLayout | None = None,
    ) -> None:
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "theme", _require_theme(theme))
        object.__setattr__(self, "show_parameter_values", show_parameter_values)
        object.__setattr__(self, "show_provenance", show_provenance)
        if layout is not None and not isinstance(layout, (SchematicComposition, SchematicLayout)):
            raise SCNSimValidationError(
                "layout must be a SchematicComposition, SchematicLayout, or None",
                stage="schematic_layout",
            )
        object.__setattr__(self, "layout", layout)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.representation not in {"authoring", "compiled"}:
            raise ValueError("invalid diagram representation")
        if self.representation == "compiled" and self.layout is not None:
            raise SCNSimValidationError(
                "schematic layout hints apply only to authoring diagrams",
                stage="schematic_layout",
            )
        if not isinstance(self.show_parameter_values, bool):
            raise TypeError("show_parameter_values must be bool")
        if not isinstance(self.show_provenance, bool):
            raise TypeError("show_provenance must be bool")
