"""Public schematic composition API, implemented in visualization."""

from .visualization.composition import (
    SchematicComposition,
    SchematicCompositionSnapshot,
    detached_composition,
)
from .visualization.composition.api import (
    SchematicArm,
    SchematicEndpoint,
    SchematicJunction,
    SchematicWiring,
    SchematicWiringRecord,
)

__all__ = ["SchematicComposition", "SchematicCompositionSnapshot"]
