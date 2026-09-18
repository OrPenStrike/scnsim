"""The one public Pint registry used by every SCNSim physical input.

Import this module as ``from scnsim import units as u``.  A physical API
accepts only quantities made by this registry; plain numbers and quantities
from another registry have no portable unit or identity meaning.
"""

from __future__ import annotations

import math
from typing import Any

from pint import Quantity, UnitRegistry


registry = UnitRegistry(on_redefinition="raise")
"""SCNSim's sole public Pint registry."""

# Pint's conventional shorthand is useful for users who need to spell a
# quantity type, while unit symbols themselves are delegated below.
Q_ = registry.Quantity


def require_quantity(value: object, unit: str, *, name: str) -> Quantity:
    """Validate and return one finite SCNSim-registry quantity.

    This is the common trust-boundary check for public physical values.  It
    deliberately accepts compatible source units, but never a bare number or a
    quantity created by a different Pint registry.
    """

    if not isinstance(value, Quantity):
        raise TypeError(f"{name} must be a Pint Quantity from scnsim.units")
    if value._REGISTRY is not registry:
        raise TypeError(f"{name} must use the scnsim.units registry")
    try:
        converted = value.to(unit)
    except Exception as exc:  # Pint's dimensional error has useful detail.
        raise ValueError(f"{name} must have dimensionality compatible with {unit}") from exc
    magnitude = converted.magnitude
    if isinstance(magnitude, bool):
        raise TypeError(f"{name} must have a real scalar magnitude")
    try:
        finite = math.isfinite(float(magnitude))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must have a real scalar magnitude") from exc
    if not finite:
        raise ValueError(f"{name} must be finite")
    # Retain the caller's spelling for attempt provenance.  Canonical identity
    # converts it once through `_canonical.quantity_envelope` at its boundary.
    return value


def require_positive_quantity(value: object, unit: str, *, name: str) -> Quantity:
    """Return a finite positive physical quantity in the requested unit."""

    converted = require_quantity(value, unit, name=name)
    if converted.to(unit).magnitude <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return converted


def __getattr__(name: str) -> Any:
    """Expose Pint unit symbols through the one SCNSim registry."""

    if name.startswith("__"):
        raise AttributeError(name)
    try:
        return getattr(registry, name)
    except AttributeError:
        raise AttributeError(f"scnsim.units has no unit {name!r}") from None


__all__ = ["Q_", "Quantity", "registry", "require_positive_quantity", "require_quantity"]
