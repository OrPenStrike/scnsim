"""Future home of SCNSim's one public Pint unit registry.

Notebook, built-in component, and custom component code will all import this
module as ``from scnsim import units as u``.  The registry is deliberately not
created in the scaffold: accepting quantities before dimensional validation,
canonicalization, and serialization exist would create a false working path.
"""

from __future__ import annotations

from ._scaffold import unavailable


def __getattr__(name: str) -> object:
    """Reject unit access until the shared Pint registry is implemented."""

    if name.startswith("__"):
        raise AttributeError(name)
    unavailable(f"scnsim.units.{name}")
