"""Project-owned reusable resonator component catalog."""

from __future__ import annotations

from scnsim import (
    ComponentInstance,
    CompositePlan,
    Library,
    ParameterRef,
    components as builtin_components,
)


class ResonatorLibrary(Library):
    """Project catalog containing only reusable resonator factories."""

    def parallel_linear_lc_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        inductance: object,
    ) -> ComponentInstance:
        """Build a grounded parallel LC with one public terminal."""

        composite = CompositePlan(id=id, library=self)
        capacitor = composite.add(
            builtin_components.capacitor(
                id="capacitor",
                capacitance=capacitance,
            )
        )
        inductor = composite.add(
            builtin_components.inductor(
                id="inductor",
                inductance=inductance,
            )
        )
        terminal = composite.bus(id="terminal")
        composite.parallel(
            id="lc",
            start=terminal,
            branches=((capacitor,), (inductor,)),
            end=composite.ground,
        )
        composite.expose_pin(id="terminal", at=terminal)
        if isinstance(capacitance, ParameterRef):
            composite.expose_parameter(id="capacitance", parameter=capacitance)
        if isinstance(inductance, ParameterRef):
            composite.expose_parameter(id="inductance", parameter=inductance)
        return composite.build()


components = ResonatorLibrary()
"""Immutable custom component catalog exported by this module."""
