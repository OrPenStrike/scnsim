"""Project-owned reusable resonator component catalog."""

from __future__ import annotations

from scnsim import (
    ComponentInstance,
    CompositePlan,
    Library,
    ParameterSpec,
    components as builtin_components,
    units as u,
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
        capacitance_ref = composite.parameter(
            id="capacitance",
            baseline=capacitance,
            spec=ParameterSpec(unit=u.fF),
        )
        inductance_ref = composite.parameter(
            id="inductance",
            baseline=inductance,
            spec=ParameterSpec(unit=u.nH),
        )
        capacitor = composite.add(
            builtin_components.capacitor(
                id="capacitor",
                capacitance=capacitance_ref,
            )
        )
        inductor = composite.add(
            builtin_components.inductor(
                id="inductor",
                inductance=inductance_ref,
            )
        )
        terminal = composite.net(
            capacitor.pin("terminal_1"),
            inductor.pin("terminal_1"),
        )
        composite.ground(
            capacitor.pin("terminal_2"),
            inductor.pin("terminal_2"),
        )
        composite.expose_pin(id="terminal", at=terminal)
        return composite.build()


components = ResonatorLibrary()
"""Immutable custom component catalog exported by this module."""
