"""Reusable parallel-LC composite used by the simple-resonator tutorials."""

from __future__ import annotations

from scnsim import (
    ComponentInstance,
    CompositePlan,
    Library,
    ParameterSpec,
    library as sc,
    units as u,
)


class ResonatorLibrary(Library):
    """Immutable catalog containing the tutorial's reusable LC package."""

    def parallel_linear_lc_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        inductance: object,
    ) -> ComponentInstance:
        """Package a grounded parallel capacitor and inductor behind one pin."""

        component = CompositePlan(id=id, library=self)
        resonator_capacitance = component.parameter(
            id="capacitance",
            baseline=capacitance,
            spec=ParameterSpec(unit=u.fF),
        )
        resonator_inductance = component.parameter(
            id="inductance",
            baseline=inductance,
            spec=ParameterSpec(unit=u.nH),
        )
        capacitor = component.add(
            sc.capacitor(
                id="capacitor",
                capacitance=resonator_capacitance,
            )
        )
        inductor = component.add(
            sc.inductor(
                id="inductor",
                inductance=resonator_inductance,
            )
        )
        terminal = component.net(
            capacitor.pin("terminal_1"),
            inductor.pin("terminal_1"),
        )
        component.ground(
            capacitor.pin("terminal_2"),
            inductor.pin("terminal_2"),
        )
        component.expose_pin(id="terminal", at=terminal)
        return component.build()


library = object.__new__(ResonatorLibrary)
"""Immutable custom Library object exported by this module."""
