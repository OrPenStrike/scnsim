"""Construction-only fixed-value Chapter 1 primitive resonator fixture."""

from __future__ import annotations

from dataclasses import dataclass

from scnsim import (
    CircuitPlan,
    ElectricNodeRef,
    PortRef,
    components,
    units as u,
)
from scnsim.authoring import SubsystemPlan


@dataclass(frozen=True)
class PrimitiveResonatorFixture:
    """Only the Plan-bound handles reused by later native-API lessons."""

    plan: CircuitPlan
    signal_port: PortRef
    resonator_node: ElectricNodeRef
    resonator: SubsystemPlan


def build_primitive_resonator() -> PrimitiveResonatorFixture:
    """Return the exact primitive circuit from lesson 1 without policy wrappers."""

    plan = CircuitPlan(id="primitive_resonator")
    resonator = plan.subsystem(id="resonator")
    resonator_cap = resonator.add(
        components.capacitor(id="capacitor", capacitance=110.0 * u.fF)
    )
    resonator_ind = resonator.add(
        components.inductor(id="inductor", inductance=5.8 * u.nH)
    )
    terminal = resonator.bus(id="terminal")
    resonator.parallel(
        id="parallel_lc",
        start=terminal,
        branches=((resonator_cap,), (resonator_ind,)),
        end=resonator.ground,
    )
    resonator_terminal = resonator.expose_pin(id="terminal", at=terminal)

    signal_boundary = plan.bus(id="signal_boundary")
    resonator_boundary = plan.bus(id="resonator_node")
    coupling_cap = plan.add(
        components.capacitor(id="coupling_cap", capacitance=6.0 * u.fF)
    )
    plan.series(
        id="coupling",
        start=signal_boundary,
        elements=(coupling_cap,),
        end=resonator_boundary,
    )
    plan.link(
        id="resonator_terminal",
        endpoints=(resonator_boundary, resonator_terminal),
    )
    resonator_node = resonator_boundary.node
    signal_port = plan.add_port(
        id="signal_in",
        at=signal_boundary,
        role="terminated",
        reference_impedance=50.0 * u.ohm,
    )
    return PrimitiveResonatorFixture(
        plan=plan,
        signal_port=signal_port,
        resonator_node=resonator_node,
        resonator=resonator,
    )
