"""Rebuild the primitive one-port resonator introduced in lesson 1."""

from __future__ import annotations

from dataclasses import dataclass

from scnsim import (
    CircuitPlan,
    ElectricNodeRef,
    ParameterRef,
    PortRef,
    components,
    units as u,
)


@dataclass(frozen=True)
class PrimitiveResonatorFixture:
    """Only the Plan-bound handles reused by later native-API lessons."""

    plan: CircuitPlan
    signal_port: PortRef
    resonator_node: ElectricNodeRef
    resonator_capacitance: ParameterRef


def build_primitive_resonator() -> PrimitiveResonatorFixture:
    """Return the exact primitive circuit from lesson 1 without policy wrappers."""

    plan = CircuitPlan(id="primitive_resonator")
    coupling_cap = plan.add(
        components.capacitor(id="coupling_cap", capacitance=6.0 * u.fF)
    )
    resonator_cap = plan.add(
        components.capacitor(id="resonator_cap", capacitance=110.0 * u.fF)
    )
    resonator_ind = plan.add(
        components.inductor(id="resonator_ind", inductance=5.8 * u.nH)
    )

    signal_boundary = plan.net(coupling_cap.pin("terminal_1"))
    resonator_node = plan.net(
        coupling_cap.pin("terminal_2"),
        resonator_cap.pin("terminal_1"),
        resonator_ind.pin("terminal_1"),
        id="resonator_node",
    )
    plan.ground(
        resonator_cap.pin("terminal_2"),
        resonator_ind.pin("terminal_2"),
    )
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
        resonator_capacitance=resonator_cap.parameter("capacitance"),
    )
