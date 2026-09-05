"""Build one floating two-probe circuit for PTC, transform, and HB lessons."""

from __future__ import annotations

from dataclasses import dataclass

from scnsim import CircuitPlan, ElectricNodeRef, PortRef, RLGC, components, units as u


@dataclass(frozen=True)
class FloatingProbeFixture:
    """Plan-bound nodes and Ports needed by lessons 10–13."""

    plan: CircuitPlan
    feedline_in: PortRef
    feedline_out: PortRef
    floating_plus: ElectricNodeRef
    floating_minus: ElectricNodeRef
    probe_plus: PortRef
    probe_minus: PortRef


def build_floating_probe_circuit() -> FloatingProbeFixture:
    """Return a feedline-coupled readout and floating probed resonator."""

    plan = CircuitPlan(id="floating_probe_circuit")
    feedline_rlgc = RLGC(
        conductors=("signal",),
        reference_conductor="ground",
        resistance_per_length=[[0.0]] * u.ohm / u.m,
        inductance_per_length=[[420.0]] * u.nH / u.m,
        conductance_per_length=[[0.0]] * u.S / u.m,
        capacitance_per_length=[[175.0]] * u.pF / u.m,
    )
    feedline_left = plan.add(
        components.transmission_line(
            id="feedline_left",
            length=1.0 * u.mm,
            rlgc=feedline_rlgc,
            n_sections=1,
        )
    )
    feedline_right = plan.add(
        components.transmission_line(
            id="feedline_right",
            length=1.0 * u.mm,
            rlgc=feedline_rlgc,
            n_sections=1,
        )
    )
    feedline_readout_coupler = plan.add(
        components.capacitor(
            id="feedline_readout_coupler",
            capacitance=6.0 * u.fF,
        )
    )
    readout = plan.add(
        components.grounded_parallel_linear_lc_resonator(
            id="readout",
            capacitance=110.0 * u.fF,
            inductance=5.8 * u.nH,
        )
    )
    readout_floating_plus = plan.add(
        components.capacitor(
            id="readout_floating_plus",
            capacitance=4.0 * u.fF,
        )
    )
    readout_floating_minus = plan.add(
        components.capacitor(
            id="readout_floating_minus",
            capacitance=3.0 * u.fF,
        )
    )
    floating = plan.add(
        components.floating_parallel_linear_lc_resonator(
            id="floating",
            terminal_1_to_reference_capacitance=45.0 * u.fF,
            terminal_2_to_reference_capacitance=42.0 * u.fF,
            terminal_mutual_capacitance=16.0 * u.fF,
            inductance=7.0 * u.nH,
        )
    )

    input_boundary = plan.net(feedline_left.pin("head", conductor="signal"))
    feedline_tap = plan.net(
        feedline_left.pin("tail", conductor="signal"),
        feedline_right.pin("head", conductor="signal"),
        feedline_readout_coupler.pin("terminal_1"),
    )
    output_boundary = plan.net(feedline_right.pin("tail", conductor="signal"))
    readout_node = plan.net(
        feedline_readout_coupler.pin("terminal_2"),
        readout.pin("terminal"),
        readout_floating_plus.pin("terminal_1"),
        readout_floating_minus.pin("terminal_1"),
        id="readout_node",
    )
    floating_plus = plan.net(
        readout_floating_plus.pin("terminal_2"),
        floating.pin("terminal_1"),
        id="floating_plus",
    )
    floating_minus = plan.net(
        readout_floating_minus.pin("terminal_2"),
        floating.pin("terminal_2"),
        id="floating_minus",
    )

    feedline_in = plan.add_port(
        id="feedline_in",
        at=input_boundary,
        role="terminated",
        reference_impedance=50.0 * u.ohm,
    )
    feedline_out = plan.add_port(
        id="feedline_out",
        at=output_boundary,
        role="terminated",
        reference_impedance=50.0 * u.ohm,
    )
    probe_plus = plan.add_port(
        id="floating_probe_plus",
        at=floating_plus,
        role="nonloading_probe",
        reference_impedance=50.0 * u.ohm,
    )
    probe_minus = plan.add_port(
        id="floating_probe_minus",
        at=floating_minus,
        role="nonloading_probe",
        reference_impedance=50.0 * u.ohm,
    )
    return FloatingProbeFixture(
        plan=plan,
        feedline_in=feedline_in,
        feedline_out=feedline_out,
        floating_plus=floating_plus,
        floating_minus=floating_minus,
        probe_plus=probe_plus,
        probe_minus=probe_minus,
    )
