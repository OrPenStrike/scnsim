"""Construction-only three-subsystem floating two-probe circuit fixture."""

from __future__ import annotations

from dataclasses import dataclass

from fixtures import resonator_library, tapped_feedline
from scnsim import (
    CircuitPlan,
    ComponentInstance,
    ElectricNodeRef,
    PortRef,
    RLGC,
    components,
    units as u,
)


@dataclass(frozen=True)
class FloatingProbeFixture:
    """Plan-bound groups, nodes, and Ports needed by Lessons 11–14."""

    plan: CircuitPlan
    feedline: ComponentInstance
    readout: ComponentInstance
    floating: ComponentInstance
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
    feedline = plan.add(
        tapped_feedline.components.tapped_feedline(
            id="feedline",
            rlgc=feedline_rlgc,
            left_length=1.0 * u.mm,
            right_length=1.0 * u.mm,
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
        resonator_library.components.parallel_linear_lc_resonator(
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

    input_boundary = plan.bus(id="feedline_input")
    feedline_tap = plan.bus(id="feedline_tap")
    output_boundary = plan.bus(id="feedline_output")
    readout_node = plan.bus(id="readout_node")
    floating_plus = plan.bus(id="floating_plus")
    floating_minus = plan.bus(id="floating_minus")

    plan.link(id="feedline_input", endpoints=(input_boundary, feedline.pin("input")))
    plan.link(id="feedline_tap", endpoints=(feedline_tap, feedline.pin("tap")))
    plan.link(id="feedline_output", endpoints=(output_boundary, feedline.pin("output")))
    plan.series(
        id="feedline_readout_coupler",
        start=feedline_tap,
        elements=(feedline_readout_coupler,),
        end=readout_node,
    )
    plan.link(id="readout_terminal", endpoints=(readout_node, readout.pin("terminal")))
    plan.branch(
        id="readout_floating_plus",
        at=readout_node,
        elements=(readout_floating_plus,),
        end=floating_plus,
    )
    plan.branch(
        id="readout_floating_minus",
        at=readout_node,
        elements=(readout_floating_minus,),
        end=floating_minus,
    )
    plan.link(id="floating_plus", endpoints=(floating_plus, floating.pin("terminal_1")))
    plan.link(id="floating_minus", endpoints=(floating_minus, floating.pin("terminal_2")))

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
        feedline=feedline,
        readout=readout,
        floating=floating,
        feedline_in=feedline_in,
        feedline_out=feedline_out,
        floating_plus=floating_plus.node,
        floating_minus=floating_minus.node,
        probe_plus=probe_plus,
        probe_minus=probe_minus,
    )
