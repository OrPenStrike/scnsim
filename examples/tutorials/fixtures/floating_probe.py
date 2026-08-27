"""Build one floating two-probe circuit for PTC, transform, and HB lessons."""

from __future__ import annotations

from dataclasses import dataclass

from scnsim import CircuitPlan, ElectricNodeRef, PortRef, components, units as u


@dataclass(frozen=True)
class FloatingProbeFixture:
    """Plan-bound nodes and Ports needed by lessons 10–13."""

    plan: CircuitPlan
    feedline_in: PortRef
    feedline_out: PortRef
    qubit_plus: ElectricNodeRef
    qubit_minus: ElectricNodeRef
    probe_plus: PortRef
    probe_minus: PortRef


def build_floating_probe_circuit() -> FloatingProbeFixture:
    """Return a connected two-port line with one floating probed resonator."""

    plan = CircuitPlan(id="floating_probe_circuit")
    feedline_cap = plan.add(
        components.capacitor(id="feedline_cap", capacitance=35.0 * u.fF)
    )
    qubit_coupler = plan.add(
        components.capacitor(id="qubit_coupler", capacitance=4.0 * u.fF)
    )
    qubit = plan.add(
        components.floating_parallel_linear_lc_resonator(
            id="qubit",
            terminal_1_to_reference_capacitance=45.0 * u.fF,
            terminal_2_to_reference_capacitance=42.0 * u.fF,
            terminal_mutual_capacitance=16.0 * u.fF,
            inductance=7.0 * u.nH,
        )
    )

    input_boundary = plan.net(feedline_cap.pin("terminal_1"))
    output_boundary = plan.net(
        feedline_cap.pin("terminal_2"),
        qubit_coupler.pin("terminal_1"),
    )
    qubit_plus = plan.net(
        qubit_coupler.pin("terminal_2"),
        qubit.pin("terminal_1"),
        id="qubit_plus",
    )
    qubit_minus = plan.net(qubit.pin("terminal_2"), id="qubit_minus")

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
        id="qubit_probe_plus",
        at=qubit_plus,
        role="nonloading_probe",
        reference_impedance=50.0 * u.ohm,
    )
    probe_minus = plan.add_port(
        id="qubit_probe_minus",
        at=qubit_minus,
        role="nonloading_probe",
        reference_impedance=50.0 * u.ohm,
    )
    return FloatingProbeFixture(
        plan=plan,
        feedline_in=feedline_in,
        feedline_out=feedline_out,
        qubit_plus=qubit_plus,
        qubit_minus=qubit_minus,
        probe_plus=probe_plus,
        probe_minus=probe_minus,
    )
