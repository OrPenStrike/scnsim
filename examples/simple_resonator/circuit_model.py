"""Team-owned façade for the simple resonator example.

This module illustrates the boundary SCNSim is intended to support.  The
circuit-model developer owns all topology and analysis choices below.  A model
user imports only ``ReadoutTarget``, ``optimize_readout()``, and
``resolve_readout()``; they do not repeat components, nets, references, ports,
or objective wiring in every Notebook.

The functions are complete UX examples but cannot execute while ``scnsim`` is
an API-only scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

from scnsim import (
    CMAESSpec,
    CircuitPlan,
    CircuitRun,
    CostObjective,
    DiagonalRootSpec,
    NetworkViewRef,
    OptimizationResult,
    OptimizationSpec,
    OptimizationVariable,
    ParameterRef,
    ReportResult,
    ReportSpec,
    library as sc,
    units as u,
)


@dataclass(frozen=True)
class ReadoutTarget:
    """Consumer-owned requirements; these are inputs, not SCNSim defaults."""

    frequency: object
    linewidth: object


def _build_model() -> tuple[CircuitPlan, ParameterRef]:
    """Build the team-owned Plan and return its private optimization handle."""

    plan = CircuitPlan(id="simple_readout")
    coupling_cap = plan.add(
        sc.capacitor(id="coupling_cap", capacitance=6.0 * u.fF)
    )
    resonator = plan.add(
        sc.grounded_parallel_linear_lc_resonator(
            id="readout",
            subsystem_capacitance=110.0 * u.fF,
            inductance=5.8 * u.nH,
        )
    )

    plan.reference("ground")
    plan.net("signal_in", coupling_cap.pin("a"))
    plan.net(
        "readout_node",
        coupling_cap.pin("b"),
        resonator.pin("signal"),
    )
    plan.add_port(
        id="signal_in",
        at="signal_in",
        role="terminated",
        reference_impedance=50.0 * u.ohm,
    )
    return plan, resonator.parameter("subsystem_capacitance")


def build_plan() -> CircuitPlan:
    """Return the reusable physical Plan for inspection by a model developer."""

    plan, _ = _build_model()
    return plan


def _optimization_request(
    target: ReadoutTarget,
    workspace: str | PathLike[str],
) -> tuple[CircuitRun, NetworkViewRef, OptimizationSpec]:
    """Reconstruct the exact Ref, model-owned root hint, and optimization."""

    plan, capacitance = _build_model()
    run = CircuitRun(plan=plan, workspace=workspace)
    readout_root = DiagonalRootSpec(
        coordinate="readout_node",
        root_hint=6.0 * u.GHz,
    )
    spec = OptimizationSpec(
        variables=(
            OptimizationVariable(
                parameter=capacitance,
                bounds=(80.0 * u.fF, 140.0 * u.fF),
            ),
        ),
        objectives=(
            CostObjective(
                id="readout_frequency",
                quantity=readout_root.frequency,
                target=target.frequency,
                weight=10.0 * u.dimensionless,
            ),
            CostObjective(
                id="readout_linewidth",
                quantity=readout_root.linewidth,
                target=target.linewidth,
                weight=1.0 * u.dimensionless,
            ),
        ),
        optimizer=CMAESSpec(seed=17, max_evaluations=200),
    )
    return run, run.original, spec


def optimize_readout(
    *,
    target: ReadoutTarget,
    workspace: str | PathLike[str],
) -> tuple[OptimizationResult, ReportResult]:
    """Execute the team-owned search and assemble a report from its exact result."""

    run, view, spec = _optimization_request(target, workspace)
    optimization = run.optimize(view, spec)
    report = run.build_report(ReportSpec(inputs=(optimization,)))
    return optimization, report


def resolve_readout(
    *,
    target: ReadoutTarget,
    workspace: str | PathLike[str],
) -> OptimizationResult:
    """Load the exact prior optimization after a kernel restart; never rerun it."""

    run, view, spec = _optimization_request(target, workspace)
    resolved = run.resolve(view, spec)
    if not isinstance(resolved, OptimizationResult):
        raise TypeError("exact request did not resolve to OptimizationResult")
    return resolved
