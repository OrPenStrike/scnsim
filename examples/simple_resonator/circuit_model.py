"""Team-owned façade for the simple resonator built-in-composite example.

This module illustrates the boundary SCNSim is intended to support.  The
circuit-model developer owns all topology and analysis choices below.  A model
user imports ``ResonatorTarget``, the default-spec inspection helper,
``optimize_resonator()``, and ``resolve_resonator()``; they do not repeat
components, electric nodes, ground attachments, logical Ports, or objective
wiring in every Notebook.

The functions use SCNSim's built-in grounded-LC composite, introduced after
the primitive and custom-composite tutorials. They cannot execute while
``scnsim`` is an API-only scaffold.
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
    ReductionPipeline,
    ReportResult,
    ReportSpec,
    library as sc,
    units as u,
)


@dataclass(frozen=True)
class ResonatorTarget:
    """Consumer-owned requirements; these are inputs, not SCNSim defaults."""

    frequency: object
    linewidth: object


def _build_model() -> tuple[CircuitPlan, ParameterRef]:
    """Build the team-owned Plan and return its private optimization handle."""

    plan = CircuitPlan(id="simple_resonator")
    coupling_cap = plan.add(
        sc.capacitor(id="coupling_cap", capacitance=6.0 * u.fF)
    )
    resonator = plan.add(
        sc.grounded_parallel_linear_lc_resonator(
            id="resonator",
            capacitance=110.0 * u.fF,
            inductance=5.8 * u.nH,
        )
    )

    signal_boundary = plan.net(coupling_cap.pin("terminal_1"))
    plan.net(
        coupling_cap.pin("terminal_2"),
        resonator.pin("terminal"),
        id="resonator_node",
    )
    plan.add_port(
        id="signal_in",
        at=signal_boundary,
        role="terminated",
        reference_impedance=50.0 * u.ohm,
    )
    return plan, resonator.parameter("capacitance")


def build_plan() -> CircuitPlan:
    """Return the reusable physical Plan for inspection by a model developer."""

    plan, _ = _build_model()
    return plan


def _build_default_spec(
    capacitance: ParameterRef,
    target: ResonatorTarget,
) -> OptimizationSpec:
    """Bind the model-owned default recipe to one exact parameter handle."""

    resonator_root = DiagonalRootSpec(
        coordinate="resonator_node",
        root_hint=6.0 * u.GHz,
    )
    return OptimizationSpec(
        variables=(
            OptimizationVariable(
                parameter=capacitance,
                bounds=(80.0 * u.fF, 140.0 * u.fF),
            ),
        ),
        objectives=(
            CostObjective(
                id="resonator_frequency",
                quantity=resonator_root.frequency,
                target=target.frequency,
                weight=10.0 * u.dimensionless,
            ),
            CostObjective(
                id="resonator_linewidth",
                quantity=resonator_root.linewidth,
                target=target.linewidth,
                weight=1.0 * u.dimensionless,
            ),
        ),
        optimizer=CMAESSpec(seed=17, max_evaluations=200),
    )


def build_default_optimization_spec(
    target: ResonatorTarget,
) -> OptimizationSpec:
    """Return the inspectable model-author-owned default optimization recipe."""

    _, capacitance = _build_model()
    return _build_default_spec(capacitance, target)


def _optimization_request(
    target: ResonatorTarget,
    workspace: str | PathLike[str],
) -> tuple[CircuitRun, NetworkViewRef, OptimizationSpec]:
    """Reconstruct the exact Ref, model-owned root hint, and optimization."""

    plan, capacitance = _build_model()
    run = CircuitRun(plan=plan, workspace=workspace)
    spec = _build_default_spec(capacitance, target)
    quantity_view = run.original.reduce(
        ReductionPipeline().retain("resonator_node")
    )
    return run, quantity_view, spec


def optimize_resonator(
    *,
    target: ResonatorTarget,
    workspace: str | PathLike[str],
) -> tuple[OptimizationResult, ReportResult]:
    """Execute the team-owned search and assemble a report from its exact result."""

    run, view, spec = _optimization_request(target, workspace)
    optimization = run.optimize(view, spec)
    report = run.build_report(ReportSpec(inputs=(optimization,)))
    return optimization, report


def resolve_resonator(
    *,
    target: ResonatorTarget,
    workspace: str | PathLike[str],
) -> OptimizationResult:
    """Load the exact prior optimization after a kernel restart; never rerun it."""

    run, view, spec = _optimization_request(target, workspace)
    resolved = run.resolve(view, spec)
    if not isinstance(resolved, OptimizationResult):
        raise TypeError("exact request did not resolve to OptimizationResult")
    return resolved
