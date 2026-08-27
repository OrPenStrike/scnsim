"""Terminal workflow for the synthetic SCNSim IPF optimization example.

This module owns requests that execute: default optimization, winner-only
Direct and pump-off HB responses, reporting, and exact receipt resolution.
Model construction remains in :mod:`circuit_model` so model inspection never
imports a terminal workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

from circuit_model import IPFModel, IPFSession, IPFTarget, build_model, build_session
from scnsim import (
    CurrentDrive,
    DirectSolveResult,
    DirectSolveSpec,
    HBCaseSpec,
    HBBatchResult,
    HBSolveSpec,
    HBTruncation,
    OptimizationResult,
    PumpAxis,
    ReportResult,
    ReportSpec,
    SParameterTrace,
    units as u,
)


@dataclass(frozen=True)
class WorkflowResult:
    """Exact immutable Results from one model-owned optimization workflow."""

    optimization: OptimizationResult
    direct: DirectSolveResult
    hb: HBBatchResult
    report: ReportResult


def build_response_specs(
    model: IPFModel,
    target: IPFTarget,
) -> tuple[DirectSolveSpec, HBSolveSpec]:
    """Build winner-only Direct and pump-off HB response requests."""

    direct_trace = SParameterTrace(
        id="transmission",
        input_port="feedline_in",
        input_mode=(),
        output_port="feedline_out",
        output_mode=(),
    )
    direct = DirectSolveSpec(
        frequencies=target.response_frequencies,
        traces=(direct_trace,),
    )
    pump_axis = PumpAxis(id="pump", frequency=9.0 * u.GHz)
    pump_drive = CurrentDrive(
        id="pump_drive",
        at=model.feedline_in_port,
        mode=(1,),
    )
    hb_trace = SParameterTrace(
        id="transmission",
        input_port="feedline_in",
        input_mode=(0,),
        output_port="feedline_out",
        output_mode=(0,),
    )
    hb = HBSolveSpec(
        pump_axes=(pump_axis,),
        drives=(pump_drive,),
        frequencies=target.response_frequencies,
        cases=(HBCaseSpec(id="pump_off", currents={}),),
        truncation=HBTruncation(
            pump_harmonics=(3,),
            modulation_harmonics=(1,),
            three_wave_mixing=False,
            four_wave_mixing=False,
        ),
        traces=(hb_trace,),
    )
    return direct, hb


def optimize_ipf(
    *,
    target: IPFTarget,
    workspace: str | PathLike[str],
) -> WorkflowResult:
    """Optimize the model default and materialize its winner-only responses."""

    model = build_model()
    session = build_session(model, workspace=workspace)
    optimization = session.run.optimize(
        session.optimization_view,
        model.build_default_optimization_spec(target),
    )
    direct_spec, hb_spec = build_response_specs(model, target)
    direct = session.run.solve(
        session.response_view,
        direct_spec,
        parameters=optimization.best.parameters,
    )
    hb = session.run.solve(
        session.response_view,
        hb_spec,
        parameters=optimization.best.parameters,
    )
    report = session.run.build_report(
        ReportSpec(inputs=(optimization, direct, hb.cases["pump_off"]))
    )
    return WorkflowResult(
        optimization=optimization,
        direct=direct,
        hb=hb,
        report=report,
    )


def resolve_ipf_optimization(
    *,
    target: IPFTarget,
    workspace: str | PathLike[str],
) -> OptimizationResult:
    """Resolve the exact model-default optimization receipt after restart."""

    model = build_model()
    session = build_session(model, workspace=workspace)
    resolved = session.run.resolve(
        session.optimization_view,
        model.build_default_optimization_spec(target),
    )
    if not isinstance(resolved, OptimizationResult):
        raise TypeError("exact request did not resolve to OptimizationResult")
    return resolved
