"""Reusable public façade for the synthetic SCNSim IPF optimization example.

The model author owns topology, Ref lineages, quantity definitions, default
bounds, objective wiring, and optimizer controls.  A notebook consumer supplies
only a target and workspace unless they deliberately create an immutable custom
optimization spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

from circuit_library import library as custom
from scnsim import (
    CMAESSpec,
    CircuitPlan,
    CircuitRun,
    CostObjective,
    DiagonalRootSpec,
    DirectSolveResult,
    DirectSolveSpec,
    CoordinateRef,
    ElectricNodeRef,
    HBCaseSpec,
    HBBatchResult,
    HBSolveSpec,
    HBTruncation,
    NetworkViewRef,
    OptimizationResult,
    OptimizationSpec,
    OptimizationVariable,
    ParameterRef,
    PortRef,
    QuantitySum,
    RLGC,
    ReductionPipeline,
    ReportResult,
    ReportSpec,
    ResidueNormalizedCouplingSpec,
    SParameterTrace,
    TransferZeroSpec,
    library as sc,
    units as u,
)


@dataclass(frozen=True)
class IPFTarget:
    """Consumer-owned synthetic targets; none is an acceptance Gate."""

    readout_frequency: object
    filter_frequency: object
    transfer_zero_frequency: object
    coupling: object
    combined_linewidth: object
    response_frequencies: object


@dataclass(frozen=True)
class IPFModel:
    """Reusable model-author surface: one Plan, public refs, and default recipe."""

    plan: CircuitPlan
    input_boundary: ElectricNodeRef
    output_boundary: ElectricNodeRef
    feedline_in: ElectricNodeRef
    feedline_out: ElectricNodeRef
    qubit_plus: ElectricNodeRef
    qubit_minus: ElectricNodeRef
    filter_open_tail: CoordinateRef
    probe_plus: PortRef
    probe_minus: PortRef
    readout_open_length: ParameterRef
    shared_short_length: ParameterRef
    coupled_length: ParameterRef
    filter_open_length: ParameterRef
    idc_finger_length: ParameterRef

    def quantity_specs(
        self,
    ) -> tuple[
        DiagonalRootSpec,
        DiagonalRootSpec,
        TransferZeroSpec,
        ResidueNormalizedCouplingSpec,
    ]:
        """Return the reusable Direct roots, zero, and coupling definitions."""

        readout_root = DiagonalRootSpec(
            coordinate=self.feedline_out,
            root_hint=6.0 * u.GHz,
        )
        filter_root = DiagonalRootSpec(
            coordinate=self.filter_open_tail,
            root_hint=7.0 * u.GHz,
        )
        transfer_zero = TransferZeroSpec(
            anchor=6.5 * u.GHz,
            family="Y",
            input_coordinate=self.feedline_out,
            output_coordinate=self.filter_open_tail,
        )
        coupling = ResidueNormalizedCouplingSpec(
            branch_a=readout_root,
            branch_b=filter_root,
            frequency=6.5 * u.GHz,
        )
        return readout_root, filter_root, transfer_zero, coupling

    def build_default_optimization_spec(self, target: IPFTarget) -> OptimizationSpec:
        """Build the model-author-owned five-variable default search recipe."""

        readout_root, filter_root, transfer_zero, coupling = self.quantity_specs()
        return OptimizationSpec(
            variables=(
                OptimizationVariable(
                    parameter=self.readout_open_length,
                    bounds=(1.8 * u.mm, 3.0 * u.mm),
                ),
                OptimizationVariable(
                    parameter=self.shared_short_length,
                    bounds=(0.6 * u.mm, 1.2 * u.mm),
                ),
                OptimizationVariable(
                    parameter=self.coupled_length,
                    bounds=(1.1 * u.mm, 2.1 * u.mm),
                ),
                OptimizationVariable(
                    parameter=self.filter_open_length,
                    bounds=(1.5 * u.mm, 2.8 * u.mm),
                ),
                OptimizationVariable(
                    parameter=self.idc_finger_length,
                    bounds=(40.0 * u.um, 68.0 * u.um),
                ),
            ),
            objectives=(
                CostObjective(
                    id="readout_frequency",
                    quantity=readout_root.frequency,
                    target=target.readout_frequency,
                    weight=100.0 * u.dimensionless,
                ),
                CostObjective(
                    id="filter_frequency",
                    quantity=filter_root.frequency,
                    target=target.filter_frequency,
                    weight=100.0 * u.dimensionless,
                ),
                CostObjective(
                    id="transfer_zero",
                    quantity=transfer_zero.frequency,
                    target=target.transfer_zero_frequency,
                    weight=30.0 * u.dimensionless,
                ),
                CostObjective(
                    id="residue_coupling",
                    quantity=coupling.magnitude,
                    target=target.coupling,
                    weight=10.0 * u.dimensionless,
                ),
                CostObjective(
                    id="combined_linewidth",
                    quantity=QuantitySum(
                        readout_root.linewidth,
                        filter_root.linewidth,
                    ),
                    target=target.combined_linewidth,
                    weight=5.0 * u.dimensionless,
                ),
            ),
            optimizer=CMAESSpec(seed=17, max_evaluations=400),
        )


@dataclass(frozen=True)
class IPFSession:
    """One sealed Run and its reusable response/optimization views."""

    run: CircuitRun
    response_view: NetworkViewRef
    optimization_view: NetworkViewRef


@dataclass(frozen=True)
class IPFWorkflowResult:
    """Exact Results returned by the consumer-level façade."""

    optimization: OptimizationResult
    direct: DirectSolveResult
    hb: HBBatchResult
    report: ReportResult


def _manual_rlgc() -> tuple[RLGC, RLGC, RLGC]:
    """Declare public synthetic HB-compatible one- and two-trace matrices."""

    readout = RLGC(
        conductors=("readout",),
        reference_conductor="ground",
        resistance_per_length=[[0.18]] * u.ohm / u.m,
        inductance_per_length=[[420.0]] * u.nH / u.m,
        conductance_per_length=[[0.0]] * u.S / u.m,
        capacitance_per_length=[[170.0]] * u.pF / u.m,
    )
    filter_line = RLGC(
        conductors=("filter",),
        reference_conductor="ground",
        resistance_per_length=[[0.22]] * u.ohm / u.m,
        inductance_per_length=[[395.0]] * u.nH / u.m,
        conductance_per_length=[[0.0]] * u.S / u.m,
        capacitance_per_length=[[162.0]] * u.pF / u.m,
    )
    coupled = RLGC(
        conductors=("readout", "filter"),
        reference_conductor="ground",
        resistance_per_length=[[0.18, 0.0], [0.0, 0.22]] * u.ohm / u.m,
        inductance_per_length=[[420.0, 75.0], [75.0, 395.0]] * u.nH / u.m,
        conductance_per_length=[[0.0, 0.0], [0.0, 0.0]] * u.S / u.m,
        capacitance_per_length=[[175.0, -22.0], [-22.0, 168.0]] * u.pF / u.m,
    )
    return readout, filter_line, coupled


def build_model() -> IPFModel:
    """Build the full synthetic Plan once for model-author inspection."""

    readout_rlgc, filter_rlgc, coupled_rlgc = _manual_rlgc()
    plan = CircuitPlan(id="synthetic_ipf")
    ipf = plan.add(
        custom.intrinsic_purcell_filter(
            id="ipf",
            readout_rlgc=readout_rlgc,
            filter_rlgc=filter_rlgc,
            coupled_rlgc=coupled_rlgc,
        )
    )
    input_cap = plan.add(sc.capacitor(id="input_cap", capacitance=12.0 * u.fF))
    output_cap = plan.add(sc.capacitor(id="output_cap", capacitance=12.0 * u.fF))
    qubit_coupler = plan.add(
        sc.capacitor(id="qubit_coupler", capacitance=4.0 * u.fF)
    )
    qubit = plan.add(
        sc.floating_parallel_single_junction_resonator(
            id="qubit",
            terminal_1_to_reference_capacitance=45.0 * u.fF,
            terminal_2_to_reference_capacitance=42.0 * u.fF,
            terminal_mutual_capacitance=16.0 * u.fF,
            josephson_inductance=420.0 * u.pH,
            junction_capacitance=2.0 * u.fF,
        )
    )

    input_boundary = plan.net(input_cap.pin("terminal_1"))
    feedline_in = plan.net(
        input_cap.pin("terminal_2"),
        ipf.pin("feedline_in"),
        id="feedline_in_node",
    )
    feedline_out = plan.net(
        ipf.pin("feedline_out"),
        output_cap.pin("terminal_1"),
        qubit_coupler.pin("terminal_2"),
        id="feedline_out_node",
    )
    output_boundary = plan.net(output_cap.pin("terminal_2"))
    qubit_plus = plan.net(
        qubit.pin("terminal_1"),
        qubit_coupler.pin("terminal_1"),
        id="qubit_plus",
    )
    qubit_minus = plan.net(qubit.pin("terminal_2"), id="qubit_minus")

    plan.add_port(
        id="feedline_in",
        at=input_boundary,
        role="terminated",
        reference_impedance=50.0 * u.ohm,
    )
    plan.add_port(
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

    return IPFModel(
        plan=plan,
        input_boundary=input_boundary,
        output_boundary=output_boundary,
        feedline_in=feedline_in,
        feedline_out=feedline_out,
        qubit_plus=qubit_plus,
        qubit_minus=qubit_minus,
        filter_open_tail=ipf.coordinate("filter_open_tail"),
        probe_plus=probe_plus,
        probe_minus=probe_minus,
        readout_open_length=ipf.parameter("readout_open_length"),
        shared_short_length=ipf.parameter("shared_short_length"),
        coupled_length=ipf.parameter("coupled_length"),
        filter_open_length=ipf.parameter("filter_open_length"),
        idc_finger_length=ipf.parameter("idc_finger_length"),
    )


def build_session(
    model: IPFModel,
    *,
    workspace: str | PathLike[str],
) -> IPFSession:
    """Seal the Plan and derive the common PTC lineage and terminal views."""

    run = CircuitRun(plan=model.plan, workspace=workspace)
    transformed = run.original.reduce(
        ReductionPipeline()
        .ptc(model.probe_plus, model.probe_minus)
        .transform_pair(model.qubit_plus, model.qubit_minus, id="qubit")
    )
    response_view = transformed.reduce(
        ReductionPipeline().retain(model.input_boundary, model.output_boundary)
    )
    optimization_view = transformed.reduce(
        ReductionPipeline().retain(
            model.feedline_out,
            model.filter_open_tail,
        )
    )
    return IPFSession(
        run=run,
        response_view=response_view,
        optimization_view=optimization_view,
    )


def build_response_specs(target: IPFTarget) -> tuple[DirectSolveSpec, HBSolveSpec]:
    """Build winner-only Direct and pump-off HB response requests."""

    trace = SParameterTrace(
        id="transmission",
        input_port="feedline_in",
        input_mode=(),
        output_port="feedline_out",
        output_mode=(),
    )
    direct = DirectSolveSpec(
        frequencies=target.response_frequencies,
        traces=(trace,),
    )
    hb = HBSolveSpec(
        pump_axes=(),
        drives=(),
        frequencies=target.response_frequencies,
        cases=(HBCaseSpec(id="pump_off", currents={}),),
        truncation=HBTruncation(
            pump_harmonics=(),
            modulation_harmonics=(),
            three_wave_mixing=False,
            four_wave_mixing=False,
        ),
        traces=(trace,),
    )
    return direct, hb


def optimize_ipf(
    *,
    target: IPFTarget,
    workspace: str | PathLike[str],
) -> IPFWorkflowResult:
    """Run the model default, then reuse its winner for Direct/HB/reporting."""

    model = build_model()
    session = build_session(model, workspace=workspace)
    spec = model.build_default_optimization_spec(target)
    optimization = session.run.optimize(session.optimization_view, spec)
    direct_spec, hb_spec = build_response_specs(target)
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
    return IPFWorkflowResult(
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
    """Reconstruct and resolve the exact model-default search after restart."""

    model = build_model()
    session = build_session(model, workspace=workspace)
    resolved = session.run.resolve(
        session.optimization_view,
        model.build_default_optimization_spec(target),
    )
    if not isinstance(resolved, OptimizationResult):
        raise TypeError("exact request did not resolve to OptimizationResult")
    return resolved
