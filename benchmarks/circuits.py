"""Public finite circuit corpus for the assigned M0 baseline, outside the wheel.

Construction returns real operation callables, never executes them. The harness
owns timing, synchronization, failure capture and Run.close(). Root hints are
local authored seeds; a failed solve is retained rather than retried or tuned.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# Human-authored M0 workload controls, not production defaults or design criteria.
LBFGSB_CONTROLS = {
    "max_evaluations": 15000,
    "max_iterations": 15000,
    "ftol": 2.220446049250313e-09,
    "gtol": 1e-05,
    "max_line_search": 20,
    "commit_every_iterations": 1,
}


@dataclass
class BenchmarkCase:
    run: object
    operations: dict[str, Callable[[], object]]
    metadata: dict[str, object]
    select_point: Callable[[int], dict[str, object]]
    select_grid: Callable[[int], dict[str, object]] | None = None

    def close(self) -> None:
        self.run.close()


def case_ids() -> tuple[str, ...]:
    return ("C1", "C2", "C3", "C4-50", "C4-200", "C4-800", "C5", "C1-matched-lc",
            "F1-singular-static", "F2-certificate-hybrid", "F3-root-nonconvergence")


def _c1(s):
    u = s.units
    inputs = s.ParameterDefinitions(id="readme_lc")
    cap = inputs.parameter(id="capacitance", baseline=110.0 * u.fF,
                           spec=s.ParameterSpec(unit=u.fF))
    plan = s.CircuitPlan(id="readme_coupled_lc")
    child = plan.subsystem(id="resonator")
    c = child.add(s.components.capacitor(id="capacitor", capacitance=cap))
    l = child.add(s.components.inductor(id="inductor", inductance=5.8 * u.nH))
    bus = child.bus(id="signal")
    child.parallel(id="parallel_lc", start=bus, branches=((c,), (l,)), end=child.ground)
    pin = child.expose_pin(id="terminal", at=bus)
    coordinate = child.expose_coordinate(id="signal", at=bus)
    boundary = plan.bus(id="signal_boundary")
    cc = plan.add(s.components.capacitor(id="coupling_capacitor", capacitance=6.0 * u.fF))
    plan.series(id="coupling", start=boundary, elements=(cc,), end=pin)
    plan.add_port(id="signal_in", at=boundary, role="terminated", reference_impedance=50.0 * u.ohm)
    return plan, cap, ("signal_in",), (coordinate,), {
        "topology": "README grounded parallel LC plus series capacitive Port coupling",
        "finite_non_ground_nodes": 2, "capacitance_fF": 110.0,
        "inductance_nH": 5.8, "coupler_fF": 6.0, "port_ohm": 50.0,
        "coordinate_exposure": "resonator.signal is an analysis alias on the README signal bus",
    }


def _tank(s, plan, name, capacitance, inductance_nH):
    bus = plan.bus(id=name)
    c = plan.add(s.components.capacitor(id=name + "_c", capacitance=capacitance))
    l = plan.add(s.components.inductor(id=name + "_l", inductance=inductance_nH * s.units.nH))
    plan.parallel(id=name + "_lc", start=bus, branches=((c,), (l,)), end=plan.ground)
    return bus


def _couple(s, plan, name, start, end, capacitance_fF):
    c = plan.add(s.components.capacitor(id=name + "_c", capacitance=capacitance_fF * s.units.fF))
    plan.series(id=name, start=start, elements=(c,), end=end)


def _c2(s):
    u = s.units
    inputs = s.ParameterDefinitions(id="m0_two_modes")
    cap = inputs.parameter(id="mode0_capacitance", baseline=110.0 * u.fF,
                           spec=s.ParameterSpec(unit=u.fF))
    plan = s.CircuitPlan(id="m0_two_modes_readout_purcell")
    nodes = {}
    records = []
    for i in range(2):
        for role, capacitance, inductance in (
            ("mode", cap if i == 0 else 110.0 * u.fF, 5.8 + 0.1 * i),
            ("readout", 100.0 * u.fF, 6.0 + 0.1 * i),
            ("purcell", 95.0 * u.fF, 5.6 + 0.1 * i),
        ):
            name = role + str(i)
            nodes[name] = _tank(s, plan, name, capacitance, inductance)
            records.append(dict(id=name, capacitance_fF=110.0 if role == "mode" else 100.0 if role == "readout" else 95.0,
                                inductance_nH=inductance))
        boundary = plan.bus(id="boundary" + str(i))
        _couple(s, plan, "port_coupler" + str(i), boundary, nodes["purcell" + str(i)], 6.0)
        _couple(s, plan, "filter_coupler" + str(i), nodes["purcell" + str(i)], nodes["readout" + str(i)], 8.0)
        _couple(s, plan, "readout_coupler" + str(i), nodes["readout" + str(i)], nodes["mode" + str(i)], 6.0)
        plan.add_port(id="p" + str(i), at=boundary, role="terminated", reference_impedance=50.0 * u.ohm)
    _couple(s, plan, "mode_coupling", nodes["mode0"], nodes["mode1"], 2.0)
    return plan, cap, ("p0", "p1"), ("mode0", "mode1"), {
        "topology": "Two capacitively coupled modes, each with readout and Purcell LC and an external Port",
        "finite_non_ground_nodes": 8, "tanks": records,
        "couplers_fF": {"port": 6.0, "filter": 8.0, "readout": 6.0, "mode_pair": 2.0},
        "port_ohm": 50.0,
    }


def _ring(s, *, medium=False):
    u = s.units
    count = 8 if medium else 36
    inputs = s.ParameterDefinitions(id="m0_medium" if medium else "m0_chip")
    cap = inputs.parameter(id="mode0_capacitance", baseline=110.0 * u.fF,
                           spec=s.ParameterSpec(unit=u.fF))
    plan = s.CircuitPlan(id="m0_medium_ptc_pair" if medium else "m0_chip_ring")
    nodes, tanks = [], []
    for i in range(count):
        c = cap if i == 0 else (100.0 + 3.0 * i) * u.fF
        l = 6.0 - 0.1 * i
        nodes.append(_tank(s, plan, "mode" + str(i), c, l))
        tanks.append(dict(id="mode" + str(i), capacitance_fF=110.0 if i == 0 else 100.0 + 3.0 * i,
                          inductance_nH=l))
    for i in range(count):
        _couple(s, plan, "ring" + str(i), nodes[i], nodes[(i + 1) % count], 8.0)
    ports, probes = [], []
    for i in range(4):
        name = "p" + str(i)
        if medium and i >= 2:
            at = nodes[i - 2]
            role = "nonloading_probe"
        else:
            at = plan.bus(id="boundary" + str(i))
            _couple(s, plan, "port_coupler" + str(i), at, nodes[i * count // 4], 6.0)
            role = "terminated"
        port = plan.add_port(id=name, at=at, role=role, reference_impedance=50.0 * u.ohm)
        ports.append(name)
        if role == "nonloading_probe":
            probes.append(port)
    return plan, cap, tuple(ports), ("mode0", "mode1"), {
        "topology": "Capacitively coupled LC ring with four explicit Ports",
        "finite_non_ground_nodes": count + (2 if medium else 4), "tanks": tanks,
        "ring_coupler_fF": 8.0, "port_coupler_fF": 6.0, "port_ohm": 50.0,
        "port_roles": ("terminated", "terminated", "nonloading_probe", "nonloading_probe") if medium else ("terminated",) * 4,
    }, tuple(probes)


def _c4(s, sections):
    u = s.units
    inputs = s.ParameterDefinitions(id="m0_ordered_line")
    length = inputs.parameter(id="line_length", baseline=1.6 * u.mm,
                              spec=s.ParameterSpec(unit=u.mm))
    plan = s.CircuitPlan(id="m0_ordered_line_" + str(sections))
    rlgc = s.RLGC(conductors=("readout", "filter"), reference_conductor="ground",
        resistance_per_length=[[0.18, 0.0], [0.0, 0.22]] * u.ohm / u.m,
        inductance_per_length=[[420.0, 75.0], [75.0, 395.0]] * u.nH / u.m,
        conductance_per_length=[[0.0, 0.0], [0.0, 0.0]] * u.S / u.m,
        capacitance_per_length=[[175.0, -22.0], [-22.0, 168.0]] * u.pF / u.m)
    line = plan.add(s.components.transmission_line(id="coupled", length=length, rlgc=rlgc, n_sections=sections))
    ports = ("readout_head", "readout_tail", "filter_head", "filter_tail")
    buses = {name: plan.bus(id=name + "_bus") for name in ports}
    plan.series(id="line_binding", start=buses["readout_head"],
        elements=(line.between(line.pin("head", conductor="readout"), line.pin("tail", conductor="readout")),),
        end=buses["readout_tail"])
    for edge in ("head", "tail"):
        plan.link(id="filter_" + edge + "_binding", endpoints=(buses["filter_" + edge], line.pin(edge, conductor="filter")))
    for name in ports:
        plan.add_port(id=name, at=buses[name], role="terminated", reference_impedance=50.0 * u.ohm)
    return plan, length, ports, (), {
        "topology": "One complete ordered two-conductor finite-pi line; no scalarization",
        "n_sections": sections, "length_mm": 1.6, "conductors": ("readout", "filter"),
        "reference_conductor": "ground", "rlgc": rlgc._record(), "port_ohm": 50.0,
    }


def _point_selector(s, np, parameter, update, metadata):
    """Select declared float64 operands, never invalidate or bypass a cache."""
    baseline = np.float64(parameter.baseline.to(parameter.spec.si_unit).magnitude)

    def point_record(ordinal):
        value = baseline
        for _ in range(ordinal):
            value = np.nextafter(value, np.float64(np.inf))
        point = s.ParameterSet({parameter: s.units.registry.Quantity(value, parameter.spec.si_unit)})
        record = dict(ordinal=ordinal, parameter=parameter._key_record(),
                      si_unit=parameter.spec.si_unit, si_f64_hex=float(value).hex(),
                      parameter_set=point._record())
        return point, record

    metadata["point_series"] = [point_record(i)[1] for i in range(6)]
    metadata["point_series_rule"] = "ordinal0 explicit baseline SI; ordinal k applies float64 nextafter toward +infinity k times"

    def select_point(ordinal):
        point, record = point_record(ordinal)
        update(point)
        metadata["current_point"] = record
        metadata["parameter_set"] = point._record()
        return record

    select_point(0)
    return select_point


def _failure_case(s, np, case_id, *, storage, workspace, chunk_size):
    """Intended failure inputs; actual native outcomes remain the authority."""
    u = s.units
    if case_id == "F2-certificate-hybrid":
        # Same finite high_q_ext(1e8) model as the named oracle-harness probe.
        definitions = s.ParameterDefinitions(id="m0_high_q")
        parameter = definitions.parameter(id="capacitance", baseline=200.0 * u.fF,
                                          spec=s.ParameterSpec(unit=u.fF))
        plan = s.CircuitPlan(id="m0_high_q_certificate")
        child = plan.subsystem(id="dut")
        n0, n1 = child.bus(id="n0"), child.bus(id="n1")
        c0 = child.expose_coordinate(id="c0", at=n0)
        c1 = child.expose_coordinate(id="c1", at=n1)
        c = child.add(s.components.capacitor(id="c", capacitance=parameter))
        l = child.add(s.components.inductor(id="l", inductance=3.0 * u.nH))
        child.parallel(id="lc", start=n1, branches=((c,), (l,)), end=child.ground)
        omega0 = 1.0 / np.sqrt(3.0e-9 * 200.0e-15)
        coupling_farad = np.sqrt(200.0e-15 / (1.0e8 * omega0 * 50.0))
        cc = child.add(s.components.capacitor(id="cc", capacitance=coupling_farad * u.F))
        child.series(id="coupling", start=n0, elements=(cc,), end=n1)
        pin = child.expose_pin(id="port", at=n0)
        boundary = plan.bus(id="port_bus")
        plan.link(id="port_binding", endpoints=(boundary, pin))
        plan.add_port(id="P0", at=boundary, role="terminated", reference_impedance=50.0 * u.ohm)
        coordinates = (c1, c0)
        spec = s.HybridizedPoleSpec(coordinates=coordinates, anchor=omega0 / (2.0 * np.pi) * u.Hz)
        name, intended = "certificate_rejection", "high-Q Hybrid formation-scale certificate rejection"
        inputs = dict(capacitance_fF=200.0, inductance_nH=3.0, approximate_Q_ext=1.0e8,
                      port_ohm=50.0, coupling_farad=float(coupling_farad),
                      anchor_hz=float(omega0 / (2.0 * np.pi)),
                      source="probes/oracle-harness/gen.py high_q_ext/_cc_for_qext; repro_hybrid_highq.py",
                      anchor_difference="Unloaded analytic LC seed; original probe used its polished loaded-root real part. No oracle solve is added.")
    else:
        plan, parameter, _, coordinates, inputs = _c1(s)
        if case_id == "F1-singular-static":
            a, b = plan.bus(id="floating_a"), plan.bus(id="floating_b")
            floating = plan.add(s.components.capacitor(id="isolated_capacitor", capacitance=6.0 * u.fF))
            plan.series(id="isolated_pair", start=a, elements=(floating,), end=b)
            spec = s.StaticCapacitanceSpec()
            name, intended = "singular_matrix", "gauge-singular Cee for isolated floating capacitor pair"
            inputs = dict(inputs, isolated_capacitor_fF=6.0,
                          elimination="retain resonator.signal; explicit finite complement contains the isolated pair")
        else:
            spec = s.DiagonalRootSpec(coordinate=coordinates[0], root_hint=1.0e30 * u.Hz)
            name, intended = "root_nonconvergence", "finite far local seed, original Newton step limit unchanged"
            inputs = dict(inputs, authored_root_hint_hz=1.0e30)
    if storage == "memory":
        if workspace is not None:
            raise ValueError("memory corpus does not accept a Workspace")
        run = s.CircuitRun(plan=plan, storage="memory", precision="float64",
            execution_policy=_execution_policy(chunk_size))
    elif storage == "workspace":
        if workspace is None:
            raise ValueError("workspace corpus requires the executor's explicit path")
        run = s.CircuitRun(plan=plan, workspace=Path(workspace), precision="float64",
            execution_policy=_execution_policy(chunk_size))
    else:
        raise ValueError("storage must be memory or workspace")
    try:
        view = run.original.reduce(s.ReductionPipeline().retain(*coordinates))
        parameters = s.ParameterSet()
        metadata = dict(case_id=case_id, storage=storage, backend="jax", precision="float64",
            declared_inputs=inputs, intended_failure_category=intended,
            outcome_rule="Record actual returned Result or typed failure, even when it differs from the intended category; no forced failure or retry.",
            parameter=parameter._key_record(), specs={name: spec._canonical_record()},
            operations={name: intended}, pending_operations={})

        def update(point):
            nonlocal parameters
            parameters = point

        select_point = _point_selector(s, np, parameter, update, metadata)
        operations = {name: lambda: run.evaluate(view, spec, parameters=parameters)}
        return BenchmarkCase(run, operations, metadata, select_point)
    except BaseException as primary:
        try:
            run.close()
        except BaseException as cleanup:
            primary.add_note("Secondary benchmark Run.close failure: " + repr(cleanup))
        raise


def _execution_policy(chunk_size):
    from scnsim.execution.policy import ExecutionPolicy
    return ExecutionPolicy(frequency_chunk_size=chunk_size)


def _matched_lc_case(s, *, storage, workspace, chunk_size):
    """The retained matched S11-zero fixture, separate from C1's Z zero."""
    u = s.units
    plan = s.CircuitPlan(id="m0_matched_parallel_lc")
    signal = plan.bus(id="p")
    resistor = plan.add(s.components.resistor(id="resistor", resistance=50.0 * u.ohm))
    capacitor = plan.add(s.components.capacitor(id="capacitor", capacitance=80.0 * u.fF))
    inductor = plan.add(s.components.inductor(id="inductor", inductance=8.0 * u.nH))
    plan.parallel(id="matched_parallel_lc", start=signal,
                  branches=((resistor,), (capacitor,), (inductor,)), end=plan.ground)
    plan.add_port(id="p", at=signal, role="terminated", reference_impedance=50.0 * u.ohm)
    if storage == "memory":
        if workspace is not None:
            raise ValueError("memory corpus does not accept a Workspace")
        run = s.CircuitRun(plan=plan, storage="memory", precision="float64",
                           execution_policy=_execution_policy(chunk_size))
    elif storage == "workspace":
        if workspace is None:
            raise ValueError("workspace corpus requires the executor's explicit path")
        run = s.CircuitRun(plan=plan, workspace=Path(workspace), precision="float64",
                           execution_policy=_execution_policy(chunk_size))
    else:
        raise ValueError("storage must be memory or workspace")
    try:
        parameters = s.ParameterSet()
        spec = s.TransferZeroSpec(anchor=6.3 * u.GHz, family="S",
                                  input_coordinate="p", output_coordinate="p")
        metadata = dict(
            case_id="C1-matched-lc", storage=storage, backend="jax", precision="float64",
            declared_inputs=dict(resistance_ohm=50.0, capacitance_fF=80.0,
                                 inductance_nH=8.0, port_ohm=50.0, anchor_GHz=6.3,
                                 source="tests/test_direct_analysis.py _matched_zero_plan"),
            port_order=("p",), trace_port_order=("p",), parameter_set=parameters._record(),
            specs={"transfer_zero": spec._canonical_record()},
            operations={"transfer_zero": "matched parallel LC S11 zero on the original one-Port View"},
            pending_operations={},
            point_policy="Fixed original authored operands. Repeated Workspace requests may be readback; native0 is not warm compute.",
        )

        def select_point(ordinal):
            record = dict(ordinal=ordinal, parameter_set=parameters._record(),
                          operand_policy="fixed authored matched-LC baseline")
            metadata["current_point"] = record
            return record

        select_point(0)
        return BenchmarkCase(run=run,
            operations={"transfer_zero": lambda: run.evaluate(run.original, spec, parameters=parameters)},
            metadata=metadata, select_point=select_point)
    except BaseException as primary:
        try:
            run.close()
        except BaseException as cleanup:
            primary.add_note("Secondary benchmark Run.close failure: " + repr(cleanup))
        raise


def build_case(case_id, *, storage, workspace=None, chunk_size=64, points=401,
               cma_evaluations=64, tail_failure=False):
    """Build one declared fixture without invoking any operation.

    C1 local optimization uses the single Human-authored LBFGSB_CONTROLS
    workload configuration. Production optimizer defaults remain unchanged.
    """
    import numpy as np
    import scnsim as s
    from scnsim.execution.policy import ExecutionPolicy

    if case_id not in case_ids():
        raise KeyError(case_id)
    if case_id == "C1-matched-lc":
        return _matched_lc_case(s, storage=storage, workspace=workspace, chunk_size=chunk_size)
    if case_id.startswith("F"):
        return _failure_case(s, np, case_id, storage=storage, workspace=workspace, chunk_size=chunk_size)
    probes = ()
    if case_id == "C1":
        plan, parameter, ports, modes, declared = _c1(s)
    elif case_id == "C2":
        plan, parameter, ports, modes, declared = _c2(s)
    elif case_id in ("C3", "C5"):
        plan, parameter, ports, modes, declared, probes = _ring(s, medium=case_id == "C5")
    else:
        plan, parameter, ports, modes, declared = _c4(s, int(case_id.split("-")[1]))
    if storage == "memory":
        if workspace is not None:
            raise ValueError("memory corpus does not accept a Workspace")
        run = s.CircuitRun(plan=plan, storage="memory", precision="float64",
            execution_policy=ExecutionPolicy(frequency_chunk_size=chunk_size))
    elif storage == "workspace":
        if workspace is None:
            raise ValueError("workspace corpus requires the executor's explicit path")
        run = s.CircuitRun(plan=plan, workspace=Path(workspace), precision="float64",
            execution_policy=ExecutionPolicy(frequency_chunk_size=chunk_size))
    else:
        raise ValueError("storage must be memory or workspace")
    try:
        u = s.units
        parameters = s.ParameterSet()
        pipeline = s.ReductionPipeline()
        view_steps = []
        if probes:
            pipeline = pipeline.ptc(*probes).transform_pair(modes[0], modes[1], id="mode_pair")
            view_steps = [{"ptc": [p.id for p in probes]}, {"transform_pair": [*modes, "mode_pair"]}]
        selected_ports = ports[:2] if probes else ports
        # Direct traces address logical Port names. Element Specs address the
        # explicitly retained physical coordinate basis through their resolver.
        direct_view = run.original.reduce(pipeline) if probes else run.original
        element_view = run.original.reduce(pipeline.retain(*selected_ports))
        element_view_steps = [*view_steps, {"retain": selected_ports}]
        traces = tuple(s.SParameterTrace(id="s_" + output + "_from_" + source,
            input_port=source, input_mode=(), output_port=output, output_mode=())
            for source in selected_ports for output in selected_ports)
        interval_hz = (5.75e9, 6.25e9) if case_id == "C1" else (5.0e9, 7.0e9)

        def direct_for_grid(count):
            frequencies = np.linspace(*interval_hz, count)
            if tail_failure:
                frequencies[-1] = 1.0e308
            return s.DirectSolveSpec(frequencies=frequencies * u.Hz, traces=traces)

        direct = direct_for_grid(points)
        static = s.StaticCapacitanceSpec()
        static_view = run.original.reduce(pipeline)
        specs = {"direct": direct, "static_capacitance": static}
        views = {"direct": direct_view, "static_capacitance": static_view}
        operations = {
            "direct": lambda: run.solve(direct_view, direct, parameters=parameters),
            "static_capacitance": lambda: run.evaluate(static_view, static, parameters=parameters),
        }
        mapping = {"direct": "named logical-Port S channels on the full RF boundary", "static_capacitance": "full finite transformed C basis"}
        if case_id in ("C1", "C2"):
            root_view = run.original.reduce(s.ReductionPipeline().retain(*modes))
            diagonal = s.DiagonalRootSpec(coordinate=modes[0], root_hint=6.0 * u.GHz)
            hybrid_coordinates = (modes[0], ports[0]) if case_id == "C1" else modes
            hybrid_view = run.original.reduce(s.ReductionPipeline().retain(*hybrid_coordinates))
            hybrid = s.HybridizedPoleSpec(coordinates=hybrid_coordinates, anchor=6.0 * u.GHz)
            zero = s.TransferZeroSpec(anchor=6.0 * u.GHz, family="Z",
                input_coordinate=ports[0], output_coordinate=ports[-1])
            specs.update(diagonal_root=diagonal, hybridized_pole=hybrid, transfer_zero=zero)
            views.update(diagonal_root=root_view, hybridized_pole=hybrid_view, transfer_zero=element_view)
            operations.update(diagonal_root=lambda: run.evaluate(root_view, diagonal, parameters=parameters),
                hybridized_pole=lambda: run.evaluate(hybrid_view, hybrid, parameters=parameters),
                transfer_zero=lambda: run.evaluate(element_view, zero, parameters=parameters))
            quantities = {"frequency": diagonal.frequency.on(root_view)}
            output = s.OutputCoordinate(quantity_label="frequency")
            weight = 1.0 / u.Hz
            mapping.update(diagonal_root="local loaded mode0 diagonal root, hint6GHz", hybridized_pole="local two-coordinate pole, anchor6GHz", transfer_zero="local Z transfer zero at actual retained Port pair, anchor6GHz")
            if case_id == "C2":
                coupling = s.ResidueNormalizedCouplingSpec(branch_a=diagonal,
                    branch_b=s.DiagonalRootSpec(coordinate=modes[1], root_hint=6.2 * u.GHz), frequency=5.5 * u.GHz)
                specs["residue_coupling"] = coupling
                views["residue_coupling"] = root_view
                operations["residue_coupling"] = lambda: run.evaluate(root_view, coupling, parameters=parameters)
                mapping["residue_coupling"] = "two physical coupled-mode residue branches at explicit5.5GHz"
        else:
            response = s.ResponseElementSpec(family="Y", input_coordinate=selected_ports[0],
                output_coordinate=selected_ports[-1], frequency=5.5 * u.GHz)
            quantities = {"response_real": response.real.on(element_view)}
            output = s.OutputCoordinate(quantity_label="response_real")
            weight = 1.0 / u.S
            specs["sensitivity_response"] = response
            views["sensitivity_response"] = element_view
        direction = 1.0 * (u.mm if case_id.startswith("C4-") else u.fF)
        for name, derivative in (("jacobian", s.Jacobian()), ("jvp", s.JVP(direction={parameter: direction})),
                                 ("vjp", s.VJP(weights={output: weight}))):
            spec = s.SensitivitySpec(quantities=quantities, wrt=(parameter,), derivative=derivative)
            specs[name] = spec
            operations[name] = lambda spec=spec: run.sensitivity(spec, parameters=parameters)
            mapping[name] = "root-frequency derivative" if case_id in ("C1", "C2") else "real Y transfer derivative"
        pending = {}
        if case_id == "C1":
            variables = (s.OptimizationVariable(parameter=parameter, bounds=(80.0 * u.fF, 140.0 * u.fF)),)
            objectives = (s.CostObjective(id="f", quantity=diagonal.frequency, target=6.2 * u.GHz, weight=1.0 * u.dimensionless),)
            cma = s.OptimizationSpec(variables=variables, objectives=objectives,
                optimizer=s.CMAESSpec(seed=7, population_size=8, max_evaluations=cma_evaluations))
            specs["cma"] = cma
            views["cma"] = root_view
            operations["cma"] = lambda: run.optimize(root_view, cma, progress=False)
            mapping["cma"] = "existing probe_ops C1 80–140fF, target6.2GHz, seed7/population8; assigned64/512 budget"
            local = s.OptimizationSpec(variables=variables, objectives=objectives,
                optimizer=s.LBFGSBSpec(**LBFGSB_CONTROLS))
            specs["lbfgsb"] = local
            views["lbfgsb"] = root_view
            operations["lbfgsb"] = lambda: run.optimize(root_view, local, progress=False)
            mapping["lbfgsb"] = "same C1 variable/objective/authored baseline initial point; Human-authored M0 LBFGSB_CONTROLS"
        metadata = dict(case_id=case_id, storage=storage, backend="jax", precision="float64",
            chunk_size=chunk_size, points=points, tail_failure=tail_failure,
            failure_frequency_hz=1.0e308 if tail_failure else None,
            failure_source_index=points - 1 if tail_failure else None,
            failure_input_authority="probes/perf-prototype/capture.py frequency_grid overflow_middle; same finite1e308Hz moved to the assigned last lane",
            declared_inputs=declared, port_order=ports, parameter=parameter._key_record(),
            parameter_set=parameters._record(), direct_view_steps=view_steps,
            element_view_steps=element_view_steps, trace_port_order=selected_ports,
            specs={name: spec._canonical_record() for name, spec in specs.items()},
            operations=mapping, pending_operations=pending,
            failure_policy="Native typed failures and lane order are recorded unchanged; no retry/anchor adjustment or fallback.")
        if case_id == "C1":
            metadata["optimization_parameter_set"] = s.ParameterSet()._record()
            metadata["optimization_initial_parameters"] = s.ParameterSet({parameter: parameter.baseline})._record()
            metadata["optimization_point_policy"] = (
                "Active variables start at their authored ParameterRef baseline; optimize parameters are fixed non-active inputs. "
                "select_point does not change optimizer initialization. Repeated identical Workspace requests may reuse a saved result; "
                "only observed native dispatch counts as numerical compute."
            )

        def update(point):
            nonlocal parameters
            parameters = point

        select_point = _point_selector(s, np, parameter, update, metadata)

        def select_grid(count):
            nonlocal direct
            direct = direct_for_grid(count)
            specs["direct"] = direct
            record = dict(points=count, authored_interval_hz=interval_hz,
                          tail_failure=tail_failure, spec=direct._canonical_record())
            metadata["points"] = count
            metadata["specs"]["direct"] = record["spec"]
            metadata["current_grid"] = record
            metadata["failure_source_index"] = count - 1 if tail_failure else None
            return record

        metadata["current_grid"] = dict(points=points, authored_interval_hz=interval_hz,
            tail_failure=tail_failure, spec=direct._canonical_record())
        return BenchmarkCase(run=run, operations=operations, metadata=metadata,
                             select_point=select_point, select_grid=select_grid)
    except BaseException as primary:
        try:
            run.close()
        except BaseException as cleanup:
            primary.add_note("Secondary benchmark Run.close failure: " + repr(cleanup))
        raise
