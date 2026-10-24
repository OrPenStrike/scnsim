"""JAX Direct physics and H2 analytic comparisons."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from tests.analytic_conditions import (assert_forward, operator_condition, response_oracle,
    two_mode_roots, pole_condition, coupling_oracle, residue_oracle, matched_zero_oracle)
from scnsim import (
    CircuitPlan, CircuitRun, DirectQuantityResult, DirectSolveSpec,
    DiagonalRootSpec, HybridizedPoleSpec, OperatorResult, OperatorSpec,
    ReductionPipeline, ResidueNormalizedCouplingSpec, ResponseElementSpec,
    RLGC, TransferZeroSpec, components, units as u,
)

def _coupled_rlgc() -> RLGC:
    return RLGC(
        conductors=("readout", "filter"),
        reference_conductor="ground",
        resistance_per_length=[[0.18, 0.0], [0.0, 0.22]] * u.ohm / u.meter,
        inductance_per_length=[[420.0, 75.0], [75.0, 395.0]] * u.nH / u.meter,
        conductance_per_length=[[0.0, 0.0], [0.0, 0.0]] * u.siemens / u.meter,
        capacitance_per_length=[[175.0, -22.0], [-22.0, 168.0]] * u.pF / u.meter,
        extraction_frequency=5.0 * u.GHz,
    )


def _advanced_direct_plan():
    plan = CircuitPlan(id="advanced_direct_stabilization")
    capacitor_a = plan.add(components.capacitor(id="capacitor_a", capacitance=80.0*u.fF))
    inductor_a = plan.add(components.inductor(id="inductor_a", inductance=8.0*u.nH))
    capacitor_b = plan.add(components.capacitor(id="capacitor_b", capacitance=95.0*u.fF))
    inductor_b = plan.add(components.inductor(id="inductor_b", inductance=7.0*u.nH))
    coupling_capacitor = plan.add(components.capacitor(id="coupling_capacitor", capacitance=2.0*u.fF))
    coupling_inductor = plan.add(components.inductor(id="coupling_inductor", inductance=500.0*u.nH))
    a, b = plan.bus(id="a"), plan.bus(id="b")
    plan.parallel(id="mode_a", start=a, branches=((capacitor_a,), (inductor_a,)), end=plan.ground)
    plan.parallel(id="mode_b", start=b, branches=((capacitor_b,), (inductor_b,)), end=plan.ground)
    plan.parallel(id="coupling", start=a, branches=((coupling_capacitor,), (coupling_inductor,)), end=b)
    return plan


def _matched_zero_plan():
    plan = CircuitPlan(id="transfer_zero_stabilization")
    resistor = plan.add(components.resistor(id="resistor", resistance=50.0*u.ohm))
    capacitor = plan.add(components.capacitor(id="capacitor", capacitance=80.0*u.fF))
    inductor = plan.add(components.inductor(id="inductor", inductance=8.0*u.nH))
    signal = plan.bus(id="p")
    plan.parallel(id="matched_lc", start=signal,
                  branches=((resistor,), (capacitor,), (inductor,)), end=plan.ground)
    plan.add_port(id="p", at=signal, role="terminated", reference_impedance=50.0*u.ohm)
    return plan


def _floating_probe_plan():
    plan = CircuitPlan(id="floating_probe_stabilization")
    feed = plan.add(components.capacitor(id="feed", capacitance=35.0*u.fF))
    coupler = plan.add(components.capacitor(id="coupler", capacitance=4.0*u.fF))
    qubit = plan.add(components.floating_parallel_linear_lc_resonator(
        id="qubit", terminal_1_to_reference_capacitance=45.0*u.fF,
        terminal_2_to_reference_capacitance=42.0*u.fF,
        terminal_mutual_capacitance=16.0*u.fF, inductance=7.0*u.nH))
    input_node, output_node = plan.bus(id="input"), plan.bus(id="output")
    plus, minus = plan.bus(id="qubit_plus"), plan.bus(id="qubit_minus")
    plan.series(id="feed_branch", start=input_node, elements=(feed,), end=output_node)
    plan.series(id="coupler_branch", start=output_node, elements=(coupler,), end=plus)
    plan.series(id="qubit_branch", start=plus,
                elements=(qubit.between(qubit.pin("terminal_1"), qubit.pin("terminal_2")),), end=minus)
    feed_in = plan.add_port(id="feedline_in", at=input_node, role="terminated", reference_impedance=50.0*u.ohm)
    plan.add_port(id="feedline_out", at=output_node, role="terminated", reference_impedance=50.0*u.ohm)
    probe_plus = plan.add_port(id="probe_plus", at=plus, role="nonloading_probe", reference_impedance=50.0*u.ohm)
    probe_minus = plan.add_port(id="probe_minus", at=minus, role="nonloading_probe", reference_impedance=50.0*u.ohm)
    return plan, feed_in, probe_plus, probe_minus, ("qubit_plus", "qubit_minus")


@contextmanager
def _closed_run(**kwargs):
    run = CircuitRun(**kwargs)
    try:
        yield run
    finally:
        run.close()


class DirectDeclarationTests(unittest.TestCase):
    def test_transmission_line_and_reduction_grammar_preserve_declaration_order(self):
        rlgc = _coupled_rlgc()
        line = components.transmission_line(id="line", length=1.6*u.mm,
                                           rlgc=rlgc, n_sections=8)
        self.assertEqual(tuple(line.pins),
                         ("head.readout", "head.filter", "tail.readout", "tail.filter"))
        self.assertEqual(line.metadata["n_sections"], 8)
        self.assertEqual(rlgc.conductors, ("readout", "filter"))
        with self.assertRaises(ValueError):
            components.transmission_line(id="bad", length=1.0*u.mm,
                                        rlgc=_coupled_rlgc(), n_sections=0)
        plan, _, probe_plus, probe_minus, nodes = _floating_probe_plan()
        pipeline = (ReductionPipeline().ptc(probe_plus, probe_minus)
                    .transform_pair(*nodes, id="qubit")
                    .retain("feedline_in", "feedline_out", "qubit.differential"))
        self.assertIsNotNone(plan)
        with self.assertRaises(ValueError):
            pipeline.ptc(probe_plus)
        with self.assertRaises(ValueError):
            pipeline.transform_pair(*nodes, id="late")
        with self.assertRaises(ValueError):
            pipeline.retain("feedline_in")


class SelectedNetworkTests(unittest.TestCase):
    def test_n_trace_pi_ladder_and_n_port_direct_are_not_scalarized(self) -> None:
        plan = CircuitPlan(id="n_trace_direct")
        line = plan.add(components.transmission_line(id="line", length=1.6 * u.mm, rlgc=_coupled_rlgc(), n_sections=2))
        for end in ("head", "tail"):
            for conductor in ("readout", "filter"):
                node = plan.bus(id=f"{end}_{conductor}")
                plan.link(id=f"connect_{end}_{conductor}", endpoints=(node, line.pin(end, conductor=conductor)))
                plan.add_port(id=f"{end}_{conductor}", at=node, role="terminated", reference_impedance=50.0 * u.ohm)
        spec = DirectSolveSpec(frequencies=[1.0, 2.0] * u.GHz)
        with TemporaryDirectory() as workspace:
            with _closed_run(plan=plan, workspace=Path(workspace), precision="float64") as run:
                compiled = run.explain(run.original, spec).evidence["compiled"]
                sections = [row for row in compiled["expanded_branch_rows"]
                            if row["kind"] == "pi_section"]
                self.assertEqual([row["section"] for row in sections], [1, 2])
                self.assertTrue(all(tuple(row["conductors"]) == ("readout", "filter")
                                    for row in sections))
                self.assertEqual(len(compiled["node_order"]), 6)
                capacitance = compiled["c_matrix"]
                encoded = capacitance["values"]
                values = np.frombuffer(bytes.fromhex(encoded["data_hex"]),
                                       dtype=np.dtype(encoded["dtype"]).newbyteorder("<"))
                self.assertTrue(any(row != column and value != 0
                                    for row, column, value in zip(capacitance["rows"],
                                        capacitance["cols"], values, strict=True)))
                result = run.solve(run.original, spec)
                self.assertEqual(result.s.view.matrix.shape, (2, 4, 4))
                self.assertEqual(result.y.view.matrix.shape, (2, 4, 4))
                self.assertEqual(result.z.view.matrix.shape, (2, 4, 4))
                self.assertEqual(tuple(result.s.view.coordinates), ("head_readout", "head_filter", "tail_readout", "tail_filter"))

    def test_ptc_transform_retain_preserves_selected_matrix_and_probe_loads(self) -> None:
        plan, _, probe_plus, probe_minus, nodes = _floating_probe_plan()
        with TemporaryDirectory() as workspace:
            with _closed_run(plan=plan, workspace=Path(workspace), precision="float64") as run:
                view = run.original.reduce(
                    ReductionPipeline()
                    .ptc(probe_plus, probe_minus)
                    .transform_pair(*nodes, id="qubit")
                    .retain("feedline_in", "feedline_out", "qubit.differential")
                )
                spec = DirectSolveSpec(frequencies=[5.5] * u.GHz)
                explanation = run.explain(view, spec).evidence
                compiled = explanation["compiled"]
                primary = compiled["views"][compiled["primary_view_id"]]
                lineage = primary["lineage"]
                declaration = lineage["declaration"]
                self.assertEqual(tuple(declaration["ptc"]["selected_ports"]),
                                 ("probe_plus", "probe_minus"))
                ports = primary["ports"]
                encoded_mask = ports["load_mask"]
                load_mask = np.frombuffer(
                    bytes.fromhex(encoded_mask["data_hex"]),
                    dtype=np.dtype(encoded_mask["dtype"]).newbyteorder("<"),
                ).reshape(encoded_mask["shape"])
                self.assertEqual(load_mask.tolist(),
                                 [0.0 if port in ("probe_plus", "probe_minus") else 1.0
                                  for port in ports["ids"]])
                self.assertEqual(tuple(port for port in ports["ids"]
                                       if port in ("probe_plus", "probe_minus")),
                                 ("probe_plus", "probe_minus"))
                port_nets = {port["id"]: port["net"]
                             for port in explanation["connectivity"]["ports"]}
                self.assertEqual(tuple(primary["terminal_ids"]),
                                 (port_nets["feedline_in"], port_nets["feedline_out"], "qubit.differential"))
                self.assertEqual(tuple(declaration["retain"]["retained_coordinates"]),
                                 tuple(primary["terminal_ids"]))
                self.assertTrue(primary["port_realizable"])
                result = run.solve(view, spec)
                self.assertEqual(result.s.view.matrix.shape, (1, 3, 3))
                self.assertEqual(result.s.view.probe_loads["probe_plus"], "compensated")
                self.assertEqual(result.s.view.probe_loads["probe_minus"], "compensated")


class DirectQuantityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = TemporaryDirectory()
        cls.addClassCleanup(cls._temporary.cleanup)
        cls.workspace = Path(cls._temporary.name)
        cls.plan = _advanced_direct_plan()
        cls.circuit_run = CircuitRun(plan=cls.plan, workspace=cls.workspace, precision="float64")
        cls.addClassCleanup(cls.circuit_run.close)
        cls.view = cls.circuit_run.original.reduce(ReductionPipeline().retain("a", "b"))
        cls.operator_spec = OperatorSpec(frequencies=[5.0, 5.5] * u.GHz)
        cls.response_spec = ResponseElementSpec(
            family="Y", input_coordinate="a", output_coordinate="b", frequency=5.5 * u.GHz
        )
        cls.pole_spec = HybridizedPoleSpec(coordinates=("a", "b"), anchor=6.2 * u.GHz)
        cls.coupling_spec = ResidueNormalizedCouplingSpec(
            branch_a=DiagonalRootSpec(coordinate="a", root_hint=6.2 * u.GHz),
            branch_b=DiagonalRootSpec(coordinate="b", root_hint=6.2 * u.GHz),
            frequency=5.5 * u.GHz,
        )
        cls.operator = cls.circuit_run.evaluate(cls.view, cls.operator_spec)
        cls.response = cls.circuit_run.evaluate(cls.view, cls.response_spec)
        cls.pole = cls.circuit_run.evaluate(cls.view, cls.pole_spec)
        cls.coupling = cls.circuit_run.evaluate(cls.view, cls.coupling_spec)

        cls._zero_temporary = TemporaryDirectory()
        cls.addClassCleanup(cls._zero_temporary.cleanup)
        cls.zero_plan = _matched_zero_plan()
        cls.zero_run = CircuitRun(plan=cls.zero_plan, workspace=Path(cls._zero_temporary.name), precision="float64")
        cls.addClassCleanup(cls.zero_run.close)
        cls.zero_spec = TransferZeroSpec(
            anchor=6.3 * u.GHz,
            family="S",
            input_coordinate="p",
            output_coordinate="p",
        )
        cls.zero = cls.zero_run.evaluate(cls.zero_run.original, cls.zero_spec)

    @staticmethod
    def _matrices(omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        capacitance = np.asarray([[82.0, -2.0], [-2.0, 97.0]]) * 1.0e-15
        stiffness = np.asarray(
            [
                [1.0 / (8.0e-9) + 1.0 / (500.0e-9), -1.0 / (500.0e-9)],
                [-1.0 / (500.0e-9), 1.0 / (7.0e-9) + 1.0 / (500.0e-9)],
            ]
        )
        return capacitance, stiffness, stiffness - omega**2 * capacitance

    def test_operator_and_response_match_the_declared_dynamic_equations(self) -> None:
        self.assertIsInstance(self.operator, OperatorResult)
        self.assertEqual(tuple(point.frequency.to("gigahertz").magnitude for point in self.operator.points), (5.0, 5.5))
        for point in self.operator.points:
            omega = 2.0 * math.pi * float(point.frequency.to("hertz").magnitude)
            _, _, expected = self._matrices(omega)
            actual = np.asarray(point.matrix.to("siemens / second").magnitude)
            assert_forward(self, "operator", actual, expected, operator_condition(omega), "S/s", frequency_perturbed=True)

        self.assertIsInstance(self.response, DirectQuantityResult)
        omega = 2.0 * math.pi * 5.5e9
        expected = 1j * (omega * 2.0e-15 - 1.0 / (omega * 500.0e-9))
        actual = complex(self.response.value.to("siemens").magnitude)
        assert_forward(self, "Yab", actual, expected, response_oracle(omega)[1], "S", frequency_perturbed=True)
        self.assertEqual(self.response.family, "Y")
        self.assertEqual(self.response.magnitude.to("siemens").magnitude, abs(actual))
        self.assertEqual(self.response.real.to("siemens").magnitude, actual.real)
        self.assertEqual(self.response.imag.to("siemens").magnitude, actual.imag)

    def test_hybridized_pole_and_residue_coupling_match_the_common_operator(self) -> None:
        self.assertIsInstance(self.pole, DirectQuantityResult)
        capacitance, stiffness, _ = self._matrices(0.0)
        roots = two_mode_roots()
        pole = complex(self.pole.root.to("radian / second").magnitude)
        oracle = min(roots, key=lambda root: abs(pole - root))
        assert_forward(self, "hybridized_pole", pole, oracle, pole_condition(oracle), "rad/s")
        self.assertEqual(self.pole.frequency.to("hertz").magnitude, pole.real / (2.0 * math.pi))
        self.assertEqual(self.pole.linewidth.to("hertz").magnitude, -pole.imag / math.pi)

        self.assertIsInstance(self.coupling, DirectQuantityResult)
        omega_a = math.sqrt(stiffness[0, 0] / capacitance[0, 0])
        omega_b = math.sqrt(stiffness[1, 1] / capacitance[1, 1])
        slope_a = 2.0 * omega_a * capacitance[0, 0]
        slope_b = 2.0 * omega_b * capacitance[1, 1]
        omega = 2.0 * math.pi * 5.5e9
        expected = (stiffness[0, 1] - omega**2 * capacitance[0, 1]) / math.sqrt(slope_a * slope_b)
        actual = complex(self.coupling.coupling.to("radian / second").magnitude)
        assert_forward(self, "residue_coupling", actual, expected, coupling_oracle(omega)[1], "rad/s", frequency_perturbed=True)
        self.assertEqual(self.coupling.magnitude.to("radian / second").magnitude, abs(actual))
        for index, residue in enumerate((self.coupling.branch_a_residue, self.coupling.branch_b_residue)):
            expected_residue, condition = residue_oracle(index)
            assert_forward(self, f"branch_{index}_residue", complex(residue.to("ohm").magnitude),
                           expected_residue, condition, "ohm")

    def test_transfer_zero_is_the_analytic_matched_parallel_lc_zero(self) -> None:
        self.assertIsInstance(self.zero, DirectQuantityResult)
        actual_omega = complex(self.zero.zero.to("radian / second").magnitude)
        oracle_omega, condition = matched_zero_oracle()
        assert_forward(self, "matched_Spp_zero", actual_omega, oracle_omega, condition, "rad/s")
        self.assertEqual(self.zero.frequency.to("hertz").magnitude, actual_omega.real / (2.0 * math.pi))
        self.assertNotEqual(complex(self.zero.denominator.to("dimensionless").magnitude), 0.0j)
        self.assertNotEqual(complex(self.zero.numerator_slope.to("dimensionless").magnitude), 0.0j)


def _two_coordinate_plan():
    plan = CircuitPlan(id="complex_anchor_stabilization")
    nodes = {name: plan.bus(id=name) for name in ("a", "b")}
    for name, capacitance in (("a", 80.0), ("b", 90.0)):
        capacitor = plan.add(components.capacitor(id=f"c_{name}", capacitance=capacitance*u.fF))
        inductor = plan.add(components.inductor(id=f"l_{name}", inductance=8.0*u.nH))
        plan.parallel(id=f"shunt_{name}", start=nodes[name],
                         branches=((capacitor,), (inductor,)), end=plan.ground)
    coupling = plan.add(components.capacitor(id="coupling", capacitance=2.0*u.fF))
    plan.series(id="coupling", start=nodes["a"], elements=(coupling,), end=nodes["b"])
    coordinates = tuple(nodes[name].node for name in ("a", "b"))
    return plan, coordinates


def _one_port_transfer_zero():
    plan = CircuitPlan(id="complex_transfer_zero_stabilization")
    resistor = plan.add(components.resistor(id="resistor", resistance=50.0*u.ohm))
    capacitor = plan.add(components.capacitor(id="capacitor", capacitance=80.0*u.fF))
    inductor = plan.add(components.inductor(id="inductor", inductance=8.0*u.nH))
    signal = plan.bus(id="p")
    plan.parallel(id="parallel_rlc", start=signal,
                  branches=((resistor,), (capacitor,), (inductor,)), end=plan.ground)
    plan.add_port(id="p", at=signal, role="terminated", reference_impedance=50.0*u.ohm)
    return plan


class ComplexAnchorTests(unittest.TestCase):
    def test_complex_newton_anchor_encoding_preserves_authored_type_and_source_unit(self) -> None:
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            plan, coordinates = _two_coordinate_plan()
            run = CircuitRun( plan=plan, workspace=Path(workspace))
            owners.callback(run.close)
            view = run.original.reduce(ReductionPipeline().retain(*coordinates))
            records = []
            for anchor in (5.0 * u.GHz, (5.0 + 0.0j) * u.GHz, (5.0 - 0.1j) * u.GHz):
                spec = HybridizedPoleSpec(coordinates=coordinates, anchor=anchor)
                explanation = run.explain(view, spec)
                request = explanation.evidence
                records.append(request)
                encoded = request["spec"]["anchor"]
                self.assertEqual(
                    encoded["type"],
                    "quantity_f64" if not isinstance(anchor.magnitude, complex) else "complex_quantity_f64",
                )
                self.assertEqual(str(spec.anchor.units), "gigahertz")
            self.assertEqual(records[1]["spec"]["anchor"]["imag_si_f64"], "0000000000000000")
            self.assertNotEqual(records[2]["spec"]["anchor"]["imag_si_f64"], "0000000000000000")
            self.assertEqual(len({record["request_sha256"] for record in records}), 3)

            zero = TransferZeroSpec(
                anchor=(5.0 + 0.0j) * u.GHz,
                family="Y",
                input_coordinate=coordinates[0],
                output_coordinate=coordinates[1],
            )
            request = run.explain(view, zero).evidence
            self.assertEqual(request["spec"]["anchor"]["type"], "complex_quantity_f64")
            self.assertEqual(str(zero.anchor.units), "gigahertz")


    def test_nonzero_imaginary_complex_anchors_execute_and_resolve(self) -> None:
        with TemporaryDirectory() as workspace, ExitStack() as owners:
            plan, coordinates = _two_coordinate_plan()
            run = CircuitRun( plan=plan, workspace=Path(workspace))
            owners.callback(run.close)
            view = run.original.reduce(ReductionPipeline().retain(*coordinates))
            spec = HybridizedPoleSpec(coordinates=coordinates, anchor=(6.0 - 1.0e-12j) * u.GHz)
            request = run.explain(view, spec).evidence
            encoded = request["spec"]["anchor"]
            result = run.evaluate(view, spec)
            self.assertTrue(np.isfinite(result.frequency.to("hertz").magnitude))
            self.assertTrue(np.isfinite(result.linewidth.to("hertz").magnitude))
            self.assertEqual(run.resolve(view, spec).identity, result.identity)

        with TemporaryDirectory() as workspace, ExitStack() as owners:
            zero_run = CircuitRun( plan=_one_port_transfer_zero(), workspace=Path(workspace))
            owners.callback(zero_run.close)
            zero_spec = TransferZeroSpec(
                anchor=(6.3 - 0.05j) * u.GHz,
                family="S",
                input_coordinate="p",
                output_coordinate="p",
            )
            zero = zero_run.evaluate(zero_run.original, zero_spec)
            self.assertTrue(np.isfinite(zero.frequency.to("hertz").magnitude))
            self.assertNotEqual(complex(zero.denominator.to("dimensionless").magnitude), 0.0j)
            self.assertEqual(zero_run.resolve(zero_run.original, zero_spec).identity, zero.identity)
