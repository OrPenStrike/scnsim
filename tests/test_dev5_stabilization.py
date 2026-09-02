"""Accepted dev5 RLGC, selected-network, and Direct regression boundary."""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scnsim import (
    AffineMap,
    CMAESSpec,
    CircuitPlan,
    CircuitRun,
    CompositePlan,
    CostObjective,
    DiagonalRootSpec,
    DirectQuantityResult,
    DirectSolveSpec,
    HybridizedPoleSpec,
    InvalidCandidatePhysicalParameter,
    InvalidOptimizationSpec,
    Library,
    OperatorResult,
    OperatorSpec,
    OptimizationSpec,
    OptimizationVariable,
    ParameterSet,
    ParameterSpec,
    RLGC,
    ReductionPipeline,
    ResidueNormalizedCouplingSpec,
    ResponseElementSpec,
    SCNSimValidationError,
    TransferZeroSpec,
    components,
    load_q2d_rlgc,
    units as u,
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


def _q2d_bytes() -> bytes:
    blocks = (
        ("Capacitance Matrix", ((175.0, -22.0), (-22.0, 168.0))),
        ("Conductance Matrix", ((0.0, 0.0), (0.0, 0.0))),
        ("Inductance Matrix", ((420.0, 75.0), (75.0, 395.0))),
        ("Resistance Matrix", ((0.18, 0.0), (0.0, 0.22))),
    )
    lines = [
        "Setup1:LastAdaptive",
        "Problem Type:  CG, RL",
        "C Units:pF/meter, G Units:mho/meter, R Units:ohm/meter, L Units:nH/meter",
        "Reduce Matrix:  Original",
        "Frequency:  5.0GHz",
        "",
    ]
    for title, matrix in blocks:
        lines.extend(
            (
                title,
                ",trace_a,trace_b",
                f"trace_a,{matrix[0][0]},{matrix[0][1]}",
                f"trace_b,{matrix[1][0]},{matrix[1][1]}",
                "",
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _floating_probe_plan() -> tuple[CircuitPlan, object, object, object, object]:
    plan = CircuitPlan(id="floating_probe_stabilization")
    feed = plan.add(components.capacitor(id="feed", capacitance=35.0 * u.fF))
    coupler = plan.add(components.capacitor(id="coupler", capacitance=4.0 * u.fF))
    qubit = plan.add(
        components.floating_parallel_linear_lc_resonator(
            id="qubit",
            terminal_1_to_reference_capacitance=45.0 * u.fF,
            terminal_2_to_reference_capacitance=42.0 * u.fF,
            terminal_mutual_capacitance=16.0 * u.fF,
            inductance=7.0 * u.nH,
        )
    )
    input_node = plan.net(feed.pin("terminal_1"))
    output_node = plan.net(feed.pin("terminal_2"), coupler.pin("terminal_1"))
    plus = plan.net(coupler.pin("terminal_2"), qubit.pin("terminal_1"), id="qubit_plus")
    minus = plan.net(qubit.pin("terminal_2"), id="qubit_minus")
    feed_in = plan.add_port(id="feedline_in", at=input_node, role="terminated", reference_impedance=50.0 * u.ohm)
    plan.add_port(id="feedline_out", at=output_node, role="terminated", reference_impedance=50.0 * u.ohm)
    probe_plus = plan.add_port(id="probe_plus", at=plus, role="nonloading_probe", reference_impedance=50.0 * u.ohm)
    probe_minus = plan.add_port(id="probe_minus", at=minus, role="nonloading_probe", reference_impedance=50.0 * u.ohm)
    return plan, feed_in, probe_plus, probe_minus, (plus, minus)


class _AffineCatalog(Library):
    """A two-consumer calibration fan-out with one accepted public input."""

    def mapped_resonator(self, *, id: str):
        composite = CompositePlan(id=id, library=self)
        length = composite.parameter(
            id="length",
            baseline=1.0 * u.meter,
            spec=ParameterSpec(unit=u.meter),
        )
        first = composite.add(
            components.capacitor(
                id="first",
                capacitance=AffineMap(
                    input=length,
                    slope=20.0 * u.fF / u.meter,
                    intercept=10.0 * u.fF,
                    support=(0.99 * u.meter, 1.01 * u.meter),
                ),
            )
        )
        second = composite.add(
            components.capacitor(
                id="second",
                capacitance=AffineMap(
                    input=length,
                    slope=30.0 * u.fF / u.meter,
                    intercept=5.0 * u.fF,
                    support=(0.99 * u.meter, 1.01 * u.meter),
                ),
            )
        )
        inductor = composite.add(components.inductor(id="inductor", inductance=8.0 * u.nH))
        terminal = composite.net(
            first.pin("terminal_1"),
            second.pin("terminal_1"),
            inductor.pin("terminal_1"),
            id="terminal",
        )
        composite.ground(
            first.pin("terminal_2"),
            second.pin("terminal_2"),
            inductor.pin("terminal_2"),
        )
        composite.expose_pin(id="terminal", at=terminal)
        composite.expose_coordinate(id="mode", at=terminal)
        return composite.build()


def _affine_plan() -> tuple[CircuitPlan, object]:
    plan = CircuitPlan(id="affine_extrapolation_stabilization")
    resonator = plan.add(_AffineCatalog().mapped_resonator(id="mapped"))
    signal = plan.net(resonator.pin("terminal"), id="signal")
    plan.add_port(id="p", at=signal, role="terminated", reference_impedance=50.0 * u.ohm)
    return plan, resonator.parameter("length")


def _advanced_direct_plan() -> tuple[CircuitPlan, object]:
    """Two lossless retained modes with an analytic parallel-LC coupling."""

    plan = CircuitPlan(id="advanced_direct_stabilization")
    capacitor_a = plan.add(components.capacitor(id="capacitor_a", capacitance=80.0 * u.fF))
    inductor_a = plan.add(components.inductor(id="inductor_a", inductance=8.0 * u.nH))
    capacitor_b = plan.add(components.capacitor(id="capacitor_b", capacitance=95.0 * u.fF))
    inductor_b = plan.add(components.inductor(id="inductor_b", inductance=7.0 * u.nH))
    coupling_capacitor = plan.add(components.capacitor(id="coupling_capacitor", capacitance=2.0 * u.fF))
    coupling_inductor = plan.add(components.inductor(id="coupling_inductor", inductance=500.0 * u.nH))
    plan.net(
        capacitor_a.pin("terminal_1"),
        inductor_a.pin("terminal_1"),
        coupling_capacitor.pin("terminal_1"),
        coupling_inductor.pin("terminal_1"),
        id="a",
    )
    plan.net(
        capacitor_b.pin("terminal_1"),
        inductor_b.pin("terminal_1"),
        coupling_capacitor.pin("terminal_2"),
        coupling_inductor.pin("terminal_2"),
        id="b",
    )
    plan.ground(
        capacitor_a.pin("terminal_2"),
        inductor_a.pin("terminal_2"),
        capacitor_b.pin("terminal_2"),
        inductor_b.pin("terminal_2"),
    )
    return plan, capacitor_a.parameter("capacitance")


def _matched_zero_plan() -> CircuitPlan:
    plan = CircuitPlan(id="transfer_zero_stabilization")
    resistor = plan.add(components.resistor(id="resistor", resistance=50.0 * u.ohm))
    capacitor = plan.add(components.capacitor(id="capacitor", capacitance=80.0 * u.fF))
    inductor = plan.add(components.inductor(id="inductor", inductance=8.0 * u.nH))
    signal = plan.net(
        resistor.pin("terminal_1"),
        capacitor.pin("terminal_1"),
        inductor.pin("terminal_1"),
        id="p",
    )
    plan.ground(resistor.pin("terminal_2"), capacitor.pin("terminal_2"), inductor.pin("terminal_2"))
    plan.add_port(id="p", at=signal, role="terminated", reference_impedance=50.0 * u.ohm)
    return plan


def _tau(order: int) -> float:
    return 256.0 * (order + 1) * np.finfo(np.float64).eps


class Dev5AuthoringTests(unittest.TestCase):
    def test_manual_rlgc_is_ordered_immutable_and_defensively_copied(self) -> None:
        rlgc = _coupled_rlgc()
        self.assertEqual(rlgc.conductors, ("readout", "filter"))
        self.assertEqual(rlgc.reference_conductor, "ground")
        self.assertEqual(rlgc._canonical_record()["source"], {"source_kind": "manual"})
        self.assertEqual(rlgc.extraction_frequency.to("gigahertz").magnitude, 5.0)
        matrix = rlgc.capacitance_per_length
        self.assertFalse(np.asarray(matrix.magnitude).flags.writeable)
        with self.assertRaises(ValueError):
            matrix.magnitude[0, 0] = 0.0
        with self.assertRaises(AttributeError):
            rlgc.other = object()

    def test_rlgc_rejects_nonphysical_or_noncanonical_matrices(self) -> None:
        kwargs = dict(
            conductors=("a", "b"),
            reference_conductor="ground",
            resistance_per_length=[[1.0, 0.0], [0.0, 1.0]] * u.ohm / u.meter,
            inductance_per_length=[[2.0, 0.2], [0.2, 2.0]] * u.nH / u.meter,
            conductance_per_length=[[1.0, -0.2], [-0.2, 1.0]] * u.siemens / u.meter,
            capacitance_per_length=[[2.0, -0.2], [-0.2, 2.0]] * u.pF / u.meter,
        )
        for field, value in (
            ("resistance_per_length", [[1.0, 0.1], [0.0, 1.0]] * u.ohm / u.meter),
            ("inductance_per_length", [[1.0, 2.0], [2.0, 1.0]] * u.nH / u.meter),
            ("conductance_per_length", [[1.0, 0.1], [0.1, 1.0]] * u.siemens / u.meter),
            ("capacitance_per_length", [[1.0, -2.0], [-2.0, 1.0]] * u.pF / u.meter),
        ):
            with self.subTest(field=field):
                invalid = dict(kwargs)
                invalid[field] = value
                with self.assertRaises(SCNSimValidationError):
                    RLGC(**invalid)
        with self.assertRaises(SCNSimValidationError):
            RLGC(**{**kwargs, "conductors": ("a", "a")})

    def test_q2d_loader_is_strict_bijective_and_digest_bound(self) -> None:
        raw = _q2d_bytes()
        with TemporaryDirectory() as temporary:
            path = Path(temporary, "rlgc.csv")
            path.write_bytes(raw)
            rlgc = load_q2d_rlgc(
                path,
                reference_conductor="ground",
                conductor_map={"trace_a": "readout", "trace_b": "filter"},
            )
            source = rlgc._canonical_record()["source"]
            self.assertEqual(source["content_sha256"], sha256(raw).hexdigest())
            self.assertEqual(source["native_conductor_order"], ["trace_a", "trace_b"])
            self.assertEqual(rlgc.conductors, ("readout", "filter"))
            self.assertEqual(rlgc.capacitance_per_length.to("pF / meter").magnitude[0, 1], -22.0)

            with self.assertRaises(SCNSimValidationError):
                load_q2d_rlgc(path, reference_conductor="ground", conductor_map={"trace_a": "only"})
            path.write_bytes(raw.replace(b"Problem Type:  CG, RL", b"Problem Type: CG, RL"))
            with self.assertRaises(SCNSimValidationError):
                load_q2d_rlgc(path, reference_conductor="ground")
            path.write_bytes(b"\xff")
            with self.assertRaises(SCNSimValidationError):
                load_q2d_rlgc(path, reference_conductor="ground")

    def test_transmission_line_and_reduction_grammar_preserve_declaration_order(self) -> None:
        line = components.transmission_line(id="line", length=1.6 * u.mm, rlgc=_coupled_rlgc(), n_sections=8)
        snapshot = line._canonical_snapshot()
        self.assertEqual(snapshot["pin_order"], ["head.readout", "head.filter", "tail.readout", "tail.filter"])
        self.assertEqual(snapshot["realization"]["n_sections"], 8)
        self.assertEqual(snapshot["realization"]["pin_conductors"], ["readout", "filter"])
        with self.assertRaises(ValueError):
            components.transmission_line(id="bad", length=1.0 * u.mm, rlgc=_coupled_rlgc(), n_sections=0)

        plan, _, probe_plus, probe_minus, nodes = _floating_probe_plan()
        pipeline = ReductionPipeline().ptc(probe_plus, probe_minus).transform_pair(*nodes, id="qubit").retain("feedline_in", "feedline_out", "qubit.differential")
        self.assertIsNotNone(plan)
        with self.assertRaises(ValueError):
            pipeline.ptc(probe_plus)
        with self.assertRaises(ValueError):
            pipeline.transform_pair(*nodes, id="late")
        with self.assertRaises(ValueError):
            pipeline.retain("feedline_in")

    def test_parameter_and_optimization_overrides_replace_authorization(self) -> None:
        capacitor = components.capacitor(id="capacitor", capacitance=80.0 * u.fF)
        parameter = capacitor.parameter("capacitance")
        root = DiagonalRootSpec(coordinate="mode", root_hint=6.0 * u.GHz)
        variable = OptimizationVariable(parameter=parameter, bounds=(50.0 * u.fF, 120.0 * u.fF), transform="linear")
        spec = OptimizationSpec(
            variables=(variable,),
            objectives=(CostObjective(id="root", quantity=root.frequency, target=6.0 * u.GHz, weight=1.0 * u.dimensionless),),
            optimizer=CMAESSpec(seed=3, max_evaluations=8),
        )
        overridden = spec.with_variable_overrides(
            bounds={parameter: (60.0 * u.fF, 100.0 * u.fF)},
            allow_extrapolation=(parameter,),
        )
        self.assertEqual(overridden.variable(parameter).model_default_bounds, variable.model_default_bounds)
        self.assertEqual(overridden.variable(parameter).bounds, (60.0 * u.fF, 100.0 * u.fF))
        self.assertEqual(overridden.allow_extrapolation, (parameter,))
        replaced = overridden.with_variable_overrides(bounds={})
        self.assertEqual(replaced.allow_extrapolation, ())
        self.assertEqual(replaced.variable(parameter).bounds, overridden.variable(parameter).bounds)
        parameter_set = ParameterSet({parameter: 90.0 * u.fF}, allow_extrapolation=(parameter,))
        self.assertEqual(parameter_set.allow_extrapolation, (parameter,))
        with self.assertRaises(InvalidOptimizationSpec):
            spec.with_variable_overrides(bounds={components.capacitor(id="foreign", capacitance=1.0 * u.fF).parameter("capacitance"): (1.0 * u.fF, 2.0 * u.fF)})

    def test_complete_accepted_selector_catalog_encodes_in_one_cma_request(self) -> None:
        plan, parameter = _advanced_direct_plan()
        with TemporaryDirectory() as workspace:
            run = CircuitRun(plan=plan, workspace=Path(workspace))
            view = run.original.reduce(ReductionPipeline().retain("a", "b"))
            pole = HybridizedPoleSpec(coordinates=("a", "b"), anchor=6.2 * u.GHz)
            zero = TransferZeroSpec(
                anchor=5.0 * u.GHz,
                family="Y",
                input_coordinate="a",
                output_coordinate="b",
            )
            coupling = ResidueNormalizedCouplingSpec(
                branch_a=DiagonalRootSpec(coordinate="a", root_hint=6.2 * u.GHz),
                branch_b=DiagonalRootSpec(coordinate="b", root_hint=6.2 * u.GHz),
                frequency=5.5 * u.GHz,
            )
            response = ResponseElementSpec(
                family="Y",
                input_coordinate="a",
                output_coordinate="b",
                frequency=5.5 * u.GHz,
            )
            objectives = (
                CostObjective(id="pole_frequency", quantity=pole.frequency, target=6.2 * u.GHz, weight=1.0 * u.dimensionless),
                CostObjective(id="pole_linewidth", quantity=pole.linewidth, target=1.0 * u.MHz, weight=1.0 * u.dimensionless),
                CostObjective(id="zero_frequency", quantity=zero.frequency, target=5.0 * u.GHz, weight=1.0 * u.dimensionless),
                CostObjective(id="coupling", quantity=coupling.magnitude, target=50.0e6 * u.radian / u.second, weight=1.0 * u.dimensionless),
                CostObjective(id="response_magnitude", quantity=response.magnitude, target=10.0 * u.uS, weight=1.0 * u.dimensionless),
                CostObjective(id="response_real", quantity=response.real, target=0.0 * u.uS, scale=1.0 * u.uS, weight=1.0 * u.dimensionless),
                CostObjective(id="response_imag", quantity=response.imag, target=10.0 * u.uS, weight=1.0 * u.dimensionless),
            )
            spec = OptimizationSpec(
                variables=(OptimizationVariable(parameter=parameter, bounds=(79.5 * u.fF, 80.5 * u.fF)),),
                objectives=objectives,
                optimizer=CMAESSpec(seed=11, max_evaluations=5),
            )
            request, _ = run._request_declaration("optimize_direct", view, spec, None)
            encoded = [
                (objective["quantity"]["type"], objective["quantity"]["projection"])
                for objective in request["spec"]["objectives"]
            ]
            self.assertEqual(
                encoded,
                [
                    ("hybridized_pole_projection", "frequency"),
                    ("hybridized_pole_projection", "linewidth"),
                    ("transfer_zero_projection", "frequency"),
                    ("residue_coupling_projection", "magnitude"),
                    ("response_element_projection", "magnitude"),
                    ("response_element_projection", "real"),
                    ("response_element_projection", "imag"),
                ],
            )
            self.assertEqual(run.inventory().requests, ())


class Dev5RuntimeTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
        "set SCNSIM_RUN_JULIA_TESTS=1 to run packaged Julia Direct regressions",
    )
    def test_n_trace_pi_ladder_and_n_port_direct_are_not_scalarized(self) -> None:
        plan = CircuitPlan(id="n_trace_direct")
        line = plan.add(components.transmission_line(id="line", length=1.6 * u.mm, rlgc=_coupled_rlgc(), n_sections=2))
        for end in ("head", "tail"):
            for conductor in ("readout", "filter"):
                node = plan.net(line.pin(end, conductor=conductor), id=f"{end}_{conductor}")
                plan.add_port(id=f"{end}_{conductor}", at=node, role="terminated", reference_impedance=50.0 * u.ohm)
        spec = DirectSolveSpec(frequencies=[1.0, 2.0] * u.GHz)
        with TemporaryDirectory() as workspace:
            run = CircuitRun(plan=plan, workspace=Path(workspace))
            explanation = run.explain(run.original, spec)
            self.assertEqual(run.inventory().requests, ())
            compiled = explanation.evidence["compiled"]
            rows = compiled["expanded_branch_rows"]
            audits = [row for row in rows if row["kind"] == "transmission_line_audit"]
            self.assertEqual(len(audits), 1)
            self.assertEqual(audits[0]["conductors"], ("readout", "filter"))
            self.assertEqual(audits[0]["n_sections"], 2)
            self.assertEqual(len(audits[0]["stations"]), 6)
            self.assertTrue(any(row.get("row_conductor") != row.get("column_conductor") and not row.get("omitted_as_zero") for row in rows if row.get("kind") in {"series_inductance", "shunt_capacitance"}))
            result = run.solve(run.original, spec)
            self.assertEqual(result.s.view.matrix.shape, (2, 4, 4))
            self.assertEqual(result.y.view.matrix.shape, (2, 4, 4))
            self.assertEqual(result.z.view.matrix.shape, (2, 4, 4))
            self.assertEqual(tuple(result.s.view.coordinates), ("head_readout", "head_filter", "tail_readout", "tail_filter"))

    @unittest.skipUnless(
        os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
        "set SCNSIM_RUN_JULIA_TESTS=1 to run packaged Julia selected-network regressions",
    )
    def test_ptc_transform_retain_uses_one_realized_lineage(self) -> None:
        plan, _, probe_plus, probe_minus, nodes = _floating_probe_plan()
        with TemporaryDirectory() as workspace:
            run = CircuitRun(plan=plan, workspace=Path(workspace))
            view = run.original.reduce(
                ReductionPipeline()
                .ptc(probe_plus, probe_minus)
                .transform_pair(*nodes, id="qubit")
                .retain("feedline_in", "feedline_out", "qubit.differential")
            )
            spec = DirectSolveSpec(frequencies=[5.5] * u.GHz)
            explanation = run.explain(view, spec)
            lineage = explanation.evidence["ref_lineage"]
            self.assertEqual(lineage["terminal_coordinates"], ("feedline_in", "feedline_out", "qubit.differential"))
            self.assertEqual(lineage["ptc"]["selected_ports"], ("probe_plus", "probe_minus"))
            self.assertEqual([row["port_id"] for row in lineage["ptc"]["loads"]], ["probe_plus", "probe_minus"])
            self.assertTrue(lineage["port_realizable"])
            result = run.solve(view, spec)
            self.assertEqual(result.s.view.matrix.shape, (1, 3, 3))
            self.assertEqual(result.s.view.probe_loads["probe_plus"], "compensated")
            self.assertEqual(result.s.view.probe_loads["probe_minus"], "compensated")


@unittest.skipUnless(
    os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
    "set SCNSIM_RUN_JULIA_TESTS=1 to run packaged Julia extrapolation regressions",
)
class Dev5ExtrapolationRuntimeTests(unittest.TestCase):
    def test_ordinary_unauthorized_extrapolation_is_typed_and_creates_no_attempt(self) -> None:
        plan, parameter = _affine_plan()
        with TemporaryDirectory() as workspace:
            run = CircuitRun(plan=plan, workspace=Path(workspace))
            view = run.original.reduce(ReductionPipeline().retain("signal"))
            with self.assertRaises(InvalidCandidatePhysicalParameter) as raised:
                run.evaluate(
                    view,
                    OperatorSpec(frequencies=[5.0] * u.GHz),
                    parameters=ParameterSet({parameter: 1.05 * u.meter}),
                )
            self.assertEqual(raised.exception.stage, "affine_support")
            self.assertEqual(run.inventory().requests, ())

    def test_authorized_extrapolation_records_every_fanout_edge(self) -> None:
        plan, parameter = _affine_plan()
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            run = CircuitRun(plan=plan, workspace=root)
            view = run.original.reduce(ReductionPipeline().retain("signal"))
            result = run.evaluate(
                view,
                OperatorSpec(frequencies=[5.0] * u.GHz),
                parameters=ParameterSet(
                    {parameter: 1.05 * u.meter},
                    allow_extrapolation=(parameter,),
                ),
            )
            receipt_path = next(
                root.glob(
                    f"leaves/*/requests/{result.identity.request_sha256}/"
                    "attempts/000001/receipt.json"
                )
            )
            evidence = json.loads(receipt_path.read_text(encoding="utf-8"))["evidence"]["extrapolation_evidence"]
            self.assertEqual(len(evidence), 2)
            self.assertEqual(
                [row["consumer_target"]["component_path"] for row in evidence],
                [["mapped", "first"], ["mapped", "second"]],
            )
            self.assertTrue(all(row["side"] == "upper" for row in evidence))
            self.assertTrue(all(row["authorization_source"] == "parameter_set" for row in evidence))
            self.assertEqual(run.resolve(view, OperatorSpec(frequencies=[5.0] * u.GHz), parameters=ParameterSet({parameter: 1.05 * u.meter}, allow_extrapolation=(parameter,))).identity, result.identity)

    def test_unauthorized_cma_candidates_are_ineligible_and_winner_drops_authority(self) -> None:
        plan, parameter = _affine_plan()
        with TemporaryDirectory() as workspace:
            run = CircuitRun(plan=plan, workspace=Path(workspace))
            view = run.original.reduce(ReductionPipeline().retain("signal"))
            response = ResponseElementSpec(
                family="Y",
                input_coordinate="signal",
                output_coordinate="signal",
                frequency=5.0 * u.GHz,
            )
            spec = OptimizationSpec(
                variables=(
                    OptimizationVariable(
                        parameter=parameter,
                        bounds=(0.9 * u.meter, 1.1 * u.meter),
                    ),
                ),
                objectives=(
                    CostObjective(
                        id="response",
                        quantity=response.magnitude,
                        target=1.0 * u.mS,
                        weight=1.0 * u.dimensionless,
                    ),
                ),
                optimizer=CMAESSpec(seed=23, max_evaluations=5),
            )
            result = run.optimize(view, spec)
            candidates = [candidate for ledger in result.ledger for candidate in ledger["candidates"]]
            rejected = [candidate for candidate in candidates if candidate["outcome"]["status"] == "failure"]
            self.assertTrue(rejected)
            for candidate in rejected:
                self.assertEqual(candidate["outcome"]["penalty"], "positive_infinity")
                self.assertEqual(candidate["outcome"]["failure"]["kind"], "invalid_candidate_physical_parameter")
                self.assertEqual(candidate["outcome"]["failure"]["stage"], "affine_support")
                self.assertTrue(candidate["extrapolation_evidence"])
                self.assertTrue(all(row["authorization_source"] == "none" for row in candidate["extrapolation_evidence"]))
            self.assertEqual(result.best.parameters.allow_extrapolation, ())


@unittest.skipUnless(
    os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
    "set SCNSIM_RUN_JULIA_TESTS=1 to run packaged Julia Direct-quantity regressions",
)
class Dev5DirectQuantityRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = TemporaryDirectory()
        cls.workspace = Path(cls._temporary.name)
        cls.plan, cls.parameter = _advanced_direct_plan()
        cls.circuit_run = CircuitRun(plan=cls.plan, workspace=cls.workspace)
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
        cls.zero_plan = _matched_zero_plan()
        cls.zero_run = CircuitRun(plan=cls.zero_plan, workspace=Path(cls._zero_temporary.name))
        cls.zero_spec = TransferZeroSpec(
            anchor=6.3 * u.GHz,
            family="S",
            input_coordinate="p",
            output_coordinate="p",
        )
        cls.zero = cls.zero_run.evaluate(cls.zero_run.original, cls.zero_spec)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._zero_temporary.cleanup()
        cls._temporary.cleanup()

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
            scale = np.max(np.abs(expected))
            self.assertLessEqual(np.max(np.abs(actual - expected)) / scale, _tau(2))

        self.assertIsInstance(self.response, DirectQuantityResult)
        omega = 2.0 * math.pi * 5.5e9
        expected = 1j * (omega * 2.0e-15 - 1.0 / (omega * 500.0e-9))
        actual = complex(self.response.value.to("siemens").magnitude)
        self.assertLessEqual(abs(actual - expected) / abs(expected), _tau(2))
        self.assertEqual(self.response.family, "Y")
        self.assertEqual(self.response.magnitude.to("siemens").magnitude, abs(actual))
        self.assertEqual(self.response.real.to("siemens").magnitude, actual.real)
        self.assertEqual(self.response.imag.to("siemens").magnitude, actual.imag)

    def test_hybridized_pole_and_residue_coupling_match_the_common_operator(self) -> None:
        self.assertIsInstance(self.pole, DirectQuantityResult)
        capacitance, stiffness, _ = self._matrices(0.0)
        roots = np.sqrt(np.linalg.eigvals(np.linalg.solve(capacitance, stiffness)))
        pole = complex(self.pole.root.to("radian / second").magnitude)
        self.assertLessEqual(min(abs(pole - root) for root in roots) / abs(pole), _tau(2))
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
        self.assertLessEqual(abs(actual - expected) / abs(expected), _tau(2))
        self.assertEqual(self.coupling.magnitude.to("radian / second").magnitude, abs(actual))
        self.assertLessEqual(
            abs(complex(self.coupling.branch_a_residue.to("ohm").magnitude) + 1.0 / slope_a) / abs(1.0 / slope_a),
            _tau(2),
        )
        self.assertLessEqual(
            abs(complex(self.coupling.branch_b_residue.to("ohm").magnitude) + 1.0 / slope_b) / abs(1.0 / slope_b),
            _tau(2),
        )

    def test_transfer_zero_is_the_analytic_matched_parallel_lc_zero(self) -> None:
        self.assertIsInstance(self.zero, DirectQuantityResult)
        expected_frequency = 1.0 / (2.0 * math.pi * math.sqrt(8.0e-9 * 80.0e-15))
        actual_omega = complex(self.zero.zero.to("radian / second").magnitude)
        expected_omega = 2.0 * math.pi * expected_frequency
        self.assertLessEqual(abs(actual_omega - expected_omega) / expected_omega, _tau(1))
        self.assertEqual(self.zero.frequency.to("hertz").magnitude, actual_omega.real / (2.0 * math.pi))
        self.assertNotEqual(complex(self.zero.denominator.to("dimensionless").magnitude), 0.0j)
        self.assertNotEqual(complex(self.zero.numerator_slope.to("dimensionless").magnitude), 0.0j)


if __name__ == "__main__":
    unittest.main()
