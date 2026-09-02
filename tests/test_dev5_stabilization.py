"""Accepted dev5 RLGC, selected-network, and Direct regression boundary."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scnsim import (
    CMAESSpec,
    CircuitPlan,
    CircuitRun,
    CostObjective,
    DiagonalRootSpec,
    DirectSolveSpec,
    InvalidOptimizationSpec,
    OptimizationSpec,
    OptimizationVariable,
    ParameterSet,
    RLGC,
    ReductionPipeline,
    SCNSimValidationError,
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


class Dev5AuthoringTests(unittest.TestCase):
    def test_manual_rlgc_is_ordered_immutable_and_defensively_copied(self) -> None:
        rlgc = _coupled_rlgc()
        self.assertEqual(rlgc.conductors, ("readout", "filter"))
        self.assertEqual(rlgc.reference_conductor, "ground")
        self.assertEqual(rlgc._canonical_record()["source"], {"source_kind": "manual"})
        self.assertAlmostEqual(rlgc.extraction_frequency.to("gigahertz").magnitude, 5.0)
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


if __name__ == "__main__":
    unittest.main()
