"""Retained Library provenance and native component-expansion regressions.

Structured authoring constructs the original physical fixtures here; these
cases do not freeze its new scope, alias or parameter-identity representation.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from tests.analytic_conditions import assert_forward, mutual_stiffness_oracle
from tempfile import TemporaryDirectory
import unittest

from scnsim import (
    AffineMap, CircuitPlan, CircuitRun, CompositePlan, DirectSolveSpec,
    Library, OperatorSpec, ParameterDefinitions, ParameterSpec, PlanSealedError,
    ReductionPipeline, ResultUnavailableError, SCNSimError, components, units as u,
)

_factory_calls: list[str] = []


class _TestCatalog(Library):
    """Real source-file catalog for the retained provenance regression."""

    def leaf(self, *, id: str, capacitance: object):
        composite = CompositePlan(id=id, library=self)
        inputs = ParameterDefinitions(id=f"{id}_inputs")
        value = inputs.parameter(id="capacitance", baseline=capacitance,
                                 spec=ParameterSpec(unit=u.farad))
        capacitor = composite.add(components.capacitor(id="capacitor", capacitance=value))
        terminal = composite.bus(id="terminal")
        composite.series(id="capacitor_branch", start=terminal,
                         elements=(capacitor,), end=composite.ground)
        composite.expose_parameter(id="capacitance", parameter=value)
        composite.expose_pin(id="terminal", at=terminal)
        return composite.build()

    def nested_lc(self, *, id: str, capacitance: object):
        _factory_calls.append(id)
        composite = CompositePlan(id=id, library=self)
        inputs = ParameterDefinitions(id=f"{id}_inputs")
        value = inputs.parameter(id="capacitance", baseline=capacitance,
                                 spec=ParameterSpec(unit=u.farad))
        length = inputs.parameter(id="length", baseline=1.0*u.meter,
                                  spec=ParameterSpec(unit=u.meter))
        inductance = inputs.parameter(id="inductance", baseline=1.0*u.nH,
                                      spec=ParameterSpec(unit=u.henry))
        capacitor_a = composite.add(components.capacitor(id="capacitor_a", capacitance=value))
        capacitor_b = composite.add(components.capacitor(id="capacitor_b", capacitance=value))
        affine_a = composite.add(components.capacitor(id="affine_a", capacitance=AffineMap(
            input=length, slope=2.0e-15*u.farad/u.meter, intercept=1.0e-15*u.farad,
            support=(0.5*u.meter, 1.5*u.meter))))
        affine_b = composite.add(components.capacitor(id="affine_b", capacitance=AffineMap(
            input=length, slope=3.0e-15*u.farad/u.meter, intercept=2.0e-15*u.farad,
            support=(0.5*u.meter, 1.5*u.meter))))
        inductor = composite.add(components.inductor(id="inductor", inductance=inductance))
        leaf = composite.add(self.leaf(id="leaf", capacitance=10.0*u.fF))
        terminal = composite.bus(id="terminal")
        composite.parallel(id="parallel_lc", start=terminal,
                           branches=((capacitor_a,), (capacitor_b,), (affine_a,),
                                     (affine_b,), (inductor,)), end=composite.ground)
        composite.link(id="nested_leaf", endpoints=(terminal, leaf.pin("terminal")))
        for name, parameter in (("capacitance", value), ("length", length),
                                ("inductance", inductance)):
            composite.expose_parameter(id=name, parameter=parameter)
        composite.expose_pin(id="terminal", at=terminal)
        composite.expose_coordinate(id="mode", at=terminal)
        composite.expose_inductive_branch(id="inductor", branch=inductor.inductive_branch("self"))
        return composite.build()


class LibraryProvenanceTests(unittest.TestCase):
    def setUp(self):
        _factory_calls.clear()
        self.catalog = _TestCatalog()

    def _plan(self, component, *, id):
        plan = CircuitPlan(id=id)
        plan.add(component)
        signal = plan.bus(id="signal")
        plan.link(id="resonator_boundary", endpoints=(signal, component.pin("terminal")))
        plan.add_port(id="port", at=signal, role="terminated", reference_impedance=50.0*u.ohm)
        return plan

    def test_factory_provenance_is_sealed_once_and_catalog_is_immutable(self):
        component = self.catalog.nested_lc(id="resonator", capacitance=100.0*u.fF)
        self.assertEqual(_factory_calls, ["resonator"])
        self.assertEqual(component.factory, "nested_lc")
        self.assertTrue(component.catalog_id.endswith(":_TestCatalog"))
        self.assertIn("source_sha256", component.catalog_source)
        with self.assertRaises(AttributeError):
            self.catalog.other = object()
        plan = self._plan(component, id="factory_once")
        with TemporaryDirectory() as workspace:
            run = CircuitRun(plan=plan, workspace=Path(workspace))
            try:
                self.assertTrue(plan.sealed)
                self.assertEqual(_factory_calls, ["resonator"])
                with self.assertRaises(PlanSealedError):
                    plan.add(components.capacitor(id="after_seal", capacitance=1.0*u.fF))
            finally:
                run.close()

    def test_custom_composite_uses_direct_resolve_without_reinvoking_factory(self):
        component = self.catalog.nested_lc(id="resonator", capacitance=100.0*u.fF)
        plan = self._plan(component, id="direct_composite")
        spec = DirectSolveSpec(frequencies=[5.0]*u.GHz)
        with TemporaryDirectory() as workspace:
            run = CircuitRun(plan=plan, workspace=Path(workspace))
            try:
                with self.assertRaises(ResultUnavailableError):
                    run.resolve(run.original, spec)
            finally:
                run.close()
        self.assertEqual(_factory_calls, ["resonator"])


class ComponentExpansionTests(unittest.TestCase):
    def test_junction_idc_squid_and_resonator_expansions_are_explicit(self):
        junction = components.josephson_junction(
            id="junction", josephson_inductance=8.0*u.nH, junction_capacitance=2.0*u.fF)
        capacitor = components.interdigitated_capacitor(
            id="idc", terminal_1_to_reference_capacitance=1.0*u.fF,
            terminal_2_to_reference_capacitance=2.0*u.fF, terminal_mutual_capacitance=3.0*u.fF)
        squid = components.symmetric_squid(id="squid", josephson_inductance=8.0*u.nH,
            junction_capacitance=2.0*u.fF, loop_inductance=1.0*u.nH)
        resonator = components.grounded_parallel_symmetric_squid_resonator(
            id="resonator", capacitance=100.0*u.fF, josephson_inductance=8.0*u.nH,
            junction_capacitance=2.0*u.fF, loop_inductance=1.0*u.nH)
        expected = ((junction, ["josephson_junction"]),
                    (capacitor, ["capacitor", "capacitor", "capacitor"]),
                    (squid, ["josephson_junction", "inductor", "josephson_junction"]))
        for component, models in expected:
            with self.subTest(component=component.id), TemporaryDirectory() as workspace:
                plan = CircuitPlan(id=f"expansion_{component.id}")
                plan.add(component)
                signal, return_bus = plan.bus(id="signal"), plan.bus(id="return")
                use = component if component is junction else component.between(
                    component.pin("terminal_1"), component.pin("terminal_2"))
                plan.series(id="body", start=signal, elements=(use,), end=return_bus)
                run = CircuitRun(plan=plan, workspace=Path(workspace), precision="float64")
                try:
                    view = run.original.reduce(ReductionPipeline().retain("signal", "return"))
                    explanation = run.explain(view, OperatorSpec(frequencies=[5.0]*u.GHz))
                    self.assertCountEqual([row["model"] for row in explanation.evidence["physical_leaves"]], models)
                finally:
                    run.close()
        self.assertEqual(resonator.inductive_branch("loop").id, "loop")

    def test_mutual_coupling_requires_one_strictly_spd_inductance_graph(self):
        plan = CircuitPlan(id="mutual_spd")
        inductors = [plan.add(components.inductor(id=f"inductor_{index}", inductance=1.0*u.nH))
                     for index in range(3)]
        signal = plan.bus(id="signal")
        plan.parallel(id="inductor_graph", start=signal,
                      branches=tuple((inductor,) for inductor in inductors), end=plan.ground)
        for name, a, b, coefficient in (("first", 0, 1, 0.9),
                                       ("second", 0, 2, 0.9),
                                       ("third", 1, 2, -0.9)):
            plan.couple_inductive(id=name, inductor_a=inductors[a].inductive_branch("self"),
                inductor_b=inductors[b].inductive_branch("self"),
                coupling_coefficient=coefficient*u.dimensionless)
        with TemporaryDirectory() as workspace:
            run = CircuitRun(plan=plan, workspace=Path(workspace), precision="float64")
            try:
                view = run.original.reduce(ReductionPipeline().retain("signal"))
                # The public compiler owns complete-graph physical rejection;
                # the retired private-snapshot timing/subclass is not promised.
                with self.assertRaises(SCNSimError):
                    run.explain(view, OperatorSpec(frequencies=[5.0]*u.GHz))
            finally:
                run.close()

    def test_valid_mutual_coupling_matches_the_independent_stiffness_stamp(self):
        plan = CircuitPlan(id="mutual_compiler")
        left = plan.add(components.symmetric_squid(id="left_squid",
            josephson_inductance=8.0*u.nH, junction_capacitance=2.0*u.fF, loop_inductance=4.0*u.nH))
        right = plan.add(components.symmetric_squid(id="right_squid",
            josephson_inductance=8.0*u.nH, junction_capacitance=2.0*u.fF, loop_inductance=9.0*u.nH))
        signal = plan.bus(id="signal")
        plan.parallel(id="parallel_squids", start=signal,
            branches=((left.between(left.pin("terminal_1"), left.pin("terminal_2")),),
                      (right.between(right.pin("terminal_1"), right.pin("terminal_2")),)), end=plan.ground)
        plan.add_port(id="port", at=signal, role="terminated", reference_impedance=50.0*u.ohm)
        plan.couple_inductive(id="mutual", inductor_a=left.inductive_branch("loop"),
            inductor_b=right.inductive_branch("loop"), coupling_coefficient=0.25*u.dimensionless)
        with TemporaryDirectory() as workspace:
            run = CircuitRun(plan=plan, workspace=Path(workspace), precision="float64")
            try:
                evidence = run.explain(run.original, DirectSolveSpec(frequencies=[5.0]*u.GHz)).evidence
            finally:
                run.close()
        coupling = next(row for row in evidence["connectivity"]["couplings"] if row["id"] == "mutual")
        self.assertEqual(coupling["inductor_a"], {"path": ("left_squid", "loop"), "branch_id": "self"})
        self.assertEqual(coupling["inductor_b"], {"path": ("right_squid", "loop"), "branch_id": "self"})
        self.assertEqual(coupling["coupling_coefficient"]["si_value_f64"], "3fd0000000000000")
        compiled = evidence["compiled"]
        order = tuple(compiled["node_order"])
        endpoints = {(tuple(row["path"]), row["pin"]): row["net"]
                     for row in evidence["connectivity"]["physical_endpoints"]}
        ground = evidence["connectivity"]["canonical_ground"]

        def incidence(path):
            vector = np.zeros(len(order))
            for pin, sign in (("terminal_1", 1.0), ("terminal_2", -1.0)):
                net = endpoints[path, pin]
                if net != ground:
                    vector[order.index(net)] += sign
            return vector

        junctions = [incidence((side, junction))
                     for side in ("left_squid", "right_squid")
                     for junction in ("junction_1", "junction_2")]
        loops = np.column_stack([incidence((side, "loop")) for side in ("left_squid", "right_squid")])
        expected, condition = mutual_stiffness_oracle(junctions, loops)
        matrix = compiled["k_matrix"]
        self.assertEqual(tuple(matrix["shape"]), (3, 3))
        encoded = matrix["values"]
        values = np.frombuffer(bytes.fromhex(encoded["data_hex"]), dtype=np.dtype(encoded["dtype"]).newbyteorder("<"))
        actual = np.zeros(tuple(matrix["shape"]))
        np.add.at(actual, (matrix["rows"], matrix["cols"]), values)
        self.assertTrue(np.isfinite(actual).all())
        self.assertNotEqual(actual.flat[0], 0.0)
        assert_forward(self, "mutual_inductance_stiffness", actual, expected, condition, "1/H")
