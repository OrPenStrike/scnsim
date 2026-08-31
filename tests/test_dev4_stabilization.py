"""Accepted dev4 Lessons 6--8 Library and Composite regression boundary."""

from __future__ import annotations

import math
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from scnsim import (
    AffineMap,
    CircuitPlan,
    CircuitRun,
    CompositePlan,
    DirectSolveSpec,
    Library,
    ParameterSpec,
    PlanSealedError,
    ResultUnavailableError,
    SCNSimValidationError,
    components,
    units as u,
)


_factory_calls: list[str] = []


class _TestCatalog(Library):
    """Catalog defined in a real source file so its sealed provenance is stable."""

    def leaf(self, *, id: str, capacitance: object):
        composite = CompositePlan(id=id, library=self)
        value = composite.parameter(
            id="capacitance", baseline=capacitance, spec=ParameterSpec(unit=u.farad)
        )
        capacitor = composite.add(components.capacitor(id="capacitor", capacitance=value))
        terminal = composite.net(capacitor.pin("terminal_1"), id="terminal")
        composite.ground(capacitor.pin("terminal_2"))
        composite.expose_pin(id="terminal", at=terminal)
        return composite.build()

    def nested_lc(self, *, id: str, capacitance: object):
        _factory_calls.append(id)
        composite = CompositePlan(id=id, library=self)
        value = composite.parameter(
            id="capacitance", baseline=capacitance, spec=ParameterSpec(unit=u.farad)
        )
        length = composite.parameter(
            id="length", baseline=1.0 * u.meter, spec=ParameterSpec(unit=u.meter)
        )
        inductance = composite.parameter(
            id="inductance", baseline=1.0 * u.nH, spec=ParameterSpec(unit=u.henry)
        )
        capacitor_a = composite.add(components.capacitor(id="capacitor_a", capacitance=value))
        capacitor_b = composite.add(components.capacitor(id="capacitor_b", capacitance=value))
        affine_a = composite.add(
            components.capacitor(
                id="affine_a",
                capacitance=AffineMap(
                    input=length,
                    slope=2.0e-15 * u.farad / u.meter,
                    intercept=1.0e-15 * u.farad,
                    support=(0.5 * u.meter, 1.5 * u.meter),
                ),
            )
        )
        affine_b = composite.add(
            components.capacitor(
                id="affine_b",
                capacitance=AffineMap(
                    input=length,
                    slope=3.0e-15 * u.farad / u.meter,
                    intercept=2.0e-15 * u.farad,
                    support=(0.5 * u.meter, 1.5 * u.meter),
                ),
            )
        )
        inductor = composite.add(components.inductor(id="inductor", inductance=inductance))
        leaf = composite.add(self.leaf(id="leaf", capacitance=10.0 * u.fF))
        terminal = composite.net(
            capacitor_a.pin("terminal_1"),
            capacitor_b.pin("terminal_1"),
            affine_a.pin("terminal_1"),
            affine_b.pin("terminal_1"),
            inductor.pin("terminal_1"),
            leaf.pin("terminal"),
            id="terminal",
        )
        composite.ground(
            capacitor_a.pin("terminal_2"),
            capacitor_b.pin("terminal_2"),
            affine_a.pin("terminal_2"),
            affine_b.pin("terminal_2"),
            inductor.pin("terminal_2"),
        )
        composite.expose_pin(id="terminal", at=terminal)
        composite.expose_coordinate(id="mode", at=terminal)
        composite.expose_inductive_branch(id="inductor", branch=inductor.inductive_branch("self"))
        return composite.build()


class _DuplicateCatalog(Library):
    def duplicate(self, *, id: str):
        composite = CompositePlan(id=id, library=self)
        capacitor = composite.add(components.capacitor(id="capacitor", capacitance=1.0 * u.fF))
        terminal = composite.net(capacitor.pin("terminal_1"), id="terminal")
        composite.ground(capacitor.pin("terminal_2"))
        composite.expose_pin(id="first", at=terminal)
        try:
            composite.expose_pin(id="second", at=terminal)
        except SCNSimValidationError:
            pass
        else:
            raise AssertionError("duplicate public pin exposure was accepted")
        composite.expose_coordinate(id="first_coordinate", at=terminal)
        try:
            composite.expose_coordinate(id="second_coordinate", at=terminal)
        except SCNSimValidationError:
            pass
        else:
            raise AssertionError("duplicate public coordinate exposure was accepted")
        return composite.build()


class _ForeignCatalog(Library):
    def foreign(self, *, id: str):
        return components.capacitor(id=id, capacitance=1.0 * u.fF)


class Dev4LibraryCompositeTests(unittest.TestCase):
    def setUp(self) -> None:
        _factory_calls.clear()
        self.catalog = _TestCatalog()

    def _component(self):
        return self.catalog.nested_lc(id="resonator", capacitance=100.0 * u.fF)

    def test_factory_provenance_is_sealed_once_and_catalog_is_immutable(self) -> None:
        component = self._component()
        self.assertEqual(_factory_calls, ["resonator"])
        self.assertEqual(component.factory, "nested_lc")
        self.assertTrue(component.catalog_id.endswith(":_TestCatalog"))
        self.assertIn("source_sha256", component._catalog_source)
        with self.assertRaises(AttributeError):
            self.catalog.other = object()

        plan = CircuitPlan(id="factory_once")
        plan.add(component)
        signal = plan.net(component.pin("terminal"), id="signal")
        plan.add_port(
            id="port",
            at=signal,
            role="terminated",
            reference_impedance=50.0 * u.ohm,
        )
        with TemporaryDirectory() as workspace:
            CircuitRun(plan=plan, workspace=Path(workspace))
        self.assertTrue(plan.sealed)
        self.assertEqual(_factory_calls, ["resonator"])
        with self.assertRaises(PlanSealedError):
            plan.add(components.capacitor(id="after_seal", capacitance=1.0 * u.fF))

    def test_nested_snapshot_preserves_parameter_fanout_affine_and_public_maps(self) -> None:
        component = self._component()
        snapshot = component._canonical_snapshot()
        realization = snapshot["realization"]
        parameter_maps = realization["public_parameter_maps"]

        self.assertEqual(
            [entry["parameter"]["parameter_id"] for entry in parameter_maps],
            ["capacitance", "length", "inductance"],
        )
        consumers = parameter_maps[0]["consumers"]
        self.assertEqual(
            [consumer["target"]["component_path"] for consumer in consumers],
            [["resonator", "capacitor_a"], ["resonator", "capacitor_b"]],
        )
        self.assertEqual([consumer["binding"]["kind"] for consumer in consumers], ["identity", "identity"])
        self.assertEqual(
            [consumer["binding"]["kind"] for consumer in parameter_maps[1]["consumers"]],
            ["affine", "affine"],
        )
        self.assertEqual(realization["public_pin_map"][0]["public_id"], "terminal")
        self.assertEqual(realization["public_coordinate_map"][0]["public_id"], "resonator.mode")
        self.assertEqual(realization["public_inductive_branch_map"][0]["public_id"], "inductor")
        self.assertEqual(
            realization["children"][-1]["realization"]["children"][0]["component_path"],
            ["resonator", "leaf", "capacitor"],
        )
        with self.assertRaises(AttributeError):
            component.parameter("capacitance").baseline = 1.0 * u.fF
        realization["public_pin_map"][0]["public_id"] = "mutated_copy"
        self.assertEqual(
            component._canonical_snapshot()["realization"]["public_pin_map"][0]["public_id"],
            "terminal",
        )

    def test_duplicate_public_exposure_and_cross_catalog_mutation_fail_closed(self) -> None:
        # A factory is not an ambient callback: returning a built-in from another
        # catalog is rejected at the owning Library boundary.
        self.assertEqual(_DuplicateCatalog().duplicate(id="duplicate").id, "duplicate")
        with self.assertRaises(SCNSimValidationError):
            _ForeignCatalog().foreign(id="foreign")

    def test_composite_coordinate_resolves_to_its_outer_port_node(self) -> None:
        component = self._component()
        plan = CircuitPlan(id="coordinate_identity")
        plan.add(component)
        signal = plan.net(component.pin("terminal"))
        port = plan.add_port(
            id="readout",
            at=signal,
            role="terminated",
            reference_impedance=50.0 * u.ohm,
        )
        snapshot = plan._canonical_snapshot()
        coordinate = snapshot["components"][0]["realization"]["public_coordinate_map"][0]
        self.assertEqual(port.node.id, "readout")
        self.assertEqual(coordinate["public_id"], port.node.id)

    def test_custom_composite_uses_the_direct_request_path_without_reinvoking_factory(self) -> None:
        component = self._component()
        plan = CircuitPlan(id="direct_composite")
        plan.add(component)
        signal = plan.net(component.pin("terminal"), id="signal")
        plan.add_port(
            id="port",
            at=signal,
            role="terminated",
            reference_impedance=50.0 * u.ohm,
        )
        spec = DirectSolveSpec(frequencies=[5.0] * u.GHz)
        with TemporaryDirectory() as workspace:
            run = CircuitRun(plan=plan, workspace=Path(workspace))
            request, _ = run._request("solve_direct", run.original, spec, None)
            with self.assertRaises(ResultUnavailableError):
                run.resolve(run.original, spec)
        self.assertEqual(request["operation"], "solve_direct")
        self.assertEqual(_factory_calls, ["resonator"])


class Dev4ComponentExpansionTests(unittest.TestCase):
    def test_junction_idc_squid_and_resonator_snapshots_are_explicit(self) -> None:
        junction = components.josephson_junction(
            id="junction", josephson_inductance=8.0 * u.nH, junction_capacitance=2.0 * u.fF
        )
        capacitor = components.interdigitated_capacitor(
            id="idc",
            terminal_1_to_reference_capacitance=1.0 * u.fF,
            terminal_2_to_reference_capacitance=2.0 * u.fF,
            terminal_mutual_capacitance=3.0 * u.fF,
        )
        squid = components.symmetric_squid(
            id="squid",
            josephson_inductance=8.0 * u.nH,
            junction_capacitance=2.0 * u.fF,
            loop_inductance=1.0 * u.nH,
        )
        resonator = components.grounded_parallel_symmetric_squid_resonator(
            id="resonator",
            capacitance=100.0 * u.fF,
            josephson_inductance=8.0 * u.nH,
            junction_capacitance=2.0 * u.fF,
            loop_inductance=1.0 * u.nH,
        )

        self.assertEqual(junction._canonical_snapshot()["realization"]["kind"], "josephson_junction")
        self.assertEqual(capacitor._canonical_snapshot()["realization"]["kind"], "composite")
        self.assertEqual(
            [child["realization"]["kind"] for child in capacitor._canonical_snapshot()["realization"]["children"]],
            ["capacitor", "capacitor", "capacitor"],
        )
        self.assertEqual(squid._canonical_snapshot()["realization"]["kind"], "composite")
        self.assertEqual(
            [child["realization"]["kind"] for child in squid._canonical_snapshot()["realization"]["children"]],
            ["josephson_junction", "inductor", "josephson_junction"],
        )
        self.assertEqual(resonator.inductive_branch("loop").id, "loop")

    def test_mutual_coupling_requires_one_strictly_spd_inductance_graph(self) -> None:
        plan = CircuitPlan(id="mutual_spd")
        inductors = [
            plan.add(components.inductor(id=f"inductor_{index}", inductance=1.0 * u.nH))
            for index in range(3)
        ]
        plan.net(*(inductor.pin("terminal_1") for inductor in inductors), id="signal")
        plan.ground(*(inductor.pin("terminal_2") for inductor in inductors))
        plan.couple_inductive(
            id="first",
            inductor_a=inductors[0].inductive_branch("self"),
            inductor_b=inductors[1].inductive_branch("self"),
            coupling_coefficient=0.9 * u.dimensionless,
        )
        plan.couple_inductive(
            id="second",
            inductor_a=inductors[0].inductive_branch("self"),
            inductor_b=inductors[2].inductive_branch("self"),
            coupling_coefficient=0.9 * u.dimensionless,
        )
        plan.couple_inductive(
            id="third",
            inductor_a=inductors[1].inductive_branch("self"),
            inductor_b=inductors[2].inductive_branch("self"),
            coupling_coefficient=-0.9 * u.dimensionless,
        )
        with self.assertRaises(SCNSimValidationError):
            plan._canonical_snapshot()

    @unittest.skipUnless(
        os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
        "set SCNSIM_RUN_JULIA_TESTS=1 to run the packaged Julia compiler regression",
    )
    def test_valid_mutual_coupling_survives_recursive_julia_compilation(self) -> None:
        plan = CircuitPlan(id="mutual_compiler")
        left = plan.add(
            components.symmetric_squid(
                id="left_squid",
                josephson_inductance=8.0 * u.nH,
                junction_capacitance=2.0 * u.fF,
                loop_inductance=4.0 * u.nH,
            )
        )
        right = plan.add(
            components.symmetric_squid(
                id="right_squid",
                josephson_inductance=8.0 * u.nH,
                junction_capacitance=2.0 * u.fF,
                loop_inductance=9.0 * u.nH,
            )
        )
        signal = plan.net(left.pin("terminal_1"), right.pin("terminal_1"), id="signal")
        plan.ground(left.pin("terminal_2"), right.pin("terminal_2"))
        plan.add_port(
            id="port",
            at=signal,
            role="terminated",
            reference_impedance=50.0 * u.ohm,
        )
        plan.couple_inductive(
            id="mutual",
            inductor_a=left.inductive_branch("loop"),
            inductor_b=right.inductive_branch("loop"),
            coupling_coefficient=0.25 * u.dimensionless,
        )

        with TemporaryDirectory() as workspace:
            run = CircuitRun(plan=plan, workspace=Path(workspace))
            explanation = run.explain(
                run.original,
                DirectSolveSpec(frequencies=[5.0] * u.GHz),
            )
        compiled = explanation.evidence["compiled"]
        rows = compiled["expanded_branch_rows"]
        mutual = next(row for row in rows if row["kind"] == "mutual_inductance")
        self.assertEqual(mutual["coupling_id"], "mutual")
        self.assertEqual(mutual["branch_a"], {"component_path": ("left_squid",), "branch_id": "loop"})
        self.assertEqual(mutual["branch_b"], {"component_path": ("right_squid",), "branch_id": "loop"})
        self.assertEqual(mutual["coupling_coefficient"]["si_value_f64"], "3fd0000000000000")
        self.assertEqual(
            mutual["derived_mutual_inductance"]["si_value_f64"],
            "3e19c511dc3a41e1",
        )
        self.assertFalse(mutual["omitted_as_zero"])
        k_matrix = compiled["k_matrix"]
        self.assertEqual(k_matrix["shape"], (3, 3))
        values = [
            struct.unpack(">d", bytes.fromhex(value))[0]
            for value in k_matrix["row_major_f64"]
        ]
        self.assertTrue(values and all(math.isfinite(value) for value in values))
        self.assertNotEqual(values[0], 0.0)


if __name__ == "__main__":
    unittest.main()
