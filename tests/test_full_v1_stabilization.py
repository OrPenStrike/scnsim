"""Accepted cross-slice identity, workspace, replay, and resolve boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
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
    OptimizationSpec,
    OptimizationVariable,
    ReductionPipeline,
    WorkspaceVersioningDowngradeForbidden,
    components,
    units as u,
)


def _primitive_plan(*, plan_id: str = "primitive_full_v1", capacitance: float = 110.0):
    plan = CircuitPlan(id=plan_id)
    coupling = plan.add(components.capacitor(id="coupling_cap", capacitance=6.0 * u.fF))
    capacitor = plan.add(components.capacitor(id="resonator_cap", capacitance=capacitance * u.fF))
    inductor = plan.add(components.inductor(id="resonator_ind", inductance=5.8 * u.nH))
    boundary = plan.net(coupling.pin("terminal_1"))
    resonator = plan.net(
        coupling.pin("terminal_2"),
        capacitor.pin("terminal_1"),
        inductor.pin("terminal_1"),
        id="resonator_node",
    )
    plan.ground(capacitor.pin("terminal_2"), inductor.pin("terminal_2"))
    plan.add_port(id="signal_in", at=boundary, role="terminated", reference_impedance=50.0 * u.ohm)
    return plan, resonator, capacitor


def _primitive_requests(run: CircuitRun, resonator: object, capacitor: object):
    direct = DirectSolveSpec(frequencies=[5.5, 6.0, 6.5] * u.GHz)
    root = DiagonalRootSpec(coordinate=resonator, root_hint=6.0 * u.GHz)
    variable = OptimizationVariable(
        parameter=capacitor.parameter("capacitance"),
        bounds=(80.0 * u.fF, 140.0 * u.fF),
    )
    optimization = OptimizationSpec(
        variables=(variable,),
        objectives=(
            CostObjective(
                id="resonance_frequency",
                quantity=root.frequency,
                target=6.2 * u.GHz,
                weight=1.0 * u.dimensionless,
            ),
        ),
        optimizer=CMAESSpec(seed=17, max_evaluations=5),
    )
    quantity_view = run.original.reduce(ReductionPipeline().retain(resonator))
    return direct, root, optimization, quantity_view


class FullV1WorkspaceTests(unittest.TestCase):
    def test_versioning_upgrade_preserves_leaf_and_forbids_downgrade(self) -> None:
        with TemporaryDirectory() as workspace:
            first_plan, _, _ = _primitive_plan(plan_id="first")
            first = CircuitRun(plan=first_plan, workspace=Path(workspace))
            first_instance = first._binding.workspace_instance_id
            upgraded_plan, _, _ = _primitive_plan(plan_id="first")
            upgraded = CircuitRun(plan=upgraded_plan, workspace=Path(workspace), versioned=True)
            self.assertEqual(upgraded._binding.workspace_instance_id, first_instance)
            self.assertEqual(first.inventory().requests, ())
            self.assertEqual(upgraded._binding.leaf, first._binding.leaf)

            second_plan, _, _ = _primitive_plan(plan_id="second", capacitance=120.0)
            second = CircuitRun(plan=second_plan, workspace=Path(workspace), versioned=True)
            self.assertNotEqual(second._binding.workspace_instance_id, first_instance)
            self.assertEqual(first.inventory().requests, ())
            self.assertEqual(second.inventory().requests, ())
            with self.assertRaises(WorkspaceVersioningDowngradeForbidden):
                plan, _, _ = _primitive_plan(plan_id="second", capacitance=120.0)
                CircuitRun(plan=plan, workspace=Path(workspace), versioned=False)


@unittest.skipUnless(
    os.environ.get("SCNSIM_RUN_JULIA_TESTS") == "1",
    "set SCNSIM_RUN_JULIA_TESTS=1 to run packaged Julia cross-slice regressions",
)
class FullV1ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = TemporaryDirectory()
        cls.workspace = Path(cls._temporary.name)
        cls.plan, cls.resonator, cls.capacitor = _primitive_plan()
        cls.circuit_run = CircuitRun(plan=cls.plan, workspace=cls.workspace)
        cls.direct_spec, cls.root_spec, cls.optimization_spec, cls.quantity_view = _primitive_requests(
            cls.circuit_run, cls.resonator, cls.capacitor
        )
        cls.direct = cls.circuit_run.solve(cls.circuit_run.original, cls.direct_spec)
        cls.root = cls.circuit_run.evaluate(cls.quantity_view, cls.root_spec)
        cls.optimization = cls.circuit_run.optimize(cls.quantity_view, cls.optimization_spec)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_direct_root_and_cma_share_one_plan_without_identity_aliasing(self) -> None:
        identities = (self.direct.identity, self.root.identity, self.optimization.identity)
        self.assertEqual(len({identity.plan_sha256 for identity in identities}), 1)
        self.assertEqual(len({identity.request_sha256 for identity in identities}), 3)
        self.assertEqual(self.direct.s.view.matrix.shape, (3, 1, 1))
        self.assertTrue(np.isfinite(self.root.frequency.to("hertz").magnitude))
        self.assertTrue(np.isfinite(self.root.linewidth.to("hertz").magnitude))
        self.assertTrue(self.optimization.ledger)
        self.assertEqual(self.optimization.best.parameters.allow_extrapolation, ())
        best = self.optimization.best.parameters.values[self.capacitor.parameter("capacitance")].to("fF").magnitude
        self.assertGreaterEqual(best, 80.0)
        self.assertLessEqual(best, 140.0)

    def test_success_reuse_inventory_and_fresh_process_resolve_are_exact(self) -> None:
        self.assertEqual(
            self.circuit_run.solve(self.circuit_run.original, self.direct_spec).identity,
            self.direct.identity,
        )
        self.assertEqual(
            self.circuit_run.evaluate(self.quantity_view, self.root_spec).identity,
            self.root.identity,
        )
        self.assertEqual(
            self.circuit_run.optimize(self.quantity_view, self.optimization_spec).identity,
            self.optimization.identity,
        )
        inventory = self.circuit_run.inventory().requests
        self.assertEqual(len(inventory), 3)
        self.assertTrue(all(row["status"] == "succeeded" for row in inventory))
        self.assertTrue(all(row["attempts"] == ("000001",) for row in inventory))

        script = """
import json
from pathlib import Path
from tests.test_full_v1_stabilization import _primitive_plan, _primitive_requests
from scnsim import CircuitRun
plan, node, capacitor = _primitive_plan()
run = CircuitRun(plan=plan, workspace=Path(__import__('sys').argv[1]))
direct, root, optimization, view = _primitive_requests(run, node, capacitor)
results = (
    run.resolve(run.original, direct),
    run.resolve(view, root),
    run.resolve(view, optimization),
)
print(json.dumps([result.identity.result_sha256 for result in results]))
"""
        output = subprocess.check_output(
            [sys.executable, "-c", script, str(self.workspace)],
            cwd=Path(__file__).parents[1],
            text=True,
        )
        self.assertEqual(
            json.loads(output),
            [self.direct.identity.result_sha256, self.root.identity.result_sha256, self.optimization.identity.result_sha256],
        )


if __name__ == "__main__":
    unittest.main()
