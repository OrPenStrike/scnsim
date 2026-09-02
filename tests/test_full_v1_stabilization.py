"""Accepted cross-slice identity, workspace, replay, and resolve boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import scnsim.runtime as runtime_module

from scnsim import (
    BackendProtocolError,
    CMAESSpec,
    CircuitPlan,
    CircuitRun,
    CostObjective,
    DiagonalRootSpec,
    DirectSolveSpec,
    EvidenceIntegrityError,
    OptimizationSpec,
    OptimizationVariable,
    ReductionPipeline,
    ResultUnavailableError,
    WorkspaceVersioningDowngradeForbidden,
    components,
    units as u,
)
from scnsim._canonical import canonical_json_bytes, sha256_hex


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

    def test_failed_and_interrupted_attempts_are_visible_and_retry_appends_success(self) -> None:
        with TemporaryDirectory() as workspace:
            plan, _, _ = _primitive_plan(plan_id="attempt_history")
            run = CircuitRun(plan=plan, workspace=Path(workspace))
            spec = DirectSolveSpec(frequencies=[5.5] * u.GHz)

            with patch.object(
                runtime_module,
                "run_terminal",
                side_effect=BackendProtocolError("deterministic launch failure", stage="process_start"),
            ):
                with self.assertRaises(BackendProtocolError):
                    run.solve(run.original, spec)
            failed = run.inventory().requests
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["status"], "failed")
            self.assertEqual(failed[0]["attempts"], ("000001",))
            with self.assertRaises(ResultUnavailableError):
                run.resolve(run.original, spec)
            self.assertEqual(run.inventory().requests[0]["attempts"], ("000001",))

            with patch.object(runtime_module, "run_terminal", side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):
                    run.solve(run.original, spec)
            interrupted = run.inventory().requests[0]
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["attempts"], ("000001", "000002"))
            with self.assertRaises(ResultUnavailableError):
                run.resolve(run.original, spec)
            self.assertEqual(run.inventory().requests[0]["attempts"], interrupted["attempts"])

            result = run.solve(run.original, spec)
            completed = run.inventory().requests[0]
            self.assertEqual(completed["status"], "succeeded")
            self.assertEqual(completed["attempts"], ("000001", "000002", "000003"))
            attempt_root = run._binding.leaf / "requests" / completed["request_sha256"] / "attempts"
            self.assertEqual(
                [json.loads((attempt_root / ordinal / "receipt.json").read_bytes())["outcome"] for ordinal in completed["attempts"]],
                ["failure", "interrupted", "success"],
            )
            self.assertEqual(run.resolve(run.original, spec).identity, result.identity)
            self.assertEqual(run.solve(run.original, spec).identity, result.identity)
            self.assertEqual(run.inventory().requests[0]["attempts"], completed["attempts"])

    def test_cma_interrupted_generation_replays_exactly_and_corruption_fails_closed(self) -> None:
        with TemporaryDirectory() as workspace:
            plan, resonator, capacitor = _primitive_plan()
            run = CircuitRun(plan=plan, workspace=Path(workspace))
            _, _, optimization, view = _primitive_requests(run, resonator, capacitor)
            real_run_terminal = runtime_module.run_terminal

            def complete_generation_then_interrupt(*args: object, **kwargs: object):
                real_run_terminal(*args, **kwargs)
                interruption = KeyboardInterrupt()
                interruption.termination = "terminated"  # type: ignore[attr-defined]
                raise interruption

            with patch.object(runtime_module, "run_terminal", complete_generation_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run.optimize(view, optimization)

            interrupted = run.inventory().requests[0]
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["attempts"], ("000001",))
            request_sha = interrupted["request_sha256"]
            first_attempt = run._binding.leaf / "requests" / request_sha / "attempts" / "000001"
            receipt_path = first_attempt / "receipt.json"
            receipt = json.loads(receipt_path.read_bytes())
            self.assertEqual(receipt["outcome"], "interrupted")
            self.assertEqual([row["id"] for row in receipt["artifacts"]], ["generation_000001"])
            ledger_path = first_attempt / "artifacts" / "generations" / "000001.json"
            ledger_bytes = ledger_path.read_bytes()
            ledger_sha = sha256_hex(ledger_bytes)
            self.assertEqual(receipt["artifacts"][0]["sha256"], ledger_sha)

            with TemporaryDirectory() as corrupted_root:
                corrupted = Path(corrupted_root, "workspace")
                shutil.copytree(workspace, corrupted)
                corrupt_ledger_path = next(
                    corrupted.glob(
                        f"leaves/*/requests/{request_sha}/attempts/000001/artifacts/generations/000001.json"
                    )
                )
                corrupt_receipt_path = corrupt_ledger_path.parents[2] / "receipt.json"
                corrupt_ledger = json.loads(corrupt_ledger_path.read_bytes())
                corrupt_ledger["candidates"][0]["evaluation_ordinal"] += 1
                corrupt_ledger_bytes = canonical_json_bytes(corrupt_ledger)
                corrupt_ledger_path.write_bytes(corrupt_ledger_bytes)
                corrupt_receipt = json.loads(corrupt_receipt_path.read_bytes())
                corrupt_receipt["artifacts"][0]["sha256"] = sha256_hex(corrupt_ledger_bytes)
                corrupt_receipt_path.write_bytes(canonical_json_bytes(corrupt_receipt))

                corrupt_plan, corrupt_resonator, corrupt_capacitor = _primitive_plan()
                corrupt_run = CircuitRun(plan=corrupt_plan, workspace=corrupted)
                _, _, corrupt_spec, corrupt_view = _primitive_requests(
                    corrupt_run, corrupt_resonator, corrupt_capacitor
                )
                with self.assertRaises(EvidenceIntegrityError):
                    corrupt_run.optimize(corrupt_view, corrupt_spec)
                with self.assertRaises(EvidenceIntegrityError):
                    corrupt_run.inventory()
                self.assertFalse((corrupt_ledger_path.parents[3] / "000002").exists())

            resumed = run.optimize(view, optimization)
            completed = run.inventory().requests[0]
            self.assertEqual(completed["status"], "succeeded")
            self.assertEqual(completed["attempts"], ("000001", "000002"))
            second_attempt = first_attempt.parent / "000002"
            second_attempt_document = json.loads((second_attempt / "attempt.json").read_bytes())
            self.assertEqual(second_attempt_document["resume_ledger_sha256"], ledger_sha)
            self.assertEqual((second_attempt / "artifacts" / "generations" / "000001.json").read_bytes(), ledger_bytes)
            self.assertEqual(canonical_json_bytes(resumed.ledger[0]), ledger_bytes)
            self.assertEqual(resumed.best.cost.hex(), self.optimization.best.cost.hex())
            resumed_best = resumed.best.parameters.values[capacitor.parameter("capacitance")].to("fF").magnitude
            reference_best = self.optimization.best.parameters.values[
                self.capacitor.parameter("capacitance")
            ].to("fF").magnitude
            self.assertEqual(float(resumed_best).hex(), float(reference_best).hex())


if __name__ == "__main__":
    unittest.main()
