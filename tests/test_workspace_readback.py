"""JAX Workspace versioning and exact saved-result readback."""

from __future__ import annotations

import json
from pathlib import Path
from contextlib import ExitStack
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from scnsim import CMAESSpec, CircuitPlan, CircuitRun, CostObjective, DiagonalRootSpec, DirectSolveSpec, OptimizationSpec, ParameterDefinitions, ParameterSpec, OptimizationVariable, ReductionPipeline, WorkspaceVersioningDowngradeForbidden, components, units as u


def _primitive_plan(*, plan_id="primitive_full_v1", capacitance=110.0):
    plan = CircuitPlan(id=plan_id)
    inputs = ParameterDefinitions(id="primitive_inputs")
    capacitance_ref = inputs.parameter(id="capacitance", baseline=capacitance*u.fF,
                                      spec=ParameterSpec(unit=u.fF))
    capacitor = plan.add(components.capacitor(id="resonator_cap", capacitance=capacitance_ref))
    inductor = plan.add(components.inductor(id="resonator_ind", inductance=5.8*u.nH))
    node = plan.bus(id="resonator_node")
    plan.parallel(id="parallel_lc", start=node, branches=((capacitor,), (inductor,)), end=plan.ground)
    coordinate = node.node
    boundary = plan.bus(id="boundary")
    coupling = plan.add(components.capacitor(id="coupling_cap", capacitance=6.0*u.fF))
    plan.series(id="coupler", start=boundary, elements=(coupling,), end=node)
    plan.add_port(id="signal_in", at=boundary, role="terminated", reference_impedance=50.0*u.ohm)
    return plan, coordinate, capacitance_ref


def _primitive_requests(run, resonator, capacitance):
    direct = DirectSolveSpec(frequencies=[5.5, 6.0, 6.5]*u.GHz)
    root = DiagonalRootSpec(coordinate=resonator, root_hint=6.0*u.GHz)
    optimization = OptimizationSpec(
        variables=(OptimizationVariable(parameter=capacitance, bounds=(80.0*u.fF, 140.0*u.fF)),),
        objectives=(CostObjective(id="resonance_frequency", quantity=root.frequency,
                                  target=6.2*u.GHz, weight=1.0*u.dimensionless),),
        optimizer=CMAESSpec(seed=17, max_evaluations=5))
    view = run.original.reduce(ReductionPipeline().retain(resonator))
    return direct, root, optimization, view


class WorkspaceVersioningTests(unittest.TestCase):
    def test_versioning_upgrade_preserves_leaf_and_forbids_downgrade(self):
        # Documented Workspace identities, not Run's private binding object.
        with TemporaryDirectory() as directory, ExitStack() as owners:
            workspace = Path(directory)
            plan, _, _ = _primitive_plan(plan_id="first")
            first = CircuitRun(plan=plan, workspace=workspace)
            owners.callback(first.close)
            original = json.loads((workspace / "workspace.json").read_bytes())["active_leaf"]
            plan, _, _ = _primitive_plan(plan_id="first")
            upgraded = CircuitRun(plan=plan, workspace=workspace, versioned=True)
            owners.callback(upgraded.close)
            index = json.loads((workspace / "workspace.json").read_bytes())
            self.assertEqual(index["iterations"][0]["workspace_instance_id"], original["workspace_instance_id"])
            self.assertEqual(index["iterations"][0]["plan_sha256"], original["plan_sha256"])
            plan, _, _ = _primitive_plan(plan_id="second", capacitance=120.0)
            second = CircuitRun(plan=plan, workspace=workspace, versioned=True)
            owners.callback(second.close)
            index = json.loads((workspace / "workspace.json").read_bytes())
            self.assertNotEqual(index["iterations"][1]["workspace_instance_id"], original["workspace_instance_id"])
            with self.assertRaises(WorkspaceVersioningDowngradeForbidden):
                plan, _, _ = _primitive_plan(plan_id="second", capacitance=120.0)
                CircuitRun(plan=plan, workspace=workspace, versioned=False)


def _resolve_jax_in_fresh_process(workspace, cpu_threads):
    from scnsim import configure_runtime
    from scnsim.execution.jax_backend import JaxBackend
    from tests.test_workspace_optimization_readback import _saved_values
    configure_runtime(cpu_threads=cpu_threads)
    plan, coordinate, capacitance = _primitive_plan()
    # Guard real readback: never supply a synthetic numerical result.
    with patch.object(JaxBackend, "__init__", side_effect=AssertionError("resolve initialized a backend")), \
            patch.object(JaxBackend, "execute_native", side_effect=AssertionError("resolve dispatched native work")):
        run = CircuitRun(plan=plan, workspace=Path(workspace), precision="float64")
        try:
            direct, root, optimization, view = _primitive_requests(run, coordinate, capacitance)
            return [_saved_values(value) for value in (
                run.resolve(run.original, direct), run.resolve(view, root),
                run.resolve(view, optimization))]
        finally:
            run.close()


class JaxWorkspaceReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = TemporaryDirectory()
        cls.addClassCleanup(cls._temporary.cleanup)
        cls.workspace = Path(cls._temporary.name)
        plan, coordinate, cls.capacitor = _primitive_plan()
        cls.circuit_run = CircuitRun(plan=plan, workspace=cls.workspace, precision="float64")
        cls.addClassCleanup(cls.circuit_run.close)
        cls.direct_spec, cls.root_spec, cls.optimization_spec, cls.view = _primitive_requests(
            cls.circuit_run, coordinate, cls.capacitor)
        cls.direct = cls.circuit_run.solve(cls.circuit_run.original, cls.direct_spec)
        cls.root = cls.circuit_run.evaluate(cls.view, cls.root_spec)
        cls.optimization = cls.circuit_run.optimize(cls.view, cls.optimization_spec, progress=False)

    def test_direct_root_and_cma_share_one_plan_without_identity_aliasing(self):
        identities = (self.direct.identity, self.root.identity, self.optimization.identity)
        self.assertEqual(len({identity.plan_sha256 for identity in identities}), 1)
        self.assertEqual(len({identity.request_sha256 for identity in identities}), 3)
        self.assertEqual(self.direct.s.view.matrix.shape, (3, 1, 1))
        self.assertTrue(np.isfinite(self.root.frequency.to("hertz").magnitude))
        self.assertTrue(np.isfinite(self.root.linewidth.to("hertz").magnitude))
        self.assertTrue(tuple(self.optimization.stage().history))
        self.assertEqual(self.optimization.best.parameters.allow_extrapolation, ())
        best = self.optimization.best.parameters.values[self.capacitor].to("fF").magnitude
        self.assertGreaterEqual(best, 80.0)
        self.assertLessEqual(best, 140.0)

    def test_success_reuse_and_fresh_process_resolve_are_exact(self):
        from scnsim.execution.config import get_runtime_configuration
        from scnsim.execution.jax_backend import JaxBackend
        from tests.test_workspace_optimization_readback import _saved_values
        expected = [_saved_values(value) for value in (self.direct, self.root, self.optimization)]
        with patch.object(JaxBackend, "execute_native", side_effect=AssertionError("warm readback dispatched native work")) as guard:
            actual = (self.circuit_run.solve(self.circuit_run.original, self.direct_spec),
                      self.circuit_run.evaluate(self.view, self.root_spec),
                      self.circuit_run.optimize(self.view, self.optimization_spec, progress=False))
            self.assertEqual([_saved_values(value) for value in actual], expected)
            guard.assert_not_called()
        threads = get_runtime_configuration().cpu_threads
        child = subprocess.run([sys.executable, "-c",
            "import json,sys; from tests.test_workspace_readback import _resolve_jax_in_fresh_process; "
            "print(json.dumps(_resolve_jax_in_fresh_process(sys.argv[1], None if sys.argv[2]=='default' else int(sys.argv[2]))))",
            str(self.workspace), "default" if threads is None else str(threads)],
            cwd=Path(__file__).parents[1], capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, f"fresh resolve stdout:\n{child.stdout}\nstderr:\n{child.stderr}")
        self.assertEqual(json.loads(child.stdout), expected, child.stderr)


if __name__ == "__main__":
    unittest.main()
