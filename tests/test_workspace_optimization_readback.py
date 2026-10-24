"""Real JAX Optimization completion reuse and process-independent readback.

The LC fixture and optimizer controls come from the assigned M0 C1 workload.
These assertions concern exact saved results, not optimizer or root accuracy.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from scnsim import (
    CMAESSpec, CircuitPlan, CircuitRun, CostObjective, DiagonalRootSpec,
    LBFGSBSpec, OptimizationResult, OptimizationSpec, OptimizationVariable,
    ParameterDefinitions, ParameterRef, ParameterSet, ParameterSpec,
    ReductionPipeline, components, configure_runtime, units as u,
)
from scnsim.execution.config import get_runtime_configuration
from scnsim.execution.jax_backend import JaxBackend


def _fixture(strategy):
    definitions = ParameterDefinitions(id="optimization_readback_inputs")
    capacitance = definitions.parameter(
        id="capacitance", baseline=110.0 * u.fF, spec=ParameterSpec(unit=u.fF),
    )
    plan = CircuitPlan(id="optimization_readback_lc")
    capacitor = plan.add(components.capacitor(id="capacitor", capacitance=capacitance))
    inductor = plan.add(components.inductor(id="inductor", inductance=5.8 * u.nH))
    signal = plan.bus(id="signal")
    plan.parallel(id="lc", start=signal, branches=((capacitor,), (inductor,)), end=plan.ground)
    boundary = plan.bus(id="boundary")
    coupler = plan.add(components.capacitor(id="coupler", capacitance=6.0 * u.fF))
    plan.series(id="coupling", start=boundary, elements=(coupler,), end=signal)
    plan.add_port(id="signal_in", at=boundary, role="terminated", reference_impedance=50.0 * u.ohm)
    if strategy == "cma":
        optimizer = CMAESSpec(seed=7, population_size=8, max_evaluations=64)
    elif strategy == "lbfgsb":
        optimizer = LBFGSBSpec(
            max_evaluations=15000, max_iterations=15000,
            ftol=2.220446049250313e-09, gtol=1e-05,
            max_line_search=20, commit_every_iterations=1,
        )
    else:
        raise ValueError(strategy)
    root = DiagonalRootSpec(coordinate=signal.node, root_hint=6.0 * u.GHz)
    spec = OptimizationSpec(
        variables=(OptimizationVariable(parameter=capacitance, bounds=(80.0 * u.fF, 140.0 * u.fF)),),
        objectives=(CostObjective(id="frequency", quantity=root.frequency,
                                 target=6.2 * u.GHz, weight=1.0 * u.dimensionless),),
        optimizer=optimizer,
    )
    return plan, signal.node, spec


def _saved_values(value):
    """Project public values losslessly; omit private reader/verification tokens."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, complex):
        return {"complex": [value.real.hex(), value.imag.hex()]}
    if isinstance(value, ParameterRef):
        return {"definitions_id": value.definitions_id, "parameter_id": value.id}
    if isinstance(value, ParameterSet):
        return {
            "values": [[_saved_values(ref), _saved_values(quantity)]
                       for ref, quantity in value.values.items()],
            "allow_extrapolation": [_saved_values(ref) for ref in value.allow_extrapolation],
        }
    if isinstance(value, u.Quantity):
        return {"unit": str(value.units), "magnitude": _saved_values(np.asarray(value.magnitude))}
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape), "bytes": value.tobytes().hex()}
    if isinstance(value, np.generic):
        return _saved_values(value.item())
    if is_dataclass(value):
        return {"type": type(value).__name__, "fields": {
            field.name: _saved_values(getattr(value, field.name))
            for field in fields(value) if not field.name.startswith("_")
        }}
    if isinstance(value, Mapping):
        return {"mapping": [[_saved_values(key), _saved_values(item)] for key, item in value.items()]}
    if isinstance(value, Sequence):
        return [_saved_values(item) for item in value]
    raise TypeError(f"Unprojected public result value: {type(value).__module__}.{type(value).__name__}")


def _resolve_in_fresh_process(workspace, strategy, cpu_threads):
    configure_runtime(cpu_threads=cpu_threads)
    plan, coordinate, spec = _fixture(strategy)
    # A backend construction or dispatch is a real readback regression, never a
    # replacement numerical result. No optimizer is called in this process.
    with patch.object(JaxBackend, "__init__", side_effect=AssertionError("resolve initialized a numerical backend")), \
            patch.object(JaxBackend, "execute_native", side_effect=AssertionError("resolve dispatched native work")):
        run = CircuitRun(plan=plan, workspace=Path(workspace), precision="float64")
        try:
            view = run.original.reduce(ReductionPipeline().retain(coordinate))
            result = run.resolve(view, spec)
        finally:
            run.close()
        return _saved_values(result)


class WorkspaceOptimizationReadbackTests(unittest.TestCase):
    def _check_readback(self, strategy):
        plan, coordinate, spec = _fixture(strategy)
        with TemporaryDirectory(prefix="optimization readback ") as directory:
            run = CircuitRun(plan=plan, workspace=Path(directory), precision="float64")
            try:
                view = run.original.reduce(ReductionPipeline().retain(coordinate))
                native_calls = []
                execute_native = JaxBackend.execute_native

                def counted(backend, *args, **kwargs):
                    native_calls.append(kwargs.get("phase"))
                    return execute_native(backend, *args, **kwargs)

                with patch.object(JaxBackend, "execute_native", new=counted):
                    first = run.optimize(view, spec, progress=False)
                self.assertGreater(len(native_calls), 0)
                self.assertIsInstance(first, OptimizationResult)
                self.assertIsNotNone(first.best)
                self.assertTrue(first.stages)
                self.assertTrue(all(stage.status == "complete" for stage in first.stages))
                expected = _saved_values(first)
                with patch.object(JaxBackend, "execute_native", side_effect=AssertionError("warm readback dispatched native work")) as guard:
                    cached = run.optimize(view, spec, progress=False)
                    resolved = run.resolve(view, spec)
                    self.assertEqual(_saved_values(cached), expected)
                    self.assertEqual(_saved_values(resolved), expected)
                    guard.assert_not_called()
                self.assertEqual(cached.identity, first.identity)
                self.assertEqual(resolved.identity, first.identity)
                # Runtime resources are part of the exact request identity.
                threads = get_runtime_configuration().cpu_threads
                child = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--resolve", directory,
                     strategy, "default" if threads is None else str(threads)],
                    capture_output=True, text=True,
                )
                self.assertEqual(child.returncode, 0,
                                 f"fresh resolve stdout:\n{child.stdout}\nstderr:\n{child.stderr}")
                self.assertEqual(json.loads(child.stdout), expected, child.stderr)
            finally:
                run.close()
            # Fixed public histories must still read after their Run is closed.
            self.assertEqual(_saved_values(first), expected)

    def test_cma_success_cache_and_fresh_process_resolve_are_exact(self):
        self._check_readback("cma")

    def test_lbfgsb_success_cache_and_fresh_process_resolve_are_exact(self):
        self._check_readback("lbfgsb")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--resolve":
        threads = None if sys.argv[4] == "default" else int(sys.argv[4])
        print(json.dumps(_resolve_in_fresh_process(sys.argv[2], sys.argv[3], threads)))
    else:
        unittest.main()
