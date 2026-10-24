"""Retained Affine-support behavior and Julia CMA evidence regressions.

These cases preserve the accepted physical validation and CMA purposes from
the original Dev5 suite.  They do not freeze the separate, still-converging
authoring-override contract.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from collections.abc import Mapping
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scnsim import AffineMap, CMAESSpec, CircuitPlan, CircuitRun, CompositePlan, CostObjective, DirectSolveSpec, InMemoryOptimizationResult, SCNSimValidationError, Library, OperatorSpec, OptimizationSpec, OptimizationVariable, ParameterDefinitions, ParameterSet, ParameterSpace, ParameterSpec, ResponseElementSpec, ReductionPipeline, components, units as u


class _AffineCatalog(Library):
    """Two accepted affine consumers with one exposed length parameter."""

    def mapped_resonator(self, *, id: str):
        composite = CompositePlan(id=id, library=self)
        definitions = ParameterDefinitions(id=f"{id}_definitions")
        length = definitions.parameter(
            id="length", baseline=1.0 * u.meter, spec=ParameterSpec(unit=u.meter)
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
        inductor = composite.add(
            components.inductor(id="inductor", inductance=8.0 * u.nH)
        )
        terminal = composite.bus(id="terminal")
        composite.parallel(
            id="parallel_resonator",
            start=terminal,
            branches=((first,), (second,), (inductor,)),
            end=composite.ground,
        )
        composite.expose_parameter(id="length", parameter=length)
        composite.expose_pin(id="terminal", at=terminal)
        return composite.build()


def _affine_plan() -> tuple[CircuitPlan, object]:
    plan = CircuitPlan(id="affine_extrapolation_stabilization")
    resonator = plan.add(_AffineCatalog().mapped_resonator(id="mapped"))
    signal = plan.bus(id="signal")
    plan.link(
        id="resonator_boundary",
        endpoints=(signal, resonator.pin("terminal")),
    )
    plan.add_port(
        id="p",
        at=signal,
        role="terminated",
        reference_impedance=50.0 * u.ohm,
    )
    return plan, resonator.parameter("length")


class AffinePhysicalValidationTests(unittest.TestCase):
    def test_ordinary_unauthorized_extrapolation_is_an_authoring_error(self) -> None:
        plan, parameter = _affine_plan()
        with TemporaryDirectory() as workspace:
            run = CircuitRun(
                plan=plan,
                workspace=Path(workspace),
                precision="float64",
            )
            try:
                view = run.original.reduce(ReductionPipeline().retain("signal"))
                with self.assertRaises(SCNSimValidationError) as raised:
                    run.evaluate(
                        view,
                        OperatorSpec(frequencies=[5.0] * u.GHz),
                        parameters=ParameterSet({parameter: 1.05 * u.meter}),
                    )
                self.assertEqual(raised.exception.stage, "authoring")
            finally:
                run.close()


def _plain_affine_evidence(value):
    if isinstance(value, Mapping):
        return {key: _plain_affine_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_affine_evidence(item) for item in value]
    return value


def _affine_grid(parameter):
    return ParameterSpace.grid(
        axes={parameter: [1.05, 1.0, 1.05] * u.meter},
        fixed=ParameterSet({}, allow_extrapolation=(parameter,)),
    )


def _resolve_affine_public_evidence(workspace, mode, cpu_threads):
    from scnsim import configure_runtime
    from scnsim.execution.jax_backend import JaxBackend

    configure_runtime(cpu_threads=cpu_threads)
    plan, parameter = _affine_plan()
    with patch.object(JaxBackend, "__init__", side_effect=AssertionError("resolve initialized a backend")), \
            patch.object(JaxBackend, "execute_native", side_effect=AssertionError("resolve dispatched native work")):
        run = CircuitRun(plan=plan, workspace=Path(workspace), precision="float64")
        try:
            if mode == "grid":
                result = run.resolve(run.original, DirectSolveSpec(frequencies=[5.0] * u.GHz),
                                     parameters=_affine_grid(parameter))
                values = [result.points[index].result.evidence for index in range(3)]
            else:
                point = ParameterSet({parameter: 1.05 * u.meter}, allow_extrapolation=(parameter,))
                view = run.original.reduce(ReductionPipeline().retain("signal"))
                values = [
                    run.resolve(run.original, DirectSolveSpec(frequencies=[5.0] * u.GHz),
                                parameters=point).evidence,
                    run.resolve(view, OperatorSpec(frequencies=[5.0] * u.GHz),
                                parameters=point).evidence,
                ]
            return _plain_affine_evidence(values)
        finally:
            run.close()


class AffineEvidenceTests(unittest.TestCase):
    """Same-call host support facts, independent of Julia receipt layout."""

    def assert_affine_rows(self, rows, parameter, value, sources):
        self.assertEqual(len(rows), 2)
        self.assertEqual([tuple(row["consumer_target"]["path"]) for row in rows],
                         [("mapped", "first"), ("mapped", "second")])
        expected_bits = struct.pack(">d", value).hex()
        for row in rows:
            self.assertEqual(row["parameter"], {"definitions_id": parameter.definitions_id,
                                                 "parameter_id": parameter.id})
            self.assertEqual(row["consumer_target"]["field"], "capacitance")
            self.assertEqual(row["input_value"]["si_unit"], "meter")
            self.assertEqual(row["input_value"]["si_value_f64"], expected_bits)
            self.assertEqual([item["si_value_f64"] for item in row["support"]],
                             [struct.pack(">d", .99).hex(), struct.pack(">d", 1.01).hex()])
            side = "lower" if value < .99 else "upper"
            self.assertEqual(row["side"], side)
            distance = .99 - value if side == "lower" else value - 1.01
            self.assertEqual(row["distance"]["si_value_f64"], struct.pack(">d", distance).hex())
            self.assertEqual(tuple(row["authorization_sources"]), tuple(sources))
            self.assertEqual(row["authorized"], bool(sources))

    def test_authorized_extrapolation_records_each_fanout_edge_in_both_storages(self):
        plan, parameter = _affine_plan()
        point = ParameterSet({parameter: 1.05 * u.meter}, allow_extrapolation=(parameter,))
        for storage in ("memory", "workspace"):
            with self.subTest(storage=storage), TemporaryDirectory() as directory:
                run = CircuitRun(plan=plan, precision="float64", storage=storage,
                                 **({"workspace": Path(directory)} if storage == "workspace" else {}))
                try:
                    view = run.original.reduce(ReductionPipeline().retain("signal"))
                    # Keep the original Operator request and saved readback purpose.
                    operator_spec = OperatorSpec(frequencies=[5.0] * u.GHz)
                    operator = run.evaluate(view, operator_spec, parameters=point)
                    self.assert_affine_rows(operator.evidence.get("affine_evidence", ()),
                                            parameter, 1.05, ("parameter_set",))
                    direct_spec = DirectSolveSpec(frequencies=[5.0] * u.GHz)
                    direct = run.solve(run.original, direct_spec, parameters=point)
                    self.assert_affine_rows(direct.evidence.get("affine_evidence", ()),
                                            parameter, 1.05, ("parameter_set",))
                    for value in (operator, direct):
                        self.assert_readonly_affine_evidence(value)
                    baseline_direct = run.solve(run.original, direct_spec)
                    baseline_operator = run.evaluate(view, operator_spec)
                    for value in (baseline_direct, baseline_operator):
                        self.assertNotIn("affine_evidence", value.evidence)
                    if storage == "workspace":
                        from scnsim.execution.jax_backend import JaxBackend
                        with patch.object(JaxBackend, "execute_native", side_effect=AssertionError("warm readback dispatched native work")):
                            warm = run.solve(run.original, direct_spec, parameters=point)
                            resolved_direct = run.resolve(run.original, direct_spec, parameters=point)
                            resolved_operator = run.resolve(view, operator_spec, parameters=point)
                        self.assertEqual(warm.identity, direct.identity)
                        self.assertEqual(resolved_direct.evidence, direct.evidence)
                        self.assertEqual(resolved_operator.evidence, operator.evidence)
                        self.assertEqual(resolved_operator.identity, operator.identity)
                    spec = ResponseElementSpec(family="Y", input_coordinate="signal",
                                               output_coordinate="signal", frequency=5.0 * u.GHz)
                    result = run.evaluate(view, spec, parameters=point)
                    self.assert_affine_rows(result.evidence["affine_evidence"], parameter,
                                            1.05, ("parameter_set",))
                    if storage == "workspace":
                        resolved = run.resolve(view, spec, parameters=point)
                        self.assertEqual(resolved.evidence["affine_evidence"],
                                         result.evidence["affine_evidence"])
                finally:
                    run.close()
                expected = _plain_affine_evidence([direct.evidence, operator.evidence])
                self.assert_affine_rows(direct.evidence.get("affine_evidence", ()),
                                        parameter, 1.05, ("parameter_set",))
                self.assert_affine_rows(operator.evidence.get("affine_evidence", ()),
                                        parameter, 1.05, ("parameter_set",))
                if storage == "workspace":
                    self.assertEqual(self.fresh_affine_evidence(directory, "point"), expected)

    def assert_readonly_affine_evidence(self, result):
        rows = result.evidence.get("affine_evidence", ())
        with self.assertRaises(TypeError):
            result.evidence["affine_evidence"] = ()
        with self.assertRaises(TypeError):
            rows[0]["input_value"]["si_value_f64"] = "mutated"
        self.assertEqual(result.evidence.get("affine_evidence", ()), rows)

    def fresh_affine_evidence(self, workspace, mode):
        from scnsim.execution.config import get_runtime_configuration

        cpu_threads = get_runtime_configuration().cpu_threads
        child = subprocess.run([
            sys.executable, "-c",
            "import json,sys; from tests.test_affine_parameters import _resolve_affine_public_evidence; "
            "print(json.dumps(_resolve_affine_public_evidence(sys.argv[1],sys.argv[2],None if sys.argv[3]=='default' else int(sys.argv[3]))))",
            str(workspace), mode, "default" if cpu_threads is None else str(cpu_threads),
        ], capture_output=True, text=True)
        self.assertEqual(child.returncode, 0,
                         f"fresh resolve stdout:\n{child.stdout}\nstderr:\n{child.stderr}")
        return json.loads(child.stdout)

    def test_direct_grid_repeated_point_preserves_original_affine_evidence(self):
        plan, parameter = _affine_plan()
        spec = DirectSolveSpec(frequencies=[5.0] * u.GHz)
        for storage in ("memory", "workspace"):
            with self.subTest(storage=storage), TemporaryDirectory() as directory:
                run = CircuitRun(plan=plan, precision="float64", storage=storage,
                                 **({"workspace": Path(directory)} if storage == "workspace" else {}))
                try:
                    space = _affine_grid(parameter)
                    result = run.solve(run.original, spec, parameters=space)
                    indices = [result.points[index].source_index for index in range(3)]
                    self.assertEqual([(index,) if type(index) is int else index
                                      for index in indices],
                                     [(0,), (1,), (2,)])
                    points = [result.points[index].result for index in range(3)]
                    original = _plain_affine_evidence([value.evidence for value in points])
                    self.assert_affine_rows(points[0].evidence.get("affine_evidence", ()),
                                            parameter, 1.05, ("parameter_set",))
                    self.assertNotIn("affine_evidence", points[1].evidence)
                    self.assertEqual(points[2].evidence, points[0].evidence)
                    self.assert_readonly_affine_evidence(points[2])
                    if storage == "workspace":
                        from scnsim.execution.jax_backend import JaxBackend
                        with patch.object(JaxBackend, "execute_native", side_effect=AssertionError("warm grid readback dispatched native work")):
                            warm = run.solve(run.original, spec, parameters=space)
                            resolved = run.resolve(run.original, spec, parameters=space)
                            self.assertEqual(_plain_affine_evidence(
                                [warm.points[index].result.evidence for index in range(3)]), original)
                            self.assertEqual(_plain_affine_evidence(
                                [resolved.points[index].result.evidence for index in range(3)]), original)
                finally:
                    run.close()
                # Access each existing point again after close, including lazy readers.
                self.assertEqual(_plain_affine_evidence(
                    [result.points[index].result.evidence for index in range(3)]), original)
                if storage == "workspace":
                    self.assertEqual(self.fresh_affine_evidence(directory, "grid"), original)

    def test_cma_records_rejected_and_authorized_affine_candidates(self):
        plan, parameter = _affine_plan()
        response = ResponseElementSpec(family="Y", input_coordinate="signal",
                                       output_coordinate="signal", frequency=5.0 * u.GHz)
        spec = OptimizationSpec(
            variables=(OptimizationVariable(parameter=parameter,
                                              bounds=(.9 * u.meter, 1.1 * u.meter)),),
            objectives=(CostObjective(id="response", quantity=response.magnitude,
                                      target=1.0 * u.mS, weight=1.0 * u.dimensionless),),
            optimizer=CMAESSpec(seed=23, max_evaluations=5),
        )
        for storage in ("memory", "workspace"):
            with self.subTest(storage=storage), TemporaryDirectory() as directory:
                run = CircuitRun(plan=plan, precision="float64", storage=storage,
                                 **({"workspace": Path(directory)} if storage == "workspace" else {}))
                try:
                    view = run.original.reduce(ReductionPipeline().retain("signal"))
                    rejected = run.optimize(view, spec)
                    self.assert_candidate_rows(rejected, parameter, ())
                    authorized_spec = spec.with_variable_overrides(
                        bounds={}, allow_extrapolation=(parameter,))
                    authorized = run.optimize(view, authorized_spec)
                    self.assert_candidate_rows(authorized, parameter, ("optimization_spec",))
                    if storage == "workspace":
                        saved = run.resolve(view, authorized_spec)
                        self.assertEqual(saved.best.parameters.allow_extrapolation, ())
                        self.assertEqual(tuple(saved.stage().history), tuple(authorized.stage().history))
                    # Keep origins call-local even when the admitted boolean/core
                    # is the same. A saved cache read retains its execution's origins.
                    both_run = run if storage == "memory" else CircuitRun(
                        plan=plan, precision="float64",
                        workspace=Path(directory) / "both sources")
                    try:
                        both_view = both_run.original.reduce(ReductionPipeline().retain("signal"))
                        both_point = ParameterSet({}, allow_extrapolation=(parameter,))
                        both = both_run.optimize(both_view, authorized_spec, parameters=both_point)
                        self.assertEqual(both_point.allow_extrapolation, (parameter,))
                        self.assert_candidate_rows(both, parameter,
                                                   ("parameter_set", "optimization_spec"))
                        if storage == "workspace":
                            cached = both_run.optimize(both_view, authorized_spec)
                            self.assertEqual(cached.identity, both.identity)
                            self.assert_candidate_rows(cached, parameter,
                                                       ("parameter_set", "optimization_spec"))
                    finally:
                        if both_run is not run:
                            both_run.close()
                finally:
                    run.close()

    def assert_candidate_rows(self, result, parameter, sources):
        self.assertEqual(result.best.parameters.allow_extrapolation, ())
        for stage in result.stages:
            for selected in (stage.observed_best, stage.accepted_iterate):
                if selected is not None:
                    self.assertEqual(selected.parameters.allow_extrapolation, ())
            # Exercise both lazy indexing/slicing and the full iteration below.
            if len(stage.history):
                for candidate in (stage.history[0], *stage.history[:1]):
                    if isinstance(result, InMemoryOptimizationResult):
                        self.assertEqual(candidate.parameters.allow_extrapolation, ())
                    else:
                        self.assertEqual(tuple(candidate["parameters"]["allow_extrapolation"]), ())
        recorded = []
        if isinstance(result, InMemoryOptimizationResult):
            def values():
                for trial in result.stage().history:
                    if trial.kind != "baseline":
                        self.assertEqual(trial.parameters.allow_extrapolation, ())
                    if trial.status == "failure":
                        self.assertEqual(trial.cost, float("inf"))
                    yield (trial.parameters.values[parameter].to("meter").magnitude,
                           trial.status, trial.failure,
                           trial.objective_components[0].get("affine_evidence", ()))
            candidates = values()
        else:
            def values():
                for row in result.stage().history:
                    self.assertEqual(tuple(row["parameters"]["allow_extrapolation"]), ())
                    if row["status"] == "failure":
                        self.assertEqual(row["cost_f64"], "7ff0000000000000")
                    binding = next(item for item in row["parameters"]["bindings"]
                                   if item["parameter"]["definitions_id"] == parameter.definitions_id
                                   and item["parameter"]["parameter_id"] == parameter.id)
                    value = struct.unpack(">d", bytes.fromhex(binding["value"]["si_value_f64"]))[0]
                    yield value, row["status"], row.get("failure"), row.get("affine_evidence", ())
            candidates = values()
        for value, status, failure, rows in candidates:
            if .99 <= value <= 1.01:
                self.assertFalse(rows)
                continue
            self.assert_affine_rows(rows, parameter, value, sources)
            recorded.append(rows)
            if sources:
                self.assertEqual(status, "success")
            else:
                self.assertEqual(status, "failure")
                self.assertEqual(failure["kind"], "invalid_candidate_physical_parameter")
                self.assertEqual(failure["stage"], "affine_support")
        self.assertTrue(recorded, "The original fixed CMA fixture did not exercise its out-of-support branch")
        self.assertEqual(result.best.parameters.allow_extrapolation, ())


if __name__ == "__main__":
    unittest.main()
