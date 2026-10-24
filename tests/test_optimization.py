"""JAX selector coverage and baseline loss, with deferred Julia evidence."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scnsim import (
    CMAESSpec,
    CircuitPlan,
    CircuitRun,
    CostObjective,
    DiagonalRootSpec,
    HybridizedPoleSpec,
    OptimizationSpec,
    OptimizationVariable,
    ParameterDefinitions,
    ParameterSpec,
    OperatorSpec,
    ResponseElementSpec,
    ResidueNormalizedCouplingSpec,
    ReductionPipeline,
    TransferZeroSpec,
    components,
    units as u,
)


def _advanced_direct_plan() -> tuple[CircuitPlan, object]:
    """Two lossless retained modes with the original parallel-LC coupling."""
    plan = CircuitPlan(id="advanced_direct_stabilization")
    definitions = ParameterDefinitions(id="advanced_direct_inputs")
    capacitance = definitions.parameter(id="capacitance", baseline=80.0*u.fF,
                                       spec=ParameterSpec(unit=u.fF))
    capacitor_a = plan.add(
        components.capacitor(id="capacitor_a", capacitance=capacitance)
    )
    inductor_a = plan.add(
        components.inductor(id="inductor_a", inductance=8.0 * u.nH)
    )
    capacitor_b = plan.add(
        components.capacitor(id="capacitor_b", capacitance=95.0 * u.fF)
    )
    inductor_b = plan.add(
        components.inductor(id="inductor_b", inductance=7.0 * u.nH)
    )
    coupling_capacitor = plan.add(
        components.capacitor(id="coupling_capacitor", capacitance=2.0 * u.fF)
    )
    coupling_inductor = plan.add(
        components.inductor(id="coupling_inductor", inductance=500.0 * u.nH)
    )
    coordinate_a = plan.bus(id="a")
    coordinate_b = plan.bus(id="b")
    plan.parallel(
        id="mode_a",
        start=coordinate_a,
        branches=((capacitor_a,), (inductor_a,)),
        end=plan.ground,
    )
    plan.parallel(
        id="mode_b",
        start=coordinate_b,
        branches=((capacitor_b,), (inductor_b,)),
        end=plan.ground,
    )
    plan.parallel(
        id="coupling",
        start=coordinate_a,
        branches=((coupling_capacitor,), (coupling_inductor,)),
        end=coordinate_b,
    )
    return plan, capacitance


def _matched_zero_plan() -> tuple[CircuitPlan, object]:
    plan = CircuitPlan(id="transfer_zero_stabilization")
    resistor = plan.add(components.resistor(id="resistor", resistance=50.0 * u.ohm))
    definitions = ParameterDefinitions(id="matched_zero_inputs")
    capacitance = definitions.parameter(id="capacitance", baseline=80.0*u.fF,
                                       spec=ParameterSpec(unit=u.fF))
    capacitor = plan.add(components.capacitor(id="capacitor", capacitance=capacitance))
    inductor = plan.add(components.inductor(id="inductor", inductance=8.0 * u.nH))
    signal = plan.bus(id="p")
    plan.parallel(
        id="matched_parallel_lc",
        start=signal,
        branches=((resistor,), (capacitor,), (inductor,)),
        end=plan.ground,
    )
    plan.add_port(
        id="p", at=signal, role="terminated", reference_impedance=50.0 * u.ohm
    )
    return plan, capacitance


def _selector_catalog(parameter: object):
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
        CostObjective(
            id="pole_frequency", quantity=pole.frequency,
            target=6.2 * u.GHz, weight=1.0 * u.dimensionless,
        ),
        CostObjective(
            id="pole_linewidth", quantity=pole.linewidth,
            target=1.0 * u.MHz, weight=1.0 * u.dimensionless,
        ),
        CostObjective(
            id="zero_frequency", quantity=zero.frequency,
            target=5.0 * u.GHz, weight=1.0 * u.dimensionless,
        ),
        CostObjective(
            id="coupling", quantity=coupling.magnitude,
            target=50.0e6 * u.radian / u.second, weight=1.0 * u.dimensionless,
        ),
        CostObjective(
            id="response_magnitude", quantity=response.magnitude,
            target=10.0 * u.uS, weight=1.0 * u.dimensionless,
        ),
        CostObjective(
            id="response_real", quantity=response.real,
            target=0.0 * u.uS, scale=1.0 * u.uS,
            weight=1.0 * u.dimensionless,
        ),
        CostObjective(
            id="response_imag", quantity=response.imag,
            target=10.0 * u.uS, weight=1.0 * u.dimensionless,
        ),
    )
    spec = OptimizationSpec(
        variables=(
            OptimizationVariable(
                parameter=parameter, bounds=(79.5 * u.fF, 80.5 * u.fF)
            ),
        ),
        objectives=objectives,
        optimizer=CMAESSpec(seed=11, max_evaluations=5),
    )
    return pole, zero, coupling, response, objectives, spec


def _stage_candidates(result):
    return tuple(row for stage in result.stages for row in stage.history)


class OptimizationTests(unittest.TestCase):
    def test_complete_selector_catalog_is_visible_in_jax_explain(self) -> None:
        plan, parameter = _advanced_direct_plan()
        with TemporaryDirectory() as workspace:
            run = CircuitRun(
                plan=plan,
                workspace=Path(workspace),
                precision="float64",
            )
            try:
                view = run.original.reduce(ReductionPipeline().retain("a", "b"))
                _pole, _zero, _coupling, _response, _objectives, spec = _selector_catalog(parameter)
                evidence = run.explain(view, spec).evidence
                self.assertEqual(
                    [
                        (item["quantity"]["type"], item["quantity"]["projection"])
                        for item in evidence["spec"]["objectives"]
                    ],
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
            finally:
                run.close()


    def test_cma_runs_all_declared_selector_branches_and_zero_baseline(self) -> None:
        plan, parameter = _advanced_direct_plan()
        with TemporaryDirectory() as workspace:
            run = CircuitRun(
                plan=plan,
                workspace=Path(workspace),
                precision="float64",
            )
            try:
                view = run.original.reduce(ReductionPipeline().retain("a", "b"))
                pole, zero, coupling, response, _original_objectives, _selector_spec = (
                    _selector_catalog(parameter)
                )
                operator = run.evaluate(view, OperatorSpec(frequencies=[5.0, 5.5] * u.GHz))
                response_value = run.evaluate(view, response)
                pole_value = run.evaluate(view, pole)
                coupling_value = run.evaluate(view, coupling)
                advanced_objectives = (
                    CostObjective(
                        id="pole_frequency", quantity=pole.frequency,
                        target=pole_value.frequency, weight=1.0 * u.dimensionless,
                    ),
                    CostObjective(
                        id="pole_linewidth", quantity=pole.linewidth,
                        target=pole_value.linewidth, scale=1.0 * u.Hz,
                        weight=1.0 * u.dimensionless,
                    ),
                    CostObjective(
                        id="coupling", quantity=coupling.magnitude,
                        target=coupling_value.magnitude,
                        weight=1.0 * u.dimensionless,
                    ),
                    CostObjective(
                        id="response_magnitude", quantity=response.magnitude,
                        target=response_value.magnitude,
                        weight=1.0 * u.dimensionless,
                    ),
                    CostObjective(
                        id="response_real", quantity=response.real,
                        target=response_value.real, scale=1.0 * u.uS,
                        weight=1.0 * u.dimensionless,
                    ),
                    CostObjective(
                        id="response_imag", quantity=response.imag,
                        target=response_value.imag,
                        weight=1.0 * u.dimensionless,
                    ),
                )
                advanced = run.optimize(
                    view,
                    OptimizationSpec(
                        variables=(
                            OptimizationVariable(
                                parameter=parameter,
                                bounds=(79.5 * u.fF, 80.5 * u.fF),
                            ),
                        ),
                        objectives=advanced_objectives,
                        optimizer=CMAESSpec(seed=29, max_evaluations=5),
                    ),
                )
                advanced_successes = [
                    candidate
                    for candidate in _stage_candidates(advanced)
                    if candidate["status"] == "success"
                ]
                self.assertTrue(advanced_successes)
                for candidate in advanced_successes:
                    self.assertEqual(
                        [row["objective_id"] for row in candidate["components"]],
                        [objective.id for objective in advanced_objectives],
                    )
                self.assertEqual(advanced.best.cost, 0.0)
            finally:
                run.close()

        with TemporaryDirectory() as zero_workspace:
            zero_plan, zero_parameter = _matched_zero_plan()
            zero_run = CircuitRun(
                plan=zero_plan,
                workspace=Path(zero_workspace),
                precision="float64",
            )
            try:
                zero = TransferZeroSpec(
                    anchor=6.3 * u.GHz,
                    family="S",
                    input_coordinate="p",
                    output_coordinate="p",
                )
                zero_value = zero_run.evaluate(zero_run.original, zero)
                zero_objectives = (
                    CostObjective(
                        id="zero_frequency", quantity=zero.frequency,
                        target=zero_value.frequency,
                        weight=1.0 * u.dimensionless,
                    ),
                )
                result = zero_run.optimize(
                    zero_run.original,
                    OptimizationSpec(
                        variables=(
                            OptimizationVariable(
                                parameter=zero_parameter,
                                bounds=(79.5 * u.fF, 80.5 * u.fF),
                            ),
                        ),
                        objectives=zero_objectives,
                        optimizer=CMAESSpec(seed=31, max_evaluations=5),
                    ),
                )
                successes = [
                    candidate
                    for candidate in _stage_candidates(result)
                    if candidate["status"] == "success"
                ]
                self.assertTrue(successes)
                for candidate in successes:
                    self.assertEqual(
                        [row["objective_id"] for row in candidate["components"]],
                        ["zero_frequency"],
                    )
                self.assertEqual(result.best.cost, 0.0)
            finally:
                zero_run.close()


if __name__ == "__main__":
    unittest.main()
