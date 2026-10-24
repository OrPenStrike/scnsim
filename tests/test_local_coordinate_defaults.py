"""Bounded local optimizer coordinates and the assigned C1 regression."""
from __future__ import annotations

import json
import unittest

import scnsim as s
from scnsim.canonical import float64_from_hex, float64_hex, quantity_envelope
from scnsim.execution.local_coordinates import LocalCoordinateMap
from scnsim.execution.prepared import _optimization_variables


def _map_for(variable, parameter, *, optimizer_scale=None):
    parameters = s.ParameterSet({parameter: parameter.baseline})
    rows, _ = _optimization_variables((variable,), parameters)
    # This is the same finalized stage-row binding performed by the local
    # optimizer before constructing its coordinate map.
    rows = tuple(dict(row, si_unit=parameter.spec.si_unit) for row in rows)
    scales = None
    if optimizer_scale is not None:
        scales = ({
            "parameter": parameter._key_record(),
            "value": quantity_envelope(
                optimizer_scale, si_unit=parameter.spec.si_unit,
                registry=s.units.registry,
            ),
        },)
    values = {
        parameter._key(): float(
            parameters.values[parameter].to(parameter.spec.si_unit).magnitude
        )
    }
    return LocalCoordinateMap(rows, values, coordinate_scales=scales)


class LocalCoordinateDefaultTests(unittest.TestCase):
    def _capacitance(self, definitions_id):
        definitions = s.ParameterDefinitions(id=definitions_id)
        return definitions.parameter(
            id="capacitance", baseline=110.0 * s.units.fF,
            spec=s.ParameterSpec(unit=s.units.fF),
        )

    def test_bounded_default_uses_width_of_resolved_bounds(self):
        parameter = self._capacitance("resolved_bounds")
        variable = s.OptimizationVariable(
            parameter=parameter,
            bounds=(80.0 * s.units.fF, 140.0 * s.units.fF),
        )._override((90.0 * s.units.fF, 130.0 * s.units.fF))

        axis = _map_for(variable, parameter).canonical_record()["coordinates"][0]
        low, high = axis["physical_bounds_f64"]
        reference = float64_from_hex(axis["reference_f64"])
        expected_width = float64_from_hex(high) - float64_from_hex(low)

        self.assertEqual(axis["mapping"], "affine")
        self.assertEqual(axis["scale_source"], "bounds_width")
        self.assertEqual(axis["scale_f64"], float64_hex(expected_width))
        self.assertEqual(axis["coordinate_unit"], "dimensionless")
        self.assertEqual(
            axis["controller_bounds_f64"],
            [
                float64_hex((float64_from_hex(low) - reference) / expected_width),
                float64_hex((float64_from_hex(high) - reference) / expected_width),
            ],
        )

        log_variable = s.OptimizationVariable(
            parameter=parameter,
            bounds=(80.0 * s.units.fF, 140.0 * s.units.fF),
            transform="log",
        )._override((90.0 * s.units.fF, 130.0 * s.units.fF))
        log_axis = _map_for(log_variable, parameter).canonical_record()["coordinates"][0]
        log_low, log_high = log_axis["physical_bounds_f64"]
        log_width = float64_from_hex(log_high) - float64_from_hex(log_low)
        self.assertEqual(log_axis["scale_source"], "bounds_width")
        self.assertEqual(log_axis["scale_f64"], float64_hex(log_width))
        self.assertEqual(
            log_axis["exponent_scale_f64"],
            float64_hex(log_width / float64_from_hex(log_axis["reference_f64"])),
        )

    def test_explicit_scale_precedes_domain_scale_and_bounded_width(self):
        parameter = self._capacitance("explicit_scale")
        bounded = s.OptimizationVariable(
            parameter=parameter,
            bounds=(80.0 * s.units.fF, 140.0 * s.units.fF),
        )
        scale_quantity = 20.0 * s.units.fF
        explicit = _map_for(
            bounded, parameter, optimizer_scale=scale_quantity,
        ).canonical_record()["coordinates"][0]
        expected_scale = quantity_envelope(
            scale_quantity,
            si_unit=parameter.spec.si_unit,
            registry=s.units.registry,
        )["si_value_f64"]
        self.assertEqual(explicit["scale_source"], "explicit")
        self.assertEqual(explicit["scale_f64"], expected_scale)

        declared_variable = s.OptimizationVariable(
            parameter=parameter,
            domain=s.OptimizationDomain.UNBOUNDED,
            scale=4.0 * s.units.fF,
        )
        declared = _map_for(declared_variable, parameter).canonical_record()["coordinates"][0]
        overridden = _map_for(
            declared_variable, parameter, optimizer_scale=2.0 * s.units.fF,
        ).canonical_record()["coordinates"][0]
        self.assertEqual(declared["scale_source"], "domain")
        self.assertEqual(overridden["scale_source"], "explicit")
        self.assertEqual(
            overridden["scale_f64"],
            float64_hex(float((2.0 * s.units.fF).to("farad").magnitude)),
        )

    def test_unscaled_signed_log_domain_retains_reference_mapping(self):
        parameter = self._capacitance("signed_log_domain")
        variable = s.OptimizationVariable(
            parameter=parameter, domain=s.OptimizationDomain.POSITIVE,
        )

        axis = _map_for(variable, parameter).canonical_record()["coordinates"][0]

        self.assertEqual(axis["mapping"], "log")
        self.assertIsNone(axis["scale_f64"])
        self.assertEqual(axis["scale_source"], "none")
        self.assertEqual(axis["exponent_scale_f64"], float64_hex(1.0))

    def test_c1_bounded_optimization_moves_toward_target_with_both_local_methods(self):
        from benchmarks.circuits import LBFGSB_CONTROLS, _c1

        target_hz = 6.2e9
        for optimizer_type, strategy in (
            (s.LBFGSBSpec, "lbfgsb"),
            (s.ProjectedLBFGSSpec, "projected_lbfgs"),
        ):
            with self.subTest(strategy=strategy):
                plan, parameter, _ports, coordinates, _declared = _c1(s)
                root = s.DiagonalRootSpec(
                    coordinate=coordinates[0], root_hint=6.0 * s.units.GHz,
                )
                spec = s.OptimizationSpec(
                    variables=(s.OptimizationVariable(
                        parameter=parameter,
                        bounds=(80.0 * s.units.fF, 140.0 * s.units.fF),
                    ),),
                    objectives=(s.CostObjective(
                        id="f", quantity=root.frequency,
                        target=6.2 * s.units.GHz,
                        weight=1.0 * s.units.dimensionless,
                    ),),
                    optimizer=optimizer_type(**LBFGSB_CONTROLS),
                )
                run = s.CircuitRun(
                    plan=plan, storage="memory", precision="float64",
                )
                try:
                    view = run.original.reduce(
                        s.ReductionPipeline().retain(*coordinates)
                    )
                    result = run.optimize(view, spec, progress=False)
                finally:
                    run.close()

                stage = result.stage()
                self.assertEqual(stage.strategy, strategy)
                self.assertEqual(stage.status, "complete")
                self.assertEqual(stage.termination_reason, "optimizer_converged")
                self.assertGreater(stage.iteration_count, 0)
                self.assertIsNotNone(result.baseline)
                self.assertIsNotNone(result.best)

                initial_hz = result.baseline.objective_components[0]["terms"][0]["value"]
                final_hz = result.best.objective_components[0]["terms"][0]["value"]
                initial_loss = ((initial_hz - target_hz) / target_hz) ** 2
                final_loss = ((final_hz - target_hz) / target_hz) ** 2
                initial_gradient = result.baseline.gradient
                gradient_summary = (
                    None if initial_gradient is None
                    else tuple(float(value) for value in initial_gradient)
                )
                observation = {
                    "event": "m2_5_c1_optimizer_observation",
                    "strategy": strategy,
                    "stage_status": stage.status,
                    "termination_reason": stage.termination_reason,
                    "iterations": stage.iteration_count,
                    "target_frequency_hz": target_hz,
                    "initial_frequency_hz": initial_hz,
                    "final_frequency_hz": final_hz,
                    "initial_target_distance_hz": abs(initial_hz - target_hz),
                    "final_target_distance_hz": abs(final_hz - target_hz),
                    "initial_normalized_loss": initial_loss,
                    "final_normalized_loss": final_loss,
                    "initial_gradient": gradient_summary,
                }
                print(json.dumps(observation, sort_keys=True), flush=True)
                self.assertLess(
                    final_loss, initial_loss,
                    msg=(f"{strategy} did not reduce the analytic target loss: "
                         f"target={target_hz:.17g} Hz, initial={initial_hz:.17g} Hz "
                         f"(loss={initial_loss:.17g}), final={final_hz:.17g} Hz "
                         f"(loss={final_loss:.17g}), "
                         f"iterations={stage.iteration_count}, "
                         f"termination={stage.termination_reason}, "
                         f"initial_gradient={gradient_summary}"),
                )


if __name__ == "__main__":
    unittest.main()
