"""H3 DUT admittance regressions with the assigned H2 condition bounds."""
from decimal import Decimal, localcontext
import json
import math
import unittest

import numpy as np

from scnsim import (CircuitPlan, CircuitRun, DirectSolveSpec, JVP, VJP,
    OutputCoordinate, ParameterDefinitions, ParameterSpec, ReductionPipeline,
    ResponseElementSpec, SensitivitySpec, TransferZeroSpec, components, units as u)
from tests.analytic_conditions import assert_forward, matched_zero_oracle, response_oracle
from tests.test_direct_analysis import _advanced_direct_plan, _matched_zero_plan


C, L, CC = 200e-15, 3e-9, 1e-15
W0 = 1 / math.sqrt(L * C)
FREQUENCIES = np.array((0.9, 0.999, 1.0, 1.001, 1.1)) * W0 / (2 * math.pi)


def _plan(quality, reference=50.0, *, parameterized=False):
    plan = CircuitPlan(id="dut_admittance")
    parameter = None
    capacitance = C * u.F
    if parameterized:
        definitions = ParameterDefinitions(id="dut_parameters")
        parameter = definitions.parameter(id="capacitance", baseline=capacitance,
                                          spec=ParameterSpec(unit=u.F))
        capacitance = parameter
    port, resonator = plan.bus(id="p"), plan.bus(id="resonator")
    cap = plan.add(components.capacitor(id="cap", capacitance=capacitance))
    ind = plan.add(components.inductor(id="ind", inductance=L * u.H))
    resistor = plan.add(components.resistor(id="loss", resistance=quality / (W0 * C) * u.ohm))
    coupling = plan.add(components.capacitor(id="coupling", capacitance=CC * u.F))
    plan.parallel(id="resonator", start=resonator, branches=((cap,), (ind,), (resistor,)), end=plan.ground)
    plan.series(id="coupling", start=port, elements=(coupling,), end=resonator)
    plan.add_port(id="p", at=port, role="terminated", reference_impedance=reference * u.ohm)
    return plan, parameter


def _oracle(omega, quality):
    """Independent closed scalar circuit; return ReY and its relative-C tangent."""
    with localcontext() as context:
        context.prec = 50  # Oracle arithmetic, not a comparison tolerance.
        w, c, l, cc, resistance = map(Decimal.from_float,
            (omega, C, L, CC, quality / (W0 * C)))
        g, a = 1 / resistance, w * cc
        d = 1 / (w * l) - w * c - a
        denominator = g*g + d*d
        real = a*a*g / denominator
        tangent = 2*w*a*a*g*d*c / denominator**2
        k = 1 / (w*l) + w*(c+cc)
        condition = sum(map(abs, (-1+2*g*g/denominator,
            2*d*w*c/denominator, 2*d/(w*l*denominator),
            2+2*d*a/denominator, 2+2*d*k/denominator)))
        derivative_condition = sum(map(abs, (-1+4*g*g/denominator,
            1-w*c/d+4*d*w*c/denominator,
            -1/(w*l*d)+4*d/(w*l*denominator),
            2-a/d+4*d*a/denominator, 3-k/d+4*d*k/denominator)))
        return real, tangent, float(condition), float(derivative_condition)


class DirectAdmittanceTests(unittest.TestCase):
    def test_dissipation_against_independent_oracle_and_numpy_schur(self):
        for quality in (1e3, 1e5, 1e7, 1e9):
            plan, _ = _plan(quality)
            run = CircuitRun(plan=plan, storage="memory", precision="float64")
            try:
                result = run.solve(run.original, DirectSolveSpec(frequencies=FREQUENCIES * u.Hz))
                values = result.y.view.matrix.to(u.S).magnitude[:, 0, 0]
                for frequency, actual in zip(FREQUENCIES, values, strict=True):
                    omega = 2 * math.pi * frequency
                    expected, _, condition, _ = _oracle(omega, quality)
                    coupling = -1j * omega * CC
                    resonator = 1 / (quality / (W0*C)) + 1j / (omega*L) - 1j*omega*C
                    schur = coupling - coupling*coupling / (coupling+resonator)
                    print("NUMPY_SCHUR " + json.dumps(dict(quality=quality, frequency_hz=float(frequency),
                        real=float(schur.real), oracle=float(expected))), flush=True)
                    assert_forward(self, "DUT_ReY", actual.real, float(expected), condition,
                                   "S", frequency_perturbed=True)
                    g = 1 / (quality / (W0*C))
                    b = 1/(omega*L)-omega*C
                    scale = g*g+b*b
                    z_condition = sum(map(abs, (-1+2*g*g/scale,
                        2*b*omega*C/scale, 2*b/(omega*L*scale),
                        2*b*(1/(omega*L)+omega*C)/scale)))
                    index = int(np.flatnonzero(FREQUENCIES == frequency)[0])
                    actual_z = result.z.view.matrix.to(u.ohm).magnitude[index, 0, 0]
                    with localcontext() as context:
                        context.prec = 50
                        wd, ld, cd, rd = map(Decimal.from_float,
                            (omega, L, C, quality/(W0*C)))
                        gd, bd = 1/rd, 1/(wd*ld)-wd*cd
                        expected_z = float(gd/(gd*gd+bd*bd))
                    assert_forward(self, "DUT_ReZ", actual_z.real, expected_z,
                                   z_condition, "ohm", frequency_perturbed=True)
            finally:
                run.close()

    def test_reference_impedance_does_not_supply_dut_loss(self):
        # Change only the reference normalization, not the dissipative DUT.
        for reference in (50.0, 100.0):
            plan, _ = _plan(1e5, reference)
            run = CircuitRun(plan=plan, storage="memory", precision="float64")
            try:
                result = run.solve(run.original, DirectSolveSpec(frequencies=[FREQUENCIES[0]] * u.Hz))
                expected, _, condition, _ = _oracle(2*math.pi*FREQUENCIES[0], 1e5)
                assert_forward(self, "reference_independent_ReY", result.y.view.matrix.to(u.S).magnitude[0, 0, 0].real,
                               float(expected), condition, "S", frequency_perturbed=True)
            finally:
                run.close()

    def test_retained_nonport_scalar_y_and_z(self):
        run = CircuitRun(plan=_advanced_direct_plan(), storage="memory", precision="float64")
        try:
            view = run.original.reduce(ReductionPipeline().retain("a", "b"))
            omega = 2*math.pi*5.5e9
            expected_y, condition_y = response_oracle(omega)
            c = np.array([[82e-15, -2e-15], [-2e-15, 97e-15]])
            k = np.array([[1/8e-9+1/500e-9, -1/500e-9], [-1/500e-9, 1/7e-9+1/500e-9]])
            y = 1j*(k/omega-omega*c)
            det = y[0, 0]*y[1, 1]-y[0, 1]**2
            local_bounds = 2*(np.abs(k)/omega+omega*np.abs(c))/np.abs(y)
            det_condition = (abs(y[0, 0]*y[1, 1])*(local_bounds[0, 0]+local_bounds[1, 1])
                             +2*abs(y[0, 1])**2*local_bounds[0, 1])/abs(det)
            for family, expected, condition, unit in (
                    ("Y", expected_y, condition_y, u.S),
                    ("Z", -y[0, 1]/det, condition_y+det_condition, u.ohm)):
                result = run.evaluate(view, ResponseElementSpec(family=family,
                    input_coordinate="a", output_coordinate="b", frequency=5.5*u.GHz))
                assert_forward(self, "retained_"+family, complex(result.value.to(unit).magnitude),
                               expected, condition, str(unit), frequency_perturbed=True)
        finally:
            run.close()

    def test_parameter_jvp_vjp_and_independent_taylor(self):
        quality, frequency = 1e5, float(FREQUENCIES[0])
        plan, parameter = _plan(quality, parameterized=True)
        run = CircuitRun(plan=plan, storage="memory", precision="float64")
        try:
            response = ResponseElementSpec(family="Y", input_coordinate="p", output_coordinate="p", frequency=frequency*u.Hz)
            quantities = {"loss": response.real.on(run.original)}
            jvp = run.sensitivity(SensitivitySpec(quantities=quantities, wrt=(parameter,),
                derivative=JVP(direction={parameter: C*u.F})))
            block = jvp.jvp.blocks[0]
            self.assertEqual(block.status, "complete")
            actual = float(block.value.to(u.S).magnitude)
            real, tangent, _, condition = _oracle(2*math.pi*frequency, quality)
            assert_forward(self, "ReY_relative_C_JVP", actual, float(tangent), condition, "S", frequency_perturbed=True)
            vjp = run.sensitivity(SensitivitySpec(quantities=quantities, wrt=(parameter,),
                derivative=VJP(weights={OutputCoordinate(quantity_label="loss"): 1/u.S})))
            block = vjp.vjp.blocks[0]
            self.assertEqual(block.status, "complete")
            assert_forward(self, "ReY_C_VJP", float(block.values[0]), float(tangent)/C,
                           condition+1, "1/F", frequency_perturbed=True)
            # Original oracle-harness samples, with no new slope/accuracy gate.
            with localcontext() as context:
                context.prec = 50
                w,c,l,cc,r = map(Decimal.from_float, (2*math.pi*frequency,C,L,CC,quality/(W0*C)))
                g,a = 1/r,w*cc
                for step in (1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10):
                    t=Decimal.from_float(step)
                    d=1/(w*l)-w*c*(1+t)-a
                    remainder=a*a*g/(g*g+d*d)-real-t*Decimal.from_float(actual)
                    print("AD_TAYLOR "+json.dumps(dict(step=step, remainder=str(remainder),
                        oracle_tangent=str(tangent), actual_jvp=actual)), flush=True)
        finally:
            run.close()

    def test_matched_s_zero_consumes_analytic_frequency_jet(self):
        run = CircuitRun(plan=_matched_zero_plan(), storage="memory", precision="float64")
        try:
            result = run.evaluate(run.original, TransferZeroSpec(anchor=6.3*u.GHz,
                family="S", input_coordinate="p", output_coordinate="p"))
            expected, condition = matched_zero_oracle()
            actual = complex(result.zero.to("radian/second").magnitude)
            assert_forward(self, "DUT_matched_S_zero", actual, expected, condition, "rad/s")
            # One-Port S uses the existing bordered numerator of order q+1=2.
            numerator_order = 2
            transfer_tau = 256 * (numerator_order + 1) * np.finfo(np.float64).eps
            self.assertLessEqual(abs(actual.imag), transfer_tau * abs(actual))
        finally:
            run.close()
