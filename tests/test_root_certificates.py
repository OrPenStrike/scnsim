"""Compatible shared passivity regressions; diagonal input grammar stays unchanged."""
from contextlib import ExitStack
import json
import struct
import unittest
import numpy as np
from scnsim import CircuitRun, HybridizedPoleSpec, ReductionPipeline, units as u
from scnsim.numerics.passivity import passive_imaginary_root
from tests.test_direct_analysis import _two_coordinate_plan

_ANCHORS = (6.0, 6.0 - 1e-12j, 6.0 + 1e-12j, 6.0 - 1e-3j, 6.0 + 1e-3j)


def _frequency_bits(value):
    return struct.pack("!d", float(value))


from decimal import Decimal, localcontext
import math
from scipy import linalg
from scnsim import CircuitPlan, DiagonalRootSpec, OperatorElementRootSpec, ResidueNormalizedCouplingSpec, components
from scnsim.errors import NumericalResolutionUnresolved, RootSlopeUnresolved
from tests.analytic_conditions import assert_forward

_Q_VALUES = (1e5, 1e6, 2e6, 4e6, 6e6, 8e6, 1e7, 3e7, 1e8, 1e9)

def _high_q_plan(quality):
    capacitance, inductance, resistance = 200e-15, 3e-9, 50.0
    omega = 1 / np.sqrt(inductance * capacitance)
    coupler = float(np.sqrt(capacitance / (quality * omega * resistance)))
    plan = CircuitPlan(id="hybrid_formation_scale")
    boundary, signal = plan.bus(id="boundary"), plan.bus(id="signal")
    c = plan.add(components.capacitor(id="c", capacitance=capacitance * u.farad))
    l = plan.add(components.inductor(id="l", inductance=inductance * u.henry))
    cc = plan.add(components.capacitor(id="cc", capacitance=coupler * u.farad))
    plan.parallel(id="lc", start=signal, branches=((c,), (l,)), end=plan.ground)
    plan.series(id="input", start=boundary, elements=(cc,), end=signal)
    plan.add_port(id="p", at=boundary, role="terminated", reference_impedance=resistance * u.ohm)
    c_matrix = np.array([[coupler, -coupler], [-coupler, capacitance + coupler]])
    g_matrix = np.diag([1 / resistance, 0.0])
    k_matrix = np.diag([0.0, 1 / inductance])
    return plan, (boundary.node, signal.node), (c_matrix, g_matrix, k_matrix), (capacitance, inductance, coupler, resistance)


def _oracle_pole(c, g, k, primitives):
    """Global QEP selection, then independent Decimal determinant polishing.

    Decimal precision and 60 oracle iterations match the reference's role:
    resolve oracle arithmetic; neither controls the producer's acceptance.
    """
    scale = 2 * np.pi * 1e9
    identity, zero = np.eye(2), np.zeros((2, 2))
    a = np.block([[zero, identity], [-k / scale**2, -g / scale]])
    b = np.block([[identity, zero], [zero, c]])
    cscale = np.max(np.abs(c))
    a[2:] /= cscale
    b[2:, 2:] /= cscale
    roots = 1j * linalg.eigvals(a, b) * scale
    initial = max((complex(w) for w in roots if np.isfinite(w) and w.real > 0), key=lambda w: w.real)
    with localcontext() as context:
        context.prec = 60
        def add(x, y):
            return x[0] + y[0], x[1] + y[1]
        def neg(x):
            return -x[0], -x[1]
        def sub(x, y):
            return add(x, neg(y))
        def mul(x, y):
            return x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]
        def div(x, y):
            denominator = y[0]**2 + y[1]**2
            return ((x[0] * y[0] + x[1] * y[1]) / denominator,
                    (x[1] * y[0] - x[0] * y[1]) / denominator)
        cap, ind, cc, resistance = map(Decimal.from_float, primitives)
        dc = ((cc, -cc), (-cc, cap + cc))
        dg = ((1 / resistance, Decimal(0)), (Decimal(0), Decimal(0)))
        dk = ((Decimal(0), Decimal(0)), (Decimal(0), 1 / ind))
        pair = lambda value: (value, Decimal(0))
        minus_i = Decimal(0), Decimal(-1)
        omega = Decimal.from_float(initial.real), Decimal.from_float(initial.imag)
        for _ in range(60):
            q, qp = {}, {}
            for i in range(2):
                for j in range(2):
                    q[i, j] = sub(add(pair(dk[i][j]), mul(mul(minus_i, omega), pair(dg[i][j]))),
                                  mul(mul(omega, omega), pair(dc[i][j])))
                    qp[i, j] = sub(mul(minus_i, pair(dg[i][j])), mul(pair(2 * dc[i][j]), omega))
            determinant = sub(mul(q[0, 0], q[1, 1]), mul(q[0, 1], q[1, 0]))
            derivative = sub(add(mul(qp[0, 0], q[1, 1]), mul(q[0, 0], qp[1, 1])),
                             add(mul(qp[0, 1], q[1, 0]), mul(q[0, 1], qp[1, 0])))
            updated = sub(omega, div(determinant, derivative))
            if updated == omega:
                break
            omega = updated
        expected = complex(float(omega[0]), float(omega[1]))
    q = k - expected**2 * c - 1j * expected * g
    left, _, right_h = linalg.svd(q)
    x, y = right_h[-1].conj(), left[:, -1]
    derivative = -2 * expected * c - 1j * g
    condition = ((linalg.norm(k, 2) + abs(expected) * linalg.norm(g, 2)
                  + abs(expected)**2 * linalg.norm(c, 2)) * linalg.norm(x) * linalg.norm(y)
                 / (abs(expected) * abs(y.conj() @ derivative @ x)))
    return expected, float(condition)


class RootCertificateTests(unittest.TestCase):
    def test_hybrid_lossless_anchor_imaginary_sign_preserves_frequency_bits(self):
        plan, coordinates = _two_coordinate_plan()
        with ExitStack() as owners:
            run = CircuitRun(plan=plan, storage="memory", precision="float64")
            owners.callback(run.close)
            view = run.original.reduce(ReductionPipeline().retain(*coordinates))
            observed = []
            for anchor in _ANCHORS:
                hybrid = run.evaluate(view, HybridizedPoleSpec(coordinates=coordinates, anchor=anchor * u.GHz))
                observed.append(_frequency_bits(hybrid.frequency.to("hertz").magnitude))
            print("PASSIVE_ANCHORS " + json.dumps({"hybrid": [value.hex() for value in observed]}), flush=True)
            for ordinal, value in enumerate(observed):
                with self.subTest(quantity="hybrid", anchor=_ANCHORS[ordinal]):
                    self.assertEqual(value, observed[0], {"hybrid": [item.hex() for item in observed]})

    def test_passivity_helper_rejects_ten_existing_tau(self):
        for dtype in (np.float32, np.float64):
            tolerance = dtype(256 * (2 + 1)) * np.finfo(dtype).eps
            omega = complex(1, float(10 * tolerance))
            self.assertFalse(passive_imaginary_root(omega, tolerance))


    def test_hybrid_high_q_matches_global_qep_in_both_coordinate_orders(self):
        for quality in _Q_VALUES:
            plan, coordinates, matrices, primitives = _high_q_plan(quality)
            expected, condition = _oracle_pole(*matrices, primitives)
            with ExitStack() as owners:
                run = CircuitRun(plan=plan, storage="memory", precision="float64")
                owners.callback(run.close)
                for order in (coordinates, coordinates[::-1]):
                    with self.subTest(quality=quality, order=tuple(ref.id for ref in order)):
                        view = run.original.reduce(ReductionPipeline().retain(*order))
                        spec = HybridizedPoleSpec(coordinates=order, anchor=expected.real / (2 * math.pi) * u.Hz)
                        result = run.evaluate(view, spec)
                        assert_forward(self, f"hybrid_high_q_{quality}_{tuple(ref.id for ref in order)}",
                                       complex(result.root.to("radian / second").magnitude), expected,
                                       condition, "radian / second")


    def test_repeated_physical_mode_is_not_certified_as_simple(self):
        plan = CircuitPlan(id="repeated_physical_pole")
        nodes = tuple(plan.bus(id=name) for name in ("a", "b"))
        for name, node in zip(("a", "b"), nodes):
            c = plan.add(components.capacitor(id=f"c_{name}", capacitance=80 * u.fF))
            l = plan.add(components.inductor(id=f"l_{name}", inductance=8 * u.nH))
            plan.parallel(id=f"lc_{name}", start=node, branches=((c,), (l,)), end=plan.ground)
        coordinates = tuple(node.node for node in nodes)
        with ExitStack() as owners:
            run = CircuitRun(plan=plan, storage="memory", precision="float64")
            owners.callback(run.close)
            view = run.original.reduce(ReductionPipeline().retain(*coordinates))
            with self.assertRaises((NumericalResolutionUnresolved, RootSlopeUnresolved)) as caught:
                run.evaluate(view, HybridizedPoleSpec(coordinates=coordinates, anchor=6.3 * u.GHz))
            print("ROOT_NEGATIVE " + json.dumps({"purpose": "multiple_mode", "kind": caught.exception.kind,
                                                 "stage": caught.exception.stage}), flush=True)


    def test_constant_nonzero_operator_element_rejects_zero_slope(self):
        # A purely inductive cross entry is constant in omega, with a
        # nonzero residual and exactly zero frequency slope.
        constant_plan = CircuitPlan(id="constant_cross_entry")
        a, b = constant_plan.bus(id="a"), constant_plan.bus(id="b")
        for name, node in (("a", a), ("b", b)):
            c = constant_plan.add(components.capacitor(id=f"c_{name}", capacitance=80 * u.fF))
            l = constant_plan.add(components.inductor(id=f"l_{name}", inductance=8 * u.nH))
            constant_plan.parallel(id=f"lc_{name}", start=node, branches=((c,), (l,)), end=constant_plan.ground)
        cross = constant_plan.add(components.inductor(id="cross", inductance=500 * u.nH))
        constant_plan.series(id="cross_branch", start=a, elements=(cross,), end=b)
        with ExitStack() as owners:
            run = CircuitRun(plan=constant_plan, storage="memory", precision="float64")
            owners.callback(run.close)
            view = run.original.reduce(ReductionPipeline().retain(a.node, b.node))
            with self.assertRaises(RootSlopeUnresolved):
                run.evaluate(view, OperatorElementRootSpec(row=a.node, column=b.node, root_hint=6.3 * u.GHz))


    def test_finite_residue_slopes_with_underflowed_product_are_rejected(self):
        plan = CircuitPlan(id="residue_denominator_underflow")
        a, b = plan.bus(id="a"), plan.bus(id="b")
        for name, node, inductance in (("a", a, 1e300), ("b", b, 2e300)):
            c = plan.add(components.capacitor(id=f"c_{name}", capacitance=1e-300 * u.farad))
            l = plan.add(components.inductor(id=f"l_{name}", inductance=inductance * u.henry))
            plan.parallel(id=f"lc_{name}", start=node, branches=((c,), (l,)), end=plan.ground)
        with ExitStack() as owners:
            run = CircuitRun(plan=plan, storage="memory", precision="float64")
            owners.callback(run.close)
            view = run.original.reduce(ReductionPipeline().retain(a.node, b.node))
            with self.assertRaises(RootSlopeUnresolved) as caught:
                run.evaluate(view, ResidueNormalizedCouplingSpec(
                    branch_a=DiagonalRootSpec(coordinate=a.node, root_hint=(1 / (2 * math.pi)) * u.Hz),
                    branch_b=DiagonalRootSpec(coordinate=b.node, root_hint=(1 / (math.sqrt(2) * 2 * math.pi)) * u.Hz),
                    frequency=1 * u.Hz))
            self.assertEqual(caught.exception.stage, "residue_coupling")


    def test_diagonal_and_coupling_lossless_anchor_imaginary_sign_preserves_frequency_bits(self):
        plan, coordinates = _two_coordinate_plan()
        with ExitStack() as owners:
            run = CircuitRun(plan=plan, storage="memory", precision="float64")
            owners.callback(run.close)
            view = run.original.reduce(ReductionPipeline().retain(*coordinates))
            observed = {name: [] for name in ("diagonal", "coupling")}
            for anchor in _ANCHORS:
                diagonal = run.evaluate(view, DiagonalRootSpec(coordinate=coordinates[0], root_hint=anchor * u.GHz))
                coupling = run.evaluate(view, ResidueNormalizedCouplingSpec(
                    branch_a=DiagonalRootSpec(coordinate=coordinates[0], root_hint=anchor * u.GHz),
                    branch_b=DiagonalRootSpec(coordinate=coordinates[1], root_hint=anchor * u.GHz),
                    frequency=5.5 * u.GHz))
                observed["diagonal"].append(_frequency_bits(diagonal.frequency.to("hertz").magnitude))
                roots = coupling.evaluation.branch_roots_rad_s
                observed["coupling"].append(tuple(_frequency_bits(complex(w).real / (2 * math.pi)) for w in roots))
            def encoded(value):
                return [item.hex() for item in value] if isinstance(value, tuple) else value.hex()
            print("PASSIVE_ANCHORS " + json.dumps({name: [encoded(value) for value in values]
                                                   for name, values in observed.items()}), flush=True)
            for name, values in observed.items():
                for ordinal, value in enumerate(values):
                    with self.subTest(quantity=name, anchor=_ANCHORS[ordinal]):
                        self.assertEqual(value, values[0], {name: [encoded(item) for item in values]})


