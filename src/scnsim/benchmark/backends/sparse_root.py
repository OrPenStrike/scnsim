"""Dtype-exact local Newton and Schur certificates over sparse systems.

The host still owns continuation, baseline anchoring and candidate policy. This
module owns exactly the existing local32-step algorithm and numerical status;
it never widens a complex64 iterate into Python complex arithmetic.
"""
from __future__ import annotations
import numpy as np
from .sparse_direct import first, maximum_abs, selected_state, tau


def _quotient(num, den, rd):
    # Root certificate formula, distinct from the solve residual's finite guard.
    return rd(0 if num == 0 else np.inf) if den == 0 else rd(num / den)


def element_state(system, omega, assembled, coordinate, measurements):
    state, status = selected_state(system, assembled, measurements)
    rd, cd = system.real_dtype, system.complex_dtype
    if state is None:
        return cd(np.nan), cd(np.nan), np.full(6, np.inf, dtype=rd), status
    F, Fp, Q, Qp, bound, X, Xp, eta_e = state
    row = system.selected[coordinate]
    x = np.zeros(system.n, dtype=cd)
    x[row] = cd(1)
    slope_scale = rd(abs(Qp[row, row]))
    if len(system.eliminated):
        x[system.eliminated] = -X[:, coordinate]
        qp_row = Qp[row, system.eliminated].toarray().ravel()
        q_row = Q[row, system.eliminated].toarray().ravel()
        slope_scale = rd(slope_scale + np.sum(np.abs(qp_row) * np.abs(X[:, coordinate]), dtype=rd)
                         + np.sum(np.abs(q_row) * np.abs(Xp[:, coordinate]), dtype=rd))
    checked = np.concatenate((np.asarray([row]), system.eliminated))
    qx, bx = Q @ x, bound @ np.abs(x)
    num, den = maximum_abs(qx[checked], rd), maximum_abs(bx[checked], rd)
    eta_q = _quotient(num, den, rd)
    f, fp = cd(F[coordinate, coordinate]), cd(Fp[coordinate, coordinate])
    eta_f = _quotient(rd(abs(f)), rd(bx[row]), rd)
    correction = rd(np.inf) if fp == 0 else rd(rd(abs(cd(f / fp))) / rd(abs(omega)))
    normalized = rd(np.inf) if slope_scale == 0 else rd(rd(abs(fp)) / slope_scale)
    return f, fp, np.asarray((eta_e, eta_q, eta_f, correction, normalized, slope_scale), dtype=rd), status


def diagonal_root(system, start, coordinate, assemble, measurements):
    rd, cd = system.real_dtype, system.complex_dtype
    omega = cd(start)
    status = first(0, not np.isfinite(omega), 14)
    steps, stopped = 0, False
    uint = np.uint32 if rd == np.float32 else np.uint64
    def bits(value):
        return np.asarray(value, dtype=rd).view(uint).item()
    with measurements.phase('root_control', coordinate=coordinate, arithmetic_precision=np.dtype(rd).name):
        while steps < 32 and status == 0 and not stopped:
            f, fp, _, code = element_state(system, omega, assemble(omega, loaded=True, derivative=True), coordinate, measurements)
            code = first(code, fp == 0, 11)
            # The original algorithm forms this candidate before selecting the
            # update; preserve arithmetic dtype even on a failing local state.
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                candidate = cd(omega - cd(f / fp))
            stopped = bits(candidate.real) == bits(omega.real) and bits(candidate.imag) == bits(omega.imag)
            if code == 0:
                omega = candidate
            status = code
            steps += 1
        f, slope, certificate, code = element_state(system, omega, assemble(omega, loaded=True, derivative=True), coordinate, measurements)
        status = first(status, code != 0, code)
        threshold = tau(system.n, rd)
        closed = bool(np.isfinite(omega) and omega.real > rd(0) and np.all(certificate[:4] <= threshold))
        status = first(status, not closed, 12)
        status = first(status, not certificate[4] > threshold, 13)
        status = first(status, omega.imag > rd(0), 12)
    return omega, slope, certificate, status, steps
