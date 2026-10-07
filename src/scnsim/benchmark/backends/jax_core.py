"""Dtype-preserving JAX kernels and complete local diagonal Newton loops.

Physical lowering, continuation and CMA stay in the Python task. Every matrix,
index and start is a runtime argument. Complex symmetry always uses transpose.
Failure codes carry the first applicable existing numerical boundary through
compiled loops; host adapters attach candidate context after synchronization.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
from jax import lax


# Codes identify existing stages, not additional numerical acceptance rules.
FAILURES = {
    1: ("direct_response_formation", "reference_matrix"),
    2: ("direct_response_formation", "series_rl"),
    3: ("eliminated_block_solve_failure", "eliminated_block"),
    4: ("eliminated_block_solve_failure", "derivative_eliminated_block"),
    5: ("direct_response_formation", "source_solve"),
    6: ("direct_response_formation", "source_admittance"),
    7: ("direct_response_formation", "y_to_z"),
    8: ("direct_response_formation", "y_to_s"),
    9: ("direct_response_formation", "deembedding"),
    10: ("direct_response_formation", "response_formation"),
    11: ("root_slope_unresolved", "newton"),
    12: ("numerical_resolution_unresolved", "newton_certificate"),
    13: ("root_slope_unresolved", "slope_certificate"),
    14: ("numerical_resolution_unresolved", "newton"),
}


def tau(n, dtype):
    return 256.0 * (n + 1) * jnp.finfo(dtype).eps


def maximum_abs(a):
    # Julia norm(A, Inf) is entrywise maximum, including matrix inputs.
    return jnp.max(jnp.abs(a))


def residual(a, x, b):
    num = maximum_abs(a @ x - b)
    den = maximum_abs(jnp.abs(a) @ jnp.abs(x) + jnp.abs(b))
    return jnp.where(jnp.isfinite(num) & jnp.isfinite(den),
                     jnp.where(den == 0, jnp.where(num == 0, 0.0, jnp.inf), num / den), jnp.inf)


def first(status, bad, code):
    return jnp.where((status == 0) & bad, jnp.int32(code), status)


def solve(a, b, n, code, status, factor=None):
    factor = jl.lu_factor(a) if factor is None else factor
    x = jl.lu_solve(factor, b)
    eta = residual(a, x, b)
    bad = ~(jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b)) & jnp.isfinite(eta) & (eta <= tau(n, a.real.dtype)))
    return x, first(status, bad, code), eta


def operator(data, omega, loaded, *, derivative):
    C, K, G, B, R, M, groups, *_ = data
    n = C.shape[0]
    status = jnp.int32(0)
    if loaded and R.shape[0]:
        load, status, _ = solve(R.astype(omega.dtype), jnp.diag(M).astype(omega.dtype), R.shape[0], 1, status)
        G = G + B @ load @ B.T
    Q = K - omega**2 * C - 1j * omega * G
    Qp = -2 * omega * C - 1j * G
    bound = jnp.abs(K) + (omega.real**2 + omega.imag**2) * jnp.abs(C) + jnp.abs(omega) * jnp.abs(G)
    for incidence, resistance, inductance in groups:
        width = resistance.shape[0]
        inv, status, _ = solve(resistance - 1j * omega * inductance, jnp.eye(width, dtype=omega.dtype), width, 2, status)
        for k in range(incidence.shape[0]):
            A = incidence[k]
            contribution = -1j * omega * (A @ inv @ A.T)
            Q = Q + contribution
            if derivative:
                Qp = Qp + A @ (-1j * inv + omega * (inv @ inductance @ inv)) @ A.T
                bound = bound + jnp.abs(contribution)
    return Q, Qp, bound, status


def network(data, omega, family):
    _, _, _, B, _, _, _, _, _, Bk, Rk, Dk, Go = data
    n, p = B.shape[0], Rk.shape[0]
    Q, _, _, status = operator(data, omega, False, derivative=False)
    H = Q / (-1j * omega) + B @ Go @ B.T
    Rinv, status, _ = solve(Rk.astype(omega.dtype), jnp.eye(p, dtype=omega.dtype), p, 1, status)
    W = H + Bk @ Rinv @ Bk.T
    factor = jl.lu_factor(W)
    X, status, _ = solve(W, Bk.astype(omega.dtype), n, 5, status, factor)
    Zsrc = Bk.T @ X
    Ysrc, status, _ = solve(Zsrc, jnp.eye(p, dtype=omega.dtype), p, 6, status)
    Y = Ysrc - Rinv
    Z = jnp.zeros_like(Y)
    S = jnp.zeros_like(Y)
    if family in ("all", "Z"):
        Z, status, _ = solve(Y, jnp.eye(p, dtype=omega.dtype), p, 7, status)
    if family in ("all", "S"):
        P = jnp.eye(p, dtype=omega.dtype) + Dk @ Y @ Dk
        N = jnp.eye(p, dtype=omega.dtype) - Dk @ Y @ Dk
        S, status, _ = solve(P, N, p, 8, status)
        dfactor = jl.lu_factor(Dk.astype(omega.dtype))
        Dinv, status, _ = solve(Dk.astype(omega.dtype), jnp.eye(p, dtype=omega.dtype), p, 1, status, dfactor)
        voltage, status, _ = solve(W, 2 * Bk @ Dinv, n, 5, status, factor)
        source_s, status, _ = solve(Dk.astype(omega.dtype), Bk.T @ voltage, p, 9, status, dfactor)
        source_s = source_s - jnp.eye(p, dtype=omega.dtype)
        eta = maximum_abs(source_s - S) / (1 + maximum_abs(source_s) + maximum_abs(S))
        status = first(status, ~(jnp.isfinite(eta) & (eta <= tau(n, omega.real.dtype))), 9)
    status = first(status, ~(jnp.all(jnp.isfinite(Y)) & jnp.all(jnp.isfinite(S)) & jnp.all(jnp.isfinite(Z))), 10)
    return S, Y, Z, status


def selected_state(data, omega):
    indices, eliminated = data[7:9]
    Q, Qp, bound, status = operator(data, omega, True, derivative=True)
    rr = Q[jnp.ix_(indices, indices)]
    rrp = Qp[jnp.ix_(indices, indices)]
    if eliminated.shape[0]:
        ee = Q[jnp.ix_(eliminated, eliminated)]
        er = Q[jnp.ix_(eliminated, indices)]
        re = Q[jnp.ix_(indices, eliminated)]
        eep = Qp[jnp.ix_(eliminated, eliminated)]
        erp = Qp[jnp.ix_(eliminated, indices)]
        rep = Qp[jnp.ix_(indices, eliminated)]
        numerator = maximum_abs(ee - ee.T)
        denominator = maximum_abs(jnp.abs(ee) + jnp.abs(ee.T))
        asym = jnp.where(denominator == 0, jnp.where(numerator == 0, 0.0, jnp.inf), numerator / denominator)
        status = first(status, ~(jnp.isfinite(asym) & (asym <= tau(ee.shape[0], omega.real.dtype))), 3)
        factor = jl.lu_factor(ee)
        X, status, ex = solve(ee, er, ee.shape[0], 3, status, factor)
        Xp, status, exp = solve(ee, erp - eep @ X, ee.shape[0], 4, status, factor)
        F = rr - re @ X
        Fp = rrp - rep @ X - re @ Xp
        eta_e = jnp.maximum(ex, exp)
    else:
        F, Fp, eta_e = rr, rrp, 0.0
        X = jnp.empty((0, indices.shape[0]), dtype=omega.dtype)
        Xp = jnp.empty_like(X)
    return F, Fp, Q, Qp, bound, X, Xp, eta_e, status


def retained_response(data, omega, family):
    F, _, _, _, _, _, _, _, status = selected_state(data, omega)
    Y = F / (-1j * omega)
    Z = jnp.zeros_like(Y)
    if family == "Z":
        Z, status, _ = solve(Y, jnp.eye(Y.shape[0], dtype=omega.dtype), Y.shape[0], 7, status)
    return jnp.zeros_like(Y), Y, Z, status


def element_state(data, omega, coordinate):
    indices, eliminated = data[7:9]
    F, Fp, Q, Qp, bound, X, Xp, eta_e, status = selected_state(data, omega)
    n = Q.shape[0]
    row = indices[coordinate]
    x = jnp.zeros(n, dtype=omega.dtype).at[row].set(1)
    slope_scale = jnp.abs(Qp[row, row])
    if eliminated.shape[0]:
        x = x.at[eliminated].set(-X[:, coordinate])
        slope_scale += jnp.sum(jnp.abs(Qp[row, eliminated]) * jnp.abs(X[:, coordinate])) + jnp.sum(jnp.abs(Q[row, eliminated]) * jnp.abs(Xp[:, coordinate]))
    checked = jnp.concatenate((jnp.reshape(row, (1,)), eliminated))
    qx = Q @ x
    bx = bound @ jnp.abs(x)
    num, den = maximum_abs(qx[checked]), maximum_abs(bx[checked])
    eta_q = jnp.where(den == 0, jnp.where(num == 0, 0.0, jnp.inf), num / den)
    f, fp = F[coordinate, coordinate], Fp[coordinate, coordinate]
    eta_f = jnp.where(bx[row] == 0, jnp.where(f == 0, 0.0, jnp.inf), jnp.abs(f) / bx[row])
    correction = jnp.where(fp == 0, jnp.inf, jnp.abs(f / fp) / jnp.abs(omega))
    normalized = jnp.where(slope_scale == 0, jnp.inf, jnp.abs(fp) / slope_scale)
    return f, fp, jnp.asarray((eta_e, eta_q, eta_f, correction, normalized, slope_scale)), status


def diagonal_root(data, start, coordinate):
    status = first(jnp.int32(0), ~jnp.isfinite(start), 14)
    def condition(state):
        step, _, code, stopped = state
        return (step < 32) & (code == 0) & ~stopped
    def advance(state):
        step, omega, _, _ = state
        f, fp, _, code = element_state(data, omega, coordinate)
        code = first(code, fp == 0, 11)
        candidate = omega - f / fp
        bits = jnp.uint32 if omega.real.dtype == jnp.float32 else jnp.uint64
        same = (lax.bitcast_convert_type(candidate.real, bits) == lax.bitcast_convert_type(omega.real, bits)) & (lax.bitcast_convert_type(candidate.imag, bits) == lax.bitcast_convert_type(omega.imag, bits))
        return step + 1, jnp.where(code == 0, candidate, omega), code, same
    steps, omega, status, _ = lax.while_loop(condition, advance, (jnp.int32(0), start, status, jnp.bool_(False)))
    _, slope, certificate, code = element_state(data, omega, coordinate)
    status = jnp.where(status == 0, code, status)
    threshold = tau(data[0].shape[0], omega.real.dtype)
    closed = jnp.isfinite(omega) & (omega.real > 0) & jnp.all(certificate[:4] <= threshold)
    status = first(status, ~closed, 12)
    status = first(status, ~(certificate[4] > threshold), 13)
    status = first(status, omega.imag > 0, 12)
    return omega, slope, certificate, status, steps
