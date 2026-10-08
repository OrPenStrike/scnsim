"""Runtime sparse stamp assembly and existing local numerical certificates.

This is the JAX numerical authority for local conductor solves and coalescing.
Full-node matrices remain indexed values; only actual local blocks are dense.
CPU sparse factors, Schur state and Newton control live in sibling owners.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl

FAILURES = {
    1: ('direct_response_formation', 'reference_matrix'),
    2: ('direct_response_formation', 'series_rl'),
    3: ('eliminated_block_solve_failure', 'eliminated_block'),
    4: ('eliminated_block_solve_failure', 'derivative_eliminated_block'),
    5: ('direct_response_formation', 'source_solve'),
    6: ('direct_response_formation', 'source_admittance'),
    7: ('direct_response_formation', 'y_to_z'),
    8: ('direct_response_formation', 'y_to_s'),
    9: ('direct_response_formation', 'deembedding'),
    10: ('direct_response_formation', 'response_formation'),
    11: ('root_slope_unresolved', 'newton'),
    12: ('numerical_resolution_unresolved', 'newton_certificate'),
    13: ('root_slope_unresolved', 'slope_certificate'),
    14: ('numerical_resolution_unresolved', 'newton'),
}


def tau(n, dtype):
    return jnp.asarray(256 * (n + 1), dtype=dtype) * jnp.finfo(dtype).eps


def maximum_abs(a):
    return jnp.max(jnp.abs(a))


def residual(a, x, b):
    num = maximum_abs(a @ x - b)
    den = maximum_abs(jnp.abs(a) @ jnp.abs(x) + jnp.abs(b))
    return jnp.where(jnp.isfinite(num) & jnp.isfinite(den),
                     jnp.where(den == 0, jnp.where(num == 0, 0.0, jnp.inf), num / den), jnp.inf)


def first(status, bad, code):
    return jnp.where((status == 0) & bad, jnp.int32(code), status)


def solve(a, b, n, code, status):
    x = jl.lu_solve(jl.lu_factor(a), b)
    eta = residual(a, x, b)
    bad = ~(jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b)) & jnp.isfinite(eta) & (eta <= tau(n, a.real.dtype)))
    return x, first(status, bad, code), eta


def assemble(data, omega, *, loaded, derivative):
    """Coalesce runtime indices, then take the contract's aggregate absolutes.

    Each series group's pair map first coalesces within a complete section.
    Only then are section absolutes added to the full-node bound. Padding has
    zero incidence weights and never changes active matrix dimension/certificates.
    """
    pattern, base, load, groups = data
    c_map, c_values, k_map, k_values, g_map, g_values = base
    zeros = jnp.zeros(pattern.shape, dtype=omega.dtype)
    C = zeros.at[c_map].add(c_values)
    K = zeros.at[k_map].add(k_values)
    G = zeros.at[g_map].add(g_values)
    status = jnp.int32(0)
    if loaded and load[0].shape[0]:
        R, M, weights, columns, pair_map = load
        inverse, status, _ = solve(R.astype(omega.dtype), jnp.diag(M).astype(omega.dtype), R.shape[0], 1, status)
        values = weights[:, None] * inverse[columns[:, None], columns[None, :]] * weights[None, :]
        G = G.at[pair_map.ravel()].add(values.ravel())
    imaginary = jnp.asarray(1j, dtype=omega.dtype)
    two = jnp.asarray(2, dtype=omega.real.dtype)
    Q = K - omega * omega * C - imaginary * omega * G
    Qp = -two * omega * C - imaginary * G
    bound = jnp.abs(K) + (omega.real * omega.real + omega.imag * omega.imag) * jnp.abs(C) + jnp.abs(omega) * jnp.abs(G)
    for R, L, weights, columns, pair_map, global_map in groups:
        width = R.shape[-1]
        eye = jnp.eye(width, dtype=omega.dtype)
        inverse, codes, _ = jax.vmap(lambda r, l: solve(r - imaginary * omega * l, eye, width, 2, jnp.int32(0)))(R, L)
        status = first(status, jnp.any(codes != 0), 2)
        section = jnp.arange(weights.shape[0])[:, None, None]
        coefficient = inverse[section, columns[:, :, None], columns[:, None, :]]
        values = -imaginary * omega * weights[:, :, None] * coefficient * weights[:, None, :]
        complete = jnp.zeros(global_map.shape, dtype=omega.dtype).at[pair_map.ravel()].add(values.ravel())
        Q = Q.at[global_map].add(complete)
        if derivative:
            inverse_prime = -imaginary * inverse + omega * (inverse @ L @ inverse)
            coefficient_prime = inverse_prime[section, columns[:, :, None], columns[:, None, :]]
            prime_values = weights[:, :, None] * coefficient_prime * weights[:, None, :]
            complete_prime = jnp.zeros(global_map.shape, dtype=omega.dtype).at[pair_map.ravel()].add(prime_values.ravel())
            Qp = Qp.at[global_map].add(complete_prime)
            bound = bound.at[global_map].add(jnp.abs(complete))
    return Q, Qp, bound, status
