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


def inductive_coefficients(groups):
    """Real-dtype L inverse once per realized exact-zero-R section.

    Other sections use an identity operand here and retain their original
    complex series-RL solve at every frequency. Numerical failures remain
    series_rl data and are consumed only when the nonzero-frequency identity
    is used; preparation does not reject the original omega=0 path.
    """
    result = []
    for R, L, weights, columns, pair_map, global_map, pure, constant, codes in groups:
        eye = jnp.eye(L.shape[-1], dtype=L.dtype)
        def local(l, use):
            a = jnp.where(use, l, eye)
            if L.shape[-1] > 2:
                inverse, status, _ = solve(a, eye, L.shape[-1], 2, jnp.int32(0))
                return inverse, status
            if L.shape[-1] == 1:
                inverse = jnp.ones_like(a) / a[0, 0]
            else:
                # Entrywise exponents avoid erasing a small diagonal through
                # global normalization. Product mantissas remain in real dtype;
                # integer exponents carry range, not additional precision.
                mantissa, exponent = jnp.frexp(a)
                ad = mantissa[0, 0] * mantissa[1, 1]
                bc = mantissa[0, 1] * mantissa[1, 0]
                ad_exponent = exponent[0, 0] + exponent[1, 1]
                bc_exponent = exponent[0, 1] + exponent[1, 0]
                # A zero term has no meaningful exponent and must not force
                # underflow of the other, possibly tiny, nonzero product.
                common = jnp.where(ad == 0, bc_exponent,
                                   jnp.where(bc == 0, ad_exponent,
                                             jnp.maximum(ad_exponent, bc_exponent)))
                aligned = (jnp.ldexp(ad, ad_exponent - common)
                           - jnp.ldexp(bc, bc_exponent - common))
                determinant_mantissa, shift = jnp.frexp(aligned)
                determinant_exponent = common + shift
                adjugate_mantissa = jnp.stack((
                    jnp.stack((mantissa[1, 1], -mantissa[0, 1])),
                    jnp.stack((-mantissa[1, 0], mantissa[0, 0]))))
                adjugate_exponent = jnp.stack((
                    jnp.stack((exponent[1, 1], exponent[0, 1])),
                    jnp.stack((exponent[1, 0], exponent[0, 0]))))
                # Distinct -b/-c entries share only the scalar determinant;
                # exact input reciprocity is evaluated identically, not imposed.
                inverse = jnp.ldexp(adjugate_mantissa / determinant_mantissa,
                                    adjugate_exponent - determinant_exponent)
            # Same local equation/certificate as solve(); singular arithmetic
            # flows to its existing series_rl failure, with no new cutoff.
            eta = residual(a, inverse, eye)
            bad = ~(jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(eye))
                    & jnp.isfinite(eta) & (eta <= tau(L.shape[-1], a.dtype)))
            return inverse, first(jnp.int32(0), bad, 2)
        inverse, status = jax.vmap(local)(L, pure)
        result.append((inverse, status))
    return tuple(result)


def _series_coefficients(R, L, pure, constant, constant_codes, omega, derivative):
    imaginary = jnp.asarray(1j, dtype=omega.dtype)
    width = R.shape[-1]
    eye = jnp.eye(width, dtype=omega.dtype)
    def local(r, l, use, fixed, fixed_code):
        def inductive(_):
            return fixed.astype(omega.dtype), jnp.zeros_like(eye), fixed_code
        def generic(_):
            inverse, code, _ = solve(r - imaginary * omega * l, eye, width, 2, jnp.int32(0))
            prime = (-imaginary * inverse + omega * (inverse @ l @ inverse)) if derivative else jnp.zeros_like(eye)
            return inverse, prime, code
        return jax.lax.cond(use & jnp.isfinite(omega) & (omega != 0), inductive, generic, operand=None)
    coefficients, primes, codes = jax.vmap(local)(R, L, pure, constant, constant_codes)
    return coefficients, primes, codes


def _assemble_standard(data, omega, *, loaded, derivative):
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
    for R, L, weights, columns, pair_map, global_map, pure, constant, constant_codes in groups:
        coefficients, primes, codes = _series_coefficients(
            R, L, pure, constant, constant_codes, omega, derivative)
        status = first(status, jnp.any(codes != 0), 2)
        section = jnp.arange(weights.shape[0])[:, None, None]
        coefficient = coefficients[section, columns[:, :, None], columns[:, None, :]]
        # Keep the original multiplication order for every generic section.
        generic_values = -imaginary * omega * weights[:, :, None] * coefficient * weights[:, None, :]
        constant_values = weights[:, :, None] * coefficient * weights[:, None, :]
        values = jnp.where((pure & jnp.isfinite(omega) & (omega != 0))[:, None, None], constant_values, generic_values)
        complete = jnp.zeros(global_map.shape, dtype=omega.dtype).at[pair_map.ravel()].add(values.ravel())
        Q = Q.at[global_map].add(complete)
        if derivative:
            coefficient_prime = primes[section, columns[:, :, None], columns[:, None, :]]
            prime_values = weights[:, :, None] * coefficient_prime * weights[:, None, :]
            complete_prime = jnp.zeros(global_map.shape, dtype=omega.dtype).at[pair_map.ravel()].add(prime_values.ravel())
            Qp = Qp.at[global_map].add(complete_prime)
            bound = bound.at[global_map].add(jnp.abs(complete))
    return Q, Qp, bound, status


def _assemble_compensated(data, omega, *, loaded, derivative):
    """Retained-Z-only indexed expansion, after selected-dtype input casts.

    Local inverse coefficients and their status retain C2 authority. Subsequent
    products and coalescing keep same-dtype arithmetic residues; no symmetry
    projection, alternate precision or full-node dense matrix is introduced.
    """
    from . import compensated as cp
    barrier = jax.lax.optimization_barrier
    kw = dict(xp=jnp, barrier=barrier)
    def pair(value):
        return cp.from_value(jnp.asarray(value, dtype=omega.dtype), xp=jnp)
    def scatter(values, indices, size):
        hi, lo = (v.ravel() for v in values)
        initial = (jnp.zeros((size,), dtype=omega.dtype), jnp.zeros((size,), dtype=omega.dtype))
        indices = indices.ravel()
        if hi.shape[0] == 0:
            return initial
        def stamp(i, carry):
            at = indices[i]
            updated = cp.add((carry[0][at], carry[1][at]), (hi[i], lo[i]), **kw)
            return carry[0].at[at].set(updated[0]), carry[1].at[at].set(updated[1])
        return jax.lax.fori_loop(0, hi.shape[0], stamp, initial)
    pattern, base, load, groups = data
    c_map, c_values, k_map, k_values, g_map, g_values = base
    C = scatter(pair(c_values), c_map, pattern.shape[0])
    K = scatter(pair(k_values), k_map, pattern.shape[0])
    G = scatter(pair(g_values), g_map, pattern.shape[0])
    status = jnp.int32(0)
    if loaded and load[0].shape[0]:
        R, M, weights, columns, pair_map = load
        inverse, status, _ = solve(R.astype(omega.dtype), jnp.diag(M).astype(omega.dtype), R.shape[0], 1, status)
        coefficient = inverse[columns[:, None], columns[None, :]]
        values = cp.multiply(cp.multiply(pair(weights[:, None]), pair(coefficient), **kw), pair(weights[None, :]), **kw)
        G = cp.add(G, scatter(values, pair_map, pattern.shape[0]), **kw)
    iw = cp.multiply(pair(1j), pair(omega), **kw)
    Q = cp.add(cp.add(K, cp.negate(cp.multiply(cp.multiply(pair(omega), pair(omega), **kw), C, **kw)), **kw),
               cp.negate(cp.multiply(iw, G, **kw)), **kw)
    Qp = cp.add(cp.negate(cp.multiply(cp.multiply(pair(2), pair(omega), **kw), C, **kw)),
                cp.negate(cp.multiply(pair(1j), G, **kw)), **kw)
    # Keep the original physical aggregate bound formula and selected dtype.
    projected_C, projected_K, projected_G = (cp.project(value, **kw) for value in (C, K, G))
    bound = (jnp.abs(projected_K) + (omega.real*omega.real + omega.imag*omega.imag)*jnp.abs(projected_C)
             + jnp.abs(omega)*jnp.abs(projected_G))
    for R, L, weights, columns, pair_map, global_map, pure, constant, constant_codes in groups:
        coefficients, primes, codes = _series_coefficients(R, L, pure, constant, constant_codes, omega, derivative)
        status = first(status, jnp.any(codes != 0), 2)
        section = jnp.arange(weights.shape[0])[:, None, None]
        coefficient = coefficients[section, columns[:, :, None], columns[:, None, :]]
        incidence = pair(weights[:, :, None])
        use = (pure & jnp.isfinite(omega) & (omega != 0))[:, None, None]
        generic = cp.multiply(cp.multiply(cp.negate(iw), incidence, **kw), pair(coefficient), **kw)
        fixed = cp.multiply(incidence, pair(coefficient), **kw)
        selected = tuple(jnp.where(use, a, b) for a, b in zip(fixed, generic))
        values = cp.multiply(selected, pair(weights[:, None, :]), **kw)
        complete = scatter(values, pair_map, global_map.shape[0])
        Q = cp.add(Q, scatter(complete, global_map, pattern.shape[0]), **kw)
        if derivative:
            coefficient_prime = primes[section, columns[:, :, None], columns[:, None, :]]
            prime_values = cp.multiply(cp.multiply(incidence, pair(coefficient_prime), **kw), pair(weights[:, None, :]), **kw)
            Qp = cp.add(Qp, scatter(scatter(prime_values, pair_map, global_map.shape[0]), global_map, pattern.shape[0]), **kw)
            bound = bound.at[global_map].add(jnp.abs(cp.project(complete, **kw)))
    return Q[0], Qp[0], bound, status, Q[1], Qp[1]


def assemble(data, omega, *, loaded, derivative, compensated=False):
    # Static request scope, never a numerical failure-triggered alternate path.
    if compensated:
        return _assemble_compensated(data, omega, loaded=loaded, derivative=derivative)
    return _assemble_standard(data, omega, loaded=loaded, derivative=derivative)
