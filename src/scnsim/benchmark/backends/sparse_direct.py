"""Call-local CSC/SuperLU network and selected Schur numerical authority.

Descriptors retain structural zeros. Only local, Port, selected Schur and actual
RHS arrays become dense. Sparse factors never survive their numerical call.
Native program/resource failures escape; only SuperLU's exact singularity is
translated to the existing formation stage. Transpose is never conjugated.
"""
from __future__ import annotations
from contextlib import contextmanager
from hashlib import sha256
from time import perf_counter_ns
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


class Measurements:
    def __init__(self, trace=None, parent=None):
        self.trace, self.parent, self.rows = trace, parent, []

    @contextmanager
    def phase(self, kind, **details):
        start = perf_counter_ns()
        def record(status):
            end = perf_counter_ns()
            self.rows.append(dict(kind=kind, start_tick_ns=start, end_tick_ns=end, details=details, status=status))
            if self.trace is not None:
                self.trace.measure(kind, start_tick_ns=start, end_tick_ns=end,
                                   parent_span_id=self.parent, details=details, status=status)
        try:
            yield
        except BaseException as original:
            try:
                record('interrupted' if isinstance(original, (KeyboardInterrupt, SystemExit)) else 'failure')
            except BaseException as secondary:
                original.add_note(f'Numerical timing publication also failed: {secondary!r}')
            raise
        else:
            record('success')


def first(status, bad, code):
    return code if status == 0 and bad else status


def maximum_abs(a, real_dtype):
    if hasattr(a, 'data') and hasattr(a, 'tocsc'):
        return real_dtype(np.max(np.abs(a.data))) if a.nnz else real_dtype(0)
    return real_dtype(np.max(np.abs(a)))


def tau(n, real_dtype):
    return real_dtype(real_dtype(256 * (n + 1)) * np.finfo(real_dtype).eps)


def ratio(num, den, real_dtype):
    if not (np.isfinite(num) and np.isfinite(den)):
        return real_dtype(np.inf)
    if den == 0:
        return real_dtype(0 if num == 0 else np.inf)
    return real_dtype(num / den)


def residual(a, x, b, real_dtype):
    num = maximum_abs(a @ x - b, real_dtype)
    den = maximum_abs(abs(a) @ np.abs(x) + np.abs(b), real_dtype)
    return ratio(num, den, real_dtype)


def dense_solve(a, b, n, code, status, real_dtype, measurements, factor=None):
    with measurements.phase('sparse_network_conversion', stage=code):
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return np.zeros(b.shape, dtype=a.dtype), first(status, True, code), real_dtype(np.inf)
        factor = lu_factor(a, check_finite=False) if factor is None else factor
        x = lu_solve(factor, b, check_finite=False)
        eta = residual(a, x, b, real_dtype)
        bad = not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.isfinite(eta) and eta <= tau(n, real_dtype))
    return x, first(status, bad, code), eta


class SparseFactor:
    def __init__(self, matrix, code, status, system, measurements):
        self.matrix, self.code, self.status = matrix, code, status
        self.system, self.measurements, self.factor = system, measurements, None
        # Do not mask a previous local failure with a consequent native factor
        # failure caused by its invalid values.
        if status:
            return
        if not np.all(np.isfinite(matrix.data)):
            self.status = code
            return
        with measurements.phase('sparse_factorization', nodes=matrix.shape[0], stage=code, nnz=matrix.nnz):
            try:
                self.factor = splu(matrix)
            except RuntimeError as error:
                if str(error) != 'Factor is exactly singular':
                    raise
                self.status = code
                return
        system.factor_facts.append(dict(nodes=matrix.shape[0], nnz=matrix.nnz,
                                        L_nnz=self.factor.L.nnz, U_nnz=self.factor.U.nnz))

    def solve(self, rhs, code=None):
        code = self.code if code is None else code
        if self.factor is None:
            return np.zeros(rhs.shape, dtype=self.system.complex_dtype), self.status, self.system.real_dtype(np.inf)
        if not np.all(np.isfinite(rhs)):
            self.status = first(self.status, True, code)
            return np.zeros(rhs.shape, dtype=self.system.complex_dtype), self.status, self.system.real_dtype(np.inf)
        with self.measurements.phase('sparse_rhs_solve', stage=code, columns=rhs.shape[1]):
            x = self.factor.solve(np.asarray(rhs, dtype=self.system.complex_dtype))
        with self.measurements.phase('sparse_residual_check', stage=code, columns=rhs.shape[1]):
            eta = residual(self.matrix, x, rhs, self.system.real_dtype)
            bad = not (np.all(np.isfinite(self.matrix.data)) and np.all(np.isfinite(rhs)) and np.isfinite(eta)
                       and eta <= tau(self.matrix.shape[0], self.system.real_dtype))
        self.status = first(self.status, bad, code)
        return x, self.status, eta


def _keys(matrix, n):
    return np.asarray(matrix.rows, dtype=np.int64) * n + np.asarray(matrix.cols, dtype=np.int64)


def _pairs(matrix, n):
    return (np.asarray(matrix.rows, dtype=np.int64)[:, None] * n
            + np.asarray(matrix.rows, dtype=np.int64)[None, :]).ravel()


def _rhs(matrix, dtype):
    # This is an actual sparse-solve RHS, not a full-node system materialization.
    result = np.zeros(matrix.shape, dtype=dtype)
    np.add.at(result, (matrix.rows, matrix.cols), np.asarray(matrix.values, dtype=dtype))
    return result


class System:
    """One candidate/View's runtime sparse operands and index preparation."""
    def __init__(self, view, real_dtype, complex_dtype):
        self.view = view
        self.real_dtype, self.complex_dtype = real_dtype, complex_dtype
        model = view.model
        self.n = len(model.node_ids)
        self.factor_facts = []
        B, Bk = model.B, view.Bk
        bases = (model.C, model.K, model.G)
        section_keys = [_pairs(block.incidence, self.n) for block in model.series_rl]
        all_keys = [_keys(m, self.n) for m in bases] + section_keys + [_pairs(B, self.n)]
        if Bk is not None:
            all_keys.append(_pairs(Bk, self.n))
        pattern = np.unique(np.concatenate(all_keys))
        self.rows, self.cols = pattern // self.n, pattern % self.n
        self.pattern_sha256 = sha256(pattern.tobytes() + np.asarray((self.n, self.n), dtype=np.int64).tobytes()).hexdigest()
        def locations(keys):
            return np.asarray(np.searchsorted(pattern, keys), dtype=np.int32)
        base = tuple(part for matrix in bases for part in
                     (locations(_keys(matrix, self.n)), np.asarray(matrix.values, dtype=real_dtype)))
        load = (np.asarray(model.R, dtype=real_dtype), np.asarray(model.M, dtype=real_dtype),
                np.asarray(B.values, dtype=real_dtype), np.asarray(B.cols, dtype=np.int32),
                locations(_pairs(B, self.n)).reshape((len(B.values),) * 2))
        grouped = {}
        for block, keys in zip(model.series_rl, section_keys, strict=True):
            grouped.setdefault(block.resistance.shape[0], []).append((block, keys))
        groups = []
        for width, blocks in grouped.items():
            count = max(len(block.incidence.values) for block, _ in blocks)
            weights = np.zeros((len(blocks), count), dtype=real_dtype)
            columns = np.zeros((len(blocks), count), dtype=np.int32)
            pair_map = np.zeros((len(blocks), count, count), dtype=np.int32)
            global_maps, offset = [], 0
            for index, (block, keys) in enumerate(blocks):
                length = len(block.incidence.values)
                weights[index, :length] = block.incidence.values
                columns[index, :length] = block.incidence.cols
                unique = np.unique(keys)
                pair_map[index, :length, :length] = (offset + np.searchsorted(unique, keys)).reshape(length, length)
                global_maps.append(locations(unique))
                offset += len(unique)
            groups.append((np.asarray([block.resistance for block, _ in blocks], dtype=real_dtype),
                           np.asarray([block.inductance for block, _ in blocks], dtype=real_dtype),
                           weights, columns, pair_map, np.concatenate(global_maps)))
        self.payload = (np.zeros(len(pattern), dtype=real_dtype), base, load, tuple(groups))
        self.B = B
        self.Bk = Bk
        self.Bk_rhs = None if Bk is None else _rhs(Bk, complex_dtype)
        self.Rk = None if view.Rk is None else np.asarray(view.Rk, dtype=complex_dtype)
        self.Dk = None if view.Dk is None else np.asarray(view.Dk, dtype=complex_dtype)
        self.Go = None if view.Go is None else np.asarray(view.Go, dtype=complex_dtype)
        self.B_pair_map = locations(_pairs(B, self.n))
        self.Bk_pair_map = None if Bk is None else locations(_pairs(Bk, self.n))
        self.selected = np.asarray(view.selected_indices, dtype=np.intp)
        selected_set = set(view.selected_indices)
        self.eliminated = np.asarray([i for i in range(self.n) if i not in selected_set], dtype=np.intp)

    def csc(self, values, measurements):
        with measurements.phase('sparse_csc_build', nodes=self.n, nnz=len(values)):
            return coo_matrix((values, (self.rows, self.cols)), shape=(self.n, self.n)).tocsc()

    def stamp(self, matrix, coefficient, pair_map):
        weights = np.asarray(matrix.values, dtype=self.complex_dtype)
        columns = np.asarray(matrix.cols)
        values = weights[:, None] * coefficient[columns[:, None], columns[None, :]] * weights[None, :]
        result = np.zeros(len(self.rows), dtype=self.complex_dtype)
        np.add.at(result, pair_map, values.ravel())
        return result


def network(system, omega, assembled, family, measurements):
    Q, _, _, status = assembled
    status = int(status)
    if status:
        return None, status
    rd, cd = system.real_dtype, system.complex_dtype
    p = system.Rk.shape[0]
    eye = np.eye(p, dtype=cd)
    Rinv, status, _ = dense_solve(system.Rk, eye, p, 1, status, rd, measurements)
    if status:
        return None, status
    w_values = (Q / cd(-cd(1j) * omega)
                + system.stamp(system.B, system.Go, system.B_pair_map)
                + system.stamp(system.Bk, Rinv, system.Bk_pair_map))
    W = system.csc(w_values, measurements)
    factor = SparseFactor(W, 5, status, system, measurements)
    X, status, _ = factor.solve(system.Bk_rhs)
    if status:
        return None, status
    Zsrc = system.Bk_rhs.T @ X
    Ysrc, status, _ = dense_solve(Zsrc, eye, p, 6, status, rd, measurements)
    if status:
        return None, status
    Y = np.asarray(Ysrc - Rinv, dtype=cd)
    Z, S = np.zeros_like(Y), np.zeros_like(Y)
    if family in ('all', 'Z'):
        Z, status, _ = dense_solve(Y, eye, p, 7, status, rd, measurements)
        if status:
            return None, status
    if family in ('all', 'S'):
        P = eye + system.Dk @ Y @ system.Dk
        N = eye - system.Dk @ Y @ system.Dk
        S, status, _ = dense_solve(P, N, p, 8, status, rd, measurements)
        if status:
            return None, status
        dfactor = lu_factor(system.Dk, check_finite=False)
        Dinv, status, _ = dense_solve(system.Dk, eye, p, 1, status, rd, measurements, dfactor)
        voltage, source_code, _ = factor.solve(cd(2) * system.Bk_rhs @ Dinv)
        status = first(status, source_code != 0, source_code)
        source_s, status, _ = dense_solve(system.Dk, system.Bk_rhs.T @ voltage, p, 9, status, rd, measurements, dfactor)
        source_s = np.asarray(source_s - eye, dtype=cd)
        eta = rd(maximum_abs(source_s - S, rd) / rd(rd(1) + maximum_abs(source_s, rd) + maximum_abs(S, rd)))
        status = first(status, not (np.isfinite(eta) and eta <= tau(system.n, rd)), 9)
    status = first(status, not all(np.all(np.isfinite(a)) for a in (Y, S, Z)), 10)
    return (S, Y, Z), status


def selected_state(system, assembled, measurements):
    q, qp, bound, status = assembled
    status = int(status)
    if status:
        return None, status
    Q, Qp = system.csc(q, measurements), system.csc(qp, measurements)
    selected, eliminated = system.selected, system.eliminated
    # Only selected Schur blocks and actual selected RHS are dense.
    rr = Q[selected][:, selected].toarray()
    rrp = Qp[selected][:, selected].toarray()
    rd = system.real_dtype
    if len(eliminated):
        ee = Q[eliminated][:, eliminated].tocsc()
        er = Q[eliminated][:, selected].toarray()
        re = Q[selected][:, eliminated]
        eep = Qp[eliminated][:, eliminated]
        erp = Qp[eliminated][:, selected].toarray()
        rep = Qp[selected][:, eliminated]
        numerator = maximum_abs(ee - ee.T, rd)
        denominator = maximum_abs(abs(ee) + abs(ee.T), rd)
        asym = ratio(numerator, denominator, rd)
        status = first(status, not (np.isfinite(asym) and asym <= tau(len(eliminated), rd)), 3)
        factor = SparseFactor(ee, 3, status, system, measurements)
        X, status, ex = factor.solve(er)
        if status:
            return None, status
        Xp, status, exp = factor.solve(np.asarray(erp - eep @ X, dtype=system.complex_dtype), 4)
        F = rr - re @ X
        Fp = rrp - rep @ X - re @ Xp
        eta_e = rd(np.maximum(ex, exp))
    else:
        F, Fp, eta_e = rr, rrp, rd(0)
        X = np.empty((0, len(selected)), dtype=system.complex_dtype)
        Xp = np.empty_like(X)
    return (F, Fp, Q, Qp, system.csc(bound, measurements), X, Xp, eta_e), status


def retained_response(system, omega, assembled, family, measurements):
    state, status = selected_state(system, assembled, measurements)
    if state is None:
        return None, status
    F = state[0]
    Y = np.asarray(F / system.complex_dtype(-system.complex_dtype(1j) * omega), dtype=system.complex_dtype)
    Z = np.zeros_like(Y)
    if family == 'Z':
        Z, status, _ = dense_solve(Y, np.eye(len(Y), dtype=system.complex_dtype), len(Y), 7, status,
                                  system.real_dtype, measurements)
    return (np.zeros_like(Y), Y, Z), status
