"""Scaled local cofactors and sparse nonsingular denominator certificates.

Local retained/numerator matrices may be singular. Their derivatives use
cofactors, never an inverse. Sparse denominators use native SuperLU with explicit
row scaling/permutation parity; streamed solves certify the ORIGINAL full matrix
with one aggregate norm, without ever constructing a global dense inverse.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .sparse_direct import maximum_abs, ratio, tau


class QuantityError(Exception):
    def __init__(self, kind, stage, detail, **facts):
        super().__init__(detail)
        self.kind, self.stage, self.detail, self.facts = kind, stage, detail, facts


def fail(kind, stage, detail, **facts):
    raise QuantityError(kind, stage, detail, **facts)


def shift_complex(value, shift, cd, rd):
    value = np.asarray(value, dtype=cd)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        return np.asarray(np.ldexp(value.real, shift), dtype=rd).astype(cd) + cd(1j) * np.asarray(np.ldexp(value.imag, shift), dtype=rd).astype(cd)


@dataclass(frozen=True)
class Scaled:
    mantissa: object
    exponent: int

    @classmethod
    def value(cls, value, cd, rd):
        value = cd(value)
        magnitude = rd(abs(value))
        if not np.isfinite(magnitude):
            fail('root_slope_unresolved', 'determinant_pivot', 'determinant entry is nonfinite')
        if magnitude == 0:
            return cls(cd(0), 0)
        _, exponent = np.frexp(magnitude)
        return cls(cd(shift_complex(value, 1-int(exponent), cd, rd)), int(exponent)-1)

    def multiply(self, other, cd, rd):
        product = Scaled.value(cd(self.mantissa * other.mantissa), cd, rd)
        return Scaled(product.mantissa, self.exponent + other.exponent + product.exponent) if product.mantissa != 0 else product

    def add(self, other, cd, rd):
        if self.mantissa == 0:
            return other
        if other.mantissa == 0:
            return self
        exponent = max(self.exponent, other.exponent)
        value = cd(shift_complex(self.mantissa, self.exponent-exponent, cd, rd) + shift_complex(other.mantissa, other.exponent-exponent, cd, rd))
        result = Scaled.value(value, cd, rd)
        return Scaled(result.mantissa, exponent + result.exponent) if result.mantissa != 0 else result

    def restore(self, cd, rd):
        if self.mantissa == 0:
            return cd(0)
        restored = cd(shift_complex(self.mantissa, self.exponent, cd, rd))
        if not np.isfinite(restored) or restored == 0:
            fail('root_slope_unresolved', 'determinant_scaling', 'determinant restoration is not representable', mantissa=self.mantissa, exponent=self.exponent)
        return restored

    def record(self):
        return {'mantissa': self.mantissa, 'exponent': self.exponent}


def local_determinant(matrix, cd, rd):
    work = np.array(matrix, dtype=cd, copy=True)
    determinant = Scaled(cd(1), 0)
    for column in range(len(work)):
        candidates = np.abs(work[column:, column])
        if not np.all(np.isfinite(candidates)):
            fail('root_slope_unresolved', 'determinant_pivot', 'local determinant pivot is nonfinite')
        row = column + int(np.argmax(candidates))
        if work[row, column] == 0:
            return Scaled(cd(0), 0)
        if row != column:
            work[[column, row]] = work[[row, column]]
            determinant = Scaled(cd(-determinant.mantissa), determinant.exponent)
        pivot = cd(work[column, column])
        determinant = determinant.multiply(Scaled.value(pivot, cd, rd), cd, rd)
        for next_row in range(column+1, len(work)):
            factor = cd(work[next_row, column] / pivot)
            work[next_row, column] = cd(0)
            work[next_row, column+1:] -= factor * work[column, column+1:]
    return determinant


def cofactors(matrix, prime, cd, rd):
    derivative = Scaled(cd(0), 0)
    terms = []
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            minor = np.delete(np.delete(matrix, row, axis=0), column, axis=1)
            cofactor = local_determinant(minor, cd, rd)
            if (row+column) % 2:
                cofactor = Scaled(cd(-cofactor.mantissa), cofactor.exponent)
            term = cofactor.multiply(Scaled.value(prime[row, column], cd, rd), cd, rd)
            derivative = derivative.add(term, cd, rd)
            terms.append((cofactor, prime[row, column]))
    return derivative, terms


def local_pair(matrix, prime, cd, rd):
    maxima = np.max(np.abs(matrix), axis=1) if len(matrix) else np.empty(0, dtype=rd)
    if not np.all(np.isfinite(maxima)):
        fail('root_slope_unresolved', 'determinant_scaling', 'local determinant row is nonfinite')
    _, powers = np.frexp(maxima)
    shifts = np.where(maxima == 0, 0, 1-powers).astype(np.int64)
    scaled = np.asarray(shift_complex(matrix, shifts[:, None], cd, rd), dtype=cd)
    scaled_prime = np.asarray(shift_complex(prime, shifts[:, None], cd, rd), dtype=cd)
    if not (np.all(np.isfinite(scaled)) and np.all(np.isfinite(scaled_prime))):
        fail('root_slope_unresolved', 'determinant_scaling', 'scaled local pair is nonfinite')
    common = -int(np.sum(shifts))
    determinant = local_determinant(scaled, cd, rd)
    derivative, terms = cofactors(scaled, scaled_prime, cd, rd)
    d = Scaled(determinant.mantissa, determinant.exponent+common)
    dp = Scaled(derivative.mantissa, derivative.exponent+common) if derivative.mantissa != 0 else derivative
    return d.restore(cd, rd), dp.restore(cd, rd), scaled, scaled_prime, dict(
        determinant=d.record(), derivative=dp.record(), row_shifts=shifts.tolist()), terms


def positive_product(factors, rd, stage='transfer_numerator_scale'):
    mantissa, exponent, zero = rd(1), 0, False
    for value in factors:
        value = rd(value)
        if not np.isfinite(value) or value < 0:
            fail('numerical_resolution_unresolved', stage, 'nonfinite or negative determinant scale')
        if value == 0:
            zero = True
        elif not zero:
            fm, fe = np.frexp(value)
            mantissa, shift = np.frexp(rd(mantissa*fm))
            exponent += int(fe)+int(shift)
    return (rd(0), 0) if zero else (rd(mantissa), exponent)


def positive_sum(terms, rd):
    mantissa, exponent = rd(0), 0
    for value, power in terms:
        if value == 0:
            continue
        if mantissa == 0:
            mantissa, exponent = rd(value), int(power)
        else:
            if power > exponent:
                mantissa, exponent = rd(np.ldexp(mantissa, exponent-power)), int(power)
            else:
                value = rd(np.ldexp(value, power-exponent))
            mantissa = rd(np.nextafter(rd(mantissa+value), rd(np.inf)))
            mantissa, shift = np.frexp(mantissa)
            exponent += int(shift)
    return rd(mantissa), exponent


def scaled_ratio(value, scale, rd):
    value = rd(value)
    if scale[0] == 0:
        return (rd(0), 0) if value == 0 else (rd(np.inf), 0)
    if value == 0:
        return rd(0), 0
    vm, ve = np.frexp(value)
    mantissa, shift = np.frexp(rd(vm / scale[0]))
    return rd(mantissa), int(ve)-int(scale[1])+int(shift)


def ratio_le(value, threshold, rd):
    if value[0] == 0:
        return True
    if not np.isfinite(value[0]):
        return False
    tm, te = np.frexp(rd(threshold))
    return value[1] < int(te) or (value[1] == int(te) and value[0] <= tm)


def _parity(permutation):
    visited = set()
    sign = 1
    for start in range(len(permutation)):
        if start in visited:
            continue
        current, length = start, 0
        while current not in visited:
            visited.add(current)
            current = int(permutation[current])
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


class Denominator:
    """Native sparse denominator factor with original-equation certificates."""
    def __init__(self, matrix, prime, system, measurements):
        from scipy.sparse.linalg import splu
        self.matrix, self.prime = matrix.tocsc(), prime.tocsc()
        self.system, self.measurements = system, measurements
        cd, rd = system.complex_dtype, system.real_dtype
        if not (np.all(np.isfinite(matrix.data)) and np.all(np.isfinite(prime.data))):
            fail('numerical_resolution_unresolved', 'transfer_denominator', 'denominator pair is nonfinite')
        # Row maxima from sparse entries; missing/zero row has no-op scale.
        maxima = np.zeros(matrix.shape[0], dtype=rd)
        np.maximum.at(maxima, matrix.tocoo().row, np.abs(matrix.tocoo().data))
        _, powers = np.frexp(maxima)
        self.shifts = np.where(maxima == 0, 0, 1-powers).astype(np.int64)
        scaled = self.matrix.copy()
        scaled_coo = scaled.tocoo()
        scaled_coo.data = np.asarray(shift_complex(scaled_coo.data, self.shifts[scaled_coo.row], cd, rd), dtype=cd)
        with measurements.phase('sparse_factorization', stage='transfer_denominator', nodes=matrix.shape[0], nnz=matrix.nnz):
            try:
                self.factor = splu(scaled_coo.tocsc(), options={'Equil': False})
            except RuntimeError as error:
                if str(error) != 'Factor is exactly singular':
                    raise
                raise QuantityError('numerical_resolution_unresolved', 'transfer_denominator', 'required denominator is singular', native_error=str(error)) from error
        determinant = Scaled(cd(_parity(self.factor.perm_r)*_parity(self.factor.perm_c)), 0)
        for diagonal in self.factor.U.diagonal():
            determinant = determinant.multiply(Scaled.value(diagonal, cd, rd), cd, rd)
        self.scaled = Scaled(determinant.mantissa, determinant.exponent-int(np.sum(self.shifts)))
        self.value = self.scaled.restore(cd, rd)
        if self.value == 0:
            fail('numerical_resolution_unresolved', 'transfer_denominator', 'denominator is zero')
        self.facts = dict(determinant=self.scaled.record(), row_shifts=self.shifts.tolist(),
                          row_permutation=self.factor.perm_r.tolist(), column_permutation=self.factor.perm_c.tolist(),
                          nnz=matrix.nnz, L_nnz=self.factor.L.nnz, U_nnz=self.factor.U.nnz)

    def solve(self, rhs):
        cd, rd = self.system.complex_dtype, self.system.real_dtype
        scaled_rhs = np.asarray(shift_complex(rhs, self.shifts[:, None], cd, rd), dtype=cd)
        with self.measurements.phase('sparse_rhs_solve', stage='transfer_denominator', columns=rhs.shape[1]):
            result = self.factor.solve(scaled_rhs)
        return np.asarray(result, dtype=cd)

    def certificate_pair(self, *, verify=True):
        cd, rd = self.system.complex_dtype, self.system.real_dtype
        n = self.matrix.shape[0]
        inverse_num = inverse_den = derivative_num = derivative_den = rd(0)
        finite = True
        trace = cd(0)
        for start in range(0, n, 8):
            columns = np.arange(start, min(start+8, n))
            rhs = np.zeros((n, len(columns)), dtype=cd)
            rhs[columns, np.arange(len(columns))] = cd(1)
            inverse = self.solve(rhs)
            prime_rhs = self.prime[:, columns].toarray()
            derivative = self.solve(prime_rhs)
            with self.measurements.phase('sparse_residual_check', stage='transfer_denominator', columns=len(columns)):
                num = maximum_abs(self.matrix @ inverse-rhs, rd)
                den = maximum_abs(abs(self.matrix) @ np.abs(inverse)+np.abs(rhs), rd)
                pnum = maximum_abs(self.matrix @ derivative-prime_rhs, rd)
                pden = maximum_abs(abs(self.matrix) @ np.abs(derivative)+np.abs(prime_rhs), rd)
                finite = finite and bool(np.all(np.isfinite(inverse)) and np.all(np.isfinite(derivative))
                                         and np.all(np.isfinite((num, den, pnum, pden))))
                inverse_num, inverse_den = rd(np.maximum(inverse_num, num)), rd(np.maximum(inverse_den, den))
                derivative_num, derivative_den = rd(np.maximum(derivative_num, pnum)), rd(np.maximum(derivative_den, pden))
            trace = cd(trace + np.sum(derivative[columns, np.arange(len(columns))], dtype=cd))
        eta = ratio(inverse_num, inverse_den, rd)
        peta = ratio(derivative_num, derivative_den, rd)
        self.facts.update(identity_solve_numerator=inverse_num, identity_solve_denominator=inverse_den,
                          identity_solve_residual=eta, derivative_solve_residual=peta, trace=trace,
                          original_dimension=n, finite=finite)
        if verify:
            self.verify_certificate()
        derivative = self.scaled.multiply(Scaled.value(trace, cd, rd), cd, rd)
        return self.value, derivative.restore(cd, rd), self.facts

    def verify_certificate(self):
        rd = self.system.real_dtype
        n = self.facts['original_dimension']
        if not self.facts['finite'] or not (self.facts['identity_solve_residual'] <= tau(n, rd)
                                          and self.facts['derivative_solve_residual'] <= tau(n, rd)):
            fail('numerical_resolution_unresolved', 'transfer_denominator',
                 'complete denominator solve residual did not close', **self.facts)
