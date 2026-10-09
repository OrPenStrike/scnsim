"""Selected-dtype compensated arithmetic for sparse solve certificates.

The JAX numerical owner supplies ``xp`` and, when needed, an optimization
barrier. Sparse multiplication and certificate reduction are host operations;
this module does not import JAX or choose numerical failure policy.
"""
from __future__ import annotations

from typing import Any, TypeAlias


Pair: TypeAlias = tuple[Any, Any]


def _rounded(value: Any, barrier: Any | None) -> Any:
    return value if barrier is None else barrier(value)


def _complex(value: Any) -> bool:
    return value.dtype.kind == "c"


def _pack_complex(real: Pair, imag: Pair, dtype: Any, *, xp: Any,
                  barrier: Any | None) -> Pair:
    unit = xp.asarray(1j, dtype=dtype)
    real_hi = xp.asarray(real[0], dtype=dtype)
    real_lo = xp.asarray(real[1], dtype=dtype)
    imag_hi = xp.asarray(imag[0], dtype=dtype)
    imag_lo = xp.asarray(imag[1], dtype=dtype)
    hi = _rounded(real_hi + _rounded(unit * imag_hi, barrier), barrier)
    lo = _rounded(real_lo + _rounded(unit * imag_lo, barrier), barrier)
    return hi, lo


def _two_sum_real(a: Any, b: Any, *, xp: Any,
                  barrier: Any | None) -> Pair:
    s = _rounded(a + b, barrier)
    b_virtual = _rounded(s - a, barrier)
    a_virtual = _rounded(s - b_virtual, barrier)
    b_roundoff = _rounded(b - b_virtual, barrier)
    a_roundoff = _rounded(a - a_virtual, barrier)
    error = _rounded(a_roundoff + b_roundoff, barrier)
    return s, error


def two_sum(a: Any, b: Any, *, xp: Any,
            barrier: Any | None = None) -> Pair:
    """Return a rounded sum and its selected-dtype addition residual."""
    a, b = xp.asarray(a), xp.asarray(b)
    if _complex(a):
        real = _two_sum_real(xp.real(a), xp.real(b), xp=xp, barrier=barrier)
        imag = _two_sum_real(xp.imag(a), xp.imag(b), xp=xp, barrier=barrier)
        return _pack_complex(real, imag, a.dtype, xp=xp, barrier=barrier)
    return _two_sum_real(a, b, xp=xp, barrier=barrier)


def _split_real(value: Any, *, xp: Any,
                barrier: Any | None) -> tuple[Any, Any, Any]:
    mantissa, exponent = xp.frexp(value)
    fraction_bits = int(xp.finfo(value.dtype).nmant)
    splitter = xp.asarray(2 ** ((fraction_bits + 2) // 2) + 1,
                          dtype=value.dtype)
    scaled = _rounded(splitter * mantissa, barrier)
    high = _rounded(scaled - _rounded(scaled - mantissa, barrier), barrier)
    low = _rounded(mantissa - high, barrier)
    return high, low, exponent


def _two_product_real(a: Any, b: Any, *, xp: Any,
                      barrier: Any | None) -> Pair:
    a_mantissa, a_exponent = xp.frexp(a)
    b_mantissa, b_exponent = xp.frexp(b)
    a_hi, a_lo, _ = _split_real(a_mantissa, xp=xp, barrier=barrier)
    b_hi, b_lo, _ = _split_real(b_mantissa, xp=xp, barrier=barrier)

    product = _rounded(a_mantissa * b_mantissa, barrier)
    hi_product = _rounded(a_hi * b_hi, barrier)
    error_1 = _rounded(product - hi_product, barrier)
    lo_hi_product = _rounded(a_lo * b_hi, barrier)
    error_2 = _rounded(error_1 - lo_hi_product, barrier)
    hi_lo_product = _rounded(a_hi * b_lo, barrier)
    error_3 = _rounded(error_2 - hi_lo_product, barrier)
    lo_product = _rounded(a_lo * b_lo, barrier)
    error = _rounded(lo_product - error_3, barrier)

    exponent = a_exponent + b_exponent
    high = _rounded(xp.ldexp(product, exponent), barrier)
    low = _rounded(xp.ldexp(error, exponent), barrier)
    return high, low


def _real_pair(value: Pair, *, xp: Any) -> Pair:
    return xp.real(value[0]), xp.real(value[1])


def _imag_pair(value: Pair, *, xp: Any) -> Pair:
    return xp.imag(value[0]), xp.imag(value[1])


def _add_real(a: Pair, b: Pair, *, xp: Any,
              barrier: Any | None) -> Pair:
    high, high_error = _two_sum_real(a[0], b[0], xp=xp, barrier=barrier)
    low, low_error = _two_sum_real(a[1], b[1], xp=xp, barrier=barrier)
    middle, middle_error = _two_sum_real(high_error, low, xp=xp,
                                          barrier=barrier)
    high, high_tail = _two_sum_real(high, middle, xp=xp, barrier=barrier)
    tail = _rounded(_rounded(low_error + middle_error, barrier) + high_tail,
                    barrier)
    high, low = _two_sum_real(high, tail, xp=xp, barrier=barrier)
    return high, low


def add(a: Pair, b: Pair, *, xp: Any,
        barrier: Any | None = None) -> Pair:
    """Add two two-component values while retaining their low components."""
    if _complex(a[0]):
        real = _add_real(_real_pair(a, xp=xp), _real_pair(b, xp=xp),
                         xp=xp, barrier=barrier)
        imag = _add_real(_imag_pair(a, xp=xp), _imag_pair(b, xp=xp),
                         xp=xp, barrier=barrier)
        return _pack_complex(real, imag, a[0].dtype, xp=xp, barrier=barrier)
    return _add_real(a, b, xp=xp, barrier=barrier)


def from_value(value: Any, *, xp: Any) -> Pair:
    """Represent an input as a high component and a zero low component."""
    high = xp.asarray(value)
    return high, xp.zeros_like(high)


def negate(a: Pair) -> Pair:
    """Negate both components without changing their dtype or shape."""
    return -a[0], -a[1]


def _multiply_real(a: Pair, b: Pair, *, xp: Any,
                   barrier: Any | None) -> Pair:
    high_product = _two_product_real(a[0], b[0], xp=xp, barrier=barrier)
    high_low = _two_product_real(a[0], b[1], xp=xp, barrier=barrier)
    low_high = _two_product_real(a[1], b[0], xp=xp, barrier=barrier)
    low_low = _two_product_real(a[1], b[1], xp=xp, barrier=barrier)
    return add(add(high_product, high_low, xp=xp, barrier=barrier),
               add(low_high, low_low, xp=xp, barrier=barrier),
               xp=xp, barrier=barrier)


def _multiply_complex(a: Pair, b: Pair, *, xp: Any,
                      barrier: Any | None) -> Pair:
    a_real, a_imag = _real_pair(a, xp=xp), _imag_pair(a, xp=xp)
    b_real, b_imag = _real_pair(b, xp=xp), _imag_pair(b, xp=xp)
    real = add(_multiply_real(a_real, b_real, xp=xp, barrier=barrier),
               negate(_multiply_real(a_imag, b_imag, xp=xp,
                                     barrier=barrier)),
               xp=xp, barrier=barrier)
    imag = add(_multiply_real(a_real, b_imag, xp=xp, barrier=barrier),
               _multiply_real(a_imag, b_real, xp=xp, barrier=barrier),
               xp=xp, barrier=barrier)
    return _pack_complex(real, imag, a[0].dtype, xp=xp, barrier=barrier)


def two_product(a: Any, b: Any, *, xp: Any,
                barrier: Any | None = None) -> Pair:
    """Return a product and its residual using selected-dtype arithmetic."""
    a, b = xp.asarray(a), xp.asarray(b)
    if _complex(a):
        return _multiply_complex(from_value(a, xp=xp), from_value(b, xp=xp),
                                 xp=xp, barrier=barrier)
    return _two_product_real(a, b, xp=xp, barrier=barrier)


def multiply(a: Pair, b: Pair, *, xp: Any,
             barrier: Any | None = None) -> Pair:
    """Multiply compensated values; complex products use real components."""
    if _complex(a[0]):
        return _multiply_complex(a, b, xp=xp, barrier=barrier)
    return _multiply_real(a, b, xp=xp, barrier=barrier)


def divide(a: Pair, b: Pair, *, xp: Any,
           barrier: Any | None = None) -> Pair:
    """Form a quotient with two residual-correction steps in the same dtype."""
    first = _rounded(a[0] / b[0], barrier)
    quotient = from_value(first, xp=xp)
    remainder = add(a, negate(multiply(quotient, b, xp=xp, barrier=barrier)),
                    xp=xp, barrier=barrier)
    second = _rounded(remainder[0] / b[0], barrier)
    quotient = add(quotient, from_value(second, xp=xp), xp=xp,
                   barrier=barrier)
    remainder = add(remainder, negate(multiply(from_value(second, xp=xp), b,
                                               xp=xp, barrier=barrier)),
                    xp=xp, barrier=barrier)
    third = _rounded(remainder[0] / b[0], barrier)
    return add(quotient, from_value(third, xp=xp), xp=xp, barrier=barrier)


def project(a: Pair, *, xp: Any,
            barrier: Any | None = None) -> Any:
    """Round a compensated value back to its selected array dtype."""
    return _rounded(a[0] + a[1], barrier)


def _row_product(matrix: Any, vector: Pair, *, row: int,
                 xp: Any, barrier: Any | None) -> Pair:
    shape = vector[0].shape[1:]
    dtype = vector[0].dtype
    total: Pair = (xp.zeros(shape, dtype=dtype), xp.zeros(shape, dtype=dtype))
    zero = xp.asarray(0, dtype=matrix.data.dtype)
    start, stop = matrix.indptr[row], matrix.indptr[row + 1]
    for offset in range(start, stop):
        column = matrix.indices[offset]
        coefficient = xp.asarray(matrix.data[offset], dtype=dtype)
        term = multiply((coefficient, zero),
                        (vector[0][column], vector[1][column]),
                        xp=xp, barrier=barrier)
        total = add(total, term, xp=xp, barrier=barrier)
    return total


def sparse_matmul(A_hi: Any, A_lo: Any, X: Pair) -> Pair:
    """Multiply caller-supplied high/low CSR matrices by a compensated RHS.

    The sparse entries are consumed directly in their stored row order; no
    dense matrix or dense sparse-operator conversion is formed. The numerical
    owner supplies CSR inputs here; factorization may retain its own CSC form.
    """
    import numpy as np

    rows = A_hi.shape[0]
    output_shape = (rows,) + X[0].shape[1:]
    high = np.zeros(output_shape, dtype=X[0].dtype)
    low = np.zeros(output_shape, dtype=X[0].dtype)
    for row in range(rows):
        hi_total = _row_product(A_hi, X, row=row, xp=np, barrier=None)
        lo_total = _row_product(A_lo, X, row=row, xp=np, barrier=None)
        combined = add(hi_total, lo_total, xp=np)
        high[row], low[row] = combined
    return high, low


def residual(A_hi: Any, A_lo: Any, X: Pair, B: Pair) -> Pair:
    """Return the compensated sparse residual ``(A_hi+A_lo)X-B``.

    ``A_hi`` and ``A_lo`` are the caller-supplied CSR matrices; this helper
    does not convert or validate sparse formats.
    """
    import numpy as np

    product = sparse_matmul(A_hi, A_lo, X)
    return add(product, negate(B), xp=np)


def solve_certificate(A_hi: Any, A_lo: Any, X: Pair, B: Pair, *,
                      real_dtype: Any) -> tuple[Any, Any, Any]:
    """Return the existing componentwise residual certificate and operands.

    Matrix inputs are the caller-supplied CSR operands. This helper does not
    perform format checks or assign a solver status.
    """
    import numpy as np

    remainder = residual(A_hi, A_lo, X, B)
    numerator = real_dtype(np.max(np.abs(project(remainder, xp=np))))

    # Addition is sparse and coalesces the represented coefficient before abs;
    # abs(A_hi)+abs(A_lo) would certify a different matrix.
    represented_A = A_hi + A_lo
    abs_A = represented_A.copy()
    abs_A.data = np.abs(abs_A.data)
    abs_X = np.abs(project(X, xp=np))
    abs_B = np.abs(project(B, xp=np))
    denominator = real_dtype(np.max(abs_A @ abs_X + abs_B))

    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        eta = real_dtype(np.inf)
    elif denominator == 0:
        eta = real_dtype(0 if numerator == 0 else np.inf)
    else:
        eta = real_dtype(numerator / denominator)
    return eta, numerator, denominator
