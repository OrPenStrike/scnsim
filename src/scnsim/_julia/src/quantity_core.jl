# Included into the single SCNSimBackend module.
# Shared Direct quantity kernels and certified root calculations.

"""One authority for the complete selected View operator and derivative."""
function selected_operator_state(compiled::CompiledPrimitive, omega::ComplexF64, coordinates::Vector{String})
    indices = selected_coordinate_indices(compiled, coordinates)
    Q = operator_at(compiled, omega; loaded = true)
    Qp = operator_derivative_at(compiled, omega; loaded = true)
    eliminated = [index for index in eachindex(compiled.nodes) if index ∉ indices]
    if isempty(eliminated)
        X = zeros(ComplexF64, 0, length(indices)); Xp = similar(X)
        return (F = Q[indices, indices], Fp = Qp[indices, indices], Q = Q, Qp = Qp,
            X = X, Xp = Xp, indices = indices, eliminated = eliminated, eta_e = 0.0)
    end
    Qee, Qer = Q[eliminated, eliminated], Q[eliminated, indices]
    X = checked_solve(Qee, Qer, "eliminated_block_solve_failure", "eliminated_block", length(eliminated))
    Qpeer, Qpee = Qp[eliminated, indices], Qp[eliminated, eliminated]
    derivative_rhs = Qpeer - Qpee * X
    Xp = checked_solve(Qee, derivative_rhs, "eliminated_block_solve_failure", "derivative_eliminated_block", length(eliminated))
    F = Q[indices, indices] - Q[indices, eliminated] * X
    Fp = Qp[indices, indices] - Qp[indices, eliminated] * X - Q[indices, eliminated] * Xp
    return (F = F, Fp = Fp, Q = Q, Qp = Qp, X = X, Xp = Xp,
        indices = indices, eliminated = eliminated,
        eta_e = max(backward_residual(Qee, X, Qer), backward_residual(Qee, Xp, derivative_rhs)))
end

"""Exact loaded operator and derivative on an ordered retained coordinate set."""
function selected_operator(compiled::CompiledPrimitive, omega::ComplexF64, coordinates::Vector{String})
    state = selected_operator_state(compiled, omega, coordinates)
    return state.F, state.Fp
end

"""One ordered scalar projection of the already selected View; never a second reduction."""
function operator_element_state(compiled::CompiledPrimitive, omega::ComplexF64,
        coordinates::Vector{String}, row::Int, column::Int)
    state = selected_operator_state(compiled, omega, coordinates)
    1 <= row <= length(coordinates) && 1 <= column <= length(coordinates) ||
        fail("validation", "port_realizability", "root", "direct_quantity", "element coordinate is absent from the final View")
    i, j = state.indices[row], state.indices[column]
    f, fp = state.F[row, column], state.Fp[row, column]
    x = zeros(ComplexF64, length(compiled.nodes)); x[j] = 1.0 + 0.0im
    !isempty(state.eliminated) && (x[state.eliminated] .= -state.X[:, column])
    bound = operator_absolute_bound(compiled, omega; loaded = true) * abs.(x)
    checked_rows = vcat([i], state.eliminated)
    residual = norm((state.Q * x)[checked_rows], Inf)
    denominator = norm(bound[checked_rows], Inf)
    eta_q = denominator == 0.0 ? (residual == 0.0 ? 0.0 : Inf) : residual / denominator
    eta_f = bound[i] == 0.0 ? (abs(f) == 0.0 ? 0.0 : Inf) : abs(f) / bound[i]
    slope_scale = abs(state.Qp[i, j])
    if !isempty(state.eliminated)
        slope_scale += sum(abs.(state.Qp[i, state.eliminated]) .* abs.(state.X[:, column])) +
            sum(abs.(state.Q[i, state.eliminated]) .* abs.(state.Xp[:, column]))
    end
    return (f = f, fp = fp, eta_e = state.eta_e, eta_q = eta_q, eta_f = eta_f,
        correction = fp == 0.0 ? Inf : abs(f / fp) / abs(omega),
        normalized_slope = slope_scale == 0.0 ? Inf : abs(fp) / slope_scale,
        scale = slope_scale, closure = eta_f)
end

function complex_frequency_value(value)::ComplexF64
    item = plain(value)
    if get(item, "type", nothing) == "quantity_f64"
        return complex(quantity_value(item))
    elseif get(item, "type", nothing) == "complex_quantity_f64"
        return complex(f64_from_hex(item["real_si_f64"]), f64_from_hex(item["imag_si_f64"]))
    end
    fail("validation", "invalid_diagonal_root_hint", "anchor", "direct_quantity", "frequency anchor is malformed")
end

"""Determinant as a bounded complex mantissa and an exact binary exponent.

This is deliberately not `LinearAlgebra.det`: identity-v1 fixes largest
complex-absolute partial pivots and resolves an equal-magnitude tie by the
smallest current row.  The product is renormalized after every pivot so the
certificate never silently overflows or underflows before its final, checked
binary64 restoration.
"""
function determinant_mantissa_exponent(matrix::Matrix{ComplexF64})::Tuple{ComplexF64,Int}
    n = size(matrix, 1); n == size(matrix, 2) || error("determinant matrix is not square")
    n == 0 && return 1.0 + 0.0im, 0
    work = copy(matrix); mantissa = 1.0 + 0.0im; exponent = 0; parity = 1
    scale_entry(value::ComplexF64, shift::Int) = complex(ldexp(real(value), shift), ldexp(imag(value), shift))
    for column in 1:n
        pivot_row = column; pivot_abs = abs(work[column, column])
        isfinite(pivot_abs) || fail("execution", "root_slope_unresolved", "determinant_pivot", "direct_quantity", "determinant pivot is non-finite")
        for row in (column + 1):n
            candidate = abs(work[row, column]); isfinite(candidate) || fail("execution", "root_slope_unresolved", "determinant_pivot", "direct_quantity", "determinant pivot is non-finite")
            # Strict `>` leaves the first (therefore smallest) row on an
            # equal complex-absolute magnitude tie.
            if candidate > pivot_abs
                pivot_row, pivot_abs = row, candidate
            end
        end
        pivot_abs == 0.0 && return 0.0 + 0.0im, 0
        if pivot_row != column
            work[column, :], work[pivot_row, :] = copy(work[pivot_row, :]), copy(work[column, :])
            parity *= -1
        end
        pivot = work[column, column]
        _, pivot_exponent = frexp(abs(pivot))
        normalized_pivot = scale_entry(pivot, 1 - pivot_exponent)
        mantissa *= normalized_pivot; exponent += pivot_exponent - 1
        magnitude = abs(mantissa)
        _, exponent_m = frexp(magnitude)
        # A nonzero product has a finite mantissa; normalize it before the
        # next multiplication while retaining its exact power-of-two scale.
        magnitude > 0.0 && isfinite(magnitude) || fail("execution", "root_slope_unresolved", "determinant_pivot", "direct_quantity", "determinant mantissa became non-finite")
        mantissa = scale_entry(mantissa, 1 - exponent_m); exponent += exponent_m - 1
        for row in (column + 1):n
            factor = work[row, column] / pivot
            work[row, column] = 0.0 + 0.0im
            for next_column in (column + 1):n
                work[row, next_column] -= factor * work[column, next_column]
            end
        end
    end
    parity < 0 && (mantissa = -mantissa)
    return mantissa, exponent
end

function mantissa_exponent_value(mantissa::ComplexF64, exponent::Int)::ComplexF64
    mantissa == 0.0 + 0.0im && return mantissa
    restored = complex(ldexp(real(mantissa), exponent), ldexp(imag(mantissa), exponent))
    finite = isfinite(real(restored)) && isfinite(imag(restored))
    underflow = restored == 0.0 + 0.0im
    finite && !underflow || fail("execution", "root_slope_unresolved", "determinant_scaling", "direct_quantity", "mantissa/exponent determinant restoration is not representable")
    return restored
end

function determinant_value(matrix::Matrix{ComplexF64})::ComplexF64
    mantissa, exponent = determinant_mantissa_exponent(matrix)
    return mantissa_exponent_value(mantissa, exponent)
end

function cofactor_derivative(F::Matrix{ComplexF64}, Fp::Matrix{ComplexF64})::ComplexF64
    q = size(F, 1); q == size(F, 2) || error("determinant matrix is not square")
    q == 1 && return Fp[1, 1]
    value = 0.0 + 0.0im
    for row in 1:q, column in 1:q
        rows = [index for index in 1:q if index != row]; columns = [index for index in 1:q if index != column]
        cofactor = (-1)^(row + column) * determinant_value(F[rows, columns])
        value += cofactor * Fp[row, column]
    end
    return value
end

"""Fixed power-of-two row scaling for determinant Newton/certificates."""
function scaled_determinant_pair(F::Matrix{ComplexF64}, Fp::Matrix{ComplexF64})
    maxima = [maximum(abs.(view(F, row, :))) for row in axes(F, 1)]
    all(isfinite, maxima) ||
        fail("execution", "root_slope_unresolved", "determinant_scaling", "direct_quantity", "determinant evidence has a non-finite row")
    # Fixed max-row mantissa/exponent scaling.  It is held constant for the
    # h/h' pair at one Newton point; no frequency-dependent normalization is
    # introduced inside a derivative evaluation.
    # `frexp.` returns one `(mantissa, exponent)` tuple per row; keep the
    # two arrays explicitly so the evidence-bearing scaling is injective.
    pairs = frexp.(maxima)
    mantissas, exponents = first.(pairs), last.(pairs)
    # `frexp(m)=a*2^e` has `floor(log2(m))=e-1`; a zero row has the
    # documented no-op scale.  Scale individual components with `ldexp`, not
    # an intermediate Float64 scale factor, so subnormal SI rows cannot make
    # the coefficient itself overflow before it is applied.
    shifts = [mantissa == 0.0 ? 0 : 1 - exponent for (mantissa, exponent) in zip(mantissas, exponents)]
    scale_entry(value::ComplexF64, shift::Int) = complex(ldexp(real(value), shift), ldexp(imag(value), shift))
    Fs = Matrix{ComplexF64}(undef, size(F)); Fps = Matrix{ComplexF64}(undef, size(Fp))
    for row in axes(F, 1), column in axes(F, 2)
        Fs[row, column] = scale_entry(F[row, column], shifts[row])
        Fps[row, column] = scale_entry(Fp[row, column], shifts[row])
    end
    finite_matrix(Fs) && finite_matrix(Fps) ||
        fail("execution", "root_slope_unresolved", "determinant_scaling", "direct_quantity", "power-of-two determinant scaling is non-finite")
    # Restore the common row exponent only after the determinant/cofactor has
    # been formed in the bounded mantissa matrix.  Returning an unscaled
    # Float64 that silently under/overflows would falsify a zero certificate,
    # so such a representation failure remains typed rather than becoming 0.
    restore_shift = sum(mantissa == 0.0 ? 0 : exponent - 1 for (mantissa, exponent) in zip(mantissas, exponents))
    determinant_mantissa, determinant_exponent = determinant_mantissa_exponent(Fs)
    determinant = mantissa_exponent_value(determinant_mantissa, determinant_exponent + restore_shift)
    derivative = mantissa_exponent_value(cofactor_derivative(Fs, Fps), restore_shift)
    return determinant, derivative, Fs, Fps, mantissas, exponents
end

function full_selected_certificate(compiled::CompiledPrimitive, omega::ComplexF64, coordinates::Vector{String}, v::Vector{ComplexF64})
    indices = selected_coordinate_indices(compiled, coordinates)
    Q = operator_at(compiled, omega; loaded = true)
    eliminated = [index for index in eachindex(compiled.nodes) if index ∉ indices]
    x = zeros(ComplexF64, length(compiled.nodes)); x[indices] .= v
    if !isempty(eliminated)
        X = checked_solve(Q[eliminated, eliminated], Q[eliminated, indices], "eliminated_block_solve_failure", "eliminated_block", length(eliminated))
        x[eliminated] .= -X * v
    end
    denominator = norm(operator_absolute_bound(compiled, omega; loaded = true) * abs.(x), Inf)
    return denominator == 0.0 ? (norm(Q * x, Inf) == 0.0 ? 0.0 : Inf) : norm(Q * x, Inf) / denominator
end

function hybridized_pole(compiled::CompiledPrimitive, coordinates::Vector{String}, anchor;
        start::Union{Nothing,ComplexF64} = nothing)::Tuple{ComplexF64,ComplexF64,Vector{ComplexF64}}
    length(coordinates) >= 2 || fail("validation", "port_realizability", "hybridized_pole", "direct_quantity", "HybridizedPoleSpec requires at least two retained coordinates")
    omega = start === nothing ? 2.0 * pi * complex_frequency_value(anchor) : start
    isfinite(real(omega)) && isfinite(imag(omega)) && real(omega) > 0.0 || fail("validation", "invalid_diagonal_root_hint", "anchor", "direct_quantity", "hybridized-pole anchor must have finite positive real frequency")
    last_F = zeros(ComplexF64, length(coordinates), length(coordinates)); last_Fp = similar(last_F)
    for _ in 1:32
        F, Fp = selected_operator(compiled, omega, coordinates); last_F, last_Fp = F, Fp
        h, hp, _, _, _, _ = scaled_determinant_pair(F, Fp)
        hp != 0.0 || fail("execution", "root_slope_unresolved", "newton", "direct_quantity", "Hybridized-pole determinant derivative is zero")
        candidate = omega - h / hp
        same_bits = reinterpret(UInt64, real(candidate)) == reinterpret(UInt64, real(omega)) && reinterpret(UInt64, imag(candidate)) == reinterpret(UInt64, imag(omega))
        omega = candidate
        same_bits && break
    end
    F, Fp = selected_operator(compiled, omega, coordinates)
    h, hp, Fs, Fps, _, _ = scaled_determinant_pair(F, Fp)
    sv = svd(F); singular = sv.S
    length(singular) >= 2 && singular[1] > 0.0 && singular[end] / singular[1] <= tau(length(coordinates)) && singular[end - 1] / singular[1] > tau(length(coordinates)) ||
        fail("execution", "numerical_resolution_unresolved", "rank_certificate", "direct_quantity", "Hybridized-pole retained operator does not have exactly one machine-null direction")
    v, u = Vector{ComplexF64}(sv.V[:, end]), Vector{ComplexF64}(sv.U[:, end])
    phase_index = findfirst(==(maximum(abs.(v))), abs.(v))::Int
    phase = exp(-im * angle(v[phase_index])); v .*= phase; u .*= phase
    residual_den = norm(abs.(F) * abs.(v), Inf); residual = residual_den == 0.0 ? (norm(F * v, Inf) == 0.0 ? 0.0 : Inf) : norm(F * v, Inf) / residual_den
    left_den = norm(abs.(transpose(u)) * abs.(F), Inf); left_residual = left_den == 0.0 ? (norm(transpose(conj.(u)) * F, Inf) == 0.0 ? 0.0 : Inf) : norm(transpose(conj.(u)) * F, Inf) / left_den
    determinant_rows = prod(norm(Base.view(F, row, :)) for row in axes(F, 1))
    eta_det = determinant_rows == 0.0 ? (abs(h) == 0.0 ? 0.0 : Inf) : abs(h) / determinant_rows
    slope = dot(u, Fp * v)
    scale = sum(abs.(u) .* (abs.(Fp) * abs.(v)))
    correction = hp == 0.0 ? Inf : abs(h / hp) / abs(omega)
    eta_q = full_selected_certificate(compiled, omega, coordinates, v)
    isfinite(real(h)) && isfinite(imag(h)) && isfinite(real(hp)) && isfinite(imag(hp)) && eta_det <= tau(length(coordinates)) && isfinite(residual) && residual <= tau(length(coordinates)) && isfinite(left_residual) && left_residual <= tau(length(coordinates)) && isfinite(eta_q) && eta_q <= tau(length(compiled.nodes)) && scale > 0.0 && abs(slope) / scale > tau(length(coordinates)) && isfinite(correction) && correction <= tau(length(coordinates)) && real(omega) > 0.0 && imag(omega) <= 0.0 ||
        fail("execution", "numerical_resolution_unresolved", "newton_certificate", "direct_quantity", "Hybridized-pole machine-resolution certificate did not close")
    return omega, slope, v
end

function operator_element_root(compiled::CompiledPrimitive, coordinates::Vector{String},
        row::String, column::String, hint::Float64; start::Union{Nothing,ComplexF64} = nothing)
    isfinite(hint) && hint > 0.0 || fail("validation", "invalid_diagonal_root_hint", "root_hint", "direct_quantity", "root_hint must be finite and strictly positive")
    ri = findfirst(==(row), coordinates); cj = findfirst(==(column), coordinates)
    (ri === nothing || cj === nothing) && fail("validation", "port_realizability", "root", "direct_quantity", "root element is absent from the final View")
    omega = isnothing(start) ? complex(2.0 * pi * hint) : start
    isfinite(real(omega)) && isfinite(imag(omega)) || fail("execution", "numerical_resolution_unresolved", "newton", "direct_quantity", "root initialization is non-finite")
    for _ in 1:32
        state = operator_element_state(compiled, omega, coordinates, ri::Int, cj::Int)
        state.fp == 0.0 && fail("execution", "root_slope_unresolved", "newton", "direct_quantity", "selected element derivative is zero")
        candidate = omega - state.f / state.fp
        same_bits = reinterpret(UInt64, real(candidate)) == reinterpret(UInt64, real(omega)) &&
            reinterpret(UInt64, imag(candidate)) == reinterpret(UInt64, imag(omega))
        omega = candidate
        if same_bits
            break
        end
    end
    state = operator_element_state(compiled, omega, coordinates, ri::Int, cj::Int)
    conditions = isfinite(real(omega)) && isfinite(imag(omega)) && real(omega) > 0.0 &&
        state.eta_e <= tau(length(compiled.nodes)) && state.eta_q <= tau(length(compiled.nodes)) &&
        state.eta_f <= tau(length(compiled.nodes)) && state.correction <= tau(length(compiled.nodes))
    conditions || fail("execution", "numerical_resolution_unresolved", "newton_certificate", "direct_quantity", "selected element Newton procedure did not reach its machine-resolution certificate")
    state.normalized_slope > tau(length(compiled.nodes)) ||
        fail("execution", "root_slope_unresolved", "slope_certificate", "direct_quantity", "selected element local slope is unresolved")
    return omega, state.fp
end

function diagonal_root(compiled::CompiledPrimitive, coordinates::Vector{String}, coordinate::String,
        hint::Float64; start::Union{Nothing,ComplexF64} = nothing)
    omega, slope = operator_element_root(compiled, coordinates, coordinate, coordinate, hint; start = start)
    imag(omega) <= 0.0 || fail("execution", "numerical_resolution_unresolved", "newton_certificate", "direct_quantity", "diagonal root violates the passive imaginary-root policy")
    return omega, slope
end
