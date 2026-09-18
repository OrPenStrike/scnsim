# Included into the single SCNSimBackend module.
# Physical compilation, shared operator assembly, and structured expansion.

struct CompiledPrimitive
    nodes::Vector{String}
    C::Matrix{Float64}
    K::Matrix{Float64}
    G::Matrix{Float64}
    series_rl::Vector{Any}
    branch_rows::Vector{Dict{String,Any}}
    # Logical ports are a boundary, not intrinsic graph elements.  Keep the
    # ordered selector/reference/mask together so every Direct operation uses
    # the same N-port realization rather than accumulating one-Port fields.
    port_ids::Vector{String}
    B::Matrix{Float64}
    R::Matrix{Float64}
    M::Vector{Float64}
end

"""Strict SPD validation plus the symmetric principal real square root."""
function principal_spd_root(matrix::Matrix{Float64}; stage::String = "reference_matrix")
    try
        cholesky(Symmetric(matrix); check = true)
    catch
        fail("validation", "port_realizability", stage, "compile", "reference matrix must be real symmetric positive definite")
    end
    decomposition = eigen(Symmetric(matrix))
    all(isfinite, decomposition.values) && all(>(0.0), decomposition.values) ||
        fail("validation", "port_realizability", stage, "compile", "reference matrix has no finite positive principal spectrum")
    root = decomposition.vectors * Diagonal(sqrt.(decomposition.values)) * transpose(decomposition.vectors)
    residual = backward_residual(root, root, matrix)
    isfinite(residual) && residual <= tau(size(matrix, 1)) ||
        fail("execution", "direct_response_formation", stage, "direct_response", "principal reference root reconstruction exceeded normalized backward-residual contract")
    return Matrix((root + transpose(root)) ./ 2.0)
end

function port_reference_root(compiled::CompiledPrimitive)
    isempty(compiled.port_ids) && return zeros(Float64, 0, 0)
    return principal_spd_root(compiled.R)
end

function port_load_admittance(compiled::CompiledPrimitive)::Matrix{Float64}
    n = length(compiled.nodes)
    isempty(compiled.port_ids) && return zeros(Float64, n, n)
    factor = try
        cholesky(Symmetric(compiled.R); check = true)
    catch
        fail("validation", "port_realizability", "reference_matrix", "compile", "Port reference matrix must be real symmetric positive definite")
    end
    # The explicit diagonal mask is defined in original logical-Port order.
    # At this stage R is diagonal; transformed views carry their own derived
    # boundary evidence and never reinterpret this intrinsic load stamp.
    reference_inverse_mask = factor \ Diagonal(compiled.M)
    residual = backward_residual(compiled.R, reference_inverse_mask, Diagonal(compiled.M))
    isfinite(residual) && residual <= tau(length(compiled.port_ids)) ||
        fail("execution", "direct_response_formation", "reference_matrix", "direct_response", "Port reference solve exceeded normalized backward-residual contract")
    return compiled.B * reference_inverse_mask * transpose(compiled.B)
end

function apply_lineage_load_mask(compiled::CompiledPrimitive, lineage)::CompiledPrimitive
    item = plain(lineage)
    ptc = get(item, "ptc", nothing)
    ptc === nothing && return compiled
    selected = get(ptc, "selected_ports", nothing)
    selected isa AbstractVector && !isempty(selected) ||
        fail("execution", "compiler_invariant", "ptc", "compile", "PTC lineage has no selected Ports")
    ids = Set(String.(selected))
    length(ids) == length(selected) || fail("execution", "compiler_invariant", "ptc", "compile", "PTC lineage repeats a Port")
    mask = copy(compiled.M)
    for (index, port_id) in enumerate(compiled.port_ids)
        if port_id in ids
            mask[index] = 0.0
            delete!(ids, port_id)
        end
    end
    isempty(ids) || fail("validation", "port_realizability", "ptc", "compile", "PTC references a Port outside the sealed Plan")
    return CompiledPrimitive(compiled.nodes, compiled.C, compiled.K, compiled.G, compiled.series_rl, compiled.branch_rows,
        compiled.port_ids, compiled.B, compiled.R, mask)
end

"""Apply declared real floating-pair maps to every compiled physical form."""
function apply_lineage_transforms(compiled::CompiledPrimitive, lineage)::CompiledPrimitive
    item = plain(lineage)
    transforms = get(item, "transforms", Any[])
    transforms isa AbstractVector || fail("execution", "compiler_invariant", "transform_pair", "compile", "View transform collection is malformed")
    current = compiled
    for transform in transforms
        inputs = get(transform, "input_coordinates", nothing)
        outputs = haskey(transform, "output_coordinates") ? get(transform, "output_coordinates", nothing) :
            Any[get(transform, "common_id", nothing), get(transform, "differential_id", nothing)]
        inputs isa AbstractVector && length(inputs) == 2 && outputs isa AbstractVector && length(outputs) == 2 ||
            fail("execution", "compiler_invariant", "transform_pair", "compile", "View transform declaration is malformed")
        left, right = String(inputs[1]), String(inputs[2]); common, differential = String(outputs[1]), String(outputs[2])
        left != right && common != differential || fail("execution", "compiler_invariant", "transform_pair", "compile", "View transform coordinates collide")
        i = findfirst(==(left), current.nodes); j = findfirst(==(right), current.nodes)
        i !== nothing && j !== nothing || fail("validation", "port_realizability", "transform_pair", "compile", "View transform input coordinate is absent")
        i, j = i::Int, j::Int
        # C[j,j]+C[j,k] is the sum of every physical capacitance branch from
        # j to the full exterior cut; the direct pair branch cancels exactly.
        c_left = current.C[i, i] + current.C[i, j]
        c_right = current.C[j, j] + current.C[i, j]
        c_total = c_left + c_right
        isfinite(c_left) && isfinite(c_right) && c_left >= 0.0 && c_right >= 0.0 && c_total > 0.0 ||
            fail("validation", "port_realizability", "transform_pair", "compile", "full external capacitance cut does not define floating-pair weights")
        alpha, beta = c_left / c_total, c_right / c_total
        n = length(current.nodes)
        retained = [index for index in 1:n if index != i && index != j]
        # Identity-v1 puts generated channels at the tail in the declared
        # common/differential order.  Keep the physical congruence in exactly
        # that coordinate order; the Python verifier must never describe a
        # different basis from the numerical operator.
        without = [current.nodes[index] for index in 1:n if index != i && index != j]
        new_names = vcat(without, [common, differential])
        length(unique(new_names)) == n || fail("validation", "port_realizability", "transform_pair", "compile", "generated View coordinates collide")
        # V_new=A*V_old, emitted as [..., common, differential], with
        # common=α*left+β*right and differential=left-right.
        A = zeros(Float64, n, n)
        for (row, name) in enumerate(new_names)
            if name == common
                A[row, i] = alpha; A[row, j] = beta
            elseif name == differential
                A[row, i] = 1.0; A[row, j] = -1.0
            else
                old = findfirst(==(name), current.nodes)::Int
                A[row, old] = 1.0
            end
        end
        T = try
            A \ Matrix{Float64}(I, n, n)
        catch
            fail("execution", "compiler_invariant", "transform_pair", "compile", "floating-pair transform is not invertible")
        end
        reconstruction = backward_residual(A, T, Matrix{Float64}(I, n, n))
        isfinite(reconstruction) && reconstruction <= tau(n) ||
            fail("execution", "compiler_invariant", "transform_pair", "compile", "floating-pair transform reconstruction exceeds the normalized backward-residual contract")
        transformed_blocks = Any[SeriesRLBlock(block.id, transpose(T) * block.incidence, block.resistance, block.inductance) for block in current.series_rl]
        # Branch rows remain compiler diagnostics rather than a second
        # physical model, but their physical incidence must follow the same
        # congruence as C.  Later transform steps classify their own external
        # cut against this current basis; leaving source-basis incidences here
        # would make the lineage evidence describe a different graph.
        transformed_rows = Dict{String,Any}[]
        for source_row in current.branch_rows
            row = copy(source_row)
            for key in ("incidence_f64", "row_incidence_f64", "column_incidence_f64",
                    "physical_positive_incidence_f64", "physical_negative_incidence_f64")
                haskey(row, key) || continue
                encoded = row[key]
                encoded isa AbstractVector && length(encoded) == n ||
                    fail("execution", "compiler_invariant", "transform_pair", "compile", "capacitance branch incidence has the wrong compiled basis width")
                incidence = Float64[f64_from_hex(value) for value in encoded]
                row[key] = f64_hex.(transpose(T) * incidence)
            end
            push!(transformed_rows, row)
        end
        current = CompiledPrimitive(new_names, transpose(T) * current.C * T, transpose(T) * current.K * T,
            transpose(T) * current.G * T, transformed_blocks, transformed_rows, current.port_ids,
            transpose(T) * current.B, current.R, current.M)
    end
    return current
end

"""A leaf in the sealed data-only expansion; no Python factory is executable here."""
struct ExpandedInductor
    id::String
    incidence::Vector{Float64}
    value::Float64
end

"""One full ordered series R/L block from a distributed-line section."""
struct SeriesRLBlock
    id::String
    incidence::Matrix{Float64}
    resistance::Matrix{Float64}
    inductance::Matrix{Float64}
end

include("structured_v2.jl")

"""Compile normalized physical tables. Ports remain outside intrinsic C/K/G."""
function compile_primitive(plan_value, values::Dict{String,Any}; context_kind::String = "compile",
        authorized::Set{String} = Set{String}(), extrapolation_evidence::Union{Nothing,Vector{Any}} = nothing,
        authorization_source::String = "none", emit_audit::Bool = false)::CompiledPrimitive
    return structured_compile(plan_value, values; context_kind = context_kind, authorized = authorized,
        extrapolation_evidence = extrapolation_evidence, authorization_source = authorization_source,
        emit_audit = emit_audit)[1]
end

function rlgc_matrix(record, name::String)::Matrix{Float64}
    item = plain(record)
    get(item, "type", nothing) == "quantity_matrix_f64" ||
        fail("execution", "compiler_invariant", "compile", "compile", "RLGC $(name) record has the wrong discriminator")
    shape = get(item, "shape", nothing); values = get(item, "values_f64", nothing)
    shape isa AbstractVector && length(shape) == 2 && shape[1] == shape[2] && shape[1] >= 1 && values isa AbstractVector && length(values) == shape[1] * shape[2] ||
        fail("execution", "compiler_invariant", "compile", "compile", "RLGC $(name) matrix shape is malformed")
    matrix = Matrix{Float64}(undef, shape[1], shape[2])
    for row in axes(matrix, 1), column in axes(matrix, 2)
        matrix[row, column] = f64_from_hex(values[(row - 1) * shape[2] + column])
    end
    matrix == transpose(matrix) || fail("execution", "compiler_invariant", "compile", "compile", "RLGC $(name) matrix is not bit-exact symmetric")
    return matrix
end

tau(n::Int) = 256.0 * (n + 1) * EPS64

function finite_matrix(value)
    return all(isfinite, real.(value)) && all(isfinite, imag.(value))
end

function backward_residual(A::AbstractMatrix, X, B)::Float64
    numerator = norm(A * X - B, Inf)
    denominator = norm(abs.(A) * abs.(X) + abs.(B), Inf)
    (!isfinite(numerator) || !isfinite(denominator)) && return Inf
    denominator == 0.0 && return numerator == 0.0 ? 0.0 : Inf
    return numerator / denominator
end

function checked_solve(A::Matrix{ComplexF64}, B, kind::String, stage::String, n::Int)
    finite_matrix(A) && finite_matrix(B) || fail("execution", kind, stage, "direct_response", "non-finite linear system")
    X = try
        A \ B
    catch
        fail("execution", kind, stage, "direct_response", "required linear solve is singular")
    end
    residual = backward_residual(A, X, B)
    isfinite(residual) && residual <= tau(n) || fail("execution", kind, stage, "direct_response", "linear solve exceeded normalized backward-residual contract")
    return X
end

function operator_at(compiled::CompiledPrimitive, omega::ComplexF64; loaded::Bool)::Matrix{ComplexF64}
    G = complex.(compiled.G)
    if loaded
        G .+= complex.(port_load_admittance(compiled))
    end
    operator = complex.(compiled.K) .- omega^2 .* complex.(compiled.C) .- im * omega .* G
    for block in compiled.series_rl
        block isa SeriesRLBlock || fail("execution", "compiler_invariant", "compile", "compile", "series branch record has the wrong type")
        impedance = complex.(block.resistance) .- im * omega .* complex.(block.inductance)
        inverse = checked_solve(impedance, Matrix{ComplexF64}(I, size(impedance, 1), size(impedance, 2)), "direct_response_formation", "series_rl", size(impedance, 1))
        operator .+= -im * omega .* (complex.(block.incidence) * inverse * transpose(complex.(block.incidence)))
    end
    return operator
end

function operator_derivative_at(compiled::CompiledPrimitive, omega::ComplexF64; loaded::Bool)::Matrix{ComplexF64}
    G = complex.(compiled.G)
    loaded && (G .+= complex.(port_load_admittance(compiled)))
    derivative = -2.0 * omega .* complex.(compiled.C) .- im .* G
    for block in compiled.series_rl
        block isa SeriesRLBlock || fail("execution", "compiler_invariant", "compile", "compile", "series branch record has the wrong type")
        impedance = complex.(block.resistance) .- im * omega .* complex.(block.inductance)
        inverse = checked_solve(impedance, Matrix{ComplexF64}(I, size(impedance, 1), size(impedance, 2)), "direct_response_formation", "series_rl", size(impedance, 1))
        derivative .+= complex.(block.incidence) * (-im .* inverse .+ omega .* (inverse * complex.(block.inductance) * inverse)) * transpose(complex.(block.incidence))
    end
    return derivative
end

function operator_absolute_bound(compiled::CompiledPrimitive, omega::ComplexF64; loaded::Bool)::Matrix{Float64}
    G = complex.(compiled.G)
    loaded && (G .+= complex.(port_load_admittance(compiled)))
    bound = abs.(complex.(compiled.K)) .+ abs2(omega) .* abs.(complex.(compiled.C)) .+ abs(omega) .* abs.(G)
    for block in compiled.series_rl
        impedance = complex.(block.resistance) .- im * omega .* complex.(block.inductance)
        inverse = checked_solve(impedance, Matrix{ComplexF64}(I, size(impedance, 1), size(impedance, 2)), "direct_response_formation", "series_rl", size(impedance, 1))
        bound .+= abs.(-im * omega .* (complex.(block.incidence) * inverse * transpose(complex.(block.incidence))))
    end
    return bound
end
