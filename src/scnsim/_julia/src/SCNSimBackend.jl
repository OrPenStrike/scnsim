module SCNSimBackend

using CMAEvolutionStrategy
using JSON3
using LinearAlgebra
using Random
using SHA

const EPS64 = eps(Float64)

"""A typed backend failure that becomes a request-scoped outcome envelope."""
struct BackendFailure <: Exception
    category::String
    kind::String
    stage::String
    context_kind::String
    message::String
end

Base.showerror(io::IO, failure::BackendFailure) = print(io, failure.message)

fail(category, kind, stage, context, message) =
    throw(BackendFailure(category, kind, stage, context, message))

function plain(value)
    if value isa JSON3.Object
        return Dict{String,Any}(String(key) => plain(item) for (key, item) in pairs(value))
    elseif value isa JSON3.Array
        return Any[plain(item) for item in value]
    end
    return value
end

"""Canonical JSON used for every backend-authored schema envelope."""
function canonical_json(value)::String
    if value isa AbstractDict
        keys_sorted = sort!(String[String(key) for key in keys(value)])
        return "{" * join((JSON3.write(key) * ":" * canonical_json(value[key]) for key in keys_sorted), ",") * "}"
    elseif value isa AbstractVector || value isa Tuple
        return "[" * join((canonical_json(item) for item in value), ",") * "]"
    elseif value === nothing
        return "null"
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa AbstractString
        return String(JSON3.write(value))
    elseif value isa Integer
        return string(value)
    end
    error("canonical JSON only accepts schema primitives, got $(typeof(value))")
end

canonical_bytes(value) = Vector{UInt8}(codeunits(canonical_json(value)))
sha256_hex(bytes::AbstractVector{UInt8}) = bytes2hex(sha256(bytes))
file_sha256(path::AbstractString) = sha256_hex(read(path))

function f64_from_hex(value)::Float64
    value isa AbstractString || fail("execution", "compiler_invariant", "compile", "compile", "binary64 value is not hexadecimal text")
    length(value) == 16 || fail("execution", "compiler_invariant", "compile", "compile", "binary64 value has wrong width")
    bits = try
        parse(UInt64, value; base = 16)
    catch
        fail("execution", "compiler_invariant", "compile", "compile", "binary64 value is not hexadecimal")
    end
    result = reinterpret(Float64, bits)
    isfinite(result) || fail("execution", "compiler_invariant", "compile", "compile", "binary64 value is non-finite")
    return result
end

f64_hex(value::Float64) = string(reinterpret(UInt64, value); base = 16, pad = 16)

function quantity_value(quantity)::Float64
    item = plain(quantity)
    get(item, "type", nothing) == "quantity_f64" || fail("execution", "compiler_invariant", "compile", "compile", "expected quantity_f64")
    return f64_from_hex(item["si_value_f64"])
end

quantity(value::Float64, unit::String, dimensionality::String) = Dict{String,Any}(
    "type" => "quantity_f64",
    "si_value_f64" => f64_hex(value),
    "si_unit" => unit,
    "dimensionality" => dimensionality,
)

complex_quantity(value::ComplexF64, unit::String, dimensionality::String) = Dict{String,Any}(
    "type" => "complex_quantity_f64",
    "real_si_f64" => f64_hex(real(value)),
    "imag_si_f64" => f64_hex(imag(value)),
    "si_unit" => unit,
    "dimensionality" => dimensionality,
)

function ref_key(reference)::String
    item = plain(reference)
    return join(String.(item["component_path"]), "\u001f") * "\u001e" * String(item["parameter_id"])
end

function endpoint_key(endpoint)::String
    item = plain(endpoint)
    return join(String.(item["component_path"]), "\u001f") * "\u001e" * String(item["pin_id"])
end

function parameter_values(request)::Dict{String,Float64}
    bindings = plain(request)["parameters"]["bindings"]
    values = Dict{String,Float64}()
    for binding in bindings
        parameter = binding["parameter"]
        key = ref_key(parameter)
        haskey(values, key) && fail("execution", "compiler_invariant", "compile", "compile", "duplicate resolved parameter binding")
        values[key] = quantity_value(binding["value"])
    end
    return values
end

function resolve_binding(binding, values::Dict{String,Float64})::Float64
    item = plain(binding)
    kind = item["kind"]
    if kind == "constant"
        return quantity_value(item["value"])
    elseif kind == "identity"
        key = ref_key(item["input"])
        haskey(values, key) || fail("execution", "compiler_invariant", "compile", "compile", "missing resolved parameter binding")
        return values[key]
    elseif kind == "affine"
        key = ref_key(item["input"])
        haskey(values, key) || fail("execution", "compiler_invariant", "compile", "compile", "missing resolved affine input")
        return quantity_value(item["slope"]) * values[key] + quantity_value(item["intercept"])
    end
    fail("execution", "compiler_invariant", "compile", "compile", "unknown parameter binding kind")
end

function primitive_value(component, parameter_id::String, binding, values::Dict{String,Float64})::Float64
    public_reference = Dict("component_path" => component["component_path"], "parameter_id" => parameter_id)
    key = ref_key(public_reference)
    return haskey(values, key) ? values[key] : resolve_binding(binding, values)
end

struct CompiledPrimitive
    nodes::Vector{String}
    C::Matrix{Float64}
    K::Matrix{Float64}
    G::Matrix{Float64}
    branch_rows::Vector{Dict{String,Any}}
    port_id::String
    port_index::Int
    reference_impedance::Float64
end

function endpoint_nodes(plan)::Dict{String,String}
    lookup = Dict{String,String}()
    for node in plan["nodes"]
        for endpoint in node["endpoints"]
            key = endpoint_key(endpoint)
            haskey(lookup, key) && fail("execution", "compiler_invariant", "compile", "endpoint belongs to multiple nodes")
            lookup[key] = String(node["node_id"])
        end
    end
    for endpoint in plan["grounded_endpoints"]
        key = endpoint_key(endpoint)
        haskey(lookup, key) && fail("execution", "compiler_invariant", "compile", "grounded endpoint also belongs to a node")
        lookup[key] = "ground"
    end
    return lookup
end

function branch_incidence(component, endpoint_to_node, node_index)::Vector{Float64}
    pins = component["pin_order"]
    length(pins) == 2 || fail("execution", "compiler_invariant", "compile", "primitive component must have exactly two ordered pins")
    path = component["component_path"]
    first_endpoint = endpoint_key(Dict("component_path" => path, "pin_id" => pins[1]))
    second_endpoint = endpoint_key(Dict("component_path" => path, "pin_id" => pins[2]))
    haskey(endpoint_to_node, first_endpoint) || fail("execution", "compiler_invariant", "compile", "primitive terminal_1 is unbound")
    haskey(endpoint_to_node, second_endpoint) || fail("execution", "compiler_invariant", "compile", "primitive terminal_2 is unbound")
    b = zeros(Float64, length(node_index))
    first_node = endpoint_to_node[first_endpoint]
    second_node = endpoint_to_node[second_endpoint]
    first_node != "ground" && (b[node_index[first_node]] += 1.0)
    second_node != "ground" && (b[node_index[second_node]] -= 1.0)
    return b
end

"""Compile the sealed primitive snapshot. Ports remain outside intrinsic C/K/G."""
function compile_primitive(plan_value, values::Dict{String,Float64})::CompiledPrimitive
    plan = plain(plan_value)
    get(plan, "schema", nothing) == "scnsim.plan" || fail("execution", "compiler_invariant", "compile", "plan schema discriminator is invalid")
    components = sort!(copy(plan["components"]); by = item -> join(String.(item["component_path"]), "\u001f"))
    all(item["realization"]["kind"] in ("resistor", "capacitor", "inductor") for item in components) ||
        fail("capability", "scaffold_unavailable", "compile", "compile", "dev3 compiler accepts only primitive R/L/C snapshots")
    nodes = sort!(String[item["node_id"] for item in plan["nodes"]])
    isempty(nodes) && fail("execution", "compiler_invariant", "compile", "primitive plan has no non-reference node")
    node_index = Dict(node => index for (index, node) in enumerate(nodes))
    endpoint_to_node = endpoint_nodes(plan)
    n = length(nodes)
    C = zeros(Float64, n, n)
    K = zeros(Float64, n, n)
    G = zeros(Float64, n, n)
    branch_rows = Dict{String,Any}[]
    for component in components
        b = branch_incidence(component, endpoint_to_node, node_index)
        realization = component["realization"]
        kind = realization["kind"]
        raw = if kind == "capacitor"
            primitive_value(component, "capacitance", realization["capacitance"], values)
        elseif kind == "inductor"
            primitive_value(component, "inductance", realization["inductance"], values)
        else
            primitive_value(component, "resistance", realization["resistance"], values)
        end
        isfinite(raw) && raw > 0.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", "optimization_candidate", "primitive R/L/C values must be finite and strictly positive")
        stamp = b * transpose(b)
        if kind == "capacitor"
            C .+= raw .* stamp
        elseif kind == "inductor"
            K .+= (1.0 / raw) .* stamp
        else
            G .+= (1.0 / raw) .* stamp
        end
        push!(branch_rows, Dict{String,Any}(
            "component_path" => component["component_path"],
            "kind" => kind,
            "terminal_1_to_terminal_2" => component["pin_order"],
            "incidence_f64" => f64_hex.(b),
            "value" => quantity(raw,
                kind == "capacitor" ? "farad" : kind == "inductor" ? "henry" : "ohm",
                kind == "capacitor" ? "capacitance" : kind == "inductor" ? "inductance" : "resistance"),
        ))
    end
    ports = plan["ports"]
    length(ports) == 1 || fail("validation", "port_realizability", "compile", "compile", "dev3 Direct requires exactly one logical Port")
    port = ports[1]
    # A raw nonloading probe is still a matched physical load. PTC is deferred
    # beyond dev3, so it receives exactly the same raw selected-network stamp.
    port["role"] in ("terminated", "nonloading_probe") ||
        fail("validation", "port_realizability", "compile", "compile", "dev3 Direct requires a terminated or raw nonloading-probe Port")
    haskey(node_index, String(port["node_id"])) || fail("execution", "compiler_invariant", "compile", "compile", "Port node is absent from compiled basis")
    z0 = quantity_value(port["reference_impedance"])
    isfinite(z0) && z0 > 0.0 || fail("execution", "compiler_invariant", "compile", "compile", "Port reference impedance must be finite and positive")
    return CompiledPrimitive(nodes, C, K, G, branch_rows, String(port["port_id"]), node_index[String(port["node_id"])], z0)
end

tau(n::Int) = 256.0 * (n + 1) * EPS64

function finite_matrix(value)
    return all(isfinite, real.(value)) && all(isfinite, imag.(value))
end

function backward_residual(A::AbstractMatrix{ComplexF64}, X, B)::Float64
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
        G[compiled.port_index, compiled.port_index] += 1.0 / compiled.reference_impedance
    end
    return complex.(compiled.K) .- omega^2 .* complex.(compiled.C) .- im * omega .* G
end

function selected_one_port_admittance(Ycirc::Matrix{ComplexF64}, port_index::Int)
    n = size(Ycirc, 1)
    eliminated = [index for index in 1:n if index != port_index]
    isempty(eliminated) && return Ycirc[port_index, port_index]
    Yee = Ycirc[eliminated, eliminated]
    Yep = Ycirc[eliminated, port_index]
    X = checked_solve(Yee, Yep, "direct_response_formation", "schur_complement", length(eliminated))
    return Ycirc[port_index, port_index] - sum(Ycirc[port_index, eliminated] .* X)
end

function response_at(compiled::CompiledPrimitive, frequency::Float64)
    omega = 2.0 * pi * frequency
    n = length(compiled.nodes)
    Q = operator_at(compiled, complex(omega); loaded = false)
    Yoperator = Q / (-im * omega)
    y = selected_one_port_admittance(Yoperator, compiled.port_index)
    isfinite(real(y)) && isfinite(imag(y)) ||
        fail("execution", "direct_response_formation", "schur_complement", "direct_response", "selected one-Port admittance is non-finite")
    Ynet = reshape(ComplexF64[y], 1, 1)
    z = checked_solve(Ynet, Matrix{ComplexF64}(I, 1, 1), "direct_response_formation", "y_to_z", 1)[1]
    y0 = 1.0 / compiled.reference_impedance
    Sleft = Matrix{ComplexF64}(I, 1, 1) .+ compiled.reference_impedance .* Ynet
    Sright = Matrix{ComplexF64}(I, 1, 1) .- compiled.reference_impedance .* Ynet
    s = checked_solve(Sleft, Sright, "direct_response_formation", "y_to_s", 1)[1]
    b = zeros(ComplexF64, n)
    b[compiled.port_index] = 1.0 + 0.0im
    loaded = copy(Yoperator)
    loaded .+= y0 .* (b * transpose(b))
    source = 2.0 * sqrt(y0) .* b
    voltage = checked_solve(loaded, source, "direct_response_formation", "source_solve", n)
    source_s = sqrt(y0) * voltage[compiled.port_index] - 1.0
    deembed_eta = abs(source_s - s) / (1.0 + abs(source_s) + abs(s))
    isfinite(deembed_eta) && deembed_eta <= tau(n) ||
        fail("execution", "direct_response_formation", "deembedding", "direct_response", "source-boundary and de-embedded selected-network responses disagree")
    all(isfinite, (real(s), imag(s), real(y), imag(y), real(z), imag(z))) ||
        fail("execution", "direct_response_formation", "response_formation", "direct_response", "one-Port Direct response is non-finite")
    return s, y, z
end

function root_certificate(compiled::CompiledPrimitive, omega::ComplexF64, coordinate::String)
    n = length(compiled.nodes)
    index = findfirst(==(coordinate), compiled.nodes)
    index === nothing && fail("validation", "port_realizability", "root", "direct_quantity", "retained coordinate is absent from the compiled basis")
    r = index::Int
    Q = operator_at(compiled, omega; loaded = true)
    Qp = -2.0 * omega .* complex.(compiled.C) .- im .* (complex.(compiled.G) .+ begin
        load = zeros(ComplexF64, n, n)
        load[compiled.port_index, compiled.port_index] = 1.0 / compiled.reference_impedance
        load
    end)
    eliminated = [i for i in 1:n if i != r]
    if isempty(eliminated)
        X = ComplexF64[]
        Xp = ComplexF64[]
        f = Q[r, r]
        fp = Qp[r, r]
        eta_e = 0.0
    else
        Qee = Q[eliminated, eliminated]
        Qer = Q[eliminated, r]
        X = try
            Qee \ Qer
        catch
            fail("execution", "eliminated_block_solve_failure", "eliminated_block", "direct_quantity", "DiagonalRoot eliminated block is singular")
        end
        eta_e = backward_residual(Qee, X, Qer)
        isfinite(eta_e) && eta_e <= tau(n) || fail("execution", "eliminated_block_solve_failure", "eliminated_block", "direct_quantity", "DiagonalRoot eliminated block lacks residual evidence")
        Qpeer = Qp[eliminated, r]
        Qpee = Qp[eliminated, eliminated]
        derivative_rhs = Qpeer - Qpee * X
        Xp = try
            Qee \ derivative_rhs
        catch
            fail("execution", "eliminated_block_solve_failure", "derivative_eliminated_block", "direct_quantity", "DiagonalRoot derivative eliminated solve is singular")
        end
        derivative_residual = backward_residual(Qee, Xp, derivative_rhs)
        isfinite(derivative_residual) && derivative_residual <= tau(n) ||
            fail("execution", "eliminated_block_solve_failure", "derivative_eliminated_block", "direct_quantity", "DiagonalRoot derivative eliminated solve exceeded its normalized backward-residual Gate")
        f = Q[r, r] - sum(Q[r, eliminated] .* X)
        fp = Qp[r, r] - sum(Qp[r, eliminated] .* X) - sum(Q[r, eliminated] .* Xp)
    end
    x = zeros(ComplexF64, n)
    x[r] = 1.0 + 0.0im
    !isempty(eliminated) && (x[eliminated] .= -X)
    abs_operator = abs.(complex.(compiled.K)) .+ abs(omega) .* abs.(complex.(compiled.G) .+ begin
        load = zeros(ComplexF64, n, n)
        load[compiled.port_index, compiled.port_index] = 1.0 / compiled.reference_impedance
        load
    end) .+ abs2(omega) .* abs.(complex.(compiled.C))
    q_residual = norm(Q * x, Inf)
    q_denominator = norm(abs_operator * abs.(x), Inf)
    eta_q = q_denominator == 0.0 ? (q_residual == 0.0 ? 0.0 : Inf) : q_residual / q_denominator
    f_denominator = (abs_operator * abs.(x))[r]
    eta_f = f_denominator == 0.0 ? (abs(f) == 0.0 ? 0.0 : Inf) : abs(f) / f_denominator
    correction = fp == 0.0 ? Inf : abs(f / fp) / abs(omega)
    slope_scale = abs(Qp[r, r])
    if !isempty(eliminated)
        slope_scale += sum(abs.(Qp[r, eliminated]) .* abs.(X)) + sum(abs.(Q[r, eliminated]) .* abs.(Xp))
    end
    normalized_slope = slope_scale == 0.0 ? Inf : abs(fp) / slope_scale
    return (f = f, fp = fp, eta_e = eta_e, eta_q = eta_q, eta_f = eta_f,
        correction = correction, normalized_slope = normalized_slope)
end

function diagonal_root(compiled::CompiledPrimitive, coordinate::String, hint::Float64; start::Union{Nothing,ComplexF64} = nothing)
    isfinite(hint) && hint > 0.0 || fail("validation", "invalid_diagonal_root_hint", "root_hint", "direct_quantity", "root_hint must be finite and strictly positive")
    try
        cholesky(Symmetric(compiled.C); check = true)
    catch
        fail("capability", "unsupported_singular_capacitance_for_diagonal_root_v1", "capacitance_positive_definiteness", "direct_quantity", "DiagonalRootSpec requires positive-definite full capacitance")
    end
    omega = isnothing(start) ? complex(2.0 * pi * hint) : start
    isfinite(real(omega)) && isfinite(imag(omega)) || fail("execution", "numerical_resolution_unresolved", "newton", "direct_quantity", "root initialization is non-finite")
    last = nothing
    for _ in 1:32
        state = root_certificate(compiled, omega, coordinate)
        last = state
        state.fp == 0.0 && fail("execution", "root_slope_unresolved", "newton", "direct_quantity", "DiagonalRoot derivative is zero")
        candidate = omega - state.f / state.fp
        same_bits = reinterpret(UInt64, real(candidate)) == reinterpret(UInt64, real(omega)) &&
            reinterpret(UInt64, imag(candidate)) == reinterpret(UInt64, imag(omega))
        omega = candidate
        if same_bits
            break
        end
    end
    state = root_certificate(compiled, omega, coordinate)
    conditions = isfinite(real(omega)) && isfinite(imag(omega)) && real(omega) > 0.0 && imag(omega) <= 0.0 &&
        state.eta_e <= tau(length(compiled.nodes)) && state.eta_q <= tau(length(compiled.nodes)) &&
        state.eta_f <= tau(length(compiled.nodes)) && state.correction <= tau(length(compiled.nodes))
    conditions || fail("execution", "numerical_resolution_unresolved", "newton_certificate", "direct_quantity", "DiagonalRoot Newton procedure did not reach its machine-resolution certificate")
    state.normalized_slope > tau(length(compiled.nodes)) ||
        fail("execution", "root_slope_unresolved", "slope_certificate", "direct_quantity", "DiagonalRoot local slope is unresolved")
    return omega, state.fp
end

function write_bytes(path::AbstractString, bytes::AbstractVector{UInt8})
    mkpath(dirname(path))
    temporary = joinpath(dirname(path), "." * basename(path) * ".tmp-" *
        string(getpid()) * "-" * string(rand(UInt64); base = 16))
    try
        open(temporary, "w") do io
            write(io, bytes)
            flush(io)
            ccall(:fsync, Cint, (Cint,), fd(io)) == 0 || error("failed to fsync staged artifact")
        end
        mv(temporary, path; force = true)
    catch
        isfile(temporary) && rm(temporary; force = true)
        rethrow()
    end
end

function write_c_f64(io, values)
    for value in values
        write(io, htol(reinterpret(UInt64, Float64(value))))
    end
end

function zarray_metadata(shape::Vector{Int}, chunks::Vector{Int})::String
    return canonical_json(Dict{String,Any}(
        "chunks" => chunks,
        "compressor" => nothing,
        "dimension_separator" => ".",
        "dtype" => "<f8",
        "fill_value" => nothing,
        "filters" => nothing,
        "order" => "C",
        "shape" => shape,
        "zarr_format" => 2,
    ))
end

function relative_files(root::AbstractString)
    files = String[]
    for (directory, _, names) in walkdir(root)
        for name in names
            path = joinpath(directory, name)
            isfile(path) || error("Zarr artifact contains a non-regular file")
            push!(files, replace(relpath(path, root), '\\' => '/'))
        end
    end
    return sort!(files)
end

function write_manifest(staging::AbstractString, artifact_id::String, artifact_path::String, datasets::Vector{Dict{String,Any}})::String
    root = joinpath(staging, artifact_path)
    entries = Any[]
    for relative in relative_files(root)
        path = joinpath(root, relative)
        push!(entries, Dict{String,Any}(
            "path" => relative,
            "mode" => "regular",
            "byte_length" => filesize(path),
            "sha256" => file_sha256(path),
        ))
    end
    manifest = Dict{String,Any}(
        "schema" => "scnsim.artifact_manifest",
        "schema_version" => 1,
        "artifact_id" => artifact_id,
        "artifact_path" => artifact_path,
        "zarr_format" => 2,
        "group_metadata_path" => ".zgroup",
        "datasets" => datasets,
        "files" => entries,
    )
    manifest_path = joinpath(staging, "artifacts", artifact_id * ".manifest.json")
    write_bytes(manifest_path, canonical_bytes(manifest))
    return file_sha256(manifest_path)
end

function dataset_entry(path::String, chunks::Vector{String})
    return Dict{String,Any}(
        "path" => path,
        "metadata_path" => path * "/.zarray",
        "chunk_paths" => chunks,
    )
end

function dataset_metadata(shape::Vector{Int}, chunks::Vector{Int})
    return Dict{String,Any}(
        "zarr_format" => 2,
        "shape" => shape,
        "chunks" => chunks,
        "dtype" => "<f8",
        "compressor" => nothing,
        "fill_value" => nothing,
        "order" => "C",
        "filters" => nothing,
        "dimension_separator" => ".",
    )
end

function write_real_zarr(staging::AbstractString, artifact_id::String, values::Vector{Float64})
    root_rel = "artifacts/" * artifact_id * ".zarr"
    root = joinpath(staging, root_rel)
    dataset = joinpath(root, "values")
    mkpath(dataset)
    write_bytes(joinpath(root, ".zgroup"), Vector{UInt8}(codeunits("{\"zarr_format\":2}")))
    chunk_size = min(length(values), 1024)
    write_bytes(joinpath(dataset, ".zarray"), Vector{UInt8}(codeunits(zarray_metadata([length(values)], [chunk_size]))))
    chunks = String[]
    for start in 1:chunk_size:length(values)
        chunk_index = (start - 1) ÷ chunk_size
        chunk_name = string(chunk_index)
        push!(chunks, "values/" * chunk_name)
        open(joinpath(dataset, chunk_name), "w") do io
            write_c_f64(io, @view values[start:min(start + chunk_size - 1, length(values))])
        end
    end
    manifest_sha = write_manifest(staging, artifact_id, root_rel, [dataset_entry("values", chunks)])
    metadata = dataset_metadata([length(values)], [chunk_size])
    return Dict{String,Any}(
        "id" => artifact_id,
        "path" => root_rel,
        "sha256" => manifest_sha,
        "media_type" => "application/vnd+zarr-v2",
        "file_manifest" => "artifacts/" * artifact_id * ".manifest.json",
        "dtype" => "float64",
        "shape" => [length(values)],
        "chunks" => [chunk_size],
        "complex_storage" => "real",
        "group_metadata" => Dict("zarr_format" => 2),
        "datasets" => [Dict("path" => "values", "metadata" => metadata)],
        "axes" => [Dict("id" => "frequency", "kind" => "frequency", "artifact_id" => "frequencies")],
        "unit" => "hertz",
        "dimensionality" => "inverse_time",
        "chunk_policy" => "frequency_capped_1024_v1",
    )
end

function write_complex_zarr(staging::AbstractString, artifact_id::String, values::Vector{ComplexF64}, port_id::String, unit::String, dimensionality::String)
    root_rel = "artifacts/" * artifact_id * ".zarr"
    root = joinpath(staging, root_rel)
    shape = [length(values), 1, 1]
    chunk_shape = [min(length(values), 1024), 1, 1]
    write_bytes(joinpath(root, ".zgroup"), Vector{UInt8}(codeunits("{\"zarr_format\":2}")))
    entries = Dict{String,Any}[]
    metadata = dataset_metadata(shape, chunk_shape)
    for (name, projection) in (("real", real), ("imag", imag))
        dataset = joinpath(root, name)
        mkpath(dataset)
        write_bytes(joinpath(dataset, ".zarray"), Vector{UInt8}(codeunits(zarray_metadata(shape, chunk_shape))))
        chunks = String[]
        for start in 1:chunk_shape[1]:length(values)
            chunk_index = (start - 1) ÷ chunk_shape[1]
            chunk_name = string(chunk_index, ".0.0")
            push!(chunks, name * "/" * chunk_name)
            open(joinpath(dataset, chunk_name), "w") do io
                for index in start:min(start + chunk_shape[1] - 1, length(values))
                    write_c_f64(io, (projection(values[index]),))
                end
            end
        end
        push!(entries, dataset_entry(name, chunks))
    end
    manifest_sha = write_manifest(staging, artifact_id, root_rel, entries)
    return Dict{String,Any}(
        "id" => artifact_id,
        "path" => root_rel,
        "sha256" => manifest_sha,
        "media_type" => "application/vnd+zarr-v2",
        "file_manifest" => "artifacts/" * artifact_id * ".manifest.json",
        "dtype" => "complex128",
        "shape" => shape,
        "chunks" => chunk_shape,
        "complex_storage" => "paired_float64_real_imag",
        "group_metadata" => Dict("zarr_format" => 2),
        "datasets" => [Dict("path" => "real", "metadata" => metadata), Dict("path" => "imag", "metadata" => metadata)],
        "axes" => [
            Dict("id" => "frequency", "kind" => "frequency", "artifact_id" => "frequencies"),
            Dict("id" => "output_coordinate", "kind" => "coordinate_output", "values" => [port_id]),
            Dict("id" => "input_coordinate", "kind" => "coordinate_input", "values" => [port_id]),
        ],
        "unit" => unit,
        "dimensionality" => dimensionality,
        "chunk_policy" => "frequency_slab_full_matrix_v1",
        "coordinate_ids" => [port_id],
        "probe_load_state" => [Dict("port_id" => port_id, "state" => "raw")],
    )
end

function result_envelope(kind::String, request_sha::String, attempt_sha::String, scalars, arrays)
    return Dict{String,Any}(
        "schema" => "scnsim.result",
        "schema_version" => 1,
        "result_kind" => kind,
        "request_sha256" => request_sha,
        "attempt_sha256" => attempt_sha,
        "scalar_catalog" => scalars,
        "array_catalog" => arrays,
    )
end

function write_success(staging::String, request, request_sha::String, attempt_sha::String, result, artifact_catalog)
    result_path = joinpath(staging, "result.json")
    write_bytes(result_path, canonical_bytes(result))
    result_sha = file_sha256(result_path)
    artifacts = Any[Dict("id" => item["id"], "sha256" => item["sha256"]) for item in artifact_catalog]
    outcome = Dict{String,Any}(
        "schema" => "scnsim.outcome",
        "schema_version" => 1,
        "request_sha256" => request_sha,
        "attempt_sha256" => attempt_sha,
        "runtime_semantic" => request["runtime_semantic"],
        "status" => "success",
        "result_sha256" => result_sha,
        "artifacts" => artifacts,
    )
    write_bytes(joinpath(staging, "outcome.json"), canonical_bytes(outcome))
end

function write_failure(staging::String, request, request_sha::String, attempt_sha::String, failure::BackendFailure)
    evidence = Dict{String,Any}(
        "type" => "failure_evidence",
        "operation" => request["operation"],
        "context_kind" => failure.context_kind,
    )
    typed = Dict{String,Any}(
        "category" => failure.category,
        "kind" => failure.kind,
        "stage" => failure.stage,
        "message" => failure.message,
        "evidence" => evidence,
    )
    outcome = Dict{String,Any}(
        "schema" => "scnsim.outcome",
        "schema_version" => 1,
        "request_sha256" => request_sha,
        "attempt_sha256" => attempt_sha,
        "runtime_semantic" => request["runtime_semantic"],
        "status" => "failure",
        "failure" => typed,
        "artifacts" => staged_generation_links(staging, request_sha, attempt_sha),
    )
    write_bytes(joinpath(staging, "outcome.json"), canonical_bytes(outcome))
end

function solve_direct(request, compiled::CompiledPrimitive, request_sha::String, attempt_sha::String, staging::String)
    lineage = request["ref_lineage"]
    original = lineage["original"]
    lineage["retain"] === nothing && lineage["port_realizable"] === true &&
        lineage["terminal_coordinates"] == original["port_order"] ||
        fail("validation", "port_realizability", "selected_network", "direct_response", "Direct response requires the exact original Port-realizable View")
    spec = request["spec"]
    frequencies = Float64[quantity_value(item) for item in spec["frequencies"]]
    isempty(frequencies) && fail("validation", "port_realizability", "frequency_grid", "direct_response", "Direct frequency grid must be nonempty")
    all(isfinite, frequencies) && all(>(0.0), frequencies) && all(diff(frequencies) .> 0.0) ||
        fail("validation", "port_realizability", "frequency_grid", "direct_response", "Direct frequency grid must be finite, positive, and strictly increasing")
    isempty(spec["traces"]) || fail("capability", "scaffold_unavailable", "direct_response", "direct_response", "dev3 Direct solve does not support named traces")
    s = ComplexF64[]
    y = ComplexF64[]
    z = ComplexF64[]
    for frequency in frequencies
        response = response_at(compiled, frequency)
        push!(s, response[1]); push!(y, response[2]); push!(z, response[3])
    end
    frequency_artifact = write_real_zarr(staging, "frequencies", frequencies)
    s_artifact = write_complex_zarr(staging, "s", s, compiled.port_id, "dimensionless", "dimensionless")
    y_artifact = write_complex_zarr(staging, "y", y, compiled.port_id, "siemens", "conductance")
    z_artifact = write_complex_zarr(staging, "z", z, compiled.port_id, "ohm", "resistance")
    arrays = Dict{String,Any}(
        "frequencies" => frequency_artifact,
        "s" => s_artifact,
        "y" => y_artifact,
        "z" => z_artifact,
    )
    result = result_envelope("direct_response", request_sha, attempt_sha, Dict{String,Any}(), arrays)
    write_success(staging, request, request_sha, attempt_sha, result, [frequency_artifact, s_artifact, y_artifact, z_artifact])
end

function evaluate_diagonal_root(request, plan, compiled::CompiledPrimitive, request_sha::String, attempt_sha::String, staging::String)
    spec = request["spec"]
    get(spec, "type", nothing) == "diagonal_root" ||
        fail("capability", "scaffold_unavailable", "evaluate_direct", "direct_quantity", "dev3 evaluates only DiagonalRootSpec")
    lineage = request["ref_lineage"]
    retained = lineage["retain"]
    retained === nothing && fail("validation", "port_realizability", "evaluate_direct", "direct_quantity", "DiagonalRootSpec requires a retained one-coordinate View")
    lineage["terminal_coordinates"] == retained["retained_coordinates"] ||
        fail("validation", "port_realizability", "evaluate_direct", "direct_quantity", "DiagonalRootSpec terminal coordinates do not match retain()")
    coordinates = retained["retained_coordinates"]
    length(coordinates) == 1 && coordinates[1] == spec["coordinate"] ||
        fail("validation", "port_realizability", "evaluate_direct", "direct_quantity", "DiagonalRootSpec coordinate must equal the retained View coordinate")
    hint = quantity_value(spec["root_hint"])
    baseline_values = plan_parameter_values(plan)
    candidate_values = parameter_values(request)
    baseline_compiled = compile_primitive(plan, baseline_values)
    baseline_root, baseline_slope = diagonal_root(baseline_compiled, String(spec["coordinate"]), hint)
    if same_parameter_values(baseline_values, candidate_values)
        omega, slope = baseline_root, baseline_slope
    else
        selector = Dict{String,Any}("spec" => spec)
        omega = root_with_continuation(plan, request, baseline_values, candidate_values, baseline_root, selector)
        slope = root_certificate(compiled, omega, String(spec["coordinate"])).fp
    end
    scalars = Dict{String,Any}(
        "root" => complex_quantity(omega, "radian / second", "inverse_time"),
        "frequency" => quantity(real(omega) / (2.0 * pi), "hertz", "inverse_time"),
        "linewidth" => quantity(-2.0 * imag(omega) / (2.0 * pi), "hertz", "inverse_time"),
        "slope" => complex_quantity(slope, "siemens", "conductance"),
    )
    result = result_envelope("diagonal_root", request_sha, attempt_sha, scalars, Dict{String,Any}())
    write_success(staging, request, request_sha, attempt_sha, result, Any[])
end

function leaf_plan_path(request_path::String)::String
    request_dir = dirname(abspath(request_path))
    basename(request_dir) == "" && error("request path is malformed")
    return joinpath(dirname(dirname(request_dir)), "plan.json")
end

function read_request_and_plan(request_path::String)
    request_bytes = read(request_path)
    request_sha = sha256_hex(request_bytes)
    request = plain(JSON3.read(String(request_bytes)))
    get(request, "schema", nothing) == "scnsim.request" || error("request schema discriminator is invalid")
    plan_path = leaf_plan_path(request_path)
    isfile(plan_path) || error("sealed plan.json is absent beside request")
    plan_bytes = read(plan_path)
    file_sha = sha256_hex(plan_bytes)
    request["plan_sha256"] == file_sha || error("request plan SHA-256 does not match sealed plan.json")
    return request, request_sha, plain(JSON3.read(String(plan_bytes)))
end

function staging_ordinal(staging::String)::Int
    match_value = match(r"^\.staging-([0-9]{6,})-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", basename(staging))
    match_value === nothing && error("staging directory does not have canonical attempt name")
    return parse(Int, match_value.captures[1])
end

function bootstrap_record(request_sha::String, ordinal::Int)
    version = string(VERSION)
    version == "1.12.6" || error("SCNSim backend must run under Julia 1.12.6")
    return Dict{String,Any}(
        "attempt_ordinal" => ordinal,
        "blas_threads" => BLAS.get_num_threads(),
        "blas_vendor" => string(BLAS.vendor()),
        "julia_threads" => Threads.nthreads(),
        "julia_version" => version,
        "request_sha256" => request_sha,
        "schema" => "scnsim.bootstrap_ready",
        "schema_version" => 1,
    )
end

function read_authorization(request_sha::String, staging::String, ordinal::Int)
    bytes = read(stdin)
    text = String(bytes)
    count(==('\n'), text) == 1 && endswith(text, "\n") || error("launch authorization must be one JSONL line followed by EOF")
    payload = text[1:end-1]
    authorization = plain(JSON3.read(payload))
    canonical_json(authorization) == payload || error("launch authorization is not canonical JSON")
    get(authorization, "schema", nothing) == "scnsim.launch_authorization" || error("launch authorization schema is invalid")
    get(authorization, "schema_version", nothing) == 1 || error("launch authorization version is invalid")
    authorization["request_sha256"] == request_sha || error("launch authorization request hash mismatches request")
    attempt_path = joinpath(staging, "attempt.json")
    isfile(attempt_path) || error("finalized attempt.json is absent before authorization")
    attempt_bytes = read(attempt_path)
    attempt_sha = sha256_hex(attempt_bytes)
    authorization["attempt_sha256"] == attempt_sha || error("launch authorization attempt hash mismatches attempt.json")
    attempt = plain(JSON3.read(String(attempt_bytes)))
    attempt["request_sha256"] == request_sha || error("attempt request hash mismatches request")
    attempt["ordinal"] == ordinal || error("attempt ordinal mismatches staging directory")
    attempt["attempt_state"] == "launched" || error("attempt is not launched")
    attempt["julia_threads"] == 1 && attempt["blas_threads"] == 1 || error("attempt thread evidence violates dev3 policy")
    return attempt_sha
end

function run_terminal(request_path::String, staging::String)
    request, request_sha, plan = read_request_and_plan(request_path)
    ordinal = staging_ordinal(staging)
    println(canonical_json(bootstrap_record(request_sha, ordinal)))
    flush(stdout)
    attempt_sha = read_authorization(request_sha, staging, ordinal)
    # Python owns this envelope and its canonical encoder. The staged attempt
    # was already bound by the authorization hash, so only parse it here.
    attempt = canonical_document(joinpath(staging, "attempt.json"); require_backend_canonical = false)
    resume_ledger_sha = get(attempt, "resume_ledger_sha256", nothing)
    (resume_ledger_sha === nothing || resume_ledger_sha isa AbstractString) ||
        fail("evidence", "evidence_integrity", "optimization_replay", "attempt", "resume ledger hash is malformed")
    try
        request["runtime_semantic"]["julia_version"] == "1.12.6" || error("request runtime identity has wrong Julia version")
        compiled = compile_primitive(plan, parameter_values(request))
        operation = request["operation"]
        if operation == "solve_direct"
            solve_direct(request, compiled, request_sha, attempt_sha, staging)
        elseif operation == "evaluate_direct"
            evaluate_diagonal_root(request, plan, compiled, request_sha, attempt_sha, staging)
        elseif operation == "optimize_direct"
            optimize_direct(request, plan, request_sha, attempt_sha, staging;
                resume_ledger_sha = resume_ledger_sha)
        else
            fail("capability", "scaffold_unavailable", operation, "scaffold", "operation is outside the dev3 backend slice")
        end
    catch failure
        if failure isa BackendFailure
            write_failure(staging, request, request_sha, attempt_sha, failure)
        else
            write_failure(staging, request, request_sha, attempt_sha,
                BackendFailure("execution", "compiler_invariant", "backend", "compile", sprint(showerror, failure)))
        end
    end
    return nothing
end

function f64_matrix_evidence(matrix::Matrix{Float64})
    values = String[]
    for row in axes(matrix, 1), column in axes(matrix, 2)
        push!(values, f64_hex(matrix[row, column]))
    end
    return Dict("shape" => [size(matrix, 1), size(matrix, 2)], "row_major_f64" => values)
end

function baseline_bindings(plan, compiled::CompiledPrimitive)
    values = Dict{String,Any}[]
    for component in plan["components"]
        realization = component["realization"]
        kind = realization["kind"]
        kind in ("capacitor", "inductor", "resistor") || continue
        parameter = kind == "capacitor" ? "capacitance" : kind == "inductor" ? "inductance" : "resistance"
        value = resolve_binding(realization[parameter], Dict{String,Float64}())
        push!(values, Dict{String,Any}(
            "parameter" => Dict("component_path" => component["component_path"], "parameter_id" => parameter),
            "value" => quantity(value,
                kind == "capacitor" ? "farad" : kind == "inductor" ? "henry" : "ohm",
                kind == "capacitor" ? "capacitance" : kind == "inductor" ? "inductance" : "resistance"),
        ))
    end
    return values
end

function preflight(plan_path::String)
    plan_bytes = read(plan_path)
    plan = plain(JSON3.read(String(plan_bytes)))
    values = Dict{String,Float64}()
    # A pure compile preflight has no request ParameterSet. Primitive baseline bindings
    # are constants in the sealed snapshot; identities correctly fail as request-bound.
    compiled = compile_primitive(plan, values)
    load = zeros(Float64, length(compiled.nodes), length(compiled.nodes))
    load[compiled.port_index, compiled.port_index] = 1.0 / compiled.reference_impedance
    runtime_path = normpath(joinpath(@__DIR__, "..", "runtime.json"))
    runtime = plain(JSON3.read(read(runtime_path, String)))
    return Dict{String,Any}(
        "schema" => "scnsim.preflight",
        "schema_version" => 1,
        "plan_sha256" => sha256_hex(plan_bytes),
        "runtime" => runtime,
        "baseline_bindings" => baseline_bindings(plan, compiled),
        "node_order" => compiled.nodes,
        "matrix_order" => "canonical_node_id",
        "primitive_branch_rows" => compiled.branch_rows,
        "c_matrix" => f64_matrix_evidence(compiled.C),
        "k_matrix" => f64_matrix_evidence(compiled.K),
        "g_matrix" => f64_matrix_evidence(compiled.G),
        "port" => Dict(
            "id" => compiled.port_id,
            "selector_f64" => [f64_hex(index == compiled.port_index ? 1.0 : 0.0) for index in eachindex(compiled.nodes)],
            "reference_admittance" => quantity(1.0 / compiled.reference_impedance, "siemens", "conductance"),
            "load_stamp" => f64_matrix_evidence(load),
            "selected_network_steps" => ["intrinsic_CKG", "port_load_BY0BT", "source_boundary", "power_wave_deembedding"],
        ),
        "root_preflight" => Dict("supported" => "single_retained_coordinate", "algorithm_id" => runtime["algorithm_ids"]["diagonal_root"]),
        "optimization_preflight" => Dict("supported" => "primitive_diagonal_root_frequency_or_linewidth", "algorithm_id" => runtime["algorithm_ids"]["optimization"]),
    )
end

function f64_matrix_hash(matrix::AbstractMatrix{Float64})::String
    # Explicit row/column order avoids Julia's column-major storage becoming evidence authority.
    values = String[]
    for row in axes(matrix, 1), column in axes(matrix, 2)
        push!(values, f64_hex(matrix[row, column]))
    end
    return sha256_hex(canonical_bytes(Dict("shape" => [size(matrix, 1), size(matrix, 2)], "values_f64" => values)))
end

function f64_matrix_projection(matrix::AbstractMatrix{Float64})
    values = String[]
    for row in axes(matrix, 1), column in axes(matrix, 2)
        push!(values, f64_hex(matrix[row, column]))
    end
    return Dict("shape" => [size(matrix, 1), size(matrix, 2)], "values_f64" => values)
end

function state_projection(value)
    if value === nothing
        return nothing
    elseif value isa Float64
        return f64_hex(value)
    elseif value isa Float32
        return f64_hex(Float64(value))
    elseif value isa BigInt
        # Julia's MersenneTwister `adv_jump` is arbitrary precision, so it has
        # no fixed machine width; its canonical signed decimal is exact.
        return string(value)
    elseif value isa Int32
        return string(reinterpret(UInt32, value); base = 16, pad = 8)
    elseif value isa Unsigned
        return string(value; base = 16, pad = 2 * sizeof(value))
    elseif value isa Signed
        return string(value)
    elseif value isa Bool
        return value
    elseif value isa Symbol
        return String(value)
    elseif value isa Tuple
        return Any[state_projection(item) for item in value]
    elseif value isa AbstractVector
        return Any[state_projection(item) for item in value]
    elseif value isa AbstractMatrix
        entries = Any[]
        for row in axes(value, 1), column in axes(value, 2)
            push!(entries, state_projection(value[row, column]))
        end
        return Dict("shape" => [size(value, 1), size(value, 2)],
            "values" => entries)
    elseif isstructtype(typeof(value))
        return Dict{String,Any}(String(field) => state_projection(getfield(value, field)) for field in fieldnames(typeof(value)))
    end
    error("unsupported CMA continuation projection value $(typeof(value))")
end

function continuation_state_sha(optimizer;
        next_raw::Union{Nothing,AbstractMatrix{Float64}} = nothing,
        next_transformed::Union{Nothing,AbstractMatrix{Float64}} = nothing)::String
    (next_raw === nothing) == (next_transformed === nothing) ||
        error("continuation projection requires both next populations or neither")
    parameter = optimizer.p
    projection = Dict{String,Any}(
        "parameters" => Dict(
            "n" => state_projection(parameter.n),
            "lambda" => state_projection(parameter.λ),
            "mean" => state_projection(parameter.mean),
            "sigma" => state_projection(parameter.sigma),
            "covariance" => state_projection(parameter.cov),
            "weights" => state_projection(parameter.weights),
            "constraints" => state_projection(parameter.constraints),
            "noise_handling" => state_projection(parameter.noise_handling),
            "parallel_evaluation" => state_projection(parameter.parallel_evaluation),
            "multi_threading" => state_projection(parameter.multi_threading),
            "seed" => state_projection(parameter.seed),
            "rng" => state_projection(parameter.rng),
        ),
        "stop" => Dict(
            "it" => state_projection(optimizer.stop.it),
            "maxiter" => state_projection(optimizer.stop.maxiter),
            "reason" => state_projection(optimizer.stop.reason),
        ),
    )
    if next_raw !== nothing
        projection["next_population"] = Dict(
            "raw" => f64_matrix_projection(next_raw),
            "transformed" => f64_matrix_projection(next_transformed),
        )
    end
    return sha256_hex(canonical_bytes(projection))
end

function candidate_parameter_set(request, values::Dict{String,Float64})
    bindings = Any[]
    for binding in request["parameters"]["bindings"]
        reference = binding["parameter"]
        key = ref_key(reference)
        haskey(values, key) || fail("execution", "compiler_invariant", "optimization", "optimization_candidate", "candidate is missing a request parameter")
        original = binding["value"]
        push!(bindings, Dict{String,Any}(
            "parameter" => reference,
            "value" => quantity(values[key], String(original["si_unit"]), String(original["dimensionality"])),
        ))
    end
    return Dict{String,Any}("type" => "parameter_set", "bindings" => bindings, "allow_extrapolation" => Any[])
end

function parameter_values_for_z(request, base::Dict{String,Float64}, z::AbstractVector{Float64})
    spec = request["spec"]
    variables = spec["variables"]
    length(variables) == length(z) || fail("execution", "compiler_invariant", "optimization", "optimization_candidate", "optimizer coordinate length mismatches variables")
    result = copy(base)
    for (index, variable) in enumerate(variables)
        coordinate = z[index]
        isfinite(coordinate) && 0.0 <= coordinate <= 1.0 ||
            fail("execution", "invalid_candidate_physical_parameter", "unit_map", "optimization_candidate", "CMA candidate left the declared unit box")
        lower = quantity_value(variable["lower"])
        upper = quantity_value(variable["upper"])
        value = if variable["transform"] == "linear"
            lower + coordinate * (upper - lower)
        elseif variable["transform"] == "log"
            lower * (upper / lower)^coordinate
        else
            fail("validation", "invalid_optimization_spec", "unit_map", "optimization_candidate", "unknown optimization transform")
        end
        isfinite(value) || fail("execution", "invalid_candidate_physical_parameter", "unit_map", "optimization_candidate", "candidate mapping is non-finite")
        result[ref_key(variable["parameter"])] = value
    end
    return result
end

function baseline_z(request, values::Dict{String,Float64})
    coordinates = Float64[]
    for variable in request["spec"]["variables"]
        key = ref_key(variable["parameter"])
        haskey(values, key) || fail("validation", "invalid_optimization_spec", "baseline", "optimization_candidate", "active variable lacks a baseline binding")
        baseline = values[key]
        lower = quantity_value(variable["lower"])
        upper = quantity_value(variable["upper"])
        coordinate = variable["transform"] == "linear" ? (baseline - lower) / (upper - lower) : log(baseline / lower) / log(upper / lower)
        isfinite(coordinate) && 0.0 <= coordinate <= 1.0 ||
            fail("validation", "invalid_optimization_spec", "baseline", "optimization_candidate", "sealed baseline lies outside resolved variable bounds")
        push!(coordinates, coordinate)
    end
    return coordinates
end

function root_selector_key(selector)
    get(selector, "type", nothing) == "diagonal_root_projection" ||
        fail("capability", "scaffold_unavailable", "optimization", "optimization_candidate", "dev3 CMA accepts only diagonal-root selectors")
    return canonical_json(selector["spec"])
end

function root_selector_specs(selector, found::Dict{String,Any} = Dict{String,Any}())
    selector_type = get(selector, "type", nothing)
    if selector_type == "diagonal_root_projection"
        found[root_selector_key(selector)] = selector["spec"]
    elseif selector_type == "quantity_sum"
        terms = get(selector, "terms", nothing)
        terms isa AbstractVector && !isempty(terms) ||
            fail("validation", "invalid_optimization_spec", "quantity_sum", "optimization_candidate", "QuantitySum requires one or more terms")
        for term in terms
            root_selector_specs(term, found)
        end
    else
        fail("capability", "scaffold_unavailable", "optimization", "optimization_candidate", "dev3 CMA accepts only diagonal-root selectors and their QuantitySum")
    end
    return found
end

function root_selector_value(selector, plan, request, baseline_values, values, baseline_roots, roots)::Float64
    selector_type = get(selector, "type", nothing)
    if selector_type == "quantity_sum"
        terms = selector["terms"]
        return sum(root_selector_value(term, plan, request, baseline_values, values, baseline_roots, roots) for term in terms)
    end
    key = root_selector_key(selector)
    root = get!(roots, key) do
        same_parameter_values(baseline_values, values) ? baseline_roots[key] :
            root_with_continuation(plan, request, baseline_values, values, baseline_roots[key], selector)
    end
    projection = selector["projection"]
    if projection == "frequency"
        return real(root) / (2.0 * pi)
    elseif projection == "linewidth"
        return -2.0 * imag(root) / (2.0 * pi)
    end
    fail("capability", "scaffold_unavailable", "optimization", "optimization_candidate", "unsupported diagonal-root projection")
end

function same_parameter_values(left::Dict{String,Float64}, right::Dict{String,Float64})::Bool
    keys(left) == keys(right) || return false
    return all(f64_hex(left[key]) == f64_hex(right[key]) for key in keys(left))
end

function plan_parameter_values(plan)::Dict{String,Float64}
    values = Dict{String,Float64}()
    for component in plan["components"]
        realization = component["realization"]
        kind = realization["kind"]
        kind in ("capacitor", "inductor", "resistor") || continue
        parameter = kind == "capacitor" ? "capacitance" : kind == "inductor" ? "inductance" : "resistance"
        key = join(String.(component["component_path"]), "\u001f") * "\u001e" * parameter
        values[key] = resolve_binding(realization[parameter], Dict{String,Float64}())
    end
    return values
end

function root_with_continuation(plan, request, baseline_values, candidate_values, baseline_root::ComplexF64, selector)
    spec = selector["spec"]
    coordinate = String(spec["coordinate"])
    hint = quantity_value(spec["root_hint"])
    function values_at(t::Float64)
        t == 0.0 && return copy(baseline_values)
        t == 1.0 && return copy(candidate_values)
        result = Dict{String,Float64}()
        for key in keys(baseline_values)
            result[key] = baseline_values[key] + t * (candidate_values[key] - baseline_values[key])
        end
        return result
    end
    function advance(left_t::Float64, left_root::ComplexF64, right_t::Float64, depth::Int)::ComplexF64
        right_values = values_at(right_t)
        try
            return diagonal_root(compile_primitive(plan, right_values), coordinate, hint; start = left_root)[1]
        catch error
            error isa BackendFailure || rethrow()
            # Continuation repairs only a Newton-resolution failure. Structural
            # eliminated-block, slope, capacitance, or physical-domain errors
            # are not alternate branches and must remain their typed failure.
            error.kind == "numerical_resolution_unresolved" || rethrow()
            depth < 32 || rethrow()
            midpoint = (left_t + right_t) / 2.0
            midpoint_root = advance(left_t, left_root, midpoint, depth + 1)
            return advance(midpoint, midpoint_root, right_t, depth + 1)
        end
    end
    return advance(0.0, baseline_root, 1.0, 0)
end

function objective_outcome(plan, request, baseline_values, values, baseline_roots)
    compiled = compile_primitive(plan, values)
    components = Any[]
    total = 0.0
    roots = Dict{String,ComplexF64}()
    for objective in request["spec"]["objectives"]
        selector = objective["quantity"]
        selector isa AbstractDict || fail("capability", "scaffold_unavailable", "optimization", "optimization_candidate", "dev3 CMA does not accept QuantitySum")
        value = root_selector_value(selector, plan, request, baseline_values, values, baseline_roots, roots)
        target = quantity_value(objective["target"])
        scale = quantity_value(objective["resolved_scale"])
        scale > 0.0 || fail("validation", "invalid_optimization_spec", "objective_scale", "optimization_candidate", "objective scale must be positive")
        residual = (value - target) / scale
        weighted = f64_from_hex(objective["weight_f64"]) * abs2(residual)
        isfinite(residual) && isfinite(weighted) || fail("execution", "numerical_resolution_unresolved", "objective", "optimization_candidate", "candidate objective is non-finite")
        total += weighted
        push!(components, Dict{String,Any}(
            "objective_id" => objective["id"],
            "value" => quantity(value, String(objective["target"]["si_unit"]), String(objective["target"]["dimensionality"])),
            "normalized_residual_f64" => f64_hex(residual),
            "weighted_cost_f64" => f64_hex(weighted),
        ))
    end
    isfinite(total) || fail("execution", "numerical_resolution_unresolved", "objective", "optimization_candidate", "candidate total cost is non-finite")
    return total, components
end

function failure_object(request, failure::BackendFailure)
    return Dict{String,Any}(
        "category" => failure.category,
        "kind" => failure.kind,
        "stage" => failure.stage,
        "message" => failure.message,
        "evidence" => Dict(
            "type" => "failure_evidence",
            "operation" => request["operation"],
            "context_kind" => failure.context_kind,
        ),
    )
end

function candidate_record(request, ordinal::Int, generation::Int, column, z::Vector{Float64}, latent, parameters, cache_hit::Bool, outcome)
    record = Dict{String,Any}(
        "evaluation_ordinal" => ordinal,
        "origin" => generation == 0 ? "baseline" : "population",
        "generation" => generation,
        "population_column" => column,
        "optimizer_coordinates_f64" => f64_hex.(z),
        "parameters" => parameters,
        "cache_hit" => cache_hit,
        "extrapolation_evidence" => Any[],
        "outcome" => outcome,
    )
    generation > 0 && latent !== nothing && (record["optimizer_latent_coordinates_f64"] = f64_hex.(latent))
    return record
end

function write_generation_ledger(staging, request, request_sha, attempt_sha, generation::Int, previous_sha, raw::Matrix{Float64}, transformed::Matrix{Float64}, candidates, certificate)
    ledger = Dict{String,Any}(
        "schema" => "scnsim.optimization_ledger",
        "schema_version" => 1,
        "request_sha256" => request_sha,
        "attempt_sha256" => attempt_sha,
        "algorithm_id" => "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v2",
        "generation" => generation,
        "previous_ledger_sha256" => previous_sha,
        "population_size" => size(transformed, 2),
        "raw_optimizer_population_sha256" => f64_matrix_hash(raw),
        "transformed_optimizer_population_sha256" => f64_matrix_hash(transformed),
        "continuation_certificate" => certificate,
        "candidates" => candidates,
    )
    path = joinpath(staging, "artifacts", "generations", lpad(string(generation), 6, '0') * ".json")
    write_bytes(path, canonical_bytes(ledger))
    return Dict{String,Any}(
        "id" => "generation_" * lpad(string(generation), 6, '0'),
        "path" => "artifacts/generations/" * lpad(string(generation), 6, '0') * ".json",
        "sha256" => file_sha256(path),
        "media_type" => "application/json",
        "byte_length" => filesize(path),
    )
end

function canonical_document(path::String; require_backend_canonical::Bool = true)
    bytes = read(path)
    # String(Vector{UInt8}) takes ownership of the vector in Julia. Preserve
    # the original bytes because their exact digest/canonical form is evidence.
    value = plain(JSON3.read(String(copy(bytes))))
    (!require_backend_canonical || canonical_bytes(value) == bytes) ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "prior ledger is not canonical JSON")
    return value
end

function staged_generation_links(staging::String, request_sha::String, attempt_sha::String)
    root = joinpath(staging, "artifacts", "generations")
    isdir(root) || return Any[]
    files = sort(readdir(root; join = true))
    links = Any[]
    previous = nothing
    for (expected, file) in enumerate(files)
        basename(file) == lpad(string(expected), 6, '0') * ".json" && isfile(file) ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "staged generation ledger filename is noncanonical")
        ledger = canonical_document(file)
        digest = file_sha256(file)
        get(ledger, "schema", nothing) == "scnsim.optimization_ledger" &&
            get(ledger, "schema_version", nothing) == 1 &&
            get(ledger, "request_sha256", nothing) == request_sha &&
            get(ledger, "attempt_sha256", nothing) isa AbstractString &&
            get(ledger, "generation", nothing) == expected &&
            get(ledger, "previous_ledger_sha256", nothing) == previous ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "staged generation ledger chain is inconsistent")
        push!(links, Dict("id" => "generation_" * lpad(string(expected), 6, '0'), "sha256" => digest))
        previous = digest
    end
    return links
end

function finalized_attempt_ledgers(entry::String, request_sha::String, current_attempt_sha::String)
    attempt_path = joinpath(entry, "attempt.json")
    receipt_path = joinpath(entry, "receipt.json")
    isfile(attempt_path) && isfile(receipt_path) ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling attempt is not finalized")
    attempt_bytes = read(attempt_path)
    attempt_sha = sha256_hex(attempt_bytes)
    attempt = plain(JSON3.read(String(copy(attempt_bytes))))
    receipt = canonical_document(receipt_path; require_backend_canonical = false)
    get(attempt, "request_sha256", nothing) == request_sha &&
        get(receipt, "request_sha256", nothing) == request_sha &&
        get(receipt, "attempt_sha256", nothing) == attempt_sha ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling attempt receipt does not bind its request and attempt")
    attempt_sha != current_attempt_sha ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "resume ledger cannot come from the current unfinalized attempt")
    get(receipt, "outcome", nothing) in ("failure", "interrupted") ||
        return Dict{String,Dict{String,Any}}()
    declared = Dict{String,String}()
    artifacts = get(receipt, "artifacts", nothing)
    artifacts isa AbstractVector ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling receipt has no artifact inventory")
    for artifact in artifacts
        artifact isa AbstractDict ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling receipt artifact is malformed")
        identifier = get(artifact, "id", nothing)
        digest = get(artifact, "sha256", nothing)
        identifier isa AbstractString && digest isa AbstractString &&
            occursin(r"^generation_[0-9]{6,}$", identifier) && occursin(r"^[0-9a-f]{64}$", digest) ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling receipt declares an invalid generation artifact")
        haskey(declared, identifier) &&
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling receipt declares a generation twice")
        declared[String(identifier)] = String(digest)
    end
    ledger_root = joinpath(entry, "artifacts", "generations")
    isdir(ledger_root) || isempty(declared) ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling receipt ledger directory is absent")
    result = Dict{String,Dict{String,Any}}()
    ledger_files = isdir(ledger_root) ? readdir(ledger_root; join = true) : String[]
    for file in ledger_files
        endswith(file, ".json") && isfile(file) ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "generation artifact is malformed")
        number = splitext(basename(file))[1]
        identifier = "generation_" * number
        haskey(declared, identifier) ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling ledger is not receipt-backed")
        digest = file_sha256(file)
        declared[identifier] == digest ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling ledger digest disagrees with its receipt")
        ledger = canonical_document(file)
        get(ledger, "schema", nothing) == "scnsim.optimization_ledger" &&
            get(ledger, "schema_version", nothing) == 1 &&
            get(ledger, "request_sha256", nothing) == request_sha &&
            get(ledger, "attempt_sha256", nothing) == attempt_sha ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling ledger has incompatible identity")
        result[digest] = ledger
    end
    length(result) == length(declared) ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling receipt ledger inventory has missing files")
    return result
end

function sibling_ledgers(staging::String, request_sha::String, attempt_sha::String, resume_sha::String)
    attempt_root = dirname(staging)
    isdir(attempt_root) || fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "attempt directory is absent")
    discovered = Dict{String,Dict{String,Any}}()
    for entry in readdir(attempt_root; join = true)
        name = basename(entry)
        occursin(r"^(?!000000$)(?:[0-9]{6}|[1-9][0-9]{6,})$", name) || continue
        isdir(entry) || fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "final attempt is not a directory")
        for (digest, ledger) in finalized_attempt_ledgers(entry, request_sha, attempt_sha)
            get(ledger, "algorithm_id", nothing) == "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v2" ||
                fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "prior generation ledger has incompatible algorithm identity")
            if haskey(discovered, digest)
                canonical_json(discovered[digest]) == canonical_json(ledger) ||
                    fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "identical ledger hash names different bytes")
            else
                discovered[digest] = ledger
            end
        end
    end
    haskey(discovered, resume_sha) ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "requested resume ledger is not present in sibling finalized attempts")
    chain = Dict{String,Any}[]
    expected = resume_sha
    while true
        haskey(discovered, expected) ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "resume ledger chain has a missing predecessor")
        ledger = discovered[expected]
        push!(chain, ledger)
        previous = ledger["previous_ledger_sha256"]
        previous === nothing && break
        previous isa AbstractString || fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "resume ledger predecessor is malformed")
        expected = previous
    end
    reverse!(chain)
    for (index, ledger) in enumerate(chain)
        ledger["generation"] == index ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "resume ledger chain has noncontiguous generation ordinals")
        index == 1 || ledger["previous_ledger_sha256"] == file_sha256_of_ledger(chain[index - 1], discovered) ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "resume ledger predecessor linkage mismatches")
    end
    return chain
end

function file_sha256_of_ledger(needle, discovered::Dict{String,Dict{String,Any}})::String
    encoded = canonical_bytes(needle)
    digest = sha256_hex(encoded)
    haskey(discovered, digest) || fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "resume ledger bytes no longer match their digest")
    return digest
end

function stored_cost(candidate)
    outcome = candidate["outcome"]
    if outcome["status"] == "success"
        return f64_from_hex(outcome["cost_f64"])
    elseif outcome["status"] == "failure" && outcome["penalty"] == "positive_infinity"
        return Inf
    end
    fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "stored candidate outcome is malformed")
end

function verify_replayed_population(ledger, transformed::Matrix{Float64})
    f64_matrix_hash(transformed) == ledger["transformed_optimizer_population_sha256"] ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed transformed CMA population differs from verified ledger")
    candidates = ledger["candidates"]
    length(candidates) == size(transformed, 2) ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed candidate count differs from verified ledger")
    for column in axes(transformed, 2)
        candidate = candidates[column]
        candidate["generation"] == ledger["generation"] && candidate["population_column"] == column ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed candidate ordinal differs from verified ledger")
        f64_hex.(collect(@view transformed[:, column])) == candidate["optimizer_coordinates_f64"] ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed candidate coordinates differ from verified ledger")
    end
    return candidates
end

function verify_prior_certificate(chain, generation::Int, optimizer, raw::Matrix{Float64}, transformed::Matrix{Float64})
    previous = generation - 1
    (previous < 1 || previous > length(chain)) && return
    ledger = chain[previous]
    certificate = ledger["continuation_certificate"]
    certificate["completed_generation"] == previous ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "resume continuation certificate names the wrong generation")
    boundary = certificate["boundary"]
    if boundary == "post_update_post_next_sample_pre_next_update"
        certificate["state_sha256"] == continuation_state_sha(optimizer;
            next_raw = raw, next_transformed = transformed) ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed CMA continuation state differs before update")
        certificate["next_raw_optimizer_population_sha256"] == f64_matrix_hash(raw) ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed next raw CMA population differs from certificate")
        certificate["next_transformed_optimizer_population_sha256"] == f64_matrix_hash(transformed) ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed next transformed CMA population differs from certificate")
    elseif boundary == "terminal_post_update"
        certificate["state_sha256"] == continuation_state_sha(optimizer) ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed CMA terminal continuation state differs")
        previous == length(chain) ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "only the terminal resume ledger may use a terminal certificate")
    else
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "resume continuation certificate has unknown boundary")
    end
end

function verify_terminal_certificate(chain, optimizer)
    isempty(chain) && return
    certificate = chain[end]["continuation_certificate"]
    certificate["boundary"] == "terminal_post_update" || return
    certificate["state_sha256"] == continuation_state_sha(optimizer) ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed terminal CMA continuation state differs")
end

function materialize_replayed_ledgers(staging::String, chain)
    artifacts = Any[]
    for (generation, ledger) in enumerate(chain)
        ledger["generation"] == generation ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed ledger generation is noncontiguous")
        bytes = canonical_bytes(ledger)
        digest = sha256_hex(bytes)
        path = joinpath(staging, "artifacts", "generations", lpad(string(generation), 6, '0') * ".json")
        write_bytes(path, bytes)
        push!(artifacts, Dict{String,Any}(
            "id" => "generation_" * lpad(string(generation), 6, '0'),
            "path" => "artifacts/generations/" * lpad(string(generation), 6, '0') * ".json",
            "sha256" => digest,
            "media_type" => "application/json",
            "byte_length" => length(bytes),
        ))
    end
    return artifacts
end

function seed_replay_cache!(cache::Dict{String,Any}, chain)
    for ledger in chain
        for candidate in ledger["candidates"]
            key = canonical_json(candidate["parameters"])
            cost = stored_cost(candidate)
            if haskey(cache, key)
                previous = cache[key]
                previous_cost = previous["cost"]
                same_cost = (isinf(previous_cost) && isinf(cost)) ||
                    (!isinf(previous_cost) && !isinf(cost) && f64_hex(previous_cost) == f64_hex(cost))
                same_cost || fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed candidate cache assigns inconsistent costs")
            else
                cache[key] = Dict("outcome" => candidate["outcome"], "cost" => cost)
            end
        end
    end
    return nothing
end

function verify_callback_costs(expected::Vector{Float64}, received)
    length(expected) == length(received) ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "CMA callback cost count differs from evaluated population")
    for index in eachindex(expected)
        same = (isinf(expected[index]) && isinf(received[index])) ||
            (!isinf(expected[index]) && !isinf(received[index]) && f64_hex(expected[index]) == f64_hex(received[index]))
        same || fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "CMA callback costs differ from stored/re-evaluated candidate costs")
    end
    return nothing
end

function optimization_baseline_roots(plan, request, values)
    compiled = compile_primitive(plan, values)
    roots = Dict{String,ComplexF64}()
    for objective in request["spec"]["objectives"]
        selector = objective["quantity"]
        for (key, root_spec) in root_selector_specs(selector)
            if !haskey(roots, key)
                roots[key] = diagonal_root(compiled, String(root_spec["coordinate"]), quantity_value(root_spec["root_hint"]))[1]
            end
        end
    end
    return roots
end

function candidate_from_z(plan, request, baseline_values, baseline_roots, z::Vector{Float64}, latent, generation::Int, column::Int, ordinal::Int, cache::Dict{String,Any})
    values = parameter_values_for_z(request, baseline_values, z)
    parameters = candidate_parameter_set(request, values)
    key = canonical_json(parameters)
    if haskey(cache, key)
        cached = cache[key]
        return candidate_record(request, ordinal, generation, column, z, latent, parameters, true, cached["outcome"]), cached["cost"]
    end
    try
        cost, components = objective_outcome(plan, request, baseline_values, values, baseline_roots)
        outcome = Dict{String,Any}(
            "status" => "success",
            "cost_f64" => f64_hex(cost),
            "objective_components" => components,
        )
        cache[key] = Dict("outcome" => outcome, "cost" => cost)
        return candidate_record(request, ordinal, generation, column, z, latent, parameters, false, outcome), cost
    catch error
        error isa BackendFailure || rethrow()
        allowed = error.kind in (
            "invalid_candidate_physical_parameter",
            "eliminated_block_solve_failure",
            "root_slope_unresolved",
            "numerical_resolution_unresolved",
        )
        allowed || rethrow()
        outcome = Dict{String,Any}(
            "status" => "failure",
            "penalty" => "positive_infinity",
            "failure" => failure_object(request, error),
        )
        cache[key] = Dict("outcome" => outcome, "cost" => Inf)
        return candidate_record(request, ordinal, generation, column, z, latent, parameters, false, outcome), Inf
    end
end

function emit_progress(request_sha::String, attempt_sha::String, generation::Int, evaluations::Int, maximum::Int)
    println(canonical_json(Dict{String,Any}(
        "schema" => "scnsim.progress",
        "schema_version" => 1,
        "request_sha256" => request_sha,
        "attempt_sha256" => attempt_sha,
        "event" => "optimization_generation_complete",
        "completed_generation" => generation,
        "completed_evaluations" => evaluations,
        "max_evaluations" => maximum,
    )))
    flush(stdout)
end

function optimize_direct(request, plan, request_sha::String, attempt_sha::String, staging::String;
        resume_ledger_sha::Union{Nothing,AbstractString} = nothing)
    spec = request["spec"]
    controls = spec["optimizer"]
    variables = spec["variables"]
    n = length(variables)
    n > 0 || fail("validation", "invalid_optimization_spec", "variables", "optimization_candidate", "optimization requires at least one variable")
    lambda = Int(controls["resolved_population_size"])
    budget = Int(controls["max_evaluations"])
    generations = Int(controls["complete_generations"])
    expected_generations = (budget - 1) ÷ lambda
    lambda >= 2 && generations >= 1 && generations == expected_generations ||
        fail("validation", "invalid_optimization_spec", "controls", "optimization_candidate", "optimization controls do not describe complete CMA generations")
    controls["unused_evaluations"] == budget - (1 + generations * lambda) ||
        fail("validation", "invalid_optimization_spec", "controls", "optimization_candidate", "optimization unused-evaluation evidence is inconsistent")
    controls["box_transform_id"] == "cmaes-jl-0.2.6-linquad-unit-box.v1" ||
        fail("validation", "invalid_optimization_spec", "controls", "optimization_candidate", "optimization box transform is unsupported")
    controls["hidden_stops"] == "disabled" ||
        fail("validation", "invalid_optimization_spec", "controls", "optimization_candidate", "optimization hidden stops must be disabled")
    sigma = f64_from_hex(controls["initial_sigma_f64"])
    isfinite(sigma) && sigma > 0.0 || fail("validation", "invalid_optimization_spec", "controls", "optimization_candidate", "initial sigma must be finite and positive")
    seed = Int64(controls["seed"])
    base_values = parameter_values(request)
    z0 = baseline_z(request, base_values)
    baseline_roots = optimization_baseline_roots(plan, request, base_values)
    cache = Dict{String,Any}()
    baseline_parameters = candidate_parameter_set(request, base_values)
    baseline_cost, baseline_components = try
        objective_outcome(plan, request, base_values, base_values, baseline_roots)
    catch error
        error isa BackendFailure || rethrow()
        rethrow()
    end
    baseline_outcome = Dict{String,Any}(
        "status" => "success",
        "cost_f64" => f64_hex(baseline_cost),
        "objective_components" => baseline_components,
    )
    baseline = candidate_record(request, 0, 0, nothing, z0, nothing, baseline_parameters, false, baseline_outcome)
    cache[canonical_json(baseline_parameters)] = Dict("outcome" => baseline_outcome, "cost" => baseline_cost)
    replay_chain = resume_ledger_sha === nothing ? Dict{String,Any}[] :
        sibling_ledgers(staging, request_sha, attempt_sha, String(resume_ledger_sha))
    length(replay_chain) <= generations ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "resume ledger chain exceeds this request's complete generation count")
    seed_replay_cache!(cache, replay_chain)
    best = baseline
    best_cost = baseline_cost
    for ledger in replay_chain, record in ledger["candidates"]
        cost = stored_cost(record)
        if isfinite(cost) && cost < best_cost
            best = record
            best_cost = cost
        end
    end
    batches = Ref{Any}(nothing)
    pending = Ref{Any}(nothing)
    replay_artifacts = materialize_replayed_ledgers(staging, replay_chain)
    prior_ledger = Ref{Union{Nothing,String}}(
        isempty(replay_chain) ? nothing : sha256_hex(canonical_bytes(replay_chain[end])))
    ledger_artifacts = Any[replay_artifacts...]
    evaluation_ordinal = Ref(1 + length(replay_chain) * lambda)

    function objective(transformed::AbstractMatrix{Float64})
        generation = (batches[] === nothing ? 1 : batches[]["generation"] + 1)
        if generation <= length(replay_chain)
            ledger = replay_chain[generation]
            records = verify_replayed_population(ledger, Matrix{Float64}(transformed))
            costs = Float64[stored_cost(record) for record in records]
            for (record, cost) in zip(records, costs)
                if isfinite(cost) && cost < best_cost
                    best = record
                    best_cost = cost
                end
            end
            batches[] = Dict(
                "generation" => generation,
                "transformed" => copy(transformed),
                "records" => records,
                "costs" => costs,
                "replay_ledger" => ledger,
            )
            return costs
        end
        records = Any[]
        costs = Float64[]
        for column in axes(transformed, 2)
            z = collect(@view transformed[:, column])
            record, cost = candidate_from_z(plan, request, base_values, baseline_roots, z, nothing, generation, column, evaluation_ordinal[], cache)
            push!(records, record)
            push!(costs, cost)
            if isfinite(cost) && cost < best_cost
                best = record
                best_cost = cost
            end
            evaluation_ordinal[] += 1
        end
        batches[] = Dict(
            "generation" => generation,
            "transformed" => copy(transformed),
            "records" => records,
            "costs" => costs,
            "replay_ledger" => nothing,
        )
        return costs
    end

    function callback(optimizer, raw, costs, permutation)
        current = batches[]
        current === nothing && error("CMA callback arrived without objective batch")
        transformed = CMAEvolutionStrategy.compute_input(optimizer.p, raw)
        f64_matrix_hash(transformed) == f64_matrix_hash(current["transformed"]) ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "CMA transformed population mismatches the evaluated candidate matrix")
        verify_callback_costs(current["costs"], costs)
        verify_prior_certificate(replay_chain, current["generation"], optimizer, raw, transformed)
        records = current["records"]
        if current["replay_ledger"] !== nothing
            ledger = current["replay_ledger"]
            f64_matrix_hash(raw) == ledger["raw_optimizer_population_sha256"] ||
                fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed raw CMA population differs from verified ledger")
            for column in axes(raw, 2)
                f64_hex.(collect(@view raw[:, column])) == records[column]["optimizer_latent_coordinates_f64"] ||
                    fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "replayed latent candidate coordinates differ from verified ledger")
            end
            # Immutable replay evidence remains byte-identical to the prior
            # attempt.  It must never become pending new-attempt evidence.
            batches[] = current
            return nothing
        else
            for column in axes(raw, 2)
                records[column]["optimizer_latent_coordinates_f64"] = f64_hex.(collect(@view raw[:, column]))
            end
        end
        if pending[] !== nothing
            previous = pending[]
            certificate = Dict{String,Any}(
                "schema" => "scnsim.cmaes_continuation_certificate",
                "schema_version" => 1,
                "projection_id" => "cmaes-jl-0.2.6-julia-1.12.6-continuation-state.v1",
                "boundary" => "post_update_post_next_sample_pre_next_update",
                "completed_generation" => previous["generation"],
                "state_sha256" => continuation_state_sha(optimizer;
                    next_raw = raw, next_transformed = transformed),
                "next_raw_optimizer_population_sha256" => f64_matrix_hash(raw),
                "next_transformed_optimizer_population_sha256" => f64_matrix_hash(transformed),
            )
            artifact = write_generation_ledger(staging, request, request_sha, attempt_sha,
                previous["generation"], prior_ledger[], previous["raw"], previous["transformed"], previous["records"], certificate)
            push!(ledger_artifacts, artifact)
            prior_ledger[] = artifact["sha256"]
            emit_progress(request_sha, attempt_sha, previous["generation"], 1 + previous["generation"] * lambda, budget)
        end
        pending[] = Dict("generation" => current["generation"], "raw" => copy(raw), "transformed" => copy(transformed), "records" => records)
        batches[] = current
    end

    optimizer = CMAEvolutionStrategy.minimize(
        objective,
        z0,
        sigma;
        lower = zeros(n), upper = ones(n), popsize = lambda, maxiter = generations,
        maxfevals = nothing, parallel_evaluation = true, multi_threading = false,
        verbosity = 0, seed = reinterpret(UInt64, seed), callback = callback,
        ftol = nothing, xtol = nothing, stagnation = nothing, ftarget = nothing,
        maxtime = nothing, noise_handling = nothing,
    )
    optimizer.stop.it == generations && optimizer.stop.reason == :maxiter ||
        fail("execution", "compiler_invariant", "optimization", "optimization_candidate", "pinned CMA package stopped outside the declared complete-generation policy")
    terminal = pending[]
    if terminal === nothing
        length(replay_chain) == generations || error("CMA completed without a new or replayed population")
        verify_terminal_certificate(replay_chain, optimizer)
    else
        terminal_certificate = Dict{String,Any}(
            "schema" => "scnsim.cmaes_continuation_certificate",
            "schema_version" => 1,
            "projection_id" => "cmaes-jl-0.2.6-julia-1.12.6-continuation-state.v1",
            "boundary" => "terminal_post_update",
            "completed_generation" => terminal["generation"],
            "state_sha256" => continuation_state_sha(optimizer),
        )
        artifact = write_generation_ledger(staging, request, request_sha, attempt_sha,
            terminal["generation"], prior_ledger[], terminal["raw"], terminal["transformed"], terminal["records"], terminal_certificate)
        push!(ledger_artifacts, artifact)
        emit_progress(request_sha, attempt_sha, terminal["generation"], 1 + terminal["generation"] * lambda, budget)
    end
    result = Dict{String,Any}(
        "schema" => "scnsim.result",
        "schema_version" => 1,
        "result_kind" => "optimization",
        "request_sha256" => request_sha,
        "attempt_sha256" => attempt_sha,
        "baseline" => baseline,
        "best" => Dict(
            "evaluation_ordinal" => best["evaluation_ordinal"],
            "cost_f64" => best["outcome"]["cost_f64"],
            "parameters" => best["parameters"],
        ),
        "completed_generations" => generations,
        "unused_evaluations" => controls["unused_evaluations"],
        "ledger_artifacts" => ledger_artifacts,
    )
    write_success(staging, request, request_sha, attempt_sha, result, ledger_artifacts)
end


end # module
