# Included into the single SCNSimBackend module.
# Canonical wire primitives and request parameter decoding.

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
        # Python's identity encoder NFC-normalizes every JSON string before it
        # sorts or writes bytes.  Do the same here, including object keys; a
        # decomposed/composed key collision is an invalid closed envelope.
        normalized = Dict{String,Any}()
        for (raw_key, item) in pairs(value)
            key = Unicode.normalize(String(raw_key), :NFC)
            haskey(normalized, key) && error("NFC-normalized object keys collide")
            normalized[key] = item
        end
        keys_sorted = sort!(collect(keys(normalized)))
        return "{" * join((JSON3.write(key) * ":" * canonical_json(normalized[key]) for key in keys_sorted), ",") * "}"
    elseif value isa AbstractVector || value isa Tuple
        return "[" * join((canonical_json(item) for item in value), ",") * "]"
    elseif value === nothing
        return "null"
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa AbstractString
        return String(JSON3.write(Unicode.normalize(String(value), :NFC)))
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

function quantity_matrix(matrix::Matrix{Float64}, unit::String, dimensionality::String)
    return Dict{String,Any}(
        "type" => "quantity_matrix_f64", "shape" => [size(matrix, 1), size(matrix, 2)],
        "values_f64" => [f64_hex(matrix[row, column]) for row in axes(matrix, 1) for column in axes(matrix, 2)],
        "si_unit" => unit, "dimensionality" => dimensionality,
    )
end

complex_quantity(value::ComplexF64, unit::String, dimensionality::String) = Dict{String,Any}(
    "type" => "complex_quantity_f64",
    "real_si_f64" => f64_hex(real(value)),
    "imag_si_f64" => f64_hex(imag(value)),
    "si_unit" => unit,
    "dimensionality" => dimensionality,
)

function ref_key(reference)::String
    item = plain(reference)
    return String(item["definitions_id"]) * "\u001e" * String(item["parameter_id"])
end

function request_parameter_set(request)
    source = plain(request)["parameter_source"]
    kind = String(source["kind"])
    if kind == "point"
        exact_keys(source, ("kind", "parameters")) || fail("execution", "compiler_invariant", "parameter_source", "compile", "point parameter-source fields are invalid")
        return source["parameters"]
    elseif kind == "grid"
        exact_keys(source, ("kind", "base_parameters", "axes", "shape")) || fail("execution", "compiler_invariant", "parameter_source", "compile", "grid parameter-source fields are invalid")
        return source["base_parameters"]
    elseif kind == "points"
        exact_keys(source, ("kind", "baseline_parameters", "points")) || fail("execution", "compiler_invariant", "parameter_source", "compile", "listed parameter-source fields are invalid")
        return source["baseline_parameters"]
    end
    fail("execution", "compiler_invariant", "compile", "compile", "unknown parameter source kind")
end

parameter_values(request)::Dict{String,Any} = structured_parameter_values(request_parameter_set(request))

function parameter_set_authorizations(request)::Set{String}
    return structured_authorizations(request_parameter_set(request))
end
