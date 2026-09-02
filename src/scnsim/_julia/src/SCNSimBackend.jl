module SCNSimBackend

using CMAEvolutionStrategy
using JSON3
using JosephsonCircuits
using LinearAlgebra
using Random
using SHA
using Unicode

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

function parameter_set_authorizations(request)::Set{String}
    parameters = plain(request)["parameters"]
    refs = get(parameters, "allow_extrapolation", Any[])
    refs isa AbstractVector || fail("execution", "compiler_invariant", "affine_support", "compile", "ParameterSet authorization collection is malformed")
    return Set(ref_key(reference) for reference in refs)
end

function resolve_binding(binding, values::Dict{String,Float64}; context_kind::String = "compile",
        authorized::Set{String} = Set{String}(), extrapolation_evidence::Union{Nothing,Vector{Any}} = nothing,
        consumer_target = nothing, authorization_source::String = "none")::Float64
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
        support = item["support"]
        length(support) == 2 || fail("execution", "compiler_invariant", "compile", "compile", "affine support must have two bounds")
        lower = quantity_value(support[1]); upper = quantity_value(support[2]); input = values[key]
        if !(lower <= input <= upper)
            consumer_target === nothing && fail("execution", "compiler_invariant", "affine_support", "compile", "authorized affine edge has no sealed consumer target")
            side, distance = input < lower ? ("lower", lower - input) : ("upper", input - upper)
            extrapolation_evidence === nothing || push!(extrapolation_evidence, Dict{String,Any}(
                "parameter" => plain(item["input"]), "consumer_target" => plain(consumer_target),
                "support" => Any[plain(support[1]), plain(support[2])],
                "input_value" => quantity(input, String(support[1]["si_unit"]), String(support[1]["dimensionality"])),
                "side" => side,
                "distance" => quantity(distance, String(support[1]["si_unit"]), String(support[1]["dimensionality"])),
                "authorization_source" => (key in authorized ? authorization_source : "none"),
            ))
            key in authorized || fail("execution", "invalid_candidate_physical_parameter", "affine_support", context_kind, "affine input is outside its declared support")
        end
        return quantity_value(item["slope"]) * input + quantity_value(item["intercept"])
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

component_path(component) = String.(component["component_path"])
component_key(component) = join(component_path(component), "\u001f")
endpoint_at(path::Vector{String}, pin) = endpoint_key(Dict("component_path" => path, "pin_id" => String(pin)))
qualified_node(path::Vector{String}, id) = join(path, "\u001f") * "\u001e" * String(id)

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

function binding_value(component, parameter_id::String, binding, values::Dict{String,Float64}; context_kind::String = "compile",
        authorized::Set{String} = Set{String}(), extrapolation_evidence::Union{Nothing,Vector{Any}} = nothing,
        authorization_source::String = "none")::Float64
    public_reference = Dict("component_path" => component["component_path"], "parameter_id" => parameter_id)
    key = ref_key(public_reference)
    return haskey(values, key) ? values[key] : resolve_binding(binding, values; context_kind = context_kind,
        authorized = authorized, extrapolation_evidence = extrapolation_evidence, consumer_target = public_reference,
        authorization_source = authorization_source)
end

function binding_for(component, parameter_id::String, realization)
    haskey(realization, parameter_id) && return realization[parameter_id]
    for entry in get(component, "parameter_bindings", Any[])
        entry["id"] == parameter_id && return entry["binding"]
    end
    fail("execution", "compiler_invariant", "compile", "compile", "sealed component is missing parameter binding $(parameter_id)")
end

function component_endpoint_incidences(component, endpoint_to_node::Dict{String,String}, node_index::Dict{String,Int})
    pins = component["pin_order"]
    length(pins) == 2 || fail("execution", "compiler_invariant", "compile", "compile", "expanded primitive must have exactly two ordered pins")
    path = component_path(component)
    left = endpoint_at(path, pins[1]); right = endpoint_at(path, pins[2])
    haskey(endpoint_to_node, left) || fail("execution", "compiler_invariant", "compile", "compile", "expanded terminal_1 is unbound")
    haskey(endpoint_to_node, right) || fail("execution", "compiler_invariant", "compile", "compile", "expanded terminal_2 is unbound")
    positive, negative = zeros(Float64, length(node_index)), zeros(Float64, length(node_index))
    endpoint_to_node[left] != "ground" && (positive[node_index[endpoint_to_node[left]]] = 1.0)
    endpoint_to_node[right] != "ground" && (negative[node_index[endpoint_to_node[right]]] = 1.0)
    return positive, negative
end

function component_incidence(component, endpoint_to_node::Dict{String,String}, node_index::Dict{String,Int})
    positive, negative = component_endpoint_incidences(component, endpoint_to_node, node_index)
    return positive .- negative
end

"""Compile the sealed primitive snapshot. Ports remain outside intrinsic C/K/G."""
function compile_primitive(plan_value, values::Dict{String,Float64}; context_kind::String = "compile",
        authorized::Set{String} = Set{String}(), extrapolation_evidence::Union{Nothing,Vector{Any}} = nothing,
        authorization_source::String = "none", emit_audit::Bool = false)::CompiledPrimitive
    return compile_recursive(plan_value, values; context_kind = context_kind, authorized = authorized,
        extrapolation_evidence = extrapolation_evidence, authorization_source = authorization_source,
        emit_audit = emit_audit)
end

branch_key(reference) = join(String.(plain(reference)["component_path"]), "\u001f") * "\u001e" * String(plain(reference)["branch_id"])

function composite_child(container, endpoint)
    path = String.(endpoint["component_path"])
    matches = [child for child in container["realization"]["children"] if component_path(child) == path]
    length(matches) == 1 || fail("execution", "compiler_invariant", "compile", "compile", "Composite endpoint does not resolve to one immediate child")
    return only(matches)
end

function composite_private_node(realization, private_id::String)
    matches = [node for node in realization["private_nodes"] if String(node["id"]) == private_id]
    length(matches) == 1 || fail("execution", "compiler_invariant", "compile", "compile", "Composite public map does not resolve to one private node")
    return only(matches)
end

function expand_private_endpoint!(leaves::Vector{Dict{String,Any}}, container, endpoint, ancestry::Set{String})
    key = endpoint_key(endpoint)
    key in ancestry && fail("execution", "compiler_invariant", "compile", "compile", "Composite private-node expansion is cyclic or duplicates an endpoint")
    push!(ancestry, key)
    child = composite_child(container, endpoint)
    realization = child["realization"]
    if String(realization["kind"]) != "composite"
        push!(leaves, Dict{String,Any}("component_path" => String.(endpoint["component_path"]), "pin_id" => String(endpoint["pin_id"])))
    else
        mappings = [item for item in realization["public_pin_map"] if String(item["public_id"]) == String(endpoint["pin_id"])]
        length(mappings) == 1 || fail("execution", "compiler_invariant", "compile", "compile", "Composite child endpoint lacks one public-pin map")
        private_node = composite_private_node(realization, String(only(mappings)["private_node_id"]))
        for nested_endpoint in private_node["endpoints"]
            expand_private_endpoint!(leaves, child, nested_endpoint, ancestry)
        end
    end
    delete!(ancestry, key)
    return nothing
end

function expanded_internal_node_id(container, private_node)::String
    leaves = Dict{String,Any}[]
    for endpoint in private_node["endpoints"]
        expand_private_endpoint!(leaves, container, endpoint, Set{String}())
    end
    sort!(leaves; by = endpoint -> (Tuple(String.(endpoint["component_path"])), String(endpoint["pin_id"])))
    any(endpoint_key(leaves[index - 1]) == endpoint_key(leaves[index]) for index in 2:length(leaves)) &&
        fail("execution", "compiler_invariant", "compile", "compile", "Composite private-node expansion duplicates a leaf endpoint")
    return "internal-" * sha256_hex(canonical_bytes(Dict(
        "schema" => "scnsim.internal_node",
        "schema_version" => 1,
        "endpoints" => leaves,
    )))
end

function recursive_nodes!(nodes::Vector{String}, component; top_level::Bool)
    realization = component["realization"]
    if String(realization["kind"]) == "transmission_line"
        conductors = String.(realization["pin_conductors"])
        sections = Int(realization["n_sections"])
        for station in 1:(sections - 1), conductor in conductors
            push!(nodes, "internal-" * sha256_hex(canonical_bytes(Dict(
                "schema" => "scnsim.line_station", "schema_version" => 1,
                "component_path" => component_path(component), "station" => station, "conductor" => conductor,
            ))))
        end
        return nothing
    end
    String(realization["kind"]) == "composite" || return nothing
    path = component_path(component)
    coordinate_targets = top_level ? Dict{String,String}(String(item["private_node_id"]) => String(item["public_id"])
        for item in realization["public_coordinate_map"]) : Dict{String,String}()
    pin_targets = Set(String(item["private_node_id"]) for item in realization["public_pin_map"])
    for private_node in realization["private_nodes"]
        id = String(private_node["id"])
        id in pin_targets && continue
        push!(nodes, get(coordinate_targets, id, expanded_internal_node_id(component, private_node)))
    end
    for child in realization["children"]
        recursive_nodes!(nodes, child; top_level = false)
    end
    return nothing
end

function validate_composite_maps!(component)
    realization = component["realization"]
    String(realization["kind"]) == "composite" || return nothing
    for (field, message) in (
        ("public_pin_map", "Composite public pin map repeats a private node target"),
        ("public_coordinate_map", "Composite public coordinate map repeats a private node target"),
    )
        targets = String[item["private_node_id"] for item in realization[field]]
        length(targets) == length(unique(targets)) ||
            fail("execution", "compiler_invariant", "compile", "compile", message)
    end
    for child in realization["children"]
        validate_composite_maps!(child)
    end
    return nothing
end

function recursive_parameter_values!(values::Dict{String,Float64}, component,
    mapped_targets::Set{String}, context_kind::String, authorized::Set{String}, extrapolation_evidence::Union{Nothing,Vector{Any}}, authorization_source::String)
    for entry in component["parameter_bindings"]
        key = ref_key(Dict("component_path" => component["component_path"], "parameter_id" => entry["id"]))
        haskey(values, key) && continue
        values[key] = resolve_binding(entry["binding"], values; context_kind = context_kind, authorized = authorized,
            extrapolation_evidence = extrapolation_evidence,
            consumer_target = Dict("component_path" => component["component_path"], "parameter_id" => entry["id"]), authorization_source = authorization_source)
    end
    realization = component["realization"]
    String(realization["kind"]) == "composite" || return nothing
    for mapping in realization["public_parameter_maps"]
        source = ref_key(mapping["parameter"])
        haskey(values, source) || fail("execution", "compiler_invariant", "compile", "compile", "Composite public parameter map has no resolved source")
        for consumer in mapping["consumers"]
            target = ref_key(consumer["target"])
            target in mapped_targets && fail("execution", "compiler_invariant", "compile", "compile", "Composite public parameter map duplicates a consumer target")
            push!(mapped_targets, target)
            mapped = resolve_binding(consumer["binding"], values; context_kind = context_kind, authorized = authorized,
                extrapolation_evidence = extrapolation_evidence, consumer_target = consumer["target"], authorization_source = authorization_source)
            if haskey(values, target)
                f64_hex(values[target]) == f64_hex(mapped) ||
                    fail("execution", "compiler_invariant", "compile", "compile", "Composite public parameter map conflicts with an existing resolved target")
            else
                values[target] = mapped
            end
        end
    end
    for child in realization["children"]
        recursive_parameter_values!(values, child, mapped_targets, context_kind, authorized, extrapolation_evidence, authorization_source)
    end
    return nothing
end

function recursive_parameter_values(plan, request_values::Dict{String,Float64}; context_kind::String = "compile",
        authorized::Set{String} = Set{String}(), extrapolation_evidence::Union{Nothing,Vector{Any}} = nothing,
        authorization_source::String = "none")
    values = copy(request_values)
    mapped_targets = Set{String}()
    for component in plan["components"]
        recursive_parameter_values!(values, component, mapped_targets, context_kind, authorized, extrapolation_evidence, authorization_source)
    end
    return values
end

recursive_baselines(plan) = recursive_parameter_values(plan, Dict{String,Float64}())

function find_component_by_path!(matches::Vector{Any}, component, path::Vector{String})
    component_path(component) == path && push!(matches, component)
    realization = component["realization"]
    String(realization["kind"]) == "composite" || return nothing
    for child in realization["children"]
        find_component_by_path!(matches, child, path)
    end
    return nothing
end

function lower_branch_reference(plan, reference; ancestry::Set{String} = Set{String}())
    item = plain(reference); key = branch_key(item)
    key in ancestry && fail("execution", "compiler_invariant", "compile", "compile", "Composite public inductive branch map is cyclic")
    push!(ancestry, key)
    path = String.(item["component_path"])
    matches = Any[]
    for component in plan["components"]
        find_component_by_path!(matches, component, path)
    end
    length(matches) == 1 || fail("execution", "compiler_invariant", "compile", "compile", "inductive branch reference does not resolve to one sealed component")
    component = only(matches); realization = component["realization"]
    if String(realization["kind"]) == "composite"
        branch_maps = [mapping for mapping in realization["public_inductive_branch_map"] if String(mapping["public_id"]) == String(item["branch_id"])]
        length(branch_maps) == 1 || fail("execution", "compiler_invariant", "compile", "compile", "Composite public inductive branch map does not resolve to one target")
        result = lower_branch_reference(plan, only(branch_maps)["target"]; ancestry = ancestry)
        delete!(ancestry, key)
        return result
    end
    branches = [branch for branch in component["inductive_branches"] if String(branch["id"]) == String(item["branch_id"])]
    length(branches) == 1 || fail("execution", "compiler_invariant", "compile", "compile", "inductive branch reference does not resolve to one leaf branch")
    delete!(ancestry, key)
    return Dict("component_path" => path, "branch_id" => String(item["branch_id"]))
end

function recursive_incidence(positive, negative, endpoint_to_node::Dict{String,String}, node_index::Dict{String,Int})
    left = endpoint_key(positive); right = endpoint_key(negative)
    haskey(endpoint_to_node, left) || fail("execution", "compiler_invariant", "compile", "compile", "expanded positive inductive endpoint is unbound")
    haskey(endpoint_to_node, right) || fail("execution", "compiler_invariant", "compile", "compile", "expanded negative inductive endpoint is unbound")
    b = zeros(Float64, length(node_index))
    endpoint_to_node[left] != "ground" && (b[node_index[endpoint_to_node[left]]] += 1.0)
    endpoint_to_node[right] != "ground" && (b[node_index[endpoint_to_node[right]]] -= 1.0)
    return b
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

function line_station_node(component, station::Int, conductor::String, endpoint_to_node::Dict{String,String})::String
    path = component_path(component)
    sections = Int(component["realization"]["n_sections"])
    if station == 0 || station == sections
        end_id = station == 0 ? "head" : "tail"
        endpoint = endpoint_at(path, end_id * "." * conductor)
        haskey(endpoint_to_node, endpoint) || fail("execution", "compiler_invariant", "compile", "compile", "transmission-line endpoint is unbound")
        return endpoint_to_node[endpoint]
    end
    return "internal-" * sha256_hex(canonical_bytes(Dict(
        "schema" => "scnsim.line_station", "schema_version" => 1,
        "component_path" => path, "station" => station, "conductor" => conductor,
    )))
end

function line_station_incidence(component, station::Int, conductors::Vector{String}, endpoint_to_node, node_index)
    B = zeros(Float64, length(node_index), length(conductors))
    for (column, conductor) in enumerate(conductors)
        node = line_station_node(component, station, conductor, endpoint_to_node)
        node == "ground" || (B[node_index[node], column] = 1.0)
    end
    return B
end

function recursive_leaf!(component, endpoint_to_node, node_index, values, capacitors, resistors, inductors, capacitance_blocks, conductance_blocks, series_rl, rows, context_kind::String;
        emit_audit::Bool = false)
    realization = component["realization"]; kind = String(realization["kind"])
    path = component_path(component); pins = component["pin_order"]
    if kind == "transmission_line"
        conductors = String.(realization["pin_conductors"])
        sections = Int(realization["n_sections"])
        length(pins) == 2 * length(conductors) && sections >= 1 ||
            fail("execution", "compiler_invariant", "compile", "compile", "transmission-line declaration is malformed")
        length_value = binding_value(component, "length", binding_for(component, "length", realization), values; context_kind = context_kind)
        isfinite(length_value) && length_value > 0.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "transmission-line length must be finite and strictly positive")
        dx = length_value / sections
        rlgc = realization["rlgc"]
        R = rlgc_matrix(rlgc["resistance_per_length"], "R") .* dx
        L = rlgc_matrix(rlgc["inductance_per_length"], "L") .* dx
        G = rlgc_matrix(rlgc["conductance_per_length"], "G") .* dx
        C = rlgc_matrix(rlgc["capacitance_per_length"], "C") .* dx
        size(R, 1) == length(conductors) && size(L) == size(R) && size(G) == size(R) && size(C) == size(R) ||
            fail("execution", "compiler_invariant", "compile", "compile", "RLGC matrix dimension disagrees with line conductors")
        all(isfinite, R) && all(isfinite, L) && all(isfinite, G) && all(isfinite, C) ||
            fail("execution", "compiler_invariant", "compile", "compile", "RLGC matrix is non-finite")
        try
            cholesky(Symmetric(L); check = true); cholesky(Symmetric(C); check = true)
            minimum(eigvals(Symmetric(R))) >= 0.0 && minimum(eigvals(Symmetric(G))) >= 0.0 || error("non-PSD")
        catch
            fail("execution", "compiler_invariant", "compile", "compile", "RLGC physical matrix validation failed")
        end
        # Preflight asks for a compiled-schematic audit, not a second line
        # expansion.  Emit it from this exact recursive lowering while its
        # endpoint map and deterministic compiler node IDs are in scope.
        if emit_audit
            stations = Dict{String,Any}[]
            for station in 0:sections, conductor in conductors
                attachment = station == 0 ? "head" : station == sections ? "tail" : "interior"
                total_factor = station == 0 || station == sections ? 0.5 : 1.0
                push!(stations, Dict{String,Any}(
                    "station" => station, "conductor" => conductor,
                    "compiled_node_id" => line_station_node(component, station, conductor, endpoint_to_node),
                    "attachment" => attachment,
                    # Interior stations receive the two explicitly recorded
                    # pi half-shunts adjacent to their left/right sections.
                    "left_half_shunt" => station == 0 ? nothing : Dict("section" => station, "end" => "right"),
                    "right_half_shunt" => station == sections ? nothing : Dict("section" => station + 1, "end" => "left"),
                    "compiled_capacitance_total" => quantity_matrix(C .* total_factor, "farad", "capacitance"),
                    "compiled_conductance_total" => quantity_matrix(G .* total_factor, "siemens", "conductance"),
                ))
            end
            push!(rows, Dict{String,Any}(
                "kind" => "transmission_line_audit", "component_path" => path,
                "conductors" => conductors, "reference_conductor" => String(rlgc["reference_conductor"]),
                "n_sections" => sections,
                "length" => quantity(length_value, "meter", "length"),
                "dx" => quantity(dx, "meter", "length"),
                "orientation" => String(rlgc["orientation"]), "rlgc_source" => plain(rlgc["source"]),
                "stations" => stations,
            ))
        end
        for section in 1:sections
            left = line_station_incidence(component, section - 1, conductors, endpoint_to_node, node_index)
            right = line_station_incidence(component, section, conductors, endpoint_to_node, node_index)
            Bseries = left - right
            push!(series_rl, SeriesRLBlock(component_key(component) * "\u001e" * "section-" * string(section), Bseries, R, L))
            for station in (section - 1, section)
                Bshunt = line_station_incidence(component, station, conductors, endpoint_to_node, node_index)
                push!(capacitance_blocks, (Bshunt, C ./ 2.0)); push!(conductance_blocks, (Bshunt, G ./ 2.0))
            end
            for (row, conductor_a) in enumerate(conductors), (column, conductor_b) in enumerate(conductors)
                for (label, matrix, unit, dimensionality) in (("series_resistance", R, "ohm", "resistance"), ("series_inductance", L, "henry", "inductance"))
                    value = matrix[row, column]
                    push!(rows, Dict{String,Any}("component_path" => path, "kind" => label, "section" => section,
                        "row_conductor" => conductor_a, "column_conductor" => conductor_b,
                        "value" => quantity(value, unit, dimensionality), "omitted_as_zero" => value == 0.0))
                end
                # Each pi section stamps a distinct half shunt at its left
                # and right station.  Preserve both in branch evidence: one
                # aggregate row would no longer prove the C/G lowering.
                for (station, end_label) in ((section - 1, "left"), (section, "right"))
                    Bstation = line_station_incidence(component, station, conductors, endpoint_to_node, node_index)
                    for (label, matrix, unit, dimensionality) in (("shunt_conductance_half", G / 2.0, "siemens", "conductance"), ("shunt_capacitance_half", C / 2.0, "farad", "capacitance"))
                        value = matrix[row, column]
                        push!(rows, Dict{String,Any}("component_path" => path, "kind" => label, "section" => section,
                            "station" => station, "end" => end_label, "row_conductor" => conductor_a, "column_conductor" => conductor_b,
                            # A matrix-valued shunt has a declared pair of
                            # physical station endpoints.  Keep each vector
                            # so transform-cut provenance can classify it
                            # after any preceding coordinate congruence.
                            "row_incidence_f64" => f64_hex.(Bstation[:, row]),
                            "column_incidence_f64" => f64_hex.(Bstation[:, column]),
                            # The diagonal Maxwell entry is a shunt from its
                            # conductor to the reference; an off-diagonal
                            # entry names the two physical conductors.  These
                            # endpoint selectors distinguish a true direct
                            # mutual from a ground branch after transforms.
                            "physical_positive_incidence_f64" => f64_hex.(Bstation[:, row]),
                            "physical_negative_incidence_f64" => f64_hex.(row == column ? zeros(Float64, length(node_index)) : Bstation[:, column]),
                            "value" => quantity(value, unit, dimensionality), "omitted_as_zero" => value == 0.0))
                    end
                end
            end
        end
        return nothing
    end
    b = component_incidence(component, endpoint_to_node, node_index)
    endpoint_positive, endpoint_negative = component_endpoint_incidences(component, endpoint_to_node, node_index)
    function record!(row_kind, value, unit, dimensionality, incidence = b; omitted_as_zero::Bool = false)
        push!(rows, Dict{String,Any}("component_path" => path, "kind" => row_kind,
            "terminal_1_to_terminal_2" => pins, "incidence_f64" => f64_hex.(incidence),
            "physical_positive_incidence_f64" => f64_hex.(endpoint_positive),
            "physical_negative_incidence_f64" => f64_hex.(endpoint_negative),
            "value" => quantity(value, unit, dimensionality), "omitted_as_zero" => omitted_as_zero))
    end
    if kind == "capacitor" || kind == "resistor"
        parameter = kind == "capacitor" ? "capacitance" : "resistance"
        value = binding_value(component, parameter, binding_for(component, parameter, realization), values; context_kind = context_kind)
        isfinite(value) && value > 0.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "primitive R/C value must be finite and strictly positive")
        kind == "capacitor" ? push!(capacitors, (b, value)) : push!(resistors, (b, value))
        record!(kind, value, kind == "capacitor" ? "farad" : "ohm", kind == "capacitor" ? "capacitance" : "resistance")
    elseif kind == "josephson_junction"
        lj = binding_value(component, "josephson_inductance", binding_for(component, "josephson_inductance", realization), values; context_kind = context_kind)
        cj = binding_value(component, "junction_capacitance", binding_for(component, "junction_capacitance", realization), values; context_kind = context_kind)
        isfinite(lj) && lj > 0.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "L_J0 must be finite and strictly positive")
        isfinite(cj) && cj >= 0.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "Cj must be finite and nonnegative")
        push!(inductors, ExpandedInductor(component_key(component) * "\u001e" * "self", b, lj)); record!("josephson_inductance", lj, "henry", "inductance")
        if cj == 0.0
            record!("junction_capacitance", cj, "farad", "capacitance"; omitted_as_zero = true)
        else
            push!(capacitors, (b, cj)); record!("junction_capacitance", cj, "farad", "capacitance")
        end
    elseif kind == "inductor"
        # The canonical branch list is the orientation authority; no drawing-derived sign exists.
        for branch in component["inductive_branches"]
            value = binding_value(component, "inductance", branch["inductance"], values; context_kind = context_kind)
            isfinite(value) && value > 0.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "inductance must be finite and strictly positive")
            branch_incidence = recursive_incidence(branch["positive_endpoint"], branch["negative_endpoint"], endpoint_to_node, node_index)
            push!(inductors, ExpandedInductor(branch_key(Dict("component_path" => component["component_path"], "branch_id" => branch["id"])), branch_incidence, value))
            record!("inductor", value, "henry", "inductance", branch_incidence)
            # HB lowering needs the same sealed branch identity as mutual
            # coupling; a drawing/name-derived association is not authority.
            rows[end]["branch_id"] = String(branch["id"])
        end
    else
        fail("capability", "scaffold_unavailable", "compile", "compile", "sealed component realization is outside the recursive Direct compiler")
    end
end

function recursive_expand!(component, endpoint_to_node, node_index, values, capacitors, resistors, inductors, capacitance_blocks, conductance_blocks, series_rl, rows, couplings, context_kind::String;
        top_level::Bool, emit_audit::Bool = false)
    realization = component["realization"]
    String(realization["kind"]) != "composite" && return recursive_leaf!(component, endpoint_to_node, node_index, values, capacitors, resistors, inductors, capacitance_blocks, conductance_blocks, series_rl, rows, context_kind; emit_audit = emit_audit)
    path = component_path(component); local_nodes = Dict{String,String}()
    for node in realization["private_nodes"]
        local_nodes[String(node["id"])] = expanded_internal_node_id(component, node)
    end
    for mapping in realization["public_pin_map"]
        private_id = String(mapping["private_node_id"]); haskey(local_nodes, private_id) || fail("execution", "compiler_invariant", "compile", "compile", "Composite public pin map targets no private node")
        public_endpoint = endpoint_at(path, mapping["public_id"])
        haskey(endpoint_to_node, public_endpoint) || fail("execution", "compiler_invariant", "compile", "compile", "Composite public pin is unbound")
        local_nodes[private_id] = endpoint_to_node[public_endpoint]
    end
    if top_level
        for mapping in realization["public_coordinate_map"]
            private_id = String(mapping["private_node_id"]); haskey(local_nodes, private_id) || fail("execution", "compiler_invariant", "compile", "compile", "Composite public coordinate map targets no private node")
            local_nodes[private_id] = String(mapping["public_id"])
        end
    end
    child_endpoints = Dict{String,String}()
    for node in realization["private_nodes"]
        for endpoint in node["endpoints"]
            key = endpoint_key(endpoint); haskey(child_endpoints, key) && fail("execution", "compiler_invariant", "compile", "compile", "Composite child endpoint belongs to multiple private nodes")
            child_endpoints[key] = local_nodes[String(node["id"])]
        end
    end
    for endpoint in realization["grounded_endpoints"]
        key = endpoint_key(endpoint); haskey(child_endpoints, key) && fail("execution", "compiler_invariant", "compile", "compile", "Composite grounded endpoint also belongs to a private node")
        child_endpoints[key] = "ground"
    end
    for child in realization["children"]
        recursive_expand!(child, child_endpoints, node_index, values, capacitors, resistors, inductors, capacitance_blocks, conductance_blocks, series_rl, rows, couplings, context_kind; top_level = false, emit_audit = emit_audit)
    end
    append!(couplings, realization["couplings"])
    return nothing
end

function compile_recursive(plan_value, request_values::Dict{String,Float64}; context_kind::String = "compile",
        authorized::Set{String} = Set{String}(), extrapolation_evidence::Union{Nothing,Vector{Any}} = nothing,
        authorization_source::String = "none", emit_audit::Bool = false)::CompiledPrimitive
    plan = plain(plan_value)
    get(plan, "schema", nothing) == "scnsim.plan" || fail("execution", "compiler_invariant", "compile", "compile", "plan schema discriminator is invalid")
    values = recursive_parameter_values(plan, request_values; context_kind = context_kind, authorized = authorized,
        extrapolation_evidence = extrapolation_evidence, authorization_source = authorization_source)
    for component in plan["components"]; validate_composite_maps!(component); end
    nodes = String[item["node_id"] for item in plan["nodes"]]
    for component in plan["components"]; recursive_nodes!(nodes, component; top_level = true); end
    nodes = unique(sort!(nodes)); isempty(nodes) && fail("execution", "compiler_invariant", "compile", "compile", "sealed Plan has no non-reference node")
    node_index = Dict(node => index for (index, node) in enumerate(nodes)); endpoint_to_node = endpoint_nodes(plan)
    capacitors = Tuple{Vector{Float64},Float64}[]; resistors = Tuple{Vector{Float64},Float64}[]; inductors = ExpandedInductor[]
    capacitance_blocks = Tuple{Matrix{Float64},Matrix{Float64}}[]; conductance_blocks = Tuple{Matrix{Float64},Matrix{Float64}}[]; series_rl = SeriesRLBlock[]
    rows = Dict{String,Any}[]; couplings = Any[]
    for component in plan["components"]
        recursive_expand!(component, endpoint_to_node, node_index, values, capacitors, resistors, inductors, capacitance_blocks, conductance_blocks, series_rl, rows, couplings, context_kind; top_level = true, emit_audit = emit_audit)
    end
    append!(couplings, plan["couplings"])
    n = length(nodes); C = zeros(Float64, n, n); G = zeros(Float64, n, n)
    for (b, value) in capacitors; C .+= value .* (b * transpose(b)); end
    for (b, value) in resistors; G .+= (1.0 / value) .* (b * transpose(b)); end
    for (Bblock, values_block) in capacitance_blocks; C .+= Bblock * values_block * transpose(Bblock); end
    for (Bblock, values_block) in conductance_blocks; G .+= Bblock * values_block * transpose(Bblock); end
    locations = Dict(item.id => index for (index, item) in enumerate(inductors))
    edges = Tuple{Int,Int,Float64}[]; resolved_pairs = Set{Tuple{Int,Int}}()
    for coupling in couplings
        left = branch_key(lower_branch_reference(plan, coupling["branch_a"])); right = branch_key(lower_branch_reference(plan, coupling["branch_b"]))
        haskey(locations, left) && haskey(locations, right) || fail("execution", "compiler_invariant", "compile", "compile", "mutual coupling references unknown expanded branch")
        i = locations[left]; j = locations[right]; i != j || fail("execution", "compiler_invariant", "compile", "compile", "mutual coupling cannot self-couple a branch")
        pair = minmax(i, j); pair in resolved_pairs &&
            fail("execution", "compiler_invariant", "compile", "compile", "mutual couplings duplicate one resolved physical branch pair")
        push!(resolved_pairs, pair)
        k = quantity_value(coupling["coupling_coefficient"]); isfinite(k) && abs(k) < 1.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "mutual coupling coefficient must satisfy abs(k) < 1")
        mutual = k * sqrt(inductors[i].value * inductors[j].value)
        push!(edges, (i, j, mutual))
        push!(rows, Dict{String,Any}(
            "kind" => "mutual_inductance",
            "coupling_id" => String(coupling["id"]),
            "branch_a" => coupling["branch_a"],
            "branch_b" => coupling["branch_b"],
            "coupling_coefficient" => quantity(k, "dimensionless", "dimensionless"),
            "derived_mutual_inductance" => quantity(mutual, "henry", "inductance"),
            "omitted_as_zero" => mutual == 0.0,
        ))
    end
    K = zeros(Float64, n, n)
    neighbors = [Int[] for _ in inductors]
    for (i, j, _) in edges
        push!(neighbors[i], j); push!(neighbors[j], i)
    end
    visited = falses(length(inductors))
    for start in eachindex(inductors)
        visited[start] && continue
        group = Int[]; pending = [start]; visited[start] = true
        while !isempty(pending)
            index = pop!(pending); push!(group, index)
            for neighbor in neighbors[index]
                visited[neighbor] && continue
                visited[neighbor] = true; push!(pending, neighbor)
            end
        end
        if length(group) == 1
            branch = inductors[only(group)]
            K .+= (1.0 / branch.value) .* (branch.incidence * transpose(branch.incidence))
            continue
        end
        local_index = Dict(member => index for (index, member) in enumerate(group))
        L = zeros(Float64, length(group), length(group))
        for (index, member) in enumerate(group); L[index, index] = inductors[member].value; end
        for (i, j, mutual) in edges
            haskey(local_index, i) && haskey(local_index, j) || continue
            left = local_index[i]; right = local_index[j]
            L[left, right] = mutual; L[right, left] = mutual
        end
        factor = try
            cholesky(Symmetric(L); check = true)
        catch
            fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "complete reciprocal inductance matrix is not positive definite")
        end
        B = hcat((inductors[index].incidence for index in group)...)
        reciprocal = factor \ transpose(B)
        residual = backward_residual(L, reciprocal, transpose(B))
        isfinite(residual) && residual <= tau(length(group)) ||
            fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "reciprocal inductance solve exceeded normalized backward-residual contract")
        K .+= B * reciprocal
    end
    ports = plan["ports"]
    port_ids = String[]
    B = zeros(Float64, n, length(ports))
    R = zeros(Float64, length(ports), length(ports))
    M = ones(Float64, length(ports))
    for (column, port) in enumerate(ports)
        port_id = String(port["port_id"])
        port_id in port_ids && fail("execution", "compiler_invariant", "compile", "compile", "sealed Plan has duplicate Port IDs")
        role = String(port["role"])
        role in ("terminated", "nonloading_probe") ||
            fail("validation", "port_realizability", "compile", "compile", "Port role is not realizable by the Direct compiler")
        node_id = String(port["node_id"])
        haskey(node_index, node_id) || fail("execution", "compiler_invariant", "compile", "compile", "Port node is absent from compiled basis")
        z0 = quantity_value(port["reference_impedance"])
        isfinite(z0) && z0 > 0.0 || fail("execution", "compiler_invariant", "compile", "compile", "Port reference impedance must be finite and positive")
        push!(port_ids, port_id)
        B[node_index[node_id], column] = 1.0
        R[column, column] = z0
    end
    return CompiledPrimitive(nodes, C, K, G, Any[series_rl...], rows, port_ids, B, R, M)
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

"""Compiler-owned realization of the terminal View boundary.

The lazy Python lineage deliberately has no numerical matrices.  This helper is
the one place that turns it into the selected B/R/M boundary used for both the
durable lineage evidence and the Direct calculation.  In particular, retaining
a node is not silently a Port: the terminal map must be square and full rank in
the original logical-Port space before a wave response is available.
"""
struct RealizedView
    compiled::CompiledPrimitive
    coordinates::Vector{String}
    terminal::Vector{String}
    coordinate_port_map::Matrix{Float64}
    selected_indices::Vector{Int}
    selected_map::Matrix{Float64}
    port_realizable::Bool
end

function lineage_matrix_evidence(label::String, matrix::AbstractMatrix{Float64}, applicability::String)
    values = String[]
    for row in axes(matrix, 1), column in axes(matrix, 2)
        push!(values, f64_hex(matrix[row, column]))
    end
    payload = Dict{String,Any}(
        "schema" => "scnsim.lineage_matrix", "schema_version" => 1,
        "label" => label, "applicability" => applicability,
        "shape" => [size(matrix, 1), size(matrix, 2)], "row_major_f64" => values,
    )
    return Dict("rows" => size(matrix, 1), "columns" => size(matrix, 2),
        "sha256" => sha256_hex(canonical_bytes(payload)))
end

"""Return a structural capacitance-branch reference for one expanded row."""
function cap_branch_ref(row)
    kind = String(get(row, "kind", ""))
    (occursin("capacitance", kind) || kind == "capacitor") || return nothing
    path = get(row, "component_path", nothing)
    path isa AbstractVector || return nothing
    suffix = if haskey(row, "section")
        station = haskey(row, "station") ? string(".station-", row["station"], ".", get(row, "end", "")) : ""
        string(kind, ".s", row["section"], station, ".", get(row, "row_conductor", ""), ".", get(row, "column_conductor", ""))
    else
        kind == "capacitor" ? "capacitance" : kind
    end
    return Dict{String,Any}("component_path" => String.(path), "branch_id" => suffix)
end

cap_ref_key(ref) = join(String.(ref["component_path"]), "\u001f") * "\u001e" * String(ref["branch_id"])

"""Current-basis physical support for one capacitance branch diagnostic.

Primitive/JJ rows carry a single oriented branch incidence.  A matrix-valued
RLGC pi shunt carries its row/column station incidences: their union is the
declared physical endpoint set for that matrix entry.  Both representations
are transformed alongside the compiled C congruence, so this helper never
classifies source-basis rows against generated coordinates.
"""
function cap_row_support(row, n::Int)
    vectors = Vector{Vector{Float64}}()
    if haskey(row, "physical_positive_incidence_f64") && haskey(row, "physical_negative_incidence_f64")
        push!(vectors, Float64[f64_from_hex(value) for value in row["physical_positive_incidence_f64"]])
        push!(vectors, Float64[f64_from_hex(value) for value in row["physical_negative_incidence_f64"]])
    elseif haskey(row, "incidence_f64")
        push!(vectors, Float64[f64_from_hex(value) for value in row["incidence_f64"]])
    elseif haskey(row, "row_incidence_f64") && haskey(row, "column_incidence_f64")
        push!(vectors, Float64[f64_from_hex(value) for value in row["row_incidence_f64"]])
        push!(vectors, Float64[f64_from_hex(value) for value in row["column_incidence_f64"]])
    else
        fail("execution", "compiler_invariant", "transform_pair", "compile", "capacitance branch has no physical incidence/station endpoints")
    end
    all(length(vector) == n && all(isfinite, vector) for vector in vectors) ||
        fail("execution", "compiler_invariant", "transform_pair", "compile", "capacitance branch incidence has the wrong current basis")
    return Set(index for vector in vectors for index in eachindex(vector) if vector[index] != 0.0)
end

function cap_row_is_direct_mutual(row, left::Int, right::Int, n::Int)
    # Endpoint selectors retain the physical distinction between a ground
    # capacitor and a two-terminal mutual capacitor after a prior transform.
    # A transformed ground branch can have support on both generated channels,
    # but it never becomes a direct pair mutual merely because of that basis
    # representation.
    if haskey(row, "physical_positive_incidence_f64") && haskey(row, "physical_negative_incidence_f64")
        positive = Float64[f64_from_hex(value) for value in row["physical_positive_incidence_f64"]]
        negative = Float64[f64_from_hex(value) for value in row["physical_negative_incidence_f64"]]
        length(positive) == n && length(negative) == n ||
            fail("execution", "compiler_invariant", "transform_pair", "compile", "capacitance endpoint incidence has the wrong current basis")
        support(vector) = Set(index for index in eachindex(vector) if vector[index] != 0.0)
        return (support(positive) == Set([left]) && support(negative) == Set([right])) ||
            (support(positive) == Set([right]) && support(negative) == Set([left]))
    end
    support = cap_row_support(row, n)
    return support == Set([left, right])
end

"""Partition the current transform's actual full external capacitance cut.

Only a branch touching exactly the selected pair and no exterior coordinate is
a direct mutual branch.  A ground branch touches one selected coordinate and
is therefore part of that coordinate's external cut; rows not touching either
member are intentionally omitted.  This is evidence-only: alpha/beta remain
the authoritative C-derived numerical weights above.
"""
function cap_branch_partition(compiled::CompiledPrimitive, left::Int, right::Int)
    included = Dict{String,Any}[]; excluded = Dict{String,Any}[]
    seen_included, seen_excluded = Set{String}(), Set{String}()
    pair = Set([left, right]); n = length(compiled.nodes)
    for row in compiled.branch_rows
        ref = cap_branch_ref(row); ref === nothing && continue
        get(row, "omitted_as_zero", false) === true && continue
        support = cap_row_support(row, n)
        isempty(support) && continue # an explicitly omitted zero branch
        key = cap_ref_key(ref)
        if cap_row_is_direct_mutual(row, left, right, n)
            key in seen_excluded || (push!(seen_excluded, key); push!(excluded, ref))
        elseif !isempty(intersect(support, pair))
            key in seen_included || (push!(seen_included, key); push!(included, ref))
        end
    end
    sort!(included; by = ref -> (Tuple(String.(ref["component_path"])), String(ref["branch_id"])))
    sort!(excluded; by = ref -> (Tuple(String.(ref["component_path"])), String(ref["branch_id"])))
    return included, excluded
end

function selected_coordinate_indices(compiled::CompiledPrimitive, coordinates::Vector{String})
    indices = Int[]
    for coordinate in coordinates
        index = findfirst(==(coordinate), compiled.nodes)
        if index === nothing
            port = findfirst(==(coordinate), compiled.port_ids)
            port === nothing && fail("validation", "port_realizability", "selected_network", "direct_response", "terminal coordinate is absent from compiled basis")
            entries = findall(!iszero, view(compiled.B, :, port::Int))
            length(entries) == 1 || fail("validation", "port_realizability", "selected_network", "direct_response", "logical Port does not select one physical coordinate")
            push!(indices, only(entries))
        else
            push!(indices, index::Int)
        end
    end
    length(indices) == length(unique(indices)) ||
        fail("validation", "port_realizability", "selected_network", "direct_response", "terminal coordinates repeat")
    return indices
end

function terminal_view(compiled::CompiledPrimitive, lineage)::RealizedView
    item = plain(lineage)
    # The compiler sorts the complete physical basis.  Build a separate map
    # from that basis to logical Port coordinates; no top-level/public-node
    # ordering is allowed to stand in for a backend node order.
    base = apply_lineage_load_mask(compiled, item)
    p = length(base.port_ids)
    coordinate_map = zeros(Float64, length(base.nodes), p)
    for (column, port) in enumerate(base.port_ids)
        # A Port ID is not generally a node ID.  Its selector column is the
        # authoritative physical binding after recursive lowering.
        entries = findall(!iszero, view(base.B, :, column))
        length(entries) == 1 || fail("validation", "port_realizability", "selected_network", "direct_response", "logical Port selector is not one physical coordinate")
        coordinate_map[only(entries), column] = 1.0
    end
    # Reconstruct the coordinate-to-Port map in the same canonical tail
    # ordering as the physical congruence below.
    original_names = String.(item["original"]["coordinate_order"])
    # Internal compiler-only nodes carry no public coordinate map.  Public
    # original coordinates map to a Port only when their ID is that Port's
    # promoted Plan node; infer this from the untransformed selector IDs.
    # The physical transform helper has transformed B, so solve the row map
    # against it: for raw public nodes it is the unique unit selector row.
    maps = Dict{String,Vector{Float64}}()
    for name in original_names
        index = findfirst(==(name), base.nodes)
        maps[name] = index === nothing ? zeros(Float64, p) : vec(copy(base.B[index, :]))
    end
    # Raw terminal channels are logical Port IDs, while original coordinate
    # names are physical Plan-node IDs.  Keep the namespaces separate so a
    # promoted node sharing its Port ID can be transformed without deleting
    # the logical boundary selector.
    logical_port_maps = Dict{String,Vector{Float64}}()
    for (column, port) in enumerate(base.port_ids)
        logical = zeros(Float64, p); logical[column] = 1.0
        logical_port_maps[port] = logical
    end
    current_names = copy(original_names)
    transforms = get(item, "transforms", Any[])
    working = base
    realized_transforms = Dict{String,Any}[]
    for transform in transforms
        inputs = String.(transform["input_coordinates"])
        length(inputs) == 2 && all(name -> haskey(maps, name), inputs) ||
            fail("validation", "port_realizability", "transform_pair", "compile", "transform input is not an original/public coordinate")
        left, right = inputs
        i = findfirst(==(left), current_names); j = findfirst(==(right), current_names)
        i !== nothing && j !== nothing || fail("validation", "port_realizability", "transform_pair", "compile", "transform input ordering is malformed")
        # Weights are compiler-derived when realizing a lazy record; a
        # previously realized record already binds exact binary64 values.
        # Recompute candidate-dependent full-cut weights from the currently
        # bound C graph.  Persisted weights bind request identity at the
        # baseline, but are never a stale numerical shortcut for CMA.
        li = findfirst(==(left), working.nodes); ri = findfirst(==(right), working.nodes)
        li !== nothing && ri !== nothing || fail("validation", "port_realizability", "transform_pair", "compile", "transform input is absent from compiled basis")
        cl = working.C[li, li] + working.C[li, ri]; cr = working.C[ri, ri] + working.C[ri, li]
        total = cl + cr
        isfinite(total) && total > 0.0 || fail("validation", "port_realizability", "transform_pair", "compile", "floating-pair external capacitance cut is invalid")
        α, β = cl / total, cr / total
        common = haskey(transform, "common_id") ? String(transform["common_id"]) : String(transform["output_coordinates"][1])
        differential = haskey(transform, "differential_id") ? String(transform["differential_id"]) : String(transform["output_coordinates"][2])
        maps[differential] = maps[left] .- maps[right]
        maps[common] = α .* maps[left] .+ β .* maps[right]
        delete!(maps, left); delete!(maps, right)
        current_names = vcat([name for name in current_names if name != left && name != right], [common, differential])
        # Apply exactly this transform to the physical graph.  An existing
        # realized record has weights, while the lazy record gets them bound
        # here; either way apply_lineage_transforms uses this declaration.
        step = Dict{String,Any}("input_coordinates" => [left, right], "output_coordinates" => [common, differential],
            "weights_f64" => [f64_hex(α), f64_hex(β)])
        working = apply_lineage_transforms(working, Dict("transforms" => Any[step]))
        push!(realized_transforms, step)
    end
    terminal = if get(item, "retain", nothing) !== nothing
        String.(item["retain"]["retained_coordinates"])
    elseif isempty(transforms)
        String.(item["original"]["port_order"])
    else
        # A transform is intermediate unless retain() chooses its generated
        # coordinates; raw Direct remains in declared logical-Port order.
        String.(item["original"]["port_order"])
    end
    terminal_maps = get(item, "retain", nothing) === nothing ? logical_port_maps : maps
    all(haskey(terminal_maps, name) for name in terminal) ||
        fail("validation", "port_realizability", "selected_network", "direct_response", "terminal View contains a non-public coordinate")
    selected_map = isempty(terminal) ? zeros(Float64, 0, p) : reduce(vcat, (reshape(terminal_maps[name], 1, :) for name in terminal))
    # Without retain(), a transformed Direct View returns to the declared
    # logical-Port boundary.  Those Port rows live in selected_map and need
    # not select individual transformed nodes.  Every coordinate-selected
    # View still proves node presence and uniqueness fail-closed.
    indices = !isempty(transforms) && get(item, "retain", nothing) === nothing ?
        Int[] : selected_coordinate_indices(working, terminal)
    # A retained subset of uniquely Port-bound coordinates is a generalized
    # wave boundary.  Omitted logical Ports remain in the matched-load
    # projector, so selected rows—not the full original Port count—govern
    # realizability.
    realizable = !isempty(terminal) && length(terminal) <= p && rank(selected_map) == length(terminal)
    return RealizedView(working, current_names, terminal, coordinate_map, indices, selected_map, realizable)
end

"""The single generalized selected-Port boundary used by evidence and solves."""
function selected_boundary(view::RealizedView)
    view.port_realizable || fail("validation", "port_realizability", "selected_network", "direct_response", "selected View is not Port-realizable")
    compiled = view.compiled; A = view.selected_map; p = length(compiled.port_ids); q = size(A, 1)
    Dp = port_reference_root(compiled)
    Dp_inv = checked_solve(complex.(Dp), Matrix{ComplexF64}(I, p, p), "direct_response_formation", "reference_matrix", p)
    Rk = A * compiled.R * transpose(A)
    Dk = principal_spd_root(Rk)
    Qk = checked_solve(complex.(Dk), complex.(A * Dp), "direct_response_formation", "reference_matrix", q)
    Pk = real.(transpose(Qk) * Qk)
    Po = Matrix{Float64}(I, p, p) - Pk
    Go = Dp_inv * complex.(Po) * complex.(Diagonal(compiled.M)) * complex.(Po) * Dp_inv
    # B_k is the source boundary transformed from the original ordered Port
    # realization.  It is not a coincidental retained-node selector.
    Bk = compiled.B * transpose(A)
    return (A = A, Bk = Bk, Rk = Rk, Dp = Dp, Dp_inv = Dp_inv, Dk = Dk,
        Qk = Qk, Pk = Pk, Po = Po, Go = Go)
end

function view_boundary_evidence(view::RealizedView)
    p = length(view.compiled.port_ids); q = length(view.terminal)
    if !view.port_realizable
        empty = zeros(Float64, 0, 0)
        matrices = Dict(label => lineage_matrix_evidence(label, empty, "not_port_realizable") for label in
            ("a", "b", "r", "d", "q", "selected_projector", "omitted_projector", "omitted_matched_loads"))
        return matrices,
            sha256_hex(canonical_bytes(Dict("schema" => "scnsim.source_boundary", "schema_version" => 1, "applicability" => "not_port_realizable"))),
            sha256_hex(canonical_bytes(Dict("schema" => "scnsim.deembedding", "schema_version" => 1, "applicability" => "not_port_realizable")))
    end
    boundary = selected_boundary(view)
    A, B, R, D = boundary.A, boundary.Bk, boundary.Rk, boundary.Dk
    selected, omitted = boundary.Pk, boundary.Po
    omitted_load = real.(boundary.Go)
    matrices = Dict(
        "a" => lineage_matrix_evidence("a", A, "port_realizable"),
        "b" => lineage_matrix_evidence("b", B, "port_realizable"),
        "r" => lineage_matrix_evidence("r", R, "port_realizable"),
        "d" => lineage_matrix_evidence("d", D, "port_realizable"),
        "q" => lineage_matrix_evidence("q", real.(boundary.Qk), "port_realizable"),
        "selected_projector" => lineage_matrix_evidence("selected_projector", selected, "port_realizable"),
        "omitted_projector" => lineage_matrix_evidence("omitted_projector", omitted, "port_realizable"),
        "omitted_matched_loads" => lineage_matrix_evidence("omitted_matched_loads", omitted_load, "port_realizable"),
    )
    source = sha256_hex(canonical_bytes(Dict("schema" => "scnsim.source_boundary", "schema_version" => 1,
        "applicability" => "port_realizable", "b" => matrices["b"], "r" => matrices["r"])))
    deembed = sha256_hex(canonical_bytes(Dict("schema" => "scnsim.deembedding", "schema_version" => 1,
        "applicability" => "port_realizable", "d" => matrices["d"], "q" => matrices["q"])))
    return matrices, source, deembed
end

"""Close a lazy public View declaration into the request-hashed evidence form."""
function realized_ref_lineage(compiled::CompiledPrimitive, lazy)
    item = plain(lazy)
    base = apply_lineage_load_mask(compiled, item)
    working = compiled
    realized_ptc = nothing
    ptc = get(item, "ptc", nothing)
    if ptc !== nothing
        selected = String.(ptc["selected_ports"])
        loads = Dict{String,Any}[]
        for port in selected
            index = findfirst(==(port), compiled.port_ids)
            index === nothing && fail("validation", "port_realizability", "ptc", "compile", "PTC references an unknown Port")
            push!(loads, Dict("port_id" => port,
                "reference_impedance" => quantity(compiled.R[index, index], "ohm", "resistance"), "before" => "raw", "after" => "compensated"))
        end
        mask_payload = Dict("schema" => "scnsim.ptc_load_mask", "schema_version" => 1,
            "port_order" => compiled.port_ids, "load_mask_f64" => f64_hex.(base.M))
        realized_ptc = Dict("type" => "ptc", "selected_ports" => selected,
            "load_mask_sha256" => sha256_hex(canonical_bytes(mask_payload)), "loads" => loads,
            "reconstruction_residual_f64" => f64_hex(0.0),
            "output_coordinate_order" => item["original"]["coordinate_order"],
            "evidence_sha256" => sha256_hex(canonical_bytes(Dict("schema" => "scnsim.ptc_evidence", "schema_version" => 1, "loads" => loads, "mask" => mask_payload))))
        working = base
    end
    transforms_out = Dict{String,Any}[]
    current_names = String.(item["original"]["coordinate_order"])
    for transform in get(item, "transforms", Any[])
        left, right = String.(transform["input_coordinates"][1]), String(transform["input_coordinates"][2])
        li = findfirst(==(left), working.nodes); ri = findfirst(==(right), working.nodes)
        li !== nothing && ri !== nothing || fail("validation", "port_realizability", "transform_pair", "compile", "transform input is absent from compiled basis")
        cl = working.C[li, li] + working.C[li, ri]; cr = working.C[ri, ri] + working.C[ri, li]; total = cl + cr
        isfinite(cl) && isfinite(cr) && cl >= 0.0 && cr >= 0.0 && total > 0.0 ||
            fail("validation", "port_realizability", "transform_pair", "compile", "floating-pair external capacitance cut is invalid")
        alpha, beta = cl / total, cr / total
        common = haskey(transform, "common_id") ? String(transform["common_id"]) : String(transform["output_coordinates"][1])
        differential = haskey(transform, "differential_id") ? String(transform["differential_id"]) : String(transform["output_coordinates"][2])
        pair = [left, right]
        all(name in current_names for name in pair) || fail("validation", "port_realizability", "transform_pair", "compile", "transform input is not current")
        output = vcat([name for name in current_names if name != left && name != right], [common, differential])
        # Transform B/R evidence is only applicable for a full-rank selected
        # channel map.  A transform can be a quantity-only coordinate map.
        temporary = terminal_view(compiled, Dict("original" => item["original"], "ptc" => realized_ptc,
            "transforms" => vcat(transforms_out, [Dict("input_coordinates" => pair, "output_coordinates" => [common, differential], "weights_f64" => [f64_hex(alpha), f64_hex(beta)])]),
            "retain" => Dict("retained_coordinates" => output), "terminal_coordinates" => output))
        matrices, _, _ = view_boundary_evidence(temporary)
        reconstruction = if temporary.port_realizable
            boundary = selected_boundary(temporary)
            backward_residual(boundary.Dk, boundary.Dk, boundary.Rk)
        else
            0.0
        end
        refs, excluded_refs = cap_branch_partition(working, li::Int, ri::Int)
        isempty(refs) && fail("validation", "port_realizability", "transform_pair", "compile", "transform external cut has no capacitance branch provenance")
        evidence_payload = Dict("schema" => "scnsim.transform_pair_evidence", "schema_version" => 1,
            "input_coordinates" => pair, "weights_f64" => [f64_hex(alpha), f64_hex(beta)], "output_coordinate_order" => output,
            "included_external_cut_branches" => refs, "excluded_direct_mutual_branches" => excluded_refs,
            "reference_matrix" => matrices["r"], "principal_root" => matrices["d"],
            "reconstruction_residual_f64" => f64_hex(reconstruction))
        push!(transforms_out, Dict("type" => "transform_pair", "input_coordinates" => pair,
            "weights_f64" => [f64_hex(alpha), f64_hex(beta)], "differential_id" => differential, "common_id" => common,
            "included_external_cut_branches" => refs, "excluded_direct_mutual_branches" => excluded_refs,
            "reference_matrix" => matrices["r"], "principal_root" => matrices["d"],
            "reconstruction_residual_f64" => f64_hex(reconstruction), "output_coordinate_order" => output,
            "evidence_sha256" => sha256_hex(canonical_bytes(evidence_payload))))
        working = apply_lineage_transforms(working, Dict("transforms" => Any[Dict("input_coordinates" => pair, "output_coordinates" => [common, differential])]))
        current_names = output
    end
    lazy_retain = get(item, "retain", nothing)
    terminal = lazy_retain === nothing ? String.(item["original"]["port_order"]) : String.(lazy_retain["retained_coordinates"])
    realization_input = Dict("original" => item["original"], "ptc" => realized_ptc, "transforms" => transforms_out,
        "retain" => lazy_retain, "terminal_coordinates" => terminal)
    view = terminal_view(compiled, realization_input)
    retain_out = nothing
    if lazy_retain !== nothing
        matrices, source, deembed = view_boundary_evidence(view)
        retain_out = Dict("type" => "retain", "retained_coordinates" => terminal,
            "eliminated_coordinates" => [name for name in current_names if name ∉ terminal], "output_coordinate_order" => terminal,
            "a_matrix" => matrices["a"], "b_matrix" => matrices["b"], "r_matrix" => matrices["r"], "d_matrix" => matrices["d"],
            "q_matrix" => matrices["q"], "selected_projector" => matrices["selected_projector"], "omitted_projector" => matrices["omitted_projector"],
            "omitted_matched_loads" => matrices["omitted_matched_loads"], "source_boundary_sha256" => source, "deembedding_evidence_sha256" => deembed)
    end
    record = Dict{String,Any}("type" => "network_view_lineage", "original" => item["original"], "ptc" => realized_ptc,
        "transforms" => transforms_out, "retain" => retain_out, "terminal_coordinates" => terminal, "port_realizable" => view.port_realizable)
    record["lineage_sha256"] = sha256_hex(canonical_bytes(record))
    return record, view
end

function selected_network_response_omega(view::RealizedView, omega::ComplexF64;
        derivative::Bool = false, families::Set{String} = Set(["S", "Y", "Z"]))
    view.port_realizable || fail("validation", "port_realizability", "selected_network", "direct_response", "Direct response requires a Port-realizable final View")
    compiled = view.compiled; p = length(view.terminal); n = length(compiled.nodes)
    Yoperator = operator_at(compiled, omega; loaded = false)
    Ycirc = Yoperator / (-im * omega)
    Ycirc_p = derivative ? (operator_derivative_at(compiled, omega; loaded = false) .* (-im * omega) .+ im .* Yoperator) ./ ((-im * omega)^2) : nothing
    boundary = selected_boundary(view)
    Dk = boundary.Dk
    H = Ycirc + complex.(compiled.B) * boundary.Go * transpose(complex.(compiled.B))
    # Full-node selected-source realization.  This holds for arbitrary
    # transformed/retained Bk maps and never substitutes a node-index Schur
    # complement for the generalized Port boundary.
    Rk_inv = checked_solve(complex.(boundary.Rk), Matrix{ComplexF64}(I, p, p), "direct_response_formation", "reference_matrix", p)
    W = H + complex.(boundary.Bk) * Rk_inv * transpose(complex.(boundary.Bk))
    X = checked_solve(W, complex.(boundary.Bk), "direct_response_formation", "source_solve", n)
    Zsrc = transpose(complex.(boundary.Bk)) * X
    Ysrc = checked_solve(Zsrc, Matrix{ComplexF64}(I, p, p), "direct_response_formation", "source_admittance", p)
    Ynet = Ysrc - Rk_inv
    Ynet_p = nothing
    if derivative
        Xp = checked_solve(W, -Ycirc_p * X, "direct_response_formation", "derivative_source_solve", n)
        Zsrc_p = transpose(complex.(boundary.Bk)) * Xp
        Ynet_p = -Ysrc * Zsrc_p * Ysrc
    end
    Znet = nothing
    if "Z" in families
        Znet = checked_solve(Ynet, Matrix{ComplexF64}(I, p, p), "direct_response_formation", "y_to_z", p)
    end
    Snet = nothing
    if "S" in families
        Dkc = complex.(Dk)
        P = Matrix{ComplexF64}(I, p, p) + Dkc * Ynet * Dkc
        N = Matrix{ComplexF64}(I, p, p) - Dkc * Ynet * Dkc
        Snet = checked_solve(P, N, "direct_response_formation", "y_to_s", p)
        # Independently reconstruct at the same selected B/R boundary.  This
        # is the execution-side companion of the lineage Bk/Dk/Qk evidence.
        source = 2.0 .* complex.(boundary.Bk) * checked_solve(complex.(Dk), Matrix{ComplexF64}(I, p, p), "direct_response_formation", "reference_matrix", p)
        voltage = checked_solve(W, source, "direct_response_formation", "source_solve", n)
        source_s = checked_solve(complex.(Dk), transpose(complex.(boundary.Bk)) * voltage, "direct_response_formation", "deembedding", p) - Matrix{ComplexF64}(I, p, p)
        eta = norm(source_s - Snet, Inf) / (1.0 + norm(source_s, Inf) + norm(Snet, Inf))
        isfinite(eta) && eta <= tau(n) || fail("execution", "direct_response_formation", "deembedding", "direct_response", "source-boundary and de-embedded selected-network responses disagree")
    end
    finite_matrix(Ynet) && (Snet === nothing || finite_matrix(Snet)) && (Znet === nothing || finite_matrix(Znet)) ||
        fail("execution", "direct_response_formation", "response_formation", "direct_response", "selected N-port response is non-finite")
    if !derivative
        return Snet, Ynet, Znet
    end
    Znet_p = Znet === nothing ? nothing : -Znet * Ynet_p * Znet
    Snet_p = nothing
    if Snet !== nothing
        Dkc = complex.(Dk)
        P = Matrix{ComplexF64}(I, p, p) + Dkc * Ynet * Dkc
        Pp = Dkc * Ynet_p * Dkc; Np = -Pp
        Snet_p = checked_solve(P, Np - Pp * Snet, "direct_response_formation", "derivative_y_to_s", p)
    end
    return Snet, Ynet, Znet, Snet_p, Ynet_p, Znet_p, H
end

function selected_network_response(view::RealizedView, frequency::Float64)
    return selected_network_response_omega(view, complex(2.0 * pi * frequency))
end

function root_certificate(compiled::CompiledPrimitive, omega::ComplexF64, coordinate::String)
    n = length(compiled.nodes)
    index = findfirst(==(coordinate), compiled.nodes)
    index === nothing && fail("validation", "port_realizability", "root", "direct_quantity", "retained coordinate is absent from the compiled basis")
    r = index::Int
    Q = operator_at(compiled, omega; loaded = true)
    loaded_g = complex.(compiled.G) .+ complex.(port_load_admittance(compiled))
    Qp = operator_derivative_at(compiled, omega; loaded = true)
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
    abs_operator = operator_absolute_bound(compiled, omega; loaded = true)
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

"""Exact loaded operator and derivative on an ordered retained coordinate set."""
function selected_operator(compiled::CompiledPrimitive, omega::ComplexF64, coordinates::Vector{String})
    indices = selected_coordinate_indices(compiled, coordinates)
    Q = operator_at(compiled, omega; loaded = true)
    Qp = operator_derivative_at(compiled, omega; loaded = true)
    eliminated = [index for index in eachindex(compiled.nodes) if index ∉ indices]
    isempty(eliminated) && return Q[indices, indices], Qp[indices, indices]
    Qee, Qer = Q[eliminated, eliminated], Q[eliminated, indices]
    X = checked_solve(Qee, Qer, "eliminated_block_solve_failure", "eliminated_block", length(eliminated))
    Qpeer, Qpee = Qp[eliminated, indices], Qp[eliminated, eliminated]
    Xp = checked_solve(Qee, Qpeer - Qpee * X, "eliminated_block_solve_failure", "derivative_eliminated_block", length(eliminated))
    F = Q[indices, indices] - Q[indices, eliminated] * X
    Fp = Qp[indices, indices] - Qp[indices, eliminated] * X - Q[indices, eliminated] * Xp
    return F, Fp
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
    matrix = Array{ComplexF64}(undef, length(values), 1, 1)
    for index in eachindex(values)
        matrix[index, 1, 1] = values[index]
    end
    return write_complex_matrix_zarr(staging, artifact_id, matrix, [port_id], unit, dimensionality,
        [Dict("port_id" => port_id, "state" => "raw")])
end

function write_complex_matrix_zarr(staging::AbstractString, artifact_id::String,
    values::Array{ComplexF64,3}, coordinate_ids::Vector{String}, unit::String,
    dimensionality::String, probe_load_state::Vector{Dict{String,Any}})
    root_rel = "artifacts/" * artifact_id * ".zarr"
    root = joinpath(staging, root_rel)
    shape = [size(values, 1), size(values, 2), size(values, 3)]
    shape[2] == shape[3] && shape[2] == length(coordinate_ids) || error("complex matrix artifact coordinate shape is invalid")
    chunk_shape = [min(shape[1], 1024), shape[2], shape[3]]
    write_bytes(joinpath(root, ".zgroup"), Vector{UInt8}(codeunits("{\"zarr_format\":2}")))
    entries = Dict{String,Any}[]
    metadata = dataset_metadata(shape, chunk_shape)
    for (name, projection) in (("real", real), ("imag", imag))
        dataset = joinpath(root, name)
        mkpath(dataset)
        write_bytes(joinpath(dataset, ".zarray"), Vector{UInt8}(codeunits(zarray_metadata(shape, chunk_shape))))
        chunks = String[]
        for start in 1:chunk_shape[1]:shape[1]
            chunk_index = (start - 1) ÷ chunk_shape[1]
            chunk_name = string(chunk_index, ".0.0")
            push!(chunks, name * "/" * chunk_name)
            open(joinpath(dataset, chunk_name), "w") do io
                # Zarr C order is frequency, output, input; Julia's normal
                # iteration is column-major, so write the declared order.
                for frequency in start:min(start + chunk_shape[1] - 1, shape[1]),
                    output in 1:shape[2], input in 1:shape[3]
                    write_c_f64(io, (projection(values[frequency, output, input]),))
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
            Dict("id" => "output_coordinate", "kind" => "coordinate_output", "values" => coordinate_ids),
            Dict("id" => "input_coordinate", "kind" => "coordinate_input", "values" => coordinate_ids),
        ],
        "unit" => unit,
        "dimensionality" => dimensionality,
        "chunk_policy" => "frequency_slab_full_matrix_v1",
        "coordinate_ids" => coordinate_ids,
        "probe_load_state" => probe_load_state,
    )
end

function write_complex_vector_zarr(staging::AbstractString, artifact_id::String,
    values::Vector{ComplexF64}, coordinate_ids::Vector{String})
    length(values) == length(coordinate_ids) && length(values) >= 2 || error("null-vector artifact shape is invalid")
    root_rel = "artifacts/" * artifact_id * ".zarr"; root = joinpath(staging, root_rel)
    write_bytes(joinpath(root, ".zgroup"), Vector{UInt8}(codeunits("{\"zarr_format\":2}")))
    shape, chunks = [length(values)], [length(values)]
    entries = Dict{String,Any}[]; metadata = dataset_metadata(shape, chunks)
    for (name, projection) in (("real", real), ("imag", imag))
        dataset = joinpath(root, name); mkpath(dataset)
        write_bytes(joinpath(dataset, ".zarray"), Vector{UInt8}(codeunits(zarray_metadata(shape, chunks))))
        open(joinpath(dataset, "0"), "w") do io
            write_c_f64(io, (projection(value) for value in values))
        end
        push!(entries, dataset_entry(name, [name * "/0"]))
    end
    manifest_sha = write_manifest(staging, artifact_id, root_rel, entries)
    return Dict{String,Any}(
        "id" => artifact_id, "path" => root_rel, "sha256" => manifest_sha,
        "media_type" => "application/vnd+zarr-v2", "file_manifest" => "artifacts/" * artifact_id * ".manifest.json",
        "dtype" => "complex128", "shape" => shape, "chunks" => chunks,
        "complex_storage" => "paired_float64_real_imag", "group_metadata" => Dict("zarr_format" => 2),
        "datasets" => [Dict("path" => "real", "metadata" => metadata), Dict("path" => "imag", "metadata" => metadata)],
        "axes" => [Dict("id" => "retained_coordinate", "kind" => "coordinate", "values" => coordinate_ids)],
        "unit" => "dimensionless", "dimensionality" => "dimensionless", "chunk_policy" => "single_complete_array_v1",
        "coordinate_ids" => coordinate_ids,
    )
end

function write_operator_zarr(staging::AbstractString, values::Array{ComplexF64,3}, coordinate_ids::Vector{String},
        probe_load_state::Vector{Dict{String,Any}})
    artifact = write_complex_matrix_zarr(staging, "operator", values, coordinate_ids, "siemens / second", "conductance_per_time", probe_load_state)
    artifact["axes"] = [
        Dict("id" => "frequency", "kind" => "frequency", "artifact_id" => "frequencies"),
        Dict("id" => "row_coordinate", "kind" => "row_coordinate", "values" => coordinate_ids),
        Dict("id" => "column_coordinate", "kind" => "column_coordinate", "values" => coordinate_ids),
    ]
    artifact["chunk_policy"] = "frequency_slab_full_matrix_v1"
    return artifact
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

"""Write HB success with case-local artifact identities.

HB has deliberately repeatable local roles (`s`, `states`, and so on) for
each named case.  Its receipt/outcome therefore cannot use Direct's global
`{id,sha256}` artifact inventory; the stable identity is case + role + path.
"""
function write_hb_success(staging::String, request, request_sha::String, attempt_sha::String, result)
    result_path = joinpath(staging, "result.json")
    write_bytes(result_path, canonical_bytes(result))
    result_sha = file_sha256(result_path)
    links = Any[]
    for case in result["cases"]
        get(case, "status", nothing) == "success" || continue
        artifacts = case["artifacts"]
        for role in ("s", "y", "z", "backend_native_s", "backend_native_z", "states", "effective_source_vectors")
            artifact = artifacts[role]
            push!(links, Dict("case_id" => case["case_id"], "id" => artifact["id"], "path" => artifact["path"], "sha256" => artifact["sha256"]))
        end
        for artifact in case["traces"]
            push!(links, Dict("case_id" => case["case_id"], "id" => artifact["id"], "path" => artifact["path"], "sha256" => artifact["sha256"]))
        end
    end
    outcome = Dict{String,Any}(
        "schema" => "scnsim.outcome",
        "schema_version" => 1,
        "request_sha256" => request_sha,
        "attempt_sha256" => attempt_sha,
        "runtime_semantic" => request["runtime_semantic"],
        "status" => "success",
        "result_sha256" => result_sha,
        "artifacts" => links,
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

function solve_direct(request, view::RealizedView, request_sha::String, attempt_sha::String, staging::String)
    lineage = request["ref_lineage"]
    compiled = view.compiled
    view.port_realizable || fail("validation", "port_realizability", "selected_network", "direct_response", "Direct response requires a Port-realizable final View")
    length(compiled.port_ids) > 0 ||
        fail("validation", "port_realizability", "selected_network", "direct_response", "Direct solve requires one or more logical Ports")
    spec = request["spec"]
    frequencies = Float64[quantity_value(item) for item in spec["frequencies"]]
    isempty(frequencies) && fail("validation", "port_realizability", "frequency_grid", "direct_response", "Direct frequency grid must be nonempty")
    all(isfinite, frequencies) && all(>(0.0), frequencies) && all(diff(frequencies) .> 0.0) ||
        fail("validation", "port_realizability", "frequency_grid", "direct_response", "Direct frequency grid must be finite, positive, and strictly increasing")
    coordinates = copy(view.terminal)
    p = length(coordinates)
    s = Array{ComplexF64}(undef, length(frequencies), p, p)
    y = similar(s)
    z = similar(s)
    for (index, frequency) in enumerate(frequencies)
        response = selected_network_response(view, frequency)
        s[index, :, :] .= response[1]
        y[index, :, :] .= response[2]
        z[index, :, :] .= response[3]
    end
    frequency_artifact = write_real_zarr(staging, "frequencies", frequencies)
    compensated = lineage["ptc"] === nothing ? Set{String}() : Set(String.(lineage["ptc"]["selected_ports"]))
    states = Dict{String,Any}[Dict("port_id" => port, "state" => (port in compensated ? "compensated" : "raw")) for port in compiled.port_ids]
    s_artifact = write_complex_matrix_zarr(staging, "s", s, coordinates, "dimensionless", "dimensionless", states)
    y_artifact = write_complex_matrix_zarr(staging, "y", y, coordinates, "siemens", "conductance", states)
    z_artifact = write_complex_matrix_zarr(staging, "z", z, coordinates, "ohm", "resistance", states)
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
        fail("capability", "scaffold_unavailable", "evaluate_direct", "direct_quantity", "operation requires a diagonal-root Spec")
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
    baseline_compiled = compile_primitive(plan, baseline_values; context_kind = "direct_quantity",
        authorized = parameter_set_authorizations(request), authorization_source = "parameter_set")
    baseline_root, baseline_slope = diagonal_root(baseline_compiled, String(spec["coordinate"]), hint)
    if same_parameter_values(baseline_values, candidate_values)
        omega, slope = baseline_root, baseline_slope
    else
        selector = Dict{String,Any}("spec" => spec)
        omega = root_with_continuation(plan, request, baseline_values, candidate_values, baseline_root, selector; context_kind = "direct_quantity")
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

function evaluate_hybridized_pole(request, plan, compiled::CompiledPrimitive, request_sha::String, attempt_sha::String, staging::String)
    spec = request["spec"]
    get(spec, "type", nothing) == "hybridized_pole" || fail("execution", "compiler_invariant", "evaluate_direct", "direct_quantity", "hybridized-pole request has the wrong Spec")
    coordinates = String.(spec["coordinates"])
    lineage = request["ref_lineage"]
    lineage["terminal_coordinates"] == coordinates || fail("validation", "port_realizability", "hybridized_pole", "direct_quantity", "HybridizedPoleSpec coordinates must equal the final retained View order")
    baseline_values, candidate_values = plan_parameter_values(plan), parameter_values(request)
    if same_parameter_values(baseline_values, candidate_values)
        omega, slope, vector = hybridized_pole(compiled, coordinates, spec["anchor"])
    else
        raw_base = compile_primitive(plan, baseline_values; context_kind = "direct_quantity", authorized = parameter_set_authorizations(request), authorization_source = "parameter_set")
        _, baseline_view = realized_ref_lineage(raw_base, request["ref_lineage"])
        baseline_root = hybridized_pole(baseline_view.compiled, coordinates, spec["anchor"])[1]
        selector = Dict{String,Any}("type" => "hybridized_pole_projection", "spec" => spec, "projection" => "frequency")
        omega = selector_root_with_continuation(plan, request, baseline_values, candidate_values, baseline_root, selector;
            context_kind = "direct_quantity")
        slope = hybridized_pole(compiled, coordinates, spec["anchor"]; start = omega)[2]
        vector = hybridized_pole(compiled, coordinates, spec["anchor"]; start = omega)[3]
    end
    evidence = sha256_hex(canonical_bytes(Dict("schema" => "scnsim.hybridized_pole_evidence", "schema_version" => 1,
        "coordinates" => coordinates, "root" => complex_quantity(omega, "radian / second", "inverse_time"))))
    scalars = Dict{String,Any}(
        "root" => complex_quantity(omega, "radian / second", "inverse_time"),
        "frequency" => quantity(real(omega) / (2.0 * pi), "hertz", "inverse_time"),
        "linewidth" => quantity(-2.0 * imag(omega) / (2.0 * pi), "hertz", "inverse_time"),
        "slope" => complex_quantity(slope, "siemens", "conductance"), "evidence_sha256" => evidence,
    )
    artifact = write_complex_vector_zarr(staging, "null_vector", vector, coordinates)
    result = result_envelope("hybridized_pole", request_sha, attempt_sha, scalars, Dict("null_vector" => artifact))
    write_success(staging, request, request_sha, attempt_sha, result, [artifact])
end

function evaluate_operator(request, compiled::CompiledPrimitive, request_sha::String, attempt_sha::String, staging::String)
    spec = request["spec"]
    get(spec, "type", nothing) == "operator" || fail("execution", "compiler_invariant", "evaluate_direct", "direct_quantity", "operator request has the wrong Spec")
    frequencies = Float64[quantity_value(item) for item in spec["frequencies"]]
    !isempty(frequencies) && all(isfinite, frequencies) && all(>(0.0), frequencies) && all(diff(frequencies) .> 0.0) ||
        fail("validation", "port_realizability", "frequency_grid", "direct_quantity", "operator frequency grid must be finite, positive, and strictly increasing")
    coordinates = String.(request["ref_lineage"]["terminal_coordinates"])
    isempty(coordinates) && fail("validation", "port_realizability", "selected_network", "direct_quantity", "operator requires a nonempty terminal View")
    values = Array{ComplexF64}(undef, length(frequencies), length(coordinates), length(coordinates))
    for (index, frequency) in enumerate(frequencies)
        values[index, :, :] .= selected_operator(compiled, complex(2.0 * pi * frequency), coordinates)[1]
    end
    frequency_artifact = write_real_zarr(staging, "frequencies", frequencies)
    compensated = request["ref_lineage"]["ptc"] === nothing ? Set{String}() : Set(String.(request["ref_lineage"]["ptc"]["selected_ports"]))
    states = Dict{String,Any}[Dict("port_id" => port, "state" => (port in compensated ? "compensated" : "raw")) for port in compiled.port_ids]
    operator_artifact = write_operator_zarr(staging, values, coordinates, states)
    result = result_envelope("operator", request_sha, attempt_sha, Dict{String,Any}(), Dict("frequencies" => frequency_artifact, "operator" => operator_artifact))
    write_success(staging, request, request_sha, attempt_sha, result, [frequency_artifact, operator_artifact])
end

function evaluate_response_element(request, view::RealizedView, request_sha::String, attempt_sha::String, staging::String)
    spec = request["spec"]
    get(spec, "type", nothing) == "response_element" || fail("execution", "compiler_invariant", "evaluate_direct", "direct_quantity", "response-element request has the wrong Spec")
    coordinates = copy(view.terminal)
    input, output = String(spec["input_coordinate"]), String(spec["output_coordinate"])
    input_index = findfirst(==(input), coordinates); output_index = findfirst(==(output), coordinates)
    (input_index === nothing || output_index === nothing) && fail("validation", "port_realizability", "selected_network", "direct_quantity", "response-element coordinate is absent from the final View")
    frequency = quantity_value(spec["frequency"])
    isfinite(frequency) && frequency > 0.0 || fail("validation", "port_realizability", "frequency", "direct_quantity", "response-element frequency must be finite and positive")
    family = String(spec["family"])
    family == "S" && !view.port_realizable && fail("validation", "port_realizability", "selected_network", "direct_quantity", "S response requires a Port-realizable final View")
    value, _, _ = transfer_family_value(view, family, output_index::Int, input_index::Int, complex(2.0 * pi * frequency))
    unit, dimension = family == "S" ? ("dimensionless", "dimensionless") : family == "Y" ? ("siemens", "conductance") : family == "Z" ? ("ohm", "resistance") : fail("validation", "port_realizability", "family", "direct_quantity", "response-element family is invalid")
    evidence = sha256_hex(canonical_bytes(Dict("schema" => "scnsim.response_element_evidence", "schema_version" => 1, "family" => family, "frequency" => spec["frequency"], "input_coordinate" => input, "output_coordinate" => output)))
    scalars = Dict{String,Any}("family" => family, "value" => complex_quantity(value, unit, dimension),
        "magnitude" => quantity(abs(value), unit, dimension), "real" => quantity(real(value), unit, dimension), "imag" => quantity(imag(value), unit, dimension), "evidence_sha256" => evidence)
    result = result_envelope("response_element", request_sha, attempt_sha, scalars, Dict{String,Any}())
    write_success(staging, request, request_sha, attempt_sha, result, Any[])
end

function transfer_family_value(view::RealizedView, family::String, output::Int, input::Int, omega::ComplexF64; derivative::Bool = false)
    if family != "S" && !view.port_realizable
        F, Fp = selected_operator(view.compiled, omega, view.terminal)
        Y = F / (-im * omega)
        Yp = (Fp .* (-im * omega) .+ im .* F) ./ ((-im * omega)^2)
        if family == "Y"
            return Y[output, input], derivative ? Yp[output, input] : nothing, (nothing, Y, nothing, nothing, Yp, nothing, nothing)
        elseif family == "Z"
            Z = checked_solve(Y, Matrix{ComplexF64}(I, size(Y, 1), size(Y, 2)), "direct_response_formation", "y_to_z", size(Y, 1))
            Zp = -Z * Yp * Z
            return Z[output, input], derivative ? Zp[output, input] : nothing, (nothing, Y, Z, nothing, Yp, Zp, nothing)
        end
    end
    values = selected_network_response_omega(view, omega; derivative = derivative, families = Set([family]))
    matrices = family == "S" ? (values[1], derivative ? values[4] : nothing) :
        family == "Y" ? (values[2], derivative ? values[5] : nothing) :
        family == "Z" ? (values[3], derivative ? values[6] : nothing) :
        fail("validation", "port_realizability", "family", "direct_quantity", "transfer family is invalid")
    return matrices[1][output, input], derivative ? matrices[2][output, input] : nothing, values
end

function transfer_certificate(view::RealizedView, family::String, output::Int, input::Int, omega::ComplexF64)
    value, value_p, values = transfer_family_value(view, family, output, input, omega; derivative = true)
    S, Y, Z, Sp, Yp, Zp, H = values
    if family == "Y"
        # For retained Y, the denominator is the exact eliminated full-node
        # admittance.  A selected source boundary uses the same H as the
        # source/de-embedding solve; a quantity-only retained View uses its
        # loaded intrinsic operator.  Do not certify a different reduction.
        if view.port_realizable
            boundary = selected_boundary(view)
            Q = operator_at(view.compiled, omega; loaded = false)
            Qp = operator_derivative_at(view.compiled, omega; loaded = false)
            Hnode = Q / (-im * omega) + complex.(view.compiled.B) * boundary.Go * transpose(complex.(view.compiled.B))
            Hpnode = (Qp .* (-im * omega) .+ im .* Q) ./ ((-im * omega)^2)
        else
            Q = operator_at(view.compiled, omega; loaded = true)
            Qp = operator_derivative_at(view.compiled, omega; loaded = true)
            Hnode = Q / (-im * omega)
            Hpnode = (Qp .* (-im * omega) .+ im .* Q) ./ ((-im * omega)^2)
        end
        selected = selected_coordinate_indices(view.compiled, view.terminal); eliminated = [k for k in eachindex(view.compiled.nodes) if k ∉ selected]
        AD = isempty(eliminated) ? Matrix{ComplexF64}(I, 1, 1) : Hnode[eliminated, eliminated]
        ADp = isempty(eliminated) ? zeros(ComplexF64, 1, 1) : Hpnode[eliminated, eliminated]
        detd = isempty(eliminated) ? 1.0 + 0.0im : determinant_value(AD)
        detdp = isempty(eliminated) ? 0.0 + 0.0im : cofactor_derivative(AD, ADp)
        HRR, HRRp = Hnode[selected, selected], Hpnode[selected, selected]
        if isempty(eliminated)
            numerator_matrix, numerator_matrix_p = HRR, HRRp
        else
            HRE, HER = Hnode[selected, eliminated], Hnode[eliminated, selected]
            HREp, HERp = Hpnode[selected, eliminated], Hpnode[eliminated, selected]
            # `det(A_D) * A_D^-1 * H_ER` is exactly adj(A_D) H_ER,
            # evaluated through the required residual-checked solve rather
            # than an explicit inverse.  Its derivative follows the same
            # analytic A X'=B'-A'X rule as every Direct Schur solve.
            X = checked_solve(AD, HER, "numerical_resolution_unresolved", "transfer_denominator", size(AD, 1))
            Xp = checked_solve(AD, HERp - ADp * X, "numerical_resolution_unresolved", "transfer_denominator_derivative", size(AD, 1))
            numerator_matrix = HRR .* detd - HRE * (detd .* X)
            numerator_matrix_p = HRRp .* detd + HRR .* detdp -
                HREp * (detd .* X) - HRE * (detdp .* X + detd .* Xp)
        end
        N, Np = numerator_matrix[output, input], numerator_matrix_p[output, input]
        return value, value_p, N, Np, detd,
            reshape(ComplexF64[N], 1, 1), reshape(ComplexF64[Np], 1, 1), AD, ADp
    elseif family == "Z"
        AD, ADp = Y, Yp
        rows = [k for k in 1:size(Y, 1) if k != input]; columns = [k for k in 1:size(Y, 2) if k != output]
        AN = isempty(rows) ? Matrix{ComplexF64}(I, 1, 1) : Y[rows, columns]
        ANp = isempty(rows) ? zeros(ComplexF64, 1, 1) : Yp[rows, columns]
        sign = (-1)^(output + input)
        # Represent the Cramer sign by one row sign, rather than multiplying
        # the whole matrix (which changes det by sign^q for q>1).  The same
        # row operation is applied to the analytic derivative.
        sign < 0 && (AN[1, :] .*= -1.0; ANp[1, :] .*= -1.0)
        return value, value_p, determinant_value(AN), cofactor_derivative(AN, ANp), determinant_value(AD), AN, ANp, AD, ADp
    elseif family == "S"
        view.port_realizable || fail("validation", "port_realizability", "selected_network", "direct_quantity", "S transfer zero requires a Port-realizable View")
        D = selected_boundary(view).Dk
        P = Matrix{ComplexF64}(I, size(Y, 1), size(Y, 2)) + complex.(D) * Y * complex.(D)
        Nmatrix = Matrix{ComplexF64}(I, size(Y, 1), size(Y, 2)) - complex.(D) * Y * complex.(D)
        Pp = complex.(D) * Yp * complex.(D); Np_matrix = -Pp
        qj = Nmatrix[:, input]; AN = [P qj; -reshape([k == output ? 1.0 + 0.0im : 0.0 + 0.0im for k in 1:size(P, 1)], 1, :) zeros(ComplexF64, 1, 1)]
        ANp = [Pp Np_matrix[:, input]; zeros(ComplexF64, 1, size(P, 1) + 1)]
        return value, value_p, determinant_value(AN), cofactor_derivative(AN, ANp), determinant_value(P), AN, ANp, P, Pp
    end
    fail("validation", "port_realizability", "family", "direct_quantity", "transfer family is invalid")
end

function transfer_zero(view::RealizedView, family::String, output::Int, input::Int, anchor;
        start::Union{Nothing,ComplexF64} = nothing)
    omega = start === nothing ? 2.0 * pi * complex_frequency_value(anchor) : start
    isfinite(real(omega)) && isfinite(imag(omega)) && real(omega) > 0.0 ||
        fail("validation", "invalid_diagonal_root_hint", "anchor", "direct_quantity", "transfer-zero anchor must have finite positive real frequency")
    last_value, last_slope, last_values = 0.0 + 0.0im, 0.0 + 0.0im, nothing
    for _ in 1:32
        value, slope, values = transfer_family_value(view, family, output, input, omega; derivative = true)
        last_value, last_slope, last_values = value, slope, values
        isfinite(real(value)) && isfinite(imag(value)) && isfinite(real(slope)) && isfinite(imag(slope)) && slope != 0.0 ||
            fail("execution", "root_slope_unresolved", "transfer_numerator", "direct_quantity", "transfer-zero numerator or analytic slope is unresolved")
        next = omega - value / slope
        if reinterpret(UInt64, real(next)) == reinterpret(UInt64, real(omega)) && reinterpret(UInt64, imag(next)) == reinterpret(UInt64, imag(omega))
            omega = next; break
        end
        omega = next
    end
    value, slope, _, _, _, AN, ANp, AD, ADp = transfer_certificate(view, family, output, input, omega)
    # Certificate the declared numerator matrix and its analytic derivative,
    # separately from the transfer Newton ratio.
    numerator, numerator_slope, ANscaled, ANpscaled, _, _ = scaled_determinant_pair(AN, ANp)
    denominator, _, ADscaled, _, _, _ = scaled_determinant_pair(AD, ADp)
    # AN/AD are coherent-SI equation/unknown numerics, hence dimensionless
    # evidence matrices.  The determinant/cofactor was formed in the bounded
    # power-of-two mantissa matrix and restored with its common exponent;
    # residual/rank use that same unscaled coherent-SI equation.
    row_product = prod(norm(Base.view(AN, row, :)) for row in axes(AN, 1))
    eta_n = row_product == 0.0 ? (abs(numerator) == 0.0 ? 0.0 : Inf) : abs(numerator) / row_product
    slope_scale = sum(abs(((-1)^(a+b) * determinant_value(AN[[k for k in 1:size(AN,1) if k != a], [k for k in 1:size(AN,2) if k != b]]))) * abs(ANp[a,b]) for a in 1:size(AN,1), b in 1:size(AN,2))
    sv = svd(AN).S
    rank_ok = length(sv) == 1 || (sv[end] / sv[1] <= tau(length(sv)) && sv[end-1] / sv[1] > tau(length(sv)))
    inverse = checked_solve(AD, Matrix{ComplexF64}(I, size(AD,1), size(AD,2)), "numerical_resolution_unresolved", "transfer_denominator", size(AD,1))
    correction = abs(value / slope) / abs(omega)
    isfinite(real(omega)) && isfinite(imag(omega)) && real(omega) > 0.0 &&
        isfinite(real(denominator)) && isfinite(imag(denominator)) && denominator != 0.0 &&
        eta_n <= tau(size(AN,1)) && slope_scale > 0.0 && abs(numerator_slope)/slope_scale > tau(size(AN,1)) && rank_ok && finite_matrix(inverse) &&
        isfinite(correction) && correction <= tau(size(AN, 1)) ||
        fail("execution", "numerical_resolution_unresolved", "transfer_denominator", "direct_quantity", "transfer-zero denominator is unresolved")
    return omega, numerator_slope, denominator
end

function evaluate_transfer_zero(request, plan, view::RealizedView, request_sha::String, attempt_sha::String, staging::String)
    spec = request["spec"]; get(spec, "type", nothing) == "transfer_zero" ||
        fail("execution", "compiler_invariant", "evaluate_direct", "direct_quantity", "transfer-zero request has the wrong Spec")
    String(spec["family"]) == "S" && !view.port_realizable &&
        fail("validation", "port_realizability", "selected_network", "direct_quantity", "S transfer zero requires a Port-realizable final View")
    coordinates = view.terminal; input = findfirst(==(String(spec["input_coordinate"])), coordinates); output = findfirst(==(String(spec["output_coordinate"])), coordinates)
    (input === nothing || output === nothing) && fail("validation", "port_realizability", "selected_network", "direct_quantity", "transfer-zero coordinate is absent from final View")
    baseline_values, candidate_values = plan_parameter_values(plan), parameter_values(request)
    if same_parameter_values(baseline_values, candidate_values)
        zero, numerator_slope, denominator = transfer_zero(view, String(spec["family"]), output::Int, input::Int, spec["anchor"])
    else
        raw_base = compile_primitive(plan, baseline_values; context_kind = "direct_quantity", authorized = parameter_set_authorizations(request), authorization_source = "parameter_set")
        _, baseline_view = realized_ref_lineage(raw_base, request["ref_lineage"])
        base_zero = transfer_zero(baseline_view, String(spec["family"]), output::Int, input::Int, spec["anchor"])[1]
        selector = Dict{String,Any}("type" => "transfer_zero_projection", "spec" => spec, "projection" => "frequency")
        zero = selector_root_with_continuation(plan, request, baseline_values, candidate_values, base_zero, selector;
            context_kind = "direct_quantity")
        _, numerator_slope, denominator = transfer_zero(view, String(spec["family"]), output::Int, input::Int, spec["anchor"]; start = zero)
    end
    evidence = sha256_hex(canonical_bytes(Dict("schema" => "scnsim.transfer_zero_evidence", "schema_version" => 1, "spec" => spec,
        "zero" => complex_quantity(zero, "radian / second", "inverse_time"))))
    scalars = Dict{String,Any}("zero" => complex_quantity(zero, "radian / second", "inverse_time"),
        "frequency" => quantity(real(zero) / (2.0 * pi), "hertz", "inverse_time"),
        "numerator_slope" => complex_quantity(numerator_slope, "dimensionless", "dimensionless"),
        "denominator" => complex_quantity(denominator, "dimensionless", "dimensionless"), "evidence_sha256" => evidence)
    write_success(staging, request, request_sha, attempt_sha, result_envelope("transfer_zero", request_sha, attempt_sha, scalars, Dict{String,Any}()), Any[])
end

function residue_branch(compiled::CompiledPrimitive, coordinates::Vector{String}, spec;
        root::Union{Nothing,ComplexF64} = nothing)
    kind = String(spec["type"])
    if kind == "diagonal_root"
        coordinate = String(spec["coordinate"]); index = findfirst(==(coordinate), coordinates)
        index === nothing && fail("validation", "port_realizability", "residue_branch", "direct_quantity", "diagonal branch coordinate is absent from the common retained basis")
        omega = root === nothing ? retained_diagonal_root(compiled, coordinates, index::Int, quantity_value(spec["root_hint"])) : root
        vector = zeros(ComplexF64, length(coordinates)); vector[index::Int] = 1.0 + 0.0im
    elseif kind == "hybridized_pole"
        String.(spec["coordinates"]) == coordinates || fail("validation", "port_realizability", "residue_branch", "direct_quantity", "hybridized branch must name the complete common retained basis")
        omega, _, vector = hybridized_pole(compiled, coordinates, spec["anchor"]; start = root)
    else
        fail("validation", "port_realizability", "residue_branch", "direct_quantity", "residue branch must be diagonal-root or hybridized-pole")
    end
    vector ./= norm(vector)
    F, Fp = selected_operator(compiled, omega, coordinates)
    slope = -transpose(vector) * Fp * vector
    slope = only(slope)
    scale = sum(abs.(vector) .* (abs.(Fp) * abs.(vector)))
    isfinite(real(slope)) && isfinite(imag(slope)) && isfinite(scale) && scale > 0.0 && abs(slope) / scale > tau(length(coordinates)) ||
        fail("execution", "root_slope_unresolved", "residue_slope", "direct_quantity", "residue branch slope is unresolved")
    return omega, vector, slope, -1.0 / slope
end

"""One diagonal branch of the common retained operator, not a separately reduced View."""
function retained_diagonal_state(compiled::CompiledPrimitive, coordinates::Vector{String}, coordinate_index::Int, omega::ComplexF64)
    indices = selected_coordinate_indices(compiled, coordinates); n = length(compiled.nodes)
    Q, Qp = operator_at(compiled, omega; loaded = true), operator_derivative_at(compiled, omega; loaded = true)
    eliminated = [k for k in 1:n if k ∉ indices]
    if isempty(eliminated)
        ri = indices[coordinate_index]
        f, fp = Q[ri, ri], Qp[ri, ri]
        x = zeros(ComplexF64, n); x[ri] = 1.0 + 0.0im
        bound = operator_absolute_bound(compiled, omega; loaded = true) * abs.(x)
        closure = bound[ri] == 0.0 ? (abs(f) == 0.0 ? 0.0 : Inf) : abs(f) / bound[ri]
        return (f = f, fp = fp, scale = abs(fp), closure = closure)
    end
    X = checked_solve(Q[eliminated, eliminated], Q[eliminated, indices], "eliminated_block_solve_failure", "eliminated_block", length(eliminated))
    Xp = checked_solve(Q[eliminated, eliminated], Qp[eliminated, indices] - Qp[eliminated, eliminated] * X, "eliminated_block_solve_failure", "derivative_eliminated_block", length(eliminated))
    F = Q[indices, indices] - Q[indices, eliminated] * X
    Fp = Qp[indices, indices] - Qp[indices, eliminated] * X - Q[indices, eliminated] * Xp
    ri = indices[coordinate_index]; col = coordinate_index
    scale = abs(Qp[ri, ri]) + sum(abs.(Qp[ri, eliminated]) .* abs.(X[:, col])) + sum(abs.(Q[ri, eliminated]) .* abs.(Xp[:, col]))
    x = zeros(ComplexF64, n); x[ri] = 1.0 + 0.0im; x[eliminated] .= -X[:, col]
    bound = operator_absolute_bound(compiled, omega; loaded = true) * abs.(x)
    closure = bound[ri] == 0.0 ? (abs(F[col, col]) == 0.0 ? 0.0 : Inf) : abs(F[col, col]) / bound[ri]
    return (f = F[col, col], fp = Fp[col, col], scale = scale, closure = closure)
end

function retained_diagonal_root(compiled::CompiledPrimitive, coordinates::Vector{String}, index::Int, hint::Float64;
        start::Union{Nothing,ComplexF64} = nothing)::ComplexF64
    isfinite(hint) && hint > 0.0 || fail("validation", "invalid_diagonal_root_hint", "root_hint", "direct_quantity", "retained diagonal root hint is invalid")
    omega = start === nothing ? complex(2.0 * pi * hint) : start
    for _ in 1:32
        state = retained_diagonal_state(compiled, coordinates, index, omega)
        value, slope = state.f, state.fp
        slope != 0.0 || fail("execution", "root_slope_unresolved", "retained_diagonal_newton", "direct_quantity", "retained diagonal slope is zero")
        candidate = omega - value / slope
        same = reinterpret(UInt64, real(candidate)) == reinterpret(UInt64, real(omega)) && reinterpret(UInt64, imag(candidate)) == reinterpret(UInt64, imag(omega))
        omega = candidate; same && break
    end
    state = retained_diagonal_state(compiled, coordinates, index, omega)
    value, slope, scale = state.f, state.fp, state.scale
    correction = slope == 0.0 ? Inf : abs(value / slope) / abs(omega)
    isfinite(real(omega)) && isfinite(imag(omega)) && real(omega) > 0.0 && imag(omega) <= 0.0 &&
        isfinite(scale) && scale > 0.0 && abs(slope) / scale > tau(length(compiled.nodes)) && state.closure <= tau(length(compiled.nodes)) &&
        isfinite(correction) && correction <= tau(length(compiled.nodes)) ||
        fail("execution", "numerical_resolution_unresolved", "retained_diagonal_newton", "direct_quantity", "retained diagonal root certificate did not close")
    return omega
end

function residue_normalized_coupling_value(compiled::CompiledPrimitive, coordinates::Vector{String}, spec;
        branch_a_root::Union{Nothing,ComplexF64} = nothing, branch_b_root::Union{Nothing,ComplexF64} = nothing)
    get(spec, "type", nothing) == "residue_normalized_coupling" ||
        fail("execution", "compiler_invariant", "evaluate_direct", "direct_quantity", "residue-normalized coupling request has the wrong Spec")
    length(coordinates) >= 2 ||
        fail("validation", "port_realizability", "residue_coupling", "direct_quantity", "residue-normalized coupling requires at least two retained coordinates")
    omega_a, va, sa, residue_a = residue_branch(compiled, coordinates, spec["branch_a"]; root = branch_a_root)
    omega_b, vb, sb, residue_b = residue_branch(compiled, coordinates, spec["branch_b"]; root = branch_b_root)
    singular = svd(hcat(va, vb)).S
    length(singular) >= 2 && singular[2] / singular[1] > tau(length(coordinates)) ||
        fail("execution", "root_slope_unresolved", "residue_rank", "direct_quantity", "residue branch vectors are not independent")
    frequency = quantity_value(spec["frequency"]); isfinite(frequency) && frequency > 0.0 ||
        fail("validation", "port_realizability", "frequency", "direct_quantity", "residue coupling frequency must be finite and positive")
    for omega in unique([omega_a, omega_b, complex(2.0 * pi * frequency)])
        Fcheck, _ = selected_operator(compiled, omega, coordinates)
        denominator = norm(abs.(Fcheck) + abs.(transpose(Fcheck)), Inf)
        asym = denominator == 0.0 ? (norm(Fcheck - transpose(Fcheck), Inf) == 0.0 ? 0.0 : Inf) : norm(Fcheck - transpose(Fcheck), Inf) / denominator
        isfinite(asym) && asym <= tau(length(coordinates)) || fail("execution", "root_slope_unresolved", "reciprocity", "direct_quantity", "selected operator is not reciprocal")
    end
    F, _ = selected_operator(compiled, complex(2.0 * pi * frequency), coordinates)
    coupling = only(transpose(va) * F * vb) / sqrt(sa * sb)
    isfinite(real(coupling)) && isfinite(imag(coupling)) || fail("execution", "root_slope_unresolved", "residue_coupling", "direct_quantity", "residue-normalized coupling is non-finite")
    return coupling, residue_a, residue_b, omega_a, omega_b
end

function evaluate_residue_normalized_coupling(request, plan, view::RealizedView, request_sha::String, attempt_sha::String, staging::String)
    spec = request["spec"]
    coordinates = copy(view.terminal); compiled = view.compiled
    baseline_values, candidate_values = plan_parameter_values(plan), parameter_values(request)
    if same_parameter_values(baseline_values, candidate_values)
        coupling, residue_a, residue_b, omega_a, omega_b = residue_normalized_coupling_value(compiled, coordinates, spec)
    else
        raw_base = compile_primitive(plan, baseline_values; context_kind = "direct_quantity",
            authorized = parameter_set_authorizations(request), authorization_source = "parameter_set")
        _, baseline_view = realized_ref_lineage(raw_base, request["ref_lineage"])
        branch_a_selector, branch_b_selector = residue_branch_selector(spec["branch_a"]), residue_branch_selector(spec["branch_b"])
        base_a = selector_root_at(branch_a_selector, baseline_view.compiled, baseline_view)
        base_b = selector_root_at(branch_b_selector, baseline_view.compiled, baseline_view)
        omega_a = selector_root_with_continuation(plan, request, baseline_values, candidate_values, base_a, branch_a_selector;
            context_kind = "direct_quantity")
        omega_b = selector_root_with_continuation(plan, request, baseline_values, candidate_values, base_b, branch_b_selector;
            context_kind = "direct_quantity")
        coupling, residue_a, residue_b, omega_a, omega_b = residue_normalized_coupling_value(compiled, coordinates, spec;
            branch_a_root = omega_a, branch_b_root = omega_b)
    end
    evidence = sha256_hex(canonical_bytes(Dict("schema" => "scnsim.residue_normalized_coupling_evidence", "schema_version" => 1,
        "branch_a_root" => complex_quantity(omega_a, "radian / second", "inverse_time"), "branch_b_root" => complex_quantity(omega_b, "radian / second", "inverse_time"))))
    scalars = Dict{String,Any}("coupling" => complex_quantity(coupling, "radian / second", "inverse_time"),
        "magnitude" => quantity(abs(coupling), "radian / second", "inverse_time"),
        "branch_a_residue" => complex_quantity(residue_a, "ohm", "resistance"), "branch_b_residue" => complex_quantity(residue_b, "ohm", "resistance"), "evidence_sha256" => evidence)
    write_success(staging, request, request_sha, attempt_sha, result_envelope("residue_normalized_coupling", request_sha, attempt_sha, scalars, Dict{String,Any}()), Any[])
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

function bootstrap_record(request, request_sha::String, ordinal::Int)
    version = string(VERSION)
    version == "1.12.6" || error("SCNSim backend must run under Julia 1.12.6")
    record = Dict{String,Any}(
        "attempt_ordinal" => ordinal,
        "blas_threads" => BLAS.get_num_threads(),
        "blas_vendor" => string(BLAS.vendor()),
        "julia_threads" => Threads.nthreads(),
        "julia_version" => version,
        "request_sha256" => request_sha,
        "schema" => "scnsim.bootstrap_ready",
        "schema_version" => 1,
    )
    if get(request, "operation", nothing) == "solve_hb"
        # HB is prepared before attempt allocation so the Python side can bind
        # its one-thread FFTW evidence into attempt.json along with BLAS.
        hb_require_runtime!()
        record["fftw_threads"] = JosephsonCircuits.FFTW.get_num_threads()
    end
    return record
end

function read_authorization(request, request_sha::String, staging::String, ordinal::Int)
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
    if get(request, "operation", nothing) == "solve_hb"
        get(attempt, "fftw_threads", nothing) == 1 || error("attempt FFTW thread evidence violates HB policy")
    end
    return attempt_sha
end

function run_terminal(request_path::String, staging::String)
    request, request_sha, plan = read_request_and_plan(request_path)
    ordinal = staging_ordinal(staging)
    println(canonical_json(bootstrap_record(request, request_sha, ordinal)))
    flush(stdout)
    attempt_sha = read_authorization(request, request_sha, staging, ordinal)
    # Python owns this envelope and its canonical encoder. The staged attempt
    # was already bound by the authorization hash, so only parse it here.
    attempt = canonical_document(joinpath(staging, "attempt.json"); require_backend_canonical = false)
    resume_ledger_sha = get(attempt, "resume_ledger_sha256", nothing)
    (resume_ledger_sha === nothing || resume_ledger_sha isa AbstractString) ||
        fail("evidence", "evidence_integrity", "optimization_replay", "attempt", "resume ledger hash is malformed")
    try
        request["runtime_semantic"]["julia_version"] == "1.12.6" || error("request runtime identity has wrong Julia version")
        operation = request["operation"]
        compile_context = operation == "evaluate_direct" ? "direct_quantity" : "compile"
        raw_compiled = compile_primitive(plan, parameter_values(request); context_kind = compile_context,
            authorized = parameter_set_authorizations(request), authorization_source = "parameter_set")
        _, view = realized_ref_lineage(raw_compiled, request["ref_lineage"])
        compiled = view.compiled
        if operation == "solve_direct"
            solve_direct(request, view, request_sha, attempt_sha, staging)
        elseif operation == "solve_hb"
            # HB lowers from the raw recursive graph so Josephson rows and
            # original B/R/M remain physical authority; `view` is the same
            # realized selected-network lineage used by Direct.
            solve_hb(request, plan, raw_compiled, view, request_sha, attempt_sha, staging)
        elseif operation == "evaluate_direct"
            kind = get(request["spec"], "type", nothing)
            if kind == "diagonal_root"
                evaluate_diagonal_root(request, plan, compiled, request_sha, attempt_sha, staging)
            elseif kind == "hybridized_pole"
                evaluate_hybridized_pole(request, plan, compiled, request_sha, attempt_sha, staging)
            elseif kind == "transfer_zero"
                evaluate_transfer_zero(request, plan, view, request_sha, attempt_sha, staging)
            elseif kind == "residue_normalized_coupling"
                evaluate_residue_normalized_coupling(request, plan, view, request_sha, attempt_sha, staging)
            elseif kind == "operator"
                evaluate_operator(request, compiled, request_sha, attempt_sha, staging)
            elseif kind == "response_element"
                evaluate_response_element(request, view, request_sha, attempt_sha, staging)
            else
                fail("capability", "scaffold_unavailable", "evaluate_direct", "direct_quantity", "Direct quantity is not implemented by this backend revision")
            end
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

function has_offdiagonal_series_resistance(compiled::CompiledPrimitive)::Bool
    for block in compiled.series_rl
        block isa SeriesRLBlock || continue
        for row in axes(block.resistance, 1), column in axes(block.resistance, 2)
            row != column && block.resistance[row, column] != 0.0 && return true
        end
    end
    return false
end

function resolved_bindings(plan, resolved::Dict{String,Float64})
    values = Dict{String,Any}[]
    for component in plan["components"]
        realization = component["realization"]
        for entry in component["parameter_bindings"]
            parameter = String(entry["id"])
            key = ref_key(Dict("component_path" => component["component_path"], "parameter_id" => parameter))
            haskey(resolved, key) || continue
            envelope = if String(realization["kind"]) == "composite"
                declarations = [item for item in realization["public_parameters"] if String(item["id"]) == parameter]
                length(declarations) == 1 || fail("execution", "compiler_invariant", "compile", "compile", "Composite public parameter declaration is missing or ambiguous")
                declaration = only(declarations)
                haskey(declaration, "spec") && haskey(declaration, "baseline") ||
                    fail("execution", "compiler_invariant", "compile", "compile", "Composite public parameter declaration lacks exact unit evidence")
                spec = declaration["spec"]
                baseline = declaration["baseline"]
                for field in ("si_unit", "dimensionality")
                    haskey(spec, field) && haskey(baseline, field) && spec[field] == baseline[field] ||
                        fail("execution", "compiler_invariant", "compile", "compile", "Composite public parameter unit evidence is inconsistent")
                end
                baseline
            else
                binding = entry["binding"]
                binding["kind"] == "constant" && haskey(binding, "value") ||
                    fail("execution", "compiler_invariant", "compile", "compile", "primitive public baseline is not a canonical constant quantity")
                binding["value"]
            end
            haskey(envelope, "si_unit") && haskey(envelope, "dimensionality") ||
                fail("execution", "compiler_invariant", "compile", "compile", "public baseline quantity lacks unit evidence")
            push!(values, Dict{String,Any}(
                "parameter" => Dict("component_path" => component["component_path"], "parameter_id" => parameter),
                "value" => quantity(resolved[key], String(envelope["si_unit"]), String(envelope["dimensionality"])),
            ))
        end
    end
    sort!(values; by = item -> ref_key(item["parameter"]))
    return values
end

function preflight(plan_path::String, request_path::String)
    plan_bytes = read(plan_path)
    plan_sha = sha256_hex(plan_bytes)
    plan = plain(JSON3.read(String(plan_bytes)))
    request = plain(JSON3.read(read(request_path, String)))
    get(request, "schema", nothing) == "scnsim.request" ||
        fail("execution", "compiler_invariant", "preflight", "compile", "preflight request schema is invalid")
    request["plan_sha256"] == plan_sha ||
        fail("execution", "compiler_invariant", "preflight", "compile", "preflight request does not bind the supplied Plan")
    operation = String(request["operation"])
    context = operation == "evaluate_direct" ? "direct_quantity" : "compile"
    request_values = parameter_values(request)
    resolved = recursive_parameter_values(plan, request_values; context_kind = context)
    raw_compiled = compile_primitive(plan, request_values; context_kind = context,
        authorized = parameter_set_authorizations(request), authorization_source = "parameter_set",
        emit_audit = true)
    realized_lineage, view = realized_ref_lineage(raw_compiled, request["ref_lineage"])
    compiled = view.compiled
    load = port_load_admittance(compiled)
    runtime_path = normpath(joinpath(@__DIR__, "..", "runtime.json"))
    runtime = plain(JSON3.read(read(runtime_path, String)))
    return Dict{String,Any}(
        "schema" => "scnsim.preflight",
        "schema_version" => 1,
        "plan_sha256" => plan_sha,
        "runtime" => runtime,
        "resolved_bindings" => resolved_bindings(plan, resolved),
        "ref_lineage" => realized_lineage,
        "node_order" => compiled.nodes,
        "matrix_order" => "canonical_node_id",
        "expanded_branch_rows" => compiled.branch_rows,
        "c_matrix" => f64_matrix_evidence(compiled.C),
        "k_matrix" => f64_matrix_evidence(compiled.K),
        "g_matrix" => f64_matrix_evidence(compiled.G),
        "ports" => Dict(
            "ids" => compiled.port_ids,
            "selector" => f64_matrix_evidence(compiled.B),
            "reference_matrix" => f64_matrix_evidence(compiled.R),
            "load_mask_f64" => f64_hex.(compiled.M),
            "load_stamp" => f64_matrix_evidence(load),
            "selected_network_steps" => ["intrinsic_CKG", "port_load_BY0BT", "source_boundary", "power_wave_deembedding"],
        ),
        "root_preflight" => Dict("supported" => "single_retained_coordinate", "algorithm_id" => runtime["algorithm_ids"]["diagonal_root"]),
        "optimization_preflight" => Dict("supported" => "dev5_full_scalar_selector_catalog", "algorithm_id" => runtime["algorithm_ids"]["optimization"]),
        "direct_hb_capability" => Dict(
            "direct" => "full_rlgc_nport_selected_network",
            "hb" => (has_offdiagonal_series_resistance(compiled) ? "unsupported_off_diagonal_series_resistance" : "josephsoncircuits_hb_candidate"),
        ),
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
        fail("capability", "scaffold_unavailable", "optimization", "optimization_candidate", "root continuation requires a diagonal-root selector")
    return canonical_json(selector["spec"])
end

selector_key(selector) = canonical_json(selector)

function root_selector_specs(selector, found::Dict{String,Any} = Dict{String,Any}())
    selector_type = get(selector, "type", nothing)
    if selector_type in ("diagonal_root_projection", "hybridized_pole_projection", "transfer_zero_projection")
        found[selector_key(selector)] = selector
    elseif selector_type == "residue_coupling_projection"
        # A residue objective has two anchored branch locators.  They are
        # private continuation dependencies, not a second public selector.
        for branch in (selector["spec"]["branch_a"], selector["spec"]["branch_b"])
            branch_selector = residue_branch_selector(branch)
            found[selector_key(branch_selector)] = branch_selector
        end
    elseif selector_type == "response_element_projection"
        nothing
    elseif selector_type == "quantity_sum"
        terms = get(selector, "terms", nothing)
        terms isa AbstractVector && !isempty(terms) ||
            fail("validation", "invalid_optimization_spec", "quantity_sum", "optimization_candidate", "QuantitySum requires one or more terms")
        for term in terms
            root_selector_specs(term, found)
        end
    else
        fail("capability", "scaffold_unavailable", "optimization", "optimization_candidate", "optimization selector is unsupported")
    end
    return found
end

function residue_branch_selector(branch)
    kind = String(branch["type"])
    if kind == "diagonal_root"
        return Dict{String,Any}("type" => "residue_diagonal_root_projection", "spec" => branch, "projection" => "frequency")
    elseif kind == "hybridized_pole"
        return Dict{String,Any}("type" => "hybridized_pole_projection", "spec" => branch, "projection" => "frequency")
    end
    fail("validation", "invalid_optimization_spec", "residue_branch", "optimization_candidate", "residue branch has an unsupported locator")
end

function selector_root_at(selector, compiled::CompiledPrimitive, view::RealizedView;
        start::Union{Nothing,ComplexF64} = nothing)::ComplexF64
    kind = String(selector["type"]); spec = selector["spec"]
    if kind == "diagonal_root_projection"
        return diagonal_root(compiled, String(spec["coordinate"]), quantity_value(spec["root_hint"]); start = start)[1]
    elseif kind == "residue_diagonal_root_projection"
        coordinate = String(spec["coordinate"]); index = findfirst(==(coordinate), view.terminal)
        index === nothing && fail("validation", "invalid_optimization_spec", "residue_branch", "optimization_candidate", "residue diagonal coordinate is absent from the terminal View")
        return retained_diagonal_root(compiled, view.terminal, index::Int, quantity_value(spec["root_hint"]); start = start)
    elseif kind == "hybridized_pole_projection"
        return hybridized_pole(compiled, String.(spec["coordinates"]), spec["anchor"]; start = start)[1]
    elseif kind == "transfer_zero_projection"
        input = findfirst(==(String(spec["input_coordinate"])), view.terminal); output = findfirst(==(String(spec["output_coordinate"])), view.terminal)
        (input === nothing || output === nothing) && fail("validation", "invalid_optimization_spec", "selector", "optimization_candidate", "transfer-zero selector coordinate is absent")
        return transfer_zero(view, String(spec["family"]), output::Int, input::Int, spec["anchor"]; start = start)[1]
    end
    fail("execution", "compiler_invariant", "optimization", "optimization_candidate", "selector has no continuation root")
end

function compiled_candidate_view(plan, request, values::Dict{String,Float64}; context_kind::String)
    raw = compile_primitive(plan, values; context_kind = context_kind,
        authorized = context_kind == "optimization_candidate" ? optimization_authorizations(request) : parameter_set_authorizations(request),
        authorization_source = context_kind == "optimization_candidate" ? "optimization_spec" : "parameter_set")
    _, view = realized_ref_lineage(raw, request["ref_lineage"])
    return view.compiled, view
end

function selector_root_with_continuation(plan, request, baseline_values, candidate_values, baseline_root::ComplexF64, selector;
        context_kind::String = "optimization_candidate")
    function values_at(t::Float64)
        t == 0.0 && return copy(baseline_values)
        t == 1.0 && return copy(candidate_values)
        return Dict(key => baseline_values[key] + t * (candidate_values[key] - baseline_values[key]) for key in keys(baseline_values))
    end
    function advance(left_t::Float64, left_root::ComplexF64, right_t::Float64, depth::Int)::ComplexF64
        try
            compiled, view = compiled_candidate_view(plan, request, values_at(right_t); context_kind = context_kind)
            return selector_root_at(selector, compiled, view; start = left_root)
        catch error
            error isa BackendFailure || rethrow()
            # At the CMA selector boundary, a candidate-only S/Y/Z formation
            # singularity/non-finite result has the documented +Inf owner.
            # It is not a Plan/reference invariant and must not turn an
            # otherwise valid population evaluation into an attempt failure.
            if context_kind == "optimization_candidate" && error.kind == "direct_response_formation"
                fail("execution", "numerical_resolution_unresolved", error.stage, "optimization_candidate", "candidate selected response is numerically unresolved")
            end
            # Only numerical resolution is repairable by the accepted dyadic
            # path.  Physical, selected-network, affine and slope failures
            # retain their original typed owner.
            error.kind == "numerical_resolution_unresolved" || rethrow()
            depth < 32 || rethrow()
            midpoint = (left_t + right_t) / 2.0
            midpoint_root = advance(left_t, left_root, midpoint, depth + 1)
            return advance(midpoint, midpoint_root, right_t, depth + 1)
        end
    end
    return advance(0.0, baseline_root, 1.0, 0)
end

"""The public unit convention of one scalar selector.

`QuantitySum` owns no new unit: Python binds its target/scale to term zero, so
the backend must fold every later equal-dimensionality term in that same public
convention.  In particular coupling is angular rate while root/zero selectors
publish cycles per second.
"""
function selector_public_unit(selector)::String
    kind = String(selector["type"])
    if kind in ("diagonal_root_projection", "hybridized_pole_projection", "transfer_zero_projection", "residue_diagonal_root_projection")
        return "hertz"
    elseif kind == "residue_coupling_projection"
        return "radian / second"
    elseif kind == "response_element_projection"
        family = String(selector["spec"]["family"])
        family == "S" && return "dimensionless"
        family == "Y" && return "siemens"
        family == "Z" && return "ohm"
    elseif kind == "quantity_sum"
        terms = selector["terms"]
        terms isa AbstractVector && !isempty(terms) ||
            fail("validation", "invalid_optimization_spec", "quantity_sum", "optimization_candidate", "QuantitySum requires one or more terms")
        return selector_public_unit(terms[1])
    end
    fail("validation", "invalid_optimization_spec", "selector", "optimization_candidate", "optimization selector has no public scalar unit")
end

function selector_value_in_unit(value::Float64, source::String, target::String)::Float64
    source == target && return value
    if source == "radian / second" && target == "hertz"
        return value / (2.0 * pi)
    elseif source == "hertz" && target == "radian / second"
        return value * (2.0 * pi)
    end
    fail("validation", "invalid_optimization_spec", "quantity_sum", "optimization_candidate", "QuantitySum terms do not share a convertible public unit convention")
end

function root_selector_value(selector, plan, request, baseline_values, values, baseline_roots, roots, compiled::CompiledPrimitive, view::RealizedView)::Float64
    selector_type = get(selector, "type", nothing)
    if selector_type == "quantity_sum"
        terms = selector["terms"]
        isempty(terms) && fail("validation", "invalid_optimization_spec", "quantity_sum", "optimization_candidate", "QuantitySum requires one or more terms")
        # Declaration order is semantic authority.  Do not let a reduction
        # implementation choose regrouping or summation order.
        public_unit = selector_public_unit(terms[1])
        total = root_selector_value(terms[1], plan, request, baseline_values, values, baseline_roots, roots, compiled, view)
        for term in terms[2:end]
            value = root_selector_value(term, plan, request, baseline_values, values, baseline_roots, roots, compiled, view)
            total = total + selector_value_in_unit(value, selector_public_unit(term), public_unit)
        end
        return total
    end
    projection = selector["projection"]
    if selector_type == "diagonal_root_projection"
        key = selector_key(selector)
        root = get!(roots, key) do
            same_parameter_values(baseline_values, values) ? baseline_roots[key] :
                selector_root_with_continuation(plan, request, baseline_values, values, baseline_roots[key], selector)
        end
        projection == "frequency" && return real(root) / (2.0 * pi)
        projection == "linewidth" && return -2.0 * imag(root) / (2.0 * pi)
    elseif selector_type == "hybridized_pole_projection"
        key = selector_key(selector)
        root = get!(roots, key) do
            same_parameter_values(baseline_values, values) ? baseline_roots[key] :
                selector_root_with_continuation(plan, request, baseline_values, values, baseline_roots[key], selector)
        end
        projection == "frequency" && return real(root) / (2.0 * pi)
        projection == "linewidth" && return -2.0 * imag(root) / (2.0 * pi)
    elseif selector_type == "transfer_zero_projection"
        spec = selector["spec"]; input = findfirst(==(String(spec["input_coordinate"])), view.terminal); output = findfirst(==(String(spec["output_coordinate"])), view.terminal)
        (input === nothing || output === nothing) && fail("validation", "invalid_optimization_spec", "selector", "optimization_candidate", "transfer-zero selector coordinate is absent")
        key = selector_key(selector)
        zero = get!(roots, key) do
            same_parameter_values(baseline_values, values) ? baseline_roots[key] :
                selector_root_with_continuation(plan, request, baseline_values, values, baseline_roots[key], selector)
        end
        projection == "frequency" && return real(zero) / (2.0 * pi)
    elseif selector_type == "response_element_projection"
        spec = selector["spec"]; input = findfirst(==(String(spec["input_coordinate"])), view.terminal); output = findfirst(==(String(spec["output_coordinate"])), view.terminal)
        (input === nothing || output === nothing) && fail("validation", "invalid_optimization_spec", "selector", "optimization_candidate", "response selector coordinate is absent")
        value = try
            transfer_family_value(view, String(spec["family"]), output::Int, input::Int, complex(2.0 * pi * quantity_value(spec["frequency"])))[1]
        catch error
            error isa BackendFailure || rethrow()
            if !same_parameter_values(baseline_values, values) && error.kind == "direct_response_formation"
                fail("execution", "numerical_resolution_unresolved", error.stage, "optimization_candidate", "candidate selected response is numerically unresolved")
            end
            rethrow()
        end
        projection == "magnitude" && return abs(value)
        projection == "real" && return real(value)
        projection == "imag" && return imag(value)
    elseif selector_type == "residue_coupling_projection"
        branch_a_selector = residue_branch_selector(selector["spec"]["branch_a"])
        branch_b_selector = residue_branch_selector(selector["spec"]["branch_b"])
        key_a, key_b = selector_key(branch_a_selector), selector_key(branch_b_selector)
        root_a = get!(roots, key_a) do
            same_parameter_values(baseline_values, values) ? baseline_roots[key_a] :
                selector_root_with_continuation(plan, request, baseline_values, values, baseline_roots[key_a], branch_a_selector)
        end
        root_b = get!(roots, key_b) do
            same_parameter_values(baseline_values, values) ? baseline_roots[key_b] :
                selector_root_with_continuation(plan, request, baseline_values, values, baseline_roots[key_b], branch_b_selector)
        end
        value = residue_normalized_coupling_value(compiled, view.terminal, selector["spec"];
            branch_a_root = root_a, branch_b_root = root_b)[1]
        projection == "magnitude" && return abs(value)
    end
    fail("validation", "invalid_optimization_spec", "selector", "optimization_candidate", "optimization selector projection is invalid")
end

function same_parameter_values(left::Dict{String,Float64}, right::Dict{String,Float64})::Bool
    keys(left) == keys(right) || return false
    return all(f64_hex(left[key]) == f64_hex(right[key]) for key in keys(left))
end

function plan_parameter_values(plan)::Dict{String,Float64}
    return recursive_baselines(plan)
end

function optimization_authorizations(request)::Set{String}
    refs = get(request["spec"], "allow_extrapolation", Any[])
    refs isa AbstractVector || fail("validation", "invalid_optimization_spec", "allow_extrapolation", "optimization_candidate", "optimization authorization collection is malformed")
    return Set(ref_key(reference) for reference in refs)
end

function root_with_continuation(plan, request, baseline_values, candidate_values, baseline_root::ComplexF64, selector; context_kind::String = "direct_quantity")
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
            return diagonal_root(compile_primitive(plan, right_values; context_kind = context_kind,
                authorized = context_kind == "optimization_candidate" ? optimization_authorizations(request) : parameter_set_authorizations(request),
                authorization_source = context_kind == "optimization_candidate" ? "optimization_spec" : "parameter_set"), coordinate, hint; start = left_root)[1]
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

function objective_outcome(plan, request, baseline_values, values, baseline_roots;
        extrapolation_evidence::Vector{Any} = Any[])
    raw_compiled = compile_primitive(plan, values; context_kind = "optimization_candidate",
        authorized = optimization_authorizations(request), extrapolation_evidence = extrapolation_evidence,
        authorization_source = "optimization_spec")
    _, view = realized_ref_lineage(raw_compiled, request["ref_lineage"])
    compiled = view.compiled
    components = Any[]
    total = 0.0
    roots = Dict{String,ComplexF64}()
    for objective in request["spec"]["objectives"]
        selector = objective["quantity"]
        selector isa AbstractDict || fail("capability", "scaffold_unavailable", "optimization", "optimization_candidate", "optimization quantity must be a selector record")
        value = root_selector_value(selector, plan, request, baseline_values, values, baseline_roots, roots, compiled, view)
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
    sort!(extrapolation_evidence; by = row -> (ref_key(row["parameter"]), ref_key(row["consumer_target"])))
    return total, components, extrapolation_evidence
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

function candidate_record(request, ordinal::Int, generation::Int, column, z::Vector{Float64}, latent, parameters, cache_hit::Bool, outcome;
        extrapolation_evidence::Vector{Any} = Any[])
    record = Dict{String,Any}(
        "evaluation_ordinal" => ordinal,
        "origin" => generation == 0 ? "baseline" : "population",
        "generation" => generation,
        "population_column" => column,
        "optimizer_coordinates_f64" => f64_hex.(z),
        "parameters" => parameters,
        "cache_hit" => cache_hit,
        "extrapolation_evidence" => extrapolation_evidence,
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
                cache[key] = Dict("outcome" => candidate["outcome"], "cost" => cost,
                    "extrapolation_evidence" => candidate["extrapolation_evidence"])
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
    # The closed failure envelope has no separate baseline context.  Baseline
    # binding uses the same declared optimization authorization and typed
    # `optimization_candidate` context; its ledger position distinguishes it.
    compiled, view = compiled_candidate_view(plan, request, values; context_kind = "optimization_candidate")
    roots = Dict{String,ComplexF64}()
    for objective in request["spec"]["objectives"]
        selector = objective["quantity"]
        for (key, root_selector) in root_selector_specs(selector)
            if !haskey(roots, key)
                roots[key] = selector_root_at(root_selector, compiled, view)
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
        return candidate_record(request, ordinal, generation, column, z, latent, parameters, true, cached["outcome"];
            extrapolation_evidence = cached["extrapolation_evidence"]), cached["cost"]
    end
    extrapolation_evidence = Any[]
    try
        cost, components, extrapolation_evidence = objective_outcome(plan, request, baseline_values, values, baseline_roots;
            extrapolation_evidence = extrapolation_evidence)
        outcome = Dict{String,Any}(
            "status" => "success",
            "cost_f64" => f64_hex(cost),
            "objective_components" => components,
        )
        cache[key] = Dict("outcome" => outcome, "cost" => cost, "extrapolation_evidence" => extrapolation_evidence)
        return candidate_record(request, ordinal, generation, column, z, latent, parameters, false, outcome;
            extrapolation_evidence = extrapolation_evidence), cost
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
        sort!(extrapolation_evidence; by = row -> (ref_key(row["parameter"]), ref_key(row["consumer_target"])))
        cache[key] = Dict("outcome" => outcome, "cost" => Inf, "extrapolation_evidence" => extrapolation_evidence)
        return candidate_record(request, ordinal, generation, column, z, latent, parameters, false, outcome;
            extrapolation_evidence = extrapolation_evidence), Inf
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
    baseline_evidence = Any[]
    baseline_cost, baseline_components, baseline_evidence = try
        objective_outcome(plan, request, base_values, base_values, baseline_roots;
            extrapolation_evidence = baseline_evidence)
    catch error
        error isa BackendFailure || rethrow()
        rethrow()
    end
    baseline_outcome = Dict{String,Any}(
        "status" => "success",
        "cost_f64" => f64_hex(baseline_cost),
        "objective_components" => baseline_components,
    )
    baseline = candidate_record(request, 0, 0, nothing, z0, nothing, baseline_parameters, false, baseline_outcome;
        extrapolation_evidence = baseline_evidence)
    cache[canonical_json(baseline_parameters)] = Dict("outcome" => baseline_outcome, "cost" => baseline_cost, "extrapolation_evidence" => baseline_evidence)
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

# The HB surface is deliberately included after the shared compiler, selected
# network realization, artifact writer, and terminal protocol are defined.
# It consumes those authorities; it does not create a second graph/runtime.
include("hb.jl")

end # module
