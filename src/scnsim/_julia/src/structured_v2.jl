"""Normalized Plan-v2 physical lowering.

This file is the sole Julia topology compiler for the structured source model.
It consumes the canonical physical-leaf/connectivity tables directly; scope
hierarchy and occurrences remain authoring/provenance facts and are never
re-expanded into a competing electrical graph.
"""

structured_path(value) = String.(plain(value))
structured_path_key(path) = join(structured_path(path), "\u001f")
structured_field_key(path, field) = structured_path_key(path) * "\u001e" * String(field)
structured_endpoint_key(path, pin) = structured_path_key(path) * "\u001e" * String(pin)
structured_branch_key(reference) = structured_path_key(plain(reference)["path"]) * "\u001e" * String(plain(reference)["branch_id"])
exact_keys(item, expected) = Set(String.(keys(item))) == Set(String.(expected))

function structured_value(value)
    item = plain(value)
    kind = get(item, "type", nothing)
    kind == "quantity_f64" && return quantity_value(item)
    kind == "rlgc" && return item
    fail("execution", "compiler_invariant", "compile", "compile", "parameter value is neither quantity_f64 nor rlgc")
end

function structured_parameter_values(parameter_set)::Dict{String,Any}
    item = plain(parameter_set)
    get(item, "type", nothing) == "parameter_set_v2" ||
        fail("execution", "compiler_invariant", "compile", "compile", "expected parameter_set_v2")
    exact_keys(item, ("type", "bindings", "allow_extrapolation")) ||
        fail("execution", "compiler_invariant", "compile", "compile", "parameter_set_v2 fields are invalid")
    bindings = get(item, "bindings", nothing)
    bindings isa AbstractVector || fail("execution", "compiler_invariant", "compile", "compile", "parameter bindings are malformed")
    values = Dict{String,Any}()
    for binding in bindings
        exact_keys(binding, ("parameter", "value")) && exact_keys(binding["parameter"], ("definitions_id", "parameter_id")) ||
            fail("execution", "compiler_invariant", "compile", "compile", "parameter binding fields are invalid")
        key = ref_key(binding["parameter"])
        haskey(values, key) && fail("execution", "compiler_invariant", "compile", "compile", "duplicate resolved parameter binding")
        values[key] = structured_value(binding["value"])
    end
    return values
end

function structured_authorizations(parameter_set)::Set{String}
    item = plain(parameter_set)
    refs = get(item, "allow_extrapolation", Any[])
    refs isa AbstractVector || fail("execution", "compiler_invariant", "affine_support", "compile", "parameter authorization collection is malformed")
    all(ref -> exact_keys(ref, ("definitions_id", "parameter_id")), refs) ||
        fail("execution", "compiler_invariant", "affine_support", "compile", "parameter authorization reference is malformed")
    return Set(ref_key(reference) for reference in refs)
end

function structured_scalar_binding(binding, values::Dict{String,Any}; context_kind::String,
        authorized::Set{String}, extrapolation_evidence::Union{Nothing,Vector{Any}},
        target, authorization_source::String, fail_unauthorized::Bool = true)::Float64
    item = plain(binding); kind = get(item, "kind", nothing)
    if kind == "constant"
        return quantity_value(item["value"])
    elseif kind == "ref"
        key = ref_key(item["parameter"])
        haskey(values, key) || fail("execution", "compiler_invariant", "compile", "compile", "missing resolved parameter binding")
        values[key] isa Float64 || fail("execution", "compiler_invariant", "compile", "compile", "scalar field references a non-scalar parameter")
        return values[key]
    elseif kind == "affine"
        key = ref_key(item["input"])
        haskey(values, key) || fail("execution", "compiler_invariant", "compile", "compile", "missing resolved affine input")
        values[key] isa Float64 || fail("execution", "compiler_invariant", "compile", "compile", "affine field references a non-scalar parameter")
        support = item["support"]
        support isa AbstractVector && length(support) == 2 || fail("execution", "compiler_invariant", "compile", "compile", "affine support must have two bounds")
        lower, upper, input = quantity_value(support[1]), quantity_value(support[2]), values[key]
        if !(lower <= input <= upper)
            side, distance = input < lower ? ("lower", lower - input) : ("upper", input - upper)
            extrapolation_evidence === nothing || push!(extrapolation_evidence, Dict{String,Any}(
                "parameter" => plain(item["input"]), "consumer_target" => plain(target),
                "support" => Any[plain(support[1]), plain(support[2])],
                "input_value" => quantity(input, String(support[1]["si_unit"]), String(support[1]["dimensionality"])),
                "side" => side,
                "distance" => quantity(distance, String(support[1]["si_unit"]), String(support[1]["dimensionality"])),
                "authorization_source" => (key in authorized ? authorization_source : "none"),
            ))
            if fail_unauthorized && !(key in authorized)
                fail("execution", "invalid_candidate_physical_parameter", "affine_support", context_kind, "affine input is outside its declared support")
            end
        end
        return quantity_value(item["slope"]) * input + quantity_value(item["intercept"])
    end
    fail("execution", "compiler_invariant", "compile", "compile", "unknown physical-field binding kind")
end

function structured_resolve_fields(plan, values::Dict{String,Any}; context_kind::String = "compile",
        authorized::Set{String} = Set{String}(), extrapolation_evidence::Union{Nothing,Vector{Any}} = nothing,
        authorization_source::String = "none", fail_unauthorized::Bool = true)
    resolved = Dict{String,Any}()
    parameter_envelopes = Dict(ref_key(definition) => definition["baseline"] for definition in plan["parameter_closure"]["definitions"])
    rows = Dict{String,Any}[]
    for leaf in plan["physical_leaves"]
        path = structured_path(leaf["path"])
        for field in leaf["fields"]
            id = String(field["id"]); binding = plain(field["binding"])
            key = structured_field_key(path, id)
            haskey(resolved, key) && fail("execution", "compiler_invariant", "compile", "compile", "physical field is duplicated")
            value = if get(binding, "kind", nothing) == "constant" && get(binding["value"], "type", nothing) == "rlgc"
                plain(binding["value"])
            elseif get(binding, "kind", nothing) == "ref"
                parameter_key = ref_key(binding["parameter"])
                haskey(values, parameter_key) || fail("execution", "compiler_invariant", "compile", "compile", "physical field references a missing parameter")
                values[parameter_key]
            else
                structured_scalar_binding(binding, values; context_kind = context_kind, authorized = authorized,
                    extrapolation_evidence = extrapolation_evidence, target = Dict("path" => path, "field" => id),
                    authorization_source = authorization_source, fail_unauthorized = fail_unauthorized)
            end
            unit = String(field["unit"])
            (unit == "rlgc") == (value isa AbstractDict && get(value, "type", nothing) == "rlgc") ||
                fail("execution", "compiler_invariant", "compile", "compile", "resolved physical field has the wrong value type")
            resolved[key] = value
            encoded = if value isa Float64
                if get(binding, "kind", nothing) == "ref"
                    parameter_key = ref_key(binding["parameter"])
                    haskey(parameter_envelopes, parameter_key) || fail("execution", "compiler_invariant", "compile", "compile", "physical field references an unknown parameter definition")
                    source = parameter_envelopes[parameter_key]
                    quantity(value, String(source["si_unit"]), String(source["dimensionality"]))
                else
                    evidence = get(binding, "kind", nothing) == "affine" ? binding["intercept"] : binding["value"]
                    quantity(value, String(evidence["si_unit"]), String(evidence["dimensionality"]))
                end
            else
                plain(value)
            end
            push!(rows, Dict{String,Any}("path" => path, "field" => id, "value" => encoded))
        end
    end
    sort!(rows; by = row -> (Tuple(String.(row["path"])), String(row["field"])))
    return resolved, rows
end

function structured_resolved_rows(plan, source_rows)
    expected = Set{String}()
    units = Dict{String,String}()
    for leaf in plan["physical_leaves"], field in leaf["fields"]
        key = structured_field_key(leaf["path"], field["id"])
        push!(expected, key); units[key] = String(field["unit"])
    end
    resolved = Dict{String,Any}(); rows = Dict{String,Any}[]
    previous = nothing
    for raw in source_rows
        row = plain(raw); key = structured_field_key(row["path"], row["field"])
        exact_keys(row, ("path", "field", "value")) || fail("execution", "compiler_invariant", "compile", "compile", "resolved field record is malformed")
        key in expected || fail("execution", "compiler_invariant", "compile", "compile", "resolved point contains an unknown physical field")
        haskey(resolved, key) && fail("execution", "compiler_invariant", "compile", "compile", "resolved point duplicates a physical field")
        ordering = (Tuple(String.(row["path"])), String(row["field"]))
        previous === nothing || previous < ordering || fail("execution", "compiler_invariant", "compile", "compile", "resolved fields are not strictly sorted")
        previous = ordering
        value = structured_value(row["value"])
        (units[key] == "rlgc") == (value isa AbstractDict) || fail("execution", "compiler_invariant", "compile", "compile", "resolved point field has the wrong value type")
        resolved[key] = value; push!(rows, row)
    end
    Set(keys(resolved)) == expected || fail("execution", "compiler_invariant", "compile", "compile", "resolved point does not exactly cover physical fields")
    return resolved, rows
end

function structured_endpoint_map(plan)
    ground = String(plan["connectivity"]["canonical_ground"])
    result = Dict{String,String}()
    for row in plan["connectivity"]["physical_endpoints"]
        key = structured_endpoint_key(row["path"], row["pin"])
        haskey(result, key) && fail("execution", "compiler_invariant", "compile", "compile", "physical endpoint is duplicated")
        result[key] = String(row["net"])
    end
    return result, ground
end

function structured_node_basis(plan, resolved)
    nodes = String[]; seen = Set{String}(); ground = String(plan["connectivity"]["canonical_ground"])
    for row in plan["connectivity"]["node_coordinates"]
        final_net, compiler_id = String(row["final_net"]), String(row["compiler_node_id"])
        final_net == compiler_id || fail("execution", "compiler_invariant", "compile", "compile", "compiler node ID differs from its canonical final net")
        compiler_id != ground || fail("execution", "compiler_invariant", "compile", "compile", "canonical ground appears in the nodal basis")
        compiler_id in seen && fail("execution", "compiler_invariant", "compile", "compile", "compiler node ID is duplicated")
        push!(seen, compiler_id); push!(nodes, compiler_id)
    end
    for leaf in plan["physical_leaves"]
        String(leaf["model"]) == "transmission_line" || continue
        rlgc = resolved[structured_field_key(leaf["path"], "rlgc")]
        conductors = String.(rlgc["conductors"]); sections = Int(leaf["model_metadata"]["n_sections"])
        for station in 1:(sections - 1), conductor in conductors
            id = "internal-" * sha256_hex(canonical_bytes(Dict(
                "schema" => "scnsim.line_station", "schema_version" => 1,
                "component_path" => structured_path(leaf["path"]), "station" => station, "conductor" => conductor,
            )))
            id in seen && fail("execution", "compiler_invariant", "compile", "compile", "transmission-line station node collides")
            push!(seen, id); push!(nodes, id)
        end
    end
    isempty(nodes) && fail("execution", "compiler_invariant", "compile", "compile", "sealed Plan has no non-reference node")
    return nodes
end

function structured_endpoint_incidences(leaf, endpoint_to_node, node_index, ground)
    pins = String.(leaf["pin_order"])
    length(pins) == 2 || fail("execution", "compiler_invariant", "compile", "compile", "primitive must have two ordered pins")
    path = structured_path(leaf["path"]); positive = zeros(Float64, length(node_index)); negative = zeros(Float64, length(node_index))
    left, right = structured_endpoint_key(path, pins[1]), structured_endpoint_key(path, pins[2])
    haskey(endpoint_to_node, left) && haskey(endpoint_to_node, right) || fail("execution", "compiler_invariant", "compile", "compile", "primitive terminal is unbound")
    endpoint_to_node[left] != ground && (positive[node_index[endpoint_to_node[left]]] = 1.0)
    endpoint_to_node[right] != ground && (negative[node_index[endpoint_to_node[right]]] = 1.0)
    return positive, negative
end

function structured_line_station_node(leaf, station::Int, conductor::String, endpoint_to_node, ground)::String
    sections = Int(leaf["model_metadata"]["n_sections"]); path = structured_path(leaf["path"])
    if station == 0 || station == sections
        pin = (station == 0 ? "head." : "tail.") * conductor
        key = structured_endpoint_key(path, pin)
        haskey(endpoint_to_node, key) || fail("execution", "compiler_invariant", "compile", "compile", "transmission-line endpoint is unbound")
        return endpoint_to_node[key]
    end
    return "internal-" * sha256_hex(canonical_bytes(Dict(
        "schema" => "scnsim.line_station", "schema_version" => 1,
        "component_path" => path, "station" => station, "conductor" => conductor,
    )))
end

function structured_line_incidence(leaf, station, conductors, endpoint_to_node, node_index, ground)
    B = zeros(Float64, length(node_index), length(conductors))
    for (column, conductor) in enumerate(conductors)
        node = structured_line_station_node(leaf, station, conductor, endpoint_to_node, ground)
        node != ground && (B[node_index[node], column] = 1.0)
    end
    return B
end

function structured_compile(plan_value, values::Dict{String,Any}; context_kind::String = "compile",
        authorized::Set{String} = Set{String}(), extrapolation_evidence::Union{Nothing,Vector{Any}} = nothing,
        authorization_source::String = "none", emit_audit::Bool = false, resolved_rows = nothing,
        fail_unauthorized::Bool = true)
    plan = plain(plan_value)
    get(plan, "schema", nothing) == "scnsim.plan" && get(plan, "schema_version", nothing) == 2 ||
        fail("execution", "compiler_invariant", "compile", "compile", "structured Plan schema/version is invalid")
    exact_keys(plan, ("schema", "schema_version", "plan_id", "scope_hierarchy", "occurrences", "physical_leaves", "connectivity", "parameter_closure")) ||
        fail("execution", "compiler_invariant", "compile", "compile", "structured Plan fields are invalid")
    resolved, evidence_rows = resolved_rows === nothing ?
        structured_resolve_fields(plan, values; context_kind = context_kind, authorized = authorized,
            extrapolation_evidence = extrapolation_evidence, authorization_source = authorization_source,
            fail_unauthorized = fail_unauthorized) : structured_resolved_rows(plan, resolved_rows)
    nodes = structured_node_basis(plan, resolved); node_index = Dict(node => index for (index, node) in enumerate(nodes))
    endpoint_to_node, ground = structured_endpoint_map(plan)
    valid_nets = Set(vcat(nodes, [ground]))
    all(net in valid_nets for net in Base.values(endpoint_to_node)) || fail("execution", "compiler_invariant", "compile", "compile", "physical endpoint targets an unknown canonical net")
    capacitors = Tuple{Vector{Float64},Float64}[]; resistors = Tuple{Vector{Float64},Float64}[]; inductors = ExpandedInductor[]
    capacitance_blocks = Tuple{Matrix{Float64},Matrix{Float64}}[]; conductance_blocks = Tuple{Matrix{Float64},Matrix{Float64}}[]
    series_rl = SeriesRLBlock[]; rows = Dict{String,Any}[]
    for leaf in plan["physical_leaves"]
        path = structured_path(leaf["path"]); model = String(leaf["model"]); pins = String.(leaf["pin_order"])
        field(id) = resolved[structured_field_key(path, id)]
        if model == "transmission_line"
            rlgc = field("rlgc"); length_value = field("length")
            length_value isa Float64 && isfinite(length_value) && length_value > 0.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "transmission-line length must be finite and positive")
            conductors = String.(rlgc["conductors"]); sections = Int(leaf["model_metadata"]["n_sections"])
            length(pins) == 2 * length(conductors) && sections >= 1 || fail("execution", "compiler_invariant", "compile", "compile", "transmission-line declaration is malformed")
            pins == vcat(["head." * conductor for conductor in conductors], ["tail." * conductor for conductor in conductors]) || fail("execution", "compiler_invariant", "compile", "compile", "transmission-line pin order disagrees with RLGC conductor order")
            dx = length_value / sections
            R = rlgc_matrix(rlgc["resistance_per_length"], "R") .* dx
            L = rlgc_matrix(rlgc["inductance_per_length"], "L") .* dx
            G = rlgc_matrix(rlgc["conductance_per_length"], "G") .* dx
            C = rlgc_matrix(rlgc["capacitance_per_length"], "C") .* dx
            size(R, 1) == length(conductors) && size(L) == size(R) && size(G) == size(R) && size(C) == size(R) || fail("execution", "compiler_invariant", "compile", "compile", "RLGC matrix dimension disagrees with line conductors")
            try
                cholesky(Symmetric(L); check = true); cholesky(Symmetric(C); check = true)
                minimum(eigvals(Symmetric(R))) >= 0.0 && minimum(eigvals(Symmetric(G))) >= 0.0 || error("non-PSD")
            catch
                fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "RLGC physical matrix validation failed")
            end
            if emit_audit
                stations = Dict{String,Any}[]
                for station in 0:sections, conductor in conductors
                    total_factor = station == 0 || station == sections ? 0.5 : 1.0
                    push!(stations, Dict{String,Any}(
                        "station" => station, "conductor" => conductor,
                        "compiled_node_id" => structured_line_station_node(leaf, station, conductor, endpoint_to_node, ground),
                        "attachment" => station == 0 ? "head" : station == sections ? "tail" : "interior",
                        "left_half_shunt" => station == 0 ? nothing : Dict("section" => station, "end" => "right"),
                        "right_half_shunt" => station == sections ? nothing : Dict("section" => station + 1, "end" => "left"),
                        "compiled_capacitance_total" => quantity_matrix(C .* total_factor, "farad", "capacitance"),
                        "compiled_conductance_total" => quantity_matrix(G .* total_factor, "siemens", "conductance"),
                    ))
                end
                push!(rows, Dict{String,Any}(
                    "kind" => "transmission_line_audit", "component_path" => path,
                    "conductors" => conductors, "reference_conductor" => String(rlgc["reference_conductor"]),
                    "n_sections" => sections, "length" => quantity(length_value, "meter", "length"),
                    "dx" => quantity(dx, "meter", "length"), "orientation" => String(rlgc["orientation"]),
                    "rlgc_source" => plain(rlgc["source"]), "stations" => stations,
                ))
            end
            for section in 1:sections
                left = structured_line_incidence(leaf, section - 1, conductors, endpoint_to_node, node_index, ground)
                right = structured_line_incidence(leaf, section, conductors, endpoint_to_node, node_index, ground)
                push!(series_rl, SeriesRLBlock(structured_path_key(path) * "\u001esection-" * string(section), left - right, R, L))
                for station in (section - 1, section)
                    Bshunt = structured_line_incidence(leaf, station, conductors, endpoint_to_node, node_index, ground)
                    push!(capacitance_blocks, (Bshunt, C ./ 2.0)); push!(conductance_blocks, (Bshunt, G ./ 2.0))
                end
                for (row_index, conductor_a) in enumerate(conductors), (column_index, conductor_b) in enumerate(conductors)
                    for (label, matrix, unit, dimensionality) in (("series_resistance", R, "ohm", "resistance"), ("series_inductance", L, "henry", "inductance"))
                        value = matrix[row_index, column_index]
                        push!(rows, Dict{String,Any}("component_path" => path, "kind" => label, "section" => section,
                            "row_conductor" => conductor_a, "column_conductor" => conductor_b,
                            "value" => quantity(value, unit, dimensionality), "omitted_as_zero" => value == 0.0))
                    end
                    for (station, end_label) in ((section - 1, "left"), (section, "right"))
                        Bstation = structured_line_incidence(leaf, station, conductors, endpoint_to_node, node_index, ground)
                        for (label, matrix, unit, dimensionality) in (("shunt_conductance_half", G / 2.0, "siemens", "conductance"), ("shunt_capacitance_half", C / 2.0, "farad", "capacitance"))
                            value = matrix[row_index, column_index]
                            push!(rows, Dict{String,Any}("component_path" => path, "kind" => label, "section" => section,
                                "station" => station, "end" => end_label, "row_conductor" => conductor_a, "column_conductor" => conductor_b,
                                "row_incidence_f64" => f64_hex.(Bstation[:, row_index]), "column_incidence_f64" => f64_hex.(Bstation[:, column_index]),
                                "physical_positive_incidence_f64" => f64_hex.(Bstation[:, row_index]),
                                "physical_negative_incidence_f64" => f64_hex.(row_index == column_index ? zeros(Float64, length(nodes)) : Bstation[:, column_index]),
                                "value" => quantity(value, unit, dimensionality), "omitted_as_zero" => value == 0.0))
                        end
                    end
                end
            end
            continue
        end
        positive, negative = structured_endpoint_incidences(leaf, endpoint_to_node, node_index, ground); b = positive - negative
        function record!(kind, value, unit, dimensionality, incidence = b; omitted_as_zero::Bool = false)
            push!(rows, Dict{String,Any}("component_path" => path, "kind" => kind,
                "terminal_1_to_terminal_2" => pins, "incidence_f64" => f64_hex.(incidence),
                "physical_positive_incidence_f64" => f64_hex.(positive), "physical_negative_incidence_f64" => f64_hex.(negative),
                "value" => quantity(value, unit, dimensionality), "omitted_as_zero" => omitted_as_zero))
        end
        if model == "capacitor" || model == "resistor"
            id = model == "capacitor" ? "capacitance" : "resistance"; value = field(id)
            value isa Float64 && isfinite(value) && value > 0.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "primitive R/C value must be finite and strictly positive")
            model == "capacitor" ? push!(capacitors, (b, value)) : push!(resistors, (b, value))
            record!(model, value, model == "capacitor" ? "farad" : "ohm", model == "capacitor" ? "capacitance" : "resistance")
        elseif model == "josephson_junction"
            lj, cj = field("josephson_inductance"), field("junction_capacitance")
            lj isa Float64 && isfinite(lj) && lj > 0.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "L_J0 must be finite and strictly positive")
            cj isa Float64 && isfinite(cj) && cj >= 0.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "Cj must be finite and nonnegative")
            branch = only(leaf["oriented_branches"])
            branch_b = positive - negative
            push!(inductors, ExpandedInductor(structured_branch_key(Dict("path" => path, "branch_id" => branch["id"])), branch_b, lj))
            record!("josephson_inductance", lj, "henry", "inductance", branch_b); rows[end]["branch_id"] = String(branch["id"])
            if cj == 0.0; record!("junction_capacitance", cj, "farad", "capacitance"; omitted_as_zero = true)
            else; push!(capacitors, (b, cj)); record!("junction_capacitance", cj, "farad", "capacitance"); end
        elseif model == "inductor"
            for branch in leaf["oriented_branches"]
                value = field(String(branch["value_field"])); value isa Float64 && isfinite(value) && value > 0.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "inductance must be finite and strictly positive")
                branch_positive = zeros(Float64, length(nodes)); branch_negative = zeros(Float64, length(nodes))
                for (target, pin) in ((branch_positive, branch["positive_pin"]), (branch_negative, branch["negative_pin"]))
                    net = endpoint_to_node[structured_endpoint_key(path, pin)]; net != ground && (target[node_index[net]] = 1.0)
                end
                incidence = branch_positive - branch_negative
                push!(inductors, ExpandedInductor(structured_branch_key(Dict("path" => path, "branch_id" => branch["id"])), incidence, value))
                record!("inductor", value, "henry", "inductance", incidence); rows[end]["branch_id"] = String(branch["id"])
            end
        else
            fail("execution", "compiler_invariant", "compile", "compile", "physical leaf model is outside the sealed native vocabulary")
        end
    end
    n = length(nodes); C = zeros(Float64, n, n); G = zeros(Float64, n, n)
    for (b, value) in capacitors; C .+= value .* (b * transpose(b)); end
    for (b, value) in resistors; G .+= (1.0 / value) .* (b * transpose(b)); end
    for (Bblock, matrix) in capacitance_blocks; C .+= Bblock * matrix * transpose(Bblock); end
    for (Bblock, matrix) in conductance_blocks; G .+= Bblock * matrix * transpose(Bblock); end
    locations = Dict(item.id => index for (index, item) in enumerate(inductors)); edges = Tuple{Int,Int,Float64}[]; pairs = Set{Tuple{Int,Int}}()
    for coupling in plan["connectivity"]["couplings"]
        left_ref, right_ref = coupling["inductor_a"], coupling["inductor_b"]
        left, right = structured_branch_key(left_ref), structured_branch_key(right_ref)
        haskey(locations, left) && haskey(locations, right) || fail("execution", "compiler_invariant", "compile", "compile", "coupling references an unknown physical branch")
        i, j = locations[left], locations[right]; i != j || fail("execution", "compiler_invariant", "compile", "compile", "coupling cannot self-couple")
        pair = minmax(i, j); pair in pairs && fail("execution", "compiler_invariant", "compile", "compile", "coupling duplicates a physical branch pair"); push!(pairs, pair)
        coefficient = quantity_value(coupling["coupling_coefficient"])
        isfinite(coefficient) && abs(coefficient) < 1.0 || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "mutual coupling coefficient must satisfy abs(k) < 1")
        mutual = coefficient * sqrt(inductors[i].value * inductors[j].value); push!(edges, (i, j, mutual))
        push!(rows, Dict{String,Any}("kind" => "mutual_inductance", "coupling_id" => String(coupling["id"]),
            "branch_a" => left_ref, "branch_b" => right_ref, "coupling_coefficient" => quantity(coefficient, "dimensionless", "dimensionless"),
            "derived_mutual_inductance" => quantity(mutual, "henry", "inductance"), "omitted_as_zero" => mutual == 0.0))
    end
    K = zeros(Float64, n, n); neighbors = [Int[] for _ in inductors]
    for (i, j, _) in edges; push!(neighbors[i], j); push!(neighbors[j], i); end
    visited = falses(length(inductors))
    for start in eachindex(inductors)
        visited[start] && continue
        group = Int[]; pending = [start]; visited[start] = true
        while !isempty(pending)
            index = pop!(pending); push!(group, index)
            for neighbor in neighbors[index]; visited[neighbor] || (visited[neighbor] = true; push!(pending, neighbor)); end
        end
        if length(group) == 1
            branch = inductors[only(group)]; K .+= (1.0 / branch.value) .* (branch.incidence * transpose(branch.incidence)); continue
        end
        local_index = Dict(member => index for (index, member) in enumerate(group)); L = zeros(Float64, length(group), length(group))
        for (index, member) in enumerate(group); L[index, index] = inductors[member].value; end
        for (i, j, mutual) in edges
            haskey(local_index, i) && haskey(local_index, j) || continue
            left, right = local_index[i], local_index[j]; L[left, right] = mutual; L[right, left] = mutual
        end
        factor = try; cholesky(Symmetric(L); check = true); catch; fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "complete reciprocal inductance matrix is not positive definite"); end
        B = hcat((inductors[index].incidence for index in group)...); reciprocal = factor \ transpose(B)
        residual = backward_residual(L, reciprocal, transpose(B))
        isfinite(residual) && residual <= tau(length(group)) || fail("execution", "invalid_candidate_physical_parameter", "physical_validation", context_kind, "reciprocal inductance solve exceeded normalized residual contract")
        K .+= B * reciprocal
    end
    ports = plan["connectivity"]["ports"]; port_ids = String[]; B = zeros(Float64, n, length(ports)); R = zeros(Float64, length(ports), length(ports)); M = ones(Float64, length(ports))
    for (column, port) in enumerate(ports)
        id, net, role = String(port["id"]), String(port["net"]), String(port["role"])
        id in port_ids && fail("execution", "compiler_invariant", "compile", "compile", "duplicate Port ID")
        role in ("terminated", "nonloading_probe") || fail("validation", "port_realizability", "compile", "compile", "Port role is not realizable")
        haskey(node_index, net) || fail("execution", "compiler_invariant", "compile", "compile", "Port net is absent from compiled basis")
        impedance = quantity_value(port["reference_impedance"]); isfinite(impedance) && impedance > 0.0 || fail("execution", "compiler_invariant", "compile", "compile", "Port impedance must be finite and positive")
        push!(port_ids, id); B[node_index[net], column] = 1.0; R[column, column] = impedance
    end
    compiled = CompiledPrimitive(nodes, C, K, G, Any[series_rl...], rows, port_ids, B, R, M)
    return compiled, evidence_rows
end
