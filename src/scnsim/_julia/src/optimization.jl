# Included into the single SCNSimBackend module.
# Direct Optimization orchestration, continuation, and ledger evidence.

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

function candidate_parameter_set(request, values::Dict{String,Any})
    bindings = Any[]
    for binding in request_parameter_set(request)["bindings"]
        reference = binding["parameter"]
        key = ref_key(reference)
        haskey(values, key) || fail("execution", "compiler_invariant", "optimization", "optimization_candidate", "candidate is missing a request parameter")
        original = binding["value"]
        encoded = values[key] isa Float64 ? quantity(values[key], String(original["si_unit"]), String(original["dimensionality"])) : plain(values[key])
        push!(bindings, Dict{String,Any}(
            "parameter" => reference,
            "value" => encoded,
        ))
    end
    return Dict{String,Any}("type" => "parameter_set_v2", "bindings" => bindings, "allow_extrapolation" => Any[])
end

function parameter_values_for_z(request, base::Dict{String,Any}, z::AbstractVector{Float64})
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

function baseline_z(request, values::Dict{String,Any})
    variables = request["spec"]["variables"]
    for variable in variables
        haskey(values, ref_key(variable["parameter"])) ||
            fail("validation", "invalid_optimization_spec", "baseline", "optimization_candidate", "active variable lacks a baseline binding")
    end
    encoded = get(request["spec"]["optimizer"], "baseline_optimizer_coordinates_f64", nothing)
    encoded isa AbstractVector ||
        fail("validation", "invalid_optimization_spec", "baseline", "optimization_candidate", "sealed baseline optimizer coordinates are absent")
    coordinates = Float64[f64_from_hex(value) for value in encoded]
    length(coordinates) == length(variables) ||
        fail("validation", "invalid_optimization_spec", "baseline", "optimization_candidate", "sealed baseline optimizer coordinate count mismatches variables")
    all(value -> isfinite(value) && 0.0 <= value <= 1.0, coordinates) ||
        fail("validation", "invalid_optimization_spec", "baseline", "optimization_candidate", "sealed baseline optimizer coordinates leave the unit box")
    return coordinates
end

selector_dependency_key(selector) = canonical_json(Dict(
    "type" => selector["type"],
    "spec" => selector["spec"],
    "view" => selector["view"],
))

function root_selector_key(selector)
    kind = get(selector, "type", nothing)
    kind in ("diagonal_root_projection", "residue_diagonal_root_projection", "hybridized_pole_projection", "transfer_zero_projection") ||
        fail("capability", "scaffold_unavailable", "optimization", "optimization_candidate", "root continuation requires a root-like selector")
    return selector_dependency_key(selector)
end

function selector_leaves(selector)::Vector{Any}
    if get(selector, "type", nothing) == "quantity_sum"
        terms = get(selector, "terms", nothing)
        terms isa AbstractVector && !isempty(terms) ||
            fail("validation", "invalid_optimization_spec", "quantity_sum", "optimization_candidate", "QuantitySum requires one or more terms")
        return Any[term for term in terms]
    end
    return Any[selector]
end

optimization_leaf_locator(objective, term_ordinal::Int) = Dict{String,Any}(
    "objective_id" => objective["id"],
    "term_ordinal" => term_ordinal,
)

function optimization_leaf_catalog(request)
    leaves = Any[]
    for objective in request["spec"]["objectives"]
        for (term_ordinal, selector) in enumerate(selector_leaves(objective["quantity"]))
            push!(leaves, Dict{String,Any}(
                "locator" => optimization_leaf_locator(objective, term_ordinal),
                "selector" => selector,
            ))
        end
    end
    return leaves
end

optimization_candidate_position(ordinal::Int, generation::Int, column) = Dict{String,Any}(
    "evaluation_ordinal" => ordinal,
    "origin" => generation == 0 ? "baseline" : "population",
    "generation" => generation,
    "population_column" => column,
)

function optimization_dependency(selector; kind::String = "quantity")
    view_sha = sha256_hex(canonical_bytes(selector["view"]))
    dependency_sha = kind == "view" ? view_sha : sha256_hex(canonical_bytes(Dict(
        "type" => selector["type"],
        "spec" => selector["spec"],
        "view" => selector["view"],
    )))
    return Dict{String,Any}(
        "kind" => kind,
        "view_sha256" => view_sha,
        "dependency_sha256" => dependency_sha,
    )
end

function optimization_context(phase::String, candidate, owner, affected;
        dependency = nothing)
    context = Dict{String,Any}(
        "schema" => "scnsim.optimization_failure_context",
        "schema_version" => 1,
        "phase" => phase,
        "candidate" => candidate,
        "owner" => owner,
        "affected_leaves" => affected,
    )
    dependency === nothing || (context["dependency"] = dependency)
    return context
end

function with_optimization_context(failure::BackendFailure, context)
    failure.optimization_context === nothing || return failure
    return BackendFailure(
        failure.category, failure.kind, failure.stage, failure.context_kind,
        failure.message, context,
    )
end

function optimization_backend_failure(error, stage::String)
    return error isa BackendFailure ? error : BackendFailure(
        "execution", "compiler_invariant", stage, "optimization_candidate",
        sprint(showerror, error),
    )
end

is_projection_only_optimization_failure(failure::BackendFailure) =
    failure.kind == "invalid_optimization_spec" &&
    failure.stage in ("selector", "quantity_sum") &&
    failure.context_kind == "optimization_candidate"

function optimization_all_leaves(request)
    return Any[item["locator"] for item in optimization_leaf_catalog(request)]
end

function optimization_view_leaves(request, declaration)
    key = canonical_json(declaration)
    return Any[item["locator"] for item in optimization_leaf_catalog(request)
        if canonical_json(item["selector"]["view"]) == key]
end

function optimization_quantity_leaves(request, selector, trigger)
    key = selector_dependency_key(selector)
    found_trigger = false
    affected = Any[]
    for item in optimization_leaf_catalog(request)
        item["locator"] == trigger && (found_trigger = true)
        found_trigger && selector_dependency_key(item["selector"]) == key &&
            push!(affected, item["locator"])
    end
    return affected
end

function optimization_root_leaves(request, root_selector; trigger = nothing)
    key = root_selector_key(root_selector)
    found_trigger = trigger === nothing
    affected = Any[]
    for item in optimization_leaf_catalog(request)
        item["locator"] == trigger && (found_trigger = true)
        roots = root_selector_specs(item["selector"])
        found_trigger && haskey(roots, key) && push!(affected, item["locator"])
    end
    return affected
end

function rebase_optimization_failure(value, candidate)
    copied = deepcopy(value)
    function visit!(item)
        if item isa AbstractDict
            if get(item, "schema", nothing) == "scnsim.optimization_failure_context"
                item["candidate"] = candidate
            else
                for nested in values(item)
                    visit!(nested)
                end
            end
        elseif item isa AbstractVector
            for nested in item
                visit!(nested)
            end
        end
    end
    visit!(copied)
    return copied
end

function root_selector_specs(selector, found::Dict{String,Any} = Dict{String,Any}())
    selector_type = get(selector, "type", nothing)
    if selector_type in ("diagonal_root_projection", "hybridized_pole_projection", "transfer_zero_projection")
        found[root_selector_key(selector)] = selector
    elseif selector_type == "residue_coupling_projection"
        # A residue objective has two anchored branch locators.  They are
        # private continuation dependencies, not a second public selector.
        for branch in (selector["spec"]["branch_a"], selector["spec"]["branch_b"])
            branch_selector = residue_branch_selector(branch, selector["view"])
            found[root_selector_key(branch_selector)] = branch_selector
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

function residue_branch_selector(branch, view_declaration = nothing)
    kind = String(branch["type"])
    selector = if kind == "diagonal_root"
        Dict{String,Any}("type" => "residue_diagonal_root_projection", "spec" => branch, "projection" => "frequency")
    elseif kind == "hybridized_pole"
        Dict{String,Any}("type" => "hybridized_pole_projection", "spec" => branch, "projection" => "frequency")
    else
        fail("validation", "invalid_optimization_spec", "residue_branch", "optimization_candidate", "residue branch has an unsupported locator")
    end
    view_declaration === nothing || (selector["view"] = view_declaration)
    return selector
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

function realize_candidate_view(plan, request, raw::CompiledPrimitive, view_declaration)
    return realized_ref_lineage(
        raw,
        declarative_lineage(plan, request, raw; view_declaration = view_declaration),
    )
end

function selector_root_with_continuation(plan, request, baseline_values, candidate_values, baseline_root::ComplexF64, selector;
        context_kind::String = "optimization_candidate",
        candidate_view::Union{Nothing,RealizedView} = nothing)
    function values_at(t::Float64)
        t == 0.0 && return copy(baseline_values)
        t == 1.0 && return copy(candidate_values)
        result = Dict{String,Any}()
        for key in keys(baseline_values)
            left, right = baseline_values[key], candidate_values[key]
            if left isa Float64 && right isa Float64
                result[key] = left + t * (right - left)
            else
                canonical_json(left) == canonical_json(right) || fail("execution", "compiler_invariant", "root_continuation", context_kind, "root continuation cannot interpolate RLGC parameters")
                result[key] = left
            end
        end
        return result
    end
    function advance(left_t::Float64, left_root::ComplexF64, right_t::Float64, depth::Int)::ComplexF64
        try
            view = if right_t == 1.0 && candidate_view !== nothing
                candidate_view::RealizedView
            else
                raw = compile_primitive(plan, values_at(right_t); context_kind = context_kind,
                    authorized = context_kind == "optimization_candidate" ? optimization_authorizations(request) : parameter_set_authorizations(request),
                    authorization_source = context_kind == "optimization_candidate" ? "optimization_spec" : "parameter_set")
                declaration = get(selector, "view", request["view"])
                realize_candidate_view(plan, request, raw, declaration)[2]
            end
            compiled = view.compiled
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

function root_selector_value(selector, plan, request, baseline_values, values, baseline_roots, roots, compiled::CompiledPrimitive, view::RealizedView, candidate, locator)::Float64
    selector_type = get(selector, "type", nothing)
    if selector_type == "quantity_sum"
        terms = selector["terms"]
        isempty(terms) && fail("validation", "invalid_optimization_spec", "quantity_sum", "optimization_candidate", "QuantitySum requires one or more terms")
        # Declaration order is semantic authority.  Do not let a reduction
        # implementation choose regrouping or summation order.
        public_unit = selector_public_unit(terms[1])
        total = root_selector_value(terms[1], plan, request, baseline_values, values, baseline_roots, roots, compiled, view, candidate, locator)
        for term in terms[2:end]
            value = root_selector_value(term, plan, request, baseline_values, values, baseline_roots, roots, compiled, view, candidate, locator)
            total = total + selector_value_in_unit(value, selector_public_unit(term), public_unit)
        end
        return total
    end
    projection = selector["projection"]
    if selector_type == "diagonal_root_projection"
        key = root_selector_key(selector)
        root = get!(roots, key) do
            same_parameter_values(baseline_values, values) ? baseline_roots[key] :
                selector_root_with_continuation(plan, request, baseline_values, values, baseline_roots[key], selector; candidate_view = view)
        end
        projection == "frequency" && return real(root) / (2.0 * pi)
        projection == "linewidth" && return -2.0 * imag(root) / (2.0 * pi)
    elseif selector_type == "hybridized_pole_projection"
        key = root_selector_key(selector)
        root = get!(roots, key) do
            same_parameter_values(baseline_values, values) ? baseline_roots[key] :
                selector_root_with_continuation(plan, request, baseline_values, values, baseline_roots[key], selector; candidate_view = view)
        end
        projection == "frequency" && return real(root) / (2.0 * pi)
        projection == "linewidth" && return -2.0 * imag(root) / (2.0 * pi)
    elseif selector_type == "transfer_zero_projection"
        spec = selector["spec"]; input = findfirst(==(String(spec["input_coordinate"])), view.terminal); output = findfirst(==(String(spec["output_coordinate"])), view.terminal)
        (input === nothing || output === nothing) && fail("validation", "invalid_optimization_spec", "selector", "optimization_candidate", "transfer-zero selector coordinate is absent")
        key = root_selector_key(selector)
        zero = get!(roots, key) do
            same_parameter_values(baseline_values, values) ? baseline_roots[key] :
                selector_root_with_continuation(plan, request, baseline_values, values, baseline_roots[key], selector; candidate_view = view)
        end
        projection == "frequency" && return real(zero) / (2.0 * pi)
    elseif selector_type == "response_element_projection"
        spec = selector["spec"]; input = findfirst(==(String(spec["input_coordinate"])), view.terminal); output = findfirst(==(String(spec["output_coordinate"])), view.terminal)
        (input === nothing || output === nothing) && fail("validation", "invalid_optimization_spec", "selector", "optimization_candidate", "response selector coordinate is absent")
        key = selector_dependency_key(selector)
        value = get!(roots, key) do
            try
                transfer_family_value(view, String(spec["family"]), output::Int, input::Int, complex(2.0 * pi * quantity_value(spec["frequency"])))[1]
            catch error
                error isa BackendFailure || rethrow()
                if !same_parameter_values(baseline_values, values) && error.kind == "direct_response_formation"
                    fail("execution", "numerical_resolution_unresolved", error.stage, "optimization_candidate", "candidate selected response is numerically unresolved")
                end
                rethrow()
            end
        end
        projection == "magnitude" && return abs(value)
        projection == "real" && return real(value)
        projection == "imag" && return imag(value)
    elseif selector_type == "residue_coupling_projection"
        branch_a_selector = residue_branch_selector(selector["spec"]["branch_a"], selector["view"])
        branch_b_selector = residue_branch_selector(selector["spec"]["branch_b"], selector["view"])
        key_a, key_b = root_selector_key(branch_a_selector), root_selector_key(branch_b_selector)
        function branch_root(branch_selector, key)
            try
                return get!(roots, key) do
                    same_parameter_values(baseline_values, values) ? baseline_roots[key] :
                        selector_root_with_continuation(plan, request, baseline_values, values, baseline_roots[key], branch_selector; candidate_view = view)
                end
            catch error
                failure = optimization_backend_failure(error, "quantity_evaluation")
                context = optimization_context(
                    "quantity_evaluation", candidate,
                    Dict("kind" => "leaf", "leaf" => locator),
                    optimization_root_leaves(request, branch_selector; trigger = locator);
                    dependency = optimization_dependency(branch_selector),
                )
                throw(with_optimization_context(failure, context))
            end
        end
        root_a = branch_root(branch_a_selector, key_a)
        root_b = branch_root(branch_b_selector, key_b)
        value = residue_normalized_coupling_value(compiled, view.terminal, selector["spec"];
            branch_a_root = root_a, branch_b_root = root_b)[1]
        projection == "magnitude" && return abs(value)
    end
    fail("validation", "invalid_optimization_spec", "selector", "optimization_candidate", "optimization selector projection is invalid")
end

function same_parameter_values(left::Dict{String,Any}, right::Dict{String,Any})::Bool
    Set(keys(left)) == Set(keys(right)) || return false
    return all((left[key] isa Float64 && right[key] isa Float64) ?
        f64_hex(left[key]) == f64_hex(right[key]) : canonical_json(left[key]) == canonical_json(right[key]) for key in keys(left))
end

function candidate_view_cache(plan, request, raw::CompiledPrimitive, candidate = nothing)
    views = Dict{String,Any}()
    for objective in request["spec"]["objectives"]
        for selector in selector_leaves(objective["quantity"])
            declaration = selector["view"]
            key = canonical_json(declaration)
            if !haskey(views, key)
                try
                    lineage, view = realize_candidate_view(plan, request, raw, declaration)
                    views[key] = Dict{String,Any}("lineage" => lineage, "view" => view)
                catch error
                    failure = optimization_backend_failure(error, "view_realization")
                    candidate === nothing && rethrow()
                    context = optimization_context(
                        "view_realization", candidate, Dict("kind" => "dependency"),
                        optimization_view_leaves(request, declaration);
                        dependency = optimization_dependency(Dict(
                            "type" => "view",
                            "spec" => Dict{String,Any}(),
                            "view" => declaration,
                        ); kind = "view"),
                    )
                    throw(with_optimization_context(failure, context))
                end
            end
        end
    end
    return views
end

function unevaluated_term(selector, ordinal::Int, request, failure::BackendFailure)
    return Dict{String,Any}(
        "term_ordinal" => ordinal,
        "selector" => selector,
        "status" => "not_evaluated",
        "failure" => failure_object(request, failure),
    )
end

function unevaluated_objective_components(request, failure::BackendFailure; first_index::Int = 1)
    components = Any[]
    for (objective_index, objective) in enumerate(request["spec"]["objectives"])
        objective_index < first_index && continue
        terms = selector_leaves(objective["quantity"])
        push!(components, Dict{String,Any}(
            "objective_id" => objective["id"],
            "quantity" => objective["quantity"],
            "status" => "not_evaluated",
            "terms" => Any[unevaluated_term(term, ordinal, request, failure) for (ordinal, term) in enumerate(terms)],
            "failure" => failure_object(request, failure),
        ))
    end
    return components
end

function plan_parameter_values(plan)::Dict{String,Any}
    result = Dict{String,Any}()
    for definition in plan["parameter_closure"]["definitions"]
        result[ref_key(definition)] = structured_value(definition["baseline"])
    end
    return result
end

function optimization_authorizations(request)::Set{String}
    refs = get(request["spec"], "allow_extrapolation", Any[])
    refs isa AbstractVector || fail("validation", "invalid_optimization_spec", "allow_extrapolation", "optimization_candidate", "optimization authorization collection is malformed")
    return Set(ref_key(reference) for reference in refs)
end

function consumer_target_key(target)::String
    item = plain(target)
    haskey(item, "path") && haskey(item, "field") && return structured_field_key(item["path"], item["field"])
    return ref_key(item)
end

function objective_outcome(plan, request, baseline_values, values, baseline_roots;
        extrapolation_evidence::Vector{Any} = Any[], prepared_raw = nothing,
        prepared_views = nothing, candidate)
    raw_compiled = if prepared_raw === nothing
        try
            compile_primitive(
                plan, values; context_kind = "optimization_candidate",
                authorized = optimization_authorizations(request), extrapolation_evidence = extrapolation_evidence,
                authorization_source = "optimization_spec",
            )
        catch error
            failure = optimization_backend_failure(error, "candidate_compile")
            throw(with_optimization_context(failure, optimization_context(
                "candidate_compile", candidate, Dict("kind" => "candidate"),
                optimization_all_leaves(request),
            )))
        end
    else
        prepared_raw
    end
    views = prepared_views === nothing ? candidate_view_cache(plan, request, raw_compiled, candidate) : prepared_views
    components = Any[]
    total = 0.0
    roots = Dict{String,ComplexF64}()
    for (objective_index, objective) in enumerate(request["spec"]["objectives"])
        selector = objective["quantity"]
        selector isa AbstractDict || fail("capability", "scaffold_unavailable", "optimization", "optimization_candidate", "optimization quantity must be a selector record")
        terms = selector_leaves(selector)
        public_unit = selector_public_unit(terms[1])
        term_records = Any[]
        value = 0.0
        failed = nothing
        for (term_index, term) in enumerate(terms)
            selected = nothing
            locator = optimization_leaf_locator(objective, term_index)
            try
                view_key = canonical_json(term["view"])
                haskey(views, view_key) ||
                    fail("execution", "compiler_invariant", "optimization", "optimization_candidate", "candidate View cache is incomplete")
                selected = views[view_key]
                view = selected["view"]::RealizedView
                term_value = root_selector_value(
                    term, plan, request, baseline_values, values, baseline_roots,
                    roots, view.compiled, view, candidate, locator,
                )
                value += selector_value_in_unit(term_value, selector_public_unit(term), public_unit)
                push!(term_records, Dict{String,Any}(
                    "term_ordinal" => term_index,
                    "selector" => term,
                    "status" => "success",
                    "ref_lineage" => selected["lineage"],
                    "value" => quantity(term_value, selector_public_unit(term), objective["target"]["dimensionality"]),
                ))
            catch error
                failure = optimization_backend_failure(error, "quantity_evaluation")
                projection_only = is_projection_only_optimization_failure(failure)
                dependency = projection_only ? nothing : optimization_dependency(term)
                affected = projection_only ? Any[locator] :
                    optimization_quantity_leaves(request, term, locator)
                context = optimization_context(
                    "quantity_evaluation", candidate,
                    Dict("kind" => "leaf", "leaf" => locator),
                    affected;
                    dependency = dependency,
                )
                failed = with_optimization_context(failure, context)
                is_candidate_failure(failed) || throw(failed)
                selected === nothing && throw(failed)
                push!(term_records, Dict{String,Any}(
                    "term_ordinal" => term_index,
                    "selector" => term,
                    "status" => "failure",
                    "ref_lineage" => selected["lineage"],
                    "failure" => failure_object(request, failed),
                ))
                for remaining_index in (term_index + 1):length(terms)
                    push!(term_records, unevaluated_term(terms[remaining_index], remaining_index, request, failed))
                end
                break
            end
        end
        if failed !== nothing
            failure = failed::BackendFailure
            push!(components, Dict{String,Any}(
                "objective_id" => objective["id"],
                "quantity" => selector,
                "status" => "failure",
                "terms" => term_records,
                "failure" => failure_object(request, failure),
            ))
            append!(components, unevaluated_objective_components(request, failure; first_index = objective_index + 1))
            sort!(extrapolation_evidence; by = row -> (ref_key(row["parameter"]), consumer_target_key(row["consumer_target"])))
            return Inf, components, extrapolation_evidence, failure
        end
        target = quantity_value(objective["target"])
        scale = quantity_value(objective["resolved_scale"])
        scale > 0.0 || fail("validation", "invalid_optimization_spec", "objective_scale", "optimization_candidate", "objective scale must be positive")
        residual = (value - target) / scale
        weighted = f64_from_hex(objective["weight_f64"]) * abs2(residual)
        if !isfinite(residual) || !isfinite(weighted)
            failure = BackendFailure("execution", "numerical_resolution_unresolved", "objective", "optimization_candidate", "candidate objective is non-finite")
            failure = with_optimization_context(failure, optimization_context(
                "objective_aggregation", candidate,
                Dict("kind" => "objective", "objective_id" => objective["id"]),
                Any[],
            ))
            push!(components, Dict{String,Any}(
                "objective_id" => objective["id"],
                "quantity" => selector,
                "status" => "failure",
                "terms" => term_records,
                "failure" => failure_object(request, failure),
            ))
            append!(components, unevaluated_objective_components(request, failure; first_index = objective_index + 1))
            sort!(extrapolation_evidence; by = row -> (ref_key(row["parameter"]), consumer_target_key(row["consumer_target"])))
            return Inf, components, extrapolation_evidence, failure
        end
        total += weighted
        push!(components, Dict{String,Any}(
            "objective_id" => objective["id"],
            "quantity" => selector,
            "status" => "success",
            "terms" => term_records,
            "value" => quantity(value, String(objective["target"]["si_unit"]), String(objective["target"]["dimensionality"])),
            "normalized_residual_f64" => f64_hex(residual),
            "weighted_cost_f64" => f64_hex(weighted),
        ))
    end
    if !isfinite(total)
        failure = BackendFailure("execution", "numerical_resolution_unresolved", "objective", "optimization_candidate", "candidate total cost is non-finite")
        failure = with_optimization_context(failure, optimization_context(
            "total_aggregation", candidate, Dict("kind" => "candidate"), Any[],
        ))
        sort!(extrapolation_evidence; by = row -> (ref_key(row["parameter"]), consumer_target_key(row["consumer_target"])))
        return Inf, components, extrapolation_evidence, failure
    end
    sort!(extrapolation_evidence; by = row -> (ref_key(row["parameter"]), consumer_target_key(row["consumer_target"])))
    return total, components, extrapolation_evidence, nothing
end

function failure_object(request, failure::BackendFailure)
    return Dict{String,Any}(
        "category" => failure.category,
        "kind" => failure.kind,
        "stage" => failure.stage,
        "message" => failure.message,
        "evidence" => failure_evidence(request, failure),
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

function write_generation_ledger(staging, request, request_sha, attempt_sha,
        checkpoint_sha, checkpoint_seal_sha, generation::Int, previous_sha,
        raw::Matrix{Float64}, transformed::Matrix{Float64}, candidates, certificate)
    ledger = Dict{String,Any}(
        "schema" => "scnsim.optimization_ledger",
        "schema_version" => 4,
        "request_sha256" => request_sha,
        "attempt_sha256" => attempt_sha,
        "algorithm_id" => "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v6",
        "baseline_checkpoint_sha256" => checkpoint_sha,
        "baseline_checkpoint_seal_sha256" => checkpoint_seal_sha,
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
            get(ledger, "schema_version", nothing) == 4 &&
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
            get(ledger, "schema_version", nothing) == 4 &&
            get(ledger, "request_sha256", nothing) == request_sha &&
            get(ledger, "attempt_sha256", nothing) == attempt_sha ||
            fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling ledger has incompatible identity")
        result[digest] = ledger
    end
    length(result) == length(declared) ||
        fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "sibling receipt ledger inventory has missing files")
    return result
end

function sibling_ledgers(staging::String, request_sha::String, attempt_sha::String,
        resume_sha::String, checkpoint_sha::String, checkpoint_seal_sha::String)
    attempt_root = dirname(staging)
    isdir(attempt_root) || fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "attempt directory is absent")
    discovered = Dict{String,Dict{String,Any}}()
    for entry in readdir(attempt_root; join = true)
        name = basename(entry)
        occursin(r"^(?!000000$)(?:[0-9]{6}|[1-9][0-9]{6,})$", name) || continue
        isdir(entry) || fail("evidence", "evidence_integrity", "optimization_replay", "artifact", "final attempt is not a directory")
        for (digest, ledger) in finalized_attempt_ledgers(entry, request_sha, attempt_sha)
            get(ledger, "algorithm_id", nothing) == "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v6" &&
                get(ledger, "baseline_checkpoint_sha256", nothing) == checkpoint_sha &&
                get(ledger, "baseline_checkpoint_seal_sha256", nothing) == checkpoint_seal_sha ||
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

function optimization_baseline_roots(plan, request, values;
        extrapolation_evidence::Vector{Any} = Any[])
    candidate = optimization_candidate_position(0, 0, nothing)
    raw = try
        compile_primitive(plan, values; context_kind = "optimization_candidate",
            authorized = optimization_authorizations(request),
            extrapolation_evidence = extrapolation_evidence,
            authorization_source = "optimization_spec")
    catch error
        failure = optimization_backend_failure(error, "candidate_compile")
        throw(with_optimization_context(failure, optimization_context(
            "candidate_compile", candidate, Dict("kind" => "candidate"),
            optimization_all_leaves(request),
        )))
    end
    views = candidate_view_cache(plan, request, raw, candidate)
    roots = Dict{String,ComplexF64}()
    for objective in request["spec"]["objectives"]
        selector = objective["quantity"]
        for (key, root_selector) in root_selector_specs(selector)
            if !haskey(roots, key)
                view = views[canonical_json(root_selector["view"])]["view"]::RealizedView
                try
                    roots[key] = selector_root_at(root_selector, view.compiled, view)
                catch error
                    failure = optimization_backend_failure(error, "baseline_root_anchor")
                    context = optimization_context(
                        "baseline_root_anchor", candidate,
                        Dict("kind" => "dependency"),
                        optimization_root_leaves(request, root_selector);
                        dependency = optimization_dependency(root_selector),
                    )
                    throw(with_optimization_context(failure, context))
                end
            end
        end
    end
    return roots, raw, views
end

function candidate_from_z(plan, request, baseline_values, baseline_roots, z::Vector{Float64}, latent, generation::Int, column::Int, ordinal::Int, cache::Dict{String,Any})
    candidate = optimization_candidate_position(ordinal, generation, column)
    values, parameters = try
        prepared_values = parameter_values_for_z(request, baseline_values, z)
        prepared_values, candidate_parameter_set(request, prepared_values)
    catch error
        failure = optimization_backend_failure(error, "candidate_prepare")
        throw(with_optimization_context(failure, optimization_context(
            "candidate_prepare", candidate, Dict("kind" => "candidate"),
            optimization_all_leaves(request),
        )))
    end
    key = canonical_json(parameters)
    if haskey(cache, key)
        cached = cache[key]
        outcome = rebase_optimization_failure(cached["outcome"], candidate)
        return candidate_record(request, ordinal, generation, column, z, latent, parameters, true, outcome;
            extrapolation_evidence = cached["extrapolation_evidence"]), cached["cost"]
    end
    extrapolation_evidence = Any[]
    try
        structured_resolve_fields(plan, values; context_kind = "optimization_candidate",
            authorized = optimization_authorizations(request), extrapolation_evidence = extrapolation_evidence,
            authorization_source = "optimization_spec", fail_unauthorized = false)
        if any(row -> row["authorization_source"] == "none", extrapolation_evidence)
            sort!(extrapolation_evidence; by = row -> (ref_key(row["parameter"]), consumer_target_key(row["consumer_target"])))
            fail("execution", "invalid_candidate_physical_parameter", "affine_support", "optimization_candidate",
                "affine input is outside its declared support")
        end
        empty!(extrapolation_evidence)
        cost, components, extrapolation_evidence, objective_failure = objective_outcome(plan, request, baseline_values, values, baseline_roots;
            extrapolation_evidence = extrapolation_evidence, candidate = candidate)
        outcome = if objective_failure === nothing
            Dict{String,Any}(
                "status" => "success",
                "cost_f64" => f64_hex(cost),
                "objective_components" => components,
            )
        else
            Dict{String,Any}(
                "status" => "failure",
                "penalty" => "positive_infinity",
                "failure" => failure_object(request, objective_failure::BackendFailure),
                "objective_components" => components,
            )
        end
        cache[key] = Dict("outcome" => outcome, "cost" => cost, "extrapolation_evidence" => extrapolation_evidence)
        return candidate_record(request, ordinal, generation, column, z, latent, parameters, false, outcome;
            extrapolation_evidence = extrapolation_evidence), cost
    catch error
        error isa BackendFailure || rethrow()
        contextual = error.optimization_context === nothing ? with_optimization_context(error, optimization_context(
            "candidate_prepare", candidate, Dict("kind" => "candidate"),
            optimization_all_leaves(request),
        )) : error
        is_candidate_failure(contextual) || throw(contextual)
        outcome = Dict{String,Any}(
            "status" => "failure",
            "penalty" => "positive_infinity",
            "failure" => failure_object(request, contextual),
            "objective_components" => unevaluated_objective_components(request, contextual),
        )
        sort!(extrapolation_evidence; by = row -> (ref_key(row["parameter"]), consumer_target_key(row["consumer_target"])))
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

function optimization_root_specs_ordered(request)
    rows = Tuple{String,Any}[]
    seen = Set{String}()
    function visit!(selector)
        kind = get(selector, "type", nothing)
        if kind in ("diagonal_root_projection", "hybridized_pole_projection", "transfer_zero_projection")
            key = root_selector_key(selector)
            if !(key in seen)
                push!(seen, key); push!(rows, (key, selector))
            end
        elseif kind == "residue_coupling_projection"
            for branch in (selector["spec"]["branch_a"], selector["spec"]["branch_b"])
                root = residue_branch_selector(branch, selector["view"])
                key = root_selector_key(root)
                if !(key in seen)
                    push!(seen, key); push!(rows, (key, root))
                end
            end
        elseif kind == "quantity_sum"
            for term in selector["terms"]
                visit!(term)
            end
        elseif kind == "response_element_projection"
            nothing
        else
            fail("capability", "scaffold_unavailable", "optimization", "optimization_candidate", "optimization selector is unsupported")
        end
    end
    for objective in request["spec"]["objectives"]
        visit!(objective["quantity"])
    end
    return rows
end

function optimization_checkpoint_document(request, request_sha::String, baseline, roots)
    root_rows = Any[]
    for (key, selector) in optimization_root_specs_ordered(request)
        root = roots[key]
        push!(root_rows, Dict{String,Any}(
            "dependency" => optimization_dependency(selector),
            "value" => Dict{String,Any}(
                "real_f64" => f64_hex(real(root)),
                "imag_f64" => f64_hex(imag(root)),
                "si_unit" => "radian / second",
                "dimensionality" => "inverse_time",
            ),
        ))
    end
    return Dict{String,Any}(
        "schema" => "scnsim.optimization_baseline_checkpoint",
        "schema_version" => 1,
        "request_sha256" => request_sha,
        "algorithm_id" => "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v6",
        "baseline" => baseline,
        "baseline_roots" => root_rows,
    )
end

function roots_from_checkpoint(request, checkpoint)
    exact_keys(checkpoint, ("schema", "schema_version", "request_sha256", "algorithm_id", "baseline", "baseline_roots")) ||
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "baseline checkpoint fields are invalid")
    get(checkpoint, "schema", nothing) == "scnsim.optimization_baseline_checkpoint" &&
        get(checkpoint, "schema_version", nothing) == 1 &&
        get(checkpoint, "algorithm_id", nothing) == "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v6" ||
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "baseline checkpoint version is unsupported")
    declared = checkpoint["baseline_roots"]
    specs = optimization_root_specs_ordered(request)
    declared isa AbstractVector && length(declared) == length(specs) ||
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "baseline checkpoint root inventory is incomplete")
    roots = Dict{String,ComplexF64}()
    for ((key, selector), row) in zip(specs, declared)
        exact_keys(row, ("dependency", "value")) && row["dependency"] == optimization_dependency(selector) ||
            fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "baseline checkpoint root dependency is out of order")
        value = row["value"]
        exact_keys(value, ("real_f64", "imag_f64", "si_unit", "dimensionality")) &&
            value["si_unit"] == "radian / second" && value["dimensionality"] == "inverse_time" ||
            fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "baseline checkpoint root value is malformed")
        roots[key] = ComplexF64(f64_from_hex(value["real_f64"]), f64_from_hex(value["imag_f64"]))
    end
    return roots
end

function baseline_primary_lineage(request, baseline)
    primary_key = canonical_json(request["view"])
    selected = nothing
    outcome = get(baseline, "outcome", nothing)
    components = outcome isa AbstractDict ? get(outcome, "objective_components", nothing) : nothing
    components isa AbstractVector ||
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "baseline checkpoint lacks objective components")
    for component in components
        terms = component isa AbstractDict ? get(component, "terms", nothing) : nothing
        terms isa AbstractVector ||
            fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "baseline checkpoint objective terms are malformed")
        for term in terms
            term isa AbstractDict && get(term, "status", nothing) == "success" ||
                fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "baseline checkpoint contains an unsuccessful objective term")
            selector = get(term, "selector", nothing)
            selector isa AbstractDict ||
                fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "baseline checkpoint objective selector is malformed")
            if canonical_json(get(selector, "view", nothing)) == primary_key
                lineage = get(term, "ref_lineage", nothing)
                lineage isa AbstractDict ||
                    fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "baseline checkpoint primary View lineage is absent")
                if selected === nothing
                    selected = lineage
                else
                    canonical_json(selected) == canonical_json(lineage) ||
                        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "baseline checkpoint primary View lineages disagree")
                end
            end
        end
    end
    selected === nothing &&
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "baseline checkpoint does not realize the primary View")
    return selected
end

function publish_baseline_checkpoint(request, request_sha::String, attempt_sha::String,
        staging::String, baseline, roots)
    checkpoint = optimization_checkpoint_document(request, request_sha, baseline, roots)
    bytes = canonical_bytes(checkpoint)
    path = joinpath(staging, "baseline-checkpoint.json")
    write_bytes(path, bytes)
    checkpoint_sha = sha256_hex(bytes)
    println(canonical_json(Dict{String,Any}(
        "schema" => "scnsim.optimization_checkpoint_ready",
        "schema_version" => 1,
        "event" => "baseline_checkpoint_ready",
        "request_sha256" => request_sha,
        "attempt_sha256" => attempt_sha,
        "checkpoint_sha256" => checkpoint_sha,
        "byte_length" => length(bytes),
    )))
    flush(stdout)
    seal_sha = read_checkpoint_committed(request_sha, attempt_sha, checkpoint_sha, nothing, "published")
    return checkpoint_sha, seal_sha
end

function load_baseline_checkpoint(request, request_sha::String, attempt_sha::String,
        request_directory::String, attempt)
    checkpoint_sha = get(attempt, "baseline_checkpoint_sha256", nothing)
    seal_sha = get(attempt, "baseline_checkpoint_seal_sha256", nothing)
    checkpoint_sha isa AbstractString && seal_sha isa AbstractString ||
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "attempt", "reused optimization attempt lacks checkpoint identities")
    directory = joinpath(request_directory, "baseline-checkpoint")
    isdir(directory) && !islink(directory) ||
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "published baseline checkpoint directory is absent")
    sort(readdir(directory)) == ["checkpoint.json", "seal.json", "source-attempt.json"] ||
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "published baseline checkpoint inventory is invalid")
    checkpoint_path = joinpath(directory, "checkpoint.json")
    seal_path = joinpath(directory, "seal.json")
    source_path = joinpath(directory, "source-attempt.json")
    any(islink, (checkpoint_path, seal_path, source_path)) &&
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "published baseline checkpoint traverses a symlink")
    checkpoint_bytes = read(checkpoint_path); seal_bytes = read(seal_path); source_bytes = read(source_path)
    sha256_hex(checkpoint_bytes) == checkpoint_sha && sha256_hex(seal_bytes) == seal_sha ||
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "published baseline checkpoint identity is corrupt")
    checkpoint = plain(JSON3.read(String(copy(checkpoint_bytes))))
    canonical_bytes(checkpoint) == checkpoint_bytes ||
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "published baseline checkpoint is noncanonical")
    get(checkpoint, "request_sha256", nothing) == request_sha ||
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "published baseline checkpoint belongs to another request")
    seal = plain(JSON3.read(String(copy(seal_bytes))))
    canonical_bytes(seal) == seal_bytes &&
        get(seal, "schema", nothing) == "scnsim.optimization_checkpoint_seal" &&
        get(seal, "schema_version", nothing) == 1 &&
        get(seal, "request_sha256", nothing) == request_sha &&
        get(seal, "checkpoint_sha256", nothing) == checkpoint_sha &&
        get(seal, "checkpoint_byte_length", nothing) == length(checkpoint_bytes) &&
        get(seal, "source_attempt_sha256", nothing) == sha256_hex(source_bytes) &&
        get(seal, "source_attempt_byte_length", nothing) == length(source_bytes) ||
        fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "published baseline checkpoint seal is corrupt")
    read_checkpoint_committed(request_sha, attempt_sha, String(checkpoint_sha), String(seal_sha), "reused")
    return checkpoint, String(checkpoint_sha), String(seal_sha)
end

function optimize_direct(request, plan, request_sha::String, attempt_sha::String, staging::String;
        resume_ledger_sha::Union{Nothing,AbstractString} = nothing,
        request_directory::String, attempt)
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
    cache = Dict{String,Any}()
    checkpoint_sha, checkpoint_seal_sha, baseline, baseline_roots, baseline_cost = if get(
            attempt, "baseline_checkpoint_sha256", nothing) !== nothing
        checkpoint, content_sha, seal_sha = load_baseline_checkpoint(
            request, request_sha, attempt_sha, request_directory, attempt,
        )
        stored = checkpoint["baseline"]
        stored["optimizer_coordinates_f64"] == f64_hex.(z0) ||
            fail("evidence", "evidence_integrity", "optimization_checkpoint", "artifact", "checkpoint baseline coordinates disagree with request")
        cost = f64_from_hex(stored["outcome"]["cost_f64"])
        roots = roots_from_checkpoint(request, checkpoint)
        (content_sha, seal_sha, stored, roots, cost)
    else
        baseline_evidence = Any[]
        roots, baseline_raw, baseline_views = optimization_baseline_roots(
            plan, request, base_values; extrapolation_evidence = baseline_evidence,
        )
        baseline_parameters = candidate_parameter_set(request, base_values)
        cost, components, baseline_evidence, baseline_failure = objective_outcome(
            plan, request, base_values, base_values, roots;
            extrapolation_evidence = baseline_evidence,
            prepared_raw = baseline_raw, prepared_views = baseline_views,
            candidate = optimization_candidate_position(0, 0, nothing),
        )
        baseline_failure === nothing || throw(baseline_failure)
        outcome = Dict{String,Any}(
            "status" => "success",
            "cost_f64" => f64_hex(cost),
            "objective_components" => components,
        )
        record = candidate_record(
            request, 0, 0, nothing, z0, nothing, baseline_parameters, false, outcome;
            extrapolation_evidence = baseline_evidence,
        )
        content_sha, seal_sha = publish_baseline_checkpoint(
            request, request_sha, attempt_sha, staging, record, roots,
        )
        (content_sha, seal_sha, record, roots, cost)
    end
    baseline_outcome = baseline["outcome"]
    baseline_parameters = baseline["parameters"]
    request["ref_lineage"] = baseline_primary_lineage(request, baseline)
    cache[canonical_json(baseline_parameters)] = Dict(
        "outcome" => baseline_outcome,
        "cost" => baseline_cost,
        "extrapolation_evidence" => baseline["extrapolation_evidence"],
    )
    replay_chain = resume_ledger_sha === nothing ? Dict{String,Any}[] :
        sibling_ledgers(staging, request_sha, attempt_sha, String(resume_ledger_sha),
            checkpoint_sha, checkpoint_seal_sha)
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
                checkpoint_sha, checkpoint_seal_sha,
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
            checkpoint_sha, checkpoint_seal_sha,
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
