# Included into the single SCNSimBackend module.
# Immutable selected-View realization and lineage evidence.

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
        # authoritative physical binding after normalized physical lowering.
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
