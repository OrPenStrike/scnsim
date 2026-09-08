"""Ordered ParameterSpace execution and point-owned artifact materialization.

The batch manifest is the sole attempt-level artifact.  Every point is
recompiled and realizes its declarative View independently; only immutable
topology records and the ordered source descriptor are shared.
"""

const BATCH_CHUNK_SIZE = 64
const POINT_NUMERICAL_FAILURES = Set([
    "port_realizability", "direct_response_formation", "invalid_candidate_physical_parameter",
    "unsupported_singular_capacitance_for_diagonal_root_v1", "eliminated_block_solve_failure",
    "root_slope_unresolved", "numerical_resolution_unresolved",
])

function merge_parameter_sets(base_value, overlay_value)
    base, overlay = plain(base_value), plain(overlay_value)
    get(base, "type", nothing) == "parameter_set_v2" && get(overlay, "type", nothing) == "parameter_set_v2" ||
        fail("execution", "compiler_invariant", "parameter_source", "compile", "batch parameter sets must use parameter_set_v2")
    by_key = Dict(ref_key(row["parameter"]) => deepcopy(row) for row in base["bindings"])
    order = String[ref_key(row["parameter"]) for row in base["bindings"]]
    for row in overlay["bindings"]
        key = ref_key(row["parameter"])
        haskey(by_key, key) || fail("execution", "compiler_invariant", "parameter_source", "compile", "batch point references a parameter outside its complete baseline")
        by_key[key] = deepcopy(row)
    end
    authorizations = Dict(ref_key(ref) => deepcopy(ref) for ref in get(base, "allow_extrapolation", Any[]))
    for ref in get(overlay, "allow_extrapolation", Any[]); authorizations[ref_key(ref)] = deepcopy(ref); end
    return Dict{String,Any}(
        "type" => "parameter_set_v2", "bindings" => Any[by_key[key] for key in order],
        "allow_extrapolation" => Any[authorizations[key] for key in sort!(collect(keys(authorizations)))],
    )
end

function parameter_points(source)
    item = plain(source); kind = String(item["kind"])
    if kind == "grid"
        axes = item["axes"]; shape = Int.(item["shape"])
        length(axes) == length(shape) && all(index -> length(axes[index]["values"]) == shape[index], eachindex(shape)) ||
            fail("execution", "compiler_invariant", "parameter_source", "compile", "grid shape disagrees with its axes")
        # Julia's CartesianIndices is first-index-fastest, while the contract
        # makes the final declared axis fastest.  Iterate a single ordinal and
        # derive its mixed-radix index explicitly without materializing tuples.
        count = prod(shape)
        return count, ((ordinal, begin
            remainder = ordinal; indices = zeros(Int, length(shape))
            for axis in reverse(eachindex(shape)); indices[axis] = remainder % shape[axis]; remainder ÷= shape[axis]; end
            overlay = Dict{String,Any}("type" => "parameter_set_v2", "bindings" => Any[
                Dict("parameter" => axes[axis]["parameter"], "value" => axes[axis]["values"][indices[axis] + 1]) for axis in eachindex(axes)
            ], "allow_extrapolation" => Any[])
            (Any[indices...], merge_parameter_sets(item["base_parameters"], overlay))
        end) for ordinal in 0:(count - 1))
    elseif kind == "points"
        listed = item["points"]
        return length(listed), ((ordinal, (ordinal, merge_parameter_sets(item["baseline_parameters"], listed[ordinal + 1]))) for ordinal in 0:(length(listed) - 1))
    end
    fail("execution", "compiler_invariant", "parameter_source", "compile", "batch execution requires grid or points source")
end

function execute_point_operation!(request, plan, raw_compiled, view, request_sha, attempt_sha, staging)
    operation = String(request["operation"]); compiled = view.compiled
    if operation == "solve_direct"
        solve_direct(request, view, request_sha, attempt_sha, staging)
    elseif operation == "solve_hb"
        solve_hb(request, plan, raw_compiled, view, request_sha, attempt_sha, staging)
    elseif operation == "evaluate_direct"
        kind = get(request["spec"], "type", nothing)
        kind == "diagonal_root" ? evaluate_diagonal_root(request, plan, compiled, request_sha, attempt_sha, staging) :
        kind == "hybridized_pole" ? evaluate_hybridized_pole(request, plan, compiled, request_sha, attempt_sha, staging) :
        kind == "transfer_zero" ? evaluate_transfer_zero(request, plan, view, request_sha, attempt_sha, staging) :
        kind == "residue_normalized_coupling" ? evaluate_residue_normalized_coupling(request, plan, view, request_sha, attempt_sha, staging) :
        kind == "operator" ? evaluate_operator(request, compiled, request_sha, attempt_sha, staging) :
        kind == "response_element" ? evaluate_response_element(request, view, request_sha, attempt_sha, staging) :
        fail("capability", "scaffold_unavailable", "evaluate_direct", "direct_quantity", "Direct quantity is not implemented")
    else
        fail("execution", "compiler_invariant", "parameter_source", "compile", "batch parameter source is invalid for this operation")
    end
end

function prefix_artifact_paths(value, prefix::String)
    if value isa AbstractDict
        return Dict{String,Any}(String(key) => prefix_artifact_paths(item, prefix) for (key, item) in pairs(value))
    elseif value isa AbstractVector
        return Any[prefix_artifact_paths(item, prefix) for item in value]
    elseif value isa AbstractString && startswith(value, "artifacts/")
        return prefix * String(value)
    end
    return value
end

function rewrite_point_manifests!(point_root::String, prefix::String)
    artifacts = joinpath(point_root, "artifacts"); isdir(artifacts) || return
    for (directory, _, names) in walkdir(artifacts), name in names
        endswith(name, ".manifest.json") || continue
        path = joinpath(directory, name); manifest = plain(JSON3.read(read(path, String)))
        get(manifest, "schema", nothing) == "scnsim.artifact_manifest" || continue
        artifact_path = String(manifest["artifact_path"])
        startswith(artifact_path, "artifacts/") || fail("execution", "compiler_invariant", "batch_artifact", "artifact", "point artifact manifest path is malformed")
        manifest["artifact_path"] = prefix * artifact_path
        write_bytes(path, canonical_bytes(manifest))
    end
end

function point_payload!(point_root::String, attempt_prefix::String)
    result_path = joinpath(point_root, "result.json"); outcome_path = joinpath(point_root, "outcome.json")
    isfile(result_path) && isfile(outcome_path) || fail("execution", "compiler_invariant", "batch_point", "artifact", "successful point produced no terminal result")
    result = plain(JSON3.read(read(result_path, String)))
    for key in ("schema", "schema_version", "request_sha256", "attempt_sha256", "parameters", "parameters_sha256", "ref_lineage"); pop!(result, key, nothing); end
    payload = Dict{String,Any}("schema" => "scnsim.parameter_point_payload", "schema_version" => 2)
    for (key, value) in result; payload[key] = value; end
    rewrite_point_manifests!(point_root, attempt_prefix)
    payload = prefix_artifact_paths(payload, attempt_prefix)
    # Catalog hashes bind the rewritten manifest bytes.  Locate them by their
    # attempt-relative paths without trusting an artifact ID convention.
    function refresh!(value)
        if value isa AbstractDict
            if haskey(value, "file_manifest") && haskey(value, "sha256")
                relative = String(value["file_manifest"])
                local_relative = relative[length(attempt_prefix) + 1:end]
                value["sha256"] = file_sha256(joinpath(point_root, local_relative))
            end
            for item in Base.values(value); refresh!(item); end
        elseif value isa AbstractVector
            for item in value; refresh!(item); end
        end
    end
    refresh!(payload)
    rm(result_path); rm(outcome_path)
    payload_path = joinpath(point_root, "payload.json"); write_bytes(payload_path, canonical_bytes(payload))
    return payload
end

function run_parameter_batch(request, plan, request_sha::String, attempt_sha::String, staging::String)
    source = request["parameter_source"]; count, points = parameter_points(source)
    root_relative = "artifacts/parameter_points/"; root = joinpath(staging, root_relative)
    mkpath(joinpath(root, "chunks")); mkpath(joinpath(root, "points"))
    chunks = Dict{String,Any}[]; pending = Dict{String,Any}[]; first_point = 0; chunk_ordinal = 0
    function flush_chunk!()
        isempty(pending) && return
        relative = "chunks/" * lpad(string(chunk_ordinal), 6, '0') * ".json"
        record = Dict{String,Any}(
            "schema" => "scnsim.parameter_point_chunk", "schema_version" => 2,
            "request_sha256" => request_sha, "attempt_sha256" => attempt_sha,
            "chunk_ordinal" => chunk_ordinal, "first_point" => first_point, "points" => copy(pending),
        )
        path = joinpath(root, relative); write_bytes(path, canonical_bytes(record))
        push!(chunks, Dict{String,Any}("chunk_ordinal" => chunk_ordinal, "first_point" => first_point,
            "point_count" => length(pending), "path" => root_relative * relative, "sha256" => file_sha256(path)))
        empty!(pending); first_point += BATCH_CHUNK_SIZE; chunk_ordinal += 1
    end
    for (ordinal, source_and_parameters) in points
        source_index, parameters = source_and_parameters
        point_request = deepcopy(request)
        point_request["parameter_source"] = Dict{String,Any}("kind" => "point", "parameters" => parameters)
        point_name = lpad(string(ordinal), 6, '0'); point_root = joinpath(root, "points", point_name); mkpath(point_root)
        point_relative = root_relative * "points/" * point_name * "/"
        metadata = Dict{String,Any}(
            "ordinal" => ordinal, "source_index" => source_index, "parameters" => parameters,
            "parameters_sha256" => point_parameters_sha(parameters),
        )
        try
            context = point_request["operation"] == "evaluate_direct" ? "direct_quantity" : "compile"
            raw = compile_primitive(plan, structured_parameter_values(parameters); context_kind = context,
                authorized = structured_authorizations(parameters), authorization_source = "parameter_set")
            lineage, view = realized_ref_lineage(raw, declarative_lineage(plan, point_request, raw))
            point_request["ref_lineage"] = lineage; metadata["ref_lineage"] = lineage
            execute_point_operation!(point_request, plan, raw, view, request_sha, attempt_sha, point_root)
            point_payload!(point_root, point_relative)
            metadata["status"] = "success"; metadata["payload_path"] = point_relative * "payload.json"
        catch error
            error isa BackendFailure || rethrow()
            error.kind in POINT_NUMERICAL_FAILURES || rethrow()
            rm(point_root; recursive = true); mkpath(point_root)
            metadata["status"] = "failure"; metadata["failure"] = failure_object(point_request, error)
        end
        push!(pending, metadata); length(pending) == BATCH_CHUNK_SIZE && flush_chunk!()
    end
    flush_chunk!()
    files = Dict{String,Any}[]
    for relative in relative_files(root)
        path = joinpath(root, relative)
        push!(files, Dict{String,Any}("path" => relative, "sha256" => file_sha256(path), "byte_length" => filesize(path)))
    end
    manifest = Dict{String,Any}(
        "schema" => "scnsim.parameter_points_manifest", "schema_version" => 2,
        "request_sha256" => request_sha, "attempt_sha256" => attempt_sha,
        "point_count" => count, "files" => files,
    )
    manifest_path = joinpath(staging, "artifacts", "parameter_points.manifest.json"); write_bytes(manifest_path, canonical_bytes(manifest))
    manifest_link = Dict{String,Any}(
        "id" => "parameter_points", "path" => "artifacts/parameter_points.manifest.json",
        "sha256" => file_sha256(manifest_path), "media_type" => "application/json", "byte_length" => filesize(manifest_path),
    )
    result = Dict{String,Any}(
        "schema" => "scnsim.result", "schema_version" => 2, "result_kind" => "parameter_sweep",
        "request_sha256" => request_sha, "attempt_sha256" => attempt_sha,
        "parameter_source_sha256" => sha256_hex(canonical_bytes(source)), "point_count" => count,
        "chunk_size" => BATCH_CHUNK_SIZE, "manifest" => manifest_link, "chunks" => chunks,
    )
    result_path = joinpath(staging, "result.json"); write_bytes(result_path, canonical_bytes(result))
    outcome = Dict{String,Any}(
        "schema" => "scnsim.outcome", "schema_version" => 1,
        "request_sha256" => request_sha, "attempt_sha256" => attempt_sha,
        "runtime_semantic" => request["runtime_semantic"], "status" => "success",
        "result_sha256" => file_sha256(result_path), "artifacts" => Any[manifest_link],
    )
    write_bytes(joinpath(staging, "outcome.json"), canonical_bytes(outcome))
end
