# Included into the single SCNSimBackend module.
# Canonical Result envelopes and exact artifact publication.

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
        "schema_version" => 2,
        "result_kind" => kind,
        "request_sha256" => request_sha,
        "attempt_sha256" => attempt_sha,
        "scalar_catalog" => scalars,
        "array_catalog" => arrays,
    )
end

function point_parameters_sha(parameters)
    return sha256_hex(canonical_bytes(Dict{String,Any}(
        "schema" => "scnsim.parameter_point_identity", "schema_version" => 2,
        "parameters" => plain(parameters),
    )))
end

function augment_single_result!(result, request)
    parameters = request_parameter_set(request)
    result["schema_version"] = 2
    result["parameters"] = parameters
    result["parameters_sha256"] = point_parameters_sha(parameters)
    result["ref_lineage"] = request["ref_lineage"]
    return result
end

function write_success(staging::String, request, request_sha::String, attempt_sha::String, result, artifact_catalog)
    augment_single_result!(result, request)
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
    augment_single_result!(result, request)
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

function failure_evidence(request, failure::BackendFailure)
    evidence = Dict{String,Any}(
        "type" => "failure_evidence",
        "operation" => request["operation"],
        "context_kind" => failure.context_kind,
    )
    failure.optimization_context === nothing ||
        (evidence["optimization_context"] = failure.optimization_context)
    return evidence
end

function write_failure(staging::String, request, request_sha::String, attempt_sha::String, failure::BackendFailure)
    evidence = failure_evidence(request, failure)
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
