"""Pinned JosephsonCircuits harmonic-balance adapter.

This file deliberately consumes `CompiledPrimitive` and `RealizedView` from
the only SCNSim compiler.  In particular, the backend package never executes
Python factories or reconstructs a parallel topology from names.  The narrow
adapter below owns only the representation conversion required by the pinned
JosephsonCircuits runtime.
"""

const HB_ALGORITHM_ID = "scnsim.hb_response.josephsoncircuits.v1"
const HB_JOSEPHSONCIRCUITS_VERSION = v"0.5.4"
const HB_REGISTRY_TREE = "43fa323200866910a93dd06ab8c002ed393ed9bb"
const HB_SOURCE_COMMIT = "a88bafb8f57ad5b43ef302ae893288f0b75c11e3"
const HB_PHI0 = 3.29105976e-16

# JosephsonCircuits 0.5.4 builds the final nonlinear residual in the closure
# passed to `nlsolve!`; it does not retain that closure in `NonlinearHB`.
# The pinned source at hbsolve.jl:1513-1721 invokes its package-local
# `nlsolve!(fj!, F, J, x; ...)` exactly once.  This more-specific adapter
# delegates to that exact generic method, then calls the same `fj!` once at
# the returned x.  It is the minimal source-level route to the independent
# Contract residual without copying/altering the backend's nonlinear model.
const HB_RESIDUAL_CAPTURE = Ref{Union{Nothing,Vector{ComplexF64}}}(nothing)
const HB_RESIDUAL_CAPTURE_ACTIVE = Ref(false)

function JosephsonCircuits.nlsolve!(fj!::F, residual::Vector{ComplexF64},
        jacobian::JosephsonCircuits.SparseArrays.SparseMatrixCSC{ComplexF64,Int},
        state::Vector{ComplexF64}; kwargs...) where {F<:Function}
    if !HB_RESIDUAL_CAPTURE_ACTIVE[]
        return invoke(JosephsonCircuits.nlsolve!,
            Tuple{Function,AbstractVector{ComplexF64},AbstractArray{ComplexF64},Vector{ComplexF64}},
            fj!, residual, jacobian, state; kwargs...)
    end
    result = invoke(JosephsonCircuits.nlsolve!,
        Tuple{Function,AbstractVector{ComplexF64},AbstractArray{ComplexF64},Vector{ComplexF64}},
        fj!, residual, jacobian, state; kwargs...)
    fj!(residual, nothing, state)
    HB_RESIDUAL_CAPTURE[] = copy(residual)
    return result
end

"""An expected numerical failure that belongs to one declared HB case."""
struct HBCaseNumericalFailure <: Exception
    stage::String
    message::String
end

Base.showerror(io::IO, failure::HBCaseNumericalFailure) = print(io, failure.message)

function hb_numeric_exception(error, stage::String)
    # Only known linear-algebra breakdowns are per-case numerical outcomes.
    # Request/schema/topology/protocol errors must propagate to the terminal
    # request failure and never be relabeled as an HB case failure.
    if error isa SingularException || error isa ZeroPivotException || error isa PosDefException || error isa LAPACKException
        return HBCaseNumericalFailure(stage, sprint(showerror, error))
    end
    rethrow(error)
end

function hb_require_runtime!()
    runtime_root = dirname(@__DIR__)
    metadata = try
        plain(JSON3.read(read(joinpath(runtime_root, "runtime.json"), String)))
    catch error
        fail("execution", "runtime_preparation", "runtime", "solve_hb", "HB runtime metadata is unreadable: " * sprint(showerror, error))
    end
    pinned = get(metadata, "josephsoncircuits", nothing)
    pinned isa AbstractDict && get(pinned, "version", nothing) == string(HB_JOSEPHSONCIRCUITS_VERSION) &&
        get(pinned, "general_registry_tree", nothing) == HB_REGISTRY_TREE &&
        get(pinned, "source_commit", nothing) == HB_SOURCE_COMMIT &&
        get(get(metadata, "algorithm_ids", Dict{String,Any}()), "harmonic_balance", nothing) == HB_ALGORITHM_ID ||
        fail("execution", "runtime_preparation", "runtime", "solve_hb", "HB runtime metadata does not match the sealed JosephsonCircuits identity")
    manifest = try
        read(joinpath(runtime_root, "Manifest.toml"), String)
    catch error
        fail("execution", "runtime_preparation", "runtime", "solve_hb", "HB Manifest is unreadable: " * sprint(showerror, error))
    end
    occursin("version = \"0.5.4\"", manifest) && occursin("git-tree-sha1 = \"" * HB_REGISTRY_TREE * "\"", manifest) ||
        fail("execution", "runtime_preparation", "runtime", "solve_hb", "HB Manifest does not contain the sealed JosephsonCircuits resolution")
    pkgversion(JosephsonCircuits) == HB_JOSEPHSONCIRCUITS_VERSION ||
        fail("execution", "runtime_preparation", "runtime", "solve_hb", "JosephsonCircuits does not have the sealed 0.5.4 version")
    Threads.nthreads() == 1 ||
        fail("execution", "runtime_preparation", "runtime", "solve_hb", "HB backend must run with exactly one Julia thread")
    BLAS.set_num_threads(1)
    JosephsonCircuits.FFTW.set_num_threads(1)
    BLAS.get_num_threads() == 1 ||
        fail("execution", "runtime_preparation", "runtime", "solve_hb", "HB backend could not set BLAS to one thread")
    JosephsonCircuits.FFTW.get_num_threads() == 1 ||
        fail("execution", "runtime_preparation", "runtime", "solve_hb", "HB backend could not set FFTW to one thread")
    return nothing
end

hb_complex_value(item)::ComplexF64 = begin
    value = plain(item)
    get(value, "type", nothing) == "complex_quantity_f64" ||
        fail("execution", "compiler_invariant", "compile", "solve_hb", "HB current coefficient must be complex_quantity_f64")
    complex(f64_from_hex(value["real_si_f64"]), f64_from_hex(value["imag_si_f64"]))
end

"""Return all finite tuple modes in the exact requested rectangular/crop basis."""
function hb_declared_modes(limits::Vector{Int}, crop)::Vector{Tuple}
    axes = [collect(-limit:limit) for limit in limits]
    tuples = isempty(axes) ? Tuple[()] : Tuple[Tuple(values) for values in Iterators.product(axes...)]
    filtered = crop === nothing ? tuples : Tuple[mode for mode in tuples if sum(abs, mode) <= Int(crop)]
    isempty(filtered) && fail("validation", "port_realizability", "compile", "solve_hb", "HB truncation has no modes")
    return sort!(filtered; by = mode -> (sum(abs, mode), mode))
end

function hb_frequency_grid(spec)::Vector{Float64}
    frequencies = Float64[quantity_value(item) for item in spec["frequencies"]]
    !isempty(frequencies) && all(isfinite, frequencies) && all(>(0.0), frequencies) && all(diff(frequencies) .> 0.0) ||
        fail("validation", "port_realizability", "compile", "solve_hb", "HB frequency grid must be nonempty, finite, positive, and strictly increasing")
    return frequencies
end

function hb_mode_frequency(mode, pump_frequencies::Vector{Float64})::Float64
    length(mode) == length(pump_frequencies) ||
        fail("execution", "compiler_invariant", "compile", "solve_hb", "HB tuple rank disagrees with pump axes")
    return sum((Float64(mode[index]) * pump_frequencies[index] for index in eachindex(mode)); init = 0.0)
end

"""Materialize the request-global public lattice and reject exact collisions."""
function hb_lattice(spec)
    axes = spec["pump_axes"]
    pumps = Float64[quantity_value(axis["frequency"]) for axis in axes]
    truncation = spec["truncation"]
    pump_modes = hb_declared_modes(Int[item for item in truncation["pump_harmonics"]], truncation["max_intermodulation_order"])
    response_modes = hb_declared_modes(Int[item for item in truncation["modulation_harmonics"]], truncation["max_intermodulation_order"])
    grid = hb_frequency_grid(spec)
    # A zero signed response frequency makes the JosephsonCircuits return-Z
    # normalization singular.  This is request-wide preflight, not a case
    # outcome, because every case shares the same lattice.
    response = Dict{String,Any}[]
    collision_values = Dict{String,Any}[]
    for (order, mode) in enumerate(response_modes)
        signed = Float64[frequency + hb_mode_frequency(mode, pumps) for frequency in grid]
        all(isfinite, signed) && all(!iszero, signed) ||
            fail("validation", "port_realizability", "compile", "solve_hb", "HB response lattice contains zero or non-finite signed frequency")
        push!(response, Dict("mode" => collect(mode), "signed_frequency_grid" => [quantity(value, "hertz", "inverse_time") for value in signed], "order" => order - 1))
        append!(collision_values, [Dict("mode" => collect(mode), "frequency" => f64_hex(value)) for value in signed])
    end
    # A lattice collision is two distinct declared tuples at one exact signed
    # physical frequency; including the tuple in the key would hide precisely
    # the degeneracy that makes the response basis non-injective.
    seen = Set{UInt64}()
    for entry in collision_values
        key = parse(UInt64, String(entry["frequency"]); base = 16)
        key in seen && fail("validation", "port_realizability", "compile", "solve_hb", "HB response lattice has an exact tuple/frequency collision")
        push!(seen, key)
    end
    operating_bits = Set{UInt64}()
    for mode in pump_modes
        bit = reinterpret(UInt64, hb_mode_frequency(mode, pumps))
        bit in operating_bits && fail("validation", "port_realizability", "compile", "solve_hb", "HB operating lattice has an exact tuple/frequency collision")
        push!(operating_bits, bit)
    end
    for drive in spec["drives"]
        Tuple(Int.(drive["mode"])) in pump_modes ||
            fail("validation", "port_realizability", "compile", "solve_hb", "HB truncation drops a declared drive mode")
    end
    for trace in spec["traces"]
        Tuple(Int.(trace["input_mode"])) in response_modes && Tuple(Int.(trace["output_mode"])) in response_modes ||
            fail("validation", "port_realizability", "compile", "solve_hb", "HB truncation drops a declared trace mode")
    end
    operating = Dict{String,Any}[
        Dict("mode" => collect(mode), "signed_frequency" => quantity(hb_mode_frequency(mode, pumps), "hertz", "inverse_time"), "order" => order - 1)
        for (order, mode) in enumerate(pump_modes)
    ]
    collision = sha256_hex(canonical_bytes(Dict("schema" => "scnsim.hb_tuple_frequency_collision", "schema_version" => 1, "entries" => collision_values)))
    return Dict{String,Any}(
        "pump_axes" => axes,
        "operating_point_modes" => operating,
        "input_modes" => response,
        "output_modes" => copy(response),
        "matrix_order" => "port_major_mode_minor",
        "tuple_frequency_collision_check_sha256" => collision,
    ), pumps, pump_modes, response_modes, grid
end

function hb_incidence_nodes(row, compiled::CompiledPrimitive)
    encoded = get(row, "incidence_f64", nothing)
    encoded isa AbstractVector && length(encoded) == length(compiled.nodes) ||
        fail("execution", "compiler_invariant", "compile", "solve_hb", "HB branch row lacks physical incidence")
    incidence = Float64[f64_from_hex(value) for value in encoded]
    positive = findall(>(0.0), incidence); negative = findall(<(0.0), incidence)
    length(positive) <= 1 && length(negative) <= 1 ||
        fail("execution", "compiler_invariant", "compile", "solve_hb", "HB branch incidence is not a physical two-terminal row")
    node_name(index) = index === nothing ? "0" : "n" * string(index)
    return node_name(isempty(positive) ? nothing : only(positive)), node_name(isempty(negative) ? nothing : only(negative))
end

"""Lower the compiler's raw primitive branch rows to pinned JC tuples.

The direct matrices are deliberately not factored or scalarized here: the
recorded primitive/JJ rows remain the physical source.  Matrix-valued RLGC
series resistance is accepted only when diagonal; its non-diagonal case has a
documented dev6 capability boundary and fails before any JC call.
"""
function hb_lower_raw(compiled::CompiledPrimitive)
    has_offdiagonal_series_resistance(compiled) &&
        fail("capability", "scaffold_unavailable", "compile", "solve_hb", "HB does not support off-diagonal RLGC series resistance")
    circuit = Tuple{String,String,String,Float64}[]
    serial = 0
    for row in compiled.branch_rows
        get(row, "omitted_as_zero", false) === true && continue
        kind = String(get(row, "kind", ""))
        kind in ("capacitor", "junction_capacitance", "resistor", "inductor", "josephson_inductance") || continue
        value_record = get(row, "value", nothing)
        value_record === nothing && fail("execution", "compiler_invariant", "compile", "solve_hb", "HB raw branch has no value")
        value = quantity_value(value_record)
        isfinite(value) && value > 0.0 || fail("execution", "compiler_invariant", "compile", "solve_hb", "HB raw branch has nonpositive value")
        left, right = hb_incidence_nodes(row, compiled)
        serial += 1
        prefix = kind == "capacitor" || kind == "junction_capacitance" ? "C" : kind == "resistor" ? "R" : kind == "inductor" ? "L" : "Lj"
        push!(circuit, (prefix * string(serial), left, right, value))
    end
    isempty(circuit) && fail("execution", "compiler_invariant", "compile", "solve_hb", "HB lowering has no primitive physical rows")
    # A JC P component is the native original-boundary source.  It remains
    # separate from every declared matched load resistance, which is stamped
    # below in the same original logical-Port order as B/R/M.
    for (index, port) in enumerate(compiled.port_ids)
        entries = findall(!iszero, view(compiled.B, :, index))
        length(entries) == 1 || fail("validation", "port_realizability", "compile", "solve_hb", "HB logical Port must bind exactly one raw node")
        node = "n" * string(only(entries))
        push!(circuit, ("P" * string(index), node, "0", Float64(index)))
        if compiled.M[index] == 1.0
            push!(circuit, ("Rload" * string(index), node, "0", compiled.R[index, index]))
        elseif compiled.M[index] != 0.0
            fail("execution", "compiler_invariant", "compile", "solve_hb", "HB Port load mask is not binary")
        end
        # Branch row source is a physical current injection at this original
        # logical boundary.  PTC handling retains this load in nonlinear
        # balance and is adapted only during selected response formation.
    end
    return circuit
end

function hb_sources(case, spec, compiled::CompiledPrimitive)
    declared = Dict(String(item["id"]) => item for item in spec["drives"])
    length(declared) == length(spec["drives"]) || fail("execution", "compiler_invariant", "compile", "solve_hb", "HB drive IDs collide")
    requested = Dict(String(item["drive_id"]) => item for item in case["currents"])
    length(requested) == length(case["currents"]) || fail("execution", "compiler_invariant", "compile", "solve_hb", "HB case repeats a drive")
    sources = NamedTuple{(:mode,:port,:current),Tuple{Tuple,Int,ComplexF64}}[]
    evidence = Dict{String,Any}[]
    for drive in spec["drives"]
        id = String(drive["id"]); binding = get(requested, id, nothing)
        coefficient = binding === nothing ? 0.0 + 0.0im : hb_complex_value(binding["coefficient"])
        mode = Tuple(Int.(drive["mode"])); port = findfirst(==(String(drive["port_id"])), compiled.port_ids)
        port === nothing && fail("validation", "port_realizability", "compile", "solve_hb", "HB drive names a Port outside the sealed Plan")
        if all(iszero, mode)
            imag(coefficient) == 0.0 || fail("validation", "port_realizability", "compile", "solve_hb", "HB DC coefficient must be real")
            push!(sources, (mode = mode, port = port::Int, current = coefficient))
        else
            # SCNSim exp(-iwt) to JC's conjugate convention.  The public
            # declaration owns only one member; generate the partner here.
            push!(sources, (mode = mode, port = port::Int, current = conj(coefficient)))
            push!(sources, (mode = Tuple(-value for value in mode), port = port::Int, current = coefficient))
        end
        injection = vec(copy(compiled.B[:, port::Int]))
        injection_sha = sha256_hex(canonical_bytes(Dict("schema" => "scnsim.hb_injection_map", "schema_version" => 1, "port_id" => drive["port_id"], "incidence_f64" => f64_hex.(injection))))
        push!(evidence, Dict("drive_id" => id, "mode" => collect(mode), "coefficient" => complex_quantity(coefficient, "ampere", "current"), "injection_map_sha256" => injection_sha))
    end
    return sources, evidence
end

function hb_case_state(effective_sources)
    bias = any(source -> all(iszero, source["mode"]) && (source["coefficient"]["real_si_f64"] != f64_hex(0.0) || source["coefficient"]["imag_si_f64"] != f64_hex(0.0)), effective_sources)
    pump = any(source -> any(!iszero, source["mode"]) && (source["coefficient"]["real_si_f64"] != f64_hex(0.0) || source["coefficient"]["imag_si_f64"] != f64_hex(0.0)), effective_sources)
    return bias ? "on" : "off", pump ? "on" : "off"
end

"""Run one JC nonlinear and linearized solve from a fresh zero state."""
function hb_run_case(circuit, spec, pumps, pump_modes, response_modes, grid, sources)
    truncation = spec["truncation"]
    pure_dc = isempty(pumps)
    # JC 0.5.4 cannot construct a rank-zero Fourier lattice.  The private
    # adapter gives it one inert axis while the public lattice remains ().
    backend_pumps = pure_dc ? (1.0,) : Tuple(2.0 * pi .* pumps)
    backend_harmonics = pure_dc ? (1,) : Tuple(Int.(truncation["pump_harmonics"]))
    backend_sources = if pure_dc
        [(mode = (0,), port = source.port, current = source.current) for source in sources]
    else
        sources
    end
    dc = any(source -> all(iszero, source.mode), backend_sources)
    # Mirror the pinned hbnlsolve setup (hbsolve.jl:1424-1441) solely to make
    # the Contract's zero initialization explicit.  This is not a second
    # topology: hbnlsolve parses the identical JC tuple circuit immediately
    # afterwards.  The shape is checked again against its returned nodeflux.
    backend_frequency = JosephsonCircuits.removeconjfreqs(
        JosephsonCircuits.truncfreqs(JosephsonCircuits.calcfreqsrdft(backend_harmonics);
            dc = dc, odd = Bool(truncation["four_wave_mixing"]), even = Bool(truncation["three_wave_mixing"]),
            maxintermodorder = something(truncation["max_intermodulation_order"], Inf)))
    backend_psc = JosephsonCircuits.parsesortcircuit(circuit; sorting = :none)
    state_length = (backend_psc.Nnodes - 1) * length(backend_frequency.modes)
    state_length > 0 || fail("execution", "compiler_invariant", "hb_case", "solve_hb", "HB backend has no non-ground state coordinates")
    initial_state = zeros(ComplexF64, state_length)
    HB_RESIDUAL_CAPTURE[] = nothing
    HB_RESIDUAL_CAPTURE_ACTIVE[] = true
    nonlinear = try
        JosephsonCircuits.hbnlsolve(
            backend_pumps, backend_harmonics, backend_sources, circuit, Dict{Symbol,ComplexF64}();
            iterations = 1000, maxintermodorder = something(truncation["max_intermodulation_order"], Inf),
            dc = dc, odd = Bool(truncation["four_wave_mixing"]), even = Bool(truncation["three_wave_mixing"]),
            x0 = initial_state, ftol = 1e-8, switchofflinesearchtol = 1e-5, alphamin = 1e-4,
            symfreqvar = nothing, sorting = :none, keyedarrays = false,
            sensitivitynames = String[], factorization = JosephsonCircuits.KLUfactorization(),
        )
    catch error
        throw(hb_numeric_exception(error, "operating_point"))
    finally
        HB_RESIDUAL_CAPTURE_ACTIVE[] = false
    end
    x = ComplexF64.(nonlinear.nodeflux[:])
    length(x) == state_length || error("pinned JosephsonCircuits returned nodeflux with a different state shape")
    all(isfinite, real.(x)) && all(isfinite, imag.(x)) || throw(HBCaseNumericalFailure("operating_point", "nonlinear operating point is non-finite"))
    residual = HB_RESIDUAL_CAPTURE[]
    residual === nothing && error("pinned JosephsonCircuits nonlinear residual hook did not execute")
    all(isfinite, real.(residual)) && all(isfinite, imag.(residual)) || throw(HBCaseNumericalFailure("operating_point", "nonlinear residual is non-finite"))
    absolute = norm(residual, Inf); relative = norm(x, 2) == 0.0 ? Inf : norm(residual, 2) / norm(x, 2)
    (absolute <= 1e-8 || relative < 1e-8) || throw(HBCaseNumericalFailure("operating_point", "nonlinear operating-point residual exceeds the pinned closure contract"))
    backend_modulation = pure_dc ? (0,) : Tuple(Int.(truncation["modulation_harmonics"]))
    linearized = try
        JosephsonCircuits.hblinsolve(
            2.0 * pi .* grid, circuit, Dict{Symbol,ComplexF64}(); nonlinear = nonlinear,
            Nmodulationharmonics = backend_modulation, threewavemixing = Bool(truncation["three_wave_mixing"]),
            fourwavemixing = Bool(truncation["four_wave_mixing"]), maxintermodorder = something(truncation["max_intermodulation_order"], Inf),
            nbatches = 1, sorting = :none, returnS = true, returnZ = true,
            returnSnoise = false, returnQE = false, returnCM = false, returnnodeflux = false,
            returnvoltage = false, returnnodefluxadjoint = false, returnvoltageadjoint = false,
            keyedarrays = false, sensitivitynames = String[], returnSsensitivity = false,
            returnZadjoint = false, returnZsensitivity = false, returnZsensitivityadjoint = false,
            factorization = JosephsonCircuits.KLUfactorization(),
        )
    catch error
        throw(hb_numeric_exception(error, "linearization"))
    end
    S = Array{ComplexF64,3}(linearized.S); Z = Array{ComplexF64,3}(linearized.Z)
    finite_matrix(S) && finite_matrix(Z) || throw(HBCaseNumericalFailure("response_formation", "linearized native S/Z is non-finite"))
    return nonlinear, linearized, S, Z
end

function hb_case_failure(ordinal::Int, case, effective, failure::HBCaseNumericalFailure)
    evidence = sha256_hex(canonical_bytes(Dict("schema" => "scnsim.hb_case_failure", "schema_version" => 1,
        "case_ordinal" => ordinal, "case_id" => case["id"], "stage" => failure.stage, "message" => failure.message,
        "effective_sources" => effective)))
    return Dict{String,Any}(
        "case_ordinal" => ordinal, "case_id" => case["id"], "status" => "failure", "effective_sources" => effective,
        "failure" => Dict("kind" => "hb_case_failure", "stage" => failure.stage, "message" => failure.message, "evidence_sha256" => evidence),
    )
end

function hb_manifest(staging::String, role::String, root_rel::String, datasets)
    root = joinpath(staging, root_rel)
    files = Any[Dict("path" => relative, "mode" => "regular", "byte_length" => filesize(joinpath(root, relative)), "sha256" => file_sha256(joinpath(root, relative))) for relative in relative_files(root)]
    manifest = Dict{String,Any}("schema" => "scnsim.artifact_manifest", "schema_version" => 1,
        "artifact_id" => role, "artifact_path" => root_rel, "zarr_format" => 2,
        "group_metadata_path" => ".zgroup", "datasets" => datasets, "files" => files)
    path = joinpath(staging, replace(root_rel, r"\.zarr$" => ".manifest.json"))
    write_bytes(path, canonical_bytes(manifest)); return file_sha256(path)
end

function hb_write_complex(staging::String, ordinal::Int, role::String, values, axes, unit::String, dimensionality::String; matrix::Bool)
    trace = startswith(role, "trace:"); local_role = trace ? role[7:end] : role
    root_rel = "artifacts/cases/" * lpad(string(ordinal), 6, '0') * "/" * (trace ? "traces/" : "") * local_role * ".zarr"
    root = joinpath(staging, root_rel); mkpath(root)
    shape = collect(size(values)); chunks = ndims(values) == 3 ? [min(shape[1], 1024), shape[2], shape[3]] : ndims(values) == 2 ? copy(shape) : [min(shape[1], 1024)]
    write_bytes(joinpath(root, ".zgroup"), Vector{UInt8}(codeunits("{\"zarr_format\":2}")))
    datasets = Dict{String,Any}[]; metadata = dataset_metadata(shape, chunks)
    for (name, projection) in (("real", real), ("imag", imag))
        data = joinpath(root, name); mkpath(data); write_bytes(joinpath(data, ".zarray"), Vector{UInt8}(codeunits(zarray_metadata(shape, chunks))))
        chunk_paths = String[]
        # Zarr v2 datasets declare C order.  Julia's CartesianIndices walks
        # column-major, so emit the C-order loops explicitly.  Frequency
        # matrices are stored in complete bounded slabs, not silently
        # truncated to the first slab.
        if ndims(values) == 3
            for start in 1:chunks[1]:shape[1]
                stop = min(start + chunks[1] - 1, shape[1]); chunk = string((start - 1) \ chunks[1], ".0.0")
                open(joinpath(data, chunk), "w") do io
                    for frequency in start:stop, output in axes(values, 2), input in axes(values, 3)
                        write_c_f64(io, (projection(values[frequency, output, input]),))
                    end
                end
                push!(chunk_paths, name * "/" * chunk)
            end
        elseif ndims(values) == 2
            chunk = "0.0"
            open(joinpath(data, chunk), "w") do io
                for row in axes(values, 1), column in axes(values, 2)
                    write_c_f64(io, (projection(values[row, column]),))
                end
            end
            push!(chunk_paths, name * "/" * chunk)
        else
            for start in 1:chunks[1]:shape[1]
                stop = min(start + chunks[1] - 1, shape[1]); chunk = string((start - 1) \ chunks[1])
                open(joinpath(data, chunk), "w") do io
                    for index in start:stop
                        write_c_f64(io, (projection(values[index]),))
                    end
                end
                push!(chunk_paths, name * "/" * chunk)
            end
        end
        push!(datasets, dataset_entry(name, chunk_paths))
    end
    # A trace's storage role is its declared trace ID.  `trace:<id>` is only
    # an internal routing prefix and must not leak into its manifest identity.
    manifest_sha = hb_manifest(staging, local_role, root_rel, datasets)
    artifact = Dict{String,Any}("id" => local_role, "path" => root_rel, "sha256" => manifest_sha,
        "media_type" => "application/vnd+zarr-v2", "file_manifest" => replace(root_rel, r"\.zarr$" => ".manifest.json"),
        "dtype" => "complex128", "shape" => shape, "chunks" => chunks,
        "complex_storage" => "paired_float64_real_imag", "group_metadata" => Dict("zarr_format" => 2),
        "datasets" => datasets, "axes" => axes, "unit" => unit, "dimensionality" => dimensionality,
        "chunk_policy" => ndims(values) == 3 ? "frequency_slab_full_matrix_v1" : ndims(values) == 2 ? "single_complete_array_v1" : "frequency_capped_1024_v1")
    return artifact
end

function hb_channels(coordinates::Vector{String}, modes::Vector{Tuple})
    return Any[Dict("coordinate" => coordinate, "mode" => collect(mode)) for coordinate in coordinates for mode in modes]
end

function hb_matrix_artifact(staging, ordinal, role, values, coordinates, modes, unit, dimensionality, probe)
    channels = hb_channels(coordinates, modes)
    artifact = hb_write_complex(staging, ordinal, role, values,
        [Dict("id" => "frequency", "kind" => "frequency", "request_field" => "spec.frequencies"),
         Dict("id" => "output_channel", "kind" => "output_channel", "values" => channels),
         Dict("id" => "input_channel", "kind" => "input_channel", "values" => channels)], unit, dimensionality; matrix = true)
    artifact["coordinate_ids"] = coordinates; artifact["probe_load_state"] = probe
    artifact["output_channels"] = channels; artifact["input_channels"] = copy(channels)
    return artifact
end

function hb_native_to_sc(array::Array{ComplexF64,3})
    # JosephsonCircuits returns [output,input,frequency] in the opposite
    # phasor convention.  SCNSim storage is [frequency,output,input].
    return permutedims(conj.(array), (3, 1, 2))
end

function hb_selected_native(native_s, native_z, view::RealizedView, modes)
    boundary = selected_boundary(view); q = boundary.Qk
    Q = kron(q, Matrix{Float64}(I, length(modes), length(modes)))
    nfrequency = size(native_s, 1); count = size(Q, 1)
    selected_s = Array{ComplexF64}(undef, nfrequency, count, count)
    selected_z = similar(selected_s); selected_y = similar(selected_s)
    for index in 1:nfrequency
        selected_s[index, :, :] .= Q * native_s[index, :, :] * transpose(Q)
        selected_z[index, :, :] .= Q * native_z[index, :, :] * transpose(Q)
        selected_y[index, :, :] .= checked_solve(selected_z[index, :, :], Matrix{ComplexF64}(I, count, count), "direct_response_formation", "hb_y_from_z", count)
    end
    return selected_s, selected_y, selected_z, Q
end

function hb_probe_state(lineage, ports::Vector{String}; native::Bool = false)
    if native || lineage["ptc"] === nothing
        return Dict{String,Any}[Dict("port_id" => id, "state" => "raw") for id in ports]
    end
    selected = Set(String.(lineage["ptc"]["selected_ports"]))
    return Dict{String,Any}[Dict("port_id" => id, "state" => (id in selected ? "compensated" : "raw")) for id in ports]
end

function hb_private_node_sources!(sources::Dict{String,Dict{String,Any}}, component)
    realization = component["realization"]
    if String(realization["kind"]) == "composite"
        path = String.(component["component_path"])
        for private_node in realization["private_nodes"]
            compiler_id = expanded_internal_node_id(component, private_node)
            source = Dict{String,Any}(
                "kind" => "component_private",
                "component_path" => path,
                "private_node_id" => String(private_node["id"]),
            )
            haskey(sources, compiler_id) && sources[compiler_id] != source &&
                fail("execution", "compiler_invariant", "hb_case", "solve_hb", "HB private-node source provenance collides")
            sources[compiler_id] = source
        end
        for child in realization["children"]
            hb_private_node_sources!(sources, child)
        end
    end
    return nothing
end

function hb_state_map(plan, compiled::CompiledPrimitive)
    plan_nodes = Dict{String,String}()
    for node in plan["nodes"]
        plan_nodes[String(node["node_id"])] = String(node["visibility"])
    end
    private_sources = Dict{String,Dict{String,Any}}()
    for component in plan["components"]
        hb_private_node_sources!(private_sources, component)
    end
    entries = Dict{String,Any}[]
    for (index, node) in enumerate(compiled.nodes)
        source = if haskey(plan_nodes, node) && plan_nodes[node] in ("public", "port_promoted")
            Dict("kind" => "plan_node", "plan_node_id" => node, "visibility" => plan_nodes[node])
        elseif haskey(private_sources, node)
            private_sources[node]
        elseif startswith(node, "internal-")
            Dict("kind" => "anonymous_internal", "internal_node_id" => node)
        else
            # Recursive compiler node IDs which are neither sealed Plan
            # coordinates nor compiler-private IDs are an integrity defect;
            # never invent a name-based provenance map.
            fail("execution", "compiler_invariant", "hb_case", "solve_hb", "HB state node lacks sealed source provenance")
        end
        push!(entries, Dict("state_index" => index - 1, "compiler_node_id" => node, "source" => source))
    end
    return entries
end

function hb_effective_source_vectors(sources, modes::Vector{Tuple}, compiled::CompiledPrimitive)
    values = zeros(ComplexF64, length(modes), length(compiled.nodes))
    node_index = Dict(node => index for (index, node) in enumerate(compiled.nodes))
    for source in sources
        mode_index = findfirst(==(source.mode), modes)
        mode_index === nothing && continue # source harmonic outside declared OP truncation is not an effective lattice row.
        for node in eachindex(compiled.nodes)
            values[mode_index::Int, node] += source.current * compiled.B[node, source.port]
        end
    end
    return values
end

"""Terminal HB entry point.  Case numerical errors never poison sibling cases."""
function solve_hb(request, plan, raw_compiled::CompiledPrimitive, view::RealizedView, request_sha::String, attempt_sha::String, staging::String)
    hb_require_runtime!()
    request["runtime_semantic"]["algorithm_id"] == HB_ALGORITHM_ID ||
        fail("execution", "compiler_invariant", "runtime", "solve_hb", "HB request has a mismatched algorithm identity")
    spec = request["spec"]
    get(spec, "type", nothing) == "hb_solve" || fail("execution", "compiler_invariant", "compile", "solve_hb", "solve_hb requires an hb_solve Spec")
    view.port_realizable || fail("validation", "port_realizability", "direct_response", "solve_hb", "HB response requires a Port-realizable final View")
    lattice, pumps, pump_modes, response_modes, grid = hb_lattice(spec)
    circuit = hb_lower_raw(raw_compiled)
    outcomes = Dict{String,Any}[]
    for (ordinal, case) in enumerate(spec["cases"])
        sources, effective = hb_sources(case, spec, raw_compiled)
        try
            nonlinear, linearized, native_s_raw, native_z_raw = hb_run_case(circuit, spec, pumps, pump_modes, response_modes, grid, sources)
            native_s = hb_native_to_sc(native_s_raw); native_z = hb_native_to_sc(native_z_raw)
            expected_native = length(raw_compiled.port_ids) * length(response_modes)
            size(native_s, 1) == length(grid) && size(native_s, 2) == expected_native && size(native_s, 3) == expected_native && size(native_z) == size(native_s) ||
                fail("execution", "backend_protocol", "protocol", "solve_hb", "pinned JosephsonCircuits returned native HB arrays with the wrong shape")
            selected_s, selected_y, selected_z, Q = try
                hb_selected_native(native_s, native_z, view, response_modes)
            catch error
                error isa BackendFailure && rethrow(); throw(HBCaseNumericalFailure("response_formation", sprint(showerror, error)))
            end
            finite_matrix(selected_s) && finite_matrix(selected_y) && finite_matrix(selected_z) || throw(HBCaseNumericalFailure("response_formation", "selected HB response is non-finite"))
            selected_probe = hb_probe_state(request["ref_lineage"], raw_compiled.port_ids)
            native_probe = hb_probe_state(request["ref_lineage"], raw_compiled.port_ids; native = true)
            selected_coordinates = copy(view.terminal); native_coordinates = copy(raw_compiled.port_ids)
            artifacts = Dict{String,Any}(
                "s" => hb_matrix_artifact(staging, ordinal, "s", selected_s, selected_coordinates, response_modes, "dimensionless", "dimensionless", selected_probe),
                "y" => hb_matrix_artifact(staging, ordinal, "y", selected_y, selected_coordinates, response_modes, "siemens", "conductance", selected_probe),
                "z" => hb_matrix_artifact(staging, ordinal, "z", selected_z, selected_coordinates, response_modes, "ohm", "resistance", selected_probe),
                "backend_native_s" => hb_matrix_artifact(staging, ordinal, "backend_native_s", native_s, native_coordinates, response_modes, "dimensionless", "dimensionless", native_probe),
                "backend_native_z" => hb_matrix_artifact(staging, ordinal, "backend_native_z", native_z, native_coordinates, response_modes, "ohm", "resistance", native_probe),
            )
            node_modes = lattice["operating_point_modes"]
            state_values = reshape(HB_PHI0 .* conj.(ComplexF64.(nonlinear.nodeflux[:])), length(node_modes), length(raw_compiled.nodes))
            source_values = hb_effective_source_vectors(sources, [Tuple(Int.(item["mode"])) for item in node_modes], raw_compiled)
            state_axes = [Dict("id" => "pump_mode", "kind" => "pump_mode", "values" => [item["mode"] for item in node_modes]), Dict("id" => "node_coordinate", "kind" => "node_coordinate", "values" => raw_compiled.nodes)]
            artifacts["states"] = hb_write_complex(staging, ordinal, "states", state_values, state_axes, "weber", "magnetic_flux"; matrix = false)
            artifacts["effective_source_vectors"] = hb_write_complex(staging, ordinal, "effective_source_vectors", source_values, state_axes, "ampere", "current"; matrix = false)
            traces = Dict{String,Any}[]
            selected_channels = hb_channels(selected_coordinates, response_modes)
            for trace in spec["traces"]
                input = findfirst(==(Dict("coordinate" => trace["input_port"], "mode" => trace["input_mode"])), selected_channels)
                output = findfirst(==(Dict("coordinate" => trace["output_port"], "mode" => trace["output_mode"])), selected_channels)
                (input === nothing || output === nothing) && fail("execution", "backend_protocol", "hb_case", "solve_hb", "HB trace is absent from selected channel basis")
                value = vec(selected_s[:, output::Int, input::Int])
                push!(traces, hb_write_complex(staging, ordinal, "trace:" * String(trace["id"]), value, [Dict("id" => "frequency", "kind" => "frequency", "request_field" => "spec.frequencies")], "dimensionless", "dimensionless"; matrix = false))
            end
            state_map = hb_state_map(plan, raw_compiled)
            reconstruction = maximum([begin denominator = norm(abs.(selected_s[index, :, :]) + abs.(Q * native_s[index, :, :] * transpose(Q)), Inf); denominator == 0.0 ? 0.0 : norm(selected_s[index, :, :] - Q * native_s[index, :, :] * transpose(Q), Inf) / denominator end for index in 1:length(grid)])
            reconciliation = Dict("comparable" => true, "reason" => nothing, "last_comparable_ancestor" => request["ref_lineage"]["lineage_sha256"], "normalization" => "backend_photon_flux_to_scnsim_power_wave", "residual_f64" => f64_hex(reconstruction), "evidence_sha256" => sha256_hex(canonical_bytes(Dict("q_f64" => f64_hex.(Q), "residual_f64" => f64_hex(reconstruction)))))
            push!(outcomes, Dict("case_ordinal" => ordinal, "case_id" => case["id"], "status" => "success", "bias_state" => hb_case_state(effective)[1], "pump_state" => hb_case_state(effective)[2], "effective_sources" => effective, "artifacts" => artifacts, "traces" => traces, "reconciliation" => reconciliation, "backend_normalization_evidence_sha256" => sha256_hex(canonical_bytes(Dict("normalization" => "backend_photon_flux_to_scnsim_power_wave"))), "state_node_map" => state_map))
        catch error
            if error isa HBCaseNumericalFailure
                push!(outcomes, hb_case_failure(ordinal, case, effective, error))
            elseif error isa BackendFailure
                rethrow()
            else
                rethrow()
            end
        end
    end
    result = Dict{String,Any}(
        "schema" => "scnsim.result", "schema_version" => 1, "result_kind" => "hb_batch",
        "request_sha256" => request_sha, "attempt_sha256" => attempt_sha,
        "lattice" => lattice, "truncation" => spec["truncation"], "cases" => outcomes,
    )
    write_hb_success(staging, request, request_sha, attempt_sha, result)
end
