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
    # Only pinned JosephsonCircuits numerical failures are case-local.  In
    # particular, JC 0.5.4 `nlsolve.jl:184-187` raises this exact
    # ErrorException when the nonlinear line search has no numerical step.
    # Request/schema/topology/protocol defects must remain request failures;
    # do not widen this to arbitrary ErrorException/ArgumentError.
    if error isa SingularException || error isa ZeroPivotException || error isa PosDefException || error isa LAPACKException ||
       (stage == "operating_point" && error isa ErrorException && sprint(showerror, error) == "NaN in nonlinear solver.")
        return HBCaseNumericalFailure(stage, sprint(showerror, error))
    end
    rethrow(error)
end

function hb_require_runtime!()
    runtime_root = dirname(@__DIR__)
    metadata = try
        plain(JSON3.read(read(joinpath(runtime_root, "runtime.json"), String)))
    catch error
        fail("execution", "runtime_preparation", "runtime", "runtime", "HB runtime metadata is unreadable: " * sprint(showerror, error))
    end
    pinned = get(metadata, "josephsoncircuits", nothing)
    pinned isa AbstractDict && get(pinned, "version", nothing) == string(HB_JOSEPHSONCIRCUITS_VERSION) &&
        get(pinned, "general_registry_tree", nothing) == HB_REGISTRY_TREE &&
        get(pinned, "source_commit", nothing) == HB_SOURCE_COMMIT &&
        get(get(metadata, "algorithm_ids", Dict{String,Any}()), "harmonic_balance", nothing) == HB_ALGORITHM_ID ||
        fail("execution", "runtime_preparation", "runtime", "runtime", "HB runtime metadata does not match the sealed JosephsonCircuits identity")
    manifest = try
        read(joinpath(runtime_root, "Manifest.toml"), String)
    catch error
        fail("execution", "runtime_preparation", "runtime", "runtime", "HB Manifest is unreadable: " * sprint(showerror, error))
    end
    occursin("version = \"0.5.4\"", manifest) && occursin("git-tree-sha1 = \"" * HB_REGISTRY_TREE * "\"", manifest) ||
        fail("execution", "runtime_preparation", "runtime", "runtime", "HB Manifest does not contain the sealed JosephsonCircuits resolution")
    pkgversion(JosephsonCircuits) == HB_JOSEPHSONCIRCUITS_VERSION ||
        fail("execution", "runtime_preparation", "runtime", "runtime", "JosephsonCircuits does not have the sealed 0.5.4 version")
    Threads.nthreads() == 1 ||
        fail("execution", "runtime_preparation", "runtime", "runtime", "HB backend must run with exactly one Julia thread")
    BLAS.set_num_threads(1)
    JosephsonCircuits.FFTW.set_num_threads(1)
    BLAS.get_num_threads() == 1 ||
        fail("execution", "runtime_preparation", "runtime", "runtime", "HB backend could not set BLAS to one thread")
    JosephsonCircuits.FFTW.get_num_threads() == 1 ||
        fail("execution", "runtime_preparation", "runtime", "runtime", "HB backend could not set FFTW to one thread")
    return nothing
end

hb_complex_value(item)::ComplexF64 = begin
    value = plain(item)
    get(value, "type", nothing) == "complex_quantity_f64" ||
        fail("execution", "compiler_invariant", "compile", "compile", "HB current coefficient must be complex_quantity_f64")
    complex(f64_from_hex(value["real_si_f64"]), f64_from_hex(value["imag_si_f64"]))
end

"""Materialize the public lattice from the pinned JC representative basis.

The nonlinear pump lattice is the RFFT representative (`removeconjfreqs`);
the small-signal lattice is JC's DFT response basis.  This is deliberately not
a hand-made rectangular enumeration: parity and diamond crop semantics belong
to the pinned backend implementation.
"""
function hb_declared_modes(limits::Vector{Int}, crop; dc::Bool, odd::Bool, even::Bool, response::Bool)::Vector{Tuple}
    # JC has no rank-zero Fourier constructor.  SCNSim's private pure-DC
    # adapter maps its `(0,)` member back to public `()` below; without an
    # explicit declared DC drive, the physical operating lattice is vacuous.
    if isempty(limits)
        return response || dc ? Tuple[()] : Tuple[]
    end
    all_frequencies = response ?
        JosephsonCircuits.calcfreqsdft(Tuple(limits)) :
        JosephsonCircuits.calcfreqsrdft(Tuple(limits))
    truncated = JosephsonCircuits.truncfreqs(all_frequencies;
        dc = dc, odd = odd, even = even,
        maxintermodorder = something(crop, Inf))
    representative = response ? truncated : JosephsonCircuits.removeconjfreqs(truncated)
    return Tuple[Tuple(Int.(mode)) for mode in representative.modes]
end

function hb_frequency_grid(spec)::Vector{Float64}
    frequencies = Float64[quantity_value(item) for item in spec["frequencies"]]
    !isempty(frequencies) && all(isfinite, frequencies) && all(>(0.0), frequencies) && all(diff(frequencies) .> 0.0) ||
        fail("validation", "port_realizability", "compile", "compile", "HB frequency grid must be nonempty, finite, positive, and strictly increasing")
    return frequencies
end

function hb_mode_frequency(mode, pump_frequencies::Vector{Float64})::Float64
    length(mode) == length(pump_frequencies) ||
        fail("execution", "compiler_invariant", "compile", "compile", "HB tuple rank disagrees with pump axes")
    return sum((Float64(mode[index]) * pump_frequencies[index] for index in eachindex(mode)); init = 0.0)
end

"""Materialize the request-global public lattice and reject exact collisions."""
function hb_lattice(spec)
    axes = spec["pump_axes"]
    pumps = Float64[quantity_value(axis["frequency"]) for axis in axes]
    truncation = spec["truncation"]
    declared_dc = any(drive -> all(iszero, Int.(drive["mode"])), spec["drives"])
    pump_modes = hb_declared_modes(Int[item for item in truncation["pump_harmonics"]], truncation["max_intermodulation_order"];
        dc = declared_dc, odd = Bool(truncation["four_wave_mixing"]), even = Bool(truncation["three_wave_mixing"]), response = false)
    response_modes = hb_declared_modes(Int[item for item in truncation["modulation_harmonics"]], truncation["max_intermodulation_order"];
        # hblinsolve's native response lattice always retains its DC signal
        # member, then selects odd 3WM and even 4WM families.
        dc = true, odd = Bool(truncation["three_wave_mixing"]), even = Bool(truncation["four_wave_mixing"]), response = true)
    grid = hb_frequency_grid(spec)
    # A zero signed response frequency makes the JosephsonCircuits return-Z
    # normalization singular.  This is request-wide preflight, not a case
    # outcome, because every case shares the same lattice.
    response = Dict{String,Any}[]
    collision_values = Dict{String,Any}[]
    for (order, mode) in enumerate(response_modes)
        signed = Float64[frequency + hb_mode_frequency(mode, pumps) for frequency in grid]
        all(isfinite, signed) && all(!iszero, signed) ||
            fail("validation", "port_realizability", "compile", "compile", "HB response lattice contains zero or non-finite signed frequency")
        push!(response, Dict("mode" => collect(mode), "signed_frequency_grid" => [quantity(value, "hertz", "inverse_time") for value in signed], "order" => order - 1))
        append!(collision_values, [Dict("mode" => collect(mode), "frequency" => f64_hex(value)) for value in signed])
    end
    # A lattice collision is two distinct declared tuples at one exact signed
    # physical frequency; including the tuple in the key would hide precisely
    # the degeneracy that makes the response basis non-injective.
    # Response grids are independent samples.  A collision is meaningful only
    # between two tuple channels *at the same declared grid ordinal*; equal
    # signed frequencies at different declared samples are not a collapsed
    # channel basis.
    for frequency_index in eachindex(grid)
        seen = Set{UInt64}()
        for mode in response_modes
            key = reinterpret(UInt64, grid[frequency_index] + hb_mode_frequency(mode, pumps))
            key in seen && fail("validation", "port_realizability", "compile", "compile", "HB response lattice has an exact tuple/frequency collision")
            push!(seen, key)
        end
    end
    operating_bits = Set{UInt64}()
    for mode in pump_modes
        bit = reinterpret(UInt64, hb_mode_frequency(mode, pumps))
        bit in operating_bits && fail("validation", "port_realizability", "compile", "compile", "HB operating lattice has an exact tuple/frequency collision")
        push!(operating_bits, bit)
    end
    # A `CurrentDrive` is an additive contribution.  Multiple declared drives
    # may therefore share one Port and *the same signed* tuple.  Only an
    # independently declared opposite tuple attempts to replace the generated
    # real-current conjugate partner and is rejected.
    signed_declarations = Dict{String,Set{Tuple}}()
    for drive in spec["drives"]
        mode = Tuple(Int.(drive["mode"]))
        # The nonlinear lattice is JC's RFFT representative basis.  A public
        # negative tuple is still a physical declared drive: it maps to its
        # retained conjugate representative rather than being rejected.
        # The coefficient conversion itself is centralized below so this
        # preflight cannot accidentally choose a second phasor convention.
        (mode in pump_modes || Tuple(-entry for entry in mode) in pump_modes) ||
            fail("validation", "port_realizability", "compile", "compile", "HB truncation drops a declared drive mode")
        if !all(iszero, mode)
            opposite = Tuple(-entry for entry in mode)
            port_id = String(drive["port_id"])
            declared = get!(signed_declarations, port_id, Set{Tuple}())
            opposite in declared &&
                fail("validation", "port_realizability", "compile", "compile", "HB independently declares a generated conjugate drive partner")
            push!(declared, mode)
        end
    end
    for trace in spec["traces"]
        Tuple(Int.(trace["input_mode"])) in response_modes && Tuple(Int.(trace["output_mode"])) in response_modes ||
            fail("validation", "port_realizability", "compile", "compile", "HB truncation drops a declared trace mode")
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
    ), pumps, pump_modes, response_modes, grid, declared_dc
end

function hb_incidence_nodes(row, compiled::CompiledPrimitive)
    encoded = get(row, "incidence_f64", nothing)
    encoded isa AbstractVector && length(encoded) == length(compiled.nodes) ||
        fail("execution", "compiler_invariant", "compile", "compile", "HB branch row lacks physical incidence")
    incidence = Float64[f64_from_hex(value) for value in encoded]
    positive = findall(>(0.0), incidence); negative = findall(<(0.0), incidence)
    length(positive) <= 1 && length(negative) <= 1 ||
        fail("execution", "compiler_invariant", "compile", "compile", "HB branch incidence is not a physical two-terminal row")
    node_name(index) = index === nothing ? "0" : "n" * string(index)
    return node_name(isempty(positive) ? nothing : only(positive)), node_name(isempty(negative) ? nothing : only(negative))
end

"""Lower a raw physical nodal C/G matrix without discarding mutual entries.

`CompiledPrimitive` is the compiler-owned raw topology basis.  Its C and G
matrices are physical Maxwell/Laplacian stamps before View transforms, so this
decomposition exactly recreates every two-terminal and reference shunt branch
for JosephsonCircuits.  It deliberately does not use an eigendecomposition or
any numerical repair: a nonphysical sign/row sum is a compiler invariant
failure, not an invitation to approximate a realizable network.
"""
function hb_lower_nodal_matrix!(circuit, matrix::Matrix{Float64}, compiled::CompiledPrimitive, prefix::String)
    size(matrix) == (length(compiled.nodes), length(compiled.nodes)) ||
        fail("execution", "compiler_invariant", "compile", "compile", "HB nodal matrix shape is malformed")
    matrix == transpose(matrix) ||
        fail("execution", "compiler_invariant", "compile", "compile", "HB nodal matrix is not bit-exact symmetric")
    serial = 0
    for row in 1:size(matrix, 1)-1, column in row+1:size(matrix, 2)
        off_diagonal = matrix[row, column]
        isfinite(off_diagonal) && off_diagonal <= 0.0 ||
            fail("execution", "compiler_invariant", "compile", "compile", "HB physical nodal matrix has a positive off-diagonal")
        value = -off_diagonal
        value == 0.0 && continue
        element_value = prefix == "R" ? 1.0 / value : value
        isfinite(element_value) && element_value > 0.0 ||
            fail("execution", "compiler_invariant", "compile", "compile", "HB physical shunt cannot be represented as a finite JC element")
        serial += 1
        push!(circuit, (prefix * "m" * string(serial), "n" * string(row), "n" * string(column), element_value))
    end
    for row in axes(matrix, 1)
        ground = sum(view(matrix, row, :))
        isfinite(ground) && ground >= 0.0 ||
            fail("execution", "compiler_invariant", "compile", "compile", "HB physical nodal matrix has a negative reference shunt")
        ground == 0.0 && continue
        element_value = prefix == "R" ? 1.0 / ground : ground
        isfinite(element_value) && element_value > 0.0 ||
            fail("execution", "compiler_invariant", "compile", "compile", "HB physical reference shunt cannot be represented as a finite JC element")
        serial += 1
        push!(circuit, (prefix * "g" * string(serial), "n" * string(row), "0", element_value))
    end
    return nothing
end

"""Lower the compiler's raw primitive branch rows to pinned JC tuples.

The direct matrices are deliberately not factored or scalarized here: the
recorded primitive/JJ rows remain the physical source.  Matrix-valued RLGC
series resistance is accepted only when diagonal; its non-diagonal case has a
documented dev6 capability boundary and fails before any JC call.
"""
function hb_lower_raw(compiled::CompiledPrimitive, plan)
    has_offdiagonal_series_resistance(compiled) &&
        fail("capability", "scaffold_unavailable", "compile", "compile", "HB does not support off-diagonal RLGC series resistance")
    circuit = Tuple{String,String,String,Any}[]
    inductors = Dict{String,String}()
    serial = 0
    for row in compiled.branch_rows
        get(row, "omitted_as_zero", false) === true && continue
        kind = String(get(row, "kind", ""))
        kind in ("inductor", "josephson_inductance") || continue
        value_record = get(row, "value", nothing)
        value_record === nothing && fail("execution", "compiler_invariant", "compile", "compile", "HB raw branch has no value")
        value = quantity_value(value_record)
        isfinite(value) && value > 0.0 || fail("execution", "compiler_invariant", "compile", "compile", "HB raw branch has nonpositive value")
        left, right = hb_incidence_nodes(row, compiled)
        serial += 1
        prefix = kind == "inductor" ? "L" : "Lj"
        name = prefix * string(serial)
        push!(circuit, (name, left, right, value))
        if kind == "inductor"
            branch = get(row, "branch_id", nothing)
            branch isa AbstractString || fail("execution", "compiler_invariant", "compile", "compile", "HB inductor row lacks sealed branch identity")
            inductors[join(String.(row["component_path"]), "\u001f") * "\u001e" * String(branch)] = name
        end
    end
    # All lumped and π-ladder C/G rows are represented once through the raw
    # compiler matrices.  This retains full RLGC C/G coupling rather than
    # silently scalarizing only the diagonal audit rows.
    hb_lower_nodal_matrix!(circuit, compiled.C, compiled, "C")
    hb_lower_nodal_matrix!(circuit, compiled.G, compiled, "R")
    isempty(circuit) && fail("execution", "compiler_invariant", "compile", "compile", "HB lowering has no primitive physical rows")
    # A JC P component is the native original-boundary source.  It remains
    # separate from every declared matched load resistance, which is stamped
    # below in the same original logical-Port order as B/R/M.
    for (index, port) in enumerate(compiled.port_ids)
        entries = findall(!iszero, view(compiled.B, :, index))
        length(entries) == 1 || fail("validation", "port_realizability", "compile", "compile", "HB logical Port must bind exactly one raw node")
        node = "n" * string(only(entries))
        push!(circuit, ("P" * string(index), node, "0", Float64(index)))
        if compiled.M[index] == 1.0 || compiled.M[index] == 0.0
            # The nonlinear operating point is always solved with every
            # physical matched boundary loaded.  PTC is an explicitly later
            # small-signal response operation, never an alteration of balance.
            push!(circuit, ("Rload" * string(index), node, "0", compiled.R[index, index]))
        else
            fail("execution", "compiler_invariant", "compile", "compile", "HB Port load mask is not binary")
        end
        # Branch row source is a physical current injection at this original
        # logical boundary.  PTC handling retains this load in nonlinear
        # balance and is adapted only during selected response formation.
    end
    # A diagonal series-R/full-L pi section is represented faithfully as R
    # then L through an adapter-private node.  Those nodes are intentionally
    # excluded from SCNSim state evidence: `hb_state_values` maps only the
    # raw compiler node IDs back to sealed topology.  Off-diagonal R was
    # rejected above; full reciprocal L is retained through JC K elements.
    for block in compiled.series_rl
        conductors = size(block.incidence, 2)
        size(block.resistance) == (conductors, conductors) && size(block.inductance) == size(block.resistance) ||
            fail("execution", "compiler_invariant", "compile", "compile", "HB series RL block shape is malformed")
        names = String[]
        for conductor in 1:conductors
            left, right = hb_incidence_nodes(Dict("incidence_f64" => f64_hex.(block.incidence[:, conductor])), compiled)
            resistance = block.resistance[conductor, conductor]; inductance = block.inductance[conductor, conductor]
            isfinite(resistance) && resistance >= 0.0 && isfinite(inductance) && inductance > 0.0 ||
                fail("execution", "compiler_invariant", "compile", "compile", "HB series RL diagonal is nonphysical")
            middle = "hbseries-" * sha256_hex(canonical_bytes(Dict("id" => block.id, "conductor" => conductor)))
            resistance > 0.0 && push!(circuit, ("Rseries" * string(length(circuit) + 1), left, middle, resistance))
            name = "Lseries" * string(length(circuit) + 1)
            push!(circuit, (name, resistance > 0.0 ? middle : left, right, inductance))
            push!(names, name)
        end
        for row in 1:conductors-1, column in row+1:conductors
            mutual = block.inductance[row, column]
            mutual == 0.0 && continue
            coefficient = mutual / sqrt(block.inductance[row, row] * block.inductance[column, column])
            isfinite(coefficient) && abs(coefficient) < 1.0 ||
                fail("execution", "compiler_invariant", "compile", "compile", "HB series inductance matrix is not strictly reciprocal-SPD")
            push!(circuit, ("Kseries" * string(length(circuit) + 1), names[row], names[column], coefficient))
        end
    end
    # Mutual rows name exact sealed inductive branches.  JC's K element uses
    # the same reciprocal coefficient k, so no derived-M round trip loses
    # orientation or precision.
    for row in compiled.branch_rows
        String(get(row, "kind", "")) == "mutual_inductance" || continue
        left_ref = lower_branch_reference(plan, row["branch_a"]); right_ref = lower_branch_reference(plan, row["branch_b"])
        left = join(String.(left_ref["component_path"]), "\u001f") * "\u001e" * String(left_ref["branch_id"])
        right = join(String.(right_ref["component_path"]), "\u001f") * "\u001e" * String(right_ref["branch_id"])
        haskey(inductors, left) && haskey(inductors, right) ||
            fail("execution", "compiler_invariant", "compile", "compile", "HB mutual row does not resolve to lowered inductors")
        coefficient = quantity_value(row["coupling_coefficient"])
        isfinite(coefficient) && abs(coefficient) < 1.0 ||
            fail("execution", "compiler_invariant", "compile", "compile", "HB mutual coefficient is not physically valid")
        coefficient == 0.0 || push!(circuit, ("K" * string(length(circuit) + 1), inductors[left], inductors[right], coefficient))
    end
    return circuit
end

function hb_sources(case, spec, compiled::CompiledPrimitive)
    declared = Dict(String(item["id"]) => item for item in spec["drives"])
    length(declared) == length(spec["drives"]) || fail("execution", "compiler_invariant", "compile", "compile", "HB drive IDs collide")
    requested = Dict(String(item["drive_id"]) => item for item in case["currents"])
    length(requested) == length(case["currents"]) || fail("execution", "compiler_invariant", "compile", "compile", "HB case repeats a drive")
    sources = NamedTuple{(:mode,:port,:current),Tuple{Tuple,Int,ComplexF64}}[]
    evidence = Dict{String,Any}[]
    for drive in spec["drives"]
        id = String(drive["id"]); binding = get(requested, id, nothing)
        coefficient = binding === nothing ? 0.0 + 0.0im : hb_complex_value(binding["coefficient"])
        mode = Tuple(Int.(drive["mode"])); port = findfirst(==(String(drive["port_id"])), compiled.port_ids)
        port === nothing && fail("validation", "port_realizability", "compile", "compile", "HB drive names a Port outside the sealed Plan")
        if all(iszero, mode)
            imag(coefficient) == 0.0 || fail("validation", "port_realizability", "compile", "compile", "HB DC coefficient must be real")
            push!(sources, (mode = mode, port = port::Int, current = coefficient))
        else
            # Keep SCNSim's declared exp(-iωt) coefficient here.  The pinned
            # RFFT representative/source conversion happens exactly once in
            # `hb_backend_sources`; inserting an opposite tuple here would
            # violate JC's representative-mode lookup.
            push!(sources, (mode = mode, port = port::Int, current = coefficient))
        end
        injection = vec(copy(compiled.B[:, port::Int]))
        injection_sha = sha256_hex(canonical_bytes(Dict("schema" => "scnsim.hb_injection_map", "schema_version" => 1, "port_id" => drive["port_id"], "incidence_f64" => f64_hex.(injection))))
        push!(evidence, Dict("drive_id" => id, "mode" => collect(mode), "coefficient" => complex_quantity(coefficient, "ampere", "current"), "injection_map_sha256" => injection_sha))
    end
    # Current sources share the physical Port/mode channel.  Sum them before
    # determining bias/pump state or constructing a JC source vector, while
    # retaining the declared-edge evidence above in declaration order.
    accumulated = Dict{Tuple{Tuple,Int},ComplexF64}()
    ordered = Tuple{Tuple,Int}[]
    for source in sources
        key = (source.mode, source.port)
        haskey(accumulated, key) || push!(ordered, key)
        accumulated[key] = get(accumulated, key, 0.0 + 0.0im) + source.current
    end
    aggregated = NamedTuple{(:mode,:port,:current),Tuple{Tuple,Int,ComplexF64}}[
        (mode = key[1], port = key[2], current = accumulated[key]) for key in ordered
    ]
    return aggregated, evidence
end

"""Return JC's one retained RFFT representative for a public source tuple.

For SCNSim's ``exp(-iωt)`` coefficient ``c_m``, the pinned JC convention has
``conj(c_m)`` at ``m`` and ``c_m`` at ``-m``.  JC receives exactly one of this
materialized pair: if the public tuple is omitted by its RFFT representative
selection, the retained opposite tuple receives the un-conjugated coefficient.
The returned boolean records whether that generated opposite tuple was used.
"""
function hb_representative_source(mode::Tuple, coefficient::ComplexF64, backend_modes, pure_dc::Bool)
    backend_mode = pure_dc && isempty(mode) ? (0,) : mode
    representative = Set(Tuple(Int.(item)) for item in backend_modes)
    if backend_mode in representative
        return backend_mode, (all(iszero, backend_mode) ? coefficient : conj(coefficient)), false
    end
    opposite = Tuple(-entry for entry in backend_mode)
    opposite in representative ||
        fail("execution", "backend_protocol", "protocol", "protocol", "HB public source mode has no pinned JC representative")
    all(iszero, opposite) &&
        fail("execution", "backend_protocol", "protocol", "protocol", "HB DC source has no pinned JC representative")
    return opposite, coefficient, true
end

"""Map sealed public source coefficients to JC's RFFT representative modes."""
function hb_backend_sources(sources, backend_modes, pure_dc::Bool)
    output = NamedTuple{(:mode,:port,:current),Tuple{Tuple,Int,ComplexF64}}[]
    for source in sources
        mode, current, _ = hb_representative_source(source.mode, source.current, backend_modes, pure_dc)
        push!(output, (mode = mode, port = source.port, current = current))
    end
    return output
end

"""Attach the sealed public-to-JC conversion audit to each declared drive.

The declaration remains in request order.  `generated_conjugate` makes the
real-current partner explicit, and `backend_binding` explains which one of the
pair the pinned JC RFFT representative receives.  Backend indices are durable
audit data only: the source identity is still the SCNSim drive ID and tuple.
"""
function hb_effective_source_evidence(evidence, backend_modes, pure_dc::Bool)
    records = Dict{String,Any}[]
    for item in evidence
        mode = Tuple(Int.(item["mode"]))
        coefficient = hb_complex_value(item["coefficient"])
        jc_mode, jc_coefficient, _ =
            hb_representative_source(mode, coefficient, backend_modes, pure_dc)
        # Public rank-zero DC remains the empty tuple; `(0,)` is solely the
        # private JC adapter and must never leak into durable SCNSim evidence.
        opposite = all(iszero, mode) ? mode : Tuple(-entry for entry in mode)
        representative_index = findfirst(==(jc_mode), Tuple[Tuple(Int.(entry)) for entry in backend_modes])
        representative_index === nothing &&
            fail("execution", "backend_protocol", "protocol", "protocol", "HB representative mapping has no pinned JC index")
        push!(records, Dict(
            "drive_id" => item["drive_id"],
            "mode" => collect(mode),
            "coefficient" => item["coefficient"],
            "injection_map_sha256" => item["injection_map_sha256"],
            "generated_conjugate" => Dict(
                "mode" => collect(opposite),
                "coefficient" => complex_quantity(all(iszero, jc_mode) ? coefficient : conj(coefficient), "ampere", "current"),
            ),
            "backend_binding" => Dict(
                "representative_mode" => collect(jc_mode),
                "representative_index" => representative_index::Int - 1,
                "coefficient" => complex_quantity(jc_coefficient, "ampere", "current"),
                "coefficient_convention" => "exp_plus_i_m_dot_omega_t_josephsoncircuits_source",
            ),
        ))
    end
    return records
end

"""Classify one case from its physical, orientation-aware node injections.

Declared drive edges remain durable evidence, but state and driven-PTC
authorization are properties of the net selected physical source.  In
particular, two logical Ports may have cancelling B columns; classifying their
coefficients before this accumulation would mislabel an undriven case.
"""
function hb_case_state(source_vectors::AbstractMatrix{ComplexF64}, modes::AbstractVector{<:Tuple})
    size(source_vectors, 1) == length(modes) ||
        fail("execution", "compiler_invariant", "hb_case", "hb_case", "HB source rows do not match the operating lattice")
    bias = false; pump = false
    for (index, mode) in enumerate(modes)
        active = any(!iszero, view(source_vectors, index, :))
        all(iszero, mode) ? (bias |= active) : (pump |= active)
    end
    return bias ? "on" : "off", pump ? "on" : "off"
end

"""Run one JC nonlinear and linearized solve from a fresh zero state."""
function hb_run_case(circuit, spec, pumps, pump_modes, response_modes, grid, sources, declared_dc::Bool)
    truncation = spec["truncation"]
    pure_dc = isempty(pumps)
    # JC 0.5.4 cannot construct a rank-zero Fourier lattice.  The private
    # adapter gives it one inert axis while the public lattice remains ().
    backend_pumps = pure_dc ? (1.0,) : Tuple(2.0 * pi .* pumps)
    backend_harmonics = pure_dc ? (0,) : Tuple(Int.(truncation["pump_harmonics"]))
    # Rank-zero public pump axes require one private static Fourier member in
    # JC.  It is exactly `(0,)`, never a synthetic harmonic, and is removed
    # from the public lattice after solving.
    dc = declared_dc
    # Mirror the pinned hbnlsolve setup (hbsolve.jl:1424-1441) solely to make
    # the Contract's zero initialization explicit.  This is not a second
    # topology: hbnlsolve parses the identical JC tuple circuit immediately
    # afterwards.  The shape is checked again against its returned nodeflux.
    backend_psc = JosephsonCircuits.parsesortcircuit(circuit; sorting = :none)
    nonlinear = nothing
    closure = Dict{String,Any}(
        "status" => "not_applicable",
        "reason" => "no_operating_point_lattice",
    )
    if !isempty(pump_modes)
        backend_frequency = JosephsonCircuits.removeconjfreqs(
            JosephsonCircuits.truncfreqs(JosephsonCircuits.calcfreqsrdft(backend_harmonics);
                dc = dc, odd = Bool(truncation["four_wave_mixing"]), even = Bool(truncation["three_wave_mixing"]),
                maxintermodorder = something(truncation["max_intermodulation_order"], Inf)))
        # The public lattice and the pinned backend must agree exactly.  The
        # rank-zero DC adapter is the sole intentional representation change.
        expected_backend_modes = pure_dc ? Tuple[(0,)] : pump_modes
        Tuple[Tuple(Int.(mode)) for mode in backend_frequency.modes] == expected_backend_modes ||
            fail("execution", "backend_protocol", "protocol", "protocol", "JC operating-point lattice disagrees with SCNSim's sealed lattice")
        backend_sources = hb_backend_sources(sources, backend_frequency.modes, pure_dc)
        state_length = (backend_psc.Nnodes - 1) * length(backend_frequency.modes)
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
        absolute = norm(residual, Inf)
        state_norm = norm(x, 2)
        relative = state_norm == 0.0 ? nothing : norm(residual, 2) / state_norm
        absolute_ok = absolute <= 1e-8
        relative_ok = relative !== nothing && relative < 1e-8
        (absolute_ok || relative_ok) || throw(HBCaseNumericalFailure("operating_point", "nonlinear operating-point residual exceeds the pinned closure contract"))
        closure = Dict{String,Any}(
            "status" => "satisfied",
            "absolute_residual_f64" => f64_hex(absolute),
            "relative_residual" => relative === nothing ?
                Dict("status" => "not_applicable", "reason" => "zero_state_norm") :
                Dict("status" => "value", "value_f64" => f64_hex(relative)),
            "successful_disjunct" => (absolute_ok && relative_ok ? "both" : absolute_ok ? "absolute" : "relative"),
        )
    end
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
    return nonlinear, linearized, S, Z, closure
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

"""HB's Result catalog carries closed Zarr metadata, unlike legacy Direct's
manifest-oriented chunk listing.  The manifest still inventories every byte;
the catalog is the semantic dtype/shape authority consumed by resolve()."""
hb_dataset_entry(path::String, shape::Vector{Int}, chunks::Vector{Int}) =
    Dict{String,Any}("path" => path, "metadata" => dataset_metadata(shape, chunks))

function hb_write_complex(staging::String, ordinal::Int, role::String, values, axis_metadata, unit::String, dimensionality::String; matrix::Bool)
    trace = startswith(role, "trace:"); local_role = trace ? role[7:end] : role
    root_rel = "artifacts/cases/" * lpad(string(ordinal), 6, '0') * "/" * (trace ? "traces/" : "") * local_role * ".zarr"
    root = joinpath(staging, root_rel); mkpath(root)
    shape = collect(size(values))
    # Zarr permits an empty leading state/source axis, but its chunk metadata
    # remains strictly positive.  An empty array has zero chunk *counts*, not
    # a zero-sized chunk and not a fabricated DC row.
    chunks = ndims(values) == 3 ? [min(max(shape[1], 1), 1024), max(shape[2], 1), max(shape[3], 1)] :
        ndims(values) == 2 ? max.(shape, 1) : [min(max(shape[1], 1), 1024)]
    write_bytes(joinpath(root, ".zgroup"), Vector{UInt8}(codeunits("{\"zarr_format\":2}")))
    # The durable result catalog owns the typed `{path, metadata}` records,
    # whereas the byte manifest owns the exact `.zarray` and chunk paths.
    # Keep those two closed envelopes distinct: Python rebuilds manifests from
    # the latter and compares their canonical bytes before promotion.
    datasets = Dict{String,Any}[]
    manifest_datasets = Dict{String,Any}[]
    metadata = dataset_metadata(shape, chunks)
    for (name, projection) in (("real", real), ("imag", imag))
        data = joinpath(root, name); mkpath(data); write_bytes(joinpath(data, ".zarray"), Vector{UInt8}(codeunits(zarray_metadata(shape, chunks))))
        chunk_paths = String[]
        # Zarr v2 datasets declare C order.  Julia's CartesianIndices walks
        # column-major, so emit the C-order loops explicitly.  Frequency
        # matrices are stored in complete bounded slabs, not silently
        # truncated to the first slab.
        if any(iszero, shape)
            # `ceil(0 / positive_chunk) == 0`: the canonical Zarr grid has
            # no payload chunks.  Preserve that exact empty physical state.
        elseif ndims(values) == 3
            for start in 1:chunks[1]:shape[1]
                stop = min(start + chunks[1] - 1, shape[1]); chunk = string((start - 1) ÷ chunks[1], ".0.0")
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
                stop = min(start + chunks[1] - 1, shape[1]); chunk = string((start - 1) ÷ chunks[1])
                open(joinpath(data, chunk), "w") do io
                    for index in start:stop
                        write_c_f64(io, (projection(values[index]),))
                    end
                end
                push!(chunk_paths, name * "/" * chunk)
            end
        end
        # Chunk paths remain in the byte manifest file inventory.  The HB
        # catalog intentionally has the closed `{path,metadata}` envelope.
        push!(datasets, hb_dataset_entry(name, shape, chunks))
        push!(manifest_datasets, dataset_entry(name, chunk_paths))
    end
    # A trace's storage role is its declared trace ID.  `trace:<id>` is only
    # an internal routing prefix and must not leak into its manifest identity.
    manifest_sha = hb_manifest(staging, local_role, root_rel, manifest_datasets)
    artifact = Dict{String,Any}("id" => local_role, "path" => root_rel, "sha256" => manifest_sha,
        "media_type" => "application/vnd+zarr-v2", "file_manifest" => replace(root_rel, r"\.zarr$" => ".manifest.json"),
        "dtype" => "complex128", "shape" => shape, "chunks" => chunks,
        "complex_storage" => "paired_float64_real_imag", "group_metadata" => Dict("zarr_format" => 2),
        "datasets" => datasets, "axes" => axis_metadata, "unit" => unit, "dimensionality" => dimensionality,
        "chunk_policy" => ndims(values) == 3 ? "frequency_slab_full_matrix_v1" : ndims(values) == 2 ? "single_complete_array_v1" : "frequency_capped_1024_v1")
    return artifact
end

function hb_channels(coordinates::Vector{String}, modes::AbstractVector{<:Tuple})
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

"""Normalize pinned JC [port×mode, port×mode, frequency] response arrays.

JC orders modes independently of SCNSim's canonical degree/tuple order and
uses exp(+iwt) photon-flux coordinates.  The exact one-conjugation plus W
normalization is applied before any de-embedding or selected View work.
"""
function hb_normalize_native(array::Array{ComplexF64,3}, solver_modes, solver_w,
        response_modes::AbstractVector{<:Tuple}, pumps::Vector{Float64}, ports::Int)
    solver = Dict{Tuple,Int}(Tuple(Int.(mode)) => index for (index, mode) in enumerate(solver_modes))
    length(solver) == length(solver_modes) || fail("execution", "backend_protocol", "protocol", "protocol", "pinned JosephsonCircuits repeats a mode tuple")
    solver_mode(mode) = isempty(pumps) && isempty(mode) ? (0,) : mode
    all(haskey(solver, solver_mode(mode)) for mode in response_modes) ||
        fail("execution", "backend_protocol", "protocol", "protocol", "pinned JosephsonCircuits omitted a declared response mode")
    mode_count = length(response_modes)
    size(array) == (ports * length(solver_modes), ports * length(solver_modes), length(solver_w)) ||
        fail("execution", "backend_protocol", "protocol", "protocol", "pinned JosephsonCircuits response array shape is malformed")
    normalized = zeros(ComplexF64, length(solver_w), ports * mode_count, ports * mode_count)
    for frequency in eachindex(solver_w), output_port in 1:ports, output_mode in 1:mode_count,
            input_port in 1:ports, input_mode in 1:mode_count
        output_tuple = response_modes[output_mode]; input_tuple = response_modes[input_mode]
        output_signed = solver_w[frequency] + 2pi * hb_mode_frequency(output_tuple, pumps)
        input_signed = solver_w[frequency] + 2pi * hb_mode_frequency(input_tuple, pumps)
        isfinite(output_signed) && isfinite(input_signed) && output_signed != 0.0 && input_signed != 0.0 ||
            fail("validation", "port_realizability", "compile", "compile", "HB normalized response has a zero or non-finite signed frequency")
        source_output = (output_port - 1) * length(solver_modes) + solver[solver_mode(output_tuple)]
        source_input = (input_port - 1) * length(solver_modes) + solver[solver_mode(input_tuple)]
        target_output = (output_port - 1) * mode_count + output_mode
        target_input = (input_port - 1) * mode_count + input_mode
        normalized[frequency, target_output, target_input] = sqrt(abs(output_signed)) * conj(array[source_output, source_input, frequency]) / sqrt(abs(input_signed))
    end
    return normalized
end

"""HB-only checked solve for the response-formation case boundary.

Direct's `checked_solve` correctly turns failures into request-level Direct
failures.  Once a sealed HB case has reached its selected response formation,
the Contract instead classifies only numerical linear-solve/residual failures
as that case's `response_formation` outcome.  Malformed shapes and protocol
identity are checked before this helper and therefore still propagate.
"""
function hb_checked_solve(A::Matrix{ComplexF64}, B, label::String, n::Int)
    finite_matrix(A) && finite_matrix(B) ||
        throw(HBCaseNumericalFailure("response_formation", "HB " * label * " has non-finite linear data"))
    X = try
        A \ B
    catch error
        throw(hb_numeric_exception(error, "response_formation"))
    end
    residual = backward_residual(A, X, B)
    isfinite(residual) && residual <= tau(n) ||
        throw(HBCaseNumericalFailure("response_formation", "HB " * label * " exceeded the normalized backward-residual contract"))
    return X
end

"""Form the one de-embedded selected HB network from normalized native Z."""
function hb_selected_native(native_z, view::RealizedView, modes)
    boundary = selected_boundary(view); compiled = view.compiled
    mode_identity = Matrix{Float64}(I, length(modes), length(modes))
    A = kron(boundary.A, mode_identity)
    all(iszero, imag.(boundary.Qk)) ||
        fail("execution", "backend_protocol", "hb_case", "hb_case", "HB selected coordinate projection is not exactly real")
    coordinate_q = real.(boundary.Qk)
    Q = kron(coordinate_q, mode_identity)
    R = kron(complex.(compiled.R), Matrix{ComplexF64}(I, length(modes), length(modes)))
    Go = kron(boundary.Go, Matrix{ComplexF64}(I, length(modes), length(modes)))
    Rk = kron(complex.(boundary.Rk), Matrix{ComplexF64}(I, length(modes), length(modes)))
    Dk = kron(complex.(boundary.Dk), Matrix{ComplexF64}(I, length(modes), length(modes)))
    original_count = size(native_z, 2); selected_count = size(A, 1)
    nfrequency = size(native_z, 1)
    selected_s = Array{ComplexF64}(undef, nfrequency, selected_count, selected_count)
    selected_z = similar(selected_s); selected_y = similar(selected_s)
    identity_original = Matrix{ComplexF64}(I, original_count, original_count)
    identity_selected = Matrix{ComplexF64}(I, selected_count, selected_count)
    Rinv = hb_checked_solve(R, identity_original, "native source de-embedding", original_count)
    Rkinv = hb_checked_solve(Rk, identity_selected, "reference-matrix inversion", selected_count)
    for index in 1:nfrequency
        # JC's port resistors are source boundary elements.  Remove all of
        # them first, then apply View PTC/transforms/retain in their declared
        # original-Port space and finally re-add/subtract the selected source.
        source_admittance = hb_checked_solve(native_z[index, :, :], identity_original, "native source admittance", original_count)
        intrinsic = source_admittance - Rinv
        # Form the selected source boundary in full original-Port space before
        # projecting it.  `A * H * A'` loses the response of eliminated Ports
        # for a retained subset, so it is not a valid Schur/source realization.
        # This is the Port-space equivalent of Direct's full-node Bk solve.
        H = intrinsic + Go
        W = H + transpose(A) * Rkinv * A
        X = hb_checked_solve(W, transpose(complex.(A)), "selected source boundary", original_count)
        source_loaded_impedance = A * X
        source_admittance = hb_checked_solve(source_loaded_impedance, identity_selected,
            "selected source de-embedding", selected_count)
        selected_y[index, :, :] .= source_admittance - Rkinv
        selected_z[index, :, :] .= hb_checked_solve(selected_y[index, :, :], identity_selected, "selected Y-to-Z", selected_count)
        selected_s[index, :, :] .= hb_checked_solve(identity_selected + Dk * selected_y[index, :, :] * Dk,
            identity_selected - Dk * selected_y[index, :, :] * Dk, "selected Y-to-S", selected_count)
    end
    return selected_s, selected_y, selected_z, Q, coordinate_q
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
                fail("execution", "compiler_invariant", "hb_case", "hb_case", "HB private-node source provenance collides")
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
            fail("execution", "compiler_invariant", "hb_case", "hb_case", "HB state node lacks sealed source provenance")
        end
        push!(entries, Dict("state_index" => index - 1, "compiler_node_id" => node, "source" => source))
    end
    return entries
end

function hb_effective_source_vectors(sources, modes::AbstractVector{<:Tuple}, compiled::CompiledPrimitive)
    values = zeros(ComplexF64, length(modes), length(compiled.nodes))
    for source in sources
        # The private rank-one `(0,)` adapter is never public evidence for a
        # no-pump request: its one static member is the public rank-zero `()`
        # operating-point tuple.
        public_mode = all(isempty, modes) && source.mode == (0,) ? () : source.mode
        mode_index = findfirst(==(public_mode), modes)
        current = source.current
        if mode_index === nothing
            # A negative public tuple is represented by the conjugate
            # positive RFFT member.  This is the SCNSim physical coefficient
            # at that retained member, distinct from the JC source convention
            # applied by `hb_backend_sources`.
            opposite = Tuple(-entry for entry in public_mode)
            mode_index = findfirst(==(opposite), modes)
            mode_index === nothing &&
                fail("execution", "backend_protocol", "protocol", "protocol", "HB effective source lacks a sealed operating-point representative")
            current = conj(current)
        end
        for node in eachindex(compiled.nodes)
            values[mode_index::Int, node] += current * compiled.B[node, source.port]
        end
    end
    return values
end

function hb_state_values(nonlinear, public_modes, compiled::CompiledPrimitive)
    backend_mode(mode) = isempty(mode) ? (0,) : mode
    solver_modes = Dict{Tuple,Int}(Tuple(Int.(mode)) => index for (index, mode) in enumerate(nonlinear.modes))
    all(haskey(solver_modes, backend_mode(Tuple(Int.(item["mode"])))) for item in public_modes) ||
        fail("execution", "backend_protocol", "protocol", "protocol", "pinned JosephsonCircuits omitted an operating-point mode")
    non_ground = String[node for node in nonlinear.nodes if node != "0"]
    length(nonlinear.nodeflux) == length(nonlinear.modes) * length(non_ground) ||
        fail("execution", "backend_protocol", "protocol", "protocol", "pinned JosephsonCircuits nodeflux shape is malformed")
    packed = reshape(ComplexF64.(nonlinear.nodeflux), length(nonlinear.modes), length(non_ground))
    values = zeros(ComplexF64, length(public_modes), length(compiled.nodes))
    for (mode_index, item) in enumerate(public_modes), (node_index, _) in enumerate(compiled.nodes)
        backend_node = "n" * string(node_index)
        backend_index = findfirst(==(backend_node), non_ground)
        backend_index === nothing && fail("execution", "backend_protocol", "protocol", "protocol", "pinned JosephsonCircuits omitted a compiled physical node")
        public_mode = Tuple(Int.(item["mode"]))
        solver_index = solver_modes[backend_mode(public_mode)]
        values[mode_index, node_index] = HB_PHI0 * conj(packed[solver_index, backend_index::Int])
    end
    return values
end

function hb_lineage_prefix_sha(lineage, reason::String)
    original = lineage["original"]
    terminal = String.(original["port_order"])
    # This is the identity of the *last comparable* step, not the first
    # non-comparable one.  Keep the normal Runtime terminal convention while
    # reconstructing each canonical prefix: transforms without retain() still
    # use original logical Port terminals, never generated node names.
    if reason in ("reference_matrix", "normalization", "signed_frequency_grid")
        value = get(lineage, "lineage_sha256", nothing)
        value isa AbstractString ||
            fail("execution", "compiler_invariant", "hb_case", "hb_case", "HB lineage lacks its canonical identity")
        return String(value)
    end
    ptc = reason in ("reference_plane", "channel_basis") ? lineage["ptc"] : nothing
    transforms = reason == "channel_basis" ? copy(lineage["transforms"]) : Any[]
    retain = nothing
    prefix = Dict{String,Any}("type" => "network_view_lineage", "original" => original,
        "ptc" => ptc, "transforms" => transforms, "retain" => retain,
        "terminal_coordinates" => terminal,
        "port_realizable" => Bool(original["port_realizable"]))
    prefix["lineage_sha256"] = sha256_hex(canonical_bytes(prefix))
    return prefix["lineage_sha256"]
end

"""Return the sealed coordinate producer of a comparable HB boundary."""
function hb_coordinate_producer_sha(lineage)
    retained = lineage["retain"]
    if retained !== nothing
        q_matrix = get(retained, "q_matrix", nothing)
        q_matrix isa AbstractDict ||
            fail("execution", "compiler_invariant", "hb_case", "hb_case", "HB comparable retain lineage lacks Q evidence")
        value = get(q_matrix, "sha256", nothing)
        value isa AbstractString ||
            fail("execution", "compiler_invariant", "hb_case", "hb_case", "HB comparable retain Q evidence lacks identity")
        return String(value)
    end
    value = get(lineage["original"], "compiled_graph_sha256", nothing)
    value isa AbstractString ||
        fail("execution", "compiler_invariant", "hb_case", "hb_case", "HB comparable original graph lacks identity")
    return String(value)
end

"""Encode the base coordinate map in deterministic row-major order.

`Q` below is expanded over sideband modes only for the native-to-selected
matrix product.  Durable coordinate provenance remains the unreplicated
physical View map, so a result does not make sideband array position an
identity authority.
"""
function hb_coordinate_projection(coordinate_q::AbstractMatrix{<:Real})
    all(isfinite, coordinate_q) ||
        fail("execution", "backend_protocol", "hb_case", "hb_case", "HB comparable coordinate projection is non-finite")
    return Dict{String,Any}(
        "shape" => [size(coordinate_q, 1), size(coordinate_q, 2)],
        "values_f64" => String[f64_hex(Float64(coordinate_q[row, column]))
            for row in axes(coordinate_q, 1) for column in axes(coordinate_q, 2)],
    )
end

"""Multiply Q*N*Qᵀ with an explicit fixed summation order for evidence."""
function hb_project_native(Q::AbstractMatrix{<:Real}, native::AbstractMatrix{ComplexF64})
    size(Q, 2) == size(native, 1) == size(native, 2) ||
        fail("execution", "backend_protocol", "hb_case", "hb_case", "HB comparable native matrix disagrees with its channel projection")
    projected = zeros(ComplexF64, size(Q, 1), size(Q, 1))
    # `a,b` are explicitly the inner summation axes.  This is intentionally
    # not BLAS multiplication: the scalar order is part of reproducible
    # diagnostic evidence and is mirrored by the Python integrity verifier.
    for row in axes(Q, 1), column in axes(Q, 1)
        value = 0.0 + 0.0im
        for a in axes(Q, 2), b in axes(Q, 2)
            value += Q[row, a] * native[a, b] * Q[column, b]
        end
        projected[row, column] = value
    end
    return projected
end

"""Return the normalized matrix infinity residual with fixed row sums."""
function hb_reconciliation_residual(selected::AbstractMatrix{ComplexF64}, projected::AbstractMatrix{ComplexF64})::Float64
    size(selected) == size(projected) ||
        fail("execution", "backend_protocol", "hb_case", "hb_case", "HB comparable selected and projected matrices have different shapes")
    numerator = 0.0
    denominator = 0.0
    for row in axes(selected, 1)
        numerator_row = 0.0
        denominator_row = 0.0
        for column in axes(selected, 2)
            numerator_row += abs(selected[row, column] - projected[row, column])
            denominator_row += abs(selected[row, column]) + abs(projected[row, column])
        end
        numerator = max(numerator, numerator_row)
        denominator = max(denominator, denominator_row)
    end
    (!isfinite(numerator) || !isfinite(denominator)) && return Inf
    return denominator == 0.0 ? (numerator == 0.0 ? 0.0 : Inf) : numerator / denominator
end

function hb_reconciliation(lineage, selected_s, native_s, Q, coordinate_q)
    original_ports = String.(lineage["original"]["port_order"])
    retained = lineage["retain"]
    plain_port_subset = retained !== nothing && lineage["ptc"] === nothing &&
        isempty(lineage["transforms"]) &&
        all(value -> value in original_ports, String.(retained["retained_coordinates"]))
    reason = if lineage["ptc"] !== nothing
        "load_or_ptc"
    elseif !isempty(lineage["transforms"])
        "reference_plane"
    elseif retained !== nothing && !plain_port_subset
        "channel_basis"
    else
        nothing
    end
    if reason !== nothing
        ancestor = hb_lineage_prefix_sha(lineage, reason)
        return Dict("comparable" => false, "reason" => reason,
            "last_comparable_ancestor" => ancestor,
            "normalization" => "backend_photon_flux_to_scnsim_power_wave",
            "evidence_sha256" => sha256_hex(canonical_bytes(Dict("reason" => reason, "last_comparable_ancestor" => ancestor))))
    end
    values = Float64[]
    for index in axes(selected_s, 1)
        projected = hb_project_native(Q, native_s[index, :, :])
        push!(values, hb_reconciliation_residual(selected_s[index, :, :], projected))
    end
    residual = maximum(values)
    isfinite(residual) || fail("execution", "backend_protocol", "hb_case", "hb_case", "HB comparable reconciliation is non-finite")
    ancestor = String(lineage["lineage_sha256"])
    projection = hb_coordinate_projection(coordinate_q)
    return Dict("comparable" => true, "reason" => nothing,
        "last_comparable_ancestor" => ancestor,
        "normalization" => "backend_photon_flux_to_scnsim_power_wave", "coordinate_projection" => projection,
        "residual_f64" => f64_hex(residual),
        "evidence_sha256" => sha256_hex(canonical_bytes(Dict(
            "coordinate_producer_sha256" => hb_coordinate_producer_sha(lineage),
            "coordinate_projection" => projection,
            "residual_f64" => f64_hex(residual)))))
end

"""Record the fixed topology split used by every case in an HB batch.

The original compiled graph and original-only lineage own nonlinear balance;
PTC is never applied there.  Response formation owns the full selected
lineage, where an explicit PTC is compensated at every sideband.
"""
function hb_topology_evidence(request)
    lineage = request["ref_lineage"]
    original = lineage["original"]
    intrinsic = get(original, "compiled_graph_sha256", nothing)
    intrinsic isa AbstractString ||
        fail("execution", "compiler_invariant", "compile", "compile", "HB lineage lacks the intrinsic compiled-graph identity")
    ptc = lineage["ptc"]
    return Dict{String,Any}(
        "allow_driven_ptc" => Bool(request["spec"]["allow_driven_ptc"]),
        "intrinsic_compiled_graph_sha256" => String(intrinsic),
        "nonlinear_balance" => Dict(
            "load_state" => "loaded",
            "lineage_sha256" => hb_lineage_prefix_sha(lineage, "load_or_ptc"),
        ),
        "response_linearization" => Dict(
            "load_state" => ptc === nothing ? "raw" : "compensated",
            "lineage_sha256" => String(lineage["lineage_sha256"]),
        ),
    )
end

"""Validate the complete HB realization without allocating an attempt/solver.

`SCNSimBackend.preflight` owns the process entry point and may call this for
`solve_hb`.  This helper deliberately executes the same sealed lattice,
lowering, representative-source and driven-PTC checks as `solve_hb`, but no
nonlinear/linearized numerical solve and no artifact write.  It is therefore
safe to use before workspace attempt allocation.
"""
function hb_preflight(request, plan, raw_compiled::CompiledPrimitive, view::RealizedView)
    hb_require_runtime!()
    request["runtime_semantic"]["algorithm_id"] == HB_ALGORITHM_ID ||
        fail("execution", "compiler_invariant", "runtime", "runtime", "HB request has a mismatched algorithm identity")
    spec = request["spec"]
    get(spec, "type", nothing) == "hb_solve" ||
        fail("execution", "compiler_invariant", "compile", "compile", "hb_preflight requires an hb_solve Spec")
    view.port_realizable ||
        fail("validation", "port_realizability", "direct_response", "direct_response", "HB response requires a Port-realizable final View")
    lattice, pumps, pump_modes, response_modes, grid, _ = hb_lattice(spec)
    circuit = hb_lower_raw(raw_compiled, plan)
    parsed = JosephsonCircuits.parsesortcircuit(circuit; sorting = :none)
    parsed.Nnodes > 1 ||
        fail("execution", "compiler_invariant", "compile", "compile", "HB backend has no non-ground state coordinates")
    backend_modes = isempty(pumps) ? Tuple[(0,)] : pump_modes
    for case in spec["cases"]
        sources, effective = hb_sources(case, spec, raw_compiled)
        # This computes (and therefore validates) every generated conjugate,
        # representative tuple/index, and coefficient exactly once in the
        # same helper that result serialization uses.
        hb_effective_source_evidence(effective, backend_modes, isempty(pumps))
        if request["ref_lineage"]["ptc"] !== nothing && !Bool(spec["allow_driven_ptc"])
            any(!iszero, hb_effective_source_vectors(sources, pump_modes, raw_compiled)) &&
                fail("validation", "port_realizability", "compile", "compile", "driven PTC requires allow_driven_ptc=True")
        end
    end
    return Dict{String,Any}(
        "lattice" => lattice,
        "response_frequency_count" => length(grid),
        "lowered_element_count" => length(circuit),
        "topology_evidence" => hb_topology_evidence(request),
    )
end

"""Terminal HB entry point.  Case numerical errors never poison sibling cases."""
function solve_hb(request, plan, raw_compiled::CompiledPrimitive, view::RealizedView, request_sha::String, attempt_sha::String, staging::String)
    hb_require_runtime!()
    request["runtime_semantic"]["algorithm_id"] == HB_ALGORITHM_ID ||
        fail("execution", "compiler_invariant", "runtime", "runtime", "HB request has a mismatched algorithm identity")
    spec = request["spec"]
    get(spec, "type", nothing) == "hb_solve" || fail("execution", "compiler_invariant", "compile", "compile", "solve_hb requires an hb_solve Spec")
    view.port_realizable || fail("validation", "port_realizability", "direct_response", "direct_response", "HB response requires a Port-realizable final View")
    lattice, pumps, pump_modes, response_modes, grid, declared_dc = hb_lattice(spec)
    circuit = hb_lower_raw(raw_compiled, plan)
    # PTC never removes matched loads from nonlinear balance.  A driven PTC
    # request is nonetheless an explicit public authorization because its
    # response-only compensation is easy to misread as an unloaded operating
    # point.  This is complete-request preflight, before any case can write
    # a staging artifact.
    if request["ref_lineage"]["ptc"] !== nothing && !Bool(spec["allow_driven_ptc"])
        for case in spec["cases"]
            case_sources, _ = hb_sources(case, spec, raw_compiled)
            any(!iszero, hb_effective_source_vectors(case_sources, pump_modes, raw_compiled)) &&
                fail("validation", "port_realizability", "compile", "compile", "driven PTC requires allow_driven_ptc=True")
        end
    end
    outcomes = Dict{String,Any}[]
    for (ordinal, case) in enumerate(spec["cases"])
        sources, effective = hb_sources(case, spec, raw_compiled)
        evidence_backend_modes = isempty(pumps) ? Tuple[(0,)] : pump_modes
        effective = hb_effective_source_evidence(effective, evidence_backend_modes, isempty(pumps))
        try
            nonlinear, linearized, native_s_raw, native_z_raw, operating_point_closure = hb_run_case(circuit, spec, pumps, pump_modes, response_modes, grid, sources, declared_dc)
            native_s = hb_normalize_native(native_s_raw, linearized.modes, linearized.w, response_modes, pumps, length(raw_compiled.port_ids))
            native_z = hb_normalize_native(native_z_raw, linearized.modes, linearized.w, response_modes, pumps, length(raw_compiled.port_ids))
            expected_native = length(raw_compiled.port_ids) * length(response_modes)
            size(native_s, 1) == length(grid) && size(native_s, 2) == expected_native && size(native_s, 3) == expected_native && size(native_z) == size(native_s) ||
                fail("execution", "backend_protocol", "protocol", "protocol", "pinned JosephsonCircuits returned native HB arrays with the wrong shape")
            selected_s, selected_y, selected_z, Q, coordinate_q = try
                hb_selected_native(native_z, view, response_modes)
            catch error
                # Compiler/shape/protocol failures are request failures.  A
                # case outcome is reserved for the bounded numerical stages
                # named by the HB contract, never a blanket rescue path.
                error isa BackendFailure && rethrow()
                throw(hb_numeric_exception(error, "response_formation"))
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
            state_values = nonlinear === nothing ? zeros(ComplexF64, 0, length(raw_compiled.nodes)) : hb_state_values(nonlinear, node_modes, raw_compiled)
            operating_modes = Tuple[Tuple(Int.(item["mode"])) for item in node_modes]
            source_values = hb_effective_source_vectors(sources, operating_modes, raw_compiled)
            state_axes = [Dict("id" => "pump_mode", "kind" => "pump_mode", "values" => [item["mode"] for item in node_modes]), Dict("id" => "node_coordinate", "kind" => "node_coordinate", "values" => raw_compiled.nodes)]
            artifacts["states"] = hb_write_complex(staging, ordinal, "states", state_values, state_axes, "weber", "magnetic_flux"; matrix = false)
            artifacts["effective_source_vectors"] = hb_write_complex(staging, ordinal, "effective_source_vectors", source_values, state_axes, "ampere", "current"; matrix = false)
            traces = Dict{String,Any}[]
            selected_channels = hb_channels(selected_coordinates, response_modes)
            for trace in spec["traces"]
                input = findfirst(==(Dict("coordinate" => trace["input_port"], "mode" => trace["input_mode"])), selected_channels)
                output = findfirst(==(Dict("coordinate" => trace["output_port"], "mode" => trace["output_mode"])), selected_channels)
                (input === nothing || output === nothing) && fail("execution", "backend_protocol", "hb_case", "hb_case", "HB trace is absent from selected channel basis")
                value = vec(selected_s[:, output::Int, input::Int])
                push!(traces, hb_write_complex(staging, ordinal, "trace:" * String(trace["id"]), value, [Dict("id" => "frequency", "kind" => "frequency", "request_field" => "spec.frequencies")], "dimensionless", "dimensionless"; matrix = false))
            end
            state_map = hb_state_map(plan, raw_compiled)
            reconciliation = hb_reconciliation(request["ref_lineage"], selected_s, native_s, Q, coordinate_q)
            state = hb_case_state(source_values, operating_modes)
            push!(outcomes, Dict("case_ordinal" => ordinal, "case_id" => case["id"], "status" => "success", "bias_state" => state[1], "pump_state" => state[2], "effective_sources" => effective, "artifacts" => artifacts, "traces" => traces, "reconciliation" => reconciliation, "operating_point_closure" => operating_point_closure, "backend_normalization_evidence_sha256" => sha256_hex(canonical_bytes(Dict("normalization" => "backend_photon_flux_to_scnsim_power_wave"))), "state_node_map" => state_map))
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
        "lattice" => lattice, "truncation" => spec["truncation"],
        "topology_evidence" => hb_topology_evidence(request), "cases" => outcomes,
    )
    write_hb_success(staging, request, request_sha, attempt_sha, result)
end
