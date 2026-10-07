#!/usr/bin/env julia
# Isolated experimental evaluator. Normal SCNSimBackend methods are never replaced.
# Whole-task mode retains Julia lowering/CMA and terminal authorization/ack flow.

using LinearAlgebra, JSON3
include(joinpath(@__DIR__, "..", "src", "SCNSimBackend.jl"))
const SB = SCNSimBackend

function decode_array(record)
    bytes = hex2bytes(record["data_hex"])
    dtype = if record["dtype"] == "complex128"
        ComplexF64
    elseif record["dtype"] == "float64"
        Float64
    else
        error("unsupported benchmark array dtype")
    end
    values = collect(reinterpret(dtype, bytes))
    shape = Int.(record["shape"])
    isempty(shape) && return values[1]
    return permutedims(reshape(values, reverse(shape)...), reverse(1:length(shape)))
end

function encode_array(value)
    array = Array(value)
    values = vec(permutedims(array, reverse(1:ndims(array))))
    return Dict("dtype" => eltype(array) <: Complex ? "complex128" : "float64",
        "shape" => collect(size(array)), "data_hex" => bytes2hex(reinterpret(UInt8, values)))
end

function timing_record(measured)
    return Dict("elapsed_ns" => round(Int,measured.time*1e9), "allocated_bytes" => measured.bytes,
        "gc_ns" => round(Int,measured.gctime*1e9),
        "compile_ns" => hasproperty(measured,:compile_time) ? round(Int,measured.compile_time*1e9) : nothing,
        "recompile_ns" => hasproperty(measured,:recompile_time) ? round(Int,measured.recompile_time*1e9) : nothing)
end

function descriptor(record)
    v = record["view"]; m = v["model"]
    blocks = Any[SB.SeriesRLBlock(String(b["id"]), decode_array(b["incidence"]),
        decode_array(b["resistance"]), decode_array(b["inductance"])) for b in m["series_rl"]]
    compiled = SB.CompiledPrimitive(String.(m["node_ids"]), decode_array(m["C"]), decode_array(m["K"]),
        decode_array(m["G"]), blocks, Dict{String,Any}[], Dict{String,Any}[], String.(m["port_ids"]),
        decode_array(m["B"]), decode_array(m["R"]), vec(decode_array(m["M"])))
    p = length(compiled.port_ids)
    selected_map = v["selected_map"] === nothing ? zeros(0, p) : decode_array(v["selected_map"])
    view = SB.RealizedView(compiled, String.(v["coordinates"]), String.(v["terminal_ids"]),
        decode_array(v["coordinate_port_map"]), Int.(v["selected_indices"]) .+ 1, selected_map, v["port_realizable"])
    boundary = v["port_realizable"] ? (Bk = decode_array(v["Bk"]), Rk = decode_array(v["Rk"]),
        Dk = decode_array(v["Dk"]), Go = decode_array(v["Go"])) : nothing
    # Share a line's impedance across its distinct pi-section incidence maps.
    groups = Any[]
    for block in blocks
        index = findfirst(g -> size(g.R) == size(block.resistance) &&
            reinterpret(UInt64,vec(g.R)) == reinterpret(UInt64,vec(block.resistance)) &&
            reinterpret(UInt64,vec(g.L)) == reinterpret(UInt64,vec(block.inductance)), groups)
        if index === nothing
            push!(groups, (R = block.resistance, L = block.inductance, incidences = [block.incidence]))
        else
            push!(groups[index].incidences, block.incidence)
        end
    end
    loaded_G = compiled.G + SB.port_load_admittance(compiled)
    return (view = view, boundary = boundary, groups = groups, loaded_G = loaded_G)
end

function checked_factor(A, kind, stage; symmetric = false, algorithm = "lu")
    SB.finite_matrix(A) || SB.fail("execution", kind, stage, "direct_quantity", "non-finite linear system")
    if symmetric
        numerator = norm(A - transpose(A), Inf)
        denominator = norm(abs.(A) + abs.(transpose(A)), Inf)
        eta = denominator == 0 ? (numerator == 0 ? 0.0 : Inf) : numerator / denominator
        isfinite(eta) && eta <= SB.tau(size(A, 1)) ||
            SB.fail("execution", kind, stage, "direct_quantity", "eliminated operator is not transpose symmetric")
    end
    try
        return symmetric && algorithm == "reuse" ? bunchkaufman(Symmetric(A, :U), false; check = true) : lu(A; check = true)
    catch error
        error isa SingularException || error isa ZeroPivotException || rethrow()
        SB.fail("execution", kind, stage, "direct_quantity", "required linear solve is singular")
    end
end

function checked_rhs(factor, A, B, kind, stage, n)
    SB.finite_matrix(B) || SB.fail("execution", kind, stage, "direct_quantity", "non-finite linear system")
    X = try
        factor \ B
    catch error
        error isa SingularException || error isa ZeroPivotException || rethrow()
        SB.fail("execution", kind, stage, "direct_quantity", "required linear solve is singular")
    end
    eta = SB.backward_residual(A, X, B)
    isfinite(eta) && eta <= SB.tau(n) || SB.fail("execution", kind, stage, "direct_quantity", "linear solve exceeded normalized backward-residual contract")
    return X, eta
end

function solve(A, B, kind, stage, n; factor = nothing)
    factor === nothing && (factor = checked_factor(A, kind, stage))
    return checked_rhs(factor, A, B, kind, stage, n)[1]
end

function operator(prepared, omega; loaded, derivative = false)
    c = prepared.view.compiled
    G = loaded ? prepared.loaded_G : c.G
    Q = complex.(c.K) - omega^2 .* c.C - im * omega .* G
    Qp = -2omega .* c.C - im .* G
    bound = abs.(c.K) + abs2(omega) .* abs.(c.C) + abs(omega) .* abs.(G)
    for block in prepared.groups
        impedance = complex.(block.R) - im * omega .* block.L
        inverse = solve(impedance, Matrix{ComplexF64}(I, size(impedance)...), "direct_response_formation", "series_rl", size(impedance, 1))
        for A in block.incidences
            contribution = -im * omega .* (A * inverse * transpose(A))
            Q += contribution
            if derivative
                Qp += A * (-im .* inverse + omega .* (inverse * block.L * inverse)) * transpose(A)
                bound += abs.(contribution)
            end
        end
    end
    return Q, Qp, bound
end

function selected_state(prepared, omega, algorithm)
    view = prepared.view; c = view.compiled
    retained = view.selected_indices
    eliminated = [i for i in eachindex(c.nodes) if i ∉ retained]
    Q, Qp, bound = operator(prepared, omega; loaded = true, derivative = true)
    F, Fp = Q[retained, retained], Qp[retained, retained]
    X = zeros(ComplexF64, 0, length(retained)); Xp = similar(X); eta_e = 0.0
    if !isempty(eliminated)
        A, rhs = Q[eliminated, eliminated], Q[eliminated, retained]
        factor = checked_factor(A, "eliminated_block_solve_failure", "eliminated_block"; symmetric = true, algorithm = algorithm)
        X, eta_x = checked_rhs(factor, A, rhs, "eliminated_block_solve_failure", "eliminated_block", length(eliminated))
        Xp, eta_xp = checked_rhs(factor, A, Qp[eliminated, retained] - Qp[eliminated, eliminated] * X,
            "eliminated_block_solve_failure", "derivative_eliminated_block", length(eliminated))
        F -= Q[retained, eliminated] * X
        Fp -= Qp[retained, eliminated] * X + Q[retained, eliminated] * Xp
        eta_e = max(eta_x, eta_xp)
    end
    return (F=F,Fp=Fp,Q=Q,Qp=Qp,bound=bound,X=X,Xp=Xp,eta_e=eta_e,eliminated=eliminated)
end

function element_state(prepared, omega, coordinate, algorithm)
    state = selected_state(prepared, omega, algorithm)
    c = prepared.view.compiled; row = prepared.view.selected_indices[coordinate]
    Q, Qp, bound = state.Q, state.Qp, state.bound
    eliminated = state.eliminated
    x = zeros(ComplexF64, length(c.nodes)); x[row] = 1
    scale = abs(Qp[row,row])
    if !isempty(eliminated)
        x[eliminated] = -state.X[:,coordinate]
        scale += sum(abs.(Qp[row,eliminated]) .* abs.(state.X[:,coordinate])) + sum(abs.(Q[row,eliminated]) .* abs.(state.Xp[:,coordinate]))
    end
    f, fp = state.F[coordinate,coordinate], state.Fp[coordinate,coordinate]
    rows = vcat([row],eliminated); bx = bound*abs.(x)
    num, den = norm((Q*x)[rows],Inf), norm(bx[rows],Inf)
    eta_q = den == 0 ? (num == 0 ? 0.0 : Inf) : num/den
    eta_f = bx[row] == 0 ? (f == 0 ? 0.0 : Inf) : abs(f)/bx[row]
    return (f=f,fp=fp,eta_e=state.eta_e,eta_q=eta_q,eta_f=eta_f,
        correction=fp == 0 ? Inf : abs(f/fp)/abs(omega),normalized_slope=scale == 0 ? Inf : abs(fp)/scale,scale=scale)
end


function root(prepared, record, algorithm)
    hint = decode_array(record["root_hint_hz"])
    isfinite(hint) && hint > 0 || SB.fail("validation", "invalid_diagonal_root_hint", "root_hint", "direct_quantity", "root_hint must be finite and strictly positive")
    omega = record["omega_start_rad_s"] === nothing ? complex(2pi*hint) : decode_array(record["omega_start_rad_s"])
    isfinite(real(omega)) && isfinite(imag(omega)) || SB.fail("execution", "numerical_resolution_unresolved", "newton", "direct_quantity", "root initialization is non-finite")
    coordinate = Int(record["coordinate_index"]) + 1
    steps = 0
    for step in 1:32
        state = element_state(prepared, omega, coordinate, algorithm)
        state.fp == 0 && SB.fail("execution", "root_slope_unresolved", "newton", "direct_quantity", "selected element derivative is zero")
        candidate = omega - state.f/state.fp
        same = reinterpret(UInt64, real(candidate)) == reinterpret(UInt64, real(omega)) && reinterpret(UInt64, imag(candidate)) == reinterpret(UInt64, imag(omega))
        omega = candidate; steps = step
        same && break
    end
    state = element_state(prepared, omega, coordinate, algorithm)
    t = SB.tau(length(prepared.view.compiled.nodes))
    isfinite(real(omega)) && isfinite(imag(omega)) && real(omega) > 0 &&
        state.eta_e <= t && state.eta_q <= t && state.eta_f <= t && state.correction <= t ||
        SB.fail("execution", "numerical_resolution_unresolved", "newton_certificate", "direct_quantity", "selected element Newton procedure did not reach its machine-resolution certificate")
    state.normalized_slope > t || SB.fail("execution", "root_slope_unresolved", "slope_certificate", "direct_quantity", "selected element local slope is unresolved")
    imag(omega) <= 0 || SB.fail("execution", "numerical_resolution_unresolved", "newton_certificate", "direct_quantity", "diagonal root violates the passive imaginary-root policy")
    return Dict("root_omega_rad_s" => encode_array(fill(omega)), "root_slope" => encode_array(fill(state.fp)),
        "evidence" => Dict("newton_steps" => steps, "certificate" => [isfinite(x) ? Dict("f64"=>SB.f64_hex(x)) : string(x) for x in (state.eta_e, state.eta_q, state.eta_f, state.correction, state.normalized_slope, state.scale)]))
end

function network(prepared, omega, family)
    view = prepared.view; c = view.compiled; b = prepared.boundary
    view.port_realizable || SB.fail("validation", "port_realizability", "selected_network", "direct_response", "Direct response requires a Port-realizable final View")
    n, p = length(c.nodes), length(view.terminal)
    Q = operator(prepared, omega; loaded = false)[1]
    H = Q/(-im*omega) + c.B*b.Go*transpose(c.B)
    Rinv = solve(complex.(b.Rk), Matrix{ComplexF64}(I,p,p), "direct_response_formation", "reference_matrix", p)
    W = H + b.Bk*Rinv*transpose(b.Bk)
    factor = checked_factor(W, "direct_response_formation", "source_solve")
    X = solve(W, complex.(b.Bk), "direct_response_formation", "source_solve", n; factor = factor)
    Zsrc = transpose(b.Bk)*X
    Y = solve(Zsrc, Matrix{ComplexF64}(I,p,p), "direct_response_formation", "source_admittance", p) - Rinv
    Z = family in ("all", "Z") ? solve(Y, Matrix{ComplexF64}(I,p,p), "direct_response_formation", "y_to_z", p) : nothing
    S = nothing
    if family in ("all", "S")
        P, N = Matrix{ComplexF64}(I,p,p) + b.Dk*Y*b.Dk, Matrix{ComplexF64}(I,p,p) - b.Dk*Y*b.Dk
        S = solve(P, N, "direct_response_formation", "y_to_s", p)
        D = complex.(b.Dk); dfactor = checked_factor(D, "direct_response_formation", "reference_matrix")
        source = 2 .* b.Bk*solve(D, Matrix{ComplexF64}(I,p,p), "direct_response_formation", "reference_matrix", p; factor = dfactor)
        voltage = solve(W, source, "direct_response_formation", "source_solve", n; factor = factor)
        source_s = solve(D, transpose(b.Bk)*voltage, "direct_response_formation", "deembedding", p; factor = dfactor) - Matrix{ComplexF64}(I,p,p)
        eta = norm(source_s-S,Inf)/(1+norm(source_s,Inf)+norm(S,Inf))
        isfinite(eta) && eta <= SB.tau(n) || SB.fail("execution", "direct_response_formation", "deembedding", "direct_response", "source-boundary and de-embedded selected-network responses disagree")
    end
    SB.finite_matrix(Y) && (S === nothing || SB.finite_matrix(S)) && (Z === nothing || SB.finite_matrix(Z)) || SB.fail("execution", "direct_response_formation", "response_formation", "direct_response", "selected N-port response is non-finite")
    return S,Y,Z
end

function evaluate(record, algorithm)
    prepared = descriptor(record)
    kind = record["kind"]
    kind == "diagonal_root" && return root(prepared, record, algorithm)
    kind in ("direct", "response_element") || error("unsupported benchmark numerical operation")
    frequencies = vec(decode_array(record["frequencies_hz"]))
    family = kind == "direct" ? "all" : record["family"]
    p = length(prepared.view.terminal)
    arrays = [Array{ComplexF64}(undef,length(frequencies),p,p) for _ in 1:3]
    for (i,f) in enumerate(frequencies)
        matrices = if kind == "response_element" && !prepared.view.port_realizable && family in ("Y","Z")
            state = selected_state(prepared,complex(2pi*f),algorithm)
            Y = state.F/(-im*2pi*f)
            Z = family == "Z" ? solve(Y,Matrix{ComplexF64}(I,size(Y)...),"direct_response_formation","y_to_z",size(Y,1)) : nothing
            (nothing,Y,Z)
        else
            network(prepared, complex(2pi*f), family)
        end
        for j in 1:3
            matrices[j] === nothing || (arrays[j][i,:,:] = matrices[j])
        end
    end
    kind == "direct" && return Dict("S" => encode_array(arrays[1]), "Y" => encode_array(arrays[2]), "Z" => encode_array(arrays[3]))
    axis = Dict("S"=>1,"Y"=>2,"Z"=>3)[family]
    value = arrays[axis][1,Int(record["output_index"])+1,Int(record["input_index"])+1]
    return Dict("response_value" => encode_array(fill(value)))
end

function serve(algorithm, blas_threads)
    BLAS.set_num_threads(blas_threads)
    println(SB.canonical_json(Dict("schema"=>"scnsim.benchmark_backend_ready", "julia_version"=>string(VERSION),
        "julia_threads"=>Threads.nthreads(), "blas_threads"=>BLAS.get_num_threads(), "algorithm"=>algorithm)))
    flush(stdout)
    for line in eachline(stdin)
        frame = SB.plain(JSON3.read(line))
        SB.canonical_json(frame) == line || error("benchmark batch is not canonical")
        get(frame,"event",nothing) == "close" && break
        results = Any[]
        for record in frame["jobs"]
            measured = @timed try
                Dict("id"=>record["id"], "values"=>evaluate(record,algorithm))
            catch error
                error isa SB.BackendFailure || rethrow()
                Dict("id"=>record["id"], "failure"=>Dict("kind"=>error.kind,"stage"=>error.stage,"detail"=>error.message))
            end
            result = measured.value; result["timing"] = timing_record(measured)
            if haskey(result,"values")
                evidence = get!(result["values"],"evidence",Dict{String,Any}())
                v = record["view"]
                evidence["axes"] = Dict("original_node_ids"=>v["original_node_ids"],
                    "model_node_ids"=>v["model"]["node_ids"],"terminal_ids"=>v["terminal_ids"],
                    "selected_indices"=>v["selected_indices"])
            end
            push!(results,result)
        end
        println(SB.canonical_json(Dict("schema"=>"scnsim.benchmark_backend_result", "batch_id"=>frame["batch_id"], "results"=>results)))
        flush(stdout)
    end
end

function whole(request_path, staging, declaration_path, observation_path, cpu_threads)
    declaration_bytes = read(declaration_path)
    declaration = SB.plain(JSON3.read(String(copy(declaration_bytes))))
    SB.canonical_bytes(declaration) == declaration_bytes || error("benchmark declaration is not canonical")
    SB.sha256_hex(read(request_path)) == declaration["source_analysis_sha256"] || error("whole-task source request does not bind benchmark declaration")
    profile = declaration["benchmark"]
    cpu_threads in profile["cpu_threads"] || error("whole-task physical quota is absent from benchmark declaration")
    requested_julia = get(profile, "julia_threads", nothing)
    requested_blas = get(profile, "julia_blas_threads", nothing)
    julia_threads = requested_julia === nothing ? cpu_threads : Int(requested_julia)
    julia_blas_threads = requested_blas === nothing ? cpu_threads : Int(requested_blas)
    BLAS.set_num_threads(julia_blas_threads)
    mesh = declaration["benchmark"]["mesh"]
    observations = Any[]
    function sections(path, values, dynamic)
        mesh["kind"] == "dynamic" && return dynamic
        rows = mesh["sections"]
        if mesh["kind"] == "grouped"
            key = mesh["parameter_key"]
            value = values[String(key[1])*"\u001e"*String(key[2])]
            found = findfirst(g -> (g["lower_inclusive"] ? value >= SB.f64_from_hex(g["lower_f64"]) : value > SB.f64_from_hex(g["lower_f64"])) &&
                (g["upper_inclusive"] ? value <= SB.f64_from_hex(g["upper_f64"]) : value < SB.f64_from_hex(g["upper_f64"])), mesh["groups"])
            found === nothing && error("candidate is outside declared benchmark mesh groups")
            rows = mesh["groups"][found]["sections"]
        end
        found = findfirst(row -> String.(row[1]) == path, rows)
        found === nothing && return dynamic
        return Int(rows[found][2])
    end
    context = (cpu_threads = cpu_threads, julia_threads = julia_threads, julia_blas_threads = julia_blas_threads,
        mesh = sections, observe = (stage, measured) -> push!(observations, Dict("stage"=>stage,"timing"=>timing_record(measured),
        "node_count"=>length(measured.value.nodes),"discretization"=>measured.value.discretization)))
    try
        measured = @timed SB.run_terminal(request_path, staging; benchmark_context = context)
        push!(observations,Dict("stage"=>"whole_terminal","timing"=>timing_record(measured)))
    finally
        payload = Dict("schema"=>"scnsim.benchmark_whole_timing", "benchmark_request_sha256"=>SB.sha256_hex(declaration_bytes),
            "mesh"=>mesh,"observations"=>observations,"julia_version"=>string(VERSION),
            "cpu_threads_requested"=>cpu_threads, "julia_threads_requested"=>julia_threads,
            "julia_blas_threads_requested"=>julia_blas_threads,
            "julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads())
        SB.write_bytes(observation_path,SB.canonical_bytes(payload))
    end
end

if length(ARGS) == 3 && ARGS[1] == "--numerical"
    ARGS[2] in ("reuse","lu") || error("unknown numerical algorithm")
    serve(ARGS[2],parse(Int,ARGS[3]))
elseif length(ARGS) in (5,6) && ARGS[1] == "--whole"
    whole(ARGS[2],ARGS[3],ARGS[4],ARGS[5],length(ARGS) == 6 ? parse(Int,ARGS[6]) : 1)
else
    error("usage: --numerical reuse|lu blas_threads | --whole request.json staging benchmark-request.json timing.json [cpu_threads]")
end
