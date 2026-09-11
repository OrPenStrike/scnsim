# Included into the single SCNSimBackend module.
# Request bootstrap, terminal dispatch, compiler audit, and preflight.

function leaf_plan_path(request_path::String)::String
    request_dir = dirname(abspath(request_path))
    basename(request_dir) == "" && error("request path is malformed")
    return joinpath(dirname(dirname(request_dir)), "plan.json")
end

function read_request_and_plan(request_path::String)
    request_bytes = read(request_path)
    request_sha = sha256_hex(request_bytes)
    request = plain(JSON3.read(String(request_bytes)))
    get(request, "schema", nothing) == "scnsim.request" && get(request, "schema_version", nothing) == 2 || error("request schema/version is invalid")
    exact_keys(request, ("schema", "schema_version", "plan_sha256", "operation", "view", "spec", "parameter_source", "runtime_semantic")) || error("request fields are invalid")
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
    attempt["julia_threads"] == 1 && attempt["blas_threads"] == 1 || error("attempt thread evidence violates the single-thread execution policy")
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
        source_kind = String(request["parameter_source"]["kind"])
        if source_kind != "point"
            operation == "optimize_direct" && fail("execution", "compiler_invariant", "parameter_source", "compile", "optimization requires one fixed point parameter source")
            run_parameter_batch(request, plan, request_sha, attempt_sha, staging)
            return nothing
        end
        compile_context = operation == "evaluate_direct" ? "direct_quantity" : "compile"
        raw_compiled = compile_primitive(plan, parameter_values(request); context_kind = compile_context,
            authorized = parameter_set_authorizations(request), authorization_source = "parameter_set")
        realized_lineage, view = realized_ref_lineage(raw_compiled, declarative_lineage(plan, request, raw_compiled))
        request["ref_lineage"] = realized_lineage
        compiled = view.compiled
        if operation == "solve_direct"
            solve_direct(request, view, request_sha, attempt_sha, staging)
        elseif operation == "solve_hb"
            # HB lowers from the raw normalized graph so Josephson rows and
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
            fail("capability", "scaffold_unavailable", operation, "scaffold", "operation is not supported by this backend revision")
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

"""Compile one immutable ResolvedPlanPoint without a Run, View, or workspace."""
function compiler_audit(plan_path::String, point_path::String)
    isabspath(plan_path) && isabspath(point_path) || error("compiler audit paths must be absolute")
    plan_bytes = read(plan_path); point_bytes = read(point_path)
    plan_sha = sha256_hex(plan_bytes)
    plan = plain(JSON3.read(String(plan_bytes))); point = plain(JSON3.read(String(point_bytes)))
    get(plan, "schema", nothing) == "scnsim.plan" && get(plan, "schema_version", nothing) == 2 ||
        fail("execution", "compiler_invariant", "compiler_audit", "compile", "compiler audit Plan schema/version is invalid")
    get(point, "schema", nothing) == "scnsim.resolved_plan_point" && get(point, "schema_version", nothing) == 2 ||
        fail("execution", "compiler_invariant", "compiler_audit", "compile", "resolved point schema/version is invalid")
    exact_keys(point, ("schema", "schema_version", "plan_sha256", "parameters", "parameters_sha256", "resolved_fields")) ||
        fail("execution", "compiler_invariant", "compiler_audit", "compile", "resolved point fields are invalid")
    String(point["plan_sha256"]) == plan_sha ||
        fail("execution", "compiler_invariant", "compiler_audit", "compile", "resolved point does not bind the supplied Plan bytes")
    parameters = plain(point["parameters"])
    parameter_values = structured_parameter_values(parameters)
    definition_keys = String[ref_key(definition) for definition in plan["parameter_closure"]["definitions"]]
    length(definition_keys) == length(unique(definition_keys)) ||
        fail("execution", "compiler_invariant", "compiler_audit", "compile", "Plan parameter closure repeats a definition")
    Set(keys(parameter_values)) == Set(definition_keys) ||
        fail("execution", "compiler_invariant", "compiler_audit", "compile", "resolved point parameters do not exactly cover the Plan closure")
    parameters_sha = sha256_hex(canonical_bytes(Dict{String,Any}(
        "schema" => "scnsim.parameter_point_identity", "schema_version" => 2, "parameters" => parameters,
    )))
    String(point["parameters_sha256"]) == parameters_sha ||
        fail("execution", "compiler_invariant", "compiler_audit", "compile", "resolved point parameter identity is invalid")
    compiled, rows = structured_compile(plan, Dict{String,Any}(); context_kind = "compile",
        emit_audit = true, resolved_rows = point["resolved_fields"])
    load = port_load_admittance(compiled)
    return Dict{String,Any}(
        "schema" => "scnsim.compiler_audit", "schema_version" => 2,
        "plan_sha256" => plan_sha, "parameters_sha256" => parameters_sha,
        "node_order" => compiled.nodes, "matrix_order" => "canonical_node_id",
        "resolved_bindings" => rows, "expanded_branch_rows" => compiled.branch_rows,
        "c_matrix" => f64_matrix_evidence(compiled.C), "k_matrix" => f64_matrix_evidence(compiled.K),
        "g_matrix" => f64_matrix_evidence(compiled.G),
        "ports" => Dict(
            "ids" => compiled.port_ids, "selector" => f64_matrix_evidence(compiled.B),
            "reference_matrix" => f64_matrix_evidence(compiled.R), "load_mask_f64" => f64_hex.(compiled.M),
            "load_stamp" => f64_matrix_evidence(load),
            "selected_network_steps" => ["intrinsic_CKG", "port_load_BY0BT", "source_boundary", "power_wave_deembedding"],
        ),
    )
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

function declarative_lineage(plan, request, compiled::CompiledPrimitive; view_declaration = nothing)
    view = plain(view_declaration === nothing ? request["view"] : view_declaration)
    get(view, "type", nothing) == "network_view" || fail("execution", "compiler_invariant", "view", "compile", "request View discriminator is invalid")
    exact_keys(view, ("type", "ptc", "transforms", "retain")) || fail("execution", "compiler_invariant", "view", "compile", "request View fields are invalid")
    ptc = get(view, "ptc", nothing); retain = get(view, "retain", nothing); transforms = get(view, "transforms", nothing)
    (ptc === nothing || exact_keys(ptc, ("selected_ports",))) &&
        (retain === nothing || exact_keys(retain, ("retained_coordinates",))) &&
        transforms isa AbstractVector && all(item -> exact_keys(item, ("id", "input_coordinates", "output_coordinates")), transforms) ||
        fail("execution", "compiler_invariant", "view", "compile", "request View declaration is malformed")
    graph_sha = sha256_hex(canonical_bytes(Dict{String,Any}(
        "schema" => "scnsim.compiled_graph_identity", "schema_version" => 1,
        "plan_sha256" => request["plan_sha256"],
        "julia_source_sha256" => request["runtime_semantic"]["julia_source_sha256"],
    )))
    coordinate_order = String[node["compiler_node_id"] for node in plan["connectivity"]["node_coordinates"]]
    all(id -> id in compiled.nodes, coordinate_order) || fail("execution", "compiler_invariant", "view", "compile", "View coordinate basis is absent from compiled graph")
    original = Dict{String,Any}(
        "type" => "original", "compiled_graph_sha256" => graph_sha,
        "coordinate_order" => coordinate_order, "port_order" => compiled.port_ids,
        "port_realizable" => !isempty(compiled.port_ids),
    )
    return Dict{String,Any}(
        "type" => "network_view_lineage", "original" => original,
        "ptc" => ptc, "transforms" => transforms, "retain" => retain,
    )
end

function preflight(plan_path::String, request_path::String)
    plan_bytes = read(plan_path)
    plan_sha = sha256_hex(plan_bytes)
    plan = plain(JSON3.read(String(plan_bytes)))
    request = plain(JSON3.read(read(request_path, String)))
    get(request, "schema", nothing) == "scnsim.request" ||
        fail("execution", "compiler_invariant", "preflight", "compile", "preflight request schema is invalid")
    get(request, "schema_version", nothing) == 2 && exact_keys(request, ("schema", "schema_version", "plan_sha256", "operation", "view", "spec", "parameter_source", "runtime_semantic")) ||
        fail("execution", "compiler_invariant", "preflight", "compile", "preflight request version/fields are invalid")
    request["plan_sha256"] == plan_sha ||
        fail("execution", "compiler_invariant", "preflight", "compile", "preflight request does not bind the supplied Plan")
    operation = String(request["operation"])
    context = operation == "evaluate_direct" ? "direct_quantity" : "compile"
    request_values = parameter_values(request)
    raw_compiled, resolved = structured_compile(plan, request_values; context_kind = context,
        authorized = parameter_set_authorizations(request), authorization_source = "parameter_set",
        emit_audit = true)
    realized_lineage, view = realized_ref_lineage(raw_compiled, declarative_lineage(plan, request, raw_compiled))
    operation == "optimize_direct" && candidate_view_cache(plan, request, raw_compiled)
    compiled = view.compiled
    load = port_load_admittance(compiled)
    realized_request = deepcopy(request); realized_request["ref_lineage"] = realized_lineage
    hb_validation = operation == "solve_hb" ? hb_preflight(realized_request, plan, raw_compiled, view) : nothing
    runtime_path = normpath(joinpath(@__DIR__, "..", "runtime.json"))
    runtime = plain(JSON3.read(read(runtime_path, String)))
    result = Dict{String,Any}(
        "schema" => "scnsim.preflight",
        "schema_version" => 2,
        "plan_sha256" => plan_sha,
        "runtime" => runtime,
        "resolved_bindings" => resolved,
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
    hb_validation === nothing || (result["hb_preflight"] = hb_validation)
    return result
end
