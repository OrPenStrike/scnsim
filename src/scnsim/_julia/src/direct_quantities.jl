# Included into the single SCNSimBackend module.
# Direct solves and typed Direct quantity terminal operations.

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
    baseline_raw = compile_primitive(plan, baseline_values; context_kind = "direct_quantity",
        authorized = parameter_set_authorizations(request), authorization_source = "parameter_set")
    _, baseline_view = realized_ref_lineage(baseline_raw, declarative_lineage(plan, request, baseline_raw))
    baseline_root, baseline_slope = diagonal_root(baseline_view.compiled, String(spec["coordinate"]), hint)
    if same_parameter_values(baseline_values, candidate_values)
        omega, slope = baseline_root, baseline_slope
    else
        selector = Dict{String,Any}(
            "type" => "diagonal_root_projection",
            "spec" => spec,
            "projection" => "frequency",
        )
        omega = selector_root_with_continuation(
            plan, request, baseline_values, candidate_values, baseline_root, selector;
            context_kind = "direct_quantity",
        )
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
        _, baseline_view = realized_ref_lineage(raw_base, declarative_lineage(plan, request, raw_base))
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
        _, baseline_view = realized_ref_lineage(raw_base, declarative_lineage(plan, request, raw_base))
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
        _, baseline_view = realized_ref_lineage(raw_base, declarative_lineage(plan, request, raw_base))
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
