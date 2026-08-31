#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "src", "SCNSimBackend.jl"))
using .SCNSimBackend

function usage()
    error("usage: scnsim_request.jl --request <absolute request.json> --staging <absolute staging-dir> | --preflight <absolute plan.json> --request <absolute request.json>")
end

function preflight_frame(plan::String, request_path::String)
    try
        return SCNSimBackend.preflight(plan, request_path)
    catch failure
        failure isa SCNSimBackend.BackendFailure || rethrow()
        request = SCNSimBackend.plain(SCNSimBackend.JSON3.read(read(request_path, String)))
        return Dict{String,Any}(
            "schema" => "scnsim.preflight_failure",
            "schema_version" => 1,
            "failure" => SCNSimBackend.failure_object(request, failure),
        )
    end
end

function main(arguments)
    if length(arguments) == 4 && arguments[1] == "--preflight" && arguments[3] == "--request"
        plan = abspath(arguments[2])
        request = abspath(arguments[4])
        isfile(plan) || error("preflight plan.json does not exist")
        isfile(request) || error("preflight request.json does not exist")
        println(SCNSimBackend.canonical_json(preflight_frame(plan, request)))
        return
    end
    length(arguments) == 4 || usage()
    arguments[1] == "--request" && arguments[3] == "--staging" || usage()
    request = abspath(arguments[2])
    staging = abspath(arguments[4])
    isfile(request) || error("request.json does not exist")
    isdir(staging) || error("attempt staging directory does not exist")
    SCNSimBackend.run_terminal(request, staging)
end

main(ARGS)
