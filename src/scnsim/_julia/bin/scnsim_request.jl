#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "src", "SCNSimBackend.jl"))
using .SCNSimBackend

function usage()
    error("usage: scnsim_request.jl --request <absolute request.json> --staging <absolute staging-dir> | --preflight <absolute plan.json>")
end

function main(arguments)
    if length(arguments) == 2 && arguments[1] == "--preflight"
        plan = abspath(arguments[2])
        isfile(plan) || error("preflight plan.json does not exist")
        println(SCNSimBackend.canonical_json(SCNSimBackend.preflight(plan)))
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
