module SCNSimBackend

using CMAEvolutionStrategy
using JSON3
using JosephsonCircuits
using LinearAlgebra
using Random
using SHA
using Unicode

const EPS64 = eps(Float64)

"""A typed backend failure that becomes a request-scoped outcome envelope."""
struct BackendFailure <: Exception
    category::String
    kind::String
    stage::String
    context_kind::String
    message::String
end

Base.showerror(io::IO, failure::BackendFailure) = print(io, failure.message)

fail(category, kind, stage, context, message) =
    throw(BackendFailure(category, kind, stage, context, message))

is_candidate_failure(error::BackendFailure) = error.kind in (
    "invalid_candidate_physical_parameter",
    "eliminated_block_solve_failure",
    "root_slope_unresolved",
    "numerical_resolution_unresolved",
)

# These files are one ordered implementation, not public submodules or fallback paths.
# Their order preserves the original definition and call graph.
include("wire.jl")
include("compile.jl")
include("view.jl")
include("quantity_core.jl")
include("result_artifacts.jl")
include("direct_quantities.jl")
include("terminal.jl")
include("optimization.jl")

# The HB surface is deliberately included after the shared compiler, selected
# network realization, artifact writer, and terminal protocol are defined.
# It consumes those authorities; it does not create a second graph/runtime.
include("hb.jl")
include("batch_v2.jl")

end # module
