const DEFAULTS_FILE = joinpath(@__DIR__, "defaults.json")
Base.include_dependency(DEFAULTS_FILE)
const DEFAULTS = JSON.parsefile(DEFAULTS_FILE)

const DEFAULT_MAX_ITERATIONS = Int(DEFAULTS["termination"]["max_iterations"])
const DEFAULT_GRADIENT_TERMINATION_TOLERANCE =
    Float64(DEFAULTS["termination"]["gradient_termination_tolerance"])
const DEFAULT_MAX_TIME = Float64(DEFAULTS["termination"]["max_time"])
const DEFAULT_STEP_SIZE_LIMIT = Float64(DEFAULTS["termination"]["step_size_limit"])
const DEFAULT_MINIMUM_OBJECTIVE_FUNCTION =
    Float64(DEFAULTS["termination"]["minimum_objective_function"])
const DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS =
    Int(DEFAULTS["termination"]["iterative_refinement_max_iterations"])

function validate_common_termination_values(
    max_iterations::Int64,
    gradient_tolerance::Float64,
    max_time::Float64,
    step_size_limit::Float64,
    minimum_objective_function::Float64,
    iterative_refinement_max_iterations::Int64,
)
    @assert max_iterations > 0
    @assert isfinite(gradient_tolerance) && gradient_tolerance > 0
    @assert isfinite(max_time) && max_time > 0
    @assert isfinite(step_size_limit) && step_size_limit > 0
    @assert isfinite(minimum_objective_function)
    @assert 0 <= iterative_refinement_max_iterations <= 3
    return nothing
end

function validate_common_termination_values(
    max_iterations::Int64,
    gradient_tolerance::Float64,
    max_time::Float64,
    step_size_limit::Float64,
    iterative_refinement_max_iterations::Int64,
)
    return validate_common_termination_values(
        max_iterations,
        gradient_tolerance,
        max_time,
        step_size_limit,
        DEFAULT_MINIMUM_OBJECTIVE_FUNCTION,
        iterative_refinement_max_iterations,
    )
end
