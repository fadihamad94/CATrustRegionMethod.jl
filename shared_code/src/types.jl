"""
Generate a solver-local `TerminationStatusCode` EnumX module.

The generated type remains owned by the calling solver module so existing
qualified status types and values remain compatible.
"""
macro define_termination_status()
    return esc(
        quote
            EnumX.@enumx TerminationStatusCode begin
                OPTIMAL
                UNBOUNDED
                ITERATION_LIMIT
                TIME_LIMIT
                MEMORY_LIMIT
                STEP_SIZE_LIMIT
                NUMERICAL_ERROR
                TRUST_REGION_SUBPROBLEM_ERROR
                OTHER_ERROR
                INVALID_MODEL
            end
        end,
    )
end

"""
Generate the existing CAT or UTR termination-criteria record in the caller.
"""
macro define_termination_criteria(variant)
    variant_value = variant isa QuoteNode ? variant.value : variant
    if variant_value == :cat
        return esc(
            quote
                mutable struct TerminationCriteria
                    MAX_ITERATIONS::Int64
                    gradient_termination_tolerance::Float64
                    MAX_TIME::Float64
                    STEP_SIZE_LIMIT::Float64
                    MINIMUM_OBJECTIVE_FUNCTION::Float64
                    iterative_refinement_max_iterations::Int64

                    function TerminationCriteria(
                        MAX_ITERATIONS::Int64 = DEFAULT_MAX_ITERATIONS,
                        gradient_termination_tolerance::Float64 =
                            DEFAULT_GRADIENT_TERMINATION_TOLERANCE,
                        MAX_TIME::Float64 = DEFAULT_MAX_TIME,
                        STEP_SIZE_LIMIT::Float64 = DEFAULT_STEP_SIZE_LIMIT,
                        MINIMUM_OBJECTIVE_FUNCTION::Float64 =
                            DEFAULT_MINIMUM_OBJECTIVE_FUNCTION,
                        iterative_refinement_max_iterations::Int64 =
                            DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS,
                    )
                        CATrustRegionShared.validate_common_termination_values(
                            MAX_ITERATIONS,
                            gradient_termination_tolerance,
                            MAX_TIME,
                            STEP_SIZE_LIMIT,
                            MINIMUM_OBJECTIVE_FUNCTION,
                            iterative_refinement_max_iterations,
                        )
                        return new(
                            MAX_ITERATIONS,
                            gradient_termination_tolerance,
                            MAX_TIME,
                            STEP_SIZE_LIMIT,
                            MINIMUM_OBJECTIVE_FUNCTION,
                            iterative_refinement_max_iterations,
                        )
                    end
                end
            end,
        )
    elseif variant_value == :utr
        return esc(
            quote
                mutable struct TerminationCriteria
                    MAX_ITERATIONS::Int64
                    MAX_INNER_ITERATIONS::Int64
                    gradient_termination_tolerance::Float64
                    MAX_TIME::Float64
                    STEP_SIZE_LIMIT::Float64
                    MINIMUM_OBJECTIVE_FUNCTION::Float64
                    iterative_refinement_max_iterations::Int64

                    function TerminationCriteria(
                        MAX_ITERATIONS::Int64 = DEFAULT_MAX_ITERATIONS,
                        MAX_INNER_ITERATIONS::Int64 = DEFAULT_MAX_INNER_ITERATIONS,
                        gradient_termination_tolerance::Float64 =
                            DEFAULT_GRADIENT_TERMINATION_TOLERANCE,
                        MAX_TIME::Float64 = DEFAULT_MAX_TIME,
                        STEP_SIZE_LIMIT::Float64 = DEFAULT_STEP_SIZE_LIMIT,
                        MINIMUM_OBJECTIVE_FUNCTION::Float64 =
                            DEFAULT_MINIMUM_OBJECTIVE_FUNCTION,
                        iterative_refinement_max_iterations::Int64 =
                            DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS,
                    )
                        CATrustRegionShared.validate_common_termination_values(
                            MAX_ITERATIONS,
                            gradient_termination_tolerance,
                            MAX_TIME,
                            STEP_SIZE_LIMIT,
                            MINIMUM_OBJECTIVE_FUNCTION,
                            iterative_refinement_max_iterations,
                        )
                        @assert MAX_INNER_ITERATIONS > 0
                        return new(
                            MAX_ITERATIONS,
                            MAX_INNER_ITERATIONS,
                            gradient_termination_tolerance,
                            MAX_TIME,
                            STEP_SIZE_LIMIT,
                            MINIMUM_OBJECTIVE_FUNCTION,
                            iterative_refinement_max_iterations,
                        )
                    end
                end
            end,
        )
    end
    error("@define_termination_criteria expects :cat or :utr")
end

function canonical_status_string(
    status;
    memory_limit::String = "MEMORY_LIMIT",
)::String
    name = try
        Symbol(status)
    catch
        return string(status)
    end
    name == :MEMORY_LIMIT && return memory_limit
    name in (
        :OPTIMAL,
        :UNBOUNDED,
        :ITERATION_LIMIT,
        :INNER_ITERATION_LIMIT,
        :TIME_LIMIT,
        :STEP_SIZE_LIMIT,
        :NUMERICAL_ERROR,
        :TRUST_REGION_SUBPROBLEM_ERROR,
        :OTHER_ERROR,
        :INVALID_MODEL,
    ) && return String(name)
    return string(status)
end
