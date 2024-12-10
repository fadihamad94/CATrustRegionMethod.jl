"""
	TerminationStatusCode

An Enum of possible values for the `TerminationStatus` attribute.
"""
@enumx TerminationStatusCode begin
    "The algorithm found an optimal solution."
    OPTIMAL
    "The algorithm stopped because it decided that the problem is unbounded."
    UNBOUNDED
    "An iterative algorithm stopped after conducting the maximum number of iterations."
    ITERATION_LIMIT
    "The algorithm stopped after a user-specified computation time."
    TIME_LIMIT
    "The algorithm stopped because it ran out of memory."
    MEMORY_LIMIT
    "The algorithm stopped because the step size is too small."
    STEP_SIZE_LIMIT
    "The algorithm stopped because it encountered unrecoverable numerical error."
    NUMERICAL_ERROR
    "The algorithm stopped because failure in solving the trust-region subproblem."
    TRUST_REGION_SUBPROBLEM_ERROR
    "The algorithm stopped because of an error not covered by one of the statuses defined above."
    OTHER_ERROR
    "The algorithm stopped because the model is invalid."
    INVALID_MODEL
end

"A description of trust-region subproblem termination failure reason."
struct TrustRegionSubproblemError <: Exception
    msg::String
    failure_reason_6a::Bool
    failure_reason_6b::Bool
    failure_reason_6c::Bool
    failure_reason_6d::Bool
end

"A description of solver termination criteria."
mutable struct TerminationCriteria
    """
     	If termination_reason = ITERATION_LIMIT then the solver has
     	taken at least MAX_ITERATIONS iterations.
     	"""
    MAX_ITERATIONS::Int64
    """
    Absolute tolerance on the gradient norm.
    """
    gradient_termination_tolerance::Float64
    """
    	If termination_reason = TIME_LIMIT then the solver has
    	taken at least MAX_TIME time (secodns).
    	"""
    MAX_TIME::Float64
    """
    	If termination_reason = STEP_SIZE_LIMIT then the solver has
    	taken a direction step with norm kess than STEP_SIZE_LIMIT.
    	"""
    STEP_SIZE_LIMIT::Float64
    """
    	The smallest value of the objective function that will be tolerated before the problem
    is declared to be unbounded from below.
    	"""
    MINIMUM_OBJECTIVE_FUNCTION::Float64

    function TerminationCriteria(
        MAX_ITERATIONS::Int64 = 100000,
        gradient_termination_tolerance::Float64 = 1e-5,
        MAX_TIME::Float64 = 5 * 60 * 60.0,
        STEP_SIZE_LIMIT::Float64 = 2.0e-16,
        MINIMUM_OBJECTIVE_FUNCTION::Float64 = -1e30,
    )
        @assert(MAX_ITERATIONS > 0)
        @assert(gradient_termination_tolerance > 0)
        @assert(MAX_TIME > 0)
        @assert(STEP_SIZE_LIMIT > 0)
        return new(
            MAX_ITERATIONS,
            gradient_termination_tolerance,
            MAX_TIME,
            STEP_SIZE_LIMIT,
            MINIMUM_OBJECTIVE_FUNCTION,
        )
    end
end

# termination_conditions_struct_default = TerminationCriteria()
# initial_radius_struct_default = INITIAL_RADIUS_STRUCT()

mutable struct AlgorithmicParameters
    """
    β param for the algorithm. It is a threshold for ρ_hat when updating the trust-region radius.
    """
    β::Float64
    """
    θ param for the algorithm. It is used for computing ρ_hat.
    """
    θ::Float64
    """
    ω_1 param for the algorithm. When ρ_hat < β, we set r_k = r_k / ω_1.
    """
    ω_1::Float64
    """
    ω_2 param for the algorithm. When ρ_hat ≧ β, we set r_k = max(ω_2 ||d_k||, r_k).
       Where d_k is the the search direction.
    """
    ω_2::Float64
    """
    γ_1 param for the algorithm. It is used for the trust-region subproblem termination criteria.
    """
    γ_1::Float64
    """
    γ_2 param for the algorithm. It is used for the trust-region subproblem termination criteria.
    """
    γ_2::Float64
    """
    γ_3 param for the algorithm. It is used for the trust-region subproblem termination criteria.
    """
    γ_3::Float64
    """
    ξ param for the algorithm. It is used for the trust-region subproblem termination criteria.
    """
    ξ::Float64
    """
    The required initial value of the trust-region radius.
    """
    r_1::Float64
    """
    If r_1 ≤ 0, then the radius will be choosen automatically based on a heursitic appraoch.
    The default is INITIAL_RADIUS_MULTIPLICATIVE_RULE * ||g_1|| / ||H_1|| where ||g_1|| is the
    l2 norm for gradient at the initial iterate and ||H_1|| is the spectral norm for the hessian
    at the initial iterate.
    """
    INITIAL_RADIUS_MULTIPLICATIVE_RULE::Float64
    """
    Specify seed level for randomness.
    """
    seed::Int64
    """
    The verbosity level of logs.
    """
    print_level::Int64
    """
    This to be able to test the performance of the algorithm for the ablation study
    when comparing versus the conference version of the paper.
    """
    radius_update_rule_approach::String
    """
    eval_offset param for the algorithm. It is used for the trust-region subproblem termination criteria.
    """
    eval_offset::Float64
    """
    trust_region_subproblem_solver param for the algorithm. It is used to determine which method to use the
    trust-region subproblem. Using the new appraoch or the old appraoch in the NEURips paper.
    """
    trust_region_subproblem_solver::Any
    # initialize parameters
    function AlgorithmicParameters(
        β::Float64 = 0.1,
        θ::Float64 = 0.1,
        ω_1::Float64 = 8.0,
        ω_2::Float64 = 16.0,
        γ_1::Float64 = 0.01,
        γ_2::Float64 = 0.8,
        γ_3::Float64 = 1.0,
        ξ::Float64 = 0.1,
        r_1::Float64 = 0.0,
        INITIAL_RADIUS_MULTIPLICATIVE_RULE::Float64 = 10.0,
        seed::Int64 = 1,
        print_level::Int64 = 0,
        radius_update_rule_approach::String = "DEFAULT",
        eval_offset::Float64 = 1e-8,
        trust_region_subproblem_solver::String = "NEW", #This is mainly for ablation study to compare against old approach (conference version)
    )
        @assert(β > 0 && β < 1)
        @assert(θ >= 0 && θ < 1)
        @assert(ω_1 >= 1)
        @assert(ω_2 >= 1)
        @assert(ω_2 >= ω_1)
        @assert(γ_3 > 0 && γ_3 <= 1)
        @assert(ξ > 0)
        @assert(eval_offset > 0)
        @assert(INITIAL_RADIUS_MULTIPLICATIVE_RULE > 0)
        @assert(0 <= γ_1 < 0.5 * (1 - ((β * θ) / (γ_3 * (1 - β)))))
        @assert(1 / ω_1 < γ_2 <= 1)
        return new(
            β,
            θ,
            ω_1,
            ω_2,
            γ_1,
            γ_2,
            γ_3,
            ξ,
            r_1,
            INITIAL_RADIUS_MULTIPLICATIVE_RULE,
            seed,
            print_level,
            radius_update_rule_approach,
            eval_offset,
            trust_region_subproblem_solver,
        )
    end
end

mutable struct AlgorithmCounter
    total_function_evaluation::Int64
    total_gradient_evaluation::Int64
    total_hessian_evaluation::Int64
    total_number_factorizations::Int64
    total_number_factorizations_findinterval::Int64
    total_number_factorizations_bisection::Int64
    total_number_factorizations_compute_search_direction::Int64
    total_number_factorizations_inverse_power_iteration::Int64
    function AlgorithmCounter()
        return new(0, 0, 0, 0, 0, 0, 0, 0)
    end

end
