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
	"The algorithm stopped because the trust region radius is too small."
	TRUST_REGION_RADIUS_LIMIT
    "The algorithm stopped because it encountered unrecoverable numerical error."
    NUMERICAL_ERROR
    "The algorithm stopped because of an error not covered by one of the statuses defined above."
    OTHER_ERROR
end

mutable struct TerminationConditions
	MAX_ITERATIONS::Int64
	gradient_termination_tolerance::Float64
	MAX_TIME::Float64
	MINIMUM_TRUST_REGION_RADIUS::Float64
	MINIMUM_OBJECTIVE_FUNCTION::Float64

	function TerminationConditions(MAX_ITERATIONS::Int64=100000, gradient_termination_tolerance::Float64=1e-5,
		MAX_TIME::Float64=30 * 60.0, MINIMUM_TRUST_REGION_RADIUS::Float64=1e-40, MINIMUM_OBJECTIVE_FUNCTION::Float64=-1e30)
		@assert(MAX_ITERATIONS > 0)
		@assert(gradient_termination_tolerance > 0)
        @assert(MAX_TIME > 0)
		@assert(MINIMUM_TRUST_REGION_RADIUS > 0)
		return new(MAX_ITERATIONS, gradient_termination_tolerance, MAX_TIME, MINIMUM_TRUST_REGION_RADIUS, MINIMUM_OBJECTIVE_FUNCTION)
	end
end

mutable struct INITIAL_RADIUS_STRUCT
	r_1::Float64
	INITIAL_RADIUS_MULTIPLICATIVE_RULEE::Int64
	function INITIAL_RADIUS_STRUCT(r_1::Float64=0.0, INITIAL_RADIUS_MULTIPLICATIVE_RULEE::Int64=10)
		@assert(INITIAL_RADIUS_MULTIPLICATIVE_RULEE > 0)
		return new(r_1, INITIAL_RADIUS_MULTIPLICATIVE_RULEE)
	end
end

termination_conditions_struct_default = TerminationConditions()
initial_radius_struct_default = INITIAL_RADIUS_STRUCT()

mutable struct Problem_Data
    nlp::Union{AbstractNLPModel, MathOptInterface.NLPBlockData, Nothing}
	termination_conditions_struct::TerminationConditions
	initial_radius_struct::INITIAL_RADIUS_STRUCT
	β_1::Float64
    θ::Float64
    ω_1::Float64
	ω_2::Float64
	γ_1::Float64
	γ_2::Float64
	print_level::Int64
	compute_ρ_hat_approach::String
	radius_update_rule_approach::String
    # initialize parameters
    function Problem_Data(nlp::Union{AbstractNLPModel, MathOptInterface.NLPBlockData, Nothing}=nothing, termination_conditions_struct::TerminationConditions=termination_conditions_struct_default,
						  initial_radius_struct::INITIAL_RADIUS_STRUCT=initial_radius_struct_default, β_1::Float64=0.1,
						  θ::Float64=0.1, ω_1::Float64=8.0, ω_2::Float64=20.0, γ_1::Float64=0.01, γ_2::Float64=0.8,
						  print_level::Int64=0, compute_ρ_hat_approach::String="DEFAULT", radius_update_rule_approach::String="DEFAULT")
		@assert(β_1 > 0 && β_1 < 1)
        @assert(θ >= 0 && θ < 1)
        @assert(ω_1 >= 1)
		@assert(ω_2 >= 1)
		@assert(ω_2 >= ω_1)
		γ_3 = 1.0 # //TODO Make param
		@assert(0 <= γ_1 < 0.5 * ( 1 - ((β_1 * θ) / (γ_3 * (1 - β_1)))))
		@assert(1/ω_1 < γ_2 <= 1)
        return new(nlp, termination_conditions_struct, initial_radius_struct, β_1, θ, ω_1, ω_2, γ_1, γ_2, print_level, compute_ρ_hat_approach, radius_update_rule_approach)
    end
end
