__precompile__()
#Main algorithm code goes here
module consistently_adaptive_trust_region_method
using NLPModels, LinearAlgebra, DataFrames, SparseArrays, EnumX
include("./trust_region_subproblem_solver.jl")

export TerminationConditions, INITIAL_RADIUS_STRUCT, Problem_Data
export phi, findinterval, bisection, computeSecondOrderModel, optimizeSecondOrderModel, compute_ρ_hat, CAT

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

	function TerminationConditions(MAX_ITERATIONS::Int64=10000, gradient_termination_tolerance::Float64=1e-5,
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
	function INITIAL_RADIUS_STRUCT(r_1::Float64, INITIAL_RADIUS_MULTIPLICATIVE_RULEE::Int64=10)
		@assert(INITIAL_RADIUS_MULTIPLICATIVE_RULEE > 0)
		return new(r_1, INITIAL_RADIUS_MULTIPLICATIVE_RULEE)
	end
end

mutable struct Problem_Data
    nlp::AbstractNLPModel
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
    function Problem_Data(nlp::AbstractNLPModel, termination_conditions_struct::TerminationConditions,
						  initial_radius_struct::INITIAL_RADIUS_STRUCT, β_1::Float64=0.1,
						  θ::Float64=0.1, ω_1::Float64=4.0, ω_2::Float64=20.0, γ_1::Float64=0.01, γ_2::Float64=0.2,
						  print_level::Int64=0, compute_ρ_hat_approach::String="DEFAULT", radius_update_rule_approach::String="DEFAULT")
		@assert(β_1 > 0 && β_1 < 1)
        @assert(θ >= 0 && θ < 1)
        @assert(ω_1 >= 1)
		@assert(ω_2 >= 1)
		@assert(ω_2 >= ω_1)
		γ_3 = 1.0 # //TODO Make param
		@assert(0 <= γ_1 < 0.5 * ( 1 - ((β_1 * θ) / (γ_3 * (1 - β_1)))))
		@assert(1/ω_1 < (1 - γ_2) <= 1)
		# @assert(1>= γ_2 > (1 / ω_1))
        return new(nlp, termination_conditions_struct, initial_radius_struct, β_1, θ, ω_1, ω_2, γ_1, γ_2, print_level, compute_ρ_hat_approach, radius_update_rule_approach)
    end
end

function computeSecondOrderModel(f::Float64, g::Vector{Float64}, H, d_k::Vector{Float64})
    return transpose(g) * d_k + 0.5 * transpose(d_k) * H * d_k
end

function compute_ρ_hat(fval_current::Float64, fval_next::Float64, gval_current::Vector{Float64}, gval_next::Vector{Float64}, H, d_k::Vector{Float64}, θ::Float64, min_gval_norm::Float64, print_level::Int64=0, approach::String="DEFAULT")
    second_order_model_value_current_iterate = computeSecondOrderModel(fval_current,  gval_current, H, d_k)
	guarantee_factor = θ * 0.5 * min(norm(gval_current, 2), norm(gval_next, 2)) * norm(d_k, 2)
	if approach != "DEFAULT"
		guarantee_factor = θ * 0.5 * norm(gval_next, 2) * norm(d_k, 2)
	end
	actual_fct_decrease = fval_current - fval_next
	predicted_fct_decrease = - second_order_model_value_current_iterate
	ρ_hat = actual_fct_decrease / (predicted_fct_decrease + guarantee_factor)
	if print_level >= 1 && ρ_hat == -Inf || isnan(ρ_hat)
		println("ρ_hat is $ρ_hat. actual_fct_decrease is $actual_fct_decrease, predicted_fct_decrease is $predicted_fct_decrease, and guarantee_factor is $guarantee_factor.")
	end
    return ρ_hat, actual_fct_decrease, predicted_fct_decrease, guarantee_factor
end

function compute_ρ_standard_trust_region_method(fval_current::Float64, fval_next::Float64, gval_current::Vector{Float64}, H, d_k::Vector{Float64}, print_level::Int64=0)
    second_order_model_value_current_iterate = computeSecondOrderModel(fval_current, gval_current, H, d_k)
	actual_fct_decrease = fval_current - fval_next
	predicted_fct_decrease = - second_order_model_value_current_iterate
	ρ = actual_fct_decrease / predicted_fct_decrease
	if print_level >= 1 && ρ == -Inf || isnan(ρ)
		println("ρ is $ρ. actual_fct_decrease is $actual_fct_decrease and predicted_fct_decrease is $predicted_fct_decrease.")
	end
    return ρ, actual_fct_decrease, predicted_fct_decrease
end

function sub_routine_trust_region_sub_problem_solver(fval_current, gval_current, hessian_current, x_k, δ_k, γ_1, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_method, print_level)
	fval_next = fval_current
	gval_next_temp = gval_current
	start_time_temp = time()
	temp_total_function_evaluation = 0

	# Solve the trust-region subproblem to generate the search direction d_k
	# success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, hard_case = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp.meta.name, subproblem_solver_method, print_level)
	success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, hard_case, temp_total_number_factorizations_findinterval, temp_total_number_factorizations_bisection, temp_total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp.meta.name, subproblem_solver_method, print_level)
	if success_subproblem_solve
		q_1 = norm(hessian_current * d_k + gval_current + δ_k * d_k)
		q_2 = γ_1 * min_gval_norm
		if q_1 > q_2
			success_subproblem_solve = false
			@warn "q_1: $q_1 is larger than q_2: $q_2."
		end
	end
	end_time_temp = time()
	total_time_temp = end_time_temp - start_time_temp
	if print_level >= 2
		println("solveTrustRegionSubproblem operation took $total_time_temp.")
	end

	start_time_temp = time()
	second_order_model_value_current_iterate = computeSecondOrderModel(fval_current, gval_current, hessian_current, d_k)
	end_time_temp = time()
	total_time_temp = end_time_temp - start_time_temp
	if print_level >= 2
		println("computeSecondOrderModel operation took $total_time_temp.")
	end

	# When we are able to solve the trust-region subproblem, we check for numerical error
	# in computing the predicted reduction from the second order model M_k. If no numerical
	# errors, we compute the objective function for the candidate solution to check if
	# we will accept the step in case this leads to reduction in the function value.
	if success_subproblem_solve && second_order_model_value_current_iterate < 0
		start_time_temp = time()
		fval_next = obj(nlp, x_k + d_k)
		temp_total_function_evaluation += 1
		end_time_temp = time()
		total_time_temp = end_time_temp - start_time_temp
		if print_level >= 2
			println("fval_next operation took $total_time_temp.")
		end
	end

	# return fval_next, success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, hard_case
	return fval_next, success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, hard_case, temp_total_number_factorizations_findinterval, temp_total_number_factorizations_bisection, temp_total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration
end

function CAT(problem::Problem_Data, x::Vector{Float64}, δ::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
    @assert(δ >= 0)
	#Termination conditions
	termination_conditions_struct = problem.termination_conditions_struct
    MAX_ITERATIONS = termination_conditions_struct.MAX_ITERATIONS
    MAX_TIME = termination_conditions_struct.MAX_TIME
    gradient_termination_tolerance = termination_conditions_struct.gradient_termination_tolerance
	MINIMUM_TRUST_REGION_RADIUS = termination_conditions_struct.MINIMUM_TRUST_REGION_RADIUS
	MINIMUM_OBJECTIVE_FUNCTION = termination_conditions_struct.MINIMUM_OBJECTIVE_FUNCTION

	#Algorithm parameters
    β_1 = problem.β_1
    ω_1 = problem.ω_1
	ω_2 = problem.ω_2
	γ_1 = problem.γ_1
	γ_2 = problem.γ_2
	γ_3 = 1.0 # //TODO Make param
	θ = problem.θ
	C = (2 + 3 * γ_3 * (1 - β_1)) / (3 * (γ_3 * (1 - 2 * γ_1) * (1 - β_1) - β_1 * θ))
	ξ = 0.1# //TODO Make param
	@assert ξ >= 1 / (6 * C)
	#Initial radius
	initial_radius_struct = problem.initial_radius_struct
	r_1 = initial_radius_struct.r_1
	INITIAL_RADIUS_MULTIPLICATIVE_RULEE = initial_radius_struct.INITIAL_RADIUS_MULTIPLICATIVE_RULEE

	#Initial conditions
    x_k = x
    δ_k = δ
    r_k = r_1

    nlp = problem.nlp

	print_level = problem.print_level
	compute_ρ_hat_approach = problem.compute_ρ_hat_approach
	radius_update_rule_approach = problem.radius_update_rule_approach

	#Algorithm history
	iteration_stats = DataFrame(k = [], fval = [], gradval = [])

	#Algorithm stats
	total_function_evaluation = 0
    total_gradient_evaluation = 0
    total_hessian_evaluation = 0
    total_number_factorizations = 0
	total_number_factorizations_findinterval = 0
	total_number_factorizations_bisection = 0
	total_number_factorizations_compute_search_direction = 0
	total_number_factorizations_inverse_power_iteration = 0

    k = 1
    try
        gval_current = grad(nlp, x_k)
        fval_current = obj(nlp, x_k)
        total_function_evaluation += 1
        total_gradient_evaluation += 1
		hessian_current = hess(nlp, x_k)
		total_hessian_evaluation += 1

		#If user doesn't change the starting radius, we select the radius as described in the paper:
		#Initial radius heuristic selection rule : r_1 = 10 * ||gval_current|| / ||hessian_current||
		if r_k <= 0.0
			r_k = INITIAL_RADIUS_MULTIPLICATIVE_RULEE * norm(gval_current, Inf) / norm(hessian_current, Inf)
		end

        compute_hessian = false
        if norm(gval_current, 2) <= gradient_termination_tolerance
            # computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
			total_number_factorization = 1
			computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations, "total_number_factorizations_findinterval" => total_number_factorizations_findinterval, "total_number_factorizations_bisection" => total_number_factorizations_bisection, "total_number_factorizations_compute_search_direction" => total_number_factorizations_compute_search_direction, "total_number_factorizations_inverse_power_iteration" => total_number_factorizations_inverse_power_iteration)
			if print_level >= 0
            	println("*********************************Iteration Count: ", 1)
			end
			push!(iteration_stats, (1, fval_current, norm(gval_current, 2)))
			return x_k, TerminationStatusCode.OPTIMAL, iteration_stats, computation_stats, 1
        end

        start_time = time()
		min_gval_norm = norm(gval_current, 2)
        while k <= MAX_ITERATIONS
			@assert total_number_factorizations == total_number_factorizations_findinterval + total_number_factorizations_bisection + total_number_factorizations_compute_search_direction + total_number_factorizations_inverse_power_iteration
			temp_grad = gval_current
			if print_level >= 1
				start_time_str = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
				println("$start_time. Iteration $k with radius $r_k and total_number_factorizations $total_number_factorizations.")
			end
            if compute_hessian
				start_time_temp = time()
                hessian_current = hess(nlp, x_k)
				total_hessian_evaluation += 1
				end_time_temp = time()
				total_time_temp = end_time_temp - start_time_temp
				if print_level >= 2
					println("hessian_current operation took $total_time_temp.")
				end
            end

			# Solve the trsut-region subproblem and generate the search direction d_k
			# fval_next, success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, hard_case = sub_routine_trust_region_sub_problem_solver(fval_current, gval_current, hessian_current, x_k, δ_k, γ_1, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_method, print_level)
			# total_number_factorizations += temp_total_number_factorizations
			# total_function_evaluation += temp_total_function_evaluation
			fval_next, success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, hard_case, temp_total_number_factorizations_findinterval, temp_total_number_factorizations_bisection, temp_total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration = sub_routine_trust_region_sub_problem_solver(fval_current, gval_current, hessian_current, x_k, δ_k, γ_1, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_method, print_level)
			total_number_factorizations_findinterval += temp_total_number_factorizations_findinterval
			total_number_factorizations_bisection += temp_total_number_factorizations_bisection
			total_number_factorizations_compute_search_direction += temp_total_number_factorizations_compute_search_direction
			total_number_factorizations_inverse_power_iteration += temp_total_number_factorizations_inverse_power_iteration

			total_number_factorizations += temp_total_number_factorizations
			total_function_evaluation += temp_total_function_evaluation
			gval_next = gval_current
			# When we are able to solve the trust-region subproblem, we compute ρ_k to check if the
			# candidate solution has a reduction in the function value so that we accept the step by
			if success_subproblem_solve
				if fval_next <= fval_current + ξ * min_gval_norm * norm(d_k) + (1 + abs(fval_current)) * 1e-8
					total_gradient_evaluation += 1
					temp_grad = grad(nlp, x_k + d_k)
					temp_norm = norm(temp_grad, 2)
					if isnan(temp_norm)
						if print_level >= 0
							println("$k. grad(nlp, x_k + d_k) is NaN.")
						end
						@warn "$k grad(nlp, x_k + d_k) is NaN."
					else
						min_gval_norm = min(min_gval_norm, temp_norm)
					end
				end

				start_time_temp = time()
				ρ_k, actual_fct_decrease, predicted_fct_decrease = compute_ρ_standard_trust_region_method(fval_current, fval_next, gval_current, hessian_current, d_k, print_level)
				end_time_temp = time()
				total_time_temp = end_time_temp - start_time_temp
				if print_level >= 2
					println("compute_ρ_standard_trust_region_method operation took $total_time_temp.")
				end

				# Check for numerical error if the predicted reduction from the second order morel is negative.
				# In case it is negative, attempt to solve the trust-region subproblem using our default approach
				# only in case the original trust-region subproblem solver was different than the default approach
				if predicted_fct_decrease <= 0 && subproblem_solver_method != subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT
					if print_level >= 1
						println("Predicted function decrease is $predicted_fct_decrease >=0. fval_current is $fval_current and fval_next is $fval_next.")
					end
					@warn "Predicted function decrease is $predicted_fct_decrease >=0. fval_current is $fval_current and fval_next is $fval_next."
					if print_level >= 3
						hessian_current_matrix = Matrix(hessian_current)
						println("Radius, Gradient, and Hessian are $r_k, $gval_current, and $hessian_current_matrix.")
					end
					if print_level >= 1
						println("Solving trust-region subproblem using our approach.")
					end

					# Solve the trsut-region subproblem and generate the search direction d_k
					# fval_next, success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, hard_case = sub_routine_trust_region_sub_problem_solver(fval_current, gval_current, hessian_current, x_k, δ_k, γ_1, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT, print_level)
					# total_number_factorizations += temp_total_number_factorizations
					# total_function_evaluation += temp_total_function_evaluation
					fval_next, success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, hard_case, temp_total_number_factorizations_findinterval, temp_total_number_factorizations_bisection, temp_total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration = sub_routine_trust_region_sub_problem_solver(fval_current, gval_current, hessian_current, x_k, δ_k, γ_1, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT, print_level)
					total_number_factorizations_findinterval += temp_total_number_factorizations_findinterval
					total_number_factorizations_bisection += temp_total_number_factorizations_bisection
					total_number_factorizations_compute_search_direction += temp_total_number_factorizations_compute_search_direction
					total_number_factorizations_inverse_power_iteration += temp_total_number_factorizations_inverse_power_iteration
					total_number_factorizations += temp_total_number_factorizations
					total_function_evaluation += temp_total_function_evaluation

					# When we are able to solve the trust-region subproblem, we compute ρ_k to check if the
					# candidate solution has a reduction in the function value so that we accept the step by
					if success_subproblem_solve
						if fval_next <= fval_current + ξ * min_gval_norm * norm(d_k) + (1 + abs(fval_current)) * 1e-8
							total_gradient_evaluation += 1
							temp_grad = grad(nlp, x_k + d_k)
							temp_norm = norm(temp_grad, 2)
							if isnan(temp_norm)
								if print_level >= 0
									println("$k. grad(nlp, x_k + d_k) is NaN.")
								end
								@warn "$k grad(nlp, x_k + d_k) is NaN."
							else
								min_gval_norm = min(min_gval_norm, temp_norm)
							end
						end

						start_time_temp = time()
						ρ_k, actual_fct_decrease, predicted_fct_decrease = compute_ρ_standard_trust_region_method(fval_current, fval_next, gval_current, hessian_current, d_k, print_level)
						end_time_temp = time()
						total_time_temp = end_time_temp - start_time_temp
						if print_level >= 2
							println("compute_ρ_standard_trust_region_method operation took $total_time_temp.")
						end
						# Check for numerical error if the predicted reduction from the second order morel is negative.
						# In case it is negative, mark the solution of the trust-region subproblem as failure
						# by setting ρ_k to a negative default value (-1.0) and  the search direction d_k to 0 vector
						if predicted_fct_decrease <= 0.0
							if print_level >= 1
								println("Predicted function decrease is $predicted_fct_decrease >=0. fval_current is $fval_current and fval_next is $fval_next.")
							end
							@warn "Predicted function decrease is $predicted_fct_decrease >=0. fval_current is $fval_current and fval_next is $fval_next."
							ρ_k = -1.0
							actual_fct_decrease = 0.0
							predicted_fct_decrease = 0.0
							d_k = zeros(length(x_k))
						end
					else
						# In case we failt to solve the trust-region subproblem using our default solver, we mark that as a failure
						# by setting ρ_k to a negative default value (-1.0) and  the search direction d_k to 0 vector
						ρ_k = -1.0
						actual_fct_decrease = 0.0
						predicted_fct_decrease = 0.0
						d_k = zeros(length(x_k))
					end
				end
			else
				# In case we failt to solve the trust-region subproblem, we mark that as a failure
				# by setting ρ_k to a negative default value (-1.0) and  the search direction d_k to 0 vector
				ρ_k = -1.0
				actual_fct_decrease = 0.0
				predicted_fct_decrease = 0.0
				d_k = zeros(length(x_k))
			end
			if print_level >= 1
				println("Iteration $k with fval_next is $fval_next and fval_current is $fval_current.")
				println("Iteration $k with actual_fct_decrease is $actual_fct_decrease and predicted_fct_decrease is $predicted_fct_decrease.")
			end

			ρ_hat_k = ρ_k
			norm_gval_current = norm(gval_current, 2)
			norm_gval_next = norm_gval_current
			# Accept the generated search direction d_k when ρ_k is positive
			# and compute ρ_hat_k for the radius update rule
			if ρ_k >= 0.0 && (fval_next <= fval_current)
				if print_level >= 1
					println("$k. =======STEP IS ACCEPTED========== $ρ_k =========fval_next is $fval_next and fval_current is $fval_current.")
				end
				x_k = x_k + d_k
				start_time_temp = time()
				gval_next = temp_grad
				if isnan(min_gval_norm)
					if print_level >= 0
						println("$k. min_gval_norm is NaN")
					end
					@warn "$k min_gval_norm is NaN."
				end
				end_time_temp = time()
				total_time_temp = end_time_temp - start_time_temp
				if print_level >= 2
					println("gval_next operation took $total_time_temp.")
				end
				start_time_temp = time()
				ρ_hat_k, actual_fct_decrease, predicted_fct_decrease, guarantee_factor = compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, d_k, θ, min_gval_norm, print_level, compute_ρ_hat_approach)
				end_time_temp = time()
				total_time_temp = end_time_temp - start_time_temp
				if print_level >= 2
					println("compute_ρ_hat operation took $total_time_temp.")
				end

                fval_current = fval_next
                gval_current = gval_next
				compute_hessian = true
            else
                #else x_k+1 = x_k, fval_current, gval_current, hessian_current will not change
                compute_hessian = false
            end
			if print_level >= 1
				println("$k. ρ_hat_k is $ρ_hat_k.")
				println("$k. hard_case is $hard_case")
				norm_d_k = norm(d_k, 2)
				println("$k. r_k is $r_k and ||d_k|| is $norm_d_k.")
			end

			# Radius update
			if radius_update_rule_approach == "DEFAULT"
				if !success_subproblem_solve || isnan(ρ_hat_k) || ρ_hat_k < β_1
					r_k = r_k / ω_1
				else
					r_k = max(ω_2 * norm(d_k, 2), r_k)
				end
			# This to be able to test the performance of the algorithm for the ablation study
			else
				if !success_subproblem_solve
					r_k = r_k / ω_1
				else
					if isnan(ρ_hat_k) || ρ_hat_k < β_1
						r_k = norm(d_k, 2) / ω_1
					else
						r_k = ω_1 * norm(d_k, 2)
					end
				end
			end

			push!(iteration_stats, (k, fval_current, norm(gval_current, 2)))
			if ρ_k < 0 && min(min_gval_norm, norm(grad(nlp, x_k + d_k), 2)) <= gradient_termination_tolerance
				@info "========Convergence without accepting step========="
				if print_level >= 0
					println("========Convergence without accepting step=========")
				end
			end
			# Check termination condition for gradient
			if norm(gval_next, 2) <= gradient_termination_tolerance ||  min_gval_norm <= gradient_termination_tolerance
				push!(iteration_stats, (k, fval_next, min_gval_norm))
	            # computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
				computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations, "total_number_factorizations_findinterval" => total_number_factorizations_findinterval, "total_number_factorizations_bisection" => total_number_factorizations_bisection, "total_number_factorizations_compute_search_direction" => total_number_factorizations_compute_search_direction, "total_number_factorizations_inverse_power_iteration" => total_number_factorizations_inverse_power_iteration)
				if print_level >= 0
	            	println("*********************************Iteration Count: ", k)
				end
				if print_level >= 2
					try
						cholesky(Matrix(hessian_current))
						println("==============Local Minimizer=============")
					catch e
						println("==============Saddle Point=============")
					end
				end

				return x_k, TerminationStatusCode.OPTIMAL, iteration_stats, computation_stats, k
	        end

			# Check termination condition for trust-region radius if it becomes too small
			if r_k <= MINIMUM_TRUST_REGION_RADIUS
				# computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
				computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations, "total_number_factorizations_findinterval" => total_number_factorizations_findinterval, "total_number_factorizations_bisection" => total_number_factorizations_bisection, "total_number_factorizations_compute_search_direction" => total_number_factorizations_compute_search_direction, "total_number_factorizations_inverse_power_iteration" => total_number_factorizations_inverse_power_iteration)
				if print_level >= 0
					println("$k. Trust region radius $r_k is too small.")
				end
				return x_k, TerminationStatusCode.TRUST_REGION_RADIUS_LIMIT, iteration_stats, computation_stats, k
			end

			# Check termination condition for function value if the objective function is unbounded (safety check)
			if fval_current <= MINIMUM_OBJECTIVE_FUNCTION || fval_next <= MINIMUM_OBJECTIVE_FUNCTION
				# computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
				computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations, "total_number_factorizations_findinterval" => total_number_factorizations_findinterval, "total_number_factorizations_bisection" => total_number_factorizations_bisection, "total_number_factorizations_compute_search_direction" => total_number_factorizations_compute_search_direction, "total_number_factorizations_inverse_power_iteration" => total_number_factorizations_inverse_power_iteration)
				if print_level >= 0
					println("$k. Function values ($fval_current, $fval_next) are too small.")
				end
				return x_k, TerminationStatusCode.UNBOUNDED, iteration_stats, computation_stats, k
			end

			# Check termination condition for time if we exceeded the time limit
	        if time() - start_time > MAX_TIME
	            # computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
				computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations, "total_number_factorizations_findinterval" => total_number_factorizations_findinterval, "total_number_factorizations_bisection" => total_number_factorizations_bisection, "total_number_factorizations_compute_search_direction" => total_number_factorizations_compute_search_direction, "total_number_factorizations_inverse_power_iteration" => total_number_factorizations_inverse_power_iteration)
				return x_k, TerminationStatusCode.TIME_LIMIT, iteration_stats, computation_stats, k
	        end
        	k += 1
        end
	# Handle exceptions
    catch e
		@error e
		# computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
		computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations, "total_number_factorizations_findinterval" => total_number_factorizations_findinterval, "total_number_factorizations_bisection" => total_number_factorizations_bisection, "total_number_factorizations_compute_search_direction" => total_number_factorizations_compute_search_direction, "total_number_factorizations_inverse_power_iteration" => total_number_factorizations_inverse_power_iteration)
		status = TerminationStatusCode.OTHER_ERROR
		if isa(e, OutOfMemoryError)
			status = TerminationStatusCode.MEMORY_LIMIT
		end
		return x_k, status, iteration_stats, computation_stats, k
    end
    # computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
	computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations, "total_number_factorizations_findinterval" => total_number_factorizations_findinterval, "total_number_factorizations_bisection" => total_number_factorizations_bisection, "total_number_factorizations_compute_search_direction" => total_number_factorizations_compute_search_direction, "total_number_factorizations_inverse_power_iteration" => total_number_factorizations_inverse_power_iteration)
	return x_k, TerminationStatusCode.ITERATION_LIMIT, iteration_stats, computation_stats, k
end


function CAT_original_alg(problem::Problem_Data, x::Vector{Float64}, δ::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
    @assert(δ >= 0)
    MAX_ITERATIONS = problem.MAX_ITERATIONS
    MAX_TIME = problem.MAX_TIME
    gradient_termination_tolerance = problem.gradient_termination_tolerance
    β_1 = problem.β_1
	compute_ρ_hat_approach = "DEFAULT"
    ω_1 = problem.ω_1
    x_k = x
    δ_k = δ
    r_k = problem.r_1
	@assert(r_k > 0)
    γ_2 = problem.γ_2
    nlp = problem.nlp
    θ = problem.θ
	print_level = problem.print_level
	iteration_stats = DataFrame(k = [], fval = [], gradval = [])
    total_function_evaluation = 0
    total_gradient_evaluation = 0
    total_hessian_evaluation = 0
    total_number_factorizations = 0
    k = 1
    try
        gval_current = grad(nlp, x_k)
        fval_current = obj(nlp, x_k)
        total_function_evaluation += 1
        total_gradient_evaluation += 1
		hessian_current = hess(nlp, x_k)
		total_hessian_evaluation += 1

        compute_hessian = false
        if norm(gval_current, 2) <= gradient_termination_tolerance
            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => 1)
			if print_level >= 0
            	println("*********************************Iteration Count: ", 1)
			end
			push!(iteration_stats, (1, fval_current, norm(gval_current, 2)))
            return x_k, TerminationStatusCode.OPTIMAL, iteration_stats, computation_stats, 1
        end
        start_time = time()
		min_gval_norm = norm(gval_current, 2)
        while k <= MAX_ITERATIONS
			temp_grad = gval_current
			if print_level >= 1
				start_time_str = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
				println("$start_time. Iteration $k with radius $r_k and total_number_factorizations $total_number_factorizations.")
			end
            if compute_hessian
				start_time_temp = time()
                hessian_current = hess(nlp, x_k)
				end_time_temp = time()
				total_time_temp = end_time_temp - start_time_temp
				if print_level >= 2
					println("hessian_current operation took $total_time_temp.")
				end
                total_hessian_evaluation += 1
            end

			fval_next, success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, hard_case = sub_routine_trust_region_sub_problem_solver(fval_current, gval_current, hessian_current, x_k, δ_k, γ_1, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_method, print_level)
			total_number_factorizations += temp_total_number_factorizations
			total_function_evaluation += temp_total_function_evaluation
			gval_next = gval_current

			if success_subproblem_solve
				temp_grad = grad(nlp, x_k + d_k)
				total_gradient_evaluation += 1
				temp_norm = norm(temp_grad, 2)

				if isnan(temp_norm)
					@warn "$k grad(nlp, x_k + d_k) is NaN."
					if print_level >= 0
						println("$k. grad(nlp, x_k + d_k) is NaN.")
					end
				else
					min_gval_norm = min(min_gval_norm, temp_norm)
				end

				start_time_temp = time()
				ρ_k, actual_fct_decrease, predicted_fct_decrease = compute_ρ_standard_trust_region_method(fval_current, fval_next, gval_current, hessian_current, d_k, print_level)
				end_time_temp = time()
				total_time_temp = end_time_temp - start_time_temp
				if print_level >= 2
					println("compute_ρ_standard_trust_region_method operation took $total_time_temp.")
				end
				if predicted_fct_decrease <= 0.0
					if print_level >= 1
						println("Predicted function decrease is $predicted_fct_decrease >=0. fval_current is $fval_current and fval_next is $fval_next.")
					end
					@warn "Predicted function decrease is $predicted_fct_decrease >=0. fval_current is $fval_current and fval_next is $fval_next."
					ρ_k = -1.0
					actual_fct_decrease = 0.0
					predicted_fct_decrease = 0.0
					d_k = zeros(length(x_k))
				end
			else
				ρ_k = -1.0
				actual_fct_decrease = 0.0
				predicted_fct_decrease = 0.0
				d_k = zeros(length(x_k))
			end
			if print_level >= 1
				println("Iteration $k with fval_next is $fval_next and fval_current is $fval_current.")
				println("Iteration $k with actual_fct_decrease is $actual_fct_decrease and predicted_fct_decrease is $predicted_fct_decrease.")
			end

			ρ_hat_k = ρ_k
			norm_gval_current = norm(gval_current, 2)
			norm_gval_next = norm_gval_current
			if ρ_k >= 0.0 && (fval_next <= fval_current)
				if print_level >= 0
					println("$k. =======STEP IS ACCEPTED========== $ρ_k =========fval_next is $fval_next and fval_current is $fval_current.")
				end
				x_k = x_k + d_k
				start_time_temp = time()
				gval_next = temp_grad
				temp_norm = norm(gval_next, 2)
				if isnan(temp_norm)
					if print_level >= 0
						println("$k. grad(nlp, x_k + d_k) is NaN.")
					end
					@warn "$k grad(nlp, x_k + d_k) is NaN."
				else
					min_gval_norm = min(min_gval_norm, temp_norm)
				end
				if isnan(min_gval_norm)
					if print_level >= 0
						println("$k. min_gval_norm is NaN")
					end
					@warn "$k min_gval_norm is NaN."
				end
				end_time_temp = time()
				total_time_temp = end_time_temp - start_time_temp
				if print_level >= 2
					println("gval_next operation took $total_time_temp.")
				end
				start_time_temp = time()
				ρ_hat_k, actual_fct_decrease, predicted_fct_decrease, guarantee_factor = compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, d_k, θ, min_gval_norm, print_level, compute_ρ_hat_approach)
				end_time_temp = time()
				total_time_temp = end_time_temp - start_time_temp
				if print_level >= 2
					println("compute_ρ_hat operation took $total_time_temp.")
				end

                fval_current = fval_next
                gval_current = gval_next
				# min_gval_norm = min(min_gval_norm, norm(gval_current, 2))
				compute_hessian = true
            else
                #else x_k+1 = x_k, fval_current, gval_current, hessian_current will not change
                compute_hessian = false
            end
			if print_level >= 1
				println("$k. ρ_hat_k is $ρ_hat_k.")
				println("$k. hard_case is $hard_case")
				norm_d_k = norm(d_k, 2)
				println("$k. r_k is $r_k and ||d_k|| is $norm_d_k.")
			end
			if !success_subproblem_solve || isnan(ρ_hat_k) || ρ_hat_k < β_1
				r_k = r_k / ω_1
			else
				r_k = ω_1 * norm(d_k, 2)
			end

			push!(iteration_stats, (k, fval_current, norm(gval_current, 2)))
			if ρ_k < 0 && min(min_gval_norm, norm(grad(nlp, x_k + d_k), 2)) <= gradient_termination_tolerance
				@info "========Convergence without accepting step========="
				if print_level >= 0
					println("========Convergence without accepting step=========")
				end
			end
			if norm(gval_next, 2) <= gradient_termination_tolerance ||  min_gval_norm <= gradient_termination_tolerance
				push!(iteration_stats, (k, fval_next, min_gval_norm))
	            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
				if print_level >= 0
	            	println("*********************************Iteration Count: ", k)
				end
				if print_level >= 2
					try
						cholesky(Matrix(hessian_current))
						println("==============Local Minimizer=============")
					catch e
						println("==============Saddle Point=============")
					end
				end

				return x_k, TerminationStatusCode.OPTIMAL, iteration_stats, computation_stats, k
	        end

			if r_k <= MINIMUM_TRUST_REGION_RADIUS
				computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
				if print_level >= 0
					println("$k. Trust region radius $r_k is too small.")
				end
				return x_k, TerminationStatusCode.TRUST_REGION_RADIUS_LIMIT, iteration_stats, computation_stats, k
			end

			if fval_current <= MINIMUM_OBJECTIVE_FUNCTION || fval_next <= MINIMUM_OBJECTIVE_FUNCTION
				computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
				if print_level >= 0
					println("$k. Function values ($fval_current, $fval_next) are too small.")
				end
				return x_k, TerminationStatusCode.UNBOUNDED, iteration_stats, computation_stats, k
			end

	        if time() - start_time > MAX_TIME
	            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
				return x_k, TerminationStatusCode.TIME_LIMIT, iteration_stats, computation_stats, k
	        end
        	k += 1
        end
    catch e
		@error e
		computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
		status = TerminationStatusCode.OTHER_ERROR
		if isa(e, OutOfMemoryError)
			status = TerminationStatusCode.MEMORY_LIMIT
		end
		return x_k, status, iteration_stats, computation_stats, k
    end
    computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
	return x_k, TerminationStatusCode.ITERATION_LIMIT, iteration_stats, computation_stats, k
end

end # module
