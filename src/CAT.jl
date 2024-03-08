__precompile__()
#Main algorithm code goes here
module consistently_adaptive_trust_region_method
using NLPModels, LinearAlgebra, DataFrames, SparseArrays
include("./trust_region_subproblem_solver.jl")

export Problem_Data
export phi, findinterval, bisection, computeSecondOrderModel, optimizeSecondOrderModel, compute_ρ_hat, CAT

mutable struct SmallTrustRegionradius
	message::String
	radius::Float64
end

mutable struct WrongFunctionPredictedReduction
	message::String
	predicted_reduction::Float64
end

mutable struct UnboundedObjective
	message::String
	fval::Float64
end

const MINIMUM_TRUST_REGION_RADIUS = 1e-40
const MINIMUM_OBJECTIVE_FUNCTION = -1e30
const RADIUS_UPDATE_SAFETY_CHECK_FAILURE_SUBPROBLEM = 2.0
const INITIAL_RADIUS_MULTIPLICATIVE_RULEE = 10

mutable struct Problem_Data
    nlp::AbstractNLPModel
	β_1::Float64
	β_2::Float64
    θ::Float64
    ω_1::Float64
	ω_2::Float64
    r_1::Float64
    MAX_ITERATION::Int64
    gradient_termination_tolerance::Float64
	γ_2::Float64
    MAX_TIME::Float64
	print_level::Int64
	compute_ρ_hat_approach::String
    # initialize parameters
    function Problem_Data(nlp::AbstractNLPModel, β_1::Float64=0.1, β_2::Float64=0.8,
                           θ::Float64=0.1, ω_1::Float64=4.0, ω_2::Float64=20.0, r_1::Float64=1.0,
                           MAX_ITERATION::Int64=10000, gradient_termination_tolerance::Float64=1e-5, γ_2::Float64=0.1,
                           MAX_TIME::Float64=30 * 60.0, print_level::Int64=0, compute_ρ_hat_approach::String="DEFAULT")
		@assert(β_1 > 0 && β_1 < 1)
		@assert(β_2 > 0 && β_2 < 1)
		@assert(β_2 >= β_1)
        @assert(θ >= 0 && θ < 1)
        # @assert(β_1 * θ < 1 - β_1)
        @assert(ω_1 >= 1)
		@assert(ω_2 >= 1)
		@assert(ω_2 >= ω_1)
        # @assert(r_1 > 0)
        @assert(MAX_ITERATION > 0)
        @assert(MAX_TIME > 0)
		@assert(1 > γ_2 > 0)
        # @assert(γ_2 > (1 / ω) && γ_2 <= 1)
        return new(nlp, β_1, β_2, θ, ω_1, ω_2, r_1, MAX_ITERATION, gradient_termination_tolerance, γ_2, MAX_TIME, print_level, compute_ρ_hat_approach)
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
	ϵ_machine = eps()
	actual_fct_decrease = fval_current - fval_next
	predicted_fct_decrease = - second_order_model_value_current_iterate
	# ρ_hat = (actual_fct_decrease + ϵ_machine) / (predicted_fct_decrease + guarantee_factor)
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
	ϵ_machine = eps()
	# ρ = (actual_fct_decrease + ϵ_machine) / predicted_fct_decrease
	ρ = actual_fct_decrease / predicted_fct_decrease
	if print_level >= 1 && ρ == -Inf || isnan(ρ)
		println("ρ is $ρ. actual_fct_decrease is $actual_fct_decrease and predicted_fct_decrease is $predicted_fct_decrease.")
	end
    return ρ, actual_fct_decrease, predicted_fct_decrease
end


#This is the final working code. It is commented because the next implementation improves the cleanliness and reusability of the code
# function CAT(problem::Problem_Data, x::Vector{Float64}, δ::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
#     @assert(δ >= 0)
#     MAX_ITERATION = problem.MAX_ITERATION
#     MAX_TIME = problem.MAX_TIME
#     gradient_termination_tolerance = problem.gradient_termination_tolerance
#     β_1 = problem.β_1
# 	β_2 = problem.β_2
#
#     ω_1 = problem.ω_1
# 	ω_2 = problem.ω_2
#     x_k = x
#     δ_k = δ
#     r_k = problem.r_1
#     γ_2 = problem.γ_2
#     nlp = problem.nlp
#     θ = problem.θ
# 	print_level = problem.print_level
# 	compute_ρ_hat_approach = problem.compute_ρ_hat_approach
# 	iteration_stats = DataFrame(k = [], deltaval = [], directionval = [], fval = [], gradval = [])
#     total_function_evaluation = 0
#     total_gradient_evaluation = 0
#     total_hessian_evaluation = 0
#     total_number_factorizations = 0
#     k = 1
#
#     try
#         gval_current = grad(nlp, x_k)
#         fval_current = obj(nlp, x_k)
#         total_function_evaluation += 1
#         total_gradient_evaluation += 1
# 		hessian_current = hess(nlp, x_k)
# 		total_hessian_evaluation += 1
# 		if r_k <= 0.0
# 			r_k = INITIAL_RADIUS_MULTIPLICATIVE_RULEE * norm(gval_current, Inf) / norm(hessian_current, Inf)
# 		end
#         compute_hessian = false
#         if norm(gval_current, 2) <= gradient_termination_tolerance
#             computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
# 			if print_level >= 0
#             	println("*********************************Iteration Count: ", 1)
# 			end
# 			push!(iteration_stats, (1, δ, [], fval_current, norm(gval_current, 2)))
#             return x_k, "SUCCESS", iteration_stats, computation_stats, 1
#         end
#         start_time = time()
# 		min_gval_norm = norm(gval_current, 2)
#         while k <= MAX_ITERATION
# 			if print_level >= 0
# 				start_time_str = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
# 				println("$start_time. Iteration $k with radius $r_k and total_number_factorizations $total_number_factorizations.")
# 			end
#             if compute_hessian
# 				start_time_temp = time()
#                 hessian_current = hess(nlp, x_k)
# 				end_time_temp = time()
# 				total_time_temp = end_time_temp - start_time_temp
# 				println("hessian_current operation took $total_time_temp.")
#                 total_hessian_evaluation += 1
#             end
#
# 			start_time_temp = time()
#             success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, hard_case = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp.meta.name, subproblem_solver_method, print_level)
# 			total_number_factorizations += temp_total_number_factorizations
# 			end_time_temp = time()
# 			total_time_temp = end_time_temp - start_time_temp
# 			println("solveTrustRegionSubproblem operation took $total_time_temp.")
#
# 			start_time_temp = time()
# 			second_order_model_value_current_iterate = computeSecondOrderModel(fval_current, gval_current, hessian_current, d_k)
# 			end_time_temp = time()
# 			total_time_temp = end_time_temp - start_time_temp
# 			println("computeSecondOrderModel operation took $total_time_temp.")
#
# 			if success_subproblem_solve && second_order_model_value_current_iterate < 0
# 				start_time_temp = time()
#             	fval_next = obj(nlp, x_k + d_k)
# 				end_time_temp = time()
# 				total_time_temp = end_time_temp - start_time_temp
# 				println("fval_next operation took $total_time_temp.")
# 				total_function_evaluation += 1
# 			else
# 				fval_next = fval_current
# 			end
# 			gval_next = gval_current
# 			if success_subproblem_solve
# 				start_time_temp = time()
# 				ρ_k, actual_fct_decrease, predicted_fct_decrease = compute_ρ_standard_trust_region_method(fval_current, fval_next, gval_current, hessian_current, d_k, print_level)
# 				end_time_temp = time()
# 				total_time_temp = end_time_temp - start_time_temp
# 				println("compute_ρ_standard_trust_region_method operation took $total_time_temp.")
# 				if predicted_fct_decrease <= 0
# 					if print_level >= 1
# 						println("Predicted function decrease is $predicted_fct_decrease >=0. fval_current is $fval_current and fval_next is $fval_next.")
# 					end
# 					@warn "Predicted function decrease is $predicted_fct_decrease >=0. fval_current is $fval_current and fval_next is $fval_next."
# 					if print_level >= 1
# 						hessian_current_matrix = Matrix(hessian_current)
# 						println("Radius, Gradient, and Hessian are $r_k, $gval_current, and $hessian_current_matrix.")
# 					end
# 					if print_level >= 0
# 						println("Solving trust-region subproblem using our approach.")
# 					end
#
# 					start_time_temp = time()
# 					success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, hard_case = optimizeSecondOrderModel(gval_current, hessian_current, δ_k, γ_2, r_k, min_gval_norm, print_level)
# 					total_number_factorizations += temp_total_number_factorizations
# 					end_time_temp = time()
# 					total_time_temp = end_time_temp - start_time_temp
# 					println("optimizeSecondOrderModel operation took $total_time_temp.")
#
# 					start_time_temp = time()
# 					second_order_model_value_current_iterate = computeSecondOrderModel(fval_current, gval_current, hessian_current, d_k)
# 					end_time_temp = time()
# 					total_time_temp = end_time_temp - start_time_temp
# 					println("computeSecondOrderModel operation took $total_time_temp.")
#
# 					if success_subproblem_solve && second_order_model_value_current_iterate < 0
# 						start_time_temp = time()
# 		            	fval_next = obj(nlp, x_k + d_k)
# 						end_time_temp = time()
# 						total_time_temp = end_time_temp - start_time_temp
# 						println("fval_next operation took $total_time_temp.")
# 						total_function_evaluation += 1
# 					else
# 						fval_next = fval_current
# 					end
# 					if success_subproblem_solve
# 						start_time_temp = time()
# 						ρ_k, actual_fct_decrease, predicted_fct_decrease = compute_ρ_standard_trust_region_method(fval_current, fval_next, gval_current, hessian_current, d_k, print_level)
# 						end_time_temp = time()
# 						total_time_temp = end_time_temp - start_time_temp
# 						println("compute_ρ_standard_trust_region_method operation took $total_time_temp.")
# 						if predicted_fct_decrease <= 0.0
# 							if print_level >= 1
# 								println("Predicted function decrease is $predicted_fct_decrease >=0. fval_current is $fval_current and fval_next is $fval_next.")
# 							end
# 							@warn "Predicted function decrease is $predicted_fct_decrease >=0. fval_current is $fval_current and fval_next is $fval_next."
# 							ρ_k = -1.0
# 							actual_fct_decrease = 0.0
# 							predicted_fct_decrease = 0.0
# 							d_k = zeros(length(x_k))
# 						end
# 					else
# 						ρ_k = -1.0
# 						actual_fct_decrease = 0.0
# 						predicted_fct_decrease = 0.0
# 						d_k = zeros(length(x_k))
# 					end
# 				end
# 			else
# 				ρ_k = -1.0
# 				actual_fct_decrease = 0.0
# 				predicted_fct_decrease = 0.0
# 				d_k = zeros(length(x_k))
# 			end
# 			if print_level >= 0
# 				println("Iteration $k with fval_next is $fval_next and fval_current is $fval_current.")
# 				println("Iteration $k with actual_fct_decrease is $actual_fct_decrease and predicted_fct_decrease is $predicted_fct_decrease.")
# 			end
#
# 			ρ_hat_k = ρ_k
# 			norm_gval_current = norm(gval_current, 2)
# 			norm_gval_next = norm_gval_current
# 			if ρ_k >= 0.0 && (fval_next <= fval_current)
# 				if print_level >= 0
# 					println("$k. =======STEP IS ACCEPTED========== $ρ_k =========fval_next is $fval_next and fval_current is $fval_current.")
# 				end
# 				x_k = x_k + d_k
# 				start_time_temp = time()
# 				gval_next = grad(nlp, x_k)
# 				total_gradient_evaluation += 1
# 				end_time_temp = time()
# 				total_time_temp = end_time_temp - start_time_temp
# 				println("gval_next operation took $total_time_temp.")
#
# 				start_time_temp = time()
# 				ρ_hat_k, actual_fct_decrease, predicted_fct_decrease, guarantee_factor = compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, d_k, θ, min_gval_norm, print_level, compute_ρ_hat_approach)
# 				end_time_temp = time()
# 				total_time_temp = end_time_temp - start_time_temp
# 				println("compute_ρ_hat operation took $total_time_temp.")
#
#                 fval_current = fval_next
#                 gval_current = gval_next
# 				min_gval_norm = min(min_gval_norm, norm(gval_current, 2))
# 				compute_hessian = true
#
#             else
#                 #else x_k+1 = x_k, fval_current, gval_current, hessian_current will not change
#                 compute_hessian = false
#             end
# 			if print_level >= 1
# 				println("$k. ρ_hat_k is $ρ_hat_k.")
# 				println("$k. hard_case is $hard_case")
# 				norm_d_k = norm(d_k, 2)
# 				println("$k. r_k is $r_k and ||d_k|| is $norm_d_k.")
# 			end
#
# 			if !success_subproblem_solve || isnan(ρ_hat_k) || ρ_hat_k < β_1
# 				if norm(d_k, 2) == 0
# 					r_k = r_k / RADIUS_UPDATE_SAFETY_CHECK_FAILURE_SUBPROBLEM
# 				else
# 					r_k = norm(d_k, 2) / ω_1
# 				end
# 			elseif ρ_k < β_2
# 				r_k = ω_1 * norm(d_k, 2)
# 			else
# 				r_k = ω_2 * norm(d_k, 2)
# 			end
#
# 			if r_k <= MINIMUM_TRUST_REGION_RADIUS
# 				if print_level >= 1
# 					println("$k. Trust region radius $r_k is too small.")
# 				end
# 				throw(SmallTrustRegionradius("Trust region radius $r_k is too small.", r_k))
# 			end
#
# 			if fval_current <= MINIMUM_OBJECTIVE_FUNCTION || fval_next <= MINIMUM_OBJECTIVE_FUNCTION
# 				throw(UnboundedObjective("Function values ($fval_current, $fval_next) are too small.", fval_current))
# 			end
#
# 			min_gval_norm = min(min_gval_norm, norm(gval_current, 2))
# 			push!(iteration_stats, (k, δ_k, d_k, fval_current, norm(gval_current, 2)))
# 	        if norm(gval_next, 2) <= gradient_termination_tolerance
# 				push!(iteration_stats, (k, δ_k, d_k, fval_next, norm(gval_next, 2)))
# 	            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
# 				if print_level >= 0
# 	            	println("*********************************Iteration Count: ", k)
# 				end
# 				if print_level >= 1
# 					try
# 						cholesky(Matrix(hessian_current))
# 						println("==============Local Minimizer=============")
# 					catch e
# 						println("==============Saddle Point=============")
# 					end
# 				end
#
# 	            return x_k, "SUCCESS", iteration_stats, computation_stats, k
# 	        end
#
# 	        if time() - start_time > MAX_TIME
# 	            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
# 	            return x_k, "MAX_TIME", iteration_stats, computation_stats, k
# 	        end
#         	k += 1
#         end
#     catch e
# 		@error e
# 		computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
# 		status = "FAILURE"
# 		if isa(e, SmallTrustRegionradius)
# 			@warn e.message
# 			status = "FAILURE_SMALL_RADIUS"
# 		elseif isa(e, WrongFunctionPredictedReduction)
# 			@warn e.message
# 			status = "FAILURE_WRONG_PREDICTED_REDUCTION"
# 		elseif isa(e, UnboundedObjective)
# 			@warn e.message
# 			status = "FAILURE_UNBOUNDED_OBJECTIVE"
# 		elseif isa(e, OutOfMemoryError)
# 			status = "FAILURE_OUT_OF_MEMORY_ERROR"
# 		else
# 			@error e
# 		end
#         return x_k, status, iteration_stats, computation_stats, k
#     end
#     computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
#     return x_k, "ITERARION_LIMIT", iteration_stats, computation_stats, k
# end

# function sub_routine_trust_region_sub_problem_solver_recursive(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_method, print_level)
# 	temp_total_gradient_evaluation = 0
# 	fval_next = fval_current
# 	gval_next_temp = gval_current
# 	start_time_temp = time()
# 	temp_total_function_evaluation = 0
# 	success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, hard_case = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp.meta.name, subproblem_solver_method, print_level)
# 	end_time_temp = time()
# 	total_time_temp = end_time_temp - start_time_temp
# 	println("solveTrustRegionSubproblem operation took $total_time_temp.")
#
# 	start_time_temp = time()
# 	second_order_model_value_current_iterate = computeSecondOrderModel(fval_current, gval_current, hessian_current, d_k)
# 	end_time_temp = time()
# 	total_time_temp = end_time_temp - start_time_temp
# 	println("computeSecondOrderModel operation took $total_time_temp.")
# 	termination_criteria_satisfied = success_subproblem_solve
# 	if success_subproblem_solve && second_order_model_value_current_iterate < 0
# 		start_time_temp = time()
# 		fval_next = obj(nlp, x_k + d_k)
# 		temp_total_function_evaluation += 1
# 		end_time_temp = time()
# 		total_time_temp = end_time_temp - start_time_temp
# 		println("fval_next operation took $total_time_temp.")
# 		if fval_next <= fval_current
# 			gval_next_temp = grad(nlp, x_k + d_k)
# 			temp_total_gradient_evaluation += 1
# 			if subproblem_solver_method == subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT
# 				q_1 = norm(hessian_current * d_k + gval_current + δ_k * d_k)
# 				norm_temp_grad_next = norm(gval_next_temp)
# 				q_2 = min(min_gval_norm, norm_temp_grad_next) / 100
# 				termination_criteria_satisfied = true
# 				if q_1 > q_2
# 					println("0==============SOLVE TRS AGAIN==============$q_1, $q_2, $norm_temp_grad_next.")
# 					if norm_temp_grad_next > 1e-5
# 						@error "==============SOLVE TRS AGAIN==============$q_1, $q_2, $norm_temp_grad_next."
# 						println("1==============SOLVE TRS AGAIN==============$q_1, $q_2, $norm_temp_grad_next.")
# 						termination_criteria_satisfied = false
# 					end
# 				end
# 			end
# 		end
# 	end
#
# 	return fval_next, gval_next_temp, success_subproblem_solve, termination_criteria_satisfied, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, temp_total_gradient_evaluation, hard_case
# end
#
# function sub_routine_trust_region_sub_problem_solver(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_method, print_level)
# 	if subproblem_solver_method != subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT
# 		fval_next, gval_next_temp, success_subproblem_solve, termination_criteria_satisfied, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, temp_total_gradient_evaluation, hard_case = sub_routine_trust_region_sub_problem_solver_recursive(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_method, print_level)
# 		min_gval_norm = min(min_gval_norm, norm(gval_next_temp))
# 		return fval_next, gval_next_temp, success_subproblem_solve, termination_criteria_satisfied, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, temp_total_gradient_evaluation, hard_case, min_gval_norm
# 	end
# 	max_attempts = 10
# 	fval_next, gval_next_temp, success_subproblem_solve, termination_criteria_satisfied, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, temp_total_gradient_evaluation, hard_case = sub_routine_trust_region_sub_problem_solver_recursive(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_method, print_level)
# 	min_gval_norm = min(min_gval_norm, norm(gval_next_temp))
# 	while success_subproblem_solve && !termination_criteria_satisfied && max_attempts >= 0
# 		fval_next, gval_next_temp, success_subproblem_solve, termination_criteria_satisfied, δ_k, d_k, temp_total_number_factorizations_, temp_total_function_evaluation_, temp_total_gradient_evaluation_, hard_case = sub_routine_trust_region_sub_problem_solver_recursive(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_method, print_level)
# 		min_gval_norm = min(min_gval_norm, norm(gval_next_temp))
# 		temp_total_number_factorizations += temp_total_number_factorizations_
# 		temp_total_function_evaluation += temp_total_function_evaluation_
# 		temp_total_gradient_evaluation += temp_total_gradient_evaluation_
# 		max_attempts -= 1
# 	end
# 	return fval_next, gval_next_temp, success_subproblem_solve, termination_criteria_satisfied, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, temp_total_gradient_evaluation, hard_case, min_gval_norm
# end

function sub_routine_trust_region_sub_problem_solver(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_method, print_level)
	fval_next = fval_current
	gval_next_temp = gval_current
	start_time_temp = time()
	temp_total_function_evaluation = 0
	success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, hard_case = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp.meta.name, subproblem_solver_method, print_level)
	end_time_temp = time()
	total_time_temp = end_time_temp - start_time_temp
	println("solveTrustRegionSubproblem operation took $total_time_temp.")

	start_time_temp = time()
	second_order_model_value_current_iterate = computeSecondOrderModel(fval_current, gval_current, hessian_current, d_k)
	end_time_temp = time()
	total_time_temp = end_time_temp - start_time_temp
	println("computeSecondOrderModel operation took $total_time_temp.")
	if success_subproblem_solve && second_order_model_value_current_iterate < 0
		start_time_temp = time()
		fval_next = obj(nlp, x_k + d_k)
		temp_total_function_evaluation += 1
		end_time_temp = time()
		total_time_temp = end_time_temp - start_time_temp
		println("fval_next operation took $total_time_temp.")
	end

	return fval_next, success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, hard_case
end

function CAT(problem::Problem_Data, x::Vector{Float64}, δ::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
    @assert(δ >= 0)
    MAX_ITERATION = problem.MAX_ITERATION
    MAX_TIME = problem.MAX_TIME
    gradient_termination_tolerance = problem.gradient_termination_tolerance
    β_1 = problem.β_1
	β_2 = problem.β_2

    ω_1 = problem.ω_1
	ω_2 = problem.ω_2
    x_k = x
    δ_k = δ
    r_k = problem.r_1
    γ_2 = problem.γ_2
    nlp = problem.nlp
    θ = problem.θ
	print_level = problem.print_level
	compute_ρ_hat_approach = problem.compute_ρ_hat_approach
	iteration_stats = DataFrame(k = [], deltaval = [], directionval = [], fval = [], gradval = [])
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
		if r_k <= 0.0
			r_k = INITIAL_RADIUS_MULTIPLICATIVE_RULEE * norm(gval_current, Inf) / norm(hessian_current, Inf)
		end
        compute_hessian = false
        if norm(gval_current, 2) <= gradient_termination_tolerance
            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
			if print_level >= 0
            	println("*********************************Iteration Count: ", 1)
			end
			push!(iteration_stats, (1, δ, [], fval_current, norm(gval_current, 2)))
            return x_k, "SUCCESS", iteration_stats, computation_stats, 1
        end
        start_time = time()
		min_gval_norm = norm(gval_current, 2)
        while k <= MAX_ITERATION
			if print_level >= 0
				start_time_str = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
				println("$start_time. Iteration $k with radius $r_k and total_number_factorizations $total_number_factorizations.")
			end
            if compute_hessian
				start_time_temp = time()
                hessian_current = hess(nlp, x_k)
				end_time_temp = time()
				total_time_temp = end_time_temp - start_time_temp
				println("hessian_current operation took $total_time_temp.")
                total_hessian_evaluation += 1
            end

			# fval_next, gval_next_temp, success_subproblem_solve, termination_criteria_satisfied, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, temp_total_gradient_evaluation, hard_case, min_gval_norm = sub_routine_trust_region_sub_problem_solver(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_method, print_level)
			fval_next, success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, hard_case = sub_routine_trust_region_sub_problem_solver(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_method, print_level)
			total_number_factorizations += temp_total_number_factorizations
			total_function_evaluation += temp_total_function_evaluation
			gval_next = gval_current
			# if success_subproblem_solve && termination_criteria_satisfied
			if success_subproblem_solve
				#This is needed - commented for debugging
				temp_ = grad(nlp, x_k + d_k)
				temp_norm = norm(temp_, 2)
				if !isnan(temp_norm)
					min_gval_norm = min(min_gval_norm, temp_norm)
				end
				if isnan(min_gval_norm)
					println("1.%%%%%%%%%%%%%%%%%$min_gval_norm%%%%%%%%%%%%%%%%%")
				end

				start_time_temp = time()
				ρ_k, actual_fct_decrease, predicted_fct_decrease = compute_ρ_standard_trust_region_method(fval_current, fval_next, gval_current, hessian_current, d_k, print_level)
				end_time_temp = time()
				total_time_temp = end_time_temp - start_time_temp
				println("compute_ρ_standard_trust_region_method operation took $total_time_temp.")
				if predicted_fct_decrease <= 0 && subproblem_solver_method != subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT
					if print_level >= 1
						println("Predicted function decrease is $predicted_fct_decrease >=0. fval_current is $fval_current and fval_next is $fval_next.")
					end
					@warn "Predicted function decrease is $predicted_fct_decrease >=0. fval_current is $fval_current and fval_next is $fval_next."
					if print_level >= 1
						hessian_current_matrix = Matrix(hessian_current)
						println("Radius, Gradient, and Hessian are $r_k, $gval_current, and $hessian_current_matrix.")
					end
					if print_level >= 0
						println("Solving trust-region subproblem using our approach.")
					end

					fval_next, success_subproblem_solve, δ_k, d_k, temp_total_number_factorizations, temp_total_function_evaluation, hard_case = sub_routine_trust_region_sub_problem_solver(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, nlp, subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT, print_level)
					total_number_factorizations += temp_total_number_factorizations
					total_function_evaluation += temp_total_function_evaluation
					# if success_subproblem_solve && termination_criteria_satisfied
					if success_subproblem_solve
						#This is needed - commented for debugging
						temp_ = grad(nlp, x_k + d_k)
						temp_norm = norm(temp_, 2)
						if !isnan(temp_norm)
							min_gval_norm = min(min_gval_norm, temp_norm)
						end
						if isnan(min_gval_norm)
							println("2.%%%%%%%%%%%%%%%%%$min_gval_norm%%%%%%%%%%%%%%%%%")
						end
						start_time_temp = time()
						ρ_k, actual_fct_decrease, predicted_fct_decrease = compute_ρ_standard_trust_region_method(fval_current, fval_next, gval_current, hessian_current, d_k, print_level)
						end_time_temp = time()
						total_time_temp = end_time_temp - start_time_temp
						println("compute_ρ_standard_trust_region_method operation took $total_time_temp.")
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
				end
			else
				ρ_k = -1.0
				actual_fct_decrease = 0.0
				predicted_fct_decrease = 0.0
				d_k = zeros(length(x_k))
			end
			if print_level >= 0
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
				gval_next = grad(nlp, x_k)
				temp_norm = norm(gval_next, 2)
				if !isnan(temp_norm)
					min_gval_norm = min(min_gval_norm, temp_norm)
				end
				if isnan(min_gval_norm)
					println("3.%%%%%%%%%%%%%%%%%$min_gval_norm%%%%%%%%%%%%%%%%%")
				end
				total_gradient_evaluation += 1
				end_time_temp = time()
				total_time_temp = end_time_temp - start_time_temp
				println("gval_next operation took $total_time_temp.")
				# gval_next = gval_next_temp
				# if norm(gval_next) != norm(grad(nlp, x_k))
				# 	@error "============GRAD COMPUTATION WRONG============"
				# end
				start_time_temp = time()
				ρ_hat_k, actual_fct_decrease, predicted_fct_decrease, guarantee_factor = compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, d_k, θ, min_gval_norm, print_level, compute_ρ_hat_approach)
				end_time_temp = time()
				total_time_temp = end_time_temp - start_time_temp
				println("compute_ρ_hat operation took $total_time_temp.")

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
				if norm(d_k, 2) == 0
					# r_k = r_k / RADIUS_UPDATE_SAFETY_CHECK_FAILURE_SUBPROBLEM
					r_k = r_k / ω_1
				else
					# r_k = norm(d_k, 2) / ω_1
					r_k = r_k / ω_1
				end
			elseif ρ_k < β_2
				r_k = max(ω_1 * norm(d_k, 2), r_k)
				# r_k = ω_1 * norm(d_k, 2)
			else
				# r_k = ω_2 * norm(d_k, 2)
				r_k = max(ω_2 * norm(d_k, 2), r_k)
			end

			# min_gval_norm = min(min_gval_norm, norm(gval_current, 2))
			push!(iteration_stats, (k, δ_k, d_k, fval_current, norm(gval_current, 2)))
	        # if norm(gval_next, 2) <= gradient_termination_tolerance
			if ρ_k < 0 && min(min_gval_norm, norm(grad(nlp, x_k + d_k), 2)) <= gradient_termination_tolerance
				@info "========Convergence even without accepting step========="
				println("========Convergence even without accepting step=========")
			end
			if norm(gval_next, 2) <= gradient_termination_tolerance ||  min_gval_norm <= gradient_termination_tolerance
				# push!(iteration_stats, (k, δ_k, d_k, fval_next, norm(gval_next, 2)))
				push!(iteration_stats, (k, δ_k, d_k, fval_next, min_gval_norm))
	            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
				if print_level >= 0
	            	println("*********************************Iteration Count: ", k)
				end
				if print_level >= 1
					try
						cholesky(Matrix(hessian_current))
						println("==============Local Minimizer=============")
					catch e
						println("==============Saddle Point=============")
					end
				end

	            return x_k, "SUCCESS", iteration_stats, computation_stats, k
	        end

			if r_k <= MINIMUM_TRUST_REGION_RADIUS
				if print_level >= 1
					println("$k. Trust region radius $r_k is too small.")
				end
				throw(SmallTrustRegionradius("Trust region radius $r_k is too small.", r_k))
			end

			if fval_current <= MINIMUM_OBJECTIVE_FUNCTION || fval_next <= MINIMUM_OBJECTIVE_FUNCTION
				throw(UnboundedObjective("Function values ($fval_current, $fval_next) are too small.", fval_current))
			end

	        if time() - start_time > MAX_TIME
	            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
	            return x_k, "MAX_TIME", iteration_stats, computation_stats, k
	        end
        	k += 1
        end
    catch e
		@error e
		computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
		status = "FAILURE"
		if isa(e, SmallTrustRegionradius)
			@warn e.message
			status = "FAILURE_SMALL_RADIUS"
		elseif isa(e, WrongFunctionPredictedReduction)
			@warn e.message
			status = "FAILURE_WRONG_PREDICTED_REDUCTION"
		elseif isa(e, UnboundedObjective)
			@warn e.message
			status = "FAILURE_UNBOUNDED_OBJECTIVE"
		elseif isa(e, OutOfMemoryError)
			status = "FAILURE_OUT_OF_MEMORY_ERROR"
		else
			@error e
		end
        return x_k, status, iteration_stats, computation_stats, k
    end
    computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
    return x_k, "ITERARION_LIMIT", iteration_stats, computation_stats, k
end

# This is the CAT original algorithm that we inlucde in the NeurIPS paper
function CAT_Original(problem::Problem_Data, x::Vector{Float64}, δ::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
    @assert(δ >= 0)
    MAX_ITERATION = problem.MAX_ITERATION
    MAX_TIME = problem.MAX_TIME
    gradient_termination_tolerance = problem.gradient_termination_tolerance
    β_1 = problem.β_1
    ω = problem.ω
    x_k = x
    δ_k = δ
    r_k = problem.r_1
    γ_2 = problem.γ_2
    nlp = problem.nlp
    θ = problem.θ
    iteration_stats = DataFrame(k = [], deltaval = [], directionval = [], fval = [], gradval = [])
    total_function_evaluation = 0
    total_gradient_evaluation = 0
    total_hessian_evaluation = 0
    total_number_factorizations = 0
    k = 1
    try
        gval_current = grad(nlp, x_k)
		r_k = 0.1 * norm(gval_current, 2)
        fval_current = obj(nlp, x_k)
        total_function_evaluation += 1
        total_gradient_evaluation += 1
        hessian_current = nothing
        compute_hessian = true
        if norm(gval_current, 2) <= gradient_termination_tolerance
            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
            println("*********************************Iteration Count: ", 1)
            push!(iteration_stats, (1, δ, [], fval_current, norm(gval_current, 2)))
            return x_k, "SUCCESS", iteration_stats, computation_stats, 1
        end
        start_time = time()
		min_gval_norm = norm(gval_current, 2)
        while k <= MAX_ITERATION
            if compute_hessian
                hessian_current = hess(nlp, x_k)
                total_hessian_evaluation += 1
            end
            δ_k, d_k, temp_total_number_factorizations = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, subproblem_solver_method)
            fval_next = obj(nlp, x_k + d_k)
            total_function_evaluation += 1
            gval_next = grad(nlp, x_k + d_k)
            total_gradient_evaluation += 1
            ρ_k = compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, d_k, θ)
            if fval_next <= fval_current
                x_k = x_k + d_k
                fval_current = fval_next
                gval_current = gval_next
				min_gval_norm = min(min_gval_norm, norm(gval_current, 2))
                compute_hessian = true
            else
                #else x_k+1 = x_k, fval_current, gval_current, hessian_current will not change
                compute_hessian = false
            end
	   	    if ρ_k <= β_1
	            r_k = norm(d_k, 2) / ω
	        else
	            r_k = ω * norm(d_k, 2)
	        end
	        push!(iteration_stats, (k, δ_k, d_k, fval_current, norm(gval_current, 2)))
	        if norm(gval_next, 2) <= gradient_termination_tolerance
	            push!(iteration_stats, (k, δ_k, d_k, fval_next, norm(gval_next, 2)))
	            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
	            println("*********************************Iteration Count: ", k)
	            return x_k, "SUCCESS", iteration_stats, computation_stats, k
	        end

	        if time() - start_time > MAX_TIME
	            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
	            return x_k, "MAX_TIME", iteration_stats, computation_stats, k
	        end
        	k += 1
        end
    catch e
        @warn e
        computation_stats = Dict("total_function_evaluation" => (MAX_ITERATION + 1), "total_gradient_evaluation" => (MAX_ITERATION + 1), "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => (MAX_ITERATION + 1))
        return x_k, "FAILURE", iteration_stats, computation_stats, (MAX_ITERATION + 1)
    end
    computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
    return x_k, "ITERARION_LIMIT", iteration_stats, computation_stats, (MAX_ITERATION + 1)
end

end # module
