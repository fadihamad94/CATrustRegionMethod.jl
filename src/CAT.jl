__precompile__()
#Main algorithm code goes here
module consistently_adaptive_trust_region_method
using NLPModels, LinearAlgebra, DataFrames, SparseArrays
include("./trust_region_subproblem_solver.jl")

export Problem_Data
export phi, findinterval, bisection, restoreFullMatrix, computeSecondOrderModel, optimizeSecondOrderModel, compute_ρ_hat, CAT

mutable struct Problem_Data
    nlp::AbstractNLPModel
    β_1::Float64
    θ::Float64
    ω::Float64
    r_1::Float64
    MAX_ITERATION::Int64
    gradient_termination_tolerance::Float64
    MAX_TIME::Float64
    γ_2::Float64

    # initialize parameters
    function Problem_Data(nlp::AbstractNLPModel, β_1::Float64=0.1,
                           θ::Float64=0.1, ω::Float64=8.0, r_1::Float64=1.0,
                           MAX_ITERATION::Int64=10000, gradient_termination_tolerance::Float64=1e-5,
                           MAX_TIME::Float64=30 * 60.0, γ_2::Float64=0.8)
        @assert(β_1 > 0 && β_1 < 1)
        @assert(θ >= 0 && θ < 1)
        @assert(β_1 * θ < 1 - β_1)
        @assert(ω > 1)
        @assert(r_1 > 0)
        @assert(MAX_ITERATION > 0)
        @assert(MAX_TIME > 0)
        @assert(γ_2 > (1 / ω) && γ_2 <= 1)
        return new(nlp, β_1, θ, ω, r_1, MAX_ITERATION, gradient_termination_tolerance, MAX_TIME, γ_2)
    end
end

function computeSecondOrderModel(f::Float64, g::Vector{Float64}, H, d_k::Vector{Float64})
    return f + transpose(g) * d_k + 0.5 * transpose(d_k) * H * d_k
end

function compute_ρ_hat(fval_current::Float64, fval_next::Float64, gval_current::Vector{Float64}, gval_next::Vector{Float64}, H, x_k::Vector{Float64}, d_k::Vector{Float64}, θ::Float64)
    second_order_model_value_current_iterate = computeSecondOrderModel(fval_current, gval_current, H, d_k)
    guarantee_factor = θ * 0.5 * norm(gval_next, 2) * norm(d_k, 2)
    ρ_hat = (fval_current - fval_next) / (fval_current - second_order_model_value_current_iterate + guarantee_factor)
    return ρ_hat
end

function compute_ρ_standard_trust_region_method(fval_current::Float64, fval_next::Float64, gval_current::Vector{Float64}, H, d_k::Vector{Float64})
    second_order_model_value_current_iterate = computeSecondOrderModel(fval_current, gval_current, H, d_k)
    ρ = (fval_current - fval_next) / (fval_current - second_order_model_value_current_iterate)
    return ρ
end

#This logic is based on "Combining Trust Region and Line Search Techniques*" by Jorge Nocedal and Ya-xiang Yuan.
function linesearch(nlp, α_k::Float64, fval_current::Float64, x_k::Vector{Float64}, d_k::Vector{Float64})
	fval_next = -1.0
	for i in 1:100
		d_k_i = α_k ^ i * d_k
		fval_next = obj(nlp, x_k + d_k_i)
		if fval_next <= fval_current
			return true, fval_next, d_k_i, i
		end
	end
	return false, NaN, d_k, 100
end

function acceptStepCheckCondition(fval_current::Float64, gval_current::Vector{Float64}, fval_next::Float64, gval_next::Vector{Float64}, min_gval_norm::Float64)
	modification_1_condition = (norm(gval_next, 2) <= 0.8 * min_gval_norm)
	return fval_next <= fval_current || modification_1_condition
	# return fval_next <= fval_current
end

function acceptStepCheckCondition_standard_trust_region_method(fval_current::Float64, fval_next::Float64, gval_current::Vector{Float64}, H, d_k::Vector{Float64}, β::Float64)
	ρ = compute_ρ_standard_trust_region_method(fval_current, fval_next, gval_current, H, d_k)
	return ρ >= β
end

function computeCandidateSolution(nlp,
								  fval_current::Float64,
	 							  gval_current::Vector{Float64},
								  hessian_current,
								  x_k::Vector{Float64},
								  δ::Float64,
								  γ_2::Float64,
								  r_k::Float64,
								  min_gval_norm::Float64,
								  θ::Float64,
								  subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
	δ_k, d_k, temp_total_number_factorizations = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ, γ_2, r_k, min_gval_norm, subproblem_solver_method)
	fval_next = obj(nlp, x_k + d_k)
	temp_total_function_evaluation = 1
	gval_next = grad(nlp, x_k + d_k)
	temp_total_gradient_evaluation = 1
	ρ_k = compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, x_k, d_k, θ)
	return fval_next, gval_next, ρ_k, δ_k, d_k, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations
end

function findInitialRadiusUsingBisection(nlp,
								 fval_current::Float64,
								 gval_current::Vector{Float64},
								 hessian_current,
								 x_k::Vector{Float64},
								 δ::Float64,
								 θ::Float64,
								 γ_2::Float64,
								 r_lower::Float64,
								 r_upper::Float64,
								 total_function_evaluation::Int64,
								 total_gradient_evaluation::Int64,
								 total_number_factorizations::Int64,
								 min_gval_norm::Float64,
								 subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)

	# @show "1-----------"
	abort = (r_upper - r_lower < 0.1)
	if abort
		return false, fval_current, gval_current, 0, 0, zeros(length(gval_current)), r_upper, total_function_evaluation, total_gradient_evaluation, total_number_factorizations
	end
	# @show "2-----------"
	r_k_temp = (r_lower + r_upper) / 2
	fval_next, gval_next, ρ_k, δ_k, d_k, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations = computeCandidateSolution(nlp, fval_current, gval_current, hessian_current, x_k, δ, γ_2, r_k_temp, min_gval_norm, θ, subproblem_solver_method)
	# @show "3-----------"
	total_function_evaluation += temp_total_function_evaluation
	total_gradient_evaluation += temp_total_gradient_evaluation
	total_number_factorizations += temp_total_number_factorizations
	# @show "4-----------"
	acceptStepCheckCondition_ = acceptStepCheckCondition(fval_current, gval_current, fval_next, gval_next, min_gval_norm)
	# @show "5-----------"
	if acceptStepCheckCondition_
		# @show "6-----------"
		return true, fval_next, gval_next, ρ_k, δ_k, d_k, r_k_temp, total_function_evaluation, total_gradient_evaluation, total_number_factorizations
	else
		# @show "7-----------"
		return findInitialRadiusUsingBisection(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_k_temp, r_upper, total_function_evaluation, total_gradient_evaluation, total_number_factorizations, min_gval_norm, subproblem_solver_method)
	end
end

function findRadiusUsingBisection(nlp,
								 fval_current::Float64,
								 gval_current::Vector{Float64},
								 hessian_current,
								 x_k::Vector{Float64},
								 δ::Float64,
								 θ::Float64,
								 γ_2::Float64,
								 r_lower::Float64,
								 r_upper::Float64,
								 total_function_evaluation::Int64,
								 total_gradient_evaluation::Int64,
								 total_number_factorizations::Int64,
								 min_gval_norm::Float64,
								 β_1::Float64,
								 β_2::Float64,
								 itr::Int64,
								 subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)

		# r_k_temp = (r_lower + r_upper) / 2
		r_k_temp = sqrt(r_lower * r_upper)
		fval_next, gval_next, ρ_k, δ_k, d_k, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations = computeCandidateSolution(nlp, fval_current, gval_current, hessian_current, x_k, δ, γ_2, r_k_temp, min_gval_norm, θ, subproblem_solver_method)
		total_function_evaluation += temp_total_function_evaluation
		total_gradient_evaluation += temp_total_gradient_evaluation
		total_number_factorizations += temp_total_number_factorizations
		stop = itr == 0
		acceptStepCheckCondition_ = acceptStepCheckCondition(fval_current, gval_current, fval_next, gval_next, min_gval_norm)
		if acceptStepCheckCondition_ || stop
			return stop, fval_next, gval_next, ρ_k, δ_k, d_k, r_k_temp, total_function_evaluation, total_gradient_evaluation, total_number_factorizations
		elseif ρ_k < β_1
			r_upper = r_k_temp
			itr -= 1
			return findRadiusUsingBisection(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_lower, r_upper, total_function_evaluation, total_gradient_evaluation, total_number_factorizations, min_gval_norm, β_1, β_2, itr, subproblem_solver_method)
		else
			r_lower = r_k_temp
			itr -= 1
			return findRadiusUsingBisection(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_lower, r_upper, total_function_evaluation, total_gradient_evaluation, total_number_factorizations, min_gval_norm, β_1, β_2, itr, subproblem_solver_method)
		end
end

function searchRadiusForSpecificBetaRange(nlp,
										 fval_current::Float64,
										 gval_current::Vector{Float64},
										 hessian_current,
										 x_k::Vector{Float64},
										 δ::Float64,
										 θ::Float64,
										 γ_2::Float64,
										 r_k::Float64,
										 min_gval_norm::Float64,
										 β::Float64,
										 inequality::String,
										 factor::Int64,
										 subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
	@show "searchRadiusForSpecificBetaRange: Search for initial radius"
	total_function_evaluation = 0
	total_gradient_evaluation = 0
	total_number_factorizations = 0
	fval_next_, gval_next_, ρ_k_temp, δ_k_, d_k_, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations = computeCandidateSolution(nlp, fval_current, gval_current, hessian_current, x_k, δ, γ_2, r_k, min_gval_norm, θ, subproblem_solver_method)
	@show "ρ_k_temp for the trial radius $r_k is $ρ_k_temp and factor is $factor"
	total_function_evaluation += temp_total_function_evaluation
	total_gradient_evaluation += temp_total_gradient_evaluation
	total_number_factorizations += temp_total_number_factorizations
	factor = factor
	if inequality == "greater_than"
		factor = 1 / factor
	end
	# if inequality == "less_than"
	# 	factor = 1 / factor
	# end
	stop = false
	itr = 5
	max_allowed_radius = 1e+6
	min_allowed_radius = 1e-6
	while !stop
		r_k = r_k * factor
		fval_next_, gval_next_, ρ_k_temp, δ_k_, d_k_, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations = computeCandidateSolution(nlp, fval_current, gval_current, hessian_current, x_k, δ, γ_2, r_k, min_gval_norm, θ, subproblem_solver_method)
		@show "searching for $inequality $β"
		@show "ρ_k_temp for the trial radius $r_k is $ρ_k_temp"
		norm_dk = norm(d_k_, 2)
		@show "search direction norm ||d_k_|| is $norm_dk"
		total_function_evaluation += temp_total_function_evaluation
		total_gradient_evaluation += temp_total_gradient_evaluation
		total_number_factorizations += temp_total_number_factorizations
		condition = ρ_k_temp <= β
		if inequality == "less_than"
			if 0 <= ρ_k_temp <= β || norm_dk < r_k
				stop = true
				break
			end
		else
			if ρ_k_temp >= β
				stop = true
				break
			end
		end
		itr -= 1
		stop = itr == 0
	end
	return r_k, total_function_evaluation, total_gradient_evaluation, total_number_factorizations
end

function searchRadiusBisectionInterval(nlp,
									  fval_current::Float64,
									  gval_current::Vector{Float64},
									  hessian_current,
									  x_k::Vector{Float64},
									  δ::Float64,
									  θ::Float64,
									  γ_2::Float64,
									  r_k::Float64,
									  min_gval_norm::Float64,
									  β_1::Float64,
									  β_2::Float64,
									  factor::Int64,
									  subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
	@show "searchRadiusBisectionInterval: Search for initial radius"
	fval_next_, gval_next_, ρ_k_temp, δ_k_, d_k_, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations = computeCandidateSolution(nlp, fval_current, gval_current, hessian_current, x_k, δ, γ_2, r_k, min_gval_norm, θ, subproblem_solver_method)
	candidate_iterate_data = (fval_next_, gval_next_, ρ_k_temp, δ_k_, d_k_)
	r_k_l = r_k
	r_k_u = r_k
	if ρ_k_temp > β_2 && norm(d_k_, 2) <= r_k
		@show "rho is greater than $β_2 and search direction norm is less than the radius. Exiting as increasing the radius will not get better solution."
		return r_k_l, r_k_u, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations, candidate_iterate_data
	end

	@show "searchRadiusBisectionInterval: ρ_k_temp for the first trial radius is $ρ_k_temp and factor is $factor"
	total_function_evaluation_l = 0
	total_gradient_evaluation_l = 0
	total_number_factorizations_l = 0
	total_function_evaluation_u = 0
	total_gradient_evaluation_u = 0
	total_number_factorizations_u = 0

	#Case (1) ρ_k(r_k) ∈ [β_1, β_2]
	#Case (2) ρ_k(r_k) < β_1
	#Case (3) ρ_k(r_k) > β_2
	r_k_l = r_k
	r_k_u = r_k
	candidate_iterate_data = ()
	if ρ_k_temp ∈ [β_1, β_2]
		#No need to search for better radius actually - just accpet this radius
		candidate_iterate_data = (fval_next_, gval_next_, ρ_k_temp, δ_k_, d_k_)
		# r_k_u, total_function_evaluation_u, total_gradient_evaluation_u, total_number_factorizations_u = searchRadiusForSpecificBetaRange(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_k, min_gval_norm, β_1, "less_than", factor, subproblem_solver_method)
		# r_k_l, total_function_evaluation_l, total_gradient_evaluation_l, total_number_factorizations_l = searchRadiusForSpecificBetaRange(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_k, min_gval_norm, β_2, "greater_than", factor, subproblem_solver_method)
	elseif 0 <= ρ_k_temp < β_1
		r_k_l, total_function_evaluation_l, total_gradient_evaluation_l, total_number_factorizations_l = searchRadiusForSpecificBetaRange(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_k, min_gval_norm, β_2, "greater_than", factor, subproblem_solver_method)
 	else
		r_k_u, total_function_evaluation_u, total_gradient_evaluation_u, total_number_factorizations_u = searchRadiusForSpecificBetaRange(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_k, min_gval_norm, β_1, "less_than", factor, subproblem_solver_method)
	end
	# if ρ_k_temp ∈ [β_1, β_2]
	# 	r_k_u, total_function_evaluation_u, total_gradient_evaluation_u, total_number_factorizations_u = searchRadiusForSpecificBetaRange(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_k, min_gval_norm, β_1, "less_than", factor, subproblem_solver_method)
	# 	r_k_l, total_function_evaluation_l, total_gradient_evaluation_l, total_number_factorizations_l = searchRadiusForSpecificBetaRange(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_k, min_gval_norm, β_2, "greater_than", factor, subproblem_solver_method)
	# elseif ρ_k_temp ∈ [0, β_1]
	# 	r_k_l, total_function_evaluation_l, total_gradient_evaluation_l, total_number_factorizations_l = searchRadiusForSpecificBetaRange(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_k, min_gval_norm, β_2, "greater_than", factor, subproblem_solver_method)
 	# elseif ρ_k_temp < 0 || ρ_k_temp > β_2
	# 	r_k_u, total_function_evaluation_u, total_gradient_evaluation_u, total_number_factorizations_u = searchRadiusForSpecificBetaRange(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_k, min_gval_norm, β_1, "less_than", factor, subproblem_solver_method)
	# end
	temp_total_function_evaluation += total_function_evaluation_l
	temp_total_gradient_evaluation += total_gradient_evaluation_l
	temp_total_number_factorizations += total_number_factorizations_l
	temp_total_function_evaluation += total_function_evaluation_u
	temp_total_gradient_evaluation += total_gradient_evaluation_u
	temp_total_number_factorizations += total_number_factorizations_u
	return r_k_l, r_k_u, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations, candidate_iterate_data
end

function searchRadius(nlp,
					 fval_current::Float64,
					 gval_current::Vector{Float64},
					 hessian_current,
					 x_k::Vector{Float64},
					 δ::Float64,
					 θ::Float64,
					 γ_2::Float64,
					 r_k::Float64,
					 min_gval_norm::Float64,
					 β_1::Float64,
					 β_2::Float64,
					 factor::Int64,
					 subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
	@show "searchRadius: Search for initial radius"
	r_k_l, r_k_u, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations, candidate_iterate_data = searchRadiusBisectionInterval(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_k, min_gval_norm, β_1, β_2, factor, subproblem_solver_method)
	@show "r_k_l, r_k_u, $r_k_l, $r_k_u"
	if r_k_l == r_k_u
		fval_next, gval_next, ρ_k, δ_k, d_k = candidate_iterate_data
		r_k_temp = r_k_l
		return fval_next, gval_next, ρ_k, δ_k, d_k, r_k_temp, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations
	end
	itr = 10
	stop, fval_next, gval_next, ρ_k, δ_k, d_k, r_k_temp, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations = findRadiusUsingBisection(nlp, fval_current, gval_current, hessian_current, x_k, δ, θ, γ_2, r_k_l, r_k_u, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations, min_gval_norm, β_1, β_2, itr, subproblem_solver_method)
	@show "stop is $stop"
	return fval_next, gval_next, ρ_k, δ_k, d_k, r_k_temp, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations
end

function CAT(problem::Problem_Data, x::Vector{Float64}, δ::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
    @assert(δ >= 0)
    MAX_ITERATION = problem.MAX_ITERATION
    MAX_TIME = problem.MAX_TIME
    gradient_termination_tolerance = problem.gradient_termination_tolerance
    β_1 = problem.β_1
	β_1 = 0.25 #value based on paper sensitivity of trust region algorithms to their parameter
	β_2 = 0.75 #value based on paper sensitivity of trust region algorithms to their parameter
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
	max_iteration_increase_initial_radius = 10
    try
        gval_current = grad(nlp, x_k)
		r_k = 0.1 * norm(gval_current, 2) #(BEST)
		# r_k = 10 * norm(gval_current, 2)
		initial_radius = 10.0
		# initial_radius = r_k
		r_k = initial_radius
        fval_current = obj(nlp, x_k)
		fval_0 = fval_current
        total_function_evaluation += 1
        total_gradient_evaluation += 1
        #hessian_current = nothing
		hessian_current = restoreFullMatrix(hess(nlp, x_k))
        total_hessian_evaluation += 1
		# r_k = (0.1 * norm(gval_current, 2)) / norm(hessian_current, 2)
        compute_hessian = false
        if norm(gval_current, 2) <= gradient_termination_tolerance
            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
            println("*********************************Iteration Count: ", 1)
            push!(iteration_stats, (1, δ, [], fval_current, norm(gval_current, 2)))
            return x_k, "SUCCESS", iteration_stats, computation_stats, 1
        end
        start_time = time()
		min_gval_norm = norm(gval_current, 2)
		min_fval = fval_current
		factor = 16

		#Finding the initial radius
		fval_next, gval_next, ρ_k, δ_k_next, d_k, r_k, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations = searchRadius(nlp, fval_current, gval_current, hessian_current, x_k, δ_k, θ, γ_2, r_k, min_gval_norm, β_1, β_2, factor, subproblem_solver_method)
		total_function_evaluation += temp_total_function_evaluation
		total_gradient_evaluation += temp_total_gradient_evaluation
		total_number_factorizations += temp_total_number_factorizations
		@show "Running iteration $k with radius $r_k and rho $ρ_k"
        while k <= MAX_ITERATION
			# @show "Running iteration $k with radius $r_k"
            if compute_hessian
                hessian_current = restoreFullMatrix(hess(nlp, x_k))
                total_hessian_evaluation += 1
            end

            # δ_k, d_k, temp_total_number_factorizations = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, subproblem_solver_method)
	    	# total_number_factorizations += temp_total_number_factorizations
            # fval_next = obj(nlp, x_k + d_k)
            # total_function_evaluation += 1
            # gval_next = grad(nlp, x_k + d_k)
            # total_gradient_evaluation += 1
            # ρ_k = compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, x_k, d_k, θ)
			if k > 1
				fval_next, gval_next, ρ_k, δ_k_next, d_k, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations = computeCandidateSolution(nlp, fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, θ, subproblem_solver_method)
				total_function_evaluation += temp_total_function_evaluation
				total_gradient_evaluation += temp_total_gradient_evaluation
				total_number_factorizations += temp_total_number_factorizations
			end
			acceptStepCheckCondition_ = acceptStepCheckCondition(fval_current, gval_current, fval_next, gval_next, min_gval_norm)

			# @show "Running iteration $k with rho $ρ_k and acceptStepCheckCondition_ $acceptStepCheckCondition_"

			# modification_1_condition = (norm(gval_next, 2) <= 0.8 * min_gval_norm)
			# F(x_k + 1) <= f(x_k) + grad_norm * d_k norm * epsilon
            # if fval_next <= fval_current
			# ϵ = 1e-3
			# if fval_next <= fval_current + norm(gval_current) * norm(d_k) * 1e-2
			# if fval_next <= fval_current + (1 / k) * (fval_0 - min_fval)
			# if (fval_next <= fval_current) || ((fval_next <= fval_current + norm(gval_current) * norm(d_k) * 1e-3) && modification_1_condition)
			# if fval_next <= fval_current || modification_1_condition
			if acceptStepCheckCondition_
				# if search_initial_radius
				# 	@show "Searching for initial radius $search_initial_radius"
				# 	@show "Initial radius value $initial_radius"
				# end
				# if search_initial_radius && initial_radius != r_k
				# 	@show "Finding radius $r_k with ρ_k value $ρ_k and factor $factor"
				# 	temp_δ_k, temp_d_k, temp_fval_next, temp_gval_next, temp_ρ_k = δ_k_next, d_k, fval_next, gval_next, ρ_k
				# 	r_lower = r_k * factor
				# 	r_upper = r_k
				# 	@show r_lower
				# 	@show r_upper
				# 	success_radius_search, fval_next, gval_next, ρ_k, δ_k_next, d_k, r_k_temp, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations = findInitialRadiusUsingBisection(nlp, fval_current, gval_current, hessian_current, x_k, δ_k, θ, γ_2, r_lower, r_upper, total_function_evaluation, total_gradient_evaluation, total_number_factorizations, min_gval_norm, subproblem_solver_method)
				# 	@show "success_radius_search $success_radius_search, radius $r_k_temp"
				# 	total_number_factorizations += temp_total_number_factorizations
				# 	total_function_evaluation += temp_total_function_evaluation
				# 	total_gradient_evaluation += temp_total_gradient_evaluation
				# 	r_k = r_k_temp
				# 	if !success_radius_search
				# 		r_k = r_upper
				# 		δ_k_next, d_k, fval_next, gval_next, ρ_k = temp_δ_k, temp_d_k, temp_fval_next, temp_gval_next, temp_ρ_k
				# 	end
				# end
				x_k = x_k + d_k
				fval_current = fval_next
				gval_current = gval_next
				δ_k = δ_k_next
				min_gval_norm = min(min_gval_norm, norm(gval_current, 2))
				min_fval = min(min_fval, fval_current)
				compute_hessian = true
				search_initial_radius = false
            else
				line_search = false
				if line_search
					#This logic is based on "Combining Trust Region and Line Search Techniques*" by Jorge Nocedal and Ya-xiang Yuan.
					α_k = max(0.1, 0.5 / (1 + (fval_current - fval_next) / dot(transpose(d_k), gval_current)))
					α_k = isnan(α_k) ? 0.1 : α_k
					success, fval_next_temp, d_k_i, i = linesearch(nlp, α_k, fval_current, x_k, d_k)
					total_function_evaluation += i
					if success
						x_k = x_k + d_k_i
		                fval_current = fval_next_temp
		                gval_current = grad(nlp, x_k + d_k_i)
						total_gradient_evaluation += 1
						min_gval_norm = min(min_gval_norm, norm(gval_current, 2))
						min_fval = min(min_fval, fval_current)
		                compute_hessian = true
					else
	                	#else x_k+1 = x_k, fval_current, gval_current, hessian_current will not change
	                	compute_hessian = false
						end
				else
					#else x_k+1 = x_k, fval_current, gval_current, hessian_current will not change
					compute_hessian = false
				end
            end
			# if search_initial_radius
			# 	max_iteration_increase_initial_radius -= 1
			# 	if isnan(ρ_k) || ρ_k <= 0
			# 		factor = 0.5
			# 		r_k = r_k * factor
			# 	else
			# 		factor = 2
			# 		r_k = factor * r_k
			# 	end
			# 	@show "Increasing initial radius candidiate value $r_k with ρ_k value $ρ_k"
			# 	if max_iteration_increase_initial_radius == 0
			# 		@show "Failed to search for initial radius"
			# 		r_k = initial_radius
			# 		search_initial_radius = false
			# 	end
			# else

			#CAT Using
			# if isnan(ρ_k) || ρ_k <= β_1
			# 	r_k = norm(d_k, 2) / ω
			# else
		    #     r_k = ω * norm(d_k, 2)
		    # end

			#Standard trust-region method
			if isnan(ρ_k) || ρ_k < β_1
				r_k = norm(d_k, 2) / ω
			elseif ρ_k < β_2
				r_k = r_k
			else
		        r_k = ω * norm(d_k, 2)
		    end

			#=if isnan(ρ_k) || ρ_k <= β_1
					modification_3 = norm(d_k, 2) / 2
					r_k = modification_3
	            	else
					modification_3 = 4 * norm(d_k, 2)
					r_k = modification_3
			end=#
	   	    	#=if isnan(ρ_k) || ρ_k <= 0
			 	modification_3 = norm(d_k, 2) / 4
			elseif ρ_k <= β_1
					modification_3 = norm(d_k, 2) / 2
					r_k = modification_3
	            	else
					modification_3 = 4 * norm(d_k, 2)
					r_k = modification_3
	            	end=#

            push!(iteration_stats, (k, δ_k, d_k, fval_current, norm(gval_current, 2)))
            if norm(gval_next, 2) <= gradient_termination_tolerance
                push!(iteration_stats, (k, δ_k, d_k, fval_next, norm(gval_next, 2)))
                computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
                println("*********************************Iteration Count: ", k)
                return x_k, "SUCCESS", iteration_stats, computation_stats, k
            end

            if time() - start_time > MAX_TIME
                # computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
				computation_stats = Dict("total_function_evaluation" => (MAX_ITERATION + 1), "total_gradient_evaluation" => (MAX_ITERATION + 1), "total_hessian_evaluation" => (MAX_ITERATION + 1), "total_number_factorizations" => (MAX_ITERATION + 1))
                return x_k, "MAX_TIME", iteration_stats, computation_stats, k
            end

            k += 1
			# end
        end
    catch e
        @warn e
		@show e
        computation_stats = Dict("total_function_evaluation" => (MAX_ITERATION + 1), "total_gradient_evaluation" => (MAX_ITERATION + 1), "total_hessian_evaluation" => (MAX_ITERATION + 1), "total_number_factorizations" => (MAX_ITERATION + 1))
        return x_k, "FAILURE", iteration_stats, computation_stats, (MAX_ITERATION + 1)
    end
    computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
	return x_k, "ITERARION_LIMIT", iteration_stats, computation_stats, (MAX_ITERATION + 1)
end


function CAT_similar_trust_region_method(problem::Problem_Data, x::Vector{Float64}, δ::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
    @assert(δ >= 0)
    MAX_ITERATION = problem.MAX_ITERATION
    MAX_TIME = problem.MAX_TIME
    gradient_termination_tolerance = problem.gradient_termination_tolerance
    β_1 = problem.β_1
	β_1 = 0.25 #value based on paper sensitivity of trust region algorithms to their parameter
	β_2 = 0.75 #value based on paper sensitivity of trust region algorithms to their parameter
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
	max_iteration_increase_initial_radius = 10
    try
        gval_current = grad(nlp, x_k)
		r_k = 0.1 * norm(gval_current, 2) #(BEST)
		# r_k = 10 * norm(gval_current, 2)
		# initial_radius = 1.0
		initial_radius = r_k
		r_k = initial_radius
        fval_current = obj(nlp, x_k)
		fval_0 = fval_current
        total_function_evaluation += 1
        total_gradient_evaluation += 1
		hessian_current = restoreFullMatrix(hess(nlp, x_k))
        total_hessian_evaluation += 1
		# r_k = (0.1 * norm(gval_current, 2)) / norm(hessian_current, 2)
        compute_hessian = false
        if norm(gval_current, 2) <= gradient_termination_tolerance
            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
            println("*********************************Iteration Count: ", 1)
            push!(iteration_stats, (1, δ, [], fval_current, norm(gval_current, 2)))
            return x_k, "SUCCESS", iteration_stats, computation_stats, 1
        end
        start_time = time()
		min_gval_norm = norm(gval_current, 2)
		min_fval = fval_current
		factor = 16

		#Finding the initial radius
		fval_next, gval_next, ρ_k, δ_k_next, d_k, r_k, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations = searchRadius(nlp, fval_current, gval_current, hessian_current, x_k, δ_k, θ, γ_2, r_k, min_gval_norm, β_1, β_2, factor, subproblem_solver_method)
		total_function_evaluation += temp_total_function_evaluation
		total_gradient_evaluation += temp_total_gradient_evaluation
		total_number_factorizations += temp_total_number_factorizations
		@show "Running iteration $k with radius $r_k and rho $ρ_k"
        while k <= MAX_ITERATION
			# @show "Running iteration $k with radius $r_k"
            if compute_hessian
                hessian_current = restoreFullMatrix(hess(nlp, x_k))
                total_hessian_evaluation += 1
            end

			if k > 1
				fval_next, gval_next, ρ_k, δ_k_next, d_k, temp_total_function_evaluation, temp_total_gradient_evaluation, temp_total_number_factorizations = computeCandidateSolution(nlp, fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, θ, subproblem_solver_method)
				total_function_evaluation += temp_total_function_evaluation
				total_gradient_evaluation += temp_total_gradient_evaluation
				total_number_factorizations += temp_total_number_factorizations
			end
			# acceptStepCheckCondition_ = acceptStepCheckCondition(fval_current, gval_current, fval_next, gval_next, min_gval_norm)
			acceptStepCheckCondition_ =	acceptStepCheckCondition_standard_trust_region_method(fval_current, fval_next, gval_current, hessian_current, d_k, β_1)
			# @show "Running iteration $k with rho $ρ_k and acceptStepCheckCondition_ $acceptStepCheckCondition_"

			if acceptStepCheckCondition_
				x_k = x_k + d_k
				fval_current = fval_next
				gval_current = gval_next
				δ_k = δ_k_next
				min_gval_norm = min(min_gval_norm, norm(gval_current, 2))
				min_fval = min(min_fval, fval_current)
				compute_hessian = true
				search_initial_radius = false
            else
				#else x_k+1 = x_k, fval_current, gval_current, hessian_current will not change
				compute_hessian = false
            end

			#Standard trust-region method
			if isnan(ρ_k) || ρ_k < β_1
				r_k = norm(d_k, 2) / ω
			elseif ρ_k < β_2
				r_k = r_k
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
				computation_stats = Dict("total_function_evaluation" => (MAX_ITERATION + 1), "total_gradient_evaluation" => (MAX_ITERATION + 1), "total_hessian_evaluation" => (MAX_ITERATION + 1), "total_number_factorizations" => (MAX_ITERATION + 1))
                return x_k, "MAX_TIME", iteration_stats, computation_stats, k
            end

            k += 1
        end
    catch e
        @warn e
		@show e
        computation_stats = Dict("total_function_evaluation" => (MAX_ITERATION + 1), "total_gradient_evaluation" => (MAX_ITERATION + 1), "total_hessian_evaluation" => (MAX_ITERATION + 1), "total_number_factorizations" => (MAX_ITERATION + 1))
        return x_k, "FAILURE", iteration_stats, computation_stats, (MAX_ITERATION + 1)
    end
    computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
	return x_k, "ITERARION_LIMIT", iteration_stats, computation_stats, (MAX_ITERATION + 1)
end


# This is the CAT algorithm wtih all improvements and tricks
function CAT_improvements_tricks(problem::Problem_Data, x::Vector{Float64}, δ::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
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
	#r_k = 0.1 * norm(gval_current, 2) #(BEST)
        fval_current = obj(nlp, x_k)
        total_function_evaluation += 1
        total_gradient_evaluation += 1
        #hessian_current = nothing
	hessian_current = restoreFullMatrix(hess(nlp, x_k))
        total_hessian_evaluation += 1
	r_k = norm(gval_current, 2) / norm(hessian_current, 2)
        compute_hessian = false
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
                hessian_current = restoreFullMatrix(hess(nlp, x_k))
                total_hessian_evaluation += 1
            end

            δ_k, d_k, temp_total_number_factorizations = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, subproblem_solver_method)
	    	total_number_factorizations += temp_total_number_factorizations
            fval_next = obj(nlp, x_k + d_k)
            total_function_evaluation += 1
            gval_next = grad(nlp, x_k + d_k)
            total_gradient_evaluation += 1
            ρ_k = compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, x_k, d_k, θ)

	    	modification_1_condition = (norm(gval_next, 2) <= 0.8 * min_gval_norm)

            if fval_next <= fval_current || modification_1_condition
                x_k = x_k + d_k
                fval_current = fval_next
                gval_current = gval_next
				min_gval_norm = min(min_gval_norm, norm(gval_current, 2))
                compute_hessian = true
            else
		line_search = false
		if line_search
				#This logic is based on "Combining Trust Region and Line Search Techniques*" by Jorge Nocedal and Ya-xiang Yuan.
				α_k = max(0.1, 0.5 / (1 + (fval_current - fval_next) / dot(transpose(d_k), gval_current)))
				α_k = isnan(α_k) ? 0.1 : α_k
				success, fval_next_temp, d_k_i, i = linesearch(nlp, α_k, fval_current, x_k, d_k)
				total_function_evaluation += i
				if success
					x_k = x_k + d_k_i
	                fval_current = fval_next_temp
	                gval_current = grad(nlp, x_k + d_k_i)
					total_gradient_evaluation += 1
					min_gval_norm = min(min_gval_norm, norm(gval_current, 2))
	                compute_hessian = true
				else
                	#else x_k+1 = x_k, fval_current, gval_current, hessian_current will not change
                	compute_hessian = false
				end
		else
			#else x_k+1 = x_k, fval_current, gval_current, hessian_current will not change
			compute_hessian = false
		end
            end
		if isnan(ρ_k) || ρ_k <= β_1
				modification_3 = norm(d_k, 2) / 2
				r_k = modification_3
            	else
				modification_3 = 4 * norm(d_k, 2)
				r_k = modification_3
		end
   	    	#=if isnan(ρ_k) || ρ_k <= 0
		 	modification_3 = norm(d_k, 2) / 4
		elseif ρ_k <= β_1
				modification_3 = norm(d_k, 2) / 2
				r_k = modification_3
            	else
				modification_3 = 4 * norm(d_k, 2)
				r_k = modification_3
            	end=#

            push!(iteration_stats, (k, δ_k, d_k, fval_current, norm(gval_current, 2)))
            if norm(gval_next, 2) <= gradient_termination_tolerance
                push!(iteration_stats, (k, δ_k, d_k, fval_next, norm(gval_next, 2)))
                computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
                println("*********************************Iteration Count: ", k)
                return x_k, "SUCCESS", iteration_stats, computation_stats, k
            end

            if time() - start_time > MAX_TIME
                # computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
				computation_stats = Dict("total_function_evaluation" => (MAX_ITERATION + 1), "total_gradient_evaluation" => (MAX_ITERATION + 1), "total_hessian_evaluation" => (MAX_ITERATION + 1), "total_number_factorizations" => (MAX_ITERATION + 1))
                return x_k, "MAX_TIME", iteration_stats, computation_stats, k
            end

            k += 1
        end
    catch e
        @warn e
        computation_stats = Dict("total_function_evaluation" => (MAX_ITERATION + 1), "total_gradient_evaluation" => (MAX_ITERATION + 1), "total_hessian_evaluation" => (MAX_ITERATION + 1), "total_number_factorizations" => (MAX_ITERATION + 1))
        return x_k, "FAILURE", iteration_stats, computation_stats, (MAX_ITERATION + 1)
    end
    computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "total_number_factorizations" => total_number_factorizations)
    return x_k, "ITERARION_LIMIT", iteration_stats, computation_stats, (MAX_ITERATION + 1)
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
                hessian_current = restoreFullMatrix(hess(nlp, x_k))
                total_hessian_evaluation += 1
            end
            δ_k, d_k, temp_total_number_factorizations = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, subproblem_solver_method)
	    	total_number_factorizations += temp_total_number_factorizations
            fval_next = obj(nlp, x_k + d_k)
            total_function_evaluation += 1
            gval_next = grad(nlp, x_k + d_k)
            total_gradient_evaluation += 1
            ρ_k = compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, x_k, d_k, θ)
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

# This is the CAT algorithm wtih a trick to make the radius problem dependent: r_1 = 0.1 * ||▽f(x_1)||
function CAT_initial_radius_trick(problem::Problem_Data, x::Vector{Float64}, δ::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
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
		r_k = 0.1 * norm(gval_current, 2) #(BEST)
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
                hessian_current = restoreFullMatrix(hess(nlp, x_k))
                total_hessian_evaluation += 1
            end
            δ_k, d_k, temp_total_number_factorizations = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, subproblem_solver_method)
	    	total_number_factorizations += temp_total_number_factorizations
            fval_next = obj(nlp, x_k + d_k)
            total_function_evaluation += 1
            gval_next = grad(nlp, x_k + d_k)
            total_gradient_evaluation += 1
            ρ_k = compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, x_k, d_k, θ)
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

# This is the CAT algorithm wtih a trick to update the dadius by using ω = 8 when increasing the dadius and ω = 4 when decreasing the radius
function CAT_radius_update_trick(problem::Problem_Data, x::Vector{Float64}, δ::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
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
                hessian_current = restoreFullMatrix(hess(nlp, x_k))
                total_hessian_evaluation += 1
            end
            δ_k, d_k, temp_total_number_factorizations = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, subproblem_solver_method)
	    	total_number_factorizations += temp_total_number_factorizations
            fval_next = obj(nlp, x_k + d_k)
            total_function_evaluation += 1
            gval_next = grad(nlp, x_k + d_k)
            total_gradient_evaluation += 1
            ρ_k = compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, x_k, d_k, θ)

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
				modification_3 = norm(d_k, 2) / 4
				r_k = modification_3
            else
				modification_3 = 8 * norm(d_k, 2)
				r_k = modification_3
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

# This is the CAT algorithm wtih a trick for the non monotone trust-region method. The point here is to accept the step even if the function value didn't decrease conditioned on the fact that ||▽f(x_x + d_k)|| <= 0.8 * ||▽f(x_x)|| ∀ t ≦ k
function CAT_non_monotone(problem::Problem_Data, x::Vector{Float64}, δ::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
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
                hessian_current = restoreFullMatrix(hess(nlp, x_k))
                total_hessian_evaluation += 1
            end

            δ_k, d_k, temp_total_number_factorizations = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, min_gval_norm, subproblem_solver_method)
	    	total_number_factorizations += temp_total_number_factorizations
            fval_next = obj(nlp, x_k + d_k)
            total_function_evaluation += 1
            gval_next = grad(nlp, x_k + d_k)
            total_gradient_evaluation += 1
            ρ_k = compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, x_k, d_k, θ)

	    	modification_1_condition = (norm(gval_next, 2) <= 0.8 * min_gval_norm)

            if fval_next <= fval_current || modification_1_condition
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
