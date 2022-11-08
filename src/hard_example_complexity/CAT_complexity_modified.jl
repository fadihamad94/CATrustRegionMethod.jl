#Main algorithm code goes here
using LinearAlgebra, DataFrames, SparseArrays, Plots
include("../trust_region_subproblem_solver.jl")
include("./hard_example_implementation_code.jl")
include("./complexity_standard_trust_region_example.jl")

mutable struct Problem_Data
    β::Float64
    θ::Float64
    ω::Float64
    r_1::Float64
    MAX_ITERATION::Int64
	MAX_TIME::Float64
    gradient_termination_tolerance::Float64

    # initialize parameters
    function Problem_Data(β::Float64=0.25,
                           θ::Float64=0.5, ω::Float64=2.0, r_1::Float64=0.5,
                           MAX_ITERATION::Int64=100, MAX_TIME::Float64=1800.0, gradient_termination_tolerance::Float64=1e-4)
        @assert(β > 0 && β < 1)
        @assert(β * θ < 1 - β)
        @assert(ω > 1)
        @assert(r_1 > 0)
        @assert(MAX_ITERATION > 0)
		@assert(MAX_TIME > 0.0)
		@assert(gradient_termination_tolerance > 0)
        return new(β, θ, ω, r_1, MAX_ITERATION, MAX_TIME, gradient_termination_tolerance)
    end
end

#Based on Theorem 4.3 in Numerical Optimization by Wright
function optimizeSecondOrderModel(g::Vector{Float64}, H::SparseMatrixCSC{Float64,Int64}, x_k::Vector{Float64}, δ::Float64, ϵ::Float64, r::Float64, k::Int64,  η::Float64)
    #When δ is 0 and the Hessian is positive semidefinite, we can directly compute the direction
    try
        cholesky(Matrix(H))
		d_k = compute_d_k(k, η)
        if norm(d_k, 2) <= r
            return 0.0, d_k
        end
    catch e
    end

    try
        δ, δ_prime = findinterval(g, H, δ, ϵ, r)
        δ_m = bisection(g, H, δ, ϵ, δ_prime, r)
        sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
        d_k = (cholesky(H + δ_m * sparse_identity) \ (-g))
        return δ_m, d_k
    catch e
        if e == ErrorException("Bisection logic failed to find a root for the phi function")
            return solveHardCaseLogic(g, H, r)
        elseif e == ErrorException("Bisection logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0.")
            return solveHardCaseLogic(g, H, r)
        else
            throw(e)
        end
    end
end

function computeSecondOrderModel(f::Float64, g::Vector{Float64}, H::SparseMatrixCSC{Float64,Int64}, d_k::Vector{Float64})
    return f + transpose(g) * d_k + 0.5 * transpose(d_k) * H * d_k
end

function compute_ρ(fval_current::Float64, fval_next::Float64, gval_current::Vector{Float64}, gval_next::Vector{Float64}, H::SparseMatrixCSC{Float64,Int64}, d_k::Vector{Float64}, θ::Float64)
    second_order_model_value_current_iterate = computeSecondOrderModel(fval_current, gval_current, H, d_k)
    guarantee_factor = θ * 0.5 * norm(gval_next, 2) * norm(d_k, 2)

    ρ = (fval_current - fval_next) / (fval_current - second_order_model_value_current_iterate + guarantee_factor)
    return ρ
end

function findAllPointsRleatedToNewtonsSteps(x_0::Vector{Float64}, η::Float64, α_k::Float64, gradient_termination_tolerance::Float64, MAX_ITERATION::Int64)
    x_k, status, iteration_stats = optimize(x_0, η, α_k, gradient_termination_tolerance, MAX_ITERATION)
    return x_k, status, iteration_stats
end

function findIntervalToComputeKForFirstComponentOfInputVectorX(xval, x_1::Float64, k_1_old)
    for i in (k_1_old + 1):size(xval)[1]
        if xval[i][1] <= x_1 && xval[i + 1][1] > x_1
            return xval[i][1], i - 1
        end
    end
end

function findIntervalToComputeKForSecondComponentOfInputVectorX(xval, x_2::Float64, k_2_old)
    for i in (k_2_old + 1):size(xval)[1]
        if xval[i][2] <= x_2 && xval[i + 1][2] > x_2
            return xval[i][2], i - 1
        end
    end
end

function CAT(problem::Problem_Data, x_0::Vector{Float64}, η::Float64, α_k::Float64, δ::Float64)
    @assert(δ >= 0)
    MAX_ITERATION = problem.MAX_ITERATION
    MAX_TIME = problem.MAX_TIME
    gradient_termination_tolerance = problem.gradient_termination_tolerance
    β = problem.β
    ω = problem.ω
    x_k = x_0
    x_k_1 = x_k_2 = 0.0
    δ_k = δ
    r_k = problem.r_1
    γ_2 = 0.8
    θ = problem.θ
    iteration_stats = DataFrame(k = [], ρ_val = [], x_val = [], deltaval = [], directionval = [], fval = [], f1val = [], f2val = [], gradval = [])
    total_function_evaluation = 0
    total_gradient_evaluation = 0
    total_hessian_evaluation = 0
    k = 0
    k_1 = k_2 = k
    f2_1_0 = 25000.3
    f2_2_0 = (π ^ 2) / 6
    d_k = Vector{Float64}([0.0, 0.0])

    f1_k_current = f2_1_0
    f2_2_comma_k_current = f2_2_0

    f1_k_next = f1_k_current
    f2_2_comma_k_next = f2_2_comma_k_current
    fval_next = 0
	iteration_stats_newtons_method = nothing
    try
        #First we need to find all the points that Newton's method or here standard trust-region will go through
        temp_x, temp_status, iteration_stats_newtons_method = findAllPointsRleatedToNewtonsSteps(x_0, η, α_k, 0.5 * gradient_termination_tolerance, 10 * MAX_ITERATION)
        allPossiblePoints =  iteration_stats_newtons_method.xval

        gval_current = compute_g2_close_form(x_k, x_k, η, k, k, α_k)
        f_1, f_1_k = compute_f1_close_form_k(x_k[1], x_k[1], η, k, α_k, f2_1_0)
        f_2, f_2_k  = compute_f2_2coma_close_form_k(x_k[2], x_k[2], η, k, f2_2_0)
		f_1_k = f2_1_0
		f_2_k = f2_2_0
        fval_current = f_1 + f_2

        total_function_evaluation += 1
        total_gradient_evaluation += 1
        hessian_current = nothing
        compute_hessian = true
        if norm(gval_current, 2) <= gradient_termination_tolerance
            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation)
            println("*********************************Iteration Count: ", 1)
            push!(iteration_stats, (1, 1, x_k, δ, [], fval_current, f_1, f_2, norm(gval_current, 2)))
            return x_k, "SUCCESS", iteration_stats, computation_stats, 1
        end

        start_time = time()
        while k <= MAX_ITERATION
            if compute_hessian
                hessian_current = compute_H2_close_form(x_k, [x_k_1, x_k_2], η, k_1, k_2, α_k)
                total_hessian_evaluation += 1
            end
			old_k_1 = k_1
			old_k_2 = k_2
            δ_k, d_k = optimizeSecondOrderModel(gval_current, hessian_current, x_k, δ_k, γ_2, r_k, k, η)
			x_k_1, k_1 = findIntervalToComputeKForFirstComponentOfInputVectorX(allPossiblePoints, x_k[1] + d_k[1], old_k_1)
            x_k_2, k_2 = findIntervalToComputeKForSecondComponentOfInputVectorX(allPossiblePoints, x_k[2] + d_k[2], old_k_2)
			if θ == 0
				f_1 = coumpute_f1_next_iterate_value(f_1, η, k_1 - 1, α_k)
				f_2 = coumpute_f2_2_next_iterate_value(f_2, η, k_2 - 1)
			else
				f_1, f_1_k_temp = compute_f1_close_form_k(x_k[1] + d_k[1], x_k_1, η, k_1, α_k, f2_1_0)
	            f_2, f_2_k_temp = compute_f2_2coma_close_form_k(x_k[2] + d_k[2], x_k_2, η, k_2, f2_2_0)
			end
            fval_next = f_1 + f_2

            total_function_evaluation += 1
            gval_next = compute_g2_close_form(x_k + d_k, [x_k_1, x_k_2], η, k_1, k_2, α_k)
            total_gradient_evaluation += 1
			ρ_k = compute_ρ(fval_current, fval_next, gval_current, gval_next, hessian_current, d_k, θ)
			push!(iteration_stats, (k, ρ_k, x_k, δ_k, d_k, fval_current, f_1, f_2, norm(gval_current, 2)))
            if fval_next <= fval_current
                x_k = x_k + d_k
                fval_current = fval_next
                gval_current = gval_next
                compute_hessian = true
            else
                #else x_k+1 = x_k, fval_current, gval_current, hessian_current will not change
                compute_hessian = false
            end

            if ρ_k < β
                r_k = norm(d_k, 2) / ω
            else
                r_k = ω * norm(d_k, 2)
            end

            if norm(gval_next, 2) <= gradient_termination_tolerance
                push!(iteration_stats, (k, ρ_k, x_k, δ_k, d_k, fval_next, f_1, f_2, norm(gval_next, 2)))
                computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation)
                println("*********************************Iteration Count: ", k)
                return x_k, "SUCCESS", iteration_stats, computation_stats, k, iteration_stats_newtons_method
            end

            if time() - start_time > MAX_TIME
                computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation)
                return x_k, "MAX_TIME", iteration_stats, computation_stats, k, iteration_stats_newtons_method
            end

            k += 1
        end
    catch e
        @warn e
        computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation)
        return x_k, "FAILURE", iteration_stats, computation_stats, k
    end
    computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation)
    return x_k, "ITERARION_LIMIT", iteration_stats, computation_stats, (MAX_ITERATION + 1), iteration_stats_newtons_method
end
