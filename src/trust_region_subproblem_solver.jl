export phi, findinterval, bisection
using LinearAlgebra
using Dates
using DelimitedFiles
using Distributions

"""
solveTrustRegionSubproblem(problem_name, f, g, H, x, δ, γ_1, γ_2, r, min_grad, print_level)
See optimizeSecondOrderModel
"""
function solveTrustRegionSubproblem(
    problem_name::String,
    f::Float64,
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
    },
    x_k::Vector{Float64},
    δ::Float64,
    γ_1::Float64,
    γ_2::Float64,
    r::Float64,
    min_grad::Float64,
    algorithm_counter::AlgorithmCounter,
    print_level::Int64 = 0,
)
    original_δ = δ
    success_subproblem_solve,
    δ_k,
    δ_prime_k,
    d_k,
    temp_total_number_factorizations,
    hard_case,
    temp_total_number_factorizations_findinterval,
    temp_total_number_factorizations_bisection,
    temp_total_number_factorizations_compute_search_direction,
    temp_total_number_factorizations_inverse_power_iteration =
        optimizeSecondOrderModel(problem_name, g, H, δ, γ_1, γ_2, r, min_grad, print_level)

    increment!(
        algorithm_counter,
        :total_number_factorizations,
        temp_total_number_factorizations,
    )
    increment!(
        algorithm_counter,
        :total_number_factorizations_findinterval,
        temp_total_number_factorizations_findinterval,
    )
    increment!(
        algorithm_counter,
        :total_number_factorizations_bisection,
        temp_total_number_factorizations_bisection,
    )
    increment!(
        algorithm_counter,
        :total_number_factorizations_compute_search_direction,
        temp_total_number_factorizations_compute_search_direction,
    )
    increment!(
        algorithm_counter,
        :total_number_factorizations_inverse_power_iteration,
        temp_total_number_factorizations_inverse_power_iteration,
    )
    @assert algorithm_counter.total_number_factorizations ==
            algorithm_counter.total_number_factorizations_findinterval +
            algorithm_counter.total_number_factorizations_bisection +
            algorithm_counter.total_number_factorizations_compute_search_direction +
            algorithm_counter.total_number_factorizations_inverse_power_iteration

    γ_3 = 0.5
    if print_level >= 2
        println("success_subproblem_solve is $success_subproblem_solve.")
        norm_d_k = norm(d_k)
        println(
            "----------validating solution for ||d_k|| = $norm_d_k and δ_prime_k = $δ_prime_k with original_δ, δ_k, δ_prime_k = $original_δ, $δ_k, $δ_prime_k and γ_1, γ_2, γ_3, r, min_grad = $γ_1, $γ_2, $γ_3, $r, $min_grad----------",
        )
        println(
            "----------validating solution  with δ_k * ||d_k|| = $(δ_k * norm_d_k) and γ_1 * min_grad = $(γ_1 * min_grad)",
        )
    end
    
    validateTrustRegionSubproblemTerminationCriteria(
        problem_name,
        d_k,
        g,
        H,
        original_δ,
        δ_k,
        δ_prime_k,
        γ_1,
        γ_2,
        γ_3,
        r,
        min_grad,
        hard_case,
        print_level,
    )
    return true, δ_k, d_k, hard_case
end

"""
  computeSearchDirection(problem_name, g, H, δ, γ_1, γ_2, r, min_grad, print_level)
  Find solution to (1).

  This is done by defining a univariate function ϕ (See (3)) in δ. First we construct an interval [δ, δ_prime]
	  such that ϕ(δ) * ϕ(δ_prime) <= 0 and then using bisection, we find a root δ_m for the ϕ function. Using δ_m
	  we compute d_k as d_k = (H + δ_m * I) ^ {-1} (-g)

  # Inputs:
    - `problem_name::String`. Name of the problem being optimized for example a CUTEst benchamrk problem SCURLY10.
	- `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
	- `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, BigInt}, Symmetric{Float64, SparseMatrixCSC{Float64, BigInt}}}`.
	See (1). The Hessian at the current iterate x.
	- `δ::BigFloat`. See (1). A warm start value for solving the above system (2).
	- `γ_1::BigFloat`. See (2). Specify how much the step d_k should be close from the exact solution.
	- `γ_2::BigFloat`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
	- `r::BigFloat`. See (1). The trsut-region radius.
	- `min_grad::BigFloat`. See (2). The minumum gradient over all iterates.
	- `print_level::BigFloat`. The verbosity level of logs.

  # Outputs:
	'success_find_interval::Bool'. See (3). It specifies if we found an interval [δ, δ_prime] such that ϕ(δ) * ϕ(δ_prime) <= 0.
	'success_bisection::Bool'. See (3). It specifies if we found δ_m ∈ [δ, δ_prime] such that ϕ(δ_m) = 0.
	'δ_m::BigFloat'. See (1), (2), and (3). The solution of the above system of equations such that ϕ(δ_m) = 0.
	'δ::BigFloat'. See (3). The lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
	'δ_prime::BigFloat'. See (3). The upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
	'd_k:Vector{Float64}`. See (1). The search direction which is the solution of (1).
	'temp_total_number_factorizations::BigInt'. The total number of choelsky factorization done for H + δ I.
	'hard_case::Bool'. See (2). It specifies if δ_m = -λ_1(H) where λ_1 is the minimum eigenvalue of the matrix H.
	'temp_total_number_factorizations_findinterval::BigInt'. The number of choelsky factorization done for H + δ I when finding the interval [δ, δ_prime].
	'temp_total_number_factorizations_bisection::BigInt'. The number of choelsky factorization done for H + δ I when doing the bisection.
	'total_number_factorizations_compute_search_direction::BigInt'. The number of choelsky factorization done when computing d_k = (H + δ_m * I) ^ {-1} (-g)
	temp_total_number_factorizations = temp_total_number_factorizations_findinterval + temp_total_number_factorizations_bisection + total_number_factorizations_compute_search_direction
"""
function computeSearchDirection(
    problem_name::String,
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
    },
    δ::Float64,
    γ_1::Float64,
    γ_2::Float64,
    r::Float64,
    min_grad::Float64,
    print_level::Int64 = 0,
)
    temp_total_number_factorizations_bisection = 0
    temp_total_number_factorizations_findinterval = 0
    temp_total_number_factorizations_compute_search_direction = 0
    temp_total_number_factorizations_ = 0
    start_time_temp = time()
    if print_level >= 2
        println("Starting Find Interval")
    end

    # Construct an interval [δ, δ_prime] such that ϕ(δ) * ϕ(δ_prime) <= 0
    success, δ, δ_prime, temp_total_number_factorizations_findinterval, temp_d_δ_prime =
        findinterval(g, H, δ, γ_2, r, print_level)
    temp_total_number_factorizations_ += temp_total_number_factorizations_findinterval
    end_time_temp = time()
    total_time_temp = end_time_temp - start_time_temp
    if print_level >= 2
        println(
            "findinterval operation finished with (δ, δ_prime) = ($δ, $δ_prime) and took $total_time_temp.",
        )
    end

    # If we fail to construct the interval [δ, δ_prime], we return failure and the problem will be solved using the
    # hard case logic.
    if !success
        @assert temp_total_number_factorizations_ ==
                temp_total_number_factorizations_findinterval +
                temp_total_number_factorizations_bisection +
                temp_total_number_factorizations_compute_search_direction
        return false,
        false,
        δ,
        δ,
        δ_prime,
        temp_d_δ_prime,
        temp_total_number_factorizations_,
        false,
        temp_total_number_factorizations_findinterval,
        temp_total_number_factorizations_bisection,
        temp_total_number_factorizations_compute_search_direction
    end

    start_time_temp = time()

    # Find a root δ_m ∈ [δ, δ_prime] for the ϕ function using bisection.
    # we compute d_k as d_k = cholesky(H + δ_m * I) \ (-g)
    success, δ_m, δ, δ_prime, temp_d_k, temp_total_number_factorizations_bisection =
        bisection(
            problem_name,
            g,
            H,
            δ,
            γ_1,
            γ_2,
            δ_prime,
            temp_d_δ_prime,
            r,
            min_grad,
            print_level,
        )
    temp_total_number_factorizations_ += temp_total_number_factorizations_bisection
    end_time_temp = time()
    total_time_temp = end_time_temp - start_time_temp
    if print_level >= 2
        println("$success. bisection operation took $total_time_temp.")
    end

    # If we fail to find δ_m using bisection, we return failure and the problem will be solved using the
    # hard case logic.
    if !success
        @assert temp_total_number_factorizations_ ==
                temp_total_number_factorizations_findinterval +
                temp_total_number_factorizations_bisection +
                temp_total_number_factorizations_compute_search_direction
        return true,
        false,
        δ_m,
        δ,
        δ_prime,
        temp_d_k,
        temp_total_number_factorizations_,
        false,
        temp_total_number_factorizations_findinterval,
        temp_total_number_factorizations_bisection,
        temp_total_number_factorizations_compute_search_direction
    end

    @assert δ <= δ_m <= δ_prime

    # Using δ_m, we compute d_k = (H + δ_m * I) \ (-g)
    # @assert (δ <= 1e-6 && norm(d_k) <= r) || γ_2 * r <= norm(d_k) <= r
    @assert temp_total_number_factorizations_ ==
            temp_total_number_factorizations_findinterval +
            temp_total_number_factorizations_bisection +
            temp_total_number_factorizations_compute_search_direction
    return true,
    true,
    δ_m,
    δ,
    δ_prime,
    temp_d_k,
    temp_total_number_factorizations_,
    false,
    temp_total_number_factorizations_findinterval,
    temp_total_number_factorizations_bisection,
    temp_total_number_factorizations_compute_search_direction
end


"""
  validateTrustRegionSubproblemTerminationCriteria(problem_name, d_k, g, H, δ_original, δ, δ_prime, γ_1, γ_2, γ_3, r, min_grad, hard_case, print_level)
  validate the trust-region subproblem termination conditions.

  # Inputs:
    - `problem_name::String`. Name of the problem being optimized for example a CUTEst benchamrk problem SCURLY10.
	- 'd_k:Vector{Float64}`. See (1). The search direction which is the solution of (1).
	- `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
	- `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, BigInt}, Symmetric{Float64, SparseMatrixCSC{Float64, BigInt}}}`.
	See (1). The Hessian at the current iterate x.
	- `δ_original::BigFloat`. See (1). A warm start value for solving the above system (2).
	- 'δ::BigFloat'. See (3). The lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
	- 'δ_prime::BigFloat'. See (3). The upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
	- `γ_1::BigFloat`. See (2). Specify how much the step d_k should be close from the exact solution.
	- `γ_2::BigFloat`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
	- `γ_3::BigFloat`. See (2). Specify upper bound on the Model value.
	- `r::BigFloat`. See (1). The trsut-region radius.
	- `min_grad::BigFloat`. See (2). The minumum gradient over all iterates.
	- 'hard_case::Bool'. See (2). It specifies if δ_m = -λ_1(H) where λ_1 is the minimum eigenvalue of the matrix H.
	- `print_level::BigFloat`. The verbosity level of logs.

  # Outputs:
	If subproblem termination conditions are satisfied or not.
"""
function validateTrustRegionSubproblemTerminationCriteria(
    problem_name::String,
    d_k::Vector{Float64},
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
    },
    original_δ::Float64,
    δ_k::Float64,
    δ_prime_k::Float64,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    r::Float64,
    min_grad::Float64,
    hard_case::Bool,
    print_level::Int64 = 0,
)
    if norm(d_k) == 0
        throw(
            TrustRegionSubproblemError(
                "Trust-region subproblem failure.",
                true,
                true,
                true,
                true,
            ),
        )
    end
    # validate conditions as if δ_k = 0.0
    condition_6a = norm(H * d_k + g) <= γ_1 * min_grad
    condition_6b = true
    condition_6c = norm(d_k) <= r
    model_val = dot(g, d_k) + 0.5 * dot(d_k, H * d_k)
    condition_6d = model_val < 0
    if condition_6a && condition_6b && condition_6c && condition_6d
        if print_level >= 2
            println("==========ACCEPTING STEP==============")
        end
        return true
    end

    γ_3 = 0.5
    # condition (6a)
    error_message = "Trust-region subproblem failure."
    failure = false
    failure_reason_6a = false
    failure_reason_6b = false
    failure_reason_6c = false
    failure_reason_6d = false
    message = "HARD CASE is $hard_case."
    condition = norm(H * d_k + g + δ_k * d_k) <= γ_1 * min_grad
    if !condition
        temp_1 = norm(H * d_k + g + δ_k * d_k)
        temp_2 = γ_1 * min_grad
        message = string(
            message,
            ". Value of norm(H * d + g + δ * d) is $temp_1, value of γ_1 * min_grad is $temp_2, and value of min_grad is $min_grad.",
        )
        failure = true
        failure_reason_6a = true
        error_message = string(error_message, " Reason (6a) failed to be satisfied.")
    end

    # condition (6b)
    if !hard_case && δ_k > 1e-6
        condition = γ_2 * r <= norm(d_k)
        if !condition
            failure = true
            failure_reason_6b = true
            message = string(
                message,
                ". Value of γ_2 * r is $(γ_2 * r), value of δ_k is $δ_k, and value of ||d_k|| = $(norm(d_k))",
            )
            error_message = string(error_message, " Reason (6b) failed to be satisfied.")
        end
    end

    # condition (6c)
    condition = (norm(d_k) - r) <= 1e-6
    if !condition
        failure = true
        failure_reason_6c = true
        message = string(message, ". Value of ||d_k|| = $(norm(d_k)) and value of r = $r")
        error_message = string(error_message, " Reason (6c) failed to be satisfied.")
    end
    # condition (6d)
    model_val = dot(g, d_k) + 0.5 * dot(d_k, H * d_k)
    condition = model_val <= -γ_3 * 0.5 * δ_k * (norm(d_k))^2
    if !condition
        temp_1 = dot(g, d_k) + 0.5 * dot(d_k, H * d_k)
        temp_2 = -γ_3 * 0.5 * δ_k * (norm(d_k))^2
        message = string(
            message,
            ". Value of dot(g, d) + 0.5 * dot(d, H * d) is $temp_1, value of -γ_3 * 0.5 * δ * (norm(d)) ^ 2 is $temp_2, and value of δ_k is $δ_k.",
        )
        failure = true
        failure_reason_6d = true
        error_message = string(error_message, " Reason (6d) failed to be satisfied.")
    end

    # print to file also to add as a unit test case
    if failure
        if print_level >= 2
            println("=============================================")
            println(message)
            println("=============================================")
            println(
                "hard_case, original_δ, δ_k, δ_prime_k, γ_1, γ_2, γ_3, r, min_grad are $hard_case, $original_δ, $δ_k, $δ_prime_k, $γ_1, $γ_2, $γ_3, $r, $min_grad.",
            )
            println("=============================================")
        end
        if print_level >= 2 && problem_name != "problem_name"
            writedlm("./$(problem_name)_gradient.txt", g, ",")
            writedlm("./$(problem_name)_hessian.txt", Matrix(H))
            writedlm(
                "./$(problem_name)_params.txt",
                (hard_case, original_δ, δ_k, δ_prime_k, γ_1, γ_2, γ_3, r, min_grad),
            )
        end
        throw(
            TrustRegionSubproblemError(
                error_message,
                failure_reason_6a,
                failure_reason_6b,
                failure_reason_6c,
                failure_reason_6d,
            ),
        )
    end
    if print_level >= 2
        println("##########ACCEPTING STEP##########")
    end
end

"""
optimizeSecondOrderModel(problem_name, g, H, δ, γ_1, γ_2, r, min_grad, print_level)
This function finds a solution to the quadratic problem

	argmin_d 1/2 d ^ T * H * d + g ^ T  * d
	s.t. || d || <= r                  (1)
	where || . || is the l2 norm of a vector.

This problem (1) is solved approximately by solving the follwoing system of equations:

||H d_k + g + δ d_k|| <= γ_1 * min_grad           (2)
γ_2 * ||d_k||) <=  γ_2 * r
||d_k|| <= r
H + δ I ≥ 0

The logic to find the solution either by defining a univariate function ϕ (See (3)) in δ. Then, we construct an interval [δ, δ_prime]
	such that ϕ(δ) * ϕ(δ_prime) <= 0 and then using bisection, we find a root δ_m for the ϕ function. Using δ_m
	we compute d_k as d_k = (H + δ_m * I) ^ {-1} (-g). If for a reason, we failed to construct the interval or the bisection failed,
	then we mark the problem as a hard case and we use inverse power iteration to find an approximate minimum eigen value
	of the Hessian and then compute the search direction using the minimum eigen value.

# Inputs:
  - `problem_name::String`. Name of the problem being optimized for example a CUTEst benchamrk problem SCURLY10.
  - `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
  - `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, BigInt}, Symmetric{Float64, SparseMatrixCSC{Float64, BigInt}}}`.
  See (1). The Hessian at the current iterate x.
  - `γ_1::BigFloat`. See (2). Specify how much the step d_k should be close from the exact solution.
  - `δ::BigFloat`. See (1). A warm start value for solving the above system (2)
  - `γ_2::BigFloat`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `r::BigFloat`. See (1). The trsut-region radius.
  - `min_grad::BigFloat`. See (2). The minumum gradient over all iterates.
  - `print_level::BigFloat`. The verbosity level of logs.
# Outputs:
  'success_subproblem_solve::Bool'
  'δ_k::BigFloat'.  See (1), (2), and (3). The solution of the above system of equations such that ϕ(δ_k) = 0.
  This is used to compute d_k.
  'd_k::Vector{Float64}`. See (1). The search direction which is the solution of (1).
  'temp_total_number_factorizations::BigInt'. The total number of choelsky factorization done for H + δ I.
  'hard_case::Bool'. See (2). It specifies if δ_k = -λ_1(H) where λ_1 is the minimum eigenvalue of the matrix H.
  'temp_total_number_factorizations_findinterval::BigInt'. The number of choelsky factorization done for H + δ I when finding the interval [δ, δ_prime].
  'temp_total_number_factorizations_bisection::BigInt'. The number of choelsky factorization done for H + δ I when doing the bisection.
  'total_number_factorizations_compute_search_direction::BigInt'. The number of choelsky factorization done when computing d_k = (H + δ_m * I) ^ {-1} (-g)
  'temp_total_number_factorizations_inverse_power_iteration::BigInt'. The number of choelsky factorization done when solving the hard case instance.
  temp_total_number_factorizations = temp_total_number_factorizations_findinterval
  + temp_total_number_factorizations_bisection + total_number_factorizations_compute_search_direction
  + temp_total_number_factorizations_inverse_power_iteration
"""
function optimizeSecondOrderModel(
    problem_name::String,
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
    },
    δ::Float64,
    γ_1::Float64,
    γ_2::Float64,
    r::Float64,
    min_grad::Float64,
    print_level::Int64 = 0,
)
    if print_level >= 2
        println("=========starting optimizeSecondOrderModel========δ is $δ========")
    end
    # When δ is 0 and the Hessian is positive semidefinite, we can directly compute the search direction
    total_number_factorizations = 0
    temp_total_number_factorizations_findinterval = 0
    temp_total_number_factorizations_bisection = 0
    temp_total_number_factorizations_compute_search_direction = 0
    temp_total_number_factorizations_inverse_power_iteration = 0
    temp_total_number_factorizations_ = 0
    try
        temp_total_number_factorizations_compute_search_direction += 1
        temp_total_number_factorizations_ +=
            temp_total_number_factorizations_compute_search_direction
        d_k = cholesky(H) \ (-g)

        if norm(d_k, 2) <= r
            @assert temp_total_number_factorizations_ ==
                    temp_total_number_factorizations_findinterval +
                    temp_total_number_factorizations_bisection +
                    temp_total_number_factorizations_compute_search_direction +
                    temp_total_number_factorizations_inverse_power_iteration
            total_number_factorizations += temp_total_number_factorizations_
            return true,
            0.0,
            0.0,
            d_k,
            total_number_factorizations,
            false,
            temp_total_number_factorizations_findinterval,
            temp_total_number_factorizations_bisection,
            temp_total_number_factorizations_compute_search_direction,
            temp_total_number_factorizations_inverse_power_iteration
        end
    catch e
        #Do nothing
    end
    δ_m = δ
    δ_prime = δ
    d_k = nothing
    try
        # Try to compute the search direction by constructing an interval [δ, δ_prime] such that ϕ(δ) * ϕ(δ_prime) <= 0
        # and then using bisection, we find a root δ_m ∈ [δ, δ_prime] for the ϕ function.
        # Using δ_m we compute d_k = cholesky(H + δ_m * I) / (-g)
        success_find_interval,
        success_bisection,
        δ_m,
        δ,
        δ_prime,
        d_k,
        temp_total_number_factorizations,
        hard_case,
        temp_total_number_factorizations_findinterval,
        temp_total_number_factorizations_bisection,
        total_number_factorizations_compute_search_direction = computeSearchDirection(
            problem_name,
            g,
            H,
            δ,
            γ_1,
            γ_2,
            r,
            min_grad,
            print_level,
        )
        @assert temp_total_number_factorizations ==
                temp_total_number_factorizations_findinterval +
                temp_total_number_factorizations_bisection +
                total_number_factorizations_compute_search_direction
        temp_total_number_factorizations_compute_search_direction +=
            total_number_factorizations_compute_search_direction # TO ACCOUNT FOR THE FIRST ATTEMP WITH d_k = cholesky(H) \ (-g)
        temp_total_number_factorizations_ += temp_total_number_factorizations
        success = success_find_interval && success_bisection
        @assert temp_total_number_factorizations_ ==
                temp_total_number_factorizations_findinterval +
                temp_total_number_factorizations_bisection +
                temp_total_number_factorizations_compute_search_direction +
                temp_total_number_factorizations_inverse_power_iteration
        if success
            try
                validateTrustRegionSubproblemTerminationCriteria(
                    problem_name,
                    d_k,
                    g,
                    H,
                    δ,
                    δ,
                    δ_prime,
                    γ_1,
                    γ_2,
                    0.5,
                    r,
                    min_grad,
                    false,
                    print_level,
                )
                total_number_factorizations += temp_total_number_factorizations_
                return true,
                δ_m,
                δ_prime,
                d_k,
                total_number_factorizations,
                hard_case,
                temp_total_number_factorizations_findinterval,
                temp_total_number_factorizations_bisection,
                temp_total_number_factorizations_compute_search_direction,
                temp_total_number_factorizations_inverse_power_iteration
            catch e
                throw(
                    error(
                        "Bisection logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0.",
                    ),
                )
            end
        end
        # If we failed to construct the interval [δ, δ_prime] or the bisection to find a root δ_m ∈ [δ, δ_prime] for
        # the ϕ function failed, then we solve the trust-region subproblem using the hard-case logic.
        if success_find_interval
            throw(error("Bisection logic failed to find a root for the phi function"))
        else
            throw(
                error(
                    "Bisection logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0.",
                ),
            )
        end
    catch e
        if e ==
           ErrorException("Bisection logic failed to find a root for the phi function") ||
           e == ErrorException(
            "Bisection logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0.",
        )
            δ_initial = δ
            δ_prime_initial = δ_prime
            δ_m_initial = δ_m

            start_time_temp = time()
            # Solve the trust-region subproblem using the hard-case logic. The root for the ϕ function is the minimum
            # eigenvalue of the Hessian matrix and the search direction is on the trust-region boundary.
            if print_level >= 2
                println("Hard case=========")
            end
            success,
            δ,
            d_k,
            temp_total_number_factorizations,
            total_number_factorizations_compute_search_direction,
            temp_total_number_factorizations_inverse_power_iteration = solveHardCaseLogic(
                problem_name,
                g,
                H,
                γ_1,
                γ_2,
                r,
                d_k,
                δ,
                δ_prime,
                min_grad,
                print_level,
            )
            @assert temp_total_number_factorizations ==
                    total_number_factorizations_compute_search_direction +
                    temp_total_number_factorizations_inverse_power_iteration
            temp_total_number_factorizations_compute_search_direction +=
                total_number_factorizations_compute_search_direction
            temp_total_number_factorizations_ += temp_total_number_factorizations
            end_time_temp = time()
            total_time_temp = end_time_temp - start_time_temp
            if print_level >= 2
                @info "$success. solveHardCaseLogic operation took $total_time_temp."
                println("$success. solveHardCaseLogic operation took $total_time_temp.")
            end
            @assert temp_total_number_factorizations_ ==
                    temp_total_number_factorizations_findinterval +
                    temp_total_number_factorizations_bisection +
                    temp_total_number_factorizations_compute_search_direction +
                    temp_total_number_factorizations_inverse_power_iteration
            if success
                total_number_factorizations += temp_total_number_factorizations_
                return success,
                δ,
                δ_prime,
                d_k,
                total_number_factorizations,
                true,
                temp_total_number_factorizations_findinterval,
                temp_total_number_factorizations_bisection,
                temp_total_number_factorizations_compute_search_direction,
                temp_total_number_factorizations_inverse_power_iteration
            end

            δ = δ_initial
            #Try bisection again with perturbed gradient
            # g_k’ = g_k + u_k \gamma_1 \varepsilon_k / 2 where u_k = randn(d), u_k = u_k / || u_k ||
            u_k = rand(Normal(), length(g))
            u_k /= norm(u_k)
            g_ = g + 0.5 * u_k * γ_1 * min_grad
            success_find_interval,
            success_bisection,
            δ_m,
            δ,
            δ_prime,
            d_k,
            temp_total_number_factorizations,
            hard_case,
            temp_total_number_factorizations_findinterval_,
            temp_total_number_factorizations_bisection_,
            temp_total_number_factorizations_compute_search_direction_ =
                computeSearchDirection(
                    problem_name,
                    g_,
                    H,
                    δ,
                    0.5 * γ_1,
                    γ_2,
                    r,
                    min_grad,
                    print_level,
                )
            success = success_find_interval && success_bisection
            if print_level >= 2
                println(
                    "==========ATTEMPT succeded $success where success_find_interval is $success_find_interval and success_bisection is $success_bisection.",
                )
            end
            @assert temp_total_number_factorizations ==
                    temp_total_number_factorizations_findinterval_ +
                    temp_total_number_factorizations_bisection_ +
                    temp_total_number_factorizations_compute_search_direction_
            temp_total_number_factorizations_findinterval +=
                temp_total_number_factorizations_findinterval_
            temp_total_number_factorizations_bisection +=
                temp_total_number_factorizations_bisection_
            temp_total_number_factorizations_compute_search_direction +=
                temp_total_number_factorizations_compute_search_direction_
            temp_total_number_factorizations_ += temp_total_number_factorizations
            @assert temp_total_number_factorizations_ ==
                    temp_total_number_factorizations_findinterval +
                    temp_total_number_factorizations_bisection +
                    temp_total_number_factorizations_compute_search_direction +
                    temp_total_number_factorizations_inverse_power_iteration
            total_number_factorizations += temp_total_number_factorizations_
            return success,
            δ_m,
            δ_prime,
            d_k,
            total_number_factorizations,
            true,
            temp_total_number_factorizations_findinterval,
            temp_total_number_factorizations_bisection,
            temp_total_number_factorizations_compute_search_direction,
            temp_total_number_factorizations_inverse_power_iteration
        else
            throw(e)
        end
    end
end

"""
phi(g, H, δ, γ_2, r, print_level)                     (3)

The function is a decreasing univariate function in δ. It is equal to -1 when H + δ I >= 0 and ||d_k|| < γ_2 r.
	It is equal to 0 when H + δ I >= 0 and γ_2 r <= ||d_k|| <= r. It is equal to 1 when H + δ I < 0 or ||d_k|| > r.
	With d_k = (H + δ * I) ^ {-1} (-g)

# Inputs:
  - `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
  - `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, BigInt}, Symmetric{Float64, SparseMatrixCSC{Float64, BigInt}}}`.
  See (1). The Hessian at the current iterate x.
  - `δ::BigFloat`. See (1). The variable in the function.
  - `γ_2::BigFloat`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `r::BigFloat`. See (1). The trsut-region radius.
  - `print_level::BigFloat`. The verbosity level of logs.

# Outputs:
  An integer values that takes the values -1, 0, or 1.
"""
function phi(
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
    },
    δ::Float64,
    γ_2::Float64,
    r::Float64,
    print_level::Int64 = 0,
)
    sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
    shifted_hessian = H + δ * sparse_identity
    temp_d = zeros(length(g))
    positive_definite = true
    try
        start_time_temp = time()
        shifted_hessian_fact = cholesky(shifted_hessian)
        end_time_temp = time()
        total_time_temp = end_time_temp - start_time_temp
        if print_level >= 2
            println("cholesky inside phi function took $total_time_temp.")
        end

        start_time_temp = time()
        temp_d = shifted_hessian_fact \ (-g)
        computed_norm = norm(temp_d, 2)
        end_time_temp = time()
        total_time_temp = end_time_temp - start_time_temp
        if print_level >= 2
            println("computed_norm opertion took $total_time_temp.")
        end
        if computed_norm > r
            return 1, temp_d, positive_definite
        elseif (δ > 1e-6 && computed_norm < γ_2 * r)
            return -1, temp_d, positive_definite
        else
            return 0, temp_d, positive_definite
        end
    catch e
        # @info e
        positive_definite = false
        return 1, temp_d, positive_definite
    end
end

"""
findinterval(g, H, δ, γ_2, r, print_level)

Constructs an interval [δ, δ_prime] based on the univariate function ϕ (See (3)) such that ϕ(δ) >= 0 and ϕ(δ_prime) <=0.
# Inputs:
  - `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
  - `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, BigInt}, Symmetric{Float64, SparseMatrixCSC{Float64, BigInt}}}`.
  See (1). The Hessian at the current iterate x.
  - `δ::BigFloat`. See (1). A warm start value for solving the above system (2).
  - `γ_2::BigFloat`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `r::BigFloat`. See (1). The trsut-region radius.
  - `print_level::BigFloat`. The verbosity level of logs.

# Outputs:
  'success_find_interval::Bool'. See (3). It specifies if we found an interval [δ, δ_prime] such that ϕ(δ) * ϕ(δ_prime) <= 0.
  'δ::BigFloat'. See (3). The lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
  'δ_prime::BigFloat'. See (3). The upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
  'd_δ_prime:Vector{Float64}`. See (1). The search direction which is computed using δ_prime.
  'temp_total_number_factorizations_findinterval::BigInt'. The number of choelsky factorization done for H + δ I when finding the interval [δ, δ_prime].
"""
function findinterval(
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
    },
    δ::Float64,
    γ_2::Float64,
    r::Float64,
    print_level::Int64 = 0,
)
    δ_original = δ
    if δ == 0.0
        δ_original = 1.0
    end

    Φ_δ_original, temp_d_original, positive_definite = phi(g, H, δ_original, γ_2, r)
    Φ_x, temp_d_x, positive_definite_x = Φ_δ_original, temp_d_original, positive_definite
    Φ_δ_original *= 1.0
    if Φ_δ_original == 0
        δ_prime = δ_original
        return true, δ_original, δ_prime, 1, temp_d_x
    end

    max_iterations = 50
    k = 1
    y = δ_original
    Φ_y, temp_d_y, positive_definite_y = Φ_x, temp_d_x, positive_definite_x

    while k <= max_iterations
        x = y

        Φ_x, temp_d_x, positive_definite_x = Φ_y, temp_d_y, positive_definite_y
        if Φ_x == 0
            δ, δ_prime = x, x
            return true, δ, δ_prime, k, temp_d_x
        end

        y = δ_original * (((2^(k^2)))^Φ_δ_original)

        #safety check
        if isnan(y) || y == Inf
            return false, x, x, k, temp_d_x
        end

        Φ_y, temp_d_y, positive_definite_y = phi(g, H, y, γ_2, r)

        if Φ_y == 0
            δ, δ_prime = y, y
            return true, δ, δ_prime, k + 1, temp_d_y
        end

        if Φ_x * Φ_y < 0
            δ, δ_prime = min(x, y), max(x, y)
            @assert δ_prime > δ
            temd_d = temp_d_y
            @assert (Φ_x == -1 && Φ_y == 1 && x == max(x, y) && y == min(x, y)) ||
                    (Φ_y == -1 && Φ_x == 1 && y == max(x, y) && x == min(x, y))
            if Φ_x == -1
                temd_d = temp_d_x
            end
            return true, δ, δ_prime, k + 1, temd_d
        end
        k = k + 1
    end
    return false, δ_original, δ_original, max_iterations + 1, temp_d_x
end

"""
bisection(problem_name, g, H, δ, γ_1, γ_2, δ_prime, d_δ_prime, r, min_grad, print_level)

Constructs an interval [δ, δ_prime] based on the univariate function ϕ (See (3)) such that ϕ(δ) >= 0 and ϕ(δ_prime) <=0.
# Inputs:
  - `problem_name::String`. Name of the problem being optimized for example a CUTEst benchamrk problem SCURLY10.
  - `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
  - `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, BigInt}, Symmetric{Float64, SparseMatrixCSC{Float64, BigInt}}}`.
  See (1). The Hessian at the current iterate x.
  - 'δ::BigFloat'. See (3). The lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
  - `γ_1::BigFloat`. See (2). Specify how much the step d_k should be close from the exact solution.
  - `γ_2::BigFloat`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - 'δ_prime::BigFloat'. See (3). The upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
  - 'd_δ_prime:Vector{Float64}`. See (1). The search direction which is computed using δ_prime.
  - `r::BigFloat`. See (1). The trsut-region radius.
  - `min_grad::BigFloat`. See (2). The minumum gradient over all iterates.
  - `print_level::BigFloat`. The verbosity level of logs.

# Outputs:
  'success_bisectionl::Bool'. See (3). It specifies if we found δ_m ∈ the interval [δ, δ_prime] such that ϕ(δ_m) = 0.
  'δ_m::BigFloat'. See (1), (2), and (3). The solution of the above system of equations such that ϕ(δ_m) = 0.
  'δ::BigFloat'. See (3). The new lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
  'δ_prime::BigFloat'. See (3). The new upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
  'd_k:Vector{Float64}`. See (1). The search direction which is the solution of (1).
  'temp_total_number_factorizations_bisection::BigInt'. The number of choelsky factorization done for H + δ I when doing the bisection.
"""
function bisection(
    problem_name::String,
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
    },
    δ::Float64,
    γ_1::Float64,
    γ_2::Float64,
    δ_prime::Float64,
    d_δ_prime::Vector{Float64},
    r::Float64,
    min_grad::Float64,
    print_level::Int64 = 0,
)
    # the input of the function is the two end of the interval (δ,δ_prime)
    # our goal here is to find the approximate δ using classic bisection method
    if print_level >= 1
        println(
            "****************************STARTING BISECTION with (δ, δ_prime) = ($δ, $δ_prime)**************",
        )
    end
    #Bisection logic
    k = 1
    original_δ = δ
    δ_m = (δ + δ_prime) / 2
    Φ_δ_m, temp_d, positive_definite = phi(g, H, δ_m, γ_2, r)
    ϕ_δ_prime, d_temp_δ_prime, positive_definite_δ_prime = -1, d_δ_prime, true
    max_iterations = 100
    while (Φ_δ_m != 0) && k <= max_iterations
        start_time_str = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
        if print_level >= 2
            println("$start_time_str. Bisection iteration $k.")
        end
        if Φ_δ_m > 0
            δ = δ_m
        else
            δ_prime = δ_m
            ϕ_δ_prime, d_temp_δ_prime, positive_definite_δ_prime =
                Φ_δ_m, temp_d, positive_definite
            try
                validateTrustRegionSubproblemTerminationCriteria(
                    problem_name,
                    d_temp_δ_prime,
                    g,
                    H,
                    original_δ,
                    δ_prime,
                    δ_prime,
                    γ_1,
                    γ_2,
                    0.5,
                    r,
                    min_grad,
                    false,
                    print_level,
                )
                if print_level >= 2
                    println(
                        "============SUCCESS BISECTION with δ = $δ, δ_m = $δ_m, and δ_prime = $δ_prime.",
                    )
                end
                return true, δ_m, δ, δ_prime, d_temp_δ_prime, k + 1
            catch e
                # nothing to do
            end
        end
        δ_m = (δ + δ_prime) / 2
        Φ_δ_m, temp_d, positive_definite = phi(g, H, δ_m, γ_2, r)
        k = k + 1
        if Φ_δ_m != 0
            q_1 = norm(H * d_temp_δ_prime + g + δ_prime * d_temp_δ_prime)
            q_2 = (γ_1 * min_grad) / 3
            if print_level >= 2
                println("$k===============Bisection entered here=================")
            end
            # if (δ_prime - δ <= ((γ_1 * min_grad) / (3 * r))) &&
            #    q_1 <= q_2
            if (δ_prime - δ <= ((γ_1 * min_grad) / (6 * r))) && q_1 <= q_2
                temp_d = d_temp_δ_prime
                if print_level >= 2
                    println("r is $r and min_grad is $min_grad.")
                    println("δ is $δ and δ_prime is $δ_prime.")
                    println(
                        "$k===================norm(H * d_temp_δ_prime + g + δ_prime * d_temp_δ_prime) is $q_1.============",
                    )
                    println(
                        "$k===================(γ_1 * min_grad / r) is $q_2.============",
                    )
                    println("$k===================ϕ_δ_prime is $ϕ_δ_prime.============")
                end
                break
            end
        end
    end

    if (Φ_δ_m != 0)
        if print_level >= 1
            println("Φ_δ_m is $Φ_δ_m.")
            println("δ, δ_prime, and δ_m are $δ, $δ_prime, and $δ_m. γ_2 is $γ_2.")
        end
        return false, δ_m, δ, δ_prime, temp_d, min(k, max_iterations) + 1
    end
    if print_level >= 1
        println(
            "****************************ENDING BISECTION with δ_m = $δ_m**************",
        )
    end
    return true, δ_m, δ, δ_prime, temp_d, min(k, max_iterations) + 1
end

"""
solveHardCaseLogic(problem_name, g, H, γ_1, γ_2, r, d_k, δ, δ_prime, min_grad, print_level)

Find a solution to (2) if for a reason, we failed to construct the interval or the bisection failed. In this case,
 we mark the problem as a hard case and we use inverse power iteration to find an approximate minimum eigen value
 of the Hessian and then compute the search direction using the minimum eigen value.

# Inputs:
  - `problem_name::String`. Name of the problem being optimized for example a CUTEst benchamrk problem SCURLY10.
  - `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
  - `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, BigInt}, Symmetric{Float64, SparseMatrixCSC{Float64, BigInt}}}`.
  See (1). The Hessian at the current iterate x.
  - `γ_1::BigFloat`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `γ_2::BigFloat`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `r::BigFloat`. See (1). The trsut-region radius.
  - 'δ::BigFloat'. See (3). The lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
  - 'δ_prime::BigFloat'. See (3). The upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
  - `min_grad::BigFloat`. See (2). The minumum gradient over all iterates.
  - `print_level::BigFloat`. The verbosity level of logs.

# Outputs:
  'success::Bool'. See (3). It specifies if we found the solution of (1).
  'δ::BigFloat'. See (1), (2), and (3). The solution of the above system of equations (2) such that ϕ(δ) = 0.
   It has the minimum eigenvalue of H.
  'd_k::Vector{Float64}'. See (1). The search direction which is the solution of (1).
   d_k = cholesky(H + δ' * I) ^ {-1} (-g) + α y where y is the eigenvector associated with the minimum eigenvalue and α computed such that ||d_k|| = r
  'temp_total_number_factorizations::BigInt'. The total number of choelsky factorization done for H + δ I.
  'total_number_factorizations_compute_search_direction::BigInt'. The number of choelsky factorization done when computing d_k
  'temp_total_number_factorizations_inverse_power_iteration::BigInt'. The number of choelsky factorization done when solving the hard case instance.
"""
function solveHardCaseLogic(
    problem_name::String,
    g::Vector{Float64},
    H::Union{SparseMatrixCSC{Float64,Int},Symmetric{Float64,SparseMatrixCSC{Float64,Int}}},
    γ_1::Float64,
    γ_2::Float64,
    r::Float64,
    d_k_δ_prime::Vector{Float64},
    δ::Float64,
    δ_prime::Float64,
    min_grad::Float64,
    print_level::Int64 = 0,
)
    if print_level >= 2
        println(
            "++++++++STARTING HARD CASE LOGIC WITH δ = $δ and δ_prime = $δ_prime++++++++",
        )
    end

    if print_level >= 2
        ϕ_δ, temp_d_δ, positive_definite = phi(g, H, δ, γ_2, r, print_level)
        norm_temp_d_δ = norm(temp_d_δ)
        ϕ_δ_prime, temp_d_δ_prime, positive_definite =
            phi(g, H, δ_prime, γ_2, r, print_level)
        norm_temp_d_δ_prime = norm(temp_d_δ_prime)
        println(
            "++++++++STARTING HARD CASE LOGIC WITH ||d_δ|| = $norm_temp_d_δ and ||d_δ_prime|| = $norm_temp_d_δ_prime++++++++",
        )
    end
    sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
    total_number_factorizations = 0
    temp_total_number_factorizations_compute_search_direction = 0
    temp_total_number_factorizations_inverse_power_iteration = 0
    temp_total_number_factorizations_ = 0
    temp_eigenvalue = 0
    try
        start_time_temp = time()
        # Compute the minimum eigenvalue of the Hessian matrix using inverse power iteration
        success,
        eigenvalue,
        eigenvector,
        temp_total_number_factorizations_inverse_power_iteration,
        temp_d_k = inverse_power_iteration(
            problem_name,
            g,
            H,
            min_grad,
            d_k_δ_prime,
            δ,
            δ_prime,
            r,
            γ_1,
            γ_2,
        )
        temp_eigenvalue = eigenvalue
        end_time_temp = time()
        total_time_temp = end_time_temp - start_time_temp
        if print_level >= 2
            @info "inverse_power_iteration operation took $total_time_temp."
        end
        eigenvalue = abs(eigenvalue)
        temp_total_number_factorizations_ +=
            temp_total_number_factorizations_inverse_power_iteration
        norm_temp_d_k = norm(temp_d_k)

        # We failed to find the minimum eigenvalue of the Hessian matrix using inverse power iteration
        if norm_temp_d_k == 0
            @assert temp_total_number_factorizations_ ==
                    temp_total_number_factorizations_compute_search_direction +
                    temp_total_number_factorizations_inverse_power_iteration
            total_number_factorizations += temp_total_number_factorizations_
            return false,
            eigenvalue,
            temp_d_k,
            total_number_factorizations,
            temp_total_number_factorizations_compute_search_direction,
            temp_total_number_factorizations_inverse_power_iteration
        end

        if print_level >= 2
            @info "candidate search direction norm is $norm_temp_d_k. r is $r."
        end
        # Validate that the search direction satisfies the trust-region subproblem termination critera
        # The search direction in the hard case should be approximately on the trust-region boundary
        if abs(norm(temp_d_k) - r) <= 1e-3
            @assert temp_total_number_factorizations_ ==
                    temp_total_number_factorizations_compute_search_direction +
                    temp_total_number_factorizations_inverse_power_iteration
            total_number_factorizations += temp_total_number_factorizations_
            return true,
            eigenvalue,
            temp_d_k,
            total_number_factorizations,
            temp_total_number_factorizations_compute_search_direction,
            temp_total_number_factorizations_inverse_power_iteration
        end
        # The computed search direction is outside the trust-region boundary.
        if abs(norm(temp_d_k) - r) > 1e-3
            if print_level >= 1
                println(
                    "This is noit a hard case. FAILURE======candidate search direction norm is $norm_temp_d_k. r is $r. γ_2 is $γ_2",
                )
                @warn "This is noit a hard case. candidate search direction norm is $norm_temp_d_k. r is $r. γ_2 is $γ_2"
            end
        end

        @assert temp_total_number_factorizations_ ==
                temp_total_number_factorizations_compute_search_direction +
                temp_total_number_factorizations_inverse_power_iteration
        total_number_factorizations += temp_total_number_factorizations_
        return false,
        eigenvalue,
        temp_d_k,
        total_number_factorizations,
        temp_total_number_factorizations_compute_search_direction,
        temp_total_number_factorizations_inverse_power_iteration
    catch e
        @info e
        # We failed to find the minimum eigenvalue of the Hessian matrix using inverse power iteration
        if print_level >= 2
            matrix_H = Matrix(H)
            mimimum_eigenvalue = eigmin(Matrix(H))
            println(
                "FAILURE+++++++inverse_power_iteration operation returned non positive matrix. retunred_eigen_value is $temp_eigenvalue and mimimum_eigenvalue is $mimimum_eigenvalue.",
            )
        end
        @assert temp_total_number_factorizations_ ==
                temp_total_number_factorizations_compute_search_direction +
                temp_total_number_factorizations_inverse_power_iteration
        total_number_factorizations += temp_total_number_factorizations_
        return false,
        δ_prime,
        zeros(length(g)),
        total_number_factorizations,
        temp_total_number_factorizations_compute_search_direction,
        temp_total_number_factorizations_inverse_power_iteration
    end
end

"""
inverse_power_iteration(problem_name, g, H, min_grad, d_k_δ_prime, δ, δ_prime, r, γ_1, γ_2, max_iter, ϵ, print_level)

Compute iteratively an approximate value to the minimum eigenvalue of H.

# Inputs:
  - `problem_name::String`. Name of the problem being optimized for example a CUTEst benchamrk problem SCURLY10.
  - `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
  - `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, BigInt}, Symmetric{Float64, SparseMatrixCSC{Float64, BigInt}}}`.
  See (1). The Hessian at the current iterate x.
  - `min_grad::BigFloat`. See (2). The minumum gradient over all iterates.
  - 'δ::BigFloat'. See (3). The lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
  - 'δ_prime::BigFloat'. See (3). The upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
  - `r::BigFloat`. See (1). The trsut-region radius.
  - `γ_1::BigFloat`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `γ_2::BigFloat`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `max_iter::BigInt`. The maximum number of iterations to run.
  - `ϵ::BigFloat`. The tolerance to specify how close the solution should be from the minimum eigenvalue.
  - `print_level::BigFloat`. The verbosity level of logs.

# Outputs:
  'success::Bool'. See (3). It specifies if we found the minimum eigenvalue of H or not.
  'eigenvalue::BigFloat'. The minimum eigenvalue of H.
  'eigenvector::::Vector{Float64}'. The eigenvector for the minimum eigenvalue of H.
  'temp_total_number_factorizations_inverse_power_iteration::BigInt'. The number of choelsky factorization done when solving the hard case instance.
  'temp_d_k::Vector{Float64}'. temp_d_k = cholesky(H + δ_prime * I) / (-g) + α * eigenvector
"""
function inverse_power_iteration(
    problem_name::String,
    g::Vector{Float64},
    H::Union{SparseMatrixCSC{Float64,Int},Symmetric{Float64,SparseMatrixCSC{Float64,Int}}},
    min_grad::Float64,
    d_k_δ_prime::Vector{Float64},
    δ::Float64,
    δ_prime::Float64,
    r::Float64,
    γ_1::Float64,
    γ_2::Float64;
    max_iter::Int64 = 50,
    ϵ::Float64 = 1e-3,
    print_level::Int64 = 0,
)
    γ_3 = 0.5
    sigma = δ_prime
    start_time_temp = time()
    n = size(H, 1)
    y = rand(Normal(), n)
    if print_level >= 2
        println(
            "++++++++STARTING INVERSE POWER ITERATION WITH δ = $δ and δ_prime = $δ_prime++++++++",
        )
    end
    sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
    y_original_fact = cholesky(H + sigma * sparse_identity)
    d_k_δ_prime = y_original_fact \ (-g)
    temp_factorization = 1
    for k = 1:max_iter
        y = (y_original_fact \ y)
        y /= norm(y)
        eigenvalue = dot(y, H * y)
        if print_level >= 2
            println("$k. eigenvalue  is $eigenvalue.")
        end

        α = -dot(d_k_δ_prime, y) + sqrt((dot(d_k_δ_prime, y))^2 + r^2 - norm(d_k_δ_prime)^2)
        τ = -dot(d_k_δ_prime, y) - sqrt((dot(d_k_δ_prime, y))^2 + r^2 - norm(d_k_δ_prime)^2)
        temp_d_k_1 = d_k_δ_prime + α * y
        temp_d_k_2 = d_k_δ_prime + τ * y
        norm_temp_d_k_1 = norm(temp_d_k_1)
        norm_temp_d_k_2 = norm(temp_d_k_2)
        model_temp_d_k_val_1 = dot(g, temp_d_k_1) + 0.5 * dot(temp_d_k_1, H * temp_d_k_1)
        model_temp_d_k_val_2 = dot(g, temp_d_k_2) + 0.5 * dot(temp_d_k_2, H * temp_d_k_2)
        if print_level >= 2
            condition_1 = norm(H * temp_d_k_1 + g + δ_prime * temp_d_k_1) <= γ_1 * min_grad
            condition_2 = norm(H * temp_d_k_2 + g + δ_prime * temp_d_k_2) <= γ_1 * min_grad
            println(
                "============condition_1 is $condition_1============condition_2 is $condition_2.",
            )
        end
        if model_temp_d_k_val_1 < model_temp_d_k_val_2
            try
                validateTrustRegionSubproblemTerminationCriteria(
                    problem_name,
                    temp_d_k_1,
                    g,
                    H,
                    δ,
                    δ_prime,
                    δ_prime,
                    γ_1,
                    γ_2,
                    γ_3,
                    r,
                    min_grad,
                    true,
                    print_level,
                )
                if print_level >= 2
                    println(
                        "===accepting search direction ||temp_d_k_1|| = $norm_temp_d_k_1 with δ, δ, δ_prime = $δ, $δ, $δ_prime. γ_1, γ_2, γ_3, r, min_grad = $γ_1, $γ_2, $γ_3, $r, $min_grad",
                    )
                end
                return true, δ_prime, y, temp_factorization, temp_d_k_1
            catch e
                try
                    validateTrustRegionSubproblemTerminationCriteria(
                        problem_name,
                        temp_d_k_2,
                        g,
                        H,
                        δ,
                        δ_prime,
                        δ_prime,
                        γ_1,
                        γ_2,
                        γ_3,
                        r,
                        min_grad,
                        true,
                        print_level,
                    )
                    if print_level >= 2
                        println(
                            "===accepting search direction ||temp_d_k_2|| = $norm_temp_d_k_2 with δ, δ, δ_prime = $δ, $δ, $δ_prime. γ_1, γ_2, γ_3, r, min_grad = $γ_1, $γ_2, $γ_3, $r, $min_grad",
                        )
                    end
                    return true, δ_prime, y, temp_factorization, temp_d_k_2
                catch e
                    # println("=======trying another eigenvalue =======")
                end
            end
        else
            try
                validateTrustRegionSubproblemTerminationCriteria(
                    problem_name,
                    temp_d_k_2,
                    g,
                    H,
                    δ,
                    δ_prime,
                    δ_prime,
                    γ_1,
                    γ_2,
                    γ_3,
                    r,
                    min_grad,
                    true,
                    print_level,
                )
                println(
                    "===accepting search direction ||temp_d_k_2|| = $norm_temp_d_k_2 with δ, δ, δ_prime = $δ, $δ, $δ_prime. γ_1, γ_2, γ_3, r, min_grad = $γ_1, $γ_2, $γ_3, $r, $min_grad",
                )
                return true, δ_prime, y, temp_factorization, temp_d_k_2
            catch e
                try
                    validateTrustRegionSubproblemTerminationCriteria(
                        problem_name,
                        temp_d_k_1,
                        g,
                        H,
                        δ,
                        δ_prime,
                        δ_prime,
                        γ_1,
                        γ_2,
                        γ_3,
                        r,
                        min_grad,
                        true,
                        print_level,
                    )
                    println(
                        "===accepting search direction ||temp_d_k_1|| = $norm_temp_d_k_1 with δ, δ, δ_prime = $δ, $δ, $δ_prime. γ_1, γ_2, γ_3, r, min_grad = $γ_1, $γ_2, $γ_3, $r, $min_grad",
                    )
                    return true, δ_prime, y, temp_factorization, temp_d_k_1
                catch e
                    # println("=======trying another eigenvalue =======")
                end
            end
        end
    end
    temp_ = dot(y, H * y)

    if print_level >= 2
        end_time_temp = time()
        total_time_temp = end_time_temp - start_time_temp
        @info "inverse_power_iteration operation took $total_time_temp."
        println("inverse_power_iteration operation took $total_time_temp.")
    end

    temp_d_k = zeros(length(g))
    return false, δ_prime, y, temp_factorization, temp_d_k
end
