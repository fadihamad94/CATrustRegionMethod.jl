export phi, findinterval, bisection
using LinearAlgebra
using Dates

"""
solveTrustRegionSubproblem(f, g, H, x, δ, γ_1, γ_2, r, min_grad, print_level)
See optimizeSecondOrderModel
"""
function solveTrustRegionSubproblem(
    f::Float64,
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
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
    success_subproblem_solve,
    δ_k,
    d_k,
    temp_total_number_factorizations,
    hard_case,
    temp_total_number_factorizations_findinterval,
    temp_total_number_factorizations_bisection,
    temp_total_number_factorizations_compute_search_direction,
    temp_total_number_factorizations_inverse_power_iteration =
        optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, min_grad, print_level)

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

    return success_subproblem_solve, δ_k, d_k, hard_case
end

"""
  computeSearchDirection(g, H, δ, γ_1, γ_2, r, min_grad, print_level)
  Find solution to (1).

  This is done by defining a univariate function ϕ (See (3)) in δ. First we construct an interval [δ, δ_prime]
	  such that ϕ(δ) * ϕ(δ_prime) <= 0 and then using bisection, we find a root δ_m for the ϕ function. Using δ_m
	  we compute d_k as d_k = (H + δ_m * I) ^ {-1} (-g)

  # Inputs:
	- `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
	- `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
	See (1). The Hessian at the current iterate x.
	- `δ::Float64`. See (1). A warm start value for solving the above system (2).
	- `γ_1::Float64`. See (2). Specify how much the step d_k should be close from the exact solution.
	- `γ_2::Float64`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
	- `r::Float64`. See (1). The trsut-region radius.
	- `min_grad::Float64`. See (2). The minumum gradient over all iterates.
	- `print_level::Float64`. The verbosity level of logs.

  # Outputs:
	'success_find_interval::Bool'. See (3). It specifies if we found an interval [δ, δ_prime] such that ϕ(δ) * ϕ(δ_prime) <= 0.
	'success_bisection::Bool'. See (3). It specifies if we found δ_m ∈ [δ, δ_prime] such that ϕ(δ_m) = 0.
	'δ_m::Float64'. See (1), (2), and (3). The solution of the above system of equations such that ϕ(δ_m) = 0.
	'δ::Float64'. See (3). The lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
	'δ_prime::Float64'. See (3). The upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
	'd_k:Vector{Float64}`. See (1). The search direction which is the solution of (1).
	'temp_total_number_factorizations::Int64'. The total number of choelsky factorization done for H + δ I.
	'hard_case::Bool'. See (2). It specifies if δ_m = -λ_1(H) where λ_1 is the minimum eigenvalue of the matrix H.
	'temp_total_number_factorizations_findinterval::Int64'. The number of choelsky factorization done for H + δ I when finding the interval [δ, δ_prime].
	'temp_total_number_factorizations_bisection::Int64'. The number of choelsky factorization done for H + δ I when doing the bisection.
	'total_number_factorizations_compute_search_direction::Int64'. The number of choelsky factorization done when computing d_k = (H + δ_m * I) ^ {-1} (-g)
	temp_total_number_factorizations = temp_total_number_factorizations_findinterval + temp_total_number_factorizations_bisection + total_number_factorizations_compute_search_direction
"""
function computeSearchDirection(
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
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
    success, δ, δ_prime, temp_total_number_factorizations_findinterval =
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
        zeros(length(g)),
        temp_total_number_factorizations_,
        false,
        temp_total_number_factorizations_findinterval,
        temp_total_number_factorizations_bisection,
        temp_total_number_factorizations_compute_search_direction
    end

    start_time_temp = time()

    # Find a root δ_m ∈ [δ, δ_prime] for the ϕ function using bisection.
    # we compute d_k as d_k = (H + δ_m * I) \ (-g)
    success, δ_m, δ, δ_prime, temp_d_k, temp_total_number_factorizations_bisection =
        bisection(g, H, δ, γ_1, γ_2, δ_prime, r, min_grad, print_level)
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
        zeros(length(g)),
        temp_total_number_factorizations_,
        false,
        temp_total_number_factorizations_findinterval,
        temp_total_number_factorizations_bisection,
        temp_total_number_factorizations_compute_search_direction
    end

    @assert δ <= δ_m <= δ_prime

    sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
    temp_total_number_factorizations_compute_search_direction += 1
    temp_total_number_factorizations_ +=
        temp_total_number_factorizations_compute_search_direction

    # Using δ_m, we compute d_k = (H + δ_m * I) \ (-g)
    # Cholesky factorization is done as an extra validation that H + δ_m * I is positive definite.
    # start_time_temp = time()
    # d_k = cholesky(H + δ_m * sparse_identity) \ (-g)
    # end_time_temp = time()
    # total_time_temp = end_time_temp - start_time_temp
    # if print_level >= 2
    #     println("d_k operation took $total_time_temp.")
    # end
    d_k = temp_d_k
    @assert γ_2 * r <= norm(d_k) <= r
    @assert temp_total_number_factorizations_ ==
            temp_total_number_factorizations_findinterval +
            temp_total_number_factorizations_bisection +
            temp_total_number_factorizations_compute_search_direction
    return true,
    true,
    δ_m,
    δ,
    δ_prime,
    d_k,
    temp_total_number_factorizations_,
    false,
    temp_total_number_factorizations_findinterval,
    temp_total_number_factorizations_bisection,
    temp_total_number_factorizations_compute_search_direction
end

"""
optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, min_grad, print_level)
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
  - `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
  - `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
  See (1). The Hessian at the current iterate x.
  - `γ_1::Float64`. See (2). Specify how much the step d_k should be close from the exact solution.
  - `δ::Float64`. See (1). A warm start value for solving the above system (2)
  - `γ_2::Float64`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `r::Float64`. See (1). The trsut-region radius.
  - `min_grad::Float64`. See (2). The minumum gradient over all iterates.
  - `print_level::Float64`. The verbosity level of logs.
# Outputs:
  'success_subproblem_solve::Bool'
  'δ_k::Float64'.  See (1), (2), and (3). The solution of the above system of equations such that ϕ(δ_k) = 0.
  This is used to compute d_k.
  'd_k::Vector{Float64}`. See (1). The search direction which is the solution of (1).
  'temp_total_number_factorizations::Int64'. The total number of choelsky factorization done for H + δ I.
  'hard_case::Bool'. See (2). It specifies if δ_k = -λ_1(H) where λ_1 is the minimum eigenvalue of the matrix H.
  'temp_total_number_factorizations_findinterval::Int64'. The number of choelsky factorization done for H + δ I when finding the interval [δ, δ_prime].
  'temp_total_number_factorizations_bisection::Int64'. The number of choelsky factorization done for H + δ I when doing the bisection.
  'total_number_factorizations_compute_search_direction::Int64'. The number of choelsky factorization done when computing d_k = (H + δ_m * I) ^ {-1} (-g)
  'temp_total_number_factorizations_inverse_power_iteration::Int64'. The number of choelsky factorization done when solving the hard case instance.
  temp_total_number_factorizations = temp_total_number_factorizations_findinterval
  + temp_total_number_factorizations_bisection + total_number_factorizations_compute_search_direction
  + temp_total_number_factorizations_inverse_power_iteration
"""
function optimizeSecondOrderModel(
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    },
    δ::Float64,
    γ_1::Float64,
    γ_2::Float64,
    r::Float64,
    min_grad::Float64,
    print_level::Int64 = 0,
)
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
    try
        # Try to compute the search direction by constructing an interval [δ, δ_prime] such that ϕ(δ) * ϕ(δ_prime) <= 0
        # and then using bisection, we find a root δ_m ∈ [δ, δ_prime] for the ϕ function.
        # Using δ_m we compute d_k = (H + δ_m * I) ^ {-1} (-g)
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
        total_number_factorizations_compute_search_direction =
            computeSearchDirection(g, H, δ, γ_1, γ_2, r, min_grad, print_level)
        @assert temp_total_number_factorizations ==
                temp_total_number_factorizations_findinterval +
                temp_total_number_factorizations_bisection +
                total_number_factorizations_compute_search_direction
        temp_total_number_factorizations_compute_search_direction +=
            total_number_factorizations_compute_search_direction # TO ACCOUNT FOR THE FIRST ATTEMP WITH d_k = cholesky(H) \ (-g)
        temp_total_number_factorizations_ += temp_total_number_factorizations
        success = success_find_interval && success_bisection
        if success
            @assert temp_total_number_factorizations_ ==
                    temp_total_number_factorizations_findinterval +
                    temp_total_number_factorizations_bisection +
                    temp_total_number_factorizations_compute_search_direction +
                    temp_total_number_factorizations_inverse_power_iteration
            total_number_factorizations += temp_total_number_factorizations_
            return true,
            δ_m,
            d_k,
            total_number_factorizations,
            hard_case,
            temp_total_number_factorizations_findinterval,
            temp_total_number_factorizations_bisection,
            temp_total_number_factorizations_compute_search_direction,
            temp_total_number_factorizations_inverse_power_iteration
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
        if e == ErrorException("Bisection logic failed to find a root for the phi function")
            start_time_temp = time()
            # Solve the trust-region subproblem using the hard-case logic. The root for the ϕ function is the minimum
            # eigenvalue of the Hessian matrix and the search direction is on the trust-region boundary.
            success,
            δ,
            d_k,
            temp_total_number_factorizations,
            total_number_factorizations_compute_search_direction,
            temp_total_number_factorizations_inverse_power_iteration =
                solveHardCaseLogic(g, H, γ_1, γ_2, r, δ, δ_prime, min_grad, print_level)
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
            total_number_factorizations += temp_total_number_factorizations_
            return success,
            δ,
            d_k,
            total_number_factorizations,
            true,
            temp_total_number_factorizations_findinterval,
            temp_total_number_factorizations_bisection,
            temp_total_number_factorizations_compute_search_direction,
            temp_total_number_factorizations_inverse_power_iteration
        elseif e == ErrorException(
            "Bisection logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0.",
        )
            start_time_temp = time()
            # Solve the trust-region subproblem using the hard-case logic. The root for the ϕ function is the minimum
            # eigenvalue of the Hessian matrix and the search direction is on the trust-region boundary.
            success,
            δ,
            d_k,
            temp_total_number_factorizations,
            total_number_factorizations_compute_search_direction,
            temp_total_number_factorizations_inverse_power_iteration =
                solveHardCaseLogic(g, H, γ_1, γ_2, r, δ, δ_prime, min_grad, print_level)
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
            total_number_factorizations += temp_total_number_factorizations_
            return success,
            δ,
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
  - `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
  See (1). The Hessian at the current iterate x.
  - `δ::Float64`. See (1). The variable in the function.
  - `γ_2::Float64`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `r::Float64`. See (1). The trsut-region radius.
  - `print_level::Float64`. The verbosity level of logs.

# Outputs:
  An integer values that takes the values -1, 0, or 1.
"""
function phi(
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
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

        if (δ <= 1e-6 && computed_norm <= r)
            return 0, temp_d, positive_definite
        elseif computed_norm < γ_2 * r
            return -1, temp_d, positive_definite
        elseif computed_norm <= r
            return 0, temp_d, positive_definite
        else
            return 1, temp_d, positive_definite
        end
    catch e
        positive_definite = false
        return 1, temp_d, positive_definite
    end
end

"""
findinterval(g, H, δ, γ_2, r, print_level)

Constructs an interval [δ, δ_prime] based on the univariate function ϕ (See (3)) such that ϕ(δ) >= 0 and ϕ(δ_prime) <=0.
# Inputs:
  - `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
  - `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
  See (1). The Hessian at the current iterate x.
  - `δ::Float64`. See (1). A warm start value for solving the above system (2).
  - `γ_2::Float64`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `r::Float64`. See (1). The trsut-region radius.
  - `print_level::Float64`. The verbosity level of logs.

# Outputs:
  'success_find_interval::Bool'. See (3). It specifies if we found an interval [δ, δ_prime] such that ϕ(δ) * ϕ(δ_prime) <= 0.
  'δ::Float64'. See (3). The lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
  'δ_prime::Float64'. See (3). The upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
  'temp_total_number_factorizations_findinterval::Int64'. The number of choelsky factorization done for H + δ I when finding the interval [δ, δ_prime].
"""
function findinterval(
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    },
    δ::Float64,
    γ_2::Float64,
    r::Float64,
    print_level::Int64 = 0,
)
    @assert δ >= 0
    if print_level >= 1
        println("STARTING WITH δ = $δ.")
    end
    Φ_δ, temp_d, positive_definite = phi(g, H, 0.0, γ_2, r)

    if Φ_δ == 0
        δ = 0.0
        δ_prime = 0.0
        return true, δ, δ_prime, 1
    end

    δ_original = δ

    Φ_δ, temp_d, positive_definite = phi(g, H, δ, γ_2, r)

    if Φ_δ == 0
        δ_prime = δ
        return true, δ, δ_prime, 2
    end

    δ_prime = δ
    Φ_δ_prime = Φ_δ
    search_δ_prime = true

    if Φ_δ > 0
        δ_prime = δ == 0.0 ? 1.0 : δ * 2
        search_δ_prime = true
    else
        # Here ϕ(δ) < 0 and we need to find new δ' >= 0 such that ϕ(δ') >= 0 and δ' < δ which is not possible
        # in case δ == 0
        @assert δ > 0
        search_δ_prime = false
        # The aim is to find [δ, δ'] such that ϕ(δ) ∈ {0, 1}, ϕ(δ') ∈ {0, -1}, and  ϕ(δ) * ϕ(δ') <= ∈ {0, -1}
        # since here ϕ(δ) < 0, we set δ' = δ and we search for δ < δ'such that ϕ(δ) ∈ {0, 1}
        δ_prime = δ
        Φ_δ_prime = -1
        δ = δ / 2
    end

    max_iterations = 50
    k = 1
    while k < max_iterations
        if search_δ_prime
            Φ_δ_prime, temp_d, positive_definite = phi(g, H, δ_prime, γ_2, r)
            if Φ_δ_prime == 0
                δ = δ_prime
                return true, δ, δ_prime, k + 2
            end
        else
            Φ_δ, temp_d, positive_definite = phi(g, H, δ, γ_2, r)
            if Φ_δ == 0
                δ_prime = δ
                return true, δ, δ_prime, k + 2
            end
        end

        if ((Φ_δ * Φ_δ_prime) < 0)
            if print_level >= 1
                println("ENDING WITH ϕ(δ) = $Φ_δ and Φ_δ_prime = $Φ_δ_prime.")
                println("ENDING WITH δ = $δ and δ_prime = $δ_prime.")
            end
            @assert δ_prime > δ
            @assert ((δ == 0.0) & (δ_prime == 1.0)) ||
                    ((δ_prime / δ) == 2^(2^(k - 1))) ||
                    ((δ_prime / δ) - 2^(2^(k - 1)) <= 1e-3)
            factor = δ_prime / δ
            return true, δ, δ_prime, k + 2
        end
        if search_δ_prime
            # Here Φ_δ_prime is still 1 and we are continue searching for δ',
            # but we can update δ to give it larger values which is the current value of δ'
            @assert Φ_δ_prime > 0
            δ = δ_prime
            δ_prime = δ_prime * (2^(2^k))
        else
            # Here Φ_δ is still -1 and we are continue searching for δ,
            # but we can update δ' to give it smaller value which is the current value of δ
            @assert Φ_δ < 0
            δ_prime = δ
            δ = δ / (2^(2^k))
        end

        k = k + 1
    end

    if (Φ_δ * Φ_δ_prime > 0)
        if print_level >= 1
            println(
                "Φ_δ is $Φ_δ and Φ_δ_prime is $Φ_δ_prime. δ is $δ and δ_prime is $δ_prime.",
            )
        end
        return false, δ, δ_prime, max_iterations + 2
    end
    factor = δ_prime / δ
    return true, δ, δ_prime, max_iterations + 2
end


"""
bisection(g, H, δ, γ_1, γ_2, δ_prime, r, min_grad, print_level)

Constructs an interval [δ, δ_prime] based on the univariate function ϕ (See (3)) such that ϕ(δ) >= 0 and ϕ(δ_prime) <=0.
# Inputs:
  - `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
  - `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
  See (1). The Hessian at the current iterate x.
  - 'δ::Float64'. See (3). The lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
  - `γ_1::Float64`. See (2). Specify how much the step d_k should be close from the exact solution.
  - `γ_2::Float64`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - 'δ_prime::Float64'. See (3). The upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
  - `r::Float64`. See (1). The trsut-region radius.
  - `min_grad::Float64`. See (2). The minumum gradient over all iterates.
  - `print_level::Float64`. The verbosity level of logs.

# Outputs:
  'success_bisectionl::Bool'. See (3). It specifies if we found δ_m ∈ the interval [δ, δ_prime] such that ϕ(δ_m) = 0.
  'δ_m::Float64'. See (1), (2), and (3). The solution of the above system of equations such that ϕ(δ_m) = 0.
  'δ::Float64'. See (3). The new lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
  'δ_prime::Float64'. See (3). The new upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
  'temp_total_number_factorizations_bisection::Int64'. The number of choelsky factorization done for H + δ I when doing the bisection.
"""
function bisection(
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    },
    δ::Float64,
    γ_1::Float64,
    γ_2::Float64,
    δ_prime::Float64,
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
    δ_m = (δ + δ_prime) / 2
    Φ_δ_m, temp_d, positive_definite = phi(g, H, δ_m, γ_2, r)
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
        end
        δ_m = (δ + δ_prime) / 2
        Φ_δ_m, temp_d, positive_definite = phi(g, H, δ_m, γ_2, r)
        k = k + 1
        if Φ_δ_m != 0
            ϕ_δ_prime, d_temp_δ_prime, positive_definite_δ_prime =
                phi(g, H, δ_prime, γ_2, r)
            ϕ_δ, d_temp_δ, positive_definite_δ = phi(g, H, δ, γ_2, r)
            q_1 = norm(H * d_temp_δ_prime + g + δ_prime * d_temp_δ_prime)
            q_2 = γ_1 * min_grad
            if print_level >= 2
                println("$k===============Bisection entered here=================")
            end
            if (δ_prime - δ <= ((γ_1 * min_grad) / r)) &&
               q_1 <= q_2 &&
               !positive_definite_δ
                if print_level >= 2
                    println(
                        "$k===================norm(H * d_temp_δ_prime + g + δ_prime * d_temp_δ_prime) is $q_1.============",
                    )
                    println("$k===================(γ_1 * min_grad / r) is $q_2.============")
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
solveHardCaseLogic(g, H, γ_1, γ_2, r, δ, δ_prime, min_grad, print_level)

Find a solution to (2) if for a reason, we failed to construct the interval or the bisection failed. In this case,
 we mark the problem as a hard case and we use inverse power iteration to find an approximate minimum eigen value
 of the Hessian and then compute the search direction using the minimum eigen value.

# Inputs:
  - `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
  - `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
  See (1). The Hessian at the current iterate x.
  - `γ_1::Float64`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `γ_2::Float64`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `r::Float64`. See (1). The trsut-region radius.
  - 'δ::Float64'. See (3). The lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
  - 'δ_prime::Float64'. See (3). The upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
  - `min_grad::Float64`. See (2). The minumum gradient over all iterates.
  - `print_level::Float64`. The verbosity level of logs.

# Outputs:
  'success::Bool'. See (3). It specifies if we found the solution of (1).
  'δ::Float64'. See (1), (2), and (3). The solution of the above system of equations (2) such that ϕ(δ) = 0.
   It has the minimum eigenvalue of H.
  'd_k::Vector{Float64}'. See (1). The search direction which is the solution of (1).
   d_k = cholesky(H + (δ + 1e-1) * I) ^ {-1} (-g)
  'temp_total_number_factorizations::Int64'. The total number of choelsky factorization done for H + δ I.
  'total_number_factorizations_compute_search_direction::Int64'. The number of choelsky factorization done when computing d_k = cholesky(H + (δ + 1e-1) * I) ^ {-1} (-g)
  'temp_total_number_factorizations_inverse_power_iteration::Int64'. The number of choelsky factorization done when solving the hard case instance.
"""
function solveHardCaseLogic(
    g::Vector{Float64},
    H::Union{
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    },
    γ_1::Float64,
    γ_2::Float64,
    r::Float64,
    δ::Float64,
    δ_prime::Float64,
    min_grad::Float64,
    print_level::Int64 = 0,
)
    sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
    total_number_factorizations = 0
    temp_total_number_factorizations_compute_search_direction = 0
    temp_total_number_factorizations_inverse_power_iteration = 0
    temp_total_number_factorizations_ = 0

    temp_eigenvalue = 0
    try
        start_time_temp = time()
        # Compute the minimum eigenvalue of the Hessian matrix using inverse power iteration
        # temp_d_k = cholesky(H + (abs(eigenvalue)) * sparse_identity) \ (-g)
        success,
        eigenvalue,
        eigenvector,
        temp_total_number_factorizations_inverse_power_iteration,
        temp_d_k = inverse_power_iteration(g, H, min_grad, δ, δ_prime, r, γ_1)
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
            zeros(length(g)),
            total_number_factorizations,
            temp_total_number_factorizations_compute_search_direction,
            temp_total_number_factorizations_inverse_power_iteration
        end

        if print_level >= 2
            @info "candidate search direction norm is $norm_temp_d_k. r is $r. γ_2 is $γ_2"
        end
        # Validate that the search direction satisfies the trust-region subproblem termination critera
        # The search direction in the hard case should be approximately on the trust-region boundary
        if γ_2 * r <= norm(temp_d_k) <= r
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
        if norm(temp_d_k) > r
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
        zeros(length(g)),
        total_number_factorizations,
        temp_total_number_factorizations_compute_search_direction,
        temp_total_number_factorizations_inverse_power_iteration
    catch e
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
inverse_power_iteration(g, H, min_grad, δ, δ_prime, r, γ_1, max_iter, ϵ, print_level)

Compute iteratively an approximate value to the minimum eigenvalue of H.

# Inputs:
  - `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
  - `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
  See (1). The Hessian at the current iterate x.
  - `min_grad::Float64`. See (2). The minumum gradient over all iterates.
  - 'δ::Float64'. See (3). The lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
  - 'δ_prime::Float64'. See (3). The upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
  - `r::Float64`. See (1). The trsut-region radius.
  - `γ_1::Float64`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
  - `max_iter::Int64`. The maximum number of iterations to run.
  - `ϵ::Float64`. The tolerance to specify how close the solution should be from the minimum eigenvalue.
  - `print_level::Float64`. The verbosity level of logs.

# Outputs:
  'success::Bool'. See (3). It specifies if we found the minimum eigenvalue of H or not.
  'eigenvalue::Float64'. The minimum eigenvalue of H.
  'eigenvector::::Vector{Float64}'. The eigenvector for the minimum eigenvalue of H.
  'temp_total_number_factorizations_inverse_power_iteration::Int64'. The number of choelsky factorization done when solving the hard case instance.
  'temp_d_k::Vector{Float64}'. temp_d_k =  cholesky(H + (abs(eigenvalue) + 1e-1) * I) ^ {-1} (-g)
"""
function inverse_power_iteration(
    g::Vector{Float64},
    H::Union{
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    },
    min_grad::Float64,
    δ::Float64,
    δ_prime::Float64,
    r::Float64,
    γ_1::Float64;
    max_iter::Int64 = 1000,
    ϵ::Float64 = 1e-3,
    print_level::Int64 = 2,
)
    sigma = δ_prime
    start_time_temp = time()
    n = size(H, 1)
    x = ones(n)
    y = ones(n)
    sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
    y_original_fact = cholesky(H + sigma * sparse_identity)
    temp_factorization = 1
    for k = 1:max_iter
        y = y_original_fact \ x
        y /= norm(y)
        eigenvalue = dot(y, H * y)

        if norm(H * y + δ_prime * y) <= abs(δ_prime - δ) + ((γ_1 * min_grad) / r)
            try
                temp_factorization += 1
                # Validate that H + eigenvalue is positive definite
                temp_d_k = cholesky(H + (abs(eigenvalue) + 1e-1) * sparse_identity) \ (-g)
                return true, eigenvalue, y, temp_factorization, temp_d_k
            catch
                #DO NOTHING
            end
        end

        #Keep as a safety check. This a sign that we can't solve the trust region subprobelm
        if norm(x + y) <= ϵ || norm(x - y) <= ϵ
            eigenvalue = dot(y, H * y)
            try
                temp_factorization += 1
                temp_d_k = cholesky(H + (abs(eigenvalue) + 1e-1) * sparse_identity) \ (-g)
                return true, eigenvalue, y, temp_factorization, temp_d_k
            catch
                #DO NOTHING
            end
        end

        x = y
    end
    temp_ = dot(y, H * y)

    if print_level >= 2
        end_time_temp = time()
        total_time_temp = end_time_temp - start_time_temp
        @info "inverse_power_iteration operation took $total_time_temp."
        println("inverse_power_iteration operation took $total_time_temp.")
    end

    temp_d_k = zeros(length(g))
    return false, temp_, y, temp_factorization, temp_d_k
end
