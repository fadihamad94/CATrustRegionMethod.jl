export phiOldApproach, findintervalOldApproach, bisectionOldApproach
#=
The big picture idea here is to optimize the trust region subproblem using a factorization method based
on the optimality conditions:
H d_k + g + δ d_k = 0
H + δ I ≥ 0
δ(r -  ||d_k ||) = 0

That is why we defined the below phiOldApproach to solve that using bisectionOldApproach logic.
=#
function solveTrustRegionSubproblemOldApproach(
    f::Float64,
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    },
    x_k::Vector{Float64},
    δ::Float64,
    γ_2::Float64,
    r::Float64,
)
    hard_case, δ_k, d_k = optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    success_subproblem_solve = false
    try
        γ_3 = 0.5
        validateTrustRegionSubproblemTerminationCriteria(
            d_k,
            g,
            H,
            δ_k,
            γ_2,
            γ_3,
            r,
            hard_case,
        )
        success_subproblem_solve = true
    catch e
        success_subproblem_solve = false
    end
    return success_subproblem_solve, δ_k, d_k, hard_case
end

function validateTrustRegionSubproblemTerminationCriteria(
    d_k::Vector{Float64},
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
    },
    δ_k::Float64,
    γ_2::Float64,
    γ_3::Float64,
    r::Float64,
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
    condition_6a = true
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
    # if model_val > 0 || !condition
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

#Based on Theorem 4.3 in Numerical Optimization by Wright
function optimizeSecondOrderModelOldApproach(
    g::Vector{Float64},
    H,
    δ::Float64,
    ϵ::Float64,
    r::Float64,
)
    #When δ is 0 and the Hessian is positive semidefinite, we can directly compute the direction
    try
        cholesky(Matrix(H))
        d_k = H \ (-g)
        if norm(d_k, 2) <= r
            return true, 0.0, d_k, false
        end
    catch e
        #Do nothing
    end

    try
        δ, δ_prime = findintervalOldApproach(g, H, δ, ϵ, r)
        δ_m = bisectionOldApproach(g, H, δ, ϵ, δ_prime, r)
        sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
        d_k = (cholesky(H + δ_m * sparse_identity) \ (-g))
        return true, δ_m, d_k, false
    catch e
        if e == ErrorException(
            "bisectionOldApproach logic failed to find a root for the phiOldApproach function",
        )
            δ, d_k = solveHardCaseLogicOldApproach(g, H, r)
            return true, δ, d_k, true
        elseif e == ErrorException(
            "bisectionOldApproach logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0.",
        )
            δ, d_k = solveHardCaseLogicOldApproach(g, H, r)
            return true, δ, d_k, true
        else
            throw(e)
        end
    end
end

function phiOldApproach(g::Vector{Float64}, H, δ::Float64, ϵ::Float64, r::Float64)
    sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
    shifted_hessian = H + δ * sparse_identity
    #cholesky factorization only works on positive definite matrices
    try
        cholesky(shifted_hessian)
        computed_norm = norm(shifted_hessian \ g, 2)
        if computed_norm < ϵ * r
            return 1
        elseif computed_norm <= r
            return 0
        else
            return -1
        end
    catch e
        return -1
    end
end

function findintervalOldApproach(g::Vector{Float64}, H, δ::Float64, ϵ::Float64, r::Float64)
    Φ_δ = phiOldApproach(g, H, 0.0, ϵ, r)

    if Φ_δ == 0
        δ = 0.0
        δ_prime = 0.0
        return δ, δ_prime
    end

    Φ_δ = phiOldApproach(g, H, δ, ϵ, r)

    if Φ_δ == 0
        δ_prime = δ
        return δ, δ_prime
    end

    δ_prime = δ == 0.0 ? 1.0 : δ * 2
    Φ_δ_prime = 0.0

    k = 1
    while k < 100
        Φ_δ_prime = phiOldApproach(g, H, δ_prime, ϵ, r)
        if Φ_δ_prime == 0
            δ = δ_prime
            return δ, δ_prime
        end

        if ((Φ_δ * Φ_δ_prime) < 0)
            break
        end
        if Φ_δ_prime > 0
            δ_prime = δ_prime / 2
        elseif Φ_δ_prime < 0
            δ_prime = δ_prime * 2
        end
        k = k + 1
    end

    #switch so that δ for ϕ_δ >= 0 and δ_prime for ϕ_δ_prime <= 0
    if Φ_δ_prime > 0 && Φ_δ < 0
        δ_temp = δ
        Φ_δ_temp = Φ_δ
        δ = δ_prime
        δ_prime = δ_temp
        Φ_δ = Φ_δ_prime
        Φ_δ_prime = Φ_δ_temp
    end

    if (Φ_δ * Φ_δ_prime > 0)
        throw(
            error(
                "bisectionOldApproach logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0.",
            ),
        )
    end
    return δ, δ_prime
end

function bisectionOldApproach(
    g::Vector{Float64},
    H,
    δ::Float64,
    ϵ::Float64,
    δ_prime::Float64,
    r::Float64,
)
    # the input of the function is the two end of the interval (δ,δ_prime)
    # our goal here is to find the approximate δ using classic bisectionOldApproach method

    #bisectionOldApproach logic
    k = 1
    δ_m = (δ + δ_prime) / 2
    Φ_δ_m = phiOldApproach(g, H, δ_m, ϵ, r)

    while (Φ_δ_m != 0) && k <= 100
        if Φ_δ_m > 0
            δ = δ_m
        else
            δ_prime = δ_m
        end
        δ_m = (δ + δ_prime) / 2
        Φ_δ_m = phiOldApproach(g, H, δ_m, ϵ, r)
        k = k + 1
    end

    if (Φ_δ_m != 0)
        throw(
            error(
                "bisectionOldApproach logic failed to find a root for the phiOldApproach function",
            ),
        )
    end
    return δ_m
end

#Based on 'THE HARD CASE' section from Numerical Optimization by Nocedal and Wright
function solveHardCaseLogicOldApproach(g::Vector{Float64}, H, r::Float64)
    minimumEigenValue = eigmin(Matrix(H))
    δ = -minimumEigenValue
    z = eigvecs(Matrix(H))[:, 1]
    Q = eigvecs(Matrix(H))
    eigenvaluesVector = eigvals(Matrix(H))

    D = zeros(size(Q))
    for i = 1:size(D)[1]
        D[i, i] = eigenvaluesVector[i]
    end

    norm_d_k_squared_without_τ_squared = 0.0

    for i = 1:length(eigenvaluesVector)
        if eigenvaluesVector[i] != minimumEigenValue
            norm_d_k_squared_without_τ_squared =
                norm_d_k_squared_without_τ_squared +
                ((Q[:, i]' * g)^2 / (eigenvaluesVector[i] + δ)^2)
        end
    end

    norm_d_k_squared = r^2
    τ = sqrt(norm_d_k_squared - norm_d_k_squared_without_τ_squared)
    d_k = τ .* z

    for i = 1:length(eigenvaluesVector)
        if eigenvaluesVector[i] != minimumEigenValue
            d_k = d_k .- ((Q[:, i]' * g) / (eigenvaluesVector[i] + δ)) * Q[:, i]
        end
    end

    return δ, d_k
end
