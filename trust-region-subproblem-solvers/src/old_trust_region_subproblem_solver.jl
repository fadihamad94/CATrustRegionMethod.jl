const OLD_SOLVER_EIGENDECOMPOSITION_DENSE_MATRIX_COUNT = UInt128(3)

function oldSolverEigendecompositionMemoryEstimate(n::Integer)::UInt128
    n >= 0 || throw(ArgumentError("Matrix dimension must be nonnegative."))
    dimension = UInt128(n)
    return OLD_SOLVER_EIGENDECOMPOSITION_DENSE_MATRIX_COUNT *
           UInt128(sizeof(Float64)) * dimension^2
end

function oldSolverEigendecompositionMemoryBudget(
    total_memory_bytes::Integer,
    memory_fraction::Real = DEFAULT_OLD_SOLVER_EIGENDECOMPOSITION_MEMORY_FRACTION,
)::UInt128
    total_memory_bytes >= 0 ||
        throw(ArgumentError("Total memory must be nonnegative."))
    0.0 < memory_fraction <= 1.0 ||
        throw(ArgumentError("Eigendecomposition memory fraction must be in (0, 1]."))
    return floor(UInt128, Float64(total_memory_bytes) * Float64(memory_fraction))
end

function guardOldSolverEigendecompositionMemory(
    H::HessianMatrix;
    total_memory_bytes::Integer = Sys.total_memory(),
    memory_fraction::Real = DEFAULT_OLD_SOLVER_EIGENDECOMPOSITION_MEMORY_FRACTION,
)::Nothing
    estimated_bytes = oldSolverEigendecompositionMemoryEstimate(size(H, 1))
    budget_bytes =
        oldSolverEigendecompositionMemoryBudget(total_memory_bytes, memory_fraction)
    estimated_bytes <= budget_bytes && return nothing

    bytes_per_gibibyte = Float64(1024^3)
    matrix_dimension = size(H, 1)
    estimated_gibibytes = Float64(estimated_bytes) / bytes_per_gibibyte
    budget_gibibytes = Float64(budget_bytes) / bytes_per_gibibyte
    @warn "Skipping old-solver dense eigendecomposition because its estimated memory exceeds the configured budget." matrix_dimension estimated_gibibytes budget_gibibytes memory_fraction
    throw(OutOfMemoryError())
end

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
    validateGammaParameters(DEFAULT_GAMMA_1, γ_2, DEFAULT_GAMMA_3)
    _, δ_k, d_k, hard_case = optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    success_subproblem_solve = false
    try
        validateTrustRegionSubproblemTerminationCriteria(
            "old_approach",
            d_k,
            g,
            H,
            δ,
            δ_k,
            δ_k,
            DEFAULT_GAMMA_1,
            γ_2,
            DEFAULT_GAMMA_3,
            r,
            norm(g),
            hard_case,
        )
        success_subproblem_solve = true
    catch e
        e isa OutOfMemoryError && rethrow()
        success_subproblem_solve = false
    end
    return success_subproblem_solve, δ_k, d_k, hard_case
end

#Based on Theorem 4.3 in Numerical Optimization by Wright
function optimizeSecondOrderModelOldApproach(
    g::Vector{Float64},
    H::HessianMatrix,
    δ::Float64,
    ϵ::Float64,
    r::Float64,
)
    #When δ is 0 and the Hessian is positive semidefinite, we can directly compute the direction
    try
        factorization = cholesky(H)
        d_k = factorization \ (-g)
        if norm(d_k, 2) <= r
            return true, 0.0, d_k, false
        end
    catch e
        e isa OutOfMemoryError && rethrow()
        #Do nothing
    end

    try
        δ, δ_prime = findintervalOldApproach(g, H, δ, ϵ, r)
        δ_m = bisectionOldApproach(g, H, δ, ϵ, δ_prime, r)
        d_k = cholesky(H + δ_m * I) \ (-g)
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

function phiOldApproach(
    g::Vector{Float64},
    H::HessianMatrix,
    δ::Float64,
    ϵ::Float64,
    r::Float64,
)
    shifted_hessian = H + δ * I
    #cholesky factorization only works on positive definite matrices
    try
        factorization = cholesky(shifted_hessian)
        computed_norm = norm(factorization \ g, 2)
        if computed_norm < ϵ * r
            return 1
        elseif computed_norm <= r
            return 0
        else
            return -1
        end
    catch e
        e isa OutOfMemoryError && rethrow()
        return -1
    end
end

function findintervalOldApproach(
    g::Vector{Float64},
    H::HessianMatrix,
    δ::Float64,
    ϵ::Float64,
    r::Float64,
)
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
    H::HessianMatrix,
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
function solveHardCaseLogicOldApproach(
    g::Vector{Float64},
    H::HessianMatrix,
    r::Float64,
)
    guardOldSolverEigendecompositionMemory(H)
    eigendecomposition = eigen!(Symmetric(Matrix(H)))
    eigenvaluesVector = eigendecomposition.values
    Q = eigendecomposition.vectors
    minimumEigenValue = first(eigenvaluesVector)
    δ = -minimumEigenValue
    projected_gradient = Q' * g
    coefficients = zeros(Float64, length(eigenvaluesVector))
    norm_d_k_squared_without_τ_squared = 0.0

    for i = 1:length(eigenvaluesVector)
        if eigenvaluesVector[i] != minimumEigenValue
            coefficient = -projected_gradient[i] / (eigenvaluesVector[i] + δ)
            coefficients[i] = coefficient
            norm_d_k_squared_without_τ_squared += abs2(coefficient)
        end
    end

    norm_d_k_squared = r^2
    τ = sqrt(norm_d_k_squared - norm_d_k_squared_without_τ_squared)
    coefficients[1] = τ
    d_k = Q * coefficients

    d_k = floatingPointSafeBallProjection(d_k, r)
    return δ, d_k
end
