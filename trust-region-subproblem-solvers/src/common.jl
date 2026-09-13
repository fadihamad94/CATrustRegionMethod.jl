"A description of trust-region subproblem termination failure reason."
struct TrustRegionSubproblemError <: Exception
    msg::String
    failure_reason_6a::Bool
    failure_reason_6b::Bool
    failure_reason_6c::Bool
    failure_reason_6d::Bool
end

Base.showerror(io::IO, error::TrustRegionSubproblemError) = print(io, error.msg)

const DEFAULTS_FILE = joinpath(@__DIR__, "defaults.json")
Base.include_dependency(DEFAULTS_FILE)
const DEFAULTS = JSON.parsefile(DEFAULTS_FILE)

const DEFAULT_GAMMA_1 = Float64(DEFAULTS["solver"]["gamma_1"])
const DEFAULT_GAMMA_2 = Float64(DEFAULTS["solver"]["gamma_2"])
const DEFAULT_GAMMA_3 = Float64(DEFAULTS["solver"]["gamma_3"])
const DEFAULT_SEED = Int(DEFAULTS["solver"]["seed"])
const DEFAULT_PRINT_LEVEL = Int(DEFAULTS["solver"]["print_level"])
const DEFAULT_DENSE_HESSIAN_THRESHOLD =
    Float64(DEFAULTS["solver"]["dense_hessian_threshold"])
const DEFAULT_REUSE_SPARSE_SYMBOLIC_FACTORIZATION =
    Bool(DEFAULTS["solver"]["reuse_sparse_symbolic_factorization"])
const DEFAULT_HANDLE_HARD_CASE = Bool(DEFAULTS["solver"]["handle_hard_case"])
const DEFAULT_USE_BACKUP_TRUST_REGION_SUBPROBLEM_SOLVER =
    Bool(DEFAULTS["solver"]["use_backup_trust_region_subproblem_solver"])
const DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS =
    Int(DEFAULTS["solver"]["iterative_refinement_max_iterations"])
const DIRECT_NEW_SOLVER = "DIRECT-NEW"

const COMMON_INTERNAL_DEFAULTS = DEFAULTS["internal"]["common"]
const DIRECT_NEW_INTERNAL_DEFAULTS = DEFAULTS["internal"][DIRECT_NEW_SOLVER]
const OLD_INTERNAL_DEFAULTS = DEFAULTS["internal"]["OLD"]

const DEFAULT_FIND_INTERVAL_MAX_ITERATIONS =
    Int(DIRECT_NEW_INTERNAL_DEFAULTS["find_interval_max_iterations"])
const DEFAULT_POWER_ITERATION_MAX_ITERATIONS =
    Int(COMMON_INTERNAL_DEFAULTS["power_iteration_max_iterations"])
const DEFAULT_POWER_ITERATION_TOLERANCE =
    Float64(COMMON_INTERNAL_DEFAULTS["power_iteration_tolerance"])
const DEFAULT_BISECTION_MAX_ITERATIONS =
    Int(DIRECT_NEW_INTERNAL_DEFAULTS["bisection_max_iterations"])
const DEFAULT_INVERSE_POWER_MAX_ITERATIONS =
    Int(DIRECT_NEW_INTERNAL_DEFAULTS["inverse_power_max_iterations"])
const DEFAULT_OLD_SOLVER_EIGENDECOMPOSITION_MEMORY_FRACTION =
    Float64(
        OLD_INTERNAL_DEFAULTS["old_solver_eigendecomposition_memory_fraction"],
    )
0.0 < DEFAULT_OLD_SOLVER_EIGENDECOMPOSITION_MEMORY_FRACTION <= 1.0 || error(
    "internal.OLD.old_solver_eigendecomposition_memory_fraction must be in (0, 1].",
)

const HessianMatrix = Union{
    Matrix{Float64},
    SparseMatrixCSC{Float64,Int},
    Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
}

finiteHessian(H::Matrix{Float64}) = all(isfinite, H)
finiteHessian(H::SparseMatrixCSC{Float64,Int}) = all(isfinite, nonzeros(H))
finiteHessian(H::Symmetric{Float64,SparseMatrixCSC{Float64,Int}}) =
    all(isfinite, nonzeros(parent(H)))

mutable struct SparseCholeskyWorkspace
    reuse_sparse_symbolic_factorization::Bool
    factor::Union{Nothing,SparseArrays.CHOLMOD.Factor{Float64,Int}}
    matrix::Union{Nothing,SparseMatrixCSC{Float64,Int}}
    triangle::Symbol
    colptr::Vector{Int}
    rowval::Vector{Int}
    symbolic_factorizations::Int
    dense_buffer::Union{Nothing,Matrix{Float64}}
    negative_gradient::Vector{Float64}
    shifted_residual::Vector{Float64}
    refinement_correction::Vector{Float64}
    iterative_refinement_max_iterations::Int

    SparseCholeskyWorkspace(
        reuse_sparse_symbolic_factorization::Bool = true,
        iterative_refinement_max_iterations::Int =
            DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS,
    ) = begin
        @assert 0 <= iterative_refinement_max_iterations <= 3
        return new(
            reuse_sparse_symbolic_factorization,
            nothing,
            nothing,
            :none,
            Int[],
            Int[],
            0,
            nothing,
            Float64[],
            Float64[],
            Float64[],
            iterative_refinement_max_iterations,
        )
    end
end

const OptionalSparseCholeskyWorkspace = Union{Nothing,SparseCholeskyWorkspace}

function validateGammaParameters(
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
)::Nothing
    0.0 < γ_1 < 1.0 || throw(ArgumentError("γ_1 must lie in (0, 1)."))
    0.0 < γ_2 < 1.0 || throw(ArgumentError("γ_2 must lie in (0, 1)."))
    0.0 < γ_3 < 1.0 || throw(ArgumentError("γ_3 must lie in (0, 1)."))
    return nothing
end

"Counts the Cholesky factorizations performed by one subproblem solve."
struct FactorizationStats
    total::Int64
    findinterval::Int64
    bisection::Int64
    compute_search_direction::Int64
    inverse_power_iteration::Int64

    function FactorizationStats(
        total::Int64,
        findinterval::Int64,
        bisection::Int64,
        compute_search_direction::Int64,
        inverse_power_iteration::Int64,
    )
        @assert total ==
                findinterval +
                bisection +
                compute_search_direction +
                inverse_power_iteration
        return new(
            total,
            findinterval,
            bisection,
            compute_search_direction,
            inverse_power_iteration,
        )
    end
end

"Counts iterative work performed by one trust-region subproblem solve."
struct IterativeStats
    native_status::Int64
    iterations::Int64
    hessian_vector_products::Int64

    function IterativeStats(
        native_status::Integer,
        iterations::Integer,
        hessian_vector_products::Integer,
    )
        iterations >= 0 || throw(ArgumentError("Iteration count must be nonnegative."))
        hessian_vector_products >= 0 ||
            throw(ArgumentError("Hessian-vector product count must be nonnegative."))
        return new(
            Int64(native_status),
            Int64(iterations),
            Int64(hessian_vector_products),
        )
    end
end

IterativeStats() = IterativeStats(0, 0, 0)

"The solution and accounting information returned by a trust-region subproblem solve."
struct TrustRegionSubproblemResult
    success::Bool
    delta::Float64
    delta_prime::Float64
    direction::Vector{Float64}
    hard_case::Bool
    factorizations::FactorizationStats
    iterative_stats::IterativeStats
end

function TrustRegionSubproblemResult(
    success,
    delta,
    delta_prime,
    direction,
    hard_case,
    factorizations,
)
    return TrustRegionSubproblemResult(
        success,
        delta,
        delta_prime,
        direction,
        hard_case,
        factorizations,
        IterativeStats(),
    )
end
