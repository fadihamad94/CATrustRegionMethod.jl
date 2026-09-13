EnumX.@enumx TerminationStatusCode begin
    OPTIMAL
    UNBOUNDED
    ITERATION_LIMIT
    TIME_LIMIT
    MEMORY_LIMIT
    STEP_SIZE_LIMIT
    NUMERICAL_ERROR
    TRUST_REGION_SUBPROBLEM_ERROR
    OTHER_ERROR
    INVALID_MODEL
    INNER_ITERATION_LIMIT
end

const DEFAULTS_FILE = joinpath(@__DIR__, "defaults.json")
Base.include_dependency(DEFAULTS_FILE)
const DEFAULTS = JSON.parsefile(DEFAULTS_FILE)

const DEFAULT_MAX_ITERATIONS = CATrustRegionShared.DEFAULT_MAX_ITERATIONS
const DEFAULT_MAX_INNER_ITERATIONS =
    Int(DEFAULTS["termination"]["max_inner_iterations"])
const DEFAULT_GRADIENT_TERMINATION_TOLERANCE =
    CATrustRegionShared.DEFAULT_GRADIENT_TERMINATION_TOLERANCE
const DEFAULT_MAX_TIME = CATrustRegionShared.DEFAULT_MAX_TIME
const DEFAULT_STEP_SIZE_LIMIT = CATrustRegionShared.DEFAULT_STEP_SIZE_LIMIT
const DEFAULT_MINIMUM_OBJECTIVE_FUNCTION =
    CATrustRegionShared.DEFAULT_MINIMUM_OBJECTIVE_FUNCTION
const DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS =
    CATrustRegionShared.DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS

const DEFAULT_RHO_0 = Float64(DEFAULTS["algorithm"]["rho_0"])
const DEFAULT_RHO_MIN = Float64(DEFAULTS["algorithm"]["rho_min"])
const DEFAULT_ETA = Float64(DEFAULTS["algorithm"]["eta"])
const DEFAULT_XI = Float64(DEFAULTS["algorithm"]["xi"])
const DEFAULT_MU_1 = Float64(DEFAULTS["algorithm"]["mu_1"])
const DEFAULT_MU_2 = Float64(DEFAULTS["algorithm"]["mu_2"])
const DEFAULT_GAMMA_1 = TrustRegionSubproblemSolvers.DEFAULT_GAMMA_1
const DEFAULT_GAMMA_2 = TrustRegionSubproblemSolvers.DEFAULT_GAMMA_2
const DEFAULT_GAMMA_3 = TrustRegionSubproblemSolvers.DEFAULT_GAMMA_3
const DEFAULT_SEED = Int(DEFAULTS["algorithm"]["seed"])
const DEFAULT_PRINT_LEVEL = Int(DEFAULTS["algorithm"]["print_level"])
const DEFAULT_TRUST_REGION_SUBPROBLEM_SOLVER =
    String(DEFAULTS["algorithm"]["trust_region_subproblem_solver"])
const DEFAULT_DENSE_HESSIAN_THRESHOLD =
    TrustRegionSubproblemSolvers.DEFAULT_DENSE_HESSIAN_THRESHOLD
const DEFAULT_REUSE_SPARSE_SYMBOLIC_FACTORIZATION =
    TrustRegionSubproblemSolvers.DEFAULT_REUSE_SPARSE_SYMBOLIC_FACTORIZATION
const DEFAULT_USE_BACKUP_TRUST_REGION_SUBPROBLEM_SOLVER =
    TrustRegionSubproblemSolvers.DEFAULT_USE_BACKUP_TRUST_REGION_SUBPROBLEM_SOLVER
const DEFAULT_HANDLE_HARD_CASE =
    TrustRegionSubproblemSolvers.DEFAULT_HANDLE_HARD_CASE
DEFAULT_TRUST_REGION_SUBPROBLEM_SOLVER == DIRECT_NEW_SOLVER || error(
    "The UTR default trust-region subproblem solver must be $DIRECT_NEW_SOLVER.",
)

@define_termination_criteria :utr

function validateTerminationCriteria(criteria::TerminationCriteria)
    CATrustRegionShared.validate_common_termination_values(
        criteria.MAX_ITERATIONS,
        criteria.gradient_termination_tolerance,
        criteria.MAX_TIME,
        criteria.STEP_SIZE_LIMIT,
        criteria.MINIMUM_OBJECTIVE_FUNCTION,
        criteria.iterative_refinement_max_iterations,
    )
    @assert criteria.MAX_INNER_ITERATIONS > 0
    return criteria
end

function _validateAlgorithmicParameterValues(
    ρ_0::Float64,
    ρ_min::Float64,
    η::Float64,
    ξ::Float64,
    μ_1::Float64,
    μ_2::Float64,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    ::Int64,
    print_level::Int64,
    dense_hessian_threshold::Float64,
    ::Bool,
    ::Bool,
    ::Bool,
    trust_region_subproblem_solver::String,
)
    @assert isfinite(ρ_0) && ρ_0 > 0
    @assert isfinite(ρ_min) && ρ_min > 0
    @assert 0 < η < 1 / 32
    @assert 1 / 4 < ξ < 1
    @assert isfinite(μ_1) && μ_1 > 1
    @assert isfinite(μ_2) && μ_2 > 1
    @assert 0 < γ_1 < 1
    @assert 0 < γ_2 < 1
    @assert 0 < γ_3 < 1
    @assert print_level >= -1
    @assert 0 <= dense_hessian_threshold <= 1
    @assert trust_region_subproblem_solver == DIRECT_NEW_SOLVER
    return nothing
end

"""
    AlgorithmicParameters

Adaptive UTR parameters together with settings forwarded to the shared
trust-region subproblem solver. `ρ_0` and `ρ_min` are independently positive;
the method intentionally does not require `ρ_0 >= ρ_min`.
"""
mutable struct AlgorithmicParameters
    ρ_0::Float64
    ρ_min::Float64
    η::Float64
    ξ::Float64
    μ_1::Float64
    μ_2::Float64
    γ_1::Float64
    γ_2::Float64
    γ_3::Float64
    seed::Int64
    print_level::Int64
    dense_hessian_threshold::Float64
    reuse_sparse_symbolic_factorization::Bool
    use_backup_trust_region_subproblem_solver::Bool
    handle_hard_case::Bool
    trust_region_subproblem_solver::String

    function AlgorithmicParameters(
        ρ_0::Float64 = DEFAULT_RHO_0,
        ρ_min::Float64 = DEFAULT_RHO_MIN,
        η::Float64 = DEFAULT_ETA,
        ξ::Float64 = DEFAULT_XI,
        μ_1::Float64 = DEFAULT_MU_1,
        μ_2::Float64 = DEFAULT_MU_2,
        γ_1::Float64 = DEFAULT_GAMMA_1,
        γ_2::Float64 = DEFAULT_GAMMA_2,
        γ_3::Float64 = DEFAULT_GAMMA_3,
        seed::Int64 = DEFAULT_SEED,
        print_level::Int64 = DEFAULT_PRINT_LEVEL,
        dense_hessian_threshold::Float64 = DEFAULT_DENSE_HESSIAN_THRESHOLD,
        reuse_sparse_symbolic_factorization::Bool =
            DEFAULT_REUSE_SPARSE_SYMBOLIC_FACTORIZATION,
        use_backup_trust_region_subproblem_solver::Bool =
            DEFAULT_USE_BACKUP_TRUST_REGION_SUBPROBLEM_SOLVER,
        handle_hard_case::Bool = DEFAULT_HANDLE_HARD_CASE,
        trust_region_subproblem_solver::String =
            DEFAULT_TRUST_REGION_SUBPROBLEM_SOLVER,
    )
        _validateAlgorithmicParameterValues(
            ρ_0,
            ρ_min,
            η,
            ξ,
            μ_1,
            μ_2,
            γ_1,
            γ_2,
            γ_3,
            seed,
            print_level,
            dense_hessian_threshold,
            reuse_sparse_symbolic_factorization,
            use_backup_trust_region_subproblem_solver,
            handle_hard_case,
            trust_region_subproblem_solver,
        )
        return new(
            ρ_0,
            ρ_min,
            η,
            ξ,
            μ_1,
            μ_2,
            γ_1,
            γ_2,
            γ_3,
            seed,
            print_level,
            dense_hessian_threshold,
            reuse_sparse_symbolic_factorization,
            use_backup_trust_region_subproblem_solver,
            handle_hard_case,
            trust_region_subproblem_solver,
        )
    end
end

function validateAlgorithmicParameters(parameters::AlgorithmicParameters)
    _validateAlgorithmicParameterValues(
        parameters.ρ_0,
        parameters.ρ_min,
        parameters.η,
        parameters.ξ,
        parameters.μ_1,
        parameters.μ_2,
        parameters.γ_1,
        parameters.γ_2,
        parameters.γ_3,
        parameters.seed,
        parameters.print_level,
        parameters.dense_hessian_threshold,
        parameters.reuse_sparse_symbolic_factorization,
        parameters.use_backup_trust_region_subproblem_solver,
        parameters.handle_hard_case,
        parameters.trust_region_subproblem_solver,
    )
    return parameters
end

"""Oracle, trial, and factorization counts accumulated by one UTR solve."""
@define_algorithm_counter :utr
