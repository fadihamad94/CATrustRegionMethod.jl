@define_termination_status

const DEFAULTS_FILE = joinpath(@__DIR__, "defaults.json")
Base.include_dependency(DEFAULTS_FILE)
const DEFAULTS = JSON.parsefile(DEFAULTS_FILE)
const DEFAULT_MAX_ITERATIONS = CATrustRegionShared.DEFAULT_MAX_ITERATIONS
const DEFAULT_GRADIENT_TERMINATION_TOLERANCE =
    CATrustRegionShared.DEFAULT_GRADIENT_TERMINATION_TOLERANCE
const DEFAULT_MAX_TIME = CATrustRegionShared.DEFAULT_MAX_TIME
const DEFAULT_STEP_SIZE_LIMIT = CATrustRegionShared.DEFAULT_STEP_SIZE_LIMIT
const DEFAULT_MINIMUM_OBJECTIVE_FUNCTION =
    CATrustRegionShared.DEFAULT_MINIMUM_OBJECTIVE_FUNCTION
const DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS =
    CATrustRegionShared.DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS

const DEFAULT_BETA = Float64(DEFAULTS["algorithm"]["beta"])
const DEFAULT_THETA = Float64(DEFAULTS["algorithm"]["theta"])
const DEFAULT_OMEGA_1 = Float64(DEFAULTS["algorithm"]["omega_1"])
const DEFAULT_OMEGA_2 = Float64(DEFAULTS["algorithm"]["omega_2"])
const DEFAULT_GAMMA_1 = TrustRegionSubproblemSolvers.DEFAULT_GAMMA_1
const DEFAULT_GAMMA_2 = TrustRegionSubproblemSolvers.DEFAULT_GAMMA_2
const DEFAULT_GAMMA_3 = TrustRegionSubproblemSolvers.DEFAULT_GAMMA_3
const DEFAULT_XI = Float64(DEFAULTS["algorithm"]["xi"])
const DEFAULT_INITIAL_RADIUS = Float64(DEFAULTS["algorithm"]["initial_radius"])
const DEFAULT_INITIAL_RADIUS_MULTIPLICATIVE_RULE =
    Float64(DEFAULTS["algorithm"]["initial_radius_multiplicative_rule"])
const DEFAULT_SEED = Int(DEFAULTS["algorithm"]["seed"])
const DEFAULT_PRINT_LEVEL = Int(DEFAULTS["algorithm"]["print_level"])
const DEFAULT_RADIUS_UPDATE_RULE_APPROACH =
    String(DEFAULTS["algorithm"]["radius_update_rule_approach"])
const DEFAULT_EVAL_OFFSET = Float64(DEFAULTS["algorithm"]["eval_offset"])
const DEFAULT_TRUST_REGION_SUBPROBLEM_SOLVER =
    String(DEFAULTS["algorithm"]["trust_region_subproblem_solver"])
const DEFAULT_DENSE_HESSIAN_THRESHOLD =
    TrustRegionSubproblemSolvers.DEFAULT_DENSE_HESSIAN_THRESHOLD
const DEFAULT_REUSE_SPARSE_SYMBOLIC_FACTORIZATION =
    TrustRegionSubproblemSolvers.DEFAULT_REUSE_SPARSE_SYMBOLIC_FACTORIZATION
const DEFAULT_HANDLE_HARD_CASE =
    TrustRegionSubproblemSolvers.DEFAULT_HANDLE_HARD_CASE
const DEFAULT_USE_BACKUP_TRUST_REGION_SUBPROBLEM_SOLVER =
    TrustRegionSubproblemSolvers.DEFAULT_USE_BACKUP_TRUST_REGION_SUBPROBLEM_SOLVER
const DEFAULT_DELTA = Float64(DEFAULTS["algorithm"]["delta"])
const DEFAULT_OLD_SOLVER_EIGENDECOMPOSITION_MEMORY_FRACTION =
    TrustRegionSubproblemSolvers.DEFAULT_OLD_SOLVER_EIGENDECOMPOSITION_MEMORY_FRACTION
0.0 < DEFAULT_OLD_SOLVER_EIGENDECOMPOSITION_MEMORY_FRACTION <= 1.0 || error(
    "TrustRegionSubproblemSolvers internal.OLD." *
    "old_solver_eigendecomposition_memory_fraction must be in (0, 1].",
)

const DEFAULT_POWER_ITERATION_MAX_ITERATIONS =
    Int(DEFAULTS["internal"]["common"]["power_iteration_max_iterations"])

DEFAULT_TRUST_REGION_SUBPROBLEM_SOLVER == DIRECT_NEW_SOLVER || error(
    "The CAT default trust-region subproblem solver must be $DIRECT_NEW_SOLVER.",
)
function validateTrustRegionSubproblemSolverParameters(
    trust_region_subproblem_solver::String,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
)
    @assert trust_region_subproblem_solver in ("OLD", DIRECT_NEW_SOLVER)
    @assert 0 < γ_1 < 1
    @assert 0 < γ_2 < 1
    @assert 0 < γ_3 < 1
    return nothing
end

@define_termination_criteria :cat

function validateTerminationCriteria(criteria::TerminationCriteria)
    CATrustRegionShared.validate_common_termination_values(
        criteria.MAX_ITERATIONS,
        criteria.gradient_termination_tolerance,
        criteria.MAX_TIME,
        criteria.STEP_SIZE_LIMIT,
        criteria.MINIMUM_OBJECTIVE_FUNCTION,
        criteria.iterative_refinement_max_iterations,
    )
    return criteria
end

mutable struct AlgorithmicParameters
    """
    β param for the algorithm. It is a threshold for ρ_hat when updating the trust-region radius.
    """
    β::Float64
    """
    θ param for the algorithm. It is used for computing ρ_hat.
    """
    θ::Float64
    """
    ω_1 param for the algorithm. When ρ_hat < β, we set r_k = r_k / ω_1.
    """
    ω_1::Float64
    """
    ω_2 param for the algorithm. When ρ_hat ≧ β, we set r_k = max(ω_2 ||d_k||, r_k).
       Where d_k is the the search direction.
    """
    ω_2::Float64
    """
    γ_1 param for the algorithm. It is used for the trust-region subproblem termination criteria.
    """
    γ_1::Float64
    """
    γ_2 param for the algorithm. It is used for the trust-region subproblem termination criteria.
    """
    γ_2::Float64
    """
    γ_3 param for the algorithm. It is used for the trust-region subproblem termination criteria.
    """
    γ_3::Float64
    """
    ξ param for the algorithm. It is used for the trust-region subproblem termination criteria.
    """
    ξ::Float64
    """
    The required initial value of the trust-region radius.
    """
    r_1::Float64
    """
    If r_1 ≤ 0, then the radius will be choosen automatically based on a heursitic appraoch.
    The default is INITIAL_RADIUS_MULTIPLICATIVE_RULE * ||g_1|| / ||H_1|| where ||g_1|| is the
    l2 norm for gradient at the initial iterate and ||H_1|| is the spectral norm for the hessian
    at the initial iterate.
    """
    INITIAL_RADIUS_MULTIPLICATIVE_RULE::Float64
    """
    Specify seed level for randomness.
    """
    seed::Int64
    """
    The verbosity level of logs.
    """
    print_level::Int64
    """
    This to be able to test the performance of the algorithm for the ablation study
    when comparing versus the conference version of the paper.
    """
    radius_update_rule_approach::String
    """
    eval_offset param for the algorithm. It is used for the trust-region subproblem termination criteria.
    """
    eval_offset::Float64
    """
    trust_region_subproblem_solver param for the algorithm. It is used to determine which method to use the
    trust-region subproblem. Using the new appraoch or the old appraoch in the NEURips paper.
    """
    trust_region_subproblem_solver::String
    "Density above which sparse Hessians are converted to dense matrices."
    dense_hessian_threshold::Float64
    "Whether sparse Cholesky factorizations reuse symbolic analysis."
    reuse_sparse_symbolic_factorization::Bool
    "Whether to retry the trust-region subproblem with a perturbed gradient."
    use_backup_trust_region_subproblem_solver::Bool
    "Whether to use inverse-power iteration to handle hard-case subproblems."
    handle_hard_case::Bool
    # initialize parameters
    function AlgorithmicParameters(
        β::Float64 = DEFAULT_BETA,
        θ::Float64 = DEFAULT_THETA,
        ω_1::Float64 = DEFAULT_OMEGA_1,
        ω_2::Float64 = DEFAULT_OMEGA_2,
        γ_1::Float64 = DEFAULT_GAMMA_1,
        γ_2::Float64 = DEFAULT_GAMMA_2,
        γ_3::Float64 = DEFAULT_GAMMA_3,
        ξ::Float64 = DEFAULT_XI,
        r_1::Float64 = DEFAULT_INITIAL_RADIUS,
        INITIAL_RADIUS_MULTIPLICATIVE_RULE::Float64 = DEFAULT_INITIAL_RADIUS_MULTIPLICATIVE_RULE,
        seed::Int64 = DEFAULT_SEED,
        print_level::Int64 = DEFAULT_PRINT_LEVEL,
        radius_update_rule_approach::String = DEFAULT_RADIUS_UPDATE_RULE_APPROACH,
        eval_offset::Float64 = DEFAULT_EVAL_OFFSET,
        trust_region_subproblem_solver::String = DEFAULT_TRUST_REGION_SUBPROBLEM_SOLVER, #This is mainly for ablation study to compare against old approach (conference version)
        dense_hessian_threshold::Float64 = DEFAULT_DENSE_HESSIAN_THRESHOLD,
        reuse_sparse_symbolic_factorization::Bool = DEFAULT_REUSE_SPARSE_SYMBOLIC_FACTORIZATION,
        use_backup_trust_region_subproblem_solver::Bool =
            DEFAULT_USE_BACKUP_TRUST_REGION_SUBPROBLEM_SOLVER,
        handle_hard_case::Bool = DEFAULT_HANDLE_HARD_CASE,
    )
        @assert(β > 0 && β < 1)
        @assert(θ >= 0 && θ < 1)
        @assert(ω_1 > 1)
        @assert(ω_2 >= 1)
        @assert(ω_2 >= ω_1)
        @assert(0 < γ_3 < 1)
        @assert(ξ >= 0)
        @assert(eval_offset >= 0)
        @assert(INITIAL_RADIUS_MULTIPLICATIVE_RULE > 0)
        @assert(0.0 <= dense_hessian_threshold <= 1.0)
        @assert(0 < γ_1 < 0.5 * (1 - ((β * θ) / (γ_3 * (1 - β)))))
        @assert(1 / ω_1 < γ_2 < 1)
        validateTrustRegionSubproblemSolverParameters(
            trust_region_subproblem_solver,
            γ_1,
            γ_2,
            γ_3,
        )
        return new(
            β,
            θ,
            ω_1,
            ω_2,
            γ_1,
            γ_2,
            γ_3,
            ξ,
            r_1,
            INITIAL_RADIUS_MULTIPLICATIVE_RULE,
            seed,
            print_level,
            radius_update_rule_approach,
            eval_offset,
            trust_region_subproblem_solver,
            dense_hessian_threshold,
            reuse_sparse_symbolic_factorization,
            use_backup_trust_region_subproblem_solver,
            handle_hard_case,
        )
    end
end

@define_algorithm_counter :cat
