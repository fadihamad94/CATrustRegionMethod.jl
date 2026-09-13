module BenchmarkPipeline

using CATrustRegionShared: canonical_status_string
using CATrustRegionMethod
using CATNeurIPS
using CSV
using CUTEst
using DataFrames
using Dates
using JSON
using LinearAlgebra
using NLPModels
using SHA
using Statistics
import TrustRegionSubproblemSolvers
import UniversalTrustRegionMethod

include("run_profiles.jl")
using .BenchmarkRunProfiles
include("ablation_variants.jl")
using .AblationVariants

const SETTINGS_SCHEMA_VERSION = 18
const RESULT_SCHEMA_VERSION = 8
const STATE_SCHEMA_VERSION = 1
const SUMMARY_SCHEMA_VERSION = 6
const SUMMARY_SHIFT = 1.0
const RUN_LIMITS_SCHEMA_VERSION = 1
const WARMUP_PROBLEM = "ARGLINA"
const WARMUP_MAX_ITERATIONS = 10
const WARMUP_MAX_TIME_SECONDS = 60.0
const CAT_NEURIPS_VARIANT = "CAT-NeurIPS"
const STANDARD_VARIANTS = ["CAT", CAT_NEURIPS_VARIANT, "UTR"]
const SUCCESS_STATUSES = Set(["SUCCESS", "OPTIMAL"])
const METRIC_NAMES = (
    "total_execution_time",
    "total_function_evaluation",
    "total_gradient_evaluation",
    "total_hessian_evaluation",
    "total_factorization_evaluation",
    "total_subproblem_iterations",
    "total_hessian_vector_products",
)

struct AlgorithmConfiguration
    beta::Float64
    theta::Float64
    omega_1::Float64
    omega_2::Float64
    gamma_1::Float64
    gamma_2::Float64
    gamma_3::Float64
    xi::Float64
    initial_radius::Float64
    initial_radius_multiplicative_rule::Float64
    seed::Int64
    print_level::Int64
    radius_update_rule_approach::String
    eval_offset::Float64
    trust_region_subproblem_solver::String
    dense_hessian_threshold::Float64
    reuse_sparse_symbolic_factorization::Bool
    use_backup_trust_region_subproblem_solver::Bool
    handle_hard_case::Bool
    delta::Float64
end

struct UTRAlgorithmConfiguration
    max_inner_iterations::Int64
    rho_0::Float64
    rho_min::Float64
    eta::Float64
    xi::Float64
    mu_1::Float64
    mu_2::Float64
    gamma_1::Float64
    gamma_2::Float64
    gamma_3::Float64
    seed::Int64
    print_level::Int64
    dense_hessian_threshold::Float64
    reuse_sparse_symbolic_factorization::Bool
    use_backup_trust_region_subproblem_solver::Bool
    handle_hard_case::Bool
    trust_region_subproblem_solver::String
end

struct CATNeurIPSConfiguration
    beta::Float64
    theta::Float64
    omega::Float64
    initial_radius::Float64
    delta::Float64
    gamma_2::Float64
    print_level::Int64
end

const SolverConfiguration =
    Union{
        AlgorithmConfiguration,
        CATNeurIPSConfiguration,
        UTRAlgorithmConfiguration,
    }

struct Experiment
    benchmark_kind::String
    problem_set::String
    results_folder::String
    batches::Vector{Vector{String}}
    problems::Vector{String}
    skipped_problems::Vector{String}
    variants::Vector{String}
    seed::Int64
    print_level::Int64
    num_threads::Int64
    max_iterations::Int64
    max_time_seconds::Float64
    gradient_tolerance::Float64
    step_size_limit::Float64
    minimum_objective_function::Float64
    iterative_refinement_max_iterations::Int64
    gamma_1::Float64
    trust_region_subproblem_solver::String
    source_fingerprint::String
end

struct SolverOutcome
    status::String
    total_execution_time::Float64
    function_value::Union{Nothing,Float64}
    gradient_value::Union{Nothing,Float64}
    terminal_objective_value::Union{Nothing,Float64}
    terminal_gradient_norm::Union{Nothing,Float64}
    best_gradient_objective_value::Union{Nothing,Float64}
    best_gradient_norm::Union{Nothing,Float64}
    total_function_evaluation::Int64
    total_gradient_evaluation::Int64
    total_hessian_evaluation::Int64
    total_factorization_evaluation::Int64
    total_subproblem_iterations::Int64
    total_hessian_vector_products::Int64
    total_iterations::Int64
    error_type::Union{Nothing,String}
    error_message::Union{Nothing,String}
    solution::Union{Nothing,Vector{Float64}}
    solver_specific::Dict{String,Any}
end

struct OptimalityValidationError <: Exception
    message::String
end

Base.showerror(io::IO, error::OptimalityValidationError) = print(io, error.message)

struct OptimalityValidation
    performed::Bool
    passed::Union{Nothing,Bool}
    gradient_norm::Union{Nothing,Float64}
    time_seconds::Float64
    error_type::Union{Nothing,String}
    error_message::Union{Nothing,String}
end

struct BenchmarkResult
    schema_version::Int64
    benchmark_kind::String
    problem_set::String
    variant::String
    problem::String
    seed::Int64
    solver_status::String
    status::String
    total_execution_time::Float64
    function_value::Union{Nothing,Float64}
    gradient_value::Union{Nothing,Float64}
    terminal_objective_value::Union{Nothing,Float64}
    terminal_gradient_norm::Union{Nothing,Float64}
    best_gradient_objective_value::Union{Nothing,Float64}
    best_gradient_norm::Union{Nothing,Float64}
    total_function_evaluation::Int64
    total_gradient_evaluation::Int64
    total_hessian_evaluation::Int64
    total_factorization_evaluation::Int64
    total_subproblem_iterations::Int64
    total_hessian_vector_products::Int64
    total_iterations::Int64
    wall_time_seconds::Float64
    allocated_bytes::Int64
    gc_time_seconds::Float64
    max_rss_bytes::Int64
    optimality_validation_performed::Bool
    optimality_validation_passed::Union{Nothing,Bool}
    optimality_validation_gradient_norm::Union{Nothing,Float64}
    optimality_validation_time_seconds::Float64
    solver_specific::Dict{String,Any}
    diagnostic_directory::String
    error_type::Union{Nothing,String}
    error_message::Union{Nothing,String}
    completed_at::String
end

function iso_timestamp()::String
    return Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")
end

function default_algorithm_configuration(
    seed::Int64;
    print_level::Int64 = CATrustRegionMethod.DEFAULT_PRINT_LEVEL,
    gamma_1::Float64 = CATrustRegionMethod.DEFAULT_GAMMA_1,
    trust_region_subproblem_solver::String =
        CATrustRegionMethod.DEFAULT_TRUST_REGION_SUBPROBLEM_SOLVER,
)::AlgorithmConfiguration
    return AlgorithmConfiguration(
        CATrustRegionMethod.DEFAULT_BETA,
        CATrustRegionMethod.DEFAULT_THETA,
        CATrustRegionMethod.DEFAULT_OMEGA_1,
        CATrustRegionMethod.DEFAULT_OMEGA_2,
        gamma_1,
        CATrustRegionMethod.DEFAULT_GAMMA_2,
        CATrustRegionMethod.DEFAULT_GAMMA_3,
        CATrustRegionMethod.DEFAULT_XI,
        CATrustRegionMethod.DEFAULT_INITIAL_RADIUS,
        CATrustRegionMethod.DEFAULT_INITIAL_RADIUS_MULTIPLICATIVE_RULE,
        seed,
        print_level,
        CATrustRegionMethod.DEFAULT_RADIUS_UPDATE_RULE_APPROACH,
        CATrustRegionMethod.DEFAULT_EVAL_OFFSET,
        trust_region_subproblem_solver,
        CATrustRegionMethod.DEFAULT_DENSE_HESSIAN_THRESHOLD,
        CATrustRegionMethod.DEFAULT_REUSE_SPARSE_SYMBOLIC_FACTORIZATION,
        CATrustRegionMethod.DEFAULT_USE_BACKUP_TRUST_REGION_SUBPROBLEM_SOLVER,
        CATrustRegionMethod.DEFAULT_HANDLE_HARD_CASE,
        CATrustRegionMethod.DEFAULT_DELTA,
    )
end

function algorithm_configuration(
    variant::String,
    seed::Int64;
    print_level::Int64 = CATrustRegionMethod.DEFAULT_PRINT_LEVEL,
    gamma_1::Float64 = CATrustRegionMethod.DEFAULT_GAMMA_1,
    trust_region_subproblem_solver::String =
        CATrustRegionMethod.DEFAULT_TRUST_REGION_SUBPROBLEM_SOLVER,
)::AlgorithmConfiguration
    base = default_algorithm_configuration(
        seed;
        print_level,
        gamma_1,
        trust_region_subproblem_solver,
    )
    (variant == "CAT" || variant in ABLATION_VARIANTS) ||
        error("Unknown benchmark variant '$variant'.")
    theta = variant == "ρ_hat_rule" ? 0.0 : base.theta
    omega_2 = variant == "radius_update_rule" ? base.omega_1 : base.omega_2
    initial_radius = variant == "initial_radius" ? 1.0 : base.initial_radius
    radius_rule =
        variant == "radius_update_rule" ? "NOT DEFAULT" : base.radius_update_rule_approach
    subproblem_solver =
        variant == "conference_subproblem_solver" ? "OLD" :
        base.trust_region_subproblem_solver
    xi = variant in ("ξ=0.0", "b_k=0.0") ? 0.0 : base.xi
    eval_offset = variant == "b_k=0.0" ? 0.0 : base.eval_offset
    return AlgorithmConfiguration(
        base.beta,
        theta,
        base.omega_1,
        omega_2,
        base.gamma_1,
        base.gamma_2,
        base.gamma_3,
        xi,
        initial_radius,
        base.initial_radius_multiplicative_rule,
        base.seed,
        base.print_level,
        radius_rule,
        eval_offset,
        subproblem_solver,
        base.dense_hessian_threshold,
        base.reuse_sparse_symbolic_factorization,
        base.use_backup_trust_region_subproblem_solver,
        base.handle_hard_case,
        base.delta,
    )
end

function default_utr_algorithm_configuration(
    seed::Int64,
    trust_region_subproblem_solver::String =
        TrustRegionSubproblemSolvers.DIRECT_NEW_SOLVER,
    gamma_1::Float64 = UniversalTrustRegionMethod.DEFAULT_GAMMA_1,
    print_level::Int64 = UniversalTrustRegionMethod.DEFAULT_PRINT_LEVEL,
)::UTRAlgorithmConfiguration
    return UTRAlgorithmConfiguration(
        UniversalTrustRegionMethod.DEFAULT_MAX_INNER_ITERATIONS,
        UniversalTrustRegionMethod.DEFAULT_RHO_0,
        UniversalTrustRegionMethod.DEFAULT_RHO_MIN,
        UniversalTrustRegionMethod.DEFAULT_ETA,
        UniversalTrustRegionMethod.DEFAULT_XI,
        UniversalTrustRegionMethod.DEFAULT_MU_1,
        UniversalTrustRegionMethod.DEFAULT_MU_2,
        gamma_1,
        UniversalTrustRegionMethod.DEFAULT_GAMMA_2,
        UniversalTrustRegionMethod.DEFAULT_GAMMA_3,
        seed,
        print_level,
        UniversalTrustRegionMethod.DEFAULT_DENSE_HESSIAN_THRESHOLD,
        UniversalTrustRegionMethod.DEFAULT_REUSE_SPARSE_SYMBOLIC_FACTORIZATION,
        UniversalTrustRegionMethod.DEFAULT_USE_BACKUP_TRUST_REGION_SUBPROBLEM_SOLVER,
        UniversalTrustRegionMethod.DEFAULT_HANDLE_HARD_CASE,
        trust_region_subproblem_solver,
    )
end

function solver_configuration(
    variant::String,
    seed::Int64;
    print_level::Int64 = CATrustRegionMethod.DEFAULT_PRINT_LEVEL,
    gamma_1::Float64 = TrustRegionSubproblemSolvers.DEFAULT_GAMMA_1,
    trust_region_subproblem_solver::String =
        TrustRegionSubproblemSolvers.DIRECT_NEW_SOLVER,
)::SolverConfiguration
    if variant == CAT_NEURIPS_VARIANT
        gamma_1 == TrustRegionSubproblemSolvers.DEFAULT_GAMMA_1 ||
            error("--gamma-1 is unavailable for $CAT_NEURIPS_VARIANT.")
        trust_region_subproblem_solver == "OLD" ||
            error("$CAT_NEURIPS_VARIANT requires --subproblem-solver OLD.")
        return CATNeurIPSConfiguration(
            CATNeurIPS.DEFAULT_BETA,
            CATNeurIPS.DEFAULT_THETA,
            CATNeurIPS.DEFAULT_OMEGA,
            CATNeurIPS.DEFAULT_INITIAL_RADIUS,
            CATNeurIPS.DEFAULT_DELTA,
            CATNeurIPS.DEFAULT_GAMMA_2,
            print_level,
        )
    end
    if variant == "UTR"
        trust_region_subproblem_solver == TrustRegionSubproblemSolvers.DIRECT_NEW_SOLVER ||
            error("UTR requires --subproblem-solver DIRECT-NEW.")
        return default_utr_algorithm_configuration(
            seed,
            trust_region_subproblem_solver,
            gamma_1,
            print_level,
        )
    end
    trust_region_subproblem_solver == "OLD" &&
        error("--subproblem-solver OLD is available only with --solver $CAT_NEURIPS_VARIANT.")
    return algorithm_configuration(
        variant,
        seed;
        print_level,
        gamma_1,
        trust_region_subproblem_solver,
    )
end

function algorithm_configuration_dict(config::CATNeurIPSConfiguration)::Dict{String,Any}
    return Dict{String,Any}(
        "algorithm" => CAT_NEURIPS_VARIANT,
        "beta" => config.beta,
        "theta" => config.theta,
        "omega" => config.omega,
        "initial_radius" => config.initial_radius,
        "delta" => config.delta,
        "gamma_2" => config.gamma_2,
        "print_level" => config.print_level,
        "trust_region_subproblem_solver" => "OLD",
        "subproblem_implementation" =>
            "TrustRegionSubproblemSolvers.solveTrustRegionSubproblemOldApproach",
        "source_repository_commit" => CATNeurIPS.SOURCE_REPOSITORY_COMMIT,
    )
end

function algorithm_configuration_dict(config::AlgorithmConfiguration)::Dict{String,Any}
    return Dict{String,Any}(
        "beta" => config.beta,
        "theta" => config.theta,
        "omega_1" => config.omega_1,
        "omega_2" => config.omega_2,
        "gamma_1" => config.gamma_1,
        "gamma_2" => config.gamma_2,
        "gamma_3" => config.gamma_3,
        "xi" => config.xi,
        "initial_radius" => config.initial_radius,
        "initial_radius_multiplicative_rule" => config.initial_radius_multiplicative_rule,
        "seed" => config.seed,
        "print_level" => config.print_level,
        "radius_update_rule_approach" => config.radius_update_rule_approach,
        "eval_offset" => config.eval_offset,
        "trust_region_subproblem_solver" => config.trust_region_subproblem_solver,
        "dense_hessian_threshold" => config.dense_hessian_threshold,
        "reuse_sparse_symbolic_factorization" => config.reuse_sparse_symbolic_factorization,
        "use_backup_trust_region_subproblem_solver" =>
            config.use_backup_trust_region_subproblem_solver,
        "handle_hard_case" => config.handle_hard_case,
        "delta" => config.delta,
    )
end

function algorithm_configuration_dict(
    config::UTRAlgorithmConfiguration,
)::Dict{String,Any}
    return Dict{String,Any}(
        "max_inner_iterations" => config.max_inner_iterations,
        "rho_0" => config.rho_0,
        "rho_min" => config.rho_min,
        "eta" => config.eta,
        "xi" => config.xi,
        "mu_1" => config.mu_1,
        "mu_2" => config.mu_2,
        "gamma_1" => config.gamma_1,
        "gamma_2" => config.gamma_2,
        "gamma_3" => config.gamma_3,
        "seed" => config.seed,
        "print_level" => config.print_level,
        "dense_hessian_threshold" => config.dense_hessian_threshold,
        "reuse_sparse_symbolic_factorization" =>
            config.reuse_sparse_symbolic_factorization,
        "use_backup_trust_region_subproblem_solver" =>
            config.use_backup_trust_region_subproblem_solver,
        "handle_hard_case" => config.handle_hard_case,
        "trust_region_subproblem_solver" =>
            config.trust_region_subproblem_solver,
    )
end

function fingerprint_files(paths::Vector{String}, root::String)::String
    normalized_root = abspath(root)
    normalized_paths =
        sort!(unique(abspath.(paths)); by = path -> relpath(path, normalized_root))
    buffer = IOBuffer()
    for path in normalized_paths
        isfile(path) || error("Cannot fingerprint missing file: $path")
        relative_path = relpath(path, normalized_root)
        write(buffer, codeunits(relative_path))
        write(buffer, UInt8(0))
        write(buffer, read(path))
        write(buffer, UInt8(0))
    end
    return bytes2hex(SHA.sha256(take!(buffer)))
end

function append_source_files!(files::Vector{String}, source_root::String)::Nothing
    for (directory, _, names) in walkdir(source_root)
        for name in names
            if endswith(name, ".jl") || endswith(name, ".json")
                push!(files, joinpath(directory, name))
            end
        end
    end
    return nothing
end

function source_fingerprint_files(benchmark_root::String)::Vector{String}
    monorepo_root = normpath(joinpath(benchmark_root, ".."))
    cat_solver_root = joinpath(monorepo_root, "CAT-solver")
    cat_neurips_root = joinpath(monorepo_root, "CAT-NeurIPS")
    utr_solver_root = joinpath(monorepo_root, "UTR")
    shared_code_root = joinpath(monorepo_root, "shared_code")
    subproblem_solver_root = joinpath(monorepo_root, "trust-region-subproblem-solvers")
    files = String[
        joinpath(benchmark_root, "Project.toml"),
        joinpath(benchmark_root, "Manifest.toml"),
        joinpath(benchmark_root, "scripts", "run_benchmark.sh"),
        joinpath(benchmark_root, "scripts", "benchmark_worker.jl"),
        joinpath(benchmark_root, "benchmark", "benchmark_pipeline.jl"),
        joinpath(benchmark_root, "benchmark", "ablation_variants.jl"),
        joinpath(benchmark_root, "benchmark", "benchmark_defaults.jl"),
        joinpath(benchmark_root, "benchmark", "run_profiles.jl"),
        joinpath(benchmark_root, "benchmark", "run-profiles.json"),
        joinpath(benchmark_root, "benchmark", "defaults.json"),
        joinpath(cat_solver_root, "Project.toml"),
        joinpath(cat_solver_root, "Manifest.toml"),
        joinpath(cat_neurips_root, "Project.toml"),
        joinpath(cat_neurips_root, "Manifest.toml"),
        joinpath(utr_solver_root, "Project.toml"),
        joinpath(utr_solver_root, "Manifest.toml"),
        joinpath(shared_code_root, "Project.toml"),
        joinpath(subproblem_solver_root, "Project.toml"),
        joinpath(subproblem_solver_root, "Manifest.toml"),
    ]
    append_source_files!(files, joinpath(cat_solver_root, "src"))
    append_source_files!(files, joinpath(cat_neurips_root, "src"))
    append_source_files!(files, joinpath(utr_solver_root, "src"))
    append_source_files!(files, joinpath(shared_code_root, "src"))
    append_source_files!(files, joinpath(subproblem_solver_root, "src"))
    return files
end

function source_fingerprint(benchmark_root::String)::String
    monorepo_root = normpath(joinpath(benchmark_root, ".."))
    return fingerprint_files(source_fingerprint_files(benchmark_root), monorepo_root)
end

function build_experiment(
    benchmark_kind::String,
    problem_set::String,
    results_folder::String,
    seed::Int64;
    solver::Union{Nothing,String} = nothing,
    print_level::Int64 = CATrustRegionMethod.DEFAULT_PRINT_LEVEL,
    num_threads::Int64 = Threads.nthreads(:default),
    gamma_1::Float64 = TrustRegionSubproblemSolvers.DEFAULT_GAMMA_1,
    subproblem_solver::String = TrustRegionSubproblemSolvers.DIRECT_NEW_SOLVER,
    benchmark_root::String = normpath(joinpath(@__DIR__, "..")),
    profiles_path::String = BenchmarkRunProfiles.DEFAULT_RUN_PROFILES_FILE,
    manually_skipped_problems::Vector{String} = String[],
    fingerprint::Union{Nothing,String} = nothing,
    ablation_variant::Union{Nothing,String} = nothing,
)::Experiment
    benchmark_kind in ("benchmark", "ablation") ||
        error("Benchmark kind must be 'benchmark' or 'ablation'.")
    seed >= 0 || error("Seed must be nonnegative.")
    print_level >= -1 || error("--print-level must be -1 or nonnegative.")
    num_threads > 0 || error("--threads must be positive.")
    0.0 < gamma_1 < 1.0 ||
        throw(ArgumentError("--gamma-1 must lie in (0, 1)."))
    subproblem_solver in ("OLD", TrustRegionSubproblemSolvers.DIRECT_NEW_SOLVER) ||
        error("--subproblem-solver must be OLD or DIRECT-NEW.")
    variants = if benchmark_kind == "benchmark"
        isnothing(ablation_variant) ||
            error("--ablation-variant is available only with --ablation.")
        solver in STANDARD_VARIANTS ||
            error("Standard benchmarks require --solver CAT, $CAT_NEURIPS_VARIANT, or UTR.")
        String[solver]
    else
        isnothing(solver) ||
            error("--solver is unavailable for the CAT ablation study.")
        isnothing(ablation_variant) &&
            error("CAT ablation runs require --ablation-variant.")
        [internal_ablation_variant(ablation_variant)]
    end
    if variants == [CAT_NEURIPS_VARIANT]
        subproblem_solver == "OLD" ||
            error("$CAT_NEURIPS_VARIANT requires --subproblem-solver OLD.")
    elseif subproblem_solver == "OLD"
        error("--subproblem-solver OLD is available only with --solver $CAT_NEURIPS_VARIANT.")
    end
    if variants == ["UTR"]
        subproblem_solver == TrustRegionSubproblemSolvers.DIRECT_NEW_SOLVER ||
            error("UTR requires --subproblem-solver DIRECT-NEW.")
    end
    for variant in variants
        config = solver_configuration(
            variant,
            seed;
            print_level,
            gamma_1,
            trust_region_subproblem_solver = subproblem_solver,
        )
        config isa CATNeurIPSConfiguration && continue
        try
            make_algorithm_parameters(config, -1)
        catch error
            error isa AssertionError || rethrow()
            throw(
                ArgumentError(
                    "--gamma-1=$gamma_1 is incompatible with $variant and " *
                    "--subproblem-solver $subproblem_solver.",
                ),
            )
        end
    end
    selection = BenchmarkRunProfiles.resolve_problem_selection(
        problem_set,
        profiles_path,
        manually_skipped_problems,
    )
    return Experiment(
        benchmark_kind,
        problem_set,
        abspath(results_folder),
        selection.batches,
        selection.problems,
        selection.skipped_problems,
        variants,
        seed,
        print_level,
        num_threads,
        selection.max_iterations,
        selection.max_time_seconds,
        CATrustRegionMethod.DEFAULT_GRADIENT_TERMINATION_TOLERANCE,
        CATrustRegionMethod.DEFAULT_STEP_SIZE_LIMIT,
        CATrustRegionMethod.DEFAULT_MINIMUM_OBJECTIVE_FUNCTION,
        CATrustRegionMethod.DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS,
        gamma_1,
        subproblem_solver,
        something(fingerprint, source_fingerprint(benchmark_root)),
    )
end

const THREAD_ENVIRONMENT_VARIABLES = (
    "CAT_BENCHMARK_STARTUP_FILE",
    "JULIA_NUM_THREADS",
    "JULIA_NUM_GC_THREADS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_CANCELLATION",
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_DEFAULT_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "MKL_DYNAMIC",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "BLIS_NT",
    "BLIS_JC_NT",
    "BLIS_PC_NT",
    "BLIS_IC_NT",
    "BLIS_JR_NT",
    "BLIS_IR_NT",
    "BLIS_THREAD_IMPL",
    "BLIS_TI",
)
const UNSET_THREAD_ENVIRONMENT_VARIABLES = (
    "BLIS_JC_NT",
    "BLIS_PC_NT",
    "BLIS_IC_NT",
    "BLIS_JR_NT",
    "BLIS_IR_NT",
    "BLIS_THREAD_IMPL",
    "BLIS_TI",
)

function expected_thread_environment(num_threads::Int64)::Dict{String,String}
    num_threads > 0 || error("The benchmark thread count must be positive.")
    count = string(num_threads)
    return Dict(
        "CAT_BENCHMARK_STARTUP_FILE" => "no",
        "JULIA_NUM_THREADS" => "$count,0",
        "JULIA_NUM_GC_THREADS" => "$count,0",
        "OMP_NUM_THREADS" => count,
        "OMP_THREAD_LIMIT" => count,
        "OMP_DYNAMIC" => "FALSE",
        "OPENBLAS_NUM_THREADS" => count,
        "OPENBLAS_DEFAULT_NUM_THREADS" => count,
        "GOTO_NUM_THREADS" => count,
        "MKL_NUM_THREADS" => count,
        "MKL_DOMAIN_NUM_THREADS" => "MKL_DOMAIN_ALL=$count",
        "MKL_DYNAMIC" => "FALSE",
        "VECLIB_MAXIMUM_THREADS" => count,
        "BLIS_NUM_THREADS" => count,
        "BLIS_NT" => count,
    )
end

function enforce_thread_configuration(num_threads::Int64)::Nothing
    num_threads > 0 || error("The benchmark thread count must be positive.")
    Threads.nthreads(:default) == num_threads || error(
        "Expected $num_threads Julia default-pool threads, found " *
        "$(Threads.nthreads(:default)).",
    )
    Threads.nthreads(:interactive) == 0 || error(
        "Expected no Julia interactive-pool threads, found " *
        "$(Threads.nthreads(:interactive)).",
    )
    Threads.ngcthreads() == num_threads || error(
        "Expected $num_threads Julia GC threads in total, found " *
        "$(Threads.ngcthreads()).",
    )

    expected_environment = expected_thread_environment(num_threads)
    mismatches = String[
        "$name=$(repr(get(ENV, name, nothing))) (expected $(repr(value)))" for
        (name, value) in sort!(collect(expected_environment); by = first) if
        get(ENV, name, nothing) != value
    ]
    isempty(mismatches) || error(
        "Benchmark thread environment is inconsistent: " * join(mismatches, ", "),
    )
    unexpected_environment = String[
        "$name=$(repr(ENV[name]))" for name in UNSET_THREAD_ENVIRONMENT_VARIABLES if
        haskey(ENV, name)
    ]
    isempty(unexpected_environment) || error(
        "Benchmark thread environment contains overriding BLIS settings: " *
        join(unexpected_environment, ", "),
    )

    LinearAlgebra.BLAS.set_num_threads(num_threads)
    actual_blas_threads = LinearAlgebra.BLAS.get_num_threads()
    if actual_blas_threads != num_threads
        configuration =
            lowercase(sprint(show, LinearAlgebra.BLAS.get_config()))
        if num_threads > 1 &&
           (occursin("accelerate", configuration) ||
            occursin("veclib", configuration))
            error(
                "Apple Accelerate cannot enforce an exact BLAS thread count " *
                "greater than one. Use --threads 1 or select a BLAS backend " *
                "with exact runtime thread-count control.",
            )
        end
        error(
            "Expected $num_threads BLAS threads after configuration, found " *
            "$actual_blas_threads.",
        )
    end
    return nothing
end

function package_version_string(package_module::Module)::Union{Nothing,String}
    version = Base.pkgversion(package_module)
    return version === nothing ? nothing : string(version)
end

function sha256_file(path::String)::Union{Nothing,String}
    isfile(path) || return nothing
    return bytes2hex(SHA.sha256(read(path)))
end

function mastsif_provenance()::Dict{String,Any}
    directory = get(ENV, "MASTSIF", "")
    normalized_directory = isempty(directory) ? nothing : normpath(directory)
    artifact_name =
        normalized_directory === nothing ? nothing : basename(normalized_directory)
    artifact_hash =
        artifact_name !== nothing && occursin(r"^[0-9a-fA-F]{40}$", artifact_name) ?
        lowercase(artifact_name) : nothing
    classification_database =
        normalized_directory === nothing ? "" :
        joinpath(normalized_directory, "CLASSF.DB")
    sif_problem_count =
        normalized_directory !== nothing && isdir(normalized_directory) ?
        count(
            name -> endswith(uppercase(name), ".SIF"),
            readdir(normalized_directory),
        ) : nothing
    return Dict{String,Any}(
        "directory" => normalized_directory,
        "artifact_hash" => artifact_hash,
        "classification_database_sha256" => sha256_file(classification_database),
        "sif_problem_count" => sif_problem_count,
    )
end

function environment_provenance()::Dict{String,Any}
    cpu_information = Sys.cpu_info()
    return Dict{String,Any}(
        "system" => Dict{String,Any}(
            "kernel" => string(Sys.KERNEL),
            "architecture" => string(Sys.ARCH),
            "machine" => Sys.MACHINE,
            "word_size" => Sys.WORD_SIZE,
            "cpu_name" => Sys.CPU_NAME,
            "cpu_model" => isempty(cpu_information) ? nothing : first(cpu_information).model,
            "logical_cpu_threads" => Sys.CPU_THREADS,
            "total_memory_bytes" => Sys.total_memory(),
        ),
        "threads" => Dict{String,Any}(
            "julia_threads" => Threads.nthreads(),
            "julia_default_pool_threads" => Threads.nthreads(:default),
            "julia_interactive_pool_threads" => Threads.nthreads(:interactive),
            "julia_gc_threads_total" => Threads.ngcthreads(),
            "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
            "startup_file" => get(ENV, "CAT_BENCHMARK_STARTUP_FILE", nothing),
            "environment" => Dict{String,Any}(
                name => get(ENV, name, nothing) for name in THREAD_ENVIRONMENT_VARIABLES
            ),
        ),
        "blas" => Dict{String,Any}(
            "vendor" => string(LinearAlgebra.BLAS.vendor()),
            "configuration" => sprint(show, LinearAlgebra.BLAS.get_config()),
        ),
        "cutest" => Dict{String,Any}(
            "cutest_version" => package_version_string(CUTEst),
            "cutest_jll_version" => package_version_string(CUTEst.CUTEst_jll),
            "sifdecode_jll_version" => package_version_string(CUTEst.SIFDecode_jll),
            "mastsif" => mastsif_provenance(),
        ),
    )
end

function settings_dict(experiment::Experiment)::Dict{String,Any}
    configurations = Dict{String,Any}(
        variant => algorithm_configuration_dict(
            solver_configuration(
                variant,
                experiment.seed;
                print_level = experiment.print_level,
                gamma_1 = experiment.gamma_1,
                trust_region_subproblem_solver =
                    experiment.trust_region_subproblem_solver,
            ),
        ) for variant in experiment.variants
    )
    return Dict{String,Any}(
        "schema_version" => SETTINGS_SCHEMA_VERSION,
        "benchmark_kind" => experiment.benchmark_kind,
        "problem_set" => experiment.problem_set,
        "batches" => experiment.batches,
        "problems" => experiment.problems,
        "skipped_problems" => experiment.skipped_problems,
        "variants" => experiment.variants,
        "ablation_variant" =>
            experiment.benchmark_kind == "ablation" ?
            ablation_variant_slug(only(experiment.variants)) : nothing,
        "solver" =>
            experiment.benchmark_kind == "benchmark" ? only(experiment.variants) : nothing,
        "seed" => experiment.seed,
        "print_level" => experiment.print_level,
        "num_threads" => experiment.num_threads,
        "max_iterations" => experiment.max_iterations,
        "max_time_seconds" => experiment.max_time_seconds,
        "gradient_tolerance" => experiment.gradient_tolerance,
        "step_size_limit" => experiment.step_size_limit,
        "minimum_objective_function" => experiment.minimum_objective_function,
        "iterative_refinement_max_iterations" =>
            experiment.iterative_refinement_max_iterations,
        "gamma_1" => experiment.gamma_1,
        "trust_region_subproblem_solver" => experiment.trust_region_subproblem_solver,
        "algorithm_configurations" => configurations,
        "timing" => Dict{String,Any}(
            "total_execution_time_scope" =>
                "solver_time_including_all_model_evaluations",
            "optimality_validation_scope" =>
                "fresh_CUTEst_gradient_outside_all_solver_timing_and_resource_metrics",
        ),
        "julia_version" => string(VERSION),
        "environment" => environment_provenance(),
        "source_fingerprint" => experiment.source_fingerprint,
    )
end

function atomic_write_json(path::String, data)::Nothing
    mkpath(dirname(path))
    temporary_path = "$path.tmp.$(getpid())"
    try
        open(temporary_path, "w") do io
            JSON.print(io, data, 2)
            println(io)
        end
        mv(temporary_path, path; force = true)
    finally
        isfile(temporary_path) && rm(temporary_path; force = true)
    end
    return nothing
end

function atomic_write_text(path::String, contents::String)::Nothing
    mkpath(dirname(path))
    temporary_path = "$path.tmp.$(getpid())"
    try
        open(temporary_path, "w") do io
            write(io, contents)
        end
        mv(temporary_path, path; force = true)
    finally
        isfile(temporary_path) && rm(temporary_path; force = true)
    end
    return nothing
end

function atomic_write_csv(path::String, data::DataFrame)::Nothing
    mkpath(dirname(path))
    temporary_path = "$path.tmp.$(getpid())"
    try
        CSV.write(temporary_path, data)
        mv(temporary_path, path; force = true)
    finally
        isfile(temporary_path) && rm(temporary_path; force = true)
    end
    return nothing
end

function prepare_results_directory(experiment::Experiment)::Nothing
    settings_path = joinpath(experiment.results_folder, "run_settings.json")
    requested = settings_dict(experiment)
    if isfile(settings_path)
        existing = try
            JSON.parsefile(settings_path)
        catch error
            throw(
                ErrorException(
                    "Cannot read current run settings at $settings_path: $(sprint(showerror, error))",
                ),
            )
        end
        compatible_reentry_settings(existing, requested)
        return nothing
    end
    if ispath(experiment.results_folder) && !isdir(experiment.results_folder)
        error("Results path exists but is not a directory: $(experiment.results_folder)")
    end
    if isdir(experiment.results_folder) && !isempty(readdir(experiment.results_folder))
        error(
            "Results directory is not a current benchmark run: $(experiment.results_folder)",
        )
    end
    mkpath(experiment.results_folder)
    atomic_write_json(settings_path, requested)
    return nothing
end

function raw_result_path(experiment::Experiment, variant::String, problem::String)::String
    return joinpath(experiment.results_folder, "raw", variant, "$problem.json")
end

function problem_log_path(experiment::Experiment, variant::String, problem::String)::String
    return joinpath(experiment.results_folder, "logs", variant, "$problem.log")
end

function diagnostic_directory(
    experiment::Experiment,
    variant::String,
    problem::String,
)::String
    return joinpath(experiment.results_folder, "diagnostics", variant, problem)
end

function result_dict(result::BenchmarkResult)::Dict{String,Any}
    return Dict{String,Any}(
        "schema_version" => result.schema_version,
        "benchmark_kind" => result.benchmark_kind,
        "problem_set" => result.problem_set,
        "variant" => result.variant,
        "problem" => result.problem,
        "seed" => result.seed,
        "solver_status" => result.solver_status,
        "status" => result.status,
        "total_execution_time" => result.total_execution_time,
        "function_value" => result.function_value,
        "gradient_value" => result.gradient_value,
        "terminal_objective_value" => result.terminal_objective_value,
        "terminal_gradient_norm" => result.terminal_gradient_norm,
        "best_gradient_objective_value" => result.best_gradient_objective_value,
        "best_gradient_norm" => result.best_gradient_norm,
        "total_function_evaluation" => result.total_function_evaluation,
        "total_gradient_evaluation" => result.total_gradient_evaluation,
        "total_hessian_evaluation" => result.total_hessian_evaluation,
        "total_factorization_evaluation" => result.total_factorization_evaluation,
        "total_subproblem_iterations" => result.total_subproblem_iterations,
        "total_hessian_vector_products" => result.total_hessian_vector_products,
        "total_iterations" => result.total_iterations,
        "wall_time_seconds" => result.wall_time_seconds,
        "allocated_bytes" => result.allocated_bytes,
        "gc_time_seconds" => result.gc_time_seconds,
        "max_rss_bytes" => result.max_rss_bytes,
        "optimality_validation_performed" =>
            result.optimality_validation_performed,
        "optimality_validation_passed" => result.optimality_validation_passed,
        "optimality_validation_gradient_norm" =>
            result.optimality_validation_gradient_norm,
        "optimality_validation_time_seconds" =>
            result.optimality_validation_time_seconds,
        "solver_specific" => result.solver_specific,
        "diagnostic_directory" => result.diagnostic_directory,
        "error_type" => result.error_type,
        "error_message" => result.error_message,
        "completed_at" => result.completed_at,
    )
end


const REQUIRED_RESULT_FIELDS = (
    "solver_status",
    "status",
    "total_execution_time",
    "function_value",
    "gradient_value",
    "terminal_objective_value",
    "terminal_gradient_norm",
    "best_gradient_objective_value",
    "best_gradient_norm",
    "total_function_evaluation",
    "total_gradient_evaluation",
    "total_hessian_evaluation",
    "total_factorization_evaluation",
    "total_subproblem_iterations",
    "total_hessian_vector_products",
    "total_iterations",
    "wall_time_seconds",
    "allocated_bytes",
    "gc_time_seconds",
    "max_rss_bytes",
    "optimality_validation_performed",
    "optimality_validation_passed",
    "optimality_validation_gradient_norm",
    "optimality_validation_time_seconds",
    "solver_specific",
    "diagnostic_directory",
    "error_type",
    "error_message",
    "completed_at",
)

function load_result(
    experiment::Experiment,
    variant::String,
    problem::String,
)::Union{Nothing,AbstractDict}
    path = raw_result_path(experiment, variant, problem)
    isfile(path) || return nothing
    data = try
        JSON.parsefile(path)
    catch error
        throw(
            ErrorException(
                "Cannot read raw benchmark result $path: $(sprint(showerror, error))",
            ),
        )
    end
    get(data, "schema_version", nothing) == RESULT_SCHEMA_VERSION ||
        error("Raw benchmark result has an unsupported schema: $path")
    expected_identity = (
        "benchmark_kind" => experiment.benchmark_kind,
        "problem_set" => experiment.problem_set,
        "variant" => variant,
        "problem" => problem,
        "seed" => experiment.seed,
    )
    for (name, expected) in expected_identity
        get(data, name, nothing) == expected ||
            error("Raw benchmark result has the wrong $name: $path")
    end
    for name in REQUIRED_RESULT_FIELDS
        haskey(data, name) || error("Raw benchmark result is missing '$name': $path")
    end
    data["status"] isa AbstractString || error("Raw benchmark status is invalid: $path")
    data["solver_status"] isa AbstractString ||
        error("Raw benchmark solver status is invalid: $path")
    data["optimality_validation_performed"] isa Bool ||
        error("Raw benchmark validation flag is invalid: $path")
    data["solver_specific"] isa AbstractDict ||
        error("Raw benchmark solver-specific report is invalid: $path")
    for name in (
        "function_value",
        "gradient_value",
        "terminal_objective_value",
        "terminal_gradient_norm",
        "best_gradient_objective_value",
        "best_gradient_norm",
        "optimality_validation_gradient_norm",
    )
        value = data[name]
        (value === nothing || (value isa Real && isfinite(value))) ||
            error("Raw benchmark field '$name' is invalid: $path")
    end
    for name in (
        "total_execution_time",
        "total_function_evaluation",
        "total_gradient_evaluation",
        "total_hessian_evaluation",
        "total_factorization_evaluation",
        "total_subproblem_iterations",
        "total_hessian_vector_products",
        "total_iterations",
        "wall_time_seconds",
        "allocated_bytes",
        "gc_time_seconds",
        "max_rss_bytes",
        "optimality_validation_time_seconds",
    )
        data[name] isa Real || error("Raw benchmark field '$name' is invalid: $path")
    end
    return data
end

function pending_items(experiment::Experiment)::Vector{Tuple{String,String}}
    pending = Tuple{String,String}[]
    for problem in experiment.problems, variant in experiment.variants
        load_result(experiment, variant, problem) === nothing &&
            push!(pending, (variant, problem))
    end
    return pending
end

function pending_runs(experiment::Experiment)::Vector{Tuple{String,Int64}}
    pending = Set(pending_items(experiment))
    return Tuple{String,Int64}[
        (variant, index) for variant in experiment.variants for
        (index, batch) in enumerate(experiment.batches) if
        any((variant, problem) in pending for problem in batch)
    ]
end

function pending_problem_runs(
    experiment::Experiment,
)::Vector{Tuple{String,Int64,String}}
    pending = Set(pending_items(experiment))
    return Tuple{String,Int64,String}[
        (variant, index, problem) for variant in experiment.variants for
        (index, batch) in enumerate(experiment.batches) for problem in batch if
        (variant, problem) in pending
    ]
end

function write_pending_runs(experiment::Experiment)::Vector{Tuple{String,Int64}}
    pending = pending_runs(experiment)
    buffer = IOBuffer()
    println(buffer, "variant\tbatch_index")
    for (variant, index) in pending
        println(buffer, "$variant\t$index")
    end
    atomic_write_text(
        joinpath(experiment.results_folder, "pending_runs.tsv"),
        String(take!(buffer)),
    )
    problem_buffer = IOBuffer()
    println(problem_buffer, "variant\tbatch_index\tproblem")
    for (variant, index, problem) in pending_problem_runs(experiment)
        println(problem_buffer, "$variant\t$index\t$problem")
    end
    atomic_write_text(
        joinpath(experiment.results_folder, "pending_problems.tsv"),
        String(take!(problem_buffer)),
    )
    return pending
end

const REENTRY_IGNORED_SYSTEM_KEYS = Set([
    "logical_cpu_threads",
    "total_memory_bytes",
])

function compatible_reentry_settings(
    existing::AbstractDict,
    requested::AbstractDict,
)::Nothing
    existing_keys = Set(String.(collect(keys(existing))))
    requested_keys = Set(String.(collect(keys(requested))))
    existing_keys == requested_keys ||
        error("Existing and requested run settings have different fields.")
    differences = String[
        key for key in existing_keys if
        key != "environment" && existing[key] != requested[key]
    ]

    existing_environment = existing["environment"]
    requested_environment = requested["environment"]
    existing_environment_keys = Set(String.(collect(keys(existing_environment))))
    requested_environment_keys = Set(String.(collect(keys(requested_environment))))
    existing_environment_keys == requested_environment_keys ||
        error("Existing and requested environments have different fields.")
    for key in existing_environment_keys
        if key != "system"
            existing_environment[key] == requested_environment[key] ||
                push!(differences, "environment.$key")
            continue
        end
        existing_system = existing_environment["system"]
        requested_system = requested_environment["system"]
        existing_system_keys = Set(String.(collect(keys(existing_system))))
        requested_system_keys = Set(String.(collect(keys(requested_system))))
        existing_system_keys == requested_system_keys ||
            error("Existing and requested system provenance have different fields.")
        append!(
            differences,
            [
                "environment.system.$system_key" for system_key in existing_system_keys if
                system_key ∉ REENTRY_IGNORED_SYSTEM_KEYS &&
                existing_system[system_key] != requested_system[system_key]
            ],
        )
    end
    sort!(differences)
    isempty(differences) || error(
        "Results directory is not reentry-compatible; changed fields: " *
        join(differences, ", "),
    )
    return nothing
end


function make_algorithm_parameters(
    config::AlgorithmConfiguration,
    print_level::Int64,
)::CATrustRegionMethod.AlgorithmicParameters
    return CATrustRegionMethod.AlgorithmicParameters(
        config.beta,
        config.theta,
        config.omega_1,
        config.omega_2,
        config.gamma_1,
        config.gamma_2,
        config.gamma_3,
        config.xi,
        config.initial_radius,
        config.initial_radius_multiplicative_rule,
        config.seed,
        print_level,
        config.radius_update_rule_approach,
        config.eval_offset,
        config.trust_region_subproblem_solver,
        config.dense_hessian_threshold,
        config.reuse_sparse_symbolic_factorization,
        config.use_backup_trust_region_subproblem_solver,
        config.handle_hard_case,
    )
end

function make_algorithm_parameters(
    config::UTRAlgorithmConfiguration,
    print_level::Int64,
)::UniversalTrustRegionMethod.AlgorithmicParameters
    return UniversalTrustRegionMethod.AlgorithmicParameters(
        config.rho_0,
        config.rho_min,
        config.eta,
        config.xi,
        config.mu_1,
        config.mu_2,
        config.gamma_1,
        config.gamma_2,
        config.gamma_3,
        config.seed,
        print_level,
        config.dense_hessian_threshold,
        config.reuse_sparse_symbolic_factorization,
        config.use_backup_trust_region_subproblem_solver,
        config.handle_hard_case,
        config.trust_region_subproblem_solver,
    )
end

function make_termination_criteria(
    experiment::Experiment,
    ::AlgorithmConfiguration,
    max_iterations::Int64 = experiment.max_iterations,
    max_time_seconds::Float64 = experiment.max_time_seconds,
)::CATrustRegionMethod.TerminationCriteria
    return CATrustRegionMethod.TerminationCriteria(
        max_iterations,
        experiment.gradient_tolerance,
        max_time_seconds,
        experiment.step_size_limit,
        experiment.minimum_objective_function,
        experiment.iterative_refinement_max_iterations,
    )
end

function make_termination_criteria(
    experiment::Experiment,
    config::UTRAlgorithmConfiguration,
    max_iterations::Int64 = experiment.max_iterations,
    max_time_seconds::Float64 = experiment.max_time_seconds,
)::UniversalTrustRegionMethod.TerminationCriteria
    return UniversalTrustRegionMethod.TerminationCriteria(
        max_iterations,
        config.max_inner_iterations,
        experiment.gradient_tolerance,
        max_time_seconds,
        experiment.step_size_limit,
        experiment.minimum_objective_function,
        experiment.iterative_refinement_max_iterations,
    )
end

function finalize_model(nlp::Any)::Nothing
    nlp === nothing && return nothing
    try
        finalize(nlp)
    catch error
        @warn "CUTEst model finalization failed" exception = (error, catch_backtrace())
    end
    return nothing
end

function status_string(status::Any)::String
    return canonical_status_string(status; memory_limit = "OUT_OF_MEMORY")
end


function finite_or_nothing(value::Real)::Union{Nothing,Float64}
    converted = Float64(value)
    return isfinite(converted) ? converted : nothing
end

finite_or_nothing(::Nothing)::Nothing = nothing

function failed_outcome(
    status::String,
    experiment::Experiment,
    error::Exception,
)::SolverOutcome
    failure_count = 2 * experiment.max_iterations
    return SolverOutcome(
        status,
        2.0 * experiment.max_time_seconds,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        failure_count,
        failure_count,
        failure_count,
        failure_count,
        failure_count,
        failure_count,
        0,
        string(typeof(error)),
        sprint(showerror, error),
        nothing,
        Dict{String,Any}(),
    )
end

function optimize_model(
    nlp,
    termination::CATrustRegionMethod.TerminationCriteria,
    parameters::CATrustRegionMethod.AlgorithmicParameters,
    config::AlgorithmConfiguration,
)
    return CATrustRegionMethod.optimize(
        nlp,
        termination,
        parameters,
        nlp.meta.x0,
        config.delta,
    )
end

function optimize_model(
    nlp,
    termination::UniversalTrustRegionMethod.TerminationCriteria,
    parameters::UniversalTrustRegionMethod.AlgorithmicParameters,
    ::UTRAlgorithmConfiguration,
)
    return UniversalTrustRegionMethod.UTR_solve(nlp, termination, parameters)
end


function run_model(
    experiment::Experiment,
    variant::String,
    problem::String,
    config::CATNeurIPSConfiguration,
)::SolverOutcome
    variant == CAT_NEURIPS_VARIANT ||
        error("CAT-NeurIPS solver configuration mismatch.")
    details_directory = diagnostic_directory(experiment, variant, problem)
    TrustRegionSubproblemSolvers.setSubproblemFailureDetailsDirectory(details_directory)
    nlp = nothing
    try
        nlp = CUTEstModel{Float64}(problem)
        parameters = CATNeurIPS.AlgorithmicParameters(
            config.beta,
            config.theta,
            config.omega,
            config.initial_radius,
            config.delta,
            config.gamma_2,
        )
        termination = CATNeurIPS.TerminationCriteria(
            experiment.max_iterations,
            experiment.gradient_tolerance,
            experiment.max_time_seconds,
        )
        result = CATNeurIPS.solve(nlp, termination, parameters)
        objective_value = finite_or_nothing(result.objective)
        gradient_norm = finite_or_nothing(result.gradient_norm)
        return SolverOutcome(
            result.status,
            result.execution_time,
            objective_value,
            gradient_norm,
            objective_value,
            gradient_norm,
            nothing,
            nothing,
            Int64(CUTEst.neval_obj(nlp)),
            Int64(CUTEst.neval_grad(nlp)),
            Int64(CUTEst.neval_hess(nlp)),
            0,
            0,
            Int64(CUTEst.neval_hprod(nlp)),
            result.iterations,
            nothing,
            nothing,
            result.solution,
            Dict{String,Any}(
                "subproblem_solves" => result.subproblem_solves,
                "subproblem_solver" => "OLD",
                "subproblem_accounting" => "unavailable",
            ),
        )
    finally
        finalize_model(nlp)
        TrustRegionSubproblemSolvers.setSubproblemFailureDetailsDirectory(nothing)
    end
end

function run_model(
    experiment::Experiment,
    variant::String,
    problem::String,
    config::SolverConfiguration,
)::SolverOutcome
    details_directory = diagnostic_directory(experiment, variant, problem)
    TrustRegionSubproblemSolvers.setSubproblemFailureDetailsDirectory(details_directory)
    nlp = nothing
    try
        nlp = CUTEstModel{Float64}(problem)
        termination = make_termination_criteria(experiment, config)
        parameters = make_algorithm_parameters(config, config.print_level)
        result = optimize_model(nlp, termination, parameters, config)
        solution, status, iteration_stats, counter, iterations, execution_time = result
        terminal_objective_value::Union{Nothing,Float64} = nothing
        terminal_gradient_norm::Union{Nothing,Float64} = nothing
        best_gradient_objective_value::Union{Nothing,Float64} = nothing
        best_gradient_norm::Union{Nothing,Float64} = nothing
        # Each history row stores the current accepted iterate's objective and gradient
        # norm, plus the objective and gradient norm at the best-gradient point seen so far
        if nrow(iteration_stats) > 0
            terminal_objective_value =
                finite_or_nothing(iteration_stats[end, "fval"])
            terminal_gradient_norm =
                finite_or_nothing(iteration_stats[end, "gradnorm"])
            best_gradient_objective_value =
                finite_or_nothing(iteration_stats[end, "min_gradnorm_fval"])
            best_gradient_norm =
                finite_or_nothing(iteration_stats[end, "min_gradnorm"])
        end
        claimed_solution = if status_string(status) == "OPTIMAL" && length(result) >= 7
            Vector{Float64}(result[7])
        else
            Vector{Float64}(solution)
        end
        return SolverOutcome(
            status_string(status),
            Float64(execution_time),
            best_gradient_objective_value,
            best_gradient_norm,
            terminal_objective_value,
            terminal_gradient_norm,
            best_gradient_objective_value,
            best_gradient_norm,
            Int64(counter.total_function_evaluation),
            Int64(counter.total_gradient_evaluation),
            Int64(counter.total_hessian_evaluation),
            Int64(counter.total_number_factorizations),
            Int64(counter.total_number_subproblem_iterations),
            Int64(counter.total_number_hessian_vector_products),
            Int64(iterations),
            nothing,
            nothing,
            claimed_solution,
            Dict{String,Any}(),
        )
    finally
        finalize_model(nlp)
        TrustRegionSubproblemSolvers.setSubproblemFailureDetailsDirectory(nothing)
    end
end


function append_to_log(f::Function, path::String, heading::String)
    mkpath(dirname(path))
    return open(path, "a") do io
        println(io, "\n[$(iso_timestamp())] $heading")
        flush(io)
        redirect_stdout(io) do
            redirect_stderr(io) do
                return f()
            end
        end
    end
end

function execute_warmup(f::Function, log_path::String, heading::String)::Nothing
    append_to_log(log_path, heading) do
        GC.gc()
        try
            f()
        catch error
            showerror(stderr, error, catch_backtrace())
            println(stderr)
            error isa OutOfMemoryError && GC.gc()
            rethrow()
        end
    end
    GC.gc()
    return nothing
end

function run_warmup(
    experiment::Experiment,
    variant::String,
    config::CATNeurIPSConfiguration,
)::Nothing
    log_path = problem_log_path(experiment, variant, "warmup_$WARMUP_PROBLEM")
    execute_warmup(log_path, "Warming up $variant/$WARMUP_PROBLEM") do
        nlp = nothing
        try
            nlp = CUTEstModel{Float64}(WARMUP_PROBLEM)
            parameters = CATNeurIPS.AlgorithmicParameters(
                config.beta,
                config.theta,
                config.omega,
                config.initial_radius,
                config.delta,
                config.gamma_2,
            )
            termination = CATNeurIPS.TerminationCriteria(
                WARMUP_MAX_ITERATIONS,
                experiment.gradient_tolerance,
                WARMUP_MAX_TIME_SECONDS,
            )
            CATNeurIPS.solve(nlp, termination, parameters)
        finally
            finalize_model(nlp)
        end
    end
    return nothing
end

function run_warmup(
    experiment::Experiment,
    variant::String,
    config::SolverConfiguration,
)::Nothing
    log_path = problem_log_path(experiment, variant, "warmup_$WARMUP_PROBLEM")
    execute_warmup(log_path, "Warming up $variant/$WARMUP_PROBLEM") do
        details_directory =
            diagnostic_directory(experiment, variant, "warmup_$WARMUP_PROBLEM")
        TrustRegionSubproblemSolvers.setSubproblemFailureDetailsDirectory(
            details_directory,
        )
        nlp = nothing
        try
            nlp = CUTEstModel{Float64}(WARMUP_PROBLEM)
            termination = make_termination_criteria(
                experiment,
                config,
                WARMUP_MAX_ITERATIONS,
                WARMUP_MAX_TIME_SECONDS,
            )
            parameters = make_algorithm_parameters(config, -1)
            optimize_model(nlp, termination, parameters, config)
        finally
            finalize_model(nlp)
            TrustRegionSubproblemSolvers.setSubproblemFailureDetailsDirectory(nothing)
        end
    end
    return nothing
end


function validate_optimality_point(
    problem::String,
    solution::Union{Nothing,Vector{Float64}},
    gradient_tolerance::Float64;
    model_factory::Function = name -> CUTEstModel{Float64}(name),
    clock_ns::Function = time_ns,
)::OptimalityValidation
    start_time_ns = UInt64(clock_ns())
    model = nothing
    gradient_norm::Union{Nothing,Float64} = nothing
    validation_error::Union{Nothing,Exception} = nothing
    passed::Union{Nothing,Bool} = nothing
    try
        isnothing(solution) && throw(
            OptimalityValidationError(
                "The solver claimed OPTIMAL without returning a point to validate.",
            ),
        )
        all(isfinite, solution) || throw(
            OptimalityValidationError(
                "The solver's claimed OPTIMAL point contains nonfinite values.",
            ),
        )
        model = model_factory(problem)
        dimension = length(model.meta.x0)
        length(solution) == dimension || throw(
            OptimalityValidationError(
                "The solver returned a point of length $(length(solution)) for a " *
                "CUTEst problem of dimension $dimension.",
            ),
        )
        gradient = zeros(Float64, dimension)
        NLPModels.grad!(model, solution, gradient)
        all(isfinite, gradient) || throw(
            OptimalityValidationError(
                "The fresh CUTEst gradient at the claimed OPTIMAL point is nonfinite.",
            ),
        )
        gradient_norm = LinearAlgebra.norm(gradient, 2)
        passed = gradient_norm <= gradient_tolerance
        passed || throw(
            OptimalityValidationError(
                "The solver claimed OPTIMAL, but a fresh CUTEst evaluation gave " *
                "norm(gradient, 2)=$gradient_norm > $gradient_tolerance.",
            ),
        )
    catch error
        error isa InterruptException && rethrow()
        validation_error = error isa OptimalityValidationError ? error :
                           OptimalityValidationError(
            "Fresh CUTEst optimality validation failed: $(sprint(showerror, error))",
        )
        passed = false
    finally
        finalize_model(model)
    end
    elapsed_time_seconds =
        Float64(UInt64(clock_ns()) - start_time_ns) / 1.0e9
    return OptimalityValidation(
        true,
        passed,
        gradient_norm,
        elapsed_time_seconds,
        isnothing(validation_error) ? nothing : "OptimalityValidationError",
        isnothing(validation_error) ? nothing : sprint(showerror, validation_error),
    )
end

function validate_optimality(
    experiment::Experiment,
    problem::String,
    outcome::SolverOutcome,
)::OptimalityValidation
    outcome.status == "OPTIMAL" || return OptimalityValidation(
        false,
        nothing,
        nothing,
        0.0,
        nothing,
        nothing,
    )
    return validate_optimality_point(
        problem,
        outcome.solution,
        experiment.gradient_tolerance,
    )
end

function solve_problem(
    experiment::Experiment,
    variant::String,
    problem::String,
    solve_problem_function::Function,
    optimality_validation_function::Function = validate_optimality,
)::BenchmarkResult
    config = solver_configuration(
        variant,
        experiment.seed;
        print_level = experiment.print_level,
        gamma_1 = experiment.gamma_1,
        trust_region_subproblem_solver = experiment.trust_region_subproblem_solver,
    )
    log_path = problem_log_path(experiment, variant, problem)
    timed = append_to_log(log_path, "Solving $variant/$problem") do
        @timed try
            solve_problem_function(experiment, variant, problem, config)
        catch error
            error isa InterruptException && rethrow()
            showerror(stderr, error, catch_backtrace())
            println(stderr)
            status = error isa OutOfMemoryError ? "OUT_OF_MEMORY" : "INCOMPLETE"
            error isa OutOfMemoryError && GC.gc()
            failed_outcome(status, experiment, error)
        end
    end
    outcome = timed.value
    outcome isa SolverOutcome || error("Problem solver returned an invalid outcome.")
    # Snapshot process-lifetime resource metrics before validation. Validation is
    # intentionally outside the @timed solver region and all solver counters.
    solver_max_rss_bytes = Int64(Sys.maxrss())
    validation = optimality_validation_function(experiment, problem, outcome)
    validation isa OptimalityValidation ||
        error("Optimality validator returned an invalid result.")
    validation_failed = validation.performed && validation.passed !== true
    final_status = validation_failed ? "GRAD_CHECK_FAILED" : outcome.status
    final_error_type = validation_failed ? validation.error_type : outcome.error_type
    final_error_message =
        validation_failed ? validation.error_message : outcome.error_message
    return BenchmarkResult(
        RESULT_SCHEMA_VERSION,
        experiment.benchmark_kind,
        experiment.problem_set,
        variant,
        problem,
        experiment.seed,
        outcome.status,
        final_status,
        outcome.total_execution_time,
        outcome.function_value,
        outcome.gradient_value,
        outcome.terminal_objective_value,
        outcome.terminal_gradient_norm,
        outcome.best_gradient_objective_value,
        outcome.best_gradient_norm,
        outcome.total_function_evaluation,
        outcome.total_gradient_evaluation,
        outcome.total_hessian_evaluation,
        outcome.total_factorization_evaluation,
        outcome.total_subproblem_iterations,
        outcome.total_hessian_vector_products,
        outcome.total_iterations,
        Float64(timed.time),
        Int64(timed.bytes),
        Float64(timed.gctime),
        # `Sys.maxrss()` is a process-lifetime high-water mark. Production
        # invokes this function in a fresh worker dedicated to this problem.
        solver_max_rss_bytes,
        validation.performed,
        validation.passed,
        validation.gradient_norm,
        validation.time_seconds,
        outcome.solver_specific,
        diagnostic_directory(experiment, variant, problem),
        final_error_type,
        final_error_message,
        iso_timestamp(),
    )
end

function solve_and_write(
    experiment::Experiment,
    variant::String,
    problem::String,
    solve_problem_function::Function,
    optimality_validation_function::Function = validate_optimality,
)::BenchmarkResult
    result = solve_problem(
        experiment,
        variant,
        problem,
        solve_problem_function,
        optimality_validation_function,
    )
    atomic_write_json(raw_result_path(experiment, variant, problem), result_dict(result))
    GC.gc()
    return result
end

function load_active_seconds(experiment::Experiment)::Float64
    path = joinpath(experiment.results_folder, "run_state.json")
    isfile(path) || return 0.0
    data = try
        JSON.parsefile(path)
    catch error
        throw(
            ErrorException(
                "Cannot read benchmark state at $path: $(sprint(showerror, error))",
            ),
        )
    end
    get(data, "schema_version", nothing) == STATE_SCHEMA_VERSION ||
        error("Unsupported benchmark state at $path")
    seconds = get(data, "total_time_script_took", nothing)
    seconds isa Real && isfinite(seconds) && seconds >= 0.0 ||
        error("Invalid benchmark state at $path")
    return Float64(seconds)
end

function write_active_seconds(experiment::Experiment, seconds::Float64)::Nothing
    atomic_write_json(
        joinpath(experiment.results_folder, "run_state.json"),
        Dict{String,Any}(
            "schema_version" => STATE_SCHEMA_VERSION,
            "total_time_script_took" => seconds,
            "updated_at" => iso_timestamp(),
        ),
    )
    return nothing
end

function result_row(
    experiment::Experiment,
    variant::String,
    problem::String,
)::Dict{String,Any}
    data = something(load_result(experiment, variant, problem))
    timeout_origin = get(data["solver_specific"], "timeout_origin", nothing)
    (timeout_origin === nothing || timeout_origin isa AbstractString) ||
        error("Raw benchmark timeout origin is invalid for $variant/$problem.")
    return Dict{String,Any}(
        "problem_name" => problem,
        "status" => String(data["status"]),
        "total_execution_time" => Float64(data["total_execution_time"]),
        "function_value" => data["function_value"],
        "gradient_value" => data["gradient_value"],
        "terminal_objective_value" => data["terminal_objective_value"],
        "terminal_gradient_norm" => data["terminal_gradient_norm"],
        "best_gradient_objective_value" => data["best_gradient_objective_value"],
        "best_gradient_norm" => data["best_gradient_norm"],
        "total_function_evaluation" => Int64(data["total_function_evaluation"]),
        "total_gradient_evaluation" => Int64(data["total_gradient_evaluation"]),
        "total_hessian_evaluation" => Int64(data["total_hessian_evaluation"]),
        "total_factorization_evaluation" => Int64(data["total_factorization_evaluation"]),
        "total_subproblem_iterations" => Int64(data["total_subproblem_iterations"]),
        "total_hessian_vector_products" => Int64(data["total_hessian_vector_products"]),
        "timeout_origin" => timeout_origin,
    )
end

function collect_variant_rows(
    experiment::Experiment,
    variant::String,
)::Vector{Dict{String,Any}}
    return [result_row(experiment, variant, problem) for problem in experiment.problems]
end

function result_dataframe(rows::Vector{Dict{String,Any}})::DataFrame
    return DataFrame(
        problem_name = String[String(row["problem_name"]) for row in rows],
        status = String[String(row["status"]) for row in rows],
        total_execution_time = Float64[
            Float64(row["total_execution_time"]) for row in rows
        ],
        function_value = Float64[
            row["function_value"] === nothing ? NaN : Float64(row["function_value"]) for
            row in rows
        ],
        gradient_value = Float64[
            row["gradient_value"] === nothing ? NaN : Float64(row["gradient_value"]) for
            row in rows
        ],
        terminal_objective_value = Float64[
            row["terminal_objective_value"] === nothing ?
            NaN : Float64(row["terminal_objective_value"]) for row in rows
        ],
        terminal_gradient_norm = Float64[
            row["terminal_gradient_norm"] === nothing ?
            NaN : Float64(row["terminal_gradient_norm"]) for row in rows
        ],
        best_gradient_objective_value = Float64[
            row["best_gradient_objective_value"] === nothing ?
            NaN : Float64(row["best_gradient_objective_value"]) for row in rows
        ],
        best_gradient_norm = Float64[
            row["best_gradient_norm"] === nothing ?
            NaN : Float64(row["best_gradient_norm"]) for row in rows
        ],
        total_function_evaluation = Int64[
            Int64(row["total_function_evaluation"]) for row in rows
        ],
        total_gradient_evaluation = Int64[
            Int64(row["total_gradient_evaluation"]) for row in rows
        ],
        total_hessian_evaluation = Int64[
            Int64(row["total_hessian_evaluation"]) for row in rows
        ],
        total_factorization_evaluation = Int64[
            Int64(row["total_factorization_evaluation"]) for row in rows
        ],
        total_subproblem_iterations = Int64[
            Int64(row["total_subproblem_iterations"]) for row in rows
        ],
        total_hessian_vector_products = Int64[
            Int64(row["total_hessian_vector_products"]) for row in rows
        ],
        timeout_origin = Union{Missing,String}[
            row["timeout_origin"] === nothing ? missing : String(row["timeout_origin"]) for
            row in rows
        ],
    )
end

function shifted_geomean(values::Vector{Float64}, shift::Float64)::Union{Nothing,Float64}
    isempty(values) && return nothing
    shifted = values .+ shift
    all(isfinite, shifted) && all(>(0.0), shifted) || return nothing
    return exp(mean(log, shifted)) - shift
end

function finite_median(values::Vector{Float64})::Union{Nothing,Float64}
    isempty(values) && return nothing
    all(isfinite, values) || return nothing
    return median(values)
end

function csv_optional_float(value::Union{Nothing,Float64})::Union{Missing,Float64}
    return value === nothing ? missing : value
end

function failure_values(experiment::Experiment, ::Bool)::Tuple{Float64,Float64}
    return (
        2.0 * experiment.max_iterations,
        2.0 * experiment.max_time_seconds,
    )
end

function run_limits_dict(experiment::Experiment)::Dict{String,Real}
    return Dict{String,Real}(
        "schema_version" => RUN_LIMITS_SCHEMA_VERSION,
        "max_iterations" => experiment.max_iterations,
        "max_time_seconds" => experiment.max_time_seconds,
    )
end

function penalized_metrics(
    rows::Vector{Dict{String,Any}},
    experiment::Experiment,
    ablation::Bool,
)::Dict{String,Vector{Float64}}
    count_penalty, runtime_penalty = failure_values(experiment, ablation)
    metrics = Dict{String,Vector{Float64}}(name => Float64[] for name in METRIC_NAMES)
    for row in rows
        successful = String(row["status"]) in SUCCESS_STATUSES
        push!(
            metrics["total_execution_time"],
            successful ? Float64(row["total_execution_time"]) : runtime_penalty,
        )
        for name in METRIC_NAMES[2:end]
            push!(metrics[name], successful ? Float64(row[name]) : count_penalty)
        end
    end
    return metrics
end

function summary_for_rows(
    rows::Vector{Dict{String,Any}},
    experiment::Experiment,
    total_time_script_took::Float64;
    ablation::Bool,
)::Dict{String,Any}
    status_counts = Dict{String,Int64}()
    for row in rows
        status = String(row["status"])
        status_counts[status] = get(status_counts, status, 0) + 1
    end
    successful_count = sum(get(status_counts, status, 0) for status in SUCCESS_STATUSES)
    metrics = penalized_metrics(rows, experiment, ablation)
    count_penalty, runtime_penalty = failure_values(experiment, ablation)
    return Dict{String,Any}(
        "schema_version" => SUMMARY_SCHEMA_VERSION,
        "generated_at" => iso_timestamp(),
        "benchmark_kind" => experiment.benchmark_kind,
        "problem_set" => experiment.problem_set,
        "seed" => experiment.seed,
        "total_time_script_took" => total_time_script_took,
        "planned_problem_count" => length(experiment.problems),
        "completed_problem_count" => length(rows),
        "successful_problem_count" => successful_count,
        "failed_problem_count" => length(rows) - successful_count,
        "complete" => length(rows) == length(experiment.problems),
        "status_counts" => status_counts,
        "run_limits" => run_limits_dict(experiment),
        "shift" => SUMMARY_SHIFT,
        "failure_values" => Dict{String,Float64}(
            "evaluation_count" => count_penalty,
            "runtime_seconds" => runtime_penalty,
        ),
        "shifted_geomeans" => Dict{String,Any}(
            name => shifted_geomean(values, SUMMARY_SHIFT) for (name, values) in metrics
        ),
        "medians" =>
            Dict{String,Any}(name => finite_median(values) for (name, values) in metrics),
    )
end

function write_variant_table(
    experiment::Experiment,
    variant::String,
    rows::Vector{Dict{String,Any}},
)::String
    path = joinpath(experiment.results_folder, variant, "table_cutest_$variant.csv")
    atomic_write_csv(path, result_dataframe(rows))
    atomic_write_json(
        joinpath(experiment.results_folder, variant, "run_limits.json"),
        run_limits_dict(experiment),
    )
    return path
end

function aggregate_standard(
    experiment::Experiment,
    total_time_script_took::Float64,
)::Dict{String,Any}
    variant = only(experiment.variants)
    rows = collect_variant_rows(experiment, variant)
    write_variant_table(experiment, variant, rows)
    summary = summary_for_rows(rows, experiment, total_time_script_took; ablation = false)
    if variant == CAT_NEURIPS_VARIANT
        summary["shifted_geomeans"]["total_factorization_evaluation"] = nothing
        summary["medians"]["total_factorization_evaluation"] = nothing
        summary["shifted_geomeans"]["total_subproblem_iterations"] = nothing
        summary["medians"]["total_subproblem_iterations"] = nothing
    end
    summary["optimization_method"] = variant
    atomic_write_json(joinpath(experiment.results_folder, "run_summary.json"), summary)
    return summary
end

function aggregate_ablation(
    experiment::Experiment,
    total_time_script_took::Float64,
)::Dict{String,Any}
    length(experiment.variants) == 1 ||
        error("Ablation aggregation requires exactly one selected variant.")
    summaries = Dict{String,Any}()
    geomean_rows = NamedTuple[]
    complete = true
    for variant in experiment.variants
        rows = collect_variant_rows(experiment, variant)
        write_variant_table(experiment, variant, rows)
        summary = summary_for_rows(rows, experiment, total_time_script_took; ablation = true)
        if variant == "conference_subproblem_solver"
            summary["shifted_geomeans"]["total_factorization_evaluation"] = nothing
            summary["medians"]["total_factorization_evaluation"] = nothing
        end
        summaries[variant] = summary
        complete &= Bool(summary["complete"])
        geomeans = summary["shifted_geomeans"]
        push!(
            geomean_rows,
            (
                criteria = variant,
                total_failure = Int(summary["failed_problem_count"]),
                geomean_total_function_evaluation = csv_optional_float(
                    geomeans["total_function_evaluation"],
                ),
                geomean_total_gradient_evaluation = csv_optional_float(
                    geomeans["total_gradient_evaluation"],
                ),
                geomean_total_hessian_evaluation = csv_optional_float(
                    geomeans["total_hessian_evaluation"],
                ),
                geomean_count_factorization = csv_optional_float(
                    geomeans["total_factorization_evaluation"],
                ),
                geomean_total_subproblem_iterations = csv_optional_float(
                    geomeans["total_subproblem_iterations"],
                ),
                geomean_total_hessian_vector_products = csv_optional_float(
                    geomeans["total_hessian_vector_products"],
                ),
                geomean_total_wall_clock_time = csv_optional_float(
                    geomeans["total_execution_time"],
                ),
            ),
        )
    end
    atomic_write_csv(
        joinpath(experiment.results_folder, "geomean_results_ablation_study.csv"),
        DataFrame(geomean_rows),
    )
    aggregate_summary = Dict{String,Any}(
        "schema_version" => SUMMARY_SCHEMA_VERSION,
        "generated_at" => iso_timestamp(),
        "benchmark_kind" => experiment.benchmark_kind,
        "problem_set" => experiment.problem_set,
        "seed" => experiment.seed,
        "total_time_script_took" => total_time_script_took,
        "complete" => complete,
        "run_limits" => run_limits_dict(experiment),
        "variants" => summaries,
    )
    atomic_write_json(
        joinpath(experiment.results_folder, "ablation_summary.json"),
        aggregate_summary,
    )
    return aggregate_summary
end

function aggregate_run(
    experiment::Experiment,
    total_time_script_took::Float64,
)::Dict{String,Any}
    return experiment.benchmark_kind == "benchmark" ?
           aggregate_standard(experiment, total_time_script_took) :
           aggregate_ablation(experiment, total_time_script_took)
end

function prepare_run(experiment::Experiment)::Vector{Tuple{String,Int64}}
    prepare_results_directory(experiment)
    pending = write_pending_runs(experiment)
    total_items = length(experiment.problems) * length(experiment.variants)
    println(
        "Prepared $(experiment.benchmark_kind) run: $(length(pending)) of " *
        "$(length(experiment.batches) * length(experiment.variants)) runs pending " *
        "($total_items work items total).",
    )
    return pending
end

function run_variant_problem(
    experiment::Experiment,
    variant::String,
    batch_index::Int64;
    problem::String,
    invocation_start::Float64 = time(),
    solve_problem_function::Function = run_model,
    warmup_function::Function = run_warmup,
    optimality_validation_function::Function = validate_optimality,
    ready_function::Function = () -> nothing,
)::Nothing
    prepare_results_directory(experiment)
    variant in experiment.variants || error("Unknown run variant '$variant'.")
    1 <= batch_index <= length(experiment.batches) ||
        error("Batch index $batch_index is outside 1:$(length(experiment.batches)).")
    batch = experiment.batches[batch_index]
    problem in batch ||
        error("Problem '$problem' is not in batch $batch_index.")
    load_result(experiment, variant, problem) === nothing || return nothing

    prior_active_seconds = load_active_seconds(experiment)
    try
        config = solver_configuration(
            variant,
            experiment.seed;
            print_level = experiment.print_level,
            gamma_1 = experiment.gamma_1,
            trust_region_subproblem_solver = experiment.trust_region_subproblem_solver,
        )
        println("Warming up $variant/$WARMUP_PROBLEM")
        warmup_function(experiment, variant, config)
        ready_function()
        log_path = problem_log_path(experiment, variant, problem)
        println("[batch $batch_index] $variant/$problem -> $log_path")
        result = solve_and_write(
            experiment,
            variant,
            problem,
            solve_problem_function,
            optimality_validation_function,
        )
        println("[batch $batch_index] $variant/$problem $(result.status)")
    finally
        elapsed = prior_active_seconds + max(0.0, time() - invocation_start)
        write_active_seconds(experiment, elapsed)
    end
    return nothing
end

function aggregate_completed_run(experiment::Experiment)::Dict{String,Any}
    prepare_results_directory(experiment)
    pending = pending_items(experiment)
    isempty(pending) ||
        error("Cannot aggregate a run with $(length(pending)) pending work items.")
    summary = aggregate_run(experiment, load_active_seconds(experiment))
    write_pending_runs(experiment)
    println("Aggregation complete: $(summary["complete"])")
    return summary
end


export ABLATION_VARIANTS,
    ABLATION_VARIANT_BY_SLUG,
    ABLATION_VARIANT_PAIRS,
    ABLATION_VARIANT_SLUGS,
    ABLATION_SLUG_BY_VARIANT,
    CAT_NEURIPS_VARIANT,
    STANDARD_VARIANTS,
    AlgorithmConfiguration,
    CATNeurIPSConfiguration,
    BenchmarkResult,
    Experiment,
    OptimalityValidation,
    OptimalityValidationError,
    SolverConfiguration,
    SolverOutcome,
    UTRAlgorithmConfiguration,
    aggregate_run,
    aggregate_completed_run,
    algorithm_configuration,
    atomic_write_json,
    atomic_write_text,
    ablation_variant_slug,
    build_experiment,
    enforce_thread_configuration,
    expected_thread_environment,
    fingerprint_files,
    internal_ablation_variant,
    load_result,
    pending_items,
    pending_problem_runs,
    pending_runs,
    prepare_run,
    raw_result_path,
    result_dict,
    run_variant_problem,
    settings_dict,
    shifted_geomean,
    solver_configuration,
    source_fingerprint,
    validate_optimality,
    validate_optimality_point

end
