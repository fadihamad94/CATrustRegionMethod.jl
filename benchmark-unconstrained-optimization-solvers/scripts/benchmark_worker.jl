import ArgParse
import CATrustRegionMethod
import TrustRegionSubproblemSolvers

include("../benchmark/benchmark_pipeline.jl")
using .BenchmarkPipeline

function parse_arguments(arguments::Vector{String})::Dict{String,Any}
    settings = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! settings begin
        "--kind"
        arg_type = String
        required = true

        "--problem-set"
        arg_type = String
        required = true

        "--results"
        arg_type = String
        required = true

        "--seed"
        arg_type = Int64
        required = true

        "--print-level"
        arg_type = Int64
        default = 0

        "--number-of-threads"
        arg_type = Int64
        required = true

        "--solver"
        arg_type = String
        default = ""

        "--ablation-variant"
        arg_type = String
        default = ""

        "--subproblem-solver"
        arg_type = String
        default = "DIRECT-NEW"

        "--gamma-1"
        arg_type = Float64
        default = TrustRegionSubproblemSolvers.DEFAULT_GAMMA_1

        "--manual-skip"
        arg_type = String
        action = :append_arg
        default = String[]

        "--batch-index"
        arg_type = Int64
        default = 0

        "--variant"
        arg_type = String
        default = ""

        "--problem"
        arg_type = String
        default = ""

        "--invocation-start"
        arg_type = Float64
        default = 0.0
    end
    return ArgParse.parse_args(arguments, settings)
end

function main(arguments::Vector{String} = ARGS)::Nothing
    isempty(arguments) && error("Missing worker mode.")
    mode = first(arguments)
    parsed = parse_arguments(arguments[2:end])
    BenchmarkPipeline.enforce_thread_configuration(parsed["number-of-threads"])
    experiment = BenchmarkPipeline.build_experiment(
        parsed["kind"],
        parsed["problem-set"],
        parsed["results"],
        parsed["seed"];
        solver = isempty(parsed["solver"]) ? nothing : parsed["solver"],
        print_level = parsed["print-level"],
        num_threads = parsed["number-of-threads"],
        gamma_1 = parsed["gamma-1"],
        subproblem_solver = parsed["subproblem-solver"],
        manually_skipped_problems = parsed["manual-skip"],
        ablation_variant = isempty(parsed["ablation-variant"]) ? nothing :
                           parsed["ablation-variant"],
    )
    if mode == "prepare"
        BenchmarkPipeline.prepare_run(experiment)
    elseif mode == "run"
        parsed["batch-index"] > 0 || error("run requires --batch-index.")
        isempty(parsed["variant"]) && error("run requires --variant.")
        isempty(parsed["problem"]) && error("run requires --problem.")
        parsed["invocation-start"] > 0.0 ||
            error("run requires --invocation-start.")
        BenchmarkPipeline.run_variant_problem(
            experiment,
            parsed["variant"],
            parsed["batch-index"];
            problem = parsed["problem"],
            invocation_start = parsed["invocation-start"],
        )
    elseif mode == "aggregate"
        BenchmarkPipeline.aggregate_completed_run(experiment)
    else
        error("Unknown worker mode '$mode'.")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
