using JSON
using Test

const BENCHMARK_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(BENCHMARK_ROOT, "benchmark", "benchmark_pipeline.jl"))
using .BenchmarkPipeline

@testset "replication experiment definitions" begin
    @test STANDARD_VARIANTS == ["CAT", "CAT-NeurIPS", "UTR"]
    @test ABLATION_VARIANT_SLUGS == [
        "original",
        "rho-hat-rule",
        "radius-update-rule",
        "initial-radius",
        "conference-subproblem-solver",
        "xi-zero",
        "b-k-zero",
    ]
    @test internal_ablation_variant("rho-hat-rule") == "ρ_hat_rule"
    @test ablation_variant_slug("b_k=0.0") == "b-k-zero"

    cat = solver_configuration("CAT", 1)
    @test cat isa AlgorithmConfiguration
    @test cat.trust_region_subproblem_solver == "DIRECT-NEW"
    neurips = solver_configuration("CAT-NeurIPS", 1; trust_region_subproblem_solver = "OLD")
    @test neurips isa CATNeurIPSConfiguration
    utr = solver_configuration("UTR", 1)
    @test utr isa UTRAlgorithmConfiguration
    @test_throws ErrorException solver_configuration("TRU", 1)
end

@testset "frozen profiles and settings schema" begin
    mktempdir() do directory
        experiment = build_experiment(
            "benchmark",
            "super_fast",
            directory,
            1;
            solver = "CAT",
            benchmark_root = BENCHMARK_ROOT,
            fingerprint = "test-fingerprint",
        )
        @test length(experiment.problems) == 30
        @test experiment.max_iterations == 100_000
        @test experiment.max_time_seconds == 120.0

        settings = settings_dict(experiment)
        @test settings["schema_version"] == 18
        @test settings["source_fingerprint"] == "test-fingerprint"
        pending = prepare_run(experiment)
        @test length(pending) == 1
        @test isfile(joinpath(directory, "run_settings.json"))
        @test length(readlines(joinpath(directory, "pending_problems.tsv"))) == 31
    end
end

@testset "CLI exposes only retained methods" begin
    script = joinpath(BENCHMARK_ROOT, "scripts", "run_benchmark.sh")
    help_text = read(`bash $script --help`, String)
    @test occursin("CAT|CAT-NeurIPS|UTR", help_text)
    invalid = run(
        pipeline(
            ignorestatus(`bash $script --super_fast --solver TRU --results ignored`),
            stdout = devnull,
            stderr = devnull,
        ),
    )
    @test !success(invalid)
end

@testset "summary statistics" begin
    @test shifted_geomean([0.0, 3.0], 1.0) ≈ 1.0
    @test shifted_geomean(Float64[], 1.0) === nothing
end
