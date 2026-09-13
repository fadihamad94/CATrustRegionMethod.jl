using LinearAlgebra
using Random
using SparseArrays
using Test
using TrustRegionSubproblemSolvers

@testset "internal defaults grouped by solver" begin
    internal_defaults = TrustRegionSubproblemSolvers.DEFAULTS["internal"]
    @test Set(keys(internal_defaults)) ==
          Set(["common", "DIRECT-NEW", "OLD"])
    @test Set(keys(internal_defaults["common"])) ==
          Set(["power_iteration_max_iterations", "power_iteration_tolerance"])
    @test Set(keys(internal_defaults["DIRECT-NEW"])) ==
          Set([
        "find_interval_max_iterations",
        "bisection_max_iterations",
        "inverse_power_max_iterations",
    ])
    @test Set(keys(internal_defaults["OLD"])) ==
          Set(["old_solver_eigendecomposition_memory_fraction"])
end

@testset "spectral norm estimate" begin
    Random.seed!(1)
    @test matrix_l2_norm(zeros(2, 2)) == 0.0
    @test matrix_l2_norm(Matrix(Diagonal([1.0, -1.0]))) ≈ 1.0
    @test matrix_l2_norm([0.0 100.0; 100.0 0.0]) ≈ 100.0
    @test matrix_l2_norm([0.0 2.0; 0.0 0.0]) ≈ 2.0
    @test matrix_l2_norm(reshape([1.0e308], 1, 1)) == 1.0e308
    @test matrix_l2_norm(
        Matrix(Diagonal([1.0e-8, 0.9e-8]));
        num_iter = 500,
        tol = 1.0e-12,
    ) ≈ 1.0e-8 rtol = 1.0e-8
    sparse_singleton = Symmetric(
        sparse([1], [1], [3.0], 100_000, 100_000),
        :L,
    )
    @test matrix_l2_norm(sparse_singleton; num_iter = 3) ≈ 3.0
    inactive_upper_entry = Symmetric(
        sparse([1, 1, 2], [1, 2, 2], [2.0, 1.0e308, 3.0], 2, 2),
        :L,
    )
    @test matrix_l2_norm(
        inactive_upper_entry;
        num_iter = 500,
        tol = 1.0e-12,
    ) ≈
          3.0 rtol = 1.0e-8
    @test matrix_l2_norm(spdiagm(0 => [3.0, -2.0]); num_iter = 100) ≈
          3.0 rtol = 1.0e-5
    @test_throws ArgumentError matrix_l2_norm(ones(2, 2); num_iter = 0)
    @test_throws ArgumentError matrix_l2_norm(ones(2, 2); tol = NaN)
    @test_throws ArgumentError matrix_l2_norm([1.0 NaN; NaN 1.0])
end

@testset "TrustRegionSubproblemSolvers" begin
    @testset "interior solution" begin
        result = solveTrustRegionSubproblem(
            "interior",
            [-2.0],
            reshape([2.0], 1, 1),
            0.0,
            0.01,
            0.8,
            0.5,
            2.0,
            2.0;
            print_level = 0,
        )

        @test result.success
        @test result.delta == 0.0
        @test result.direction ≈ [1.0]
        @test !result.hard_case
        @test result.factorizations.total ==
              result.factorizations.findinterval +
              result.factorizations.bisection +
              result.factorizations.compute_search_direction +
              result.factorizations.inverse_power_iteration
    end

    @testset "boundary solution" begin
        result = solveTrustRegionSubproblem(
            "boundary",
            [-2.0],
            reshape([2.0], 1, 1),
            0.0,
            0.01,
            0.8,
            0.5,
            0.5,
            2.0;
            print_level = 0,
        )

        @test result.success
        @test result.delta > 0.0
        @test norm(result.direction) <= 0.5
        @test norm(result.direction) >= 0.8 * 0.5
    end

    @testset "dense and sparse workspaces" begin
        dense_hessian = [4.0 1.0; 1.0 3.0]
        sparse_hessian = sparse(dense_hessian)
        gradient = [1.0, -2.0]
        workspace = SparseCholeskyWorkspace()

        dense_factor = TrustRegionSubproblemSolvers.factorizeShiftedHessian!(
            workspace,
            dense_hessian,
            0.5,
        )
        dense_direction = TrustRegionSubproblemSolvers.solveFactorizedSystem(
            dense_factor,
            gradient,
            workspace,
        )
        @test dense_direction ≈ -(dense_hessian + 0.5I) \ gradient

        sparse_factor = TrustRegionSubproblemSolvers.factorizeShiftedHessian!(
            workspace,
            sparse_hessian,
            0.5,
        )
        @test issuccess(sparse_factor)
        @test workspace.symbolic_factorizations == 1
    end

    @testset "old solver remains available" begin
        success, delta, direction, hard_case = solveTrustRegionSubproblemOldApproach(
            0.0,
            [-2.0],
            reshape([2.0], 1, 1),
            [0.0],
            0.0,
            0.8,
            2.0,
        )
        @test success
        @test delta == 0.0
        @test direction ≈ [1.0]
        @test !hard_case
    end

    @testset "projection validates radius" begin
        @test TrustRegionSubproblemSolvers.floatingPointSafeBallProjection(
            [3.0, 4.0],
            1.0,
        ) |> norm <= 1.0
        @test_throws ArgumentError TrustRegionSubproblemSolvers.floatingPointSafeBallProjection(
            [1.0],
            -1.0,
        )
    end
end

include("test_TRS_solver.jl")
include("test_fixtures.jl")
include("test_TRS_solver_old_approach.jl")
