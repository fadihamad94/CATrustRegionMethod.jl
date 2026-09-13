using LinearAlgebra
using Random
using SparseArrays
using Test

using TrustRegionSubproblemSolvers

const TRSS = TrustRegionSubproblemSolvers

function result_satisfies_termination_criteria(
    result::TrustRegionSubproblemResult,
    g::Vector{Float64},
    H,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    radius::Float64,
    min_grad::Float64,
)
    direction = result.direction
    shift = result.delta
    direction_norm = norm(direction)
    model_value = dot(g, direction) + 0.5 * dot(direction, H * direction)
    return norm(H * direction + g + shift * direction) <= γ_1 * min_grad &&
           γ_2 * shift * radius <= shift * direction_norm &&
           direction_norm <= radius &&
           model_value <= -γ_3 * 0.5 * shift * direction_norm^2
end

@testset "trust-region solver" begin
    @testset "public API surface" begin
        exported_names = names(TrustRegionSubproblemSolvers)
        @test :solveTrustRegionSubproblem in exported_names
        @test :solveTrustRegionSubproblemOldApproach in exported_names
        @test :phi ∉ exported_names
        @test :findinterval ∉ exported_names
        @test :bisection ∉ exported_names
        @test :phiOldApproach ∉ exported_names
        @test :findintervalOldApproach ∉ exported_names
        @test :bisectionOldApproach ∉ exported_names
        @test :optimizeSecondOrderModelOldApproach ∉ exported_names
        @test isdefined(TrustRegionSubproblemSolvers, :phi)
    end

    @testset "public input validation" begin
        valid_arguments = (
            "input-validation",
            [0.0],
            reshape([2.0], 1, 1),
            0.0,
            0.01,
            0.8,
            0.5,
            1.0,
            0.0,
        )
        @test_throws ArgumentError solveTrustRegionSubproblem(
            valid_arguments[1],
            [NaN],
            valid_arguments[3:end]...,
        )
        @test_throws ArgumentError solveTrustRegionSubproblem(
            valid_arguments[1:2]...,
            reshape([NaN], 1, 1),
            valid_arguments[4:end]...,
        )
        @test_throws ArgumentError solveTrustRegionSubproblem(
            valid_arguments[1:3]...,
            Inf,
            valid_arguments[5:end]...,
        )
        @test_throws ArgumentError solveTrustRegionSubproblem(
            valid_arguments[1:7]...,
            Inf,
            valid_arguments[9],
        )
        @test_throws ArgumentError solveTrustRegionSubproblem(
            valid_arguments[1:8]...,
            Inf,
        )
        @test_throws ArgumentError solveTrustRegionSubproblem(
            "nonsymmetric-hessian",
            zeros(2),
            [1.0 1.0; 0.0 1.0],
            0.0,
            0.01,
            0.8,
            0.5,
            1.0,
            0.0,
        )
        @test_throws ArgumentError TRSS.checkCandidateTerminationCriteria(
            [NaN],
            valid_arguments[2],
            valid_arguments[3:end]...,
        )
    end

    @testset "Newton, boundary, and literal zero-step criteria" begin
        γ_1, γ_2, γ_3 = 0.01, 0.8, 0.5

        interior = solveTrustRegionSubproblem(
            "primary-interior",
            [-2.0],
            reshape([2.0], 1, 1),
            0.0,
            γ_1,
            γ_2,
            γ_3,
            2.0,
            2.0;
            print_level = 0,
        )
        @test interior.success
        @test interior.delta == 0.0
        @test interior.direction ≈ [1.0]
        @test result_satisfies_termination_criteria(
            interior,
            [-2.0],
            reshape([2.0], 1, 1),
            γ_1,
            γ_2,
            γ_3,
            2.0,
            2.0,
        )

        boundary = solveTrustRegionSubproblem(
            "primary-boundary",
            [-2.0],
            reshape([2.0], 1, 1),
            0.0,
            γ_1,
            γ_2,
            γ_3,
            0.5,
            2.0;
            print_level = 0,
        )
        @test boundary.success
        @test boundary.delta > 0.0
        @test γ_2 * 0.5 <= norm(boundary.direction) <= 0.5
        @test result_satisfies_termination_criteria(
            boundary,
            [-2.0],
            reshape([2.0], 1, 1),
            γ_1,
            γ_2,
            γ_3,
            0.5,
            2.0,
        )

        zero_step = solveTrustRegionSubproblem(
            "primary-zero",
            [0.0],
            reshape([2.0], 1, 1),
            0.0,
            γ_1,
            γ_2,
            γ_3,
            1.0,
            0.0;
            print_level = 0,
        )
        @test zero_step.success
        @test zero_step.delta == 0.0
        @test iszero(norm(zero_step.direction))
        @test result_satisfies_termination_criteria(
            zero_step,
            [0.0],
            reshape([2.0], 1, 1),
            γ_1,
            γ_2,
            γ_3,
            1.0,
            0.0,
        )

        zero_check = checkTrustRegionSubproblemTerminationCriteria(
            zero_step.direction,
            [0.0],
            reshape([2.0], 1, 1),
            zero_step.delta,
            γ_1,
            γ_2,
            γ_3,
            1.0,
            0.0,
        )
        @test zero_check.valid
        @test zero_check.zero_step
        @test isnothing(
            validateTrustRegionSubproblemTerminationCriteria(
                "primary-zero",
                zero_step.direction,
                [0.0],
                reshape([2.0], 1, 1),
                0.0,
                zero_step.delta,
                zero_step.delta_prime,
                γ_1,
                γ_2,
                γ_3,
                1.0,
                0.0,
                zero_step.hard_case,
                0,
            ),
        )

        invalid_zero_check = checkTrustRegionSubproblemTerminationCriteria(
            [0.0],
            [1.0],
            reshape([2.0], 1, 1),
            0.0,
            γ_1,
            γ_2,
            γ_3,
            1.0,
            1.0,
        )
        @test !invalid_zero_check.valid
        @test invalid_zero_check.zero_step
        @test invalid_zero_check.failure_reason_6a
        @test_throws TrustRegionSubproblemError validateTrustRegionSubproblemTerminationCriteria(
            "invalid-zero",
            [0.0],
            [1.0],
            reshape([2.0], 1, 1),
            0.0,
            0.0,
            0.0,
            γ_1,
            γ_2,
            γ_3,
            1.0,
            1.0,
            false,
            0,
        )

        valid_arguments = (
            [0.0],
            [0.0],
            reshape([2.0], 1, 1),
            0.0,
            γ_1,
            γ_2,
            γ_3,
            1.0,
            0.0,
        )
        @test_throws ArgumentError checkTrustRegionSubproblemTerminationCriteria(
            [NaN],
            valid_arguments[2:end]...,
        )
        @test_throws ArgumentError checkTrustRegionSubproblemTerminationCriteria(
            valid_arguments[1],
            [Inf],
            valid_arguments[3:end]...,
        )
        @test_throws ArgumentError checkTrustRegionSubproblemTerminationCriteria(
            valid_arguments[1:2]...,
            reshape([NaN], 1, 1),
            valid_arguments[4:end]...,
        )
        @test_throws ArgumentError checkTrustRegionSubproblemTerminationCriteria(
            valid_arguments[1:3]...,
            NaN,
            valid_arguments[5:end]...,
        )
        @test_throws ArgumentError checkTrustRegionSubproblemTerminationCriteria(
            valid_arguments[1:7]...,
            Inf,
            valid_arguments[9],
        )
        @test_throws ArgumentError checkTrustRegionSubproblemTerminationCriteria(
            valid_arguments...,
            [NaN],
        )
        @test_throws ArgumentError checkTrustRegionSubproblemTerminationCriteria(
            valid_arguments...,
            nothing,
            NaN,
        )
        @test_throws OverflowError checkTrustRegionSubproblemTerminationCriteria(
            [1.0e308],
            [0.0],
            reshape([1.0e308], 1, 1),
            0.0,
            γ_1,
            γ_2,
            γ_3,
            1.0e308,
            0.0,
        )
    end

    @testset "classifier and effective zero shift" begin
        γ_1, γ_2, γ_3 = 0.01, 0.8, 0.5
        counter = TRSS.FactorizationCounter()

        indefinite = TRSS.phi(
            [1.0],
            reshape([-1.0], 1, 1),
            0.0,
            γ_1,
            γ_2,
            γ_3,
            1.0,
            1.0,
            counter,
            :findinterval,
            nothing,
        )
        @test indefinite.value == TRSS.PHI_NEGATIVE
        @test !indefinite.positive_definite

        outside = TRSS.phi(
            [-2.0],
            reshape([2.0], 1, 1),
            0.0,
            γ_1,
            γ_2,
            γ_3,
            0.5,
            2.0,
            counter,
            :findinterval,
            nothing,
        )
        @test outside.value == TRSS.PHI_NEGATIVE
        @test outside.positive_definite

        boundary = TRSS.phi(
            [-2.0],
            reshape([2.0], 1, 1),
            2.0,
            γ_1,
            γ_2,
            γ_3,
            0.5,
            2.0,
            counter,
            :findinterval,
            nothing,
        )
        @test boundary.value == TRSS.PHI_ZERO
        @test boundary.accepted_shift == 2.0

        below_boundary = TRSS.phi(
            [-2.0],
            reshape([2.0], 1, 1),
            20.0,
            γ_1,
            γ_2,
            γ_3,
            0.5,
            2.0,
            counter,
            :findinterval,
            nothing,
        )
        @test below_boundary.value == TRSS.PHI_POSITIVE

        effective_zero = TRSS.phi(
            [-0.001],
            reshape([1.0], 1, 1),
            1.0,
            γ_1,
            γ_2,
            γ_3,
            1.0,
            1.0,
            counter,
            :findinterval,
            nothing,
        )
        @test effective_zero.value == TRSS.PHI_ZERO
        @test effective_zero.shift == 1.0
        @test effective_zero.accepted_shift == 0.0
    end

    @testset "certified interval and hard case" begin
        γ_1, γ_2, γ_3 = 0.2, 0.8, 0.5
        H = [-1.0 0.0; 0.0 2.0]
        g = [0.0, 1.0]
        counter = TRSS.FactorizationCounter()
        zero_evaluation = TRSS.phi(
            g,
            H,
            0.0,
            γ_1,
            γ_2,
            γ_3,
            1.0,
            1.0,
            counter,
            :compute_search_direction,
            nothing,
        )
        interval = TRSS.findinterval(
            g,
            H,
            0.0,
            γ_1,
            γ_2,
            γ_3,
            1.0,
            1.0,
            counter,
            nothing,
            zero_evaluation = zero_evaluation,
        )
        @test interval.status == :interval
        @test interval.lower.value == TRSS.PHI_NEGATIVE
        @test interval.upper.value == TRSS.PHI_POSITIVE
        @test interval.lower.shift < interval.upper.shift

        hard_interval = TRSS.bisection(
            g,
            H,
            interval,
            γ_1,
            γ_2,
            γ_3,
            1.0,
            1.0,
            counter,
            nothing,
        )
        @test hard_interval.status == :hard_case
        @test !hard_interval.lower.positive_definite
        @test hard_interval.upper.value == TRSS.PHI_POSITIVE
        @test hard_interval.upper.shift - hard_interval.lower.shift <=
              TRSS.hardCaseTolerance(γ_1, γ_3, 1.0, 1.0)

        Random.seed!(1)
        result = solveTrustRegionSubproblem(
            "primary-hard",
            g,
            H,
            0.0,
            γ_1,
            γ_2,
            γ_3,
            1.0,
            1.0;
            print_level = 0,
        )
        @test result.success
        @test result.hard_case
        @test result.factorizations.inverse_power_iteration == 1
        @test result_satisfies_termination_criteria(
            result,
            g,
            H,
            γ_1,
            γ_2,
            γ_3,
            1.0,
            1.0,
        )
    end

    @testset "refinement violation continuation and perturbed-gradient backup" begin
        H = reshape([0.1], 1, 1)
        g = [0.3]
        workspace = SparseCholeskyWorkspace(true, 0)
        factorization = TRSS.factorizeShiftedHessian!(workspace, H, 0.2)
        certificate = TRSS.certifiedShiftedSolve(
            factorization,
            g,
            H,
            0.2,
            0.0,
            workspace,
        )
        @test !certificate.certified
        @test certificate.residual_norm > 0.0

        accepted_uncertified = TRSS.phi(
            g,
            H,
            0.2,
            0.01,
            0.8,
            0.5,
            1.1,
            3.0e-15,
            TRSS.FactorizationCounter(),
            :bisection,
            SparseCholeskyWorkspace(true, 0),
        )
        @test accepted_uncertified.value == TRSS.PHI_ZERO
        @test !accepted_uncertified.certified
        @test TRSS.checkCandidateTerminationCriteria(
            accepted_uncertified.direction,
            g,
            H,
            accepted_uncertified.accepted_shift,
            0.01,
            0.8,
            0.5,
            1.1,
            3.0e-15,
        ).valid

        outside_uncertified = TRSS.phi(
            g,
            H,
            0.2,
            0.01,
            0.8,
            0.5,
            0.5,
            3.0e-15,
            TRSS.FactorizationCounter(),
            :bisection,
            SparseCholeskyWorkspace(true, 0),
        )
        @test outside_uncertified.value == TRSS.PHI_NEGATIVE
        @test !outside_uncertified.certified

        below_boundary_uncertified = TRSS.phi(
            g,
            H,
            0.2,
            0.01,
            0.8,
            0.5,
            2.0,
            3.0e-15,
            TRSS.FactorizationCounter(),
            :bisection,
            SparseCholeskyWorkspace(true, 0),
        )
        @test below_boundary_uncertified.value == TRSS.PHI_POSITIVE
        @test !below_boundary_uncertified.certified

        invalid_zero_evaluation = TRSS.phi(
            g,
            H,
            0.0,
            0.01,
            0.8,
            0.5,
            3.1,
            3.0e-15,
            TRSS.FactorizationCounter(),
            :compute_search_direction,
            SparseCholeskyWorkspace(true, 0),
        )
        @test invalid_zero_evaluation.value == TRSS.PHI_ZERO
        @test !invalid_zero_evaluation.certified
        @test !TRSS.checkCandidateTerminationCriteria(
            invalid_zero_evaluation.direction,
            g,
            H,
            invalid_zero_evaluation.accepted_shift,
            0.01,
            0.8,
            0.5,
            3.1,
            3.0e-15,
        ).valid

        result_after_invalid_zero = TRSS.optimizeSecondOrderModel(
            "primary-invalid-zero",
            g,
            H,
            0.0,
            0.01,
            0.8,
            0.5,
            3.1,
            3.0e-15;
            print_level = 0,
            use_backup_trust_region_subproblem_solver = false,
            sparse_cholesky_workspace = SparseCholeskyWorkspace(true, 0),
        )
        result_after_invalid_zero_check =
            TRSS.checkCandidateTerminationCriteria(
                result_after_invalid_zero.direction,
                g,
                H,
                result_after_invalid_zero.delta,
                0.01,
                0.8,
                0.5,
                3.1,
                3.0e-15,
            )
        @test result_after_invalid_zero.factorizations.total > 1
        @test !result_after_invalid_zero.success || result_after_invalid_zero_check.valid

        continued_after_refinement_violation =
            solveTrustRegionSubproblem(
                "primary-uncertified",
                g,
                H,
                0.0,
                0.01,
                0.8,
                0.5,
                1.0,
                0.0;
                print_level = 0,
                sparse_cholesky_workspace = SparseCholeskyWorkspace(true, 0),
            )
        @test continued_after_refinement_violation.success
        @test result_satisfies_termination_criteria(
            continued_after_refinement_violation,
            g,
            H,
            0.01,
            0.8,
            0.5,
            1.0,
            0.0,
        )

        hard_H = [-1.0 0.0; 0.0 2.0]
        hard_g = [0.0, 1.0]
        Random.seed!(1)
        without_backup = TRSS.optimizeSecondOrderModel(
            "primary-no-backup",
            hard_g,
            hard_H,
            0.0,
            0.2,
            0.8,
            0.5,
            1.0,
            1.0;
            print_level = 0,
            use_backup_trust_region_subproblem_solver = false,
            inverse_power_max_iterations = 0,
        )
        Random.seed!(1)
        with_backup = TRSS.optimizeSecondOrderModel(
            "primary-with-backup",
            hard_g,
            hard_H,
            0.0,
            0.2,
            0.8,
            0.5,
            1.0,
            1.0;
            print_level = 0,
            use_backup_trust_region_subproblem_solver = true,
            inverse_power_max_iterations = 0,
        )
        @test !without_backup.success
        @test with_backup.success
        @test with_backup.hard_case
        @test with_backup.factorizations.total > without_backup.factorizations.total
        @test result_satisfies_termination_criteria(
            with_backup,
            hard_g,
            hard_H,
            0.2,
            0.8,
            0.5,
            1.0,
            1.0,
        )
    end

    @testset "dense and sparse parity and accounting" begin
        dense_H = [4.0 1.0; 1.0 3.0]
        sparse_H = sparse(dense_H)
        symmetric_sparse_H = Symmetric(sparse(tril(dense_H)), :L)
        g = [1.0, -2.0]
        results = TrustRegionSubproblemResult[]
        for H in (dense_H, sparse_H, symmetric_sparse_H)
            workspace = SparseCholeskyWorkspace()
            result = solveTrustRegionSubproblem(
                "primary-representation",
                g,
                H,
                0.0,
                0.01,
                0.8,
                0.5,
                10.0,
                norm(g);
                print_level = 0,
                sparse_cholesky_workspace = workspace,
            )
            push!(results, result)
            @test result.success
            @test result.factorizations.total ==
                  result.factorizations.findinterval +
                  result.factorizations.bisection +
                  result.factorizations.compute_search_direction +
                  result.factorizations.inverse_power_iteration
            if !(H isa Matrix)
                @test workspace.symbolic_factorizations == 1
            end
        end
        @test results[1].direction ≈ results[2].direction
        @test results[1].direction ≈ results[3].direction
    end

    @testset "input validation" begin
        valid_arguments =
            ("invalid", [1.0], reshape([1.0], 1, 1), 0.0, 0.01, 0.8, 0.5, 1.0, 1.0)

        for (name, γ_1, γ_2, γ_3) in (
            ("gamma-1-zero", 0.0, 0.8, 0.5),
            ("gamma-1-one", 1.0, 0.8, 0.5),
            ("gamma-2-zero", 0.01, 0.0, 0.5),
            ("gamma-3-zero", 0.01, 0.8, 0.0),
        )
            @test_throws ArgumentError solveTrustRegionSubproblem(
                name,
                [-2.0],
                reshape([2.0], 1, 1),
                0.0,
                γ_1,
                γ_2,
                γ_3,
                0.5,
                2.0;
                print_level = 0,
            )
        end

        @test_throws ArgumentError solveTrustRegionSubproblem(
            "gamma-2-one-without-backup",
            [-1.0],
            reshape([2.0], 1, 1),
            0.0,
            0.01,
            1.0,
            0.5,
            0.31,
            1.0;
            print_level = 0,
            use_backup_trust_region_subproblem_solver = false,
        )
        @test_throws ArgumentError solveTrustRegionSubproblem(
            "gamma-3-one-hard-case-without-backup",
            [0.0, 1.0],
            [-1.0 0.0; 0.0 2.0],
            0.0,
            0.2,
            0.8,
            1.0,
            1.0,
            1.0;
            print_level = 0,
            use_backup_trust_region_subproblem_solver = false,
        )
        @test_throws ArgumentError checkTrustRegionSubproblemTerminationCriteria(
            [0.0],
            [0.0],
            reshape([2.0], 1, 1),
            0.0,
            0.0,
            0.8,
            0.5,
            1.0,
            0.0,
        )
        @test_throws ArgumentError solveTrustRegionSubproblem(
            valid_arguments[1:3]...,
            -1.0,
            valid_arguments[5:end]...;
            print_level = 0,
        )
        @test_throws ArgumentError solveTrustRegionSubproblem(
            valid_arguments[1:7]...,
            -1.0,
            valid_arguments[9];
            print_level = 0,
        )
        @test_throws DimensionMismatch solveTrustRegionSubproblem(
            "invalid-dimensions",
            [1.0, 2.0],
            reshape([1.0], 1, 1),
            0.0,
            0.01,
            0.8,
            0.5,
            1.0,
            1.0;
            print_level = 0,
        )
    end
end
