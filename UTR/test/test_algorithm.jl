using LinearAlgebra
using SparseArrays

@testset "UTR algorithm helpers" begin
    function select_parameters(H, gradient_norm = 1.0, ρ = 1.0)
        counter = UTR.AlgorithmCounter()
        workspace = UTR.SparseCholeskyWorkspace()
        selection = UTR.selectUTRSubproblemParameters(
            H,
            gradient_norm,
            ρ,
            counter,
            workspace,
        )
        return selection, counter
    end

    strongly_negative, negative_counter =
        select_parameters(reshape([-2.0], 1, 1))
    @test strongly_negative.selection_case == 1
    @test strongly_negative.σ == 0.0
    @test strongly_negative.r == 0.5
    @test strongly_negative.hessian_shift == 0.0
    @test strongly_negative.radius == 0.5
    @test negative_counter.total_number_factorizations_parameter_selection == 1

    central, central_counter = select_parameters(reshape([0.0], 1, 1))
    @test central.selection_case == 2
    @test central.σ == 1.0
    @test central.r == 0.25
    @test central.hessian_shift == 1.0
    @test central.radius == 0.25
    @test central_counter.total_number_factorizations_parameter_selection == 2

    strongly_positive, positive_counter =
        select_parameters(reshape([2.0], 1, 1))
    @test strongly_positive.selection_case == 1
    @test strongly_positive.σ == 0.0
    @test strongly_positive.radius == 0.5
    @test positive_counter.total_number_factorizations_parameter_selection == 2

    # The strict-PD Cholesky convention puts the exact positive edge in Case 2.
    positive_boundary, boundary_counter =
        select_parameters(reshape([1.0], 1, 1))
    @test positive_boundary.selection_case == 2
    @test positive_boundary.hessian_shift == 1.0
    @test boundary_counter.total_number_factorizations_parameter_selection == 2

    # The exact negative edge is not positive definite after the +τ shift.
    negative_boundary, negative_boundary_counter =
        select_parameters(reshape([-1.0], 1, 1))
    @test negative_boundary.selection_case == 1
    @test negative_boundary.hessian_shift == 0.0
    @test negative_boundary_counter.total_number_factorizations_parameter_selection ==
          1

    scaled, _ = select_parameters(reshape([0.0], 1, 1), 16.0, 2.0)
    @test scaled.gradient_scale == 4.0
    @test scaled.τ == 8.0
    @test scaled.hessian_shift == 8.0
    @test scaled.radius == 0.5
    @test scaled.σ == 2.0
    @test scaled.r == 0.125

    large_penalty, _ =
        select_parameters(reshape([-1e155], 1, 1), 1e-308, 1e308)
    @test large_penalty.selection_case == 1
    @test large_penalty.r == 5e-309
    @test isfinite(large_penalty.r)

    sparse_selection, sparse_counter =
        select_parameters(sparse(Diagonal([0.0, 0.5])))
    @test sparse_selection.selection_case == 2
    @test sparse_counter.total_number_factorizations_parameter_selection == 2
    @test UTR.assertFactorizationAccounting(sparse_counter) === sparse_counter

    dense_hessian = [2.0 0.0; 0.0 3.0]
    @test UTR.regularizeHessian(dense_hessian, 0.0) === dense_hessian
    @test UTR.regularizeHessian(dense_hessian, 1.5) ==
          [3.5 0.0; 0.0 4.5]
    sparse_hessian = sparse(dense_hessian)
    @test UTR.regularizeHessian(sparse_hessian, 1.5) isa SparseMatrixCSC
    @test Matrix(UTR.regularizeHessian(sparse_hessian, 1.5)) ==
          [3.5 0.0; 0.0 4.5]

    η = 0.02
    ξ = 0.5
    @test UTR.trialAccepted(10.0, 9.9, 4.0, 4.0, 2.0, η, ξ)
    @test UTR.trialAccepted(10.0, 9.99, 4.0, 1.0, 2.0, η, ξ)
    @test !UTR.trialAccepted(10.0, 10.01, 4.0, 0.0, 2.0, η, ξ)
    @test !UTR.trialAccepted(10.0, 9.99, 4.0, 3.0, 2.0, η, ξ)
    @test UTR.trialAccepted(
        0.0,
        -1e299,
        1e240,
        1e240,
        1e60,
        η,
        ξ,
    )

    @test UTR.rejectedPenalty(2.0, 2.25) == 4.5
    @test UTR.acceptedPenalty(4.5, 1e-5, 1.75) == 4.5 / 1.75
    @test UTR.acceptedPenalty(1e-6, 1e-5, 1.75) == 1e-5

    @test UTR._normalizeHessian([1.0 0.0; 0.0 2.0], 2) ==
          [1.0 0.0; 0.0 2.0]
    @test_throws UTR.InvalidHessianError UTR._normalizeHessian(
        [1.0 2.0; 0.0 1.0],
        2,
    )
    @test_throws UTR.InvalidHessianError UTR._normalizeHessian(
        reshape([1.0], 1, 1),
        2,
    )
    @test_throws DomainError UTR._normalizeHessian(
        reshape([NaN], 1, 1),
        1,
    )
end
