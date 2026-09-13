using Test

import CATrustRegionMethod
import CATrustRegionShared
import TrustRegionSubproblemSolvers

@testset "CAT delegates trust-region subproblems" begin
    @test CATrustRegionMethod.SparseCholeskyWorkspace ===
          TrustRegionSubproblemSolvers.SparseCholeskyWorkspace
    @test CATrustRegionMethod.Optimizer <: CATrustRegionShared.AbstractUnconstrainedOptimizer
    @test CATrustRegionMethod.DEFAULT_GAMMA_1 ===
          TrustRegionSubproblemSolvers.DEFAULT_GAMMA_1
    @test fieldnames(CATrustRegionMethod.AlgorithmCounter) == (
        :total_function_evaluation,
        :total_gradient_evaluation,
        :total_hessian_evaluation,
        :total_number_factorizations,
        :total_number_subproblem_iterations,
        :total_number_hessian_vector_products,
        :total_number_factorizations_findinterval,
        :total_number_factorizations_bisection,
        :total_number_factorizations_compute_search_direction,
        :total_number_factorizations_inverse_power_iteration,
    )

    counter = CATrustRegionMethod.AlgorithmCounter()
    success, delta, direction, hard_case = CATrustRegionMethod.solveTrustRegionSubproblem(
        "package_boundary",
        0.0,
        [-2.0],
        reshape([2.0], 1, 1),
        [0.0],
        0.0,
        0.01,
        0.8,
        0.5,
        0.5,
        2.0,
        counter,
        0,
    )

    @test success
    @test delta > 0.0
    @test direction ≈ [0.5]
    @test hard_case isa Bool
    @test counter.total_number_factorizations > 0
    @test counter.total_number_factorizations ==
          counter.total_number_factorizations_findinterval +
          counter.total_number_factorizations_bisection +
          counter.total_number_factorizations_compute_search_direction +
          counter.total_number_factorizations_inverse_power_iteration
    @test counter.total_number_subproblem_iterations == 0
    @test counter.total_number_hessian_vector_products == 0

end
