const UTR = UniversalTrustRegionMethod
import CATrustRegionShared
import TrustRegionSubproblemSolvers

@testset "UTR parameters and counters" begin
    @test UTR.Optimizer <: CATrustRegionShared.AbstractUnconstrainedOptimizer
    @test UTR.DEFAULT_GAMMA_1 === TrustRegionSubproblemSolvers.DEFAULT_GAMMA_1
    @test parentmodule(UTR.TerminationStatusCode.T) === UTR.TerminationStatusCode
    @test CATrustRegionShared.canonical_status_string(
        UTR.TerminationStatusCode.INNER_ITERATION_LIMIT,
    ) == "INNER_ITERATION_LIMIT"
    criteria = UTR.TerminationCriteria()
    @test criteria.MAX_ITERATIONS == 100_000
    @test criteria.MAX_INNER_ITERATIONS == 100
    @test criteria.gradient_termination_tolerance == 1e-5
    @test criteria.MAX_TIME == 18_000.0
    @test criteria.STEP_SIZE_LIMIT == 2e-16
    @test criteria.MINIMUM_OBJECTIVE_FUNCTION == -1e30
    @test criteria.iterative_refinement_max_iterations == 3

    @test_throws AssertionError UTR.TerminationCriteria(0)
    @test_throws AssertionError UTR.TerminationCriteria(1, 0)
    @test_throws AssertionError UTR.TerminationCriteria(1, 1, 0.0)

    parameters = UTR.AlgorithmicParameters()
    @test UTR.DEFAULT_PRINT_LEVEL == 0
    @test parameters.print_level == 0
    @test parameters.ρ_0 == 1.0
    @test parameters.ρ_min == 1e-5
    @test parameters.η == 29 / 1200
    @test parameters.ξ == 0.5
    @test parameters.μ_1 == 2.25
    @test parameters.μ_2 == 1.75
    @test parameters.γ_1 == 0.01
    @test parameters.γ_2 == 0.8
    @test parameters.γ_3 == 0.5
    @test parameters.dense_hessian_threshold == 0.2
    @test parameters.reuse_sparse_symbolic_factorization
    @test parameters.use_backup_trust_region_subproblem_solver
    @test parameters.handle_hard_case
    @test parameters.trust_region_subproblem_solver ==
          TrustRegionSubproblemSolvers.DIRECT_NEW_SOLVER
    # The TeX requires both penalties to be positive, but does not require ρ₀ ≥ ρmin.
    @test UTR.AlgorithmicParameters(1e-6, 1e-5).ρ_0 == 1e-6
    @test_throws AssertionError UTR.AlgorithmicParameters(0.0)
    @test_throws AssertionError UTR.AlgorithmicParameters(1.0, 0.0)
    @test_throws AssertionError UTR.AlgorithmicParameters(1.0, 1e-5, 1 / 32)
    @test_throws AssertionError UTR.AlgorithmicParameters(1.0, 1e-5, 0.01, 0.25)
    @test_throws AssertionError UTR.AlgorithmicParameters(
        1.0,
        1e-5,
        0.01,
        0.5,
        1.0,
    )

    parameter_values =
        map(field -> getfield(parameters, field), fieldnames(typeof(parameters)))
    @test length(parameter_values) == 16
    @test UTR.AlgorithmicParameters(parameter_values...).handle_hard_case

    invalid_solver_values = collect(parameter_values)
    invalid_solver_values[end] = "unsupported"
    @test_throws AssertionError UTR.AlgorithmicParameters(invalid_solver_values...)

    for solver in (TrustRegionSubproblemSolvers.DIRECT_NEW_SOLVER,)
        for (field_index, boundary) in (
            (7, 0.0),
            (7, 1.0),
            (8, 0.0),
            (8, 1.0),
            (9, 0.0),
            (9, 1.0),
        )
            boundary_values = collect(parameter_values)
            boundary_values[field_index] = boundary
            boundary_values[end] = solver
            @test_throws AssertionError UTR.AlgorithmicParameters(
                boundary_values...,
            )
        end
    end

    counter = UTR.AlgorithmCounter()
    @test all(field -> getfield(counter, field) == 0, fieldnames(UTR.AlgorithmCounter))
    UTR.recordParameterSelectionFactorizations!(counter, 2)
    @test counter.total_number_factorizations == 2
    @test counter.total_number_factorizations_parameter_selection == 2
    @test counter.total_number_subproblem_iterations == 0
    @test counter.total_number_hessian_vector_products == 0
    @test UTR.assertFactorizationAccounting(counter) === counter
end
