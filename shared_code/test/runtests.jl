using CATrustRegionShared
using MathOptInterface
using Test

const MOI = MathOptInterface

module CATLayout
using CATrustRegionShared
using EnumX
const DEFAULT_MAX_ITERATIONS = CATrustRegionShared.DEFAULT_MAX_ITERATIONS
const DEFAULT_GRADIENT_TERMINATION_TOLERANCE =
    CATrustRegionShared.DEFAULT_GRADIENT_TERMINATION_TOLERANCE
const DEFAULT_MAX_TIME = CATrustRegionShared.DEFAULT_MAX_TIME
const DEFAULT_STEP_SIZE_LIMIT = CATrustRegionShared.DEFAULT_STEP_SIZE_LIMIT
const DEFAULT_MINIMUM_OBJECTIVE_FUNCTION =
    CATrustRegionShared.DEFAULT_MINIMUM_OBJECTIVE_FUNCTION
const DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS =
    CATrustRegionShared.DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS
@define_termination_status
@define_termination_criteria :cat
@define_algorithm_counter :cat
end

module UTRLayout
using CATrustRegionShared
using EnumX
const DEFAULT_MAX_ITERATIONS = CATrustRegionShared.DEFAULT_MAX_ITERATIONS
const DEFAULT_MAX_INNER_ITERATIONS = 100
const DEFAULT_GRADIENT_TERMINATION_TOLERANCE =
    CATrustRegionShared.DEFAULT_GRADIENT_TERMINATION_TOLERANCE
const DEFAULT_MAX_TIME = CATrustRegionShared.DEFAULT_MAX_TIME
const DEFAULT_STEP_SIZE_LIMIT = CATrustRegionShared.DEFAULT_STEP_SIZE_LIMIT
const DEFAULT_MINIMUM_OBJECTIVE_FUNCTION =
    CATrustRegionShared.DEFAULT_MINIMUM_OBJECTIVE_FUNCTION
const DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS =
    CATrustRegionShared.DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS
@define_termination_status
@define_termination_criteria :utr
@define_algorithm_counter :utr
end

@testset "CATrustRegionShared" begin
    @test fieldnames(CATLayout.AlgorithmCounter) == (
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
    @test length(fieldnames(UTRLayout.AlgorithmCounter)) == 13
    counter = CATLayout.AlgorithmCounter()
    @test increment!(counter, :total_function_evaluation) == 1
    @test_throws ArgumentError increment!(counter, :not_a_field)
    iterative_stats = (
        native_status = Int64(0),
        iterations = Int64(3),
        hessian_vector_products = Int64(4),
    )
    @test record_iterative_stats!(counter, iterative_stats) === counter
    @test counter.total_number_subproblem_iterations == 3
    @test counter.total_number_hessian_vector_products == 4
    @test_throws ArgumentError record_iterative_stats!(
        counter,
        merge(iterative_stats, (iterations = Int64(-1),)),
    )

    @test CATLayout.TerminationCriteria().MAX_ITERATIONS == 100000
    @test UTRLayout.TerminationCriteria().MAX_INNER_ITERATIONS == 100
    for invalid_value in (NaN, Inf, -Inf)
        @test_throws AssertionError CATLayout.TerminationCriteria(
            100,
            invalid_value,
        )
        @test_throws AssertionError CATLayout.TerminationCriteria(
            100,
            1.0e-5,
            invalid_value,
        )
        @test_throws AssertionError CATLayout.TerminationCriteria(
            100,
            1.0e-5,
            1.0,
            invalid_value,
        )
        @test_throws AssertionError CATLayout.TerminationCriteria(
            100,
            1.0e-5,
            1.0,
            1.0e-16,
            invalid_value,
        )
    end
    @test CATLayout.TerminationStatusCode.T !== UTRLayout.TerminationStatusCode.T
    @test canonical_status_string(CATLayout.TerminationStatusCode.OPTIMAL) ==
          "OPTIMAL"
    @test canonical_status_string(UTRLayout.TerminationStatusCode.MEMORY_LIMIT) ==
          "MEMORY_LIMIT"
    @test canonical_status_string(
        UTRLayout.TerminationStatusCode.MEMORY_LIMIT;
        memory_limit = "OUT_OF_MEMORY",
    ) ==
          "OUT_OF_MEMORY"

    evaluator = EmptyNLPEvaluator()
    gradient = [1.0, 1.0]
    MOI.eval_objective_gradient(evaluator, gradient, zeros(2))
    @test gradient == zeros(2)
    @test MOI.features_available(evaluator) == [:Grad, :Hess]
    @test format_to_six_decimals(1 / 3) == 0.333333
end
