using Test,
    NLPModels,
    NLPModelsJuMP,
    JuMP,
    LinearAlgebra,
    DataFrames,
    SparseArrays,
    JSON

import CATrustRegionMethod
import TrustRegionSubproblemSolvers

const EXPECTED_ITERATION_HISTORY_COLUMNS = [
    "k",
    "fval",
    "gradnorm",
    "min_gradnorm_fval",
    "min_gradnorm",
]

function assert_iteration_history_contract(iteration_stats::DataFrame)
    @test names(iteration_stats) == EXPECTED_ITERATION_HISTORY_COLUMNS
    @test eltype(iteration_stats.k) == Int64
    for column in EXPECTED_ITERATION_HISTORY_COLUMNS[2:end]
        @test eltype(iteration_stats[!, column]) == Float64
    end
    @test all(isfinite, iteration_stats.fval)
    @test all(isfinite, iteration_stats.gradnorm)
    @test all(isfinite, iteration_stats.min_gradnorm_fval)
    @test all(isfinite, iteration_stats.min_gradnorm)
    @test all(iteration_stats.min_gradnorm .<= iteration_stats.gradnorm)
    @test issorted(iteration_stats.min_gradnorm; rev = true)
end

function solve_NLP1_starting_at_global_optimum()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()
    x = [1.0, 1.0]
    δ = 0.0
    x, status, iteration_stats, algorithm_counter =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, x, δ)
    @test x == [1.0, 1.0]
    assert_iteration_history_contract(iteration_stats)
    @test iteration_stats[1, :fval] ==
          iteration_stats[1, :min_gradnorm_fval]
    @test iteration_stats[1, :gradnorm] ==
          iteration_stats[1, :min_gradnorm]
    assert_oracle_counters_match(algorithm_counter, nlp)

    computation_stats = Dict(
        "total_number_factorizations_compute_search_direction" => 0,
        "total_hessian_evaluation" => 1,
        "total_number_factorizations_findinterval" => 0,
        "total_gradient_evaluation" => 1,
        "total_number_factorizations" => 0,
        "total_number_factorizations_bisection" => 0,
        "total_function_evaluation" => 1,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    assert_counter_matches(algorithm_counter, computation_stats)

    @test obj(nlp, x) == 0.0
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
end

function solveSimpleConvexNLPModel()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_simple_convex_nlp_model()
    x = [0.0, 0.0]
    δ = 0.0
    algorithm_params.r_1 = -1.0
    x, status, iteration_stats, algorithm_counter =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, x, δ)
    @test norm(x - [0.0, 1.0], 2) <= tol
    assert_iteration_history_contract(iteration_stats)
    assert_oracle_counters_match(algorithm_counter, nlp)

    computation_stats = Dict(
        "total_number_factorizations_compute_search_direction" => 1,
        "total_hessian_evaluation" => 1,
        "total_number_factorizations_findinterval" => 0,
        "total_gradient_evaluation" => 2,
        "total_number_factorizations" => 1,
        "total_number_factorizations_bisection" => 0,
        "total_function_evaluation" => 2,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    assert_counter_matches(algorithm_counter, computation_stats)

    @test norm(obj(nlp, x) - 0, 2) <= tol
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
end

function solveComplexConvexNLPModel()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()
    x = [0.0, 0.0]
    δ = 0.0
    x, status, iteration_stats, algorithm_counter =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, x, δ)
    @test norm(x[1] - 1, 2) <= tol
    @test norm(x[2] - 1, 2) <= tol
    assert_oracle_counters_match(algorithm_counter, nlp)

    @test algorithm_counter.total_function_evaluation <= 20
    @test algorithm_counter.total_gradient_evaluation <= 20
    @test algorithm_counter.total_hessian_evaluation <= 15
    @test algorithm_counter.total_number_factorizations <= 55

    @test norm(obj(nlp, x) - 0, 2) <= tol
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
end

function solveSimpleConvexNLPModelDifferentStartingPoint()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_simple_convex_nlp_model()
    x = [0.1, 0.1]
    δ = 0.0
    algorithm_params.r_1 = -1.0
    x, status, iteration_stats, algorithm_counter =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, x, δ)
    assert_oracle_counters_match(algorithm_counter, nlp)

    computation_stats = Dict(
        "total_number_factorizations_compute_search_direction" => 1,
        "total_hessian_evaluation" => 1,
        "total_number_factorizations_findinterval" => 0,
        "total_gradient_evaluation" => 2,
        "total_number_factorizations" => 1,
        "total_number_factorizations_bisection" => 0,
        "total_function_evaluation" => 2,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    assert_counter_matches(algorithm_counter, computation_stats)
    @test norm(obj(nlp, x) - 0.0, 2) <= tol
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
end

function solveSimpleConvexNLPModelAnotherStartingPoint()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_simple_convex_nlp_model()
    x = [20.01, -10.01]
    δ = 0.0
    algorithm_params.r_1 = -1.0
    x, status, iteration_stats, algorithm_counter =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, x, δ)
    assert_oracle_counters_match(algorithm_counter, nlp)

    computation_stats = Dict(
        "total_number_factorizations_compute_search_direction" => 1,
        "total_hessian_evaluation" => 1,
        "total_number_factorizations_findinterval" => 0,
        "total_gradient_evaluation" => 2,
        "total_number_factorizations" => 1,
        "total_number_factorizations_bisection" => 0,
        "total_function_evaluation" => 2,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    assert_counter_matches(algorithm_counter, computation_stats)

    @test norm(obj(nlp, x) - 0.0, 2) <= tol
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
end

function solveComplexConvexNLP1()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_complex_convex_nlp1_model()
    termination_criteria.MAX_ITERATIONS = 10
    x = [0.0, 0.0]
    δ = 0.0
    x, status, iteration_stats, algorithm_counter =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, x, δ)
    assert_oracle_counters_match(algorithm_counter, nlp)

    computation_stats = Dict(
        "total_number_factorizations_compute_search_direction" => 1,
        "total_hessian_evaluation" => 1,
        "total_number_factorizations_findinterval" => 0,
        "total_gradient_evaluation" => 2,
        "total_number_factorizations" => 1,
        "total_number_factorizations_bisection" => 0,
        "total_function_evaluation" => 2,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    assert_counter_matches(algorithm_counter, computation_stats)

    @test norm(obj(nlp, x) - 0.750000000125, 2) <= tol
    @test norm(x[1] - 0.33332500000000004, 2) <= tol
    @test norm(x[2] - 0.166665, 2) <= tol
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
end

function solveComplexNLPModeL1()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_complex_nlp_modeL1()
    x = [0.0, 0.0]
    δ = 0.0
    x, status, iteration_stats, algorithm_counter =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, x, δ)
    assert_oracle_counters_match(algorithm_counter, nlp)

    @test algorithm_counter.total_function_evaluation <= 10
    @test algorithm_counter.total_gradient_evaluation <= 10
    @test algorithm_counter.total_hessian_evaluation <= 10
    @test algorithm_counter.total_number_factorizations <= 35

    @test norm(obj(nlp, x) - 0.183430792966865, 2) <= tol
    @test norm(x[1] - 0.7221896985843893, 2) <= tol
    @test norm(x[2] - (-0.5819243669997765), 2) <= tol
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
end

function solveNLPSinCosModel1()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_problem_sin_cos_mode_nlp1()
    termination_criteria.gradient_termination_tolerance = 2e-2
    x = [0.0, 0.0]
    δ = 0.049
    x, status, iteration_stats, algorithm_counter =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, x, δ)
    assert_oracle_counters_match(algorithm_counter, nlp)

    @test algorithm_counter.total_function_evaluation <= 5
    @test algorithm_counter.total_gradient_evaluation <= 5
    @test algorithm_counter.total_hessian_evaluation <= 5
    @test algorithm_counter.total_number_factorizations <= 35

    @test norm(obj(nlp, x) + 1, 2) <= tol
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
end

function solveNLPSinCosModel1DifferentStartingPoint()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_problem_sin_cos_mode_nlp1()
    x = [10.0, 0.0]
    δ = 0.0
    x, status, iteration_stats, algorithm_counter =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, x, δ)
    assert_oracle_counters_match(algorithm_counter, nlp)

    @test algorithm_counter.total_function_evaluation <= 5
    @test algorithm_counter.total_gradient_evaluation <= 5
    @test algorithm_counter.total_hessian_evaluation <= 5
    @test algorithm_counter.total_number_factorizations <= 15

    @test norm(obj(nlp, x) + 1, 2) <= tol
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
end

function solveNLPSinCosModel1DeltaNotZero()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_problem_sin_cos_mode_nlp1()
    x = [0.0, 0.0]
    δ = 1.0
    x, status, iteration_stats, algorithm_counter =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, x, δ)
    assert_oracle_counters_match(algorithm_counter, nlp)

    @test algorithm_counter.total_function_evaluation <= 10
    @test algorithm_counter.total_gradient_evaluation <= 10
    @test algorithm_counter.total_hessian_evaluation <= 10
    @test algorithm_counter.total_number_factorizations <= 25

    @test norm(obj(nlp, x) + 1, 2) <= tol
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
end

function solveNLPSinCosModel2()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_problem_sin_cos_mode_nlp2()
    termination_criteria.MAX_ITERATIONS = 1000
    x = [10.0, 10.0]
    δ = 1.0
    x, status, iteration_stats, algorithm_counter =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, x, δ)
    assert_oracle_counters_match(algorithm_counter, nlp)

    @test algorithm_counter.total_function_evaluation <= 10
    @test algorithm_counter.total_gradient_evaluation <= 10
    @test algorithm_counter.total_hessian_evaluation <= 10
    @test algorithm_counter.total_number_factorizations <= 15

    @test norm(obj(nlp, x) - (-2), 2) <= tol
    @test norm(x[1] - 10.995653476776056, 2) <= tol
    @test norm(x[2] - 9.424777960768635, 2) <= tol
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
end

function rejected_trial_returns_best_gradient_witness()
    model = Model()
    @variable(model, x, start = 0.0)
    @NLobjective(
        model,
        Min,
        -1.500000003 * x^4 + 2.000000004 * x^3 + 0.5 * x^2 - x,
    )
    nlp = MathOptNLPModel(model)
    termination_criteria = CATrustRegionMethod.TerminationCriteria()
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters()
    algorithm_params.print_level = -1

    current_x,
    status,
    iteration_stats,
    _,
    _,
    _,
    best_gradient_x = CATrustRegionMethod.optimize(
        nlp,
        termination_criteria,
        algorithm_params,
        [0.0],
        0.0,
    )

    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
    @test current_x ≈ [0.0] atol = 1.0e-12
    @test best_gradient_x ≈ [1.0] atol = 1.0e-12
    @test iteration_stats[end, :fval] ≈ 0.0 atol = 1.0e-12
    @test iteration_stats[end, :gradnorm] ≈ 1.0 atol = 1.0e-12
    @test iteration_stats[end, :min_gradnorm_fval] ≈ 1.0e-9 atol = 1.0e-12
    @test iteration_stats[end, :min_gradnorm] <=
          termination_criteria.gradient_termination_tolerance
end

function optimize_models()
    #--------------------------------------------------------------
    #-----TESTING SOLVING NLP STARTING AT THE GLOBAL MINIMIZER-----
    #--------------------------------------------------------------
    solve_NLP1_starting_at_global_optimum()
    #---------------------------------------------------------------
    #------------TESTING SOLVING SIMPLE CONVEX NLP MODEL------------
    #---------------------------------------------------------------
    solveSimpleConvexNLPModel()
    #----------------------------------------------------------------
    #------------TESTING SOLVING COMPLEX CONVEX NLP MODEL------------
    #----------------------------------------------------------------
    solveComplexConvexNLPModel()
    #-------------------------------------------------------------------
    #-TESTING SOLVING COMPLEX CONVEX NLP MODEL DIFFERENT STARTING POINT-
    #-------------------------------------------------------------------
    solveSimpleConvexNLPModelDifferentStartingPoint()
    #-------------------------------------------------------------------
    #--TESTING SOLVING COMPLEX CONVEX NLP MODEL ANOTHER STARTING POINT--
    #-------------------------------------------------------------------
    solveSimpleConvexNLPModelAnotherStartingPoint()
    #----------------------------------------------------------------
    #------------TESTING SOLVING COMPLEX CONVEX NLP MODEL------------
    #----------------------------------------------------------------
    solveComplexConvexNLP1()
    #---------------------------------------------------------
    #------------TESTING SOLVING COMPLEX NLP MODEL------------
    #---------------------------------------------------------
    solveComplexNLPModeL1()
    #---------------------------------------------------------
    #------------TESTING SOLVING SIN COS NLP MODEL------------
    #---------------------------------------------------------
    solveNLPSinCosModel1()
    #-------------------------------------------------------------
    #-TESTING SOLVING SIN COS NLP MODEL  DIFFERENT STARTING POINT-
    #-------------------------------------------------------------
    solveNLPSinCosModel1DifferentStartingPoint()
    #----------------------------------------------------------
    #-----TESTING SOLVING SIN COS NLP MODEL DELTA NOT ZERO-----
    #----------------------------------------------------------
    solveNLPSinCosModel1DeltaNotZero()
    #---------------------------------------------------------
    #------------TESTING SOLVING SIN COS NLP MODEL------------
    #---------------------------------------------------------
    solveNLPSinCosModel2()
    rejected_trial_returns_best_gradient_witness()
end

@testset "optimization_CAT_tests" begin
    optimize_models()
end
