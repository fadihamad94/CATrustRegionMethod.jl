using Test, NLPModels, NLPModelsJuMP, JuMP, LinearAlgebra, DataFrames, SparseArrays

include("../src/CAT_Module.jl")
include("./test_TRS_solver.jl")

function solve_NLP1_starting_at_global_optimum()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()
    x = [1.0, 1.0]
    δ = 0.0
    x, status, iteration_stats, computation_stats =
        consistently_adaptive_trust_region_method.CAT(
            nlp,
            termination_criteria,
            algorithm_params,
            x,
            δ,
        )
    @test x == [1.0, 1.0]
    @test computation_stats["total_function_evaluation"] == nlp.counters.neval_obj
    @test computation_stats["total_gradient_evaluation"] == nlp.counters.neval_grad
    @test computation_stats["total_hessian_evaluation"] == nlp.counters.neval_hess
    @test computation_stats == Dict(
        "total_number_factorizations_compute_search_direction" => 0,
        "total_hessian_evaluation" => 1,
        "total_number_factorizations_findinterval" => 0,
        "total_gradient_evaluation" => 1,
        "total_number_factorizations" => 1,
        "total_number_factorizations_bisection" => 0,
        "total_function_evaluation" => 1,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    @test obj(nlp, x) == 0.0
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveSimpleConvexNLPModel()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_simple_convex_nlp_model()
    x = [0.0, 0.0]
    δ = 0.0
    algorithm_params.r_1 = -1.0
    x, status, iteration_stats, computation_stats =
        consistently_adaptive_trust_region_method.CAT(
            nlp,
            termination_criteria,
            algorithm_params,
            x,
            δ,
        )
    @test norm(x - [0.0, 1.0], 2) <= tol
    @test computation_stats["total_function_evaluation"] == nlp.counters.neval_obj
    @test computation_stats["total_gradient_evaluation"] == nlp.counters.neval_grad
    @test computation_stats["total_hessian_evaluation"] == nlp.counters.neval_hess
    @test computation_stats == Dict(
        "total_number_factorizations_compute_search_direction" => 1,
        "total_hessian_evaluation" => 1,
        "total_number_factorizations_findinterval" => 0,
        "total_gradient_evaluation" => 2,
        "total_number_factorizations" => 1,
        "total_number_factorizations_bisection" => 0,
        "total_function_evaluation" => 2,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    @test norm(obj(nlp, x) - 0, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveComplexConvexNLPModel()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()
    x = [0.0, 0.0]
    δ = 0.0
    x, status, iteration_stats, computation_stats =
        consistently_adaptive_trust_region_method.CAT(
            nlp,
            termination_criteria,
            algorithm_params,
            x,
            δ,
        )
    @test norm(x[1] - 1, 2) <= tol
    @test norm(x[2] - 1, 2) <= tol
    @test computation_stats["total_function_evaluation"] == nlp.counters.neval_obj
    @test computation_stats["total_gradient_evaluation"] == nlp.counters.neval_grad
    @test computation_stats["total_hessian_evaluation"] == nlp.counters.neval_hess
    @test computation_stats == Dict(
        "total_number_factorizations_compute_search_direction" => 19,
        "total_hessian_evaluation" => 12,
        "total_number_factorizations_findinterval" => 16,
        "total_gradient_evaluation" => 13,
        "total_number_factorizations" => 46,
        "total_number_factorizations_bisection" => 11,
        "total_function_evaluation" => 16,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    @test norm(obj(nlp, x) - 0, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveSimpleConvexNLPModelDifferentStartingPoint()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_simple_convex_nlp_model()
    x = [0.1, 0.1]
    δ = 0.0
    algorithm_params.r_1 = -1.0
    x, status, iteration_stats, computation_stats =
        consistently_adaptive_trust_region_method.CAT(
            nlp,
            termination_criteria,
            algorithm_params,
            x,
            δ,
        )
    @test computation_stats["total_function_evaluation"] == nlp.counters.neval_obj
    @test computation_stats["total_gradient_evaluation"] == nlp.counters.neval_grad
    @test computation_stats["total_hessian_evaluation"] == nlp.counters.neval_hess
    @test computation_stats == Dict(
        "total_number_factorizations_compute_search_direction" => 1,
        "total_hessian_evaluation" => 1,
        "total_number_factorizations_findinterval" => 0,
        "total_gradient_evaluation" => 2,
        "total_number_factorizations" => 1,
        "total_number_factorizations_bisection" => 0,
        "total_function_evaluation" => 2,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    @test norm(obj(nlp, x) - 0.0, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveSimpleConvexNLPModelAnotherStartingPoint()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_simple_convex_nlp_model()
    x = [20.01, -10.01]
    δ = 0.0
    algorithm_params.r_1 = -1.0
    x, status, iteration_stats, computation_stats =
        consistently_adaptive_trust_region_method.CAT(
            nlp,
            termination_criteria,
            algorithm_params,
            x,
            δ,
        )
    @test computation_stats["total_function_evaluation"] == nlp.counters.neval_obj
    @test computation_stats["total_gradient_evaluation"] == nlp.counters.neval_grad
    @test computation_stats["total_hessian_evaluation"] == nlp.counters.neval_hess
    @test computation_stats == Dict(
        "total_number_factorizations_compute_search_direction" => 1,
        "total_hessian_evaluation" => 1,
        "total_number_factorizations_findinterval" => 0,
        "total_gradient_evaluation" => 2,
        "total_number_factorizations" => 1,
        "total_number_factorizations_bisection" => 0,
        "total_function_evaluation" => 2,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    @test norm(obj(nlp, x) - 0.0, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveComplexConvexNLP1()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_complex_convex_nlp1_model()
    termination_criteria.MAX_ITERATIONS = 10
    x = [0.0, 0.0]
    δ = 0.0
    x, status, iteration_stats, computation_stats =
        consistently_adaptive_trust_region_method.CAT(
            nlp,
            termination_criteria,
            algorithm_params,
            x,
            δ,
        )
    @test computation_stats["total_function_evaluation"] == nlp.counters.neval_obj
    @test computation_stats["total_gradient_evaluation"] == nlp.counters.neval_grad
    @test computation_stats["total_hessian_evaluation"] == nlp.counters.neval_hess
    @test computation_stats == Dict(
        "total_number_factorizations_compute_search_direction" => 1,
        "total_hessian_evaluation" => 1,
        "total_number_factorizations_findinterval" => 0,
        "total_gradient_evaluation" => 2,
        "total_number_factorizations" => 1,
        "total_number_factorizations_bisection" => 0,
        "total_function_evaluation" => 2,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    @test norm(obj(nlp, x) - 0.750000000125, 2) <= tol
    @test norm(x[1] - 0.33332500000000004, 2) <= tol
    @test norm(x[2] - 0.166665, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveComplexNLPModeL1()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_complex_nlp_modeL1()
    x = [0.0, 0.0]
    δ = 0.0
    x, status, iteration_stats, computation_stats =
        consistently_adaptive_trust_region_method.CAT(
            nlp,
            termination_criteria,
            algorithm_params,
            x,
            δ,
        )
    @test computation_stats["total_function_evaluation"] == nlp.counters.neval_obj
    @test computation_stats["total_gradient_evaluation"] == nlp.counters.neval_grad
    @test computation_stats["total_hessian_evaluation"] == nlp.counters.neval_hess
    @test computation_stats == Dict(
        "total_number_factorizations_compute_search_direction" => 8,
        "total_hessian_evaluation" => 4,
        "total_number_factorizations_findinterval" => 10,
        "total_gradient_evaluation" => 5,
        "total_number_factorizations" => 27,
        "total_number_factorizations_bisection" => 9,
        "total_function_evaluation" => 6,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    @test norm(obj(nlp, x) - 0.183430792966865, 2) <= tol
    @test norm(x[1] - 0.7221896985843893, 2) <= tol
    @test norm(x[2] - (-0.5819243669997765), 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveNLPSinCosModel1()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_problem_sin_cos_mode_nlp1()
    termination_criteria.gradient_termination_tolerance = 2e-2
    x = [0.0, 0.0]
    δ = 0.049
    x, status, iteration_stats, computation_stats =
        consistently_adaptive_trust_region_method.CAT(
            nlp,
            termination_criteria,
            algorithm_params,
            x,
            δ,
        )
    @test computation_stats["total_function_evaluation"] == nlp.counters.neval_obj
    @test computation_stats["total_gradient_evaluation"] == nlp.counters.neval_grad
    @test computation_stats["total_hessian_evaluation"] == nlp.counters.neval_hess
    @test computation_stats == Dict(
        "total_number_factorizations_compute_search_direction" => 5,
        "total_hessian_evaluation" => 3,
        "total_number_factorizations_findinterval" => 9,
        "total_gradient_evaluation" => 4,
        "total_number_factorizations" => 21,
        "total_number_factorizations_bisection" => 7,
        "total_function_evaluation" => 4,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    @test norm(obj(nlp, x) + 1, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveNLPSinCosModel1DifferentStartingPoint()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_problem_sin_cos_mode_nlp1()
    x = [10.0, 0.0]
    δ = 0.0
    x, status, iteration_stats, computation_stats =
        consistently_adaptive_trust_region_method.CAT(
            nlp,
            termination_criteria,
            algorithm_params,
            x,
            δ,
        )
    @test computation_stats["total_function_evaluation"] == nlp.counters.neval_obj
    @test computation_stats["total_gradient_evaluation"] == nlp.counters.neval_grad
    @test computation_stats["total_hessian_evaluation"] == nlp.counters.neval_hess
    @test computation_stats == Dict(
        "total_number_factorizations_compute_search_direction" => 4,
        "total_hessian_evaluation" => 3,
        "total_number_factorizations_findinterval" => 4,
        "total_gradient_evaluation" => 4,
        "total_number_factorizations" => 12,
        "total_number_factorizations_bisection" => 4,
        "total_function_evaluation" => 4,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    @test norm(obj(nlp, x) + 1, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveNLPSinCosModel1DeltaNotZero()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_problem_sin_cos_mode_nlp1()
    x = [0.0, 0.0]
    δ = 1.0
    x, status, iteration_stats, computation_stats =
        consistently_adaptive_trust_region_method.CAT(
            nlp,
            termination_criteria,
            algorithm_params,
            x,
            δ,
        )
    @test computation_stats["total_function_evaluation"] == nlp.counters.neval_obj
    @test computation_stats["total_gradient_evaluation"] == nlp.counters.neval_grad
    @test computation_stats["total_hessian_evaluation"] == nlp.counters.neval_hess
    @test computation_stats == Dict(
        "total_number_factorizations_compute_search_direction" => 6,
        "total_hessian_evaluation" => 4,
        "total_number_factorizations_findinterval" => 7,
        "total_gradient_evaluation" => 5,
        "total_number_factorizations" => 18,
        "total_number_factorizations_bisection" => 5,
        "total_function_evaluation" => 5,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    @test norm(obj(nlp, x) + 1, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveNLPSinCosModel2()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_problem_sin_cos_mode_nlp2()
    termination_criteria.MAX_ITERATIONS = 1000
    x = [10.0, 10.0]
    δ = 1.0
    x, status, iteration_stats, computation_stats =
        consistently_adaptive_trust_region_method.CAT(
            nlp,
            termination_criteria,
            algorithm_params,
            x,
            δ,
        )
    @test computation_stats["total_function_evaluation"] == nlp.counters.neval_obj
    @test computation_stats["total_gradient_evaluation"] == nlp.counters.neval_grad
    @test computation_stats["total_hessian_evaluation"] == nlp.counters.neval_hess
    @test computation_stats == Dict(
        "total_number_factorizations_compute_search_direction" => 5,
        "total_hessian_evaluation" => 4,
        "total_number_factorizations_findinterval" => 3,
        "total_gradient_evaluation" => 5,
        "total_number_factorizations" => 10,
        "total_number_factorizations_bisection" => 2,
        "total_function_evaluation" => 5,
        "total_number_factorizations_inverse_power_iteration" => 0,
    )
    @test norm(obj(nlp, x) - (-2), 2) <= tol
    @test norm(x[1] - 10.995653476776056, 2) <= tol
    @test norm(x[2] - 9.424777960768635, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
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
end

@testset "optimization_CAT_tests" begin
    optimize_models()
end
