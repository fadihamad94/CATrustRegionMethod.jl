using Test, NLPModels, NLPModelsJuMP, JuMP, LinearAlgebra, DataFrames, SparseArrays

include("../src/CAT_Module.jl")
include("./test_TRS_solver.jl")

function solve_NLP1_starting_at_global_optimum()
    problem = test_create_dummy_problem()
    x = [1.0, 1.0]
    δ = 0.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test x == [1.0, 1.0]
    @test obj(problem.nlp, x) == 0.0
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveSimpleConvexNLPModel()
    tol = 1e-3
    problem = test_create_simple_convex_nlp_model()
    x = [0.0, 0.0]
    δ = 0.0
    problem.initial_radius_struct.r_1 = -1.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(x - [0.0, 1.0], 2) <= tol
    @test norm(obj(problem.nlp, x) - 0, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveComplexConvexNLPModel()
    tol = 1e-3
    problem = test_create_dummy_problem()
    x = [0.0, 0.0]
    δ = 0.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(x[1] - 1, 2) <= tol
    @test norm(x[2] - 1, 2) <= tol
    @test norm(obj(problem.nlp, x) - 0, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveSimpleConvexNLPModelDifferentStartingPoint()
    tol = 1e-3
    problem = test_create_simple_convex_nlp_model()
    x = [0.1, 0.1]
    δ = 0.0
    problem.initial_radius_struct.r_1 = -1.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) - 0.0, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveSimpleConvexNLPModelAnotherStartingPoint()
    tol = 1e-3
    problem = test_create_simple_convex_nlp_model()
    x = [20.01, -10.01]
    δ = 0.0
    problem.initial_radius_struct.r_1 = -1.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) - 0.0, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveComplexConvexNLP1()
    tol = 1e-3
    problem = test_create_complex_convex_nlp1_model()
    problem.termination_conditions_struct.MAX_ITERATIONS = 10
    x = [0.0, 0.0]
    δ = 0.0
    x, status , iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) - 0.750000000125, 2) <= tol
    @test norm(x[1] - 0.33332500000000004, 2) <= tol
    @test norm(x[2] - 0.166665, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveComplexNLPModeL1()
    tol = 1e-3
    problem = test_create_complex_nlp_modeL1()
    x = [0.0, 0.0]
    δ = 0.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) - 0.183430792966865, 2) <= tol
    @test norm(x[1] - 0.7221896985843893, 2) <= tol
    @test norm(x[2] - (-0.5819243669997765), 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveNLPSinCosModel1()
    tol = 1e-3
    problem = test_create_problem_sin_cos_mode_nlp1()
    problem.termination_conditions_struct.gradient_termination_tolerance = 2e-2
    x = [0.0, 0.0]
    δ = 0.049
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) + 1, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveNLPSinCosModel1DifferentStartingPoint()
    tol = 1e-3
    problem = test_create_problem_sin_cos_mode_nlp1()
    x = [10.0, 0.0]
    δ = 0.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) + 1, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveNLPSinCosModel1DeltaNotZero()
    tol = 1e-3
    problem = test_create_problem_sin_cos_mode_nlp1()
    x = [0.0, 0.0]
    δ = 1.0
    x, status = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) + 1, 2) <= tol
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
end

function solveNLPSinCosModel2()
    tol = 1e-3
    problem = test_create_problem_sin_cos_mode_nlp2()
    problem.termination_conditions_struct.MAX_ITERATIONS = 1000
    x = [10.0, 10.0]
    δ = 1.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) - (-2), 2) <= tol
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
