using Test, NLPModels, NLPModelsJuMP, JuMP, LinearAlgebra

include("../src/trust_region_subproblem_solver.jl")

#Functions to create NLP models

function createDummyNLPModel()
    x0 = [-1.2; 1.0]
    model = Model()
    @variable(model, x[i=1:2], start=x0[i])
    @NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createDummyNLPModel2()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (2 * x + y - 1) ^ 2 + x + y + (x ^ 2 - 2 * y ^ 2) ^ 3)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createSimpleUnivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @NLobjective(model, Min, (x - 1) ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createSimpleConvexNLPModeL()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (x + y - 1) ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createComplexConvexNLPModeL1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (x + y - 1) ^ 2 + x + y + (x - 2 * y) ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createComplexNLPModeL1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (2 * x + y - 1) ^ 2 + x + y + (x ^ 2 - 2 * y ^ 2) ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createSinCosNLPModeL1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, sin(x) * cos(y))
    nlp = MathOptNLPModel(model)
    return nlp
end

function createSinCosNLPModeL2()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, sin(x) + cos(y))
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleUnivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @NLobjective(model, Min, -x ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, -x ^ 2 - y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, -x ^ 2 - 2 * y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem2()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x ^ 2 + 0.01 * x - y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem3()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x ^ 2 - 10 * x * y + y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end


function createSimpleUnivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @NLobjective(model, Min, (x - 1) ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleUnivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @NLobjective(model, Min, -x ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, -x ^ 2 - y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, -x ^ 2 - 2 * y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem2()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x ^ 2 + 0.01 * x - y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem3()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x ^ 2 - 10 * x * y + y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function test_create_dummy_problem()
    nlp = createDummyNLPModel()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(0.5)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_dummy_problem2()
    nlp = createDummyNLPModel2()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(0.5)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_simple_convex_nlp_model()
    nlp = createSimpleConvexNLPModeL()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(0.5)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_complex_convex_nlp1_model()
    nlp = createComplexConvexNLPModeL1()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(0.5)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_complex_nlp_modeL1()
    nlp = createComplexNLPModeL1()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(0.5)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_problem_sin_cos_mode_nlp1()
    nlp = createSinCosNLPModeL1()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(0.5)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_problem_sin_cos_mode_nlp2()
    nlp = createSinCosNLPModeL2()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(0.5)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_simple_univariate_convex_model()
    nlp = createSimpleUnivariateConvexProblem()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(0.5)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_simple_univariate_convex_model_solved_same_as_Newton()
    nlp = createSimpleUnivariateConvexProblem()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(2.0)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_hard_case_using_simple_univariate_convex_model()
    nlp = createHardCaseUsingSimpleUnivariateConvexProblem()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(1.0)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_hard_case_using_simple_bivariate_convex_model()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(1.0)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_hard_case_using_bivariate_convex_model_1()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem1()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(1.0)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_hard_case_using_bivariate_convex_model_2()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem2()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(1.0)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

function test_create_hard_case_using_bivariate_convex_model_3()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem3()
    termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(100, 1e-4)
    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(5.0)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, 0.25, 0.5, 2.0, 2.0)
    return problem
end

#Unit test optimize second order model function
function test_optimize_second_order_model_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = [1.0, 1.0]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test d_k == [-0.0, -0.0]
    @test δ_k == δ
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_phi_zero()
    tol = 1e-3
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = nlp.meta.x0
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test γ_2 * r <= norm(d_k) <= r
    @test norm(d_k - [0.106, 0.139], 2) <= tol
    @test abs(δ_k - 64.0) <= tol
    @test abs(norm(d_k, 2) - r) <= γ_2
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_phi_δ_positive_phi_δ_prime_negative()
    tol = 1e-3
    problem = test_create_dummy_problem2()
    nlp = problem.nlp
    x_k = [0.0, 1.0]
    δ = 250.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test γ_2 * r <= norm(d_k) <= r
    @test norm(d_k - [-0.0032, 0.179], 2) <= tol
    @test abs(δ_k - 500.0) <= tol
    @test abs(norm(d_k) - r) <= γ_2
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_for_simple_univariate_convex_model()
    tol = 1e-3
    problem = test_create_simple_univariate_convex_model()
    nlp = problem.nlp
    x_k = [0.0]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 0.5
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test γ_2 * r <= norm(d_k) <= r
    # @test norm((x_k + d_k) - [0.5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    # @test norm(obj(nlp, x_k + d_k) - 0.25, 2) <= tol
end

function test_optimize_second_order_model_for_simple_univariate_convex_model_solved_same_as_Newton()
    tol = 1e-3
    problem = test_create_simple_univariate_convex_model_solved_same_as_Newton()
    nlp = problem.nlp
    x_k = [0.0]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 2.0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test δ_k == 0.0
    @test norm((H + δ_k * I) \ g, 2) <= r
    @test norm(d_k) <= r
    @test norm((x_k + d_k) - [1.0], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - 0, 2) <= tol
end

function test_optimize_second_order_model_for_simple_bivariate_convex_model()
    tol = 1e-3
    problem = test_create_simple_convex_nlp_model()
    nlp = problem.nlp
    x_k = [0.0, 0.0]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 0.5
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(norm((H + δ_k * I) \ g, 2) - r) <= γ_2
    # @test norm((x_k + d_k) - [0.333, 0.333], 2) <= tol
    @test δ_k == 2.5
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function  test_optimize_second_order_model_hard_case_using_simple_univariate_convex_model()
    tol = 1e-3
    problem = test_create_hard_case_using_simple_univariate_convex_model()
    nlp = problem.nlp
    # x_k = [0.01]
    x_k = [1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 1 - 1e-5
    # r = 2.0
    r = 0.0002
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    temp_ = norm(g)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(δ_k == 2.1) <= tol
    @test norm((x_k + d_k) - [0.00021], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - (-4.00004e-8)) <= tol
end

function  test_optimize_second_order_model_hard_case_using_simple_bivariate_convex_model()
    tol = 1e-3
    problem = test_create_hard_case_using_simple_bivariate_convex_model()
    nlp = problem.nlp
    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 1 - 1e-5
    r = 0.00029
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(δ_k - 2.097) <= tol
    @test norm((x_k + d_k) - [2.00001e-5, 2.00001e-5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - (-8.000079e-8), 2) <= tol
end

function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_1()
    tol = 1e-3
    problem = test_create_hard_case_using_bivariate_convex_model_1()
    nlp = problem.nlp
    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 1 - 1e-5
    r = 4.0e-4
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(norm(d_k) - r) <= tol
    @test abs(δ_k - 4.1) <= tol
    @test norm((x_k + d_k) - [2e-5, 4.00001e-5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - (-32.004e-8)) <= tol
end

function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_2()
    tol = 1e-2
    problem = test_create_hard_case_using_bivariate_convex_model_2()
    nlp = problem.nlp
    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 1 - 1e-5
    r = 0.00245
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test γ_2 * r <= norm(d_k) <= r
    @test norm(d_k, 2) - r <= tol
    @test abs(δ_k - 2.102) <= tol
    @test norm((x_k + d_k) - [-0.0025, 2.00001e-5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - (-1.8459e-5)) <= tol
end

function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_3()
    tol = 1e-3
    problem = test_create_hard_case_using_bivariate_convex_model_3()
    nlp = problem.nlp
    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 1 - 1e-5
    r = 0.00114
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(δ_k - 8.099) <= tol
    @test norm((x_k + d_k) - [8.1e-4, 8.1e-4], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - (-5.2488e-6)) <= tol
end

function test_optimize_second_order_model_bisection_logic_bug_fix()
    tol = 1e-3
    r = 0.08343452704764227
    g = [3.4679032978601754e-9, 7.39587593251434e-9, 2.7183407851072428e-8, -0.003000483357027406, 0.008419134290306829]
    H = [66.0 65.0743102550725 65.67263629092243 -1.2900082661017744e7 3.6178787247129455e7; 65.0743102550725 64.1661603269859 64.75315415279051 -1.265594646809823e7 3.549303722643331e7; 65.67263629092243 64.75315415279051 65.34746955501134 -1.2813679600892197e7 3.59360895721625e7; -1.2900082661017744e7 -1.265594646809823e7 -1.2813679600892197e7 3.3981511777610044e12 -9.544912385973707e12; 3.6178787247129455e7 3.549303722643331e7 3.59360895721625e7 -9.544912385973707e12 2.6810622545473246e13]
    γ_1 = 0.01
    γ_2 = 1 - 1e-8
    δ = 6.205227748467783e-12

    success, δ, δ_prime, temp_total_number_factorizations = consistently_adaptive_trust_region_method.findinterval(g, H, δ, γ_2, r)
    @test success
    @test abs(δ - 2.5e-8) <= tol
    @test abs(δ_prime - 0.0133096) <= tol
    min_grad = norm(g, 2)
    success, δ_m, temp_total_number_factorizations = consistently_adaptive_trust_region_method.bisection(g, H, δ, γ_1, γ_2, δ_prime, r, min_grad, 0)
    @test success
    @test abs(δ_m - 5.173e-7) <= tol

    r = 0.0018
    γ_2 = 1 - 1e-5
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, γ_1, γ_2, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(δ_k - 5.173e-7) <= tol
    @test abs(δ_k - 1e-5) <= γ_2 && abs(norm(d_k, 2) - r) <= γ_2
end

function test_optimize_second_order_model_bisection_failure_non_hard_case()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem3()
    x = [3.0, 2.0]
    g = grad(nlp, x)
    H = hess(nlp, x)
    r = 1e-8
    δ = 1e-10
    γ_2 = 1 - 1e-5
    print_level = 0
    status, δ_k, d_k, temp_total_number_factorizations, hard_case = optimizeSecondOrderModel(g, H, δ, γ_2, r, norm(g),print_level)
    @test status == false
    @test norm(d_k) == 0.0
    @test hard_case == true
end

function optimize_models()
    test_optimize_second_order_model_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    test_optimize_second_order_model_phi_zero()
    test_optimize_second_order_model_phi_δ_positive_phi_δ_prime_negative()
    test_optimize_second_order_model_for_simple_univariate_convex_model()
    test_optimize_second_order_model_for_simple_univariate_convex_model_solved_same_as_Newton()
    test_optimize_second_order_model_for_simple_bivariate_convex_model()
    test_optimize_second_order_model_hard_case_using_simple_univariate_convex_model()
    test_optimize_second_order_model_hard_case_using_simple_bivariate_convex_model()
    test_optimize_second_order_model_hard_case_using_bivariate_convex_model_1()
    test_optimize_second_order_model_hard_case_using_bivariate_convex_model_2()
    test_optimize_second_order_model_hard_case_using_bivariate_convex_model_3()
    test_optimize_second_order_model_bisection_logic_bug_fix()
    test_optimize_second_order_model_bisection_failure_non_hard_case()
end

function test_findMinimumEigenValue_example_1()
    H = sparse([10.0 6.0 2.0; 6.0 4.0 2.0; 2.0 2.0 0.0])
    δ = 3.0
    success, eigenvalue, eigenvector, itr = consistently_adaptive_trust_region_method.findMinimumEigenValue(H, δ)
    @test success
    @test abs(eigenvalue - eigmin(Matrix(H))) <= 1e-1
end

function test_findMinimumEigenValue_example_2()
    H = sparse([10.0 6.0 2.0; 6.0 4.0 2.0; 2.0 2.0 0.0])
    δ = 4.0
    success, eigenvalue, eigenvector, itr = consistently_adaptive_trust_region_method.findMinimumEigenValue(H, δ)
    @test success
    @test abs(eigenvalue - eigmin(Matrix(H))) <= 1e-1
end

function test_findMinimumEigenValue_example_3()
    H = sparse([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0])
    δ = 0.0
    success, eigenvalue, eigenvector, itr = consistently_adaptive_trust_region_method.findMinimumEigenValue(H, δ)
    @test success
    @test abs(eigenvalue - eigmin(Matrix(H))) <= 1e-1
end

function test_findMinimumEigenValue_example_4()
    H = sparse([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0])
    δ = -1.0
    success, eigenvalue, eigenvector, itr = consistently_adaptive_trust_region_method.findMinimumEigenValue(H, δ)
    @test success
    @test abs(eigenvalue - eigmin(Matrix(H))) <= 1e-1
end

function test_findMinimumEigenValue_example_5()
    H = sparse([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0])
    δ = 3.4
    success, eigenvalue, eigenvector, itr = consistently_adaptive_trust_region_method.findMinimumEigenValue(H, δ, ϵ = 1e-4)
    @test success
    @show eigenvalue
    @test abs(eigenvalue - eigmin(Matrix(H))) <= 1e-1
end

function test_findMinimumEigenValue_example_6()
    H = sparse([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0])
    δ = 5.0
    success, eigenvalue, eigenvector, itr = consistently_adaptive_trust_region_method.findMinimumEigenValue(H, δ, ϵ = 1e-5)
    @test success
    @test abs(eigenvalue - eigmin(Matrix(H))) <= 1e-1
end


function findMinimumEigenValue()
    test_findMinimumEigenValue_example_1()
    test_findMinimumEigenValue_example_2()
    test_findMinimumEigenValue_example_3()
    test_findMinimumEigenValue_example_4()
    test_findMinimumEigenValue_example_5()
    test_findMinimumEigenValue_example_6()
end

@testset "TRS_Solver_Tests" begin
    optimize_models()
end
#
# @testset "findMinimumEigenValue" begin
#     findMinimumEigenValue()
# end
