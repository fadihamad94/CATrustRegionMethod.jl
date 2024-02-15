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
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 0.5, 100, 1e-4)
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_dummy_problem2()
    nlp = createDummyNLPModel2()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_simple_convex_nlp_model()
    nlp = createSimpleConvexNLPModeL()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_complex_convex_nlp1_model()
    nlp = createComplexConvexNLPModeL1()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_complex_nlp_modeL1()
    nlp = createComplexNLPModeL1()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_problem_sin_cos_mode_nlp1()
    nlp = createSinCosNLPModeL1()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_problem_sin_cos_mode_nlp2()
    nlp = createSinCosNLPModeL2()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_simple_univariate_convex_model()
    nlp = createSimpleUnivariateConvexProblem()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_simple_univariate_convex_model_solved_same_as_Newton()
    nlp = createSimpleUnivariateConvexProblem()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 2.0, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 2.0, 100, 1e-4)
    return problem
end

function test_create_hard_case_using_simple_univariate_convex_model()
    nlp = createHardCaseUsingSimpleUnivariateConvexProblem()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 1.00, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 1.0, 100, 1e-4)
    return problem
end

function test_create_hard_case_using_simple_bivariate_convex_model()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 1.00, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 1.0, 100, 1e-4)
    return problem
end

function test_create_hard_case_using_bivariate_convex_model_1()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem1()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 1.00, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 1.0, 100, 1e-4)
    return problem
end

function test_create_hard_case_using_bivariate_convex_model_2()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem2()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 1.00, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 1.0, 100, 1e-4)
    return problem
end

function test_create_hard_case_using_bivariate_convex_model_3()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem3()
    # problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 5.00, 100, 1e-4)
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.25, 0.5, 2.0, 2.0, 5.0, 100, 1e-4)
    return problem
end

#Unit test optimize second order model function
function test_optimize_second_order_model_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = [1.0, 1.0]
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    success, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g))
    @test success
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
    ϵ = 0.2
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g))
    @test norm(d_k - [0.09089549448495053, 0.1912507010656195], 2) <= tol
    @test norm(δ_k - 32.0, 2) <= tol
    @test abs(norm(d_k, 2) - r) <= ϵ
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_phi_δ_positive_phi_δ_prime_negative()
    tol = 1e-3
    problem = test_create_dummy_problem2()
    nlp = problem.nlp
    x_k = [0.0, 1.0]
    δ = 250.0
    ϵ = 0.2
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g))
    #These results when using δ_m = (δ + δ') / 2
    # @test norm(d_k - [-0.0033190803461683517, 0.18495510723928077], 2) <= tol
    # @test norm(δ_k - 492.1875, 2) <= tol
    #These results when using δ_m = sqrt(δ + δ')
    @test norm(d_k - [-0.003486, 0.19523], 2) <= tol
    @test norm(δ_k - 478.8016, 2) <= tol
    @test abs(norm(d_k) - r) <= ϵ
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_for_simple_univariate_convex_model()
    tol = 1e-3
    problem = test_create_simple_univariate_convex_model()
    nlp = problem.nlp
    x_k = [0.0]
    δ = 0.0
    ϵ = 0.2
    r = 0.5
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g))
    @test ϵ * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test ϵ * r <= norm(d_k) <= r
    @test norm((x_k + d_k) - [0.5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - 0.25, 2) <= tol
end

function test_optimize_second_order_model_for_simple_univariate_convex_model_solved_same_as_Newton()
    tol = 1e-3
    problem = test_create_simple_univariate_convex_model_solved_same_as_Newton()
    nlp = problem.nlp
    x_k = [0.0]
    δ = 0.0
    ϵ = 0.2
    r = 2.0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g))
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
    ϵ = 0.2
    r = 0.5
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g))
    @test abs(norm((H + δ_k * I) \ g, 2) - r) <= ϵ
    @test (1 - ϵ) * r <= norm(d_k) <= (1 + ϵ) * r
    @test norm((x_k + d_k) - [0.4, 0.4], 2) <= tol
    @test δ_k == 1.0
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - 0.04) <= tol
end

#Ceck this later
function  test_optimize_second_order_model_hard_case_using_simple_univariate_convex_model()
    tol = 1e-3
    problem = test_create_hard_case_using_simple_univariate_convex_model()
    nlp = problem.nlp
    # x_k = [0.01]
    x_k = [1e-5]
    δ = 0.0
    ϵ = 1e-5
    # r = 2.0
    r = 0.0002
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    temp_ = norm(g)
    @show "norm(g) is $temp_"
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g))
    @test status
    @test norm(d_k) <= r
    #OLD HARD CASE LOGIC
    @test δ_k == 2.0
    # @test norm((x_k + d_k) - [1.0], 2) <= tol
    @test norm((x_k + d_k) - [0.00021], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    # @test abs(obj(nlp, x_k + d_k) - (-1)) <= tol
    @test abs(obj(nlp, x_k + d_k) - (-4.00004e-8)) <= tol
end

#Check this
function  test_optimize_second_order_model_hard_case_using_simple_bivariate_convex_model()
    tol = 1e-3
    problem = test_create_hard_case_using_simple_bivariate_convex_model()
    nlp = problem.nlp
    x_k = [1e-5, 1e-5]
    δ = 0.0
    ϵ = 1e-5
    r = 0.00029
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g))
    @test status
    @test norm(d_k) <= r
    @test abs(δ_k - 2.0) <= tol
    # @test norm((x_k + d_k) - [1.0, 0.0], 2) <= tol
    # @test norm((x_k + d_k) - [0.0, 0.0], 2) <= tol
    @test norm((x_k + d_k) - [2.00001e-5, 2.00001e-5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    # @test norm(obj(nlp, x_k + d_k) - (-1), 2) <= tol
    @test norm(obj(nlp, x_k + d_k) - (-8.000079e-8), 2) <= tol
end

# Check this
function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_1()
    tol = 1e-3
    problem = test_create_hard_case_using_bivariate_convex_model_1()
    nlp = problem.nlp
    x_k = [1e-5, 1e-5]
    δ = 0.0
    ϵ = 1e-5
    r = 4.0e-4
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g))
    @test status
    @test abs(norm(d_k) - r) <= tol
    @test abs(δ_k - 4.0) <= tol
    # @test norm((x_k + d_k) - [0.0, 1.0], 2) <= tol
    @test norm((x_k + d_k) - [2e-5, 4.00001e-5], 2) <= tol
    # @test norm((x_k + d_k) - [0.0, 0.0], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    # @test abs(obj(nlp, x_k + d_k) - (-2.0)) <= tol
    @test abs(obj(nlp, x_k + d_k) - (-32.004e-8)) <= tol
end

# Check this
function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_2()
    tol = 1e-2
    problem = test_create_hard_case_using_bivariate_convex_model_2()
    nlp = problem.nlp
    x_k = [1e-5, 1e-5]
    δ = 0.0
    ϵ = 1e-5
    r = 0.00245
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g))
    @test status
    @test norm(d_k, 2) - r <= tol
    @test abs(δ_k - 2.0) <= tol
    # @test norm((x_k + d_k) - [-0.0025, 0.999], 2) <= tol
    @test norm((x_k + d_k) - [-0.0025, 2.00001e-5], 2) <= tol
    # @test norm((x_k + d_k) - [0.0, 0.0], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    # @test abs(obj(nlp, x_k + d_k) - (-1.0000125)) <= tol
    @test abs(obj(nlp, x_k + d_k) - (-1.8459e-5)) <= tol
end

# Check this
function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_3()
    tol = 1e-3
    problem = test_create_hard_case_using_bivariate_convex_model_3()
    nlp = problem.nlp
    x_k = [1e-5, 1e-5]
    δ = 0.0
    ϵ = 1e-5
    r = 0.00114
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g))
    @test status
    @test norm(d_k) <= r
    @test abs(δ_k - 8.0) <= tol
    # @test norm((x_k + d_k) - [-3.5355339059327373, -3.5355339059327373], 2) <= tol
    @test norm((x_k + d_k) - [8.1e-4, 8.1e-4], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    # @test abs(obj(nlp, x_k + d_k) - (-100)) <= tol
    @test abs(obj(nlp, x_k + d_k) - (-5.2488e-6)) <= tol
end

function test_optimize_second_order_model_bisection_logic_bug_fix()
    tol = 1e-3
    r = 0.08343452704764227
    g = [3.4679032978601754e-9, 7.39587593251434e-9, 2.7183407851072428e-8, -0.003000483357027406, 0.008419134290306829]
    H = [66.0 65.0743102550725 65.67263629092243 -1.2900082661017744e7 3.6178787247129455e7; 65.0743102550725 64.1661603269859 64.75315415279051 -1.265594646809823e7 3.549303722643331e7; 65.67263629092243 64.75315415279051 65.34746955501134 -1.2813679600892197e7 3.59360895721625e7; -1.2900082661017744e7 -1.265594646809823e7 -1.2813679600892197e7 3.3981511777610044e12 -9.544912385973707e12; 3.6178787247129455e7 3.549303722643331e7 3.59360895721625e7 -9.544912385973707e12 2.6810622545473246e13]
    ϵ = 1e-8
    δ = 6.205227748467783e-12

    success, δ, δ_prime, temp_total_number_factorizations = consistently_adaptive_trust_region_method.findinterval(g, H, δ, ϵ, r)
    @test success
    @test abs(δ - 2.5e-8) <= tol
    @test abs(δ_prime - 1.6e-5) <= tol

    success, δ_m, temp_total_number_factorizations = consistently_adaptive_trust_region_method.bisection(g, H, δ, ϵ, δ_prime, r)
    @test success
    @test abs(δ_m - 5.173e-7) <= tol

    r = 0.0018
    ϵ = 1e-5
    status, δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g))
    @test status
    @test abs(δ_k - 5.173e-7) <= tol
    @test abs(δ_k - 1e-5) <= ϵ && abs(norm(d_k, 2) - r) <= ϵ
end

function test_optimize_second_order_model_bisection_failure_non_hard_case()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem3()
    x = [3.0, 2.0]
    g = grad(nlp, x)
    H = hess(nlp, x)
    r = 1e-8
    δ = 1e-10
    ϵ = 1e-5
    success, δ, d_k, temp_total_number_factorizations, hard_case = optimizeSecondOrderModel(g, H, δ, ϵ, r, norm(g),print_level)
    @test success
    @test hard_case
    @test abs(norm(d_k) - r) <= ϵ
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

@testset "findMinimumEigenValue" begin
    findMinimumEigenValue()
end
