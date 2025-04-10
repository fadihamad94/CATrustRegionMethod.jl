import CATrustRegionMethod

using DelimitedFiles: readdlm
using EnumX
using JuMP
using LinearAlgebra
using NLPModels
using NLPModelsJuMP
using SparseArrays
using Test

#Functions to create NLP models

function createDummyNLPModel()
    x0 = [-1.2; 1.0]
    model = Model()
    @variable(model, x[i = 1:2], start = x0[i])
    @NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createDummyNLPModel2()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (2 * x + y - 1)^2 + x + y + (x^2 - 2 * y^2)^3)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createSimpleUnivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @NLobjective(model, Min, (x - 1)^2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createSimpleConvexNLPModeL()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (x + y - 1)^2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createComplexConvexNLPModeL1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (x + y - 1)^2 + x + y + (x - 2 * y)^2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createComplexNLPModeL1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (2 * x + y - 1)^2 + x + y + (x^2 - 2 * y^2)^2)
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
    @NLobjective(model, Min, -x^2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, -x^2 - y^2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, -x^2 - 2 * y^2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem2()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x^2 + 0.01 * x - y^2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem3()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x^2 - 10 * x * y + y^2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem4()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x + x^2 - y^2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function test_create_dummy_problem()
    nlp = createDummyNLPModel()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 0.5
    return nlp, termination_criteria, algorithm_params
end

function test_create_dummy_problem2()
    nlp = createDummyNLPModel2()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 0.5
    return nlp, termination_criteria, algorithm_params
end

function test_create_simple_convex_nlp_model()
    nlp = createSimpleConvexNLPModeL()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 0.5
    return nlp, termination_criteria, algorithm_params
end

function test_create_complex_convex_nlp1_model()
    nlp = createComplexConvexNLPModeL1()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 0.5
    return nlp, termination_criteria, algorithm_params
end

function test_create_complex_nlp_modeL1()
    nlp = createComplexNLPModeL1()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 0.5
    return nlp, termination_criteria, algorithm_params
end

function test_create_problem_sin_cos_mode_nlp1()
    nlp = createSinCosNLPModeL1()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 0.5
    return nlp, termination_criteria, algorithm_params
end

function test_create_problem_sin_cos_mode_nlp2()
    nlp = createSinCosNLPModeL2()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 0.5
    return nlp, termination_criteria, algorithm_params
end

function test_create_simple_univariate_convex_model()
    nlp = createSimpleUnivariateConvexProblem()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 0.5
    return nlp, termination_criteria, algorithm_params
end

function test_create_simple_univariate_convex_model_solved_same_as_Newton()
    nlp = createSimpleUnivariateConvexProblem()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 2.0
    return nlp, termination_criteria, algorithm_params
end

function test_create_hard_case_using_simple_univariate_convex_model()
    nlp = createHardCaseUsingSimpleUnivariateConvexProblem()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 1.0
    return nlp, termination_criteria, algorithm_params
end

function test_create_hard_case_using_simple_bivariate_convex_model()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 1.0
    return nlp, termination_criteria, algorithm_params
end

function test_create_hard_case_using_bivariate_convex_model_1()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem1()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 1.0
    return nlp, termination_criteria, algorithm_params
end

function test_create_hard_case_using_bivariate_convex_model_2()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem2()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 1.0
    return nlp, termination_criteria, algorithm_params
end

function test_create_hard_case_using_bivariate_convex_model_3()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem3()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 5.0
    return nlp, termination_criteria, algorithm_params
end

function test_create_hard_case_using_bivariate_convex_model_4()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem4()
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.1, 0.1, 8.0, 16.0)
    algorithm_params.r_1 = 10.0
    return nlp, termination_criteria, algorithm_params
end

#Unit test optimize second order model function
function test_optimize_second_order_model_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = [1.0, 1.0]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    γ_3 = 0.5
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, δ_prime_k, d_k =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
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
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = nlp.meta.x0
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    γ_3 = 0.5
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, δ_prime_k, d_k =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_phi_δ_positive_phi_δ_prime_negative()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem2()

    x_k = [0.0, 1.0]
    δ = 250.0
    γ_1 = 0.01
    γ_2 = 0.8
    γ_3 = 0.5
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, δ_prime_k, d_k =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(δ_k - 500.0) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_for_simple_univariate_convex_model()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_simple_univariate_convex_model()

    x_k = [0.0]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    γ_3 = 0.5
    r = 0.5
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, δ_prime_k, d_k =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_for_simple_univariate_convex_model_solved_same_as_Newton()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_simple_univariate_convex_model_solved_same_as_Newton()

    x_k = [0.0]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    γ_3 = 0.5
    r = 2.0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, δ_prime_k, d_k =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
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
    nlp, termination_criteria, algorithm_params = test_create_simple_convex_nlp_model()

    x_k = [0.0, 0.0]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    γ_3 = 0.5
    r = 0.5
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, δ_prime_k, d_k =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test δ_k == 2.0
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_hard_case_using_simple_univariate_convex_model()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_hard_case_using_simple_univariate_convex_model()

    x_k = [1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    γ_3 = 0.5
    r = 0.0002
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    temp_ = norm(g)
    status, δ_k, δ_prime_k, d_k, temp_total_number_factorizations, hard_case =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2 + tol
    @test γ_2 * r - tol <= norm((H + δ_k * I) \ g, 2) <= r
    @test γ_2 * r - tol <= norm(d_k) <= r
    @test abs(norm(d_k) - r) <= tol
    @test norm((x_k + d_k) - [0.00021], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - (-4.00004e-8)) <= tol
end

function test_optimize_second_order_model_hard_case_using_simple_bivariate_convex_model()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_hard_case_using_simple_bivariate_convex_model()

    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    γ_3 = 0.5
    r = 0.00029
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, δ_prime_k, d_k =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(δ_k - 2.109) <= tol
    @test norm((x_k + d_k) - [2.00001e-5, 2.00001e-5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - (-8.000079e-8), 2) <= tol
end

function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_1()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_hard_case_using_bivariate_convex_model_1()

    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    γ_3 = 0.5
    r = 4.0e-4
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, δ_prime_k, d_k =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(norm(d_k) - r) <= tol
    @test abs(δ_k - 4.1099) <= tol
    @test norm((x_k + d_k) - [2e-5, 4.00001e-5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - (-32.004e-8)) <= tol
end

function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_2()
    tol = 1e-2
    nlp, termination_criteria, algorithm_params =
        test_create_hard_case_using_bivariate_convex_model_2()

    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    γ_3 = 0.5
    r = 0.00245
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, δ_prime_k, d_k, temp_total_number_factorizations, hard_case =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test norm(d_k, 2) - r <= tol
    @test abs(δ_k - 3.0) <= tol
    @test norm((x_k + d_k) - [-0.0025, 2.00001e-5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - (-1.8459e-5)) <= tol
end

function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_3()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_hard_case_using_bivariate_convex_model_3()

    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    γ_3 = 0.5
    r = 0.00114
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, δ_prime_k, d_k =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(δ_k - 8.113) <= tol
    @test norm((x_k + d_k) - [8.1e-4, 8.1e-4], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - (-5.2488e-6)) <= tol
end

function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_4()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_hard_case_using_bivariate_convex_model_4()

    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    γ_3 = 0.5
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    r = algorithm_params.r_1

    status, δ_k, δ_prime_k, d_k, temp_total_number_factorizations, hard_case =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test hard_case
    @test q_1 <= q_2
    # @test abs(norm(d_k) - r) <= tol
    @test norm(d_k) <= r
    @test abs(δ_k - 2.0) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_bisection_logic_bug_fix()
    tol = 1e-3
    r = 0.08343452704764227
    g = [
        3.4679032978601754e-9,
        7.39587593251434e-9,
        2.7183407851072428e-8,
        -0.003000483357027406,
        0.008419134290306829,
    ]
    H = [
        66.0 65.0743102550725 65.67263629092243 -1.2900082661017744e7 3.6178787247129455e7
        65.0743102550725 64.1661603269859 64.75315415279051 -1.265594646809823e7 3.549303722643331e7
        65.67263629092243 64.75315415279051 65.34746955501134 -1.2813679600892197e7 3.59360895721625e7
        -1.2900082661017744e7 -1.265594646809823e7 -1.2813679600892197e7 3.3981511777610044e12 -9.544912385973707e12
        3.6178787247129455e7 3.549303722643331e7 3.59360895721625e7 -9.544912385973707e12 2.6810622545473246e13
    ]
    γ_1 = 0.01
    γ_2 = 1 - 1e-8
    δ = 6.205227748467783e-12

    success, δ, δ_prime, temp_total_number_factorizations, temp_d_δ_prime =
        CATrustRegionMethod.findinterval(g, H, δ, γ_2, r)
    @test success
    @test abs(δ - 1.5e-8) <= tol
    @test abs(δ_prime - 1.5e-8) <= tol
    min_grad = norm(g, 2)
    success, δ_m, temp_total_number_factorizations = CATrustRegionMethod.bisection(
        "problem_name",
        g,
        H,
        δ,
        γ_1,
        γ_2,
        δ_prime,
        temp_d_δ_prime,
        r,
        min_grad,
        0,
    )
    @test success
    @test abs(δ_m - 5.173e-7) <= tol

    r = 0.08343452704764227
    g = [
        3.4679032978601754e-9,
        7.39587593251434e-9,
        2.7183407851072428e-8,
        -0.003000483357027406,
        0.008419134290306829,
    ]
    H = [
        66.0 65.0743102550725 65.67263629092243 -1.2900082661017744e7 3.6178787247129455e7
        65.0743102550725 64.1661603269859 64.75315415279051 -1.265594646809823e7 3.549303722643331e7
        65.67263629092243 64.75315415279051 65.34746955501134 -1.2813679600892197e7 3.59360895721625e7
        -1.2900082661017744e7 -1.265594646809823e7 -1.2813679600892197e7 3.3981511777610044e12 -9.544912385973707e12
        3.6178787247129455e7 3.549303722643331e7 3.59360895721625e7 -9.544912385973707e12 2.6810622545473246e13
    ]
    γ_1 = 0.01
    γ_2 = 1 - 1e-8
    γ_3 = 0.5
    δ = 6.205227748467783e-12

    status, δ_k, δ_prime_k, d_k =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g))
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test δ_k <= 1e-6 && norm((H + δ_k * I) \ g, 2) <= r
    @test δ_k <= 1e-6 && norm(d_k) <= r
    @test abs(δ_k - 5.173e-7) <= tol
    @test abs(δ_k - 1e-5) <= tol
end

function test_hard_case_failure()
    tol = 1e-3
    problem_name = "MSQRTBLS"
    dir = joinpath(@__DIR__, "examples_CUTEST")
    g = vec(readdlm(joinpath(dir, "$(problem_name)_gradient.txt"), ','))
    g = map(Float64, g)
    H = readdlm(joinpath(dir, "$(problem_name)_hessian.txt"))
    H = sparse(H)
    hard_case_sol, δ, γ_1, γ_2, γ_3, r, min_grad =
        readdlm(joinpath(dir, "$(problem_name)_params.txt"))
    status, δ_k, δ_prime_k, d_k, temp_total_number_factorizations, hard_case =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, min_grad, 0)
    @test status == true
    @test !hard_case == hard_case_sol == true
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * min_grad
    @test q_1 <= q_2
    @test norm(d_k) <= r

    problem_name = "SPIN2LS"
    g = vec(readdlm(joinpath(dir, "$(problem_name)_gradient.txt"), ','))
    g = map(Float64, g)
    H = readdlm(joinpath(dir, "$(problem_name)_hessian.txt"))
    H = sparse(H)
    hard_case_sol, δ, γ_1, γ_2, γ_3, r, min_grad =
        readdlm(joinpath(dir, "$(problem_name)_params.txt"))
    status, δ_k, δ_prime_k, d_k, temp_total_number_factorizations, hard_case =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, min_grad, 0)
    @test status == true
    @test !hard_case == hard_case_sol == true
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * min_grad
    @test q_1 <= q_2
    @test norm(d_k) <= r
end

function test_optimize_second_order_model_bisection_failure_non_hard_case()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem3()
    x = [3.0, 2.0]
    g = grad(nlp, x)
    H = hess(nlp, x)
    r = 1e-8
    δ = 1e-10
    γ_1 = 0.01
    γ_2 = 1 - 1e-5
    γ_3 = 0.5
    print_level = 0
    status, δ_k, δ_prime_k, d_k, temp_total_number_factorizations, hard_case =
        CATrustRegionMethod.optimizeSecondOrderModel("problem_name", g, H, δ, γ_1, γ_2, γ_3, r, norm(g), print_level)
    @test status == false
    @test norm(d_k) == 0.0
    @test hard_case == true
end


function test_phi_positive_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = [0.0, 0.0]
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    Φ_δ, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ, ϵ, r)
    @test Φ_δ == 1
    @test positive_definite
end

function test_phi_zero()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 0.0
    ϵ = 0.8
    r = 0.4
    Φ_δ, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ, ϵ, r)
    @test Φ_δ == 0
    @test positive_definite
end

function test_phi_negative_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 3.0
    γ_2 = 1.0
    r = 1.0
    Φ_δ, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ, γ_2, r)
    @test Φ_δ == -1
    @test positive_definite
end

function test_find_interval_with_both_phi_zero_starting_from_phi_zero()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    success, δ, δ_prime, temp_total_number_factorizations, temp_d_δ_prime =
        CATrustRegionMethod.findinterval(g, H, δ, ϵ, r)
    @test δ == 16.0
    @test δ_prime == 512.0
    Φ_δ, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ, ϵ, r)
    @test Φ_δ == 1
    @test positive_definite
    Φ_δ_prime, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ_prime, ϵ, r)
    @test Φ_δ_prime == -1
    @test positive_definite
end

function test_find_interval_with_both_phi_0_starting_from_phi_negative_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = [0.0, 0.0]
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    success, δ, δ_prime, temp_total_number_factorizations = CATrustRegionMethod.findinterval(g, H, δ, ϵ, r)
    @test δ == 2.0
    @test δ_prime == 16.0
    Φ_δ, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ, ϵ, r)
    @test Φ_δ == 1
    @test positive_definite
    Φ_δ_prime, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ_prime, ϵ, r)
    @test Φ_δ_prime == -1
    @test positive_definite
end

function test_find_interval_with_both_phi_0_starting_from_phi_positive_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem2()

    x_k = [0.0, 0.0]
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 9.0
    ϵ = 0.8
    r = 0.2
    success, δ, δ_prime, temp_total_number_factorizations = CATrustRegionMethod.findinterval(g, H, δ, ϵ, r)
    @test δ == δ_prime == 9.0
    Φ_δ, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ, ϵ, r)
    @test Φ_δ == 0
    @test positive_definite
    Φ_δ_prime, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ_prime, ϵ, r)
    @test Φ_δ_prime == 0
    @test positive_definite
end

function test_find_interval_with_phi_δ_positive_one_phi_δ_prime_negative_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem2()

    x_k = [0.0, 1.0]
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 250.0
    γ_2 = 0.2
    r = 0.3
    success, δ, δ_prime, temp_total_number_factorizations =
        CATrustRegionMethod.findinterval(g, H, δ, γ_2, r)
    @test (δ, δ_prime) == (500.0, 500.0)
    Φ_δ, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ, γ_2, r)
    @test Φ_δ == 0
    @test positive_definite
    Φ_δ_prime, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ_prime, γ_2, r)
    @test Φ_δ_prime == 0
    @test positive_definite
end

function test_bisection_with_starting_on_root_δ_zero()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 64.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 0.2
    success, δ, δ_prime, temp_total_number_factorizations, temp_d_δ_prime =
        CATrustRegionMethod.findinterval(g, H, δ, γ_2, r)
    min_grad = norm(g, 2)
    success, δ_m, temp_total_number_factorizations = CATrustRegionMethod.bisection(
        "problem_name",
        g,
        H,
        δ,
        γ_1,
        γ_2,
        δ_prime,
        temp_d_δ_prime,
        r,
        min_grad,
        0,
    )
    @test success
    @test δ_m == δ == δ_prime
    Φ_δ, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ, γ_2, r)
    @test Φ_δ == 0
    @test positive_definite
    Φ_δ_prime, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ_prime, γ_2, r)
    @test Φ_δ_prime == 0
    @test positive_definite
    Φ_δ_m, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ_m, γ_2, r)
    @test Φ_δ_prime == 0
    @test positive_definite
end

function test_bisection_with_starting_on_root_δ_not_zero()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = [0.0, 0.0]
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.2
    r = 0.2
    success, δ, δ_prime, temp_total_number_factorizations, temp_d_δ_prime =
        CATrustRegionMethod.findinterval(g, H, δ, γ_2, r)
    min_grad = norm(g, 2)
    success, δ_m, temp_total_number_factorizations = CATrustRegionMethod.bisection(
        "problem_name",
        g,
        H,
        δ,
        γ_1,
        γ_2,
        δ_prime,
        temp_d_δ_prime,
        r,
        min_grad,
        0,
    )
    @test success
    @test δ_m == 16.0
    @test δ == 16.0
    @test δ_prime == 16.0
    Φ_δ, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ, γ_2, r)
    @test Φ_δ == 0
    @test positive_definite
    Φ_δ_prime, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ_prime, γ_2, r)
    @test Φ_δ_prime == 0
    @test positive_definite
    Φ_δ_m, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ_m, γ_2, r)
    @test Φ_δ_m == 0
    @test positive_definite
end

function test_bisection_with_starting_from_negative_one_and_positive_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem2()

    x_k = [0.0, 1.0]
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 250.0
    γ_1 = 0.01
    γ_2 = 0.2
    r = 0.3
    success, δ, δ_prime, temp_total_number_factorizations, temp_d_δ_prime =
        CATrustRegionMethod.findinterval(g, H, δ, γ_2, r)
    min_grad = norm(g, 2)
    success, δ_m, temp_total_number_factorizations = CATrustRegionMethod.bisection(
        "problem_name",
        g,
        H,
        δ,
        γ_1,
        γ_2,
        δ_prime,
        temp_d_δ_prime,
        r,
        min_grad,
        0,
    )
    @test success
    @test abs(δ_m - 500.0) <= 1e-3
    Φ_δ, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ, γ_2, r)
    @test Φ_δ == 0
    @test positive_definite
    Φ_δ_prime, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ_prime, γ_2, r)
    @test Φ_δ_prime == 0
    @test positive_definite
    Φ_δ_m, temp_d, positive_definite = CATrustRegionMethod.phi(g, H, δ_m, γ_2, r)
    @test Φ_δ_prime == 0
    @test positive_definite
end

function test_compute_second_order_model_negative_direction()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()
    x_k = [0.0, 0.0]
    d_k = [-1.0, -1.0]

    function_value = obj(nlp, x_k)
    gradient_value = grad(nlp, x_k)
    hessian_value = hess(nlp, x_k)
    second_order_model_value =
        CATrustRegionMethod.computeSecondOrderModel(gradient_value, hessian_value, d_k)
    @test second_order_model_value == 104.0 - function_value
end

function test_compute_second_order_model_zero_direction()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()
    x_k = [0.0, 0.0]
    d_k = [0.0, 0.0]

    function_value = obj(nlp, x_k)
    gradient_value = grad(nlp, x_k)
    hessian_value = hess(nlp, x_k)
    second_order_model_value =
        CATrustRegionMethod.computeSecondOrderModel(gradient_value, hessian_value, d_k)
    @test second_order_model_value == 1.0 - function_value
end

function test_compute_second_order_model_positive_direction()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()
    x_k = [0.0, 0.0]
    d_k = [1.0, 1.0]

    function_value = obj(nlp, x_k)
    gradient_value = grad(nlp, x_k)
    hessian_value = hess(nlp, x_k)
    second_order_model_value =
        CATrustRegionMethod.computeSecondOrderModel(gradient_value, hessian_value, d_k)
    @test second_order_model_value == 100.0 - function_value
end

function test_compute_ρ_hat_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = [1.0, 1.0]
    δ = 0.0
    ϵ = 0.2
    r = 0.2
    d_k = [-0.0, -0.0]
    θ = algorithm_params.θ
    fval_current = obj(nlp, x_k)
    fval_next = obj(nlp, x_k + d_k)
    gval_current = grad(nlp, x_k)
    gval_next = grad(nlp, x_k + d_k)
    H = hess(nlp, x_k)
    ρ = CATrustRegionMethod.compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, H, d_k, θ)
end

function test_compute_ρ_hat_phi_zero()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = nlp.meta.x0
    δ = 0.0
    ϵ = 1.2
    r = 0.2
    d_k = [0.02471910112359557, 0.3806741573033706]
    θ = algorithm_params.θ
    fval_current = obj(nlp, x_k)
    fval_next = obj(nlp, x_k + d_k)
    gval_current = grad(nlp, x_k)
    gval_next = grad(nlp, x_k + d_k)
    H = hess(nlp, x_k)
    ρ = CATrustRegionMethod.compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, H, d_k, θ)[1]
    @test norm(ρ - 0.980423689675886, 2) <= tol
end

function test_compute_ρ_hat_phi_δ_positive_phi_δ_prime_negative()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem2()

    x_k = [0.0, 1.0]
    δ = 250.0
    ϵ = 1.2
    r = 0.2
    d_k = [-0.005830328471736362, 0.34323592199917485]
    θ = algorithm_params.θ
    fval_current = obj(nlp, x_k)
    fval_next = obj(nlp, x_k + d_k)
    gval_current = grad(nlp, x_k)
    gval_next = grad(nlp, x_k + d_k)
    H = hess(nlp, x_k)
    ρ = CATrustRegionMethod.compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, H, d_k, θ)[1]
    @test norm(ρ - 1.126954013438328, 2) <= tol
end

function test_compute_l_2_norm_diagonal_matrix()
    tol = 1e-3

    # Create a diagonal sparse matrix
    i = [1, 2, 3]
    j = [1, 2, 3]
    v = [10.0, 20.0, 30.0]
    diagonal_sparse_matrix = sparse(i, j, v)

    # Create a symmetric sparse matrix
    symmetric_diagonal_matrix = Symmetric(diagonal_sparse_matrix)

    l2_norm_our_approach = CATrustRegionMethod.matrix_l2_norm(symmetric_diagonal_matrix)

    l2_norm_using_linear_algebra = opnorm(Matrix(symmetric_diagonal_matrix), 2)

    @test abs(l2_norm_our_approach - l2_norm_using_linear_algebra) <= tol
end

function test_compute_l_2_norm_symmetric_matrix_2_by_2()
    tol = 1e-3

    # Define the indices and values for a sparse matrix
    i = [1, 1, 2, 2]
    j = [1, 2, 1, 2]
    v = [10.0, 20.0, 20.0, 10.0]
    sparse_matrix = sparse(i, j, v)

    # Create a symmetric sparse matrix
    symmetric_matrix = Symmetric(sparse_matrix)

    # Create a symmetric sparse matrix
    symmetric_matrix = Symmetric(sparse_matrix, :U)

    l2_norm_our_approach = CATrustRegionMethod.matrix_l2_norm(symmetric_matrix)

    l2_norm_using_linear_algebra = opnorm(Matrix(symmetric_matrix), 2)

    @test abs(l2_norm_our_approach - l2_norm_using_linear_algebra) <= tol
end

function test_compute_l_2_norms_ymmetric_matrix_3_by_3()
    tol = 1e-3

    # Create a sparse matrix
    i = [1, 2, 3, 1]
    j = [1, 2, 3, 2]
    v = [10.0, 20.0, 30.0, 40.0]
    sparse_matrix = sparse(i, j, v)

    # Create a symmetric sparse matrix
    symmetric_matrix = Symmetric(sparse_matrix, :U) # ':U' means to use the upper triangle

    l2_norm_our_approach = CATrustRegionMethod.matrix_l2_norm(symmetric_matrix)

    l2_norm_using_linear_algebra = opnorm(Matrix(symmetric_matrix), 2)

    @test abs(l2_norm_our_approach - l2_norm_using_linear_algebra) <= tol
end

function unit_tests()
    #Unit test for the ϕ function
    test_phi_negative_one()
    test_phi_zero()
    test_phi_positive_one()

    #Unit test for the find interval function
    test_find_interval_with_both_phi_zero_starting_from_phi_zero()
    test_find_interval_with_both_phi_0_starting_from_phi_negative_one()
    test_find_interval_with_both_phi_0_starting_from_phi_positive_one()
    test_find_interval_with_phi_δ_positive_one_phi_δ_prime_negative_one()

    #Unit test for the bisection function
    test_bisection_with_starting_on_root_δ_zero()
    test_bisection_with_starting_on_root_δ_not_zero()
    test_bisection_with_starting_from_negative_one_and_positive_one()

    #Unit test compute second order model function
    test_compute_second_order_model_negative_direction()
    test_compute_second_order_model_zero_direction()
    test_compute_second_order_model_positive_direction()

    #Unit test compute ρ function
    test_compute_ρ_hat_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    test_compute_ρ_hat_phi_zero()
    test_compute_ρ_hat_phi_δ_positive_phi_δ_prime_negative()

    #Unit test for the matrix l2 norm function
    test_compute_l_2_norm_diagonal_matrix()
    test_compute_l_2_norm_symmetric_matrix_2_by_2()
    test_compute_l_2_norms_ymmetric_matrix_3_by_3()
end

function optimize_models_test()
    test_optimize_second_order_model_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    test_optimize_second_order_model_phi_zero()
    test_optimize_second_order_model_phi_δ_positive_phi_δ_prime_negative()
    test_optimize_second_order_model_for_simple_univariate_convex_model()
    test_optimize_second_order_model_for_simple_univariate_convex_model_solved_same_as_Newton()
    test_optimize_second_order_model_for_simple_bivariate_convex_model()
    test_optimize_second_order_model_hard_case_using_simple_univariate_convex_model() #This i snot a hard case
    test_optimize_second_order_model_hard_case_using_simple_bivariate_convex_model()  #This i snot a hard case
    test_optimize_second_order_model_hard_case_using_bivariate_convex_model_1() #This i snot a hard case
    test_optimize_second_order_model_hard_case_using_bivariate_convex_model_2() #This i snot a hard case
    test_optimize_second_order_model_hard_case_using_bivariate_convex_model_3() #This i snot a hard case
    test_optimize_second_order_model_hard_case_using_bivariate_convex_model_4()
    test_optimize_second_order_model_bisection_logic_bug_fix()
    test_optimize_second_order_model_bisection_failure_non_hard_case()
    test_hard_case_failure()
end

@testset "basic_unit_tests" begin
    unit_tests()
end

@testset "TRS_Solver_Tests" begin
    optimize_models_test()
end
