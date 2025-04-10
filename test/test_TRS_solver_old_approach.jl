using EnumX
using JuMP
using LinearAlgebra
using NLPModels
using NLPModelsJuMP
using SparseArrays
using Test

import CATrustRegionMethod

#Unit test optimize second order model function
function test_old_optimize_second_order_model_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = [1.0, 1.0]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = CATrustRegionMethod.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test d_k == [-0.0, -0.0]
    @test δ_k == δ
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_old_optimize_second_order_model_phi_zero()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = nlp.meta.x0
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = CATrustRegionMethod.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test norm(d_k - [0.106, 0.139], 2) <= tol
    @test abs(δ_k - 64.0) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_old_optimize_second_order_model_phi_δ_positive_phi_δ_prime_negative()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem2()

    x_k = [0.0, 1.0]
    δ = 250.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = CATrustRegionMethod.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test norm(d_k - [-0.0032, 0.179], 2) <= tol
    @test abs(δ_k - 500.0) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_old_optimize_second_order_model_for_simple_univariate_convex_model()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_simple_univariate_convex_model()

    x_k = [0.0]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 0.5
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = CATrustRegionMethod.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_old_optimize_second_order_model_for_simple_univariate_convex_model_solved_same_as_Newton()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_simple_univariate_convex_model_solved_same_as_Newton()

    x_k = [0.0]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 2.0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = CATrustRegionMethod.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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

function test_old_optimize_second_order_model_for_simple_bivariate_convex_model()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params = test_create_simple_convex_nlp_model()

    x_k = [0.0, 0.0]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 0.5
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = CATrustRegionMethod.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test δ_k == 2.0
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_old_optimize_second_order_model_hard_case_using_simple_univariate_convex_model()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_hard_case_using_simple_univariate_convex_model()

    x_k = [1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 1 - 1e-5
    r = 0.0002
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    temp_ = norm(g)
    status, δ_k, d_k = CATrustRegionMethod.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test norm((x_k + d_k) - [0.00021], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - (-4.00004e-8)) <= tol
end

function test_old_optimize_second_order_model_hard_case_using_simple_bivariate_convex_model()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_hard_case_using_simple_bivariate_convex_model()

    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 1 - 1e-5
    r = 0.00029
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = CATrustRegionMethod.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(δ_k - 2.097) <= tol
    @test norm((x_k + d_k) - [2.00001e-5, 2.00001e-5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - (-8.000079e-8), 2) <= tol
end

function test_old_optimize_second_order_model_hard_case_using_bivariate_convex_model_1()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_hard_case_using_bivariate_convex_model_1()

    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 1 - 1e-5
    r = 4.0e-4
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = CATrustRegionMethod.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(norm(d_k) - r) <= tol
    @test abs(δ_k - 4.1) <= tol
    @test norm((x_k + d_k) - [2e-5, 4.00001e-5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - (-32.004e-8)) <= tol
end

function test_old_optimize_second_order_model_hard_case_using_bivariate_convex_model_2()
    tol = 1e-2
    nlp, termination_criteria, algorithm_params =
        test_create_hard_case_using_bivariate_convex_model_2()

    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 1 - 1e-5
    r = 0.00245
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = CATrustRegionMethod.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test norm(d_k, 2) - r <= tol
    @test abs(δ_k - 2.102) <= tol
    @test norm((x_k + d_k) - [-0.0025, 2.00001e-5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - (-1.8459e-5)) <= tol
end

function test_old_optimize_second_order_model_hard_case_using_bivariate_convex_model_3()
    tol = 1e-3
    nlp, termination_criteria, algorithm_params =
        test_create_hard_case_using_bivariate_convex_model_3()

    x_k = [1e-5, 1e-5]
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 1 - 1e-5
    r = 0.00114
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    status, δ_k, d_k = CATrustRegionMethod.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    γ_1 = 1e-2
    q_1 = norm(H * d_k + g + δ_k * d_k)
    q_2 = γ_1 * norm(g)
    @test status
    @test q_1 <= q_2
    @test γ_2 * r <= norm(d_k) <= r
    @test abs(δ_k - 8.099) <= tol
    @test norm((x_k + d_k) - [8.1e-4, 8.1e-4], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test abs(obj(nlp, x_k + d_k) - (-5.2488e-6)) <= tol
end

function test_old_optimize_second_order_model_bisection_failure_non_hard_case()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem3()
    x = [3.0, 2.0]
    g = grad(nlp, x)
    H = hess(nlp, x)
    r = 1e-8
    δ = 1e-10
    γ_1 = 0.01
    γ_2 = 1 - 1e-5
    print_level = 0
    status, δ_k, d_k, hard_case = CATrustRegionMethod.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
    @test status == true
    @test abs(norm(d_k) - 0.0) <= 1e-3
    @test hard_case == false
end


function test_old_phi_negative_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = [0.0, 0.0]
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    Φ_δ = CATrustRegionMethod.phiOldApproach(g, H, δ, ϵ, r)
    @test Φ_δ == -1
end

function test_old_phi_zero()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 0.0
    ϵ = 0.8
    r = 0.4
    Φ_δ = CATrustRegionMethod.phiOldApproach(g, H, δ, ϵ, r)
    @test Φ_δ == 0

end

function test_old_phi_positive_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 3.0
    γ_2 = 1.0
    r = 1.0
    Φ_δ = CATrustRegionMethod.phiOldApproach(g, H, δ, γ_2, r)
    @test Φ_δ == 1
end

function test_old_find_interval_with_both_phi_zero_starting_from_phi_zero()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    δ, δ_prime = CATrustRegionMethod.findintervalOldApproach(g, H, δ, ϵ, r)
    @test δ == δ_prime == 64.0
    Φ_δ = CATrustRegionMethod.phiOldApproach(g, H, δ, ϵ, r)
    @test Φ_δ == 0

    Φ_δ_prime = CATrustRegionMethod.phiOldApproach(g, H, δ_prime, ϵ, r)
    @test Φ_δ_prime == 0

end

function test_old_find_interval_with_both_phi_0_starting_from_phi_negative_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = [0.0, 0.0]
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    δ, δ_prime = CATrustRegionMethod.findintervalOldApproach(g, H, δ, ϵ, r)
    @test δ == 8.0
    @test δ_prime == 8.0
    Φ_δ = CATrustRegionMethod.phiOldApproach(g, H, δ, ϵ, r)
    @test Φ_δ == 0

    Φ_δ_prime = CATrustRegionMethod.phiOldApproach(g, H, δ_prime, ϵ, r)
    @test Φ_δ_prime == 0

end

function test_old_find_interval_with_both_phi_0_starting_from_phi_positive_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem2()

    x_k = [0.0, 0.0]
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 9.0
    ϵ = 0.8
    r = 0.2
    δ, δ_prime = CATrustRegionMethod.findintervalOldApproach(g, H, δ, ϵ, r)
    @test δ == δ_prime == 9.0
    Φ_δ = CATrustRegionMethod.phiOldApproach(g, H, δ, ϵ, r)
    @test Φ_δ == 0

    Φ_δ_prime = CATrustRegionMethod.phiOldApproach(g, H, δ_prime, ϵ, r)
    @test Φ_δ_prime == 0

end

function test_old_find_interval_with_phi_δ_positive_one_phi_δ_prime_negative_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem2()

    x_k = [0.0, 1.0]
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 250.0
    γ_2 = 0.2
    r = 0.3
    δ, δ_prime = CATrustRegionMethod.findintervalOldApproach(g, H, δ, γ_2, r)
    @test (δ, δ_prime) == (500.0, 500.0)
    Φ_δ = CATrustRegionMethod.phiOldApproach(g, H, δ, γ_2, r)
    @test Φ_δ == 0

    Φ_δ_prime = CATrustRegionMethod.phiOldApproach(g, H, δ_prime, γ_2, r)
    @test Φ_δ_prime == 0

end

function test_old_bisection_with_starting_on_root_δ_zero()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 64.0
    γ_1 = 0.01
    γ_2 = 0.8
    r = 0.2
    δ, δ_prime = CATrustRegionMethod.findintervalOldApproach(g, H, δ, γ_2, r)
    min_grad = norm(g, 2)
    δ_m = CATrustRegionMethod.bisectionOldApproach(g, H, δ, γ_2, δ_prime, r)
    @test δ_m == δ == δ_prime
    Φ_δ = CATrustRegionMethod.phiOldApproach(g, H, δ, γ_2, r)
    @test Φ_δ == 0

    Φ_δ_prime = CATrustRegionMethod.phiOldApproach(g, H, δ_prime, γ_2, r)
    @test Φ_δ_prime == 0

    Φ_δ_m = CATrustRegionMethod.phiOldApproach(g, H, δ_m, γ_2, r)
    @test Φ_δ_prime == 0

end

function test_old_bisection_with_starting_on_root_δ_not_zero()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = [0.0, 0.0]
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 0.0
    γ_1 = 0.01
    γ_2 = 0.2
    r = 0.2
    δ, δ_prime = CATrustRegionMethod.findintervalOldApproach(g, H, δ, γ_2, r)
    min_grad = norm(g, 2)
    δ_m = CATrustRegionMethod.bisectionOldApproach(g, H, δ, γ_2, δ_prime, r)
    @test δ_m == 8.0
    @test δ == 8.0
    @test δ_prime == 8.0
    Φ_δ = CATrustRegionMethod.phiOldApproach(g, H, δ, γ_2, r)
    @test Φ_δ == 0

    Φ_δ_prime = CATrustRegionMethod.phiOldApproach(g, H, δ_prime, γ_2, r)
    @test Φ_δ_prime == 0

    Φ_δ_m = CATrustRegionMethod.phiOldApproach(g, H, δ_m, γ_2, r)
    @test Φ_δ_m == 0

end

function test_old_bisection_with_starting_from_negative_one_and_positive_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem2()

    x_k = [0.0, 1.0]
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 250.0
    γ_1 = 0.01
    γ_2 = 0.2
    r = 0.3
    δ, δ_prime = CATrustRegionMethod.findintervalOldApproach(g, H, δ, γ_2, r)
    min_grad = norm(g, 2)
    δ_m = CATrustRegionMethod.bisectionOldApproach(g, H, δ, γ_2, δ_prime, r)
    @test abs(δ_m - 500.0) <= 1e-3
    Φ_δ = CATrustRegionMethod.phiOldApproach(g, H, δ, γ_2, r)
    @test Φ_δ == 0

    Φ_δ_prime = CATrustRegionMethod.phiOldApproach(g, H, δ_prime, γ_2, r)
    @test Φ_δ_prime == 0

    Φ_δ_m = CATrustRegionMethod.phiOldApproach(g, H, δ_m, γ_2, r)
    @test Φ_δ_prime == 0

end

function test_old_compute_second_order_model_negative_direction()
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

function test_old_compute_second_order_model_zero_direction()
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

function test_old_compute_second_order_model_positive_direction()
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

function test_old_compute_ρ_hat_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
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

function test_old_compute_ρ_hat_phi_zero()
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

function test_old_compute_ρ_hat_phi_δ_positive_phi_δ_prime_negative()
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

function test_old_compute_l_2_norm_diagonal_matrix()
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

function test_old_compute_l_2_norm_symmetric_matrix_2_by_2()
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

function test_old_compute_l_2_norms_ymmetric_matrix_3_by_3()
    tol = 1e-3

    # Create a sparse matrix
    i = [1, 2, 3, 1]
    j = [1, 2, 3, 2]
    v = [10.0, 20.0, 30.0, 40.0]
    sparse_matrix = sparse(i, j, v)

    # Create a symmetric sparse matrix
    symmetric_matrix = Symmetric(sparse_matrix, :U)

    l2_norm_our_approach = CATrustRegionMethod.matrix_l2_norm(symmetric_matrix)

    l2_norm_using_linear_algebra = opnorm(Matrix(symmetric_matrix), 2)

    @test abs(l2_norm_our_approach - l2_norm_using_linear_algebra) <= tol
end

function old_unit_tests()
    #Unit test for the ϕ function
    test_old_phi_negative_one()
    test_old_phi_zero()
    test_old_phi_positive_one()

    #Unit test for the find interval function
    test_old_find_interval_with_both_phi_zero_starting_from_phi_zero()
    test_old_find_interval_with_both_phi_0_starting_from_phi_negative_one()
    test_old_find_interval_with_both_phi_0_starting_from_phi_positive_one()
    test_old_find_interval_with_phi_δ_positive_one_phi_δ_prime_negative_one()

    #Unit test for the bisection function
    test_old_bisection_with_starting_on_root_δ_zero()
    test_old_bisection_with_starting_on_root_δ_not_zero()
    test_old_bisection_with_starting_from_negative_one_and_positive_one()

    #Unit test compute second order model function
    test_old_compute_second_order_model_negative_direction()
    test_old_compute_second_order_model_zero_direction()
    test_old_compute_second_order_model_positive_direction()

    #Unit test compute ρ function
    test_old_compute_ρ_hat_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    test_old_compute_ρ_hat_phi_zero()
    test_old_compute_ρ_hat_phi_δ_positive_phi_δ_prime_negative()

    #Unit test for the matrix l2 norm function
    test_old_compute_l_2_norm_diagonal_matrix()
    test_old_compute_l_2_norm_symmetric_matrix_2_by_2()
    test_old_compute_l_2_norms_ymmetric_matrix_3_by_3()
end

function old_optimize_models_test()
    test_old_optimize_second_order_model_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    test_old_optimize_second_order_model_phi_zero()
    test_old_optimize_second_order_model_phi_δ_positive_phi_δ_prime_negative()
    test_old_optimize_second_order_model_for_simple_univariate_convex_model()
    test_old_optimize_second_order_model_for_simple_univariate_convex_model_solved_same_as_Newton()
    test_old_optimize_second_order_model_for_simple_bivariate_convex_model()
    test_old_optimize_second_order_model_hard_case_using_simple_univariate_convex_model()
    test_old_optimize_second_order_model_hard_case_using_simple_bivariate_convex_model()
    test_old_optimize_second_order_model_hard_case_using_bivariate_convex_model_1()
    test_old_optimize_second_order_model_hard_case_using_bivariate_convex_model_2()
    test_old_optimize_second_order_model_hard_case_using_bivariate_convex_model_3()
    test_old_optimize_second_order_model_bisection_failure_non_hard_case()
end

@testset "basic_unit_tests_old_TRS" begin
    old_unit_tests()
end

@testset "OLD_TRS_Solver_Tests" begin
    old_optimize_models_test()
end
