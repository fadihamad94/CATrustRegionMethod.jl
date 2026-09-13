using JuMP
using LinearAlgebra
using NLPModels
using NLPModelsJuMP
using Random
using SparseArrays
using Test

import TrustRegionSubproblemSolvers

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
    status, δ_k, d_k = TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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
    status, δ_k, d_k = TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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
    status, δ_k, d_k = TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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
    status, δ_k, d_k = TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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
    status, δ_k, d_k = TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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
    status, δ_k, d_k = TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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
    status, δ_k, d_k = TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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
    status, δ_k, d_k = TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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
    status, δ_k, d_k = TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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
    status, δ_k, d_k = TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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
    status, δ_k, d_k = TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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
    status, δ_k, d_k, hard_case = TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(g, H, δ, γ_2, r)
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
    Φ_δ = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ, ϵ, r)
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
    Φ_δ = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ, ϵ, r)
    @test Φ_δ == 0

end

function test_old_phi_positive_one()
    nlp, termination_criteria, algorithm_params = test_create_dummy_problem()

    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ = 3.0
    γ_2 = 1 - 1e-5
    r = 1.0
    Φ_δ = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ, γ_2, r)
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
    δ, δ_prime = TrustRegionSubproblemSolvers.findintervalOldApproach(g, H, δ, ϵ, r)
    @test δ == δ_prime == 64.0
    Φ_δ = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ, ϵ, r)
    @test Φ_δ == 0

    Φ_δ_prime = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ_prime, ϵ, r)
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
    δ, δ_prime = TrustRegionSubproblemSolvers.findintervalOldApproach(g, H, δ, ϵ, r)
    @test δ == 8.0
    @test δ_prime == 8.0
    Φ_δ = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ, ϵ, r)
    @test Φ_δ == 0

    Φ_δ_prime = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ_prime, ϵ, r)
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
    δ, δ_prime = TrustRegionSubproblemSolvers.findintervalOldApproach(g, H, δ, ϵ, r)
    @test δ == δ_prime == 9.0
    Φ_δ = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ, ϵ, r)
    @test Φ_δ == 0

    Φ_δ_prime = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ_prime, ϵ, r)
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
    δ, δ_prime = TrustRegionSubproblemSolvers.findintervalOldApproach(g, H, δ, γ_2, r)
    @test (δ, δ_prime) == (500.0, 500.0)
    Φ_δ = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ, γ_2, r)
    @test Φ_δ == 0

    Φ_δ_prime = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ_prime, γ_2, r)
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
    δ, δ_prime = TrustRegionSubproblemSolvers.findintervalOldApproach(g, H, δ, γ_2, r)
    min_grad = norm(g, 2)
    δ_m = TrustRegionSubproblemSolvers.bisectionOldApproach(g, H, δ, γ_2, δ_prime, r)
    @test δ_m == δ == δ_prime
    Φ_δ = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ, γ_2, r)
    @test Φ_δ == 0

    Φ_δ_prime = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ_prime, γ_2, r)
    @test Φ_δ_prime == 0

    Φ_δ_m = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ_m, γ_2, r)
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
    δ, δ_prime = TrustRegionSubproblemSolvers.findintervalOldApproach(g, H, δ, γ_2, r)
    min_grad = norm(g, 2)
    δ_m = TrustRegionSubproblemSolvers.bisectionOldApproach(g, H, δ, γ_2, δ_prime, r)
    @test δ_m == 8.0
    @test δ == 8.0
    @test δ_prime == 8.0
    Φ_δ = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ, γ_2, r)
    @test Φ_δ == 0

    Φ_δ_prime = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ_prime, γ_2, r)
    @test Φ_δ_prime == 0

    Φ_δ_m = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ_m, γ_2, r)
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
    δ, δ_prime = TrustRegionSubproblemSolvers.findintervalOldApproach(g, H, δ, γ_2, r)
    min_grad = norm(g, 2)
    δ_m = TrustRegionSubproblemSolvers.bisectionOldApproach(g, H, δ, γ_2, δ_prime, r)
    @test abs(δ_m - 500.0) <= 1e-3
    Φ_δ = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ, γ_2, r)
    @test Φ_δ == 0

    Φ_δ_prime = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ_prime, γ_2, r)
    @test Φ_δ_prime == 0

    Φ_δ_m = TrustRegionSubproblemSolvers.phiOldApproach(g, H, δ_m, γ_2, r)
    @test Φ_δ_prime == 0

end

function test_old_eigendecomposition_memory_guard()
    @test TrustRegionSubproblemSolvers.DEFAULT_OLD_SOLVER_EIGENDECOMPOSITION_MEMORY_FRACTION ==
          0.5
    @test TrustRegionSubproblemSolvers.oldSolverEigendecompositionMemoryEstimate(10) == 2400
    @test TrustRegionSubproblemSolvers.oldSolverEigendecompositionMemoryBudget(4800) == 2400
    @test_throws ArgumentError TrustRegionSubproblemSolvers.oldSolverEigendecompositionMemoryEstimate(-1)
    @test_throws ArgumentError TrustRegionSubproblemSolvers.oldSolverEigendecompositionMemoryBudget(-1)
    @test_throws ArgumentError TrustRegionSubproblemSolvers.oldSolverEigendecompositionMemoryBudget(
        4800,
        0.0,
    )
    @test_throws ArgumentError TrustRegionSubproblemSolvers.oldSolverEigendecompositionMemoryBudget(
        4800,
        1.1,
    )

    H = spdiagm(0 => ones(10))
    @test isnothing(
        TrustRegionSubproblemSolvers.guardOldSolverEigendecompositionMemory(
            H;
            total_memory_bytes = 4800,
        ),
    )
    @test_logs (:warn, r"Skipping old-solver dense eigendecomposition") begin
        @test_throws OutOfMemoryError TrustRegionSubproblemSolvers.guardOldSolverEigendecompositionMemory(
            H;
            total_memory_bytes = 4799,
        )
    end
end

function test_old_sparse_positive_definite_path_matches_dense_path()
    H_sparse = spdiagm(0 => [2.0, 3.0, 4.0])
    H_dense = Matrix(H_sparse)
    g = [1.0, -2.0, 3.0]
    r = 10.0

    sparse_status, sparse_delta, sparse_direction, sparse_hard_case =
        TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(
            g,
            H_sparse,
            0.0,
            0.8,
            r,
        )
    dense_status, dense_delta, dense_direction, dense_hard_case =
        TrustRegionSubproblemSolvers.optimizeSecondOrderModelOldApproach(
            g,
            H_dense,
            0.0,
            0.8,
            r,
        )

    @test sparse_status
    @test dense_status
    @test sparse_delta == dense_delta == 0.0
    @test !sparse_hard_case
    @test !dense_hard_case
    @test sparse_direction ≈ dense_direction
    @test sparse_direction ≈ -(H_sparse \ g)
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

    # Unit tests for the old-solver memory guard and sparse positive-definite path
    test_old_eigendecomposition_memory_guard()
    test_old_sparse_positive_definite_path_matches_dense_path()
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

@testset "OLD_TRS gamma parameter boundaries" begin
    for γ_2 in (0.0, 1.0)
        @test_throws ArgumentError TrustRegionSubproblemSolvers.solveTrustRegionSubproblemOldApproach(
            0.0,
            [1.0],
            reshape([1.0], 1, 1),
            [0.0],
            0.0,
            γ_2,
            1.0,
        )
    end
end
