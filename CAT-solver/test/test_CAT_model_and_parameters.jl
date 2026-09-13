using LinearAlgebra
using NLPModels
using Test

import CATrustRegionMethod

function test_compute_second_order_model_negative_direction()
    nlp, _, _ = test_create_dummy_problem()
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
    nlp, _, _ = test_create_dummy_problem()
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
    nlp, _, _ = test_create_dummy_problem()
    x_k = [0.0, 0.0]
    d_k = [1.0, 1.0]

    function_value = obj(nlp, x_k)
    gradient_value = grad(nlp, x_k)
    hessian_value = hess(nlp, x_k)
    second_order_model_value =
        CATrustRegionMethod.computeSecondOrderModel(gradient_value, hessian_value, d_k)
    @test second_order_model_value == 100.0 - function_value
end

function test_compute_rho_hat_at_global_minimizer()
    nlp, _, algorithm_params = test_create_dummy_problem()
    x_k = [1.0, 1.0]
    d_k = [-0.0, -0.0]
    fval_current = obj(nlp, x_k)
    fval_next = obj(nlp, x_k + d_k)
    gval_current = grad(nlp, x_k)
    gval_next = grad(nlp, x_k + d_k)
    H = hess(nlp, x_k)

    @test CATrustRegionMethod.compute_ρ_hat(
        fval_current,
        fval_next,
        gval_current,
        gval_next,
        H,
        d_k,
        algorithm_params.θ,
    ) isa Tuple
end

function test_compute_rho_hat_phi_zero()
    nlp, _, algorithm_params = test_create_dummy_problem()
    x_k = nlp.meta.x0
    d_k = [0.02471910112359557, 0.3806741573033706]
    fval_current = obj(nlp, x_k)
    fval_next = obj(nlp, x_k + d_k)
    gval_current = grad(nlp, x_k)
    gval_next = grad(nlp, x_k + d_k)
    H = hess(nlp, x_k)
    rho = CATrustRegionMethod.compute_ρ_hat(
        fval_current,
        fval_next,
        gval_current,
        gval_next,
        H,
        d_k,
        algorithm_params.θ,
    )[1]

    @test norm(rho - 0.980423689675886, 2) <= 1e-3
end

function test_compute_rho_hat_mixed_phi_signs()
    nlp, _, algorithm_params = test_create_dummy_problem2()
    x_k = [0.0, 1.0]
    d_k = [-0.005830328471736362, 0.34323592199917485]
    fval_current = obj(nlp, x_k)
    fval_next = obj(nlp, x_k + d_k)
    gval_current = grad(nlp, x_k)
    gval_next = grad(nlp, x_k + d_k)
    H = hess(nlp, x_k)
    rho = CATrustRegionMethod.compute_ρ_hat(
        fval_current,
        fval_next,
        gval_current,
        gval_next,
        H,
        d_k,
        algorithm_params.θ,
    )[1]

    @test norm(rho - 1.126954013438328, 2) <= 1e-3
end

function test_cat_solve_uses_nlpmodels_constraint_metadata()
    nlp = createSimpleUnivariateConvexProblem(0.0)
    nlp_result = CATrustRegionMethod.CAT_solve(nlp)
    @test nlp_result[2] == CATrustRegionMethod.TerminationStatusCode.OPTIMAL

    model = Model()
    @variable(model, x, start = 0.0)
    @NLobjective(model, Min, (x - 1.0)^2)
    jump_result = CATrustRegionMethod.CAT_solve(model)
    @test jump_result[2] == CATrustRegionMethod.TerminationStatusCode.OPTIMAL

    constrained_model = Model()
    @variable(constrained_model, constrained_x)
    @variable(constrained_model, constrained_y)
    @NLobjective(
        constrained_model,
        Min,
        constrained_x^2 + constrained_y^2,
    )
    @constraint(constrained_model, constrained_x + constrained_y == 1.0)
    @test_throws ErrorException CATrustRegionMethod.CAT_solve(constrained_model)
end


function test_algorithmic_parameter_defaults_and_validation()
    @test Set(keys(CATrustRegionMethod.DEFAULTS["internal"])) == Set(["common"])
    @test CATrustRegionMethod.DEFAULTS["internal"]["common"][
        "power_iteration_max_iterations"
    ] == 20
    @test CATrustRegionMethod.DEFAULT_PRINT_LEVEL == 0
    @test CATrustRegionMethod.AlgorithmicParameters().print_level == 0
    @test CATrustRegionMethod.DEFAULT_TRUST_REGION_SUBPROBLEM_SOLVER ==
          TrustRegionSubproblemSolvers.DIRECT_NEW_SOLVER
    @test CATrustRegionMethod.AlgorithmicParameters().trust_region_subproblem_solver ==
          TrustRegionSubproblemSolvers.DIRECT_NEW_SOLVER
    parameter_defaults = (
        CATrustRegionMethod.DEFAULT_BETA,
        CATrustRegionMethod.DEFAULT_THETA,
        CATrustRegionMethod.DEFAULT_OMEGA_1,
        CATrustRegionMethod.DEFAULT_OMEGA_2,
        CATrustRegionMethod.DEFAULT_GAMMA_1,
        CATrustRegionMethod.DEFAULT_GAMMA_2,
        CATrustRegionMethod.DEFAULT_GAMMA_3,
        CATrustRegionMethod.DEFAULT_XI,
        CATrustRegionMethod.DEFAULT_INITIAL_RADIUS,
        CATrustRegionMethod.DEFAULT_INITIAL_RADIUS_MULTIPLICATIVE_RULE,
        CATrustRegionMethod.DEFAULT_SEED,
        CATrustRegionMethod.DEFAULT_PRINT_LEVEL,
        CATrustRegionMethod.DEFAULT_RADIUS_UPDATE_RULE_APPROACH,
        CATrustRegionMethod.DEFAULT_EVAL_OFFSET,
        CATrustRegionMethod.DEFAULT_TRUST_REGION_SUBPROBLEM_SOLVER,
    )
    @test CATrustRegionMethod.AlgorithmicParameters().dense_hessian_threshold == 0.2
    @test CATrustRegionMethod.AlgorithmicParameters().reuse_sparse_symbolic_factorization
    disabled_reuse_parameters = CATrustRegionMethod.AlgorithmicParameters()
    disabled_reuse_parameters.reuse_sparse_symbolic_factorization = false
    @test !disabled_reuse_parameters.reuse_sparse_symbolic_factorization

    zero_b_k_parameters = CATrustRegionMethod.AlgorithmicParameters(
        parameter_defaults[1:7]...,
        0.0,
        parameter_defaults[9:13]...,
        0.0,
        parameter_defaults[15],
    )
    @test zero_b_k_parameters.ξ == 0.0
    @test zero_b_k_parameters.eval_offset == 0.0
    @test_throws AssertionError CATrustRegionMethod.AlgorithmicParameters(
        parameter_defaults[1:7]...,
        -0.1,
        parameter_defaults[9:end]...,
    )
    @test_throws AssertionError CATrustRegionMethod.AlgorithmicParameters(
        parameter_defaults[1:13]...,
        -0.1,
        parameter_defaults[15],
    )
    @test_throws AssertionError CATrustRegionMethod.AlgorithmicParameters(
        parameter_defaults...,
        -0.1,
    )

    defaults = CATrustRegionMethod.AlgorithmicParameters()
    default_values =
        map(field -> getfield(defaults, field), fieldnames(typeof(defaults)))
    @test_throws AssertionError CATrustRegionMethod.AlgorithmicParameters(
        default_values[1:14]...,
        "NEW",
    )
    for (field_index, boundary) in (
        (5, 0.0),
        (5, 1.0),
        (6, 0.0),
        (6, 1.0),
        (7, 0.0),
        (7, 1.0),
    )
        boundary_values = collect(default_values)
        boundary_values[field_index] = boundary
        @test_throws AssertionError CATrustRegionMethod.AlgorithmicParameters(
            boundary_values...,
        )
    end

    for solver in ("OLD", TrustRegionSubproblemSolvers.DIRECT_NEW_SOLVER)
        for (boundary_γ_1, boundary_γ_2, boundary_γ_3) in (
            (0.0, defaults.γ_2, defaults.γ_3),
            (1.0, defaults.γ_2, defaults.γ_3),
            (defaults.γ_1, 0.0, defaults.γ_3),
            (defaults.γ_1, 1.0, defaults.γ_3),
            (defaults.γ_1, defaults.γ_2, 0.0),
            (defaults.γ_1, defaults.γ_2, 1.0),
        )
            @test_throws AssertionError CATrustRegionMethod.validateTrustRegionSubproblemSolverParameters(
                solver,
                boundary_γ_1,
                boundary_γ_2,
                boundary_γ_3,
            )
        end
    end

    for field in (
        :gradient_termination_tolerance,
        :MAX_TIME,
        :STEP_SIZE_LIMIT,
        :MINIMUM_OBJECTIVE_FUNCTION,
    ), invalid_value in (NaN, Inf, -Inf)
        mutated_criteria = CATrustRegionMethod.TerminationCriteria()
        setfield!(mutated_criteria, field, invalid_value)
        @test_throws AssertionError CATrustRegionMethod.validateTerminationCriteria(
            mutated_criteria,
        )
    end

    nlp, mutated_criteria, valid_parameters = test_create_dummy_problem()
    mutated_criteria.MINIMUM_OBJECTIVE_FUNCTION = NaN
    @test_throws AssertionError CATrustRegionMethod.optimize(
        nlp,
        mutated_criteria,
        valid_parameters,
        copy(nlp.meta.x0),
        0.0,
    )
end

@testset "CAT model and parameter utilities" begin
    test_compute_second_order_model_negative_direction()
    test_compute_second_order_model_zero_direction()
    test_compute_second_order_model_positive_direction()
    test_compute_rho_hat_at_global_minimizer()
    test_compute_rho_hat_phi_zero()
    test_compute_rho_hat_mixed_phi_signs()
    test_cat_solve_uses_nlpmodels_constraint_metadata()
    test_algorithmic_parameter_defaults_and_validation()
    @test CATrustRegionMethod.TerminationCriteria().iterative_refinement_max_iterations == 3
end
