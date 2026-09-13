using MathOptInterface
######################
##### ROSENBROOK #####
######################
function rosenbrook1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (2.0 - x)^2 + 100 * (y - x^2)^2)
    return model
end

function rosenbrook2()
    model = Model()
    @variable(model, x >= 0.0)
    @variable(model, y >= 0.0)
    @NLobjective(model, Min, (2.0 - x)^2 + 100 * (y - x^2)^2)
    @constraint(model, x + y >= 0.1)
    @NLconstraint(model, x * y + x >= 0.1)
    return model
end

function createHardCaseUsingSimpleBivariateConvexProblemJuMP()
    model = Model(CATrustRegionMethod.Optimizer)
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x^2 - 10 * x * y + y^2)
    return model
end

######################
####Utility Method####
######################
function attachSolverWithAttributesToJuMPModel(model::Model, options::Dict{String,Any})
    set_optimizer(model, CATrustRegionMethod.Optimizer)
    for (name, value) in options
        sname = string(name)
        set_optimizer_attribute(model, sname, value)
    end
end

function optimize_rosenbrook1_model_MOI_wrapper_with_default_arguments()
    default_β = 0.1
    default_θ = 0.1
    default_ω_1 = 8.0
    default_ω_2 = 16.0
    default_γ_1 = 1e-2
    default_γ_2 = 0.8
    default_r_1 = 0.0
    default_print_level = 0
    default_max_iterations = 100000
    default_gradient_termination_tolerance = 1e-5
    default_max_time = 5 * 60 * 60.0
    default_step_size_limit = 2.0e-16
    options = Dict{String,Any}(
        "algorithm_params!r_1" => default_r_1,
        "algorithm_params!β" => default_β,
        "algorithm_params!ω_2" => default_ω_2,
        "algorithm_params!print_level" => default_print_level,
        "termination_criteria!MAX_ITERATIONS" => default_max_iterations,
        "termination_criteria!gradient_termination_tolerance" =>
            default_gradient_termination_tolerance,
    )
    model = rosenbrook1()
    attachSolverWithAttributesToJuMPModel(model, options)

    #Test using JUMP
    optimize!(model)
    x = JuMP.value.(model[:x])
    y = JuMP.value.(model[:y])
    status = MOI.get(model, MOI.TerminationStatus())
    @test status == MOI.OPTIMAL
    @test MOI.get(model, MOI.RawStatusString()) == "Optimal"
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test abs(MOI.get(model, MOI.ObjectiveValue()) - 4.68876e-19) <= 1e-3

    @test MOI.get(model, MOI.Silent()) == true
    @test MOI.get(model, MOI.RawOptimizerAttribute("time_limit")) == default_max_time

    # Retrieve the solver instance
    optimizer = backend(model).optimizer.model

    nlp = MathOptNLPModel(model)
    termination_criteria = CATrustRegionMethod.TerminationCriteria()
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters()

    x_k, status, iteration_stats, algorithm_counter, itr =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, nlp.meta.x0, 0.0)

    assert_oracle_counters_match(algorithm_counter, nlp)

    @test algorithm_counter.total_function_evaluation <= 35
    @test algorithm_counter.total_gradient_evaluation <= 25
    @test algorithm_counter.total_hessian_evaluation <= 25
    @test algorithm_counter.total_number_factorizations <= 110

    @test x_k == [x, y]
    @test itr == optimizer.inner.itr
    @test x_k == optimizer.inner.x
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
    @test iteration_stats == optimizer.inner.iteration_stats

    @test algorithm_counter.total_function_evaluation ==
          optimizer.inner.algorithm_counter.total_function_evaluation
    @test algorithm_counter.total_gradient_evaluation ==
          optimizer.inner.algorithm_counter.total_gradient_evaluation
    @test algorithm_counter.total_hessian_evaluation ==
          optimizer.inner.algorithm_counter.total_hessian_evaluation
    @test algorithm_counter.total_number_factorizations ==
          optimizer.inner.algorithm_counter.total_number_factorizations
    @test algorithm_counter.total_number_factorizations_findinterval ==
          optimizer.inner.algorithm_counter.total_number_factorizations_findinterval
    @test algorithm_counter.total_number_factorizations_bisection ==
          optimizer.inner.algorithm_counter.total_number_factorizations_bisection
    @test algorithm_counter.total_number_factorizations_compute_search_direction ==
          optimizer.inner.algorithm_counter.total_number_factorizations_compute_search_direction
    @test algorithm_counter.total_number_factorizations_inverse_power_iteration ==
          optimizer.inner.algorithm_counter.total_number_factorizations_inverse_power_iteration

    @test optimizer.inner.algorithm_params.β == default_β
    @test optimizer.inner.algorithm_params.θ == default_θ
    @test optimizer.inner.algorithm_params.ω_1 == default_ω_1
    @test optimizer.inner.algorithm_params.ω_2 == default_ω_2
    @test optimizer.inner.algorithm_params.γ_1 == default_γ_1
    @test optimizer.inner.algorithm_params.γ_2 == default_γ_2
    @test optimizer.inner.algorithm_params.r_1 == default_r_1
    @test optimizer.inner.algorithm_params.print_level == default_print_level
    @test optimizer.inner.termination_criteria.MAX_ITERATIONS == default_max_iterations
    @test optimizer.inner.termination_criteria.gradient_termination_tolerance ==
          default_gradient_termination_tolerance
    @test optimizer.inner.termination_criteria.MAX_TIME == default_max_time
    @test optimizer.inner.termination_criteria.STEP_SIZE_LIMIT == default_step_size_limit
end

function optimize_rosenbrook1_model_MOI_wrapper_with_user_specified_arguments()
    β = 0.2
    ω_2 = 8.0
    r_1 = 100.0
    print_level = -1
    MAX_ITERATIONS = 10
    gradient_termination_tolerance = 1e-3
    iterative_refinement_max_iterations = 2
    dense_hessian_threshold = 0.35
    reuse_sparse_symbolic_factorization = false
    options = Dict{String,Any}(
        "algorithm_params!r_1" => r_1,
        "algorithm_params!β" => β,
        "algorithm_params!ω_2" => ω_2,
        "algorithm_params!print_level" => print_level,
        "algorithm_params!dense_hessian_threshold" => dense_hessian_threshold,
        "algorithm_params!reuse_sparse_symbolic_factorization" =>
            reuse_sparse_symbolic_factorization,
        "termination_criteria!MAX_ITERATIONS" => MAX_ITERATIONS,
        "termination_criteria!gradient_termination_tolerance" =>
            gradient_termination_tolerance,
        "termination_criteria!iterative_refinement_max_iterations" =>
            iterative_refinement_max_iterations,
    )
    model = rosenbrook1()
    attachSolverWithAttributesToJuMPModel(model, options)

    #Test using JUMP (UserLimit due to MAX_ITERATIONS = 10)
    optimize!(model)
    x = JuMP.value.(model[:x])
    y = JuMP.value.(model[:y])
    status = MOI.get(model, MOI.TerminationStatus())
    @test status == MOI.OTHER_LIMIT

    # Retrieve the solver instance
    optimizer = backend(model).optimizer.model

    nlp = MathOptNLPModel(model)
    termination_criteria = CATrustRegionMethod.TerminationCriteria()
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters()

    algorithm_params.β = β
    algorithm_params.ω_2 = ω_2
    algorithm_params.r_1 = r_1
    algorithm_params.print_level = print_level
    termination_criteria.MAX_ITERATIONS = MAX_ITERATIONS
    termination_criteria.gradient_termination_tolerance = gradient_termination_tolerance
    termination_criteria.iterative_refinement_max_iterations =
        iterative_refinement_max_iterations
    x_k, status, iteration_stats, algorithm_counter, itr =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, nlp.meta.x0, 0.0)

    assert_oracle_counters_match(algorithm_counter, nlp)

    @test algorithm_counter.total_function_evaluation <= 15
    @test algorithm_counter.total_gradient_evaluation <= 10
    @test algorithm_counter.total_hessian_evaluation <= 10
    @test algorithm_counter.total_number_factorizations <= 45

    @test x_k == [x, y]
    @test itr == optimizer.inner.itr
    @test x_k == optimizer.inner.x
    @test status == CATrustRegionMethod.TerminationStatusCode.ITERATION_LIMIT
    @test iteration_stats == optimizer.inner.iteration_stats

    @test algorithm_counter.total_function_evaluation ==
          optimizer.inner.algorithm_counter.total_function_evaluation
    @test algorithm_counter.total_gradient_evaluation ==
          optimizer.inner.algorithm_counter.total_gradient_evaluation
    @test algorithm_counter.total_hessian_evaluation ==
          optimizer.inner.algorithm_counter.total_hessian_evaluation
    @test algorithm_counter.total_number_factorizations ==
          optimizer.inner.algorithm_counter.total_number_factorizations
    @test algorithm_counter.total_number_factorizations_findinterval ==
          optimizer.inner.algorithm_counter.total_number_factorizations_findinterval
    @test algorithm_counter.total_number_factorizations_bisection ==
          optimizer.inner.algorithm_counter.total_number_factorizations_bisection
    @test algorithm_counter.total_number_factorizations_compute_search_direction ==
          optimizer.inner.algorithm_counter.total_number_factorizations_compute_search_direction
    @test algorithm_counter.total_number_factorizations_inverse_power_iteration ==
          optimizer.inner.algorithm_counter.total_number_factorizations_inverse_power_iteration

    @test optimizer.inner.algorithm_params.β == β
    @test optimizer.inner.algorithm_params.ω_2 == ω_2
    @test optimizer.inner.algorithm_params.r_1 == r_1
    @test optimizer.inner.algorithm_params.dense_hessian_threshold == dense_hessian_threshold
    @test optimizer.inner.algorithm_params.reuse_sparse_symbolic_factorization ==
          reuse_sparse_symbolic_factorization
    @test optimizer.inner.algorithm_params.print_level == print_level
    @test optimizer.inner.termination_criteria.MAX_ITERATIONS == MAX_ITERATIONS
    @test optimizer.inner.termination_criteria.gradient_termination_tolerance ==
          gradient_termination_tolerance
    @test optimizer.inner.termination_criteria.iterative_refinement_max_iterations ==
          iterative_refinement_max_iterations
end

function optimize_model_with_constraints_failure_expected()
    model = rosenbrook2()
    set_optimizer(model, CATrustRegionMethod.Optimizer)
    @test_throws MOI.UnsupportedConstraint{
        MathOptInterface.VariableIndex,
        MathOptInterface.GreaterThan{Float64},
    } optimize!(model)
end

function unsupported_nlp_constraints_and_max_sense_are_rejected()
    optimizer = CATrustRegionMethod.Optimizer()
    constrained_nlp_data = MOI.NLPBlockData(
        [MOI.NLPBoundsPair(0.0, Inf)],
        CATrustRegionMethod.EmptyNLPEvaluator(),
        true,
    )

    @test_throws MOI.SetAttributeNotAllowed MOI.set(
        optimizer,
        MOI.NLPBlock(),
        constrained_nlp_data,
    )
    @test isempty(optimizer.nlp_data.constraint_bounds)

    @test MOI.get(optimizer, MOI.ObjectiveSense()) == MOI.FEASIBILITY_SENSE
    @test_throws MOI.SetAttributeNotAllowed MOI.set(
        optimizer,
        MOI.ObjectiveSense(),
        MOI.MAX_SENSE,
    )
    @test MOI.get(optimizer, MOI.ObjectiveSense()) == MOI.FEASIBILITY_SENSE

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    @test MOI.get(optimizer, MOI.ObjectiveSense()) == MOI.MIN_SENSE
end

function direct_scalar_objectives_are_optimized()
    quadratic_model = Model(CATrustRegionMethod.Optimizer)
    set_silent(quadratic_model)
    @variable(quadratic_model, x, start = -3.0)
    @objective(quadratic_model, Min, (x - 2.0)^2 + 5.0)

    optimize!(quadratic_model)

    @test termination_status(quadratic_model) == MOI.OPTIMAL
    @test value(x) ≈ 2.0 atol = 1.0e-8
    @test objective_value(quadratic_model) ≈ 5.0 atol = 1.0e-12
    @test objective_value(quadratic_model) ≈
          (value(x) - 2.0)^2 + 5.0 atol = 1.0e-12

    # The temporary NLP evaluator must not replace the stored scalar objective.
    optimize!(quadratic_model)
    @test value(x) ≈ 2.0 atol = 1.0e-8
    @objective(quadratic_model, Min, (x + 1.0)^2 + 7.0)
    optimize!(quadratic_model)
    @test value(x) ≈ -1.0 atol = 1.0e-8
    @test objective_value(quadratic_model) ≈ 7.0 atol = 1.0e-12

    nonlinear_model = Model(CATrustRegionMethod.Optimizer)
    set_silent(nonlinear_model)
    @variable(nonlinear_model, y, start = 0.5)
    @objective(nonlinear_model, Min, (y^2 - 1.0)^2 + 2.0)

    optimize!(nonlinear_model)

    @test termination_status(nonlinear_model) == MOI.OPTIMAL
    @test abs(abs(value(y)) - 1.0) <= 1.0e-6
    @test objective_value(nonlinear_model) ≈
          (value(y)^2 - 1.0)^2 + 2.0 atol = 1.0e-12
end

function direct_objective_storage_obeys_moi()
    optimizer = CATrustRegionMethod.Optimizer()
    variable = MOI.add_variable(optimizer)
    objective = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(1.0, variable, variable)],
        [MOI.ScalarAffineTerm(-4.0, variable)],
        4.0,
    )
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        optimizer,
        MOI.ObjectiveFunction{typeof(objective)}(),
        objective,
    )

    @test MOI.get(optimizer, MOI.ObjectiveFunctionType()) == typeof(objective)
    @test MOI.get(
        optimizer,
        MOI.ObjectiveFunction{typeof(objective)}(),
    ) == objective
    @test MOI.get(
        optimizer,
        MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(),
    ) isa MOI.ScalarNonlinearFunction

    # MOI defines FEASIBILITY_SENSE as removing a scalar objective. Since CAT is
    # unconstrained, the resulting zero-objective model returns its starting point.
    MOI.set(optimizer, MOI.VariablePrimalStart(), variable, 3.0)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
    @test_throws MOI.GetAttributeNotAllowed MOI.get(
        optimizer,
        MOI.ObjectiveFunctionType(),
    )
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(optimizer, MOI.VariablePrimal(), variable) == 3.0
    @test MOI.get(optimizer, MOI.ObjectiveValue()) == 0.0
end

function rejected_trial_moi_result_is_coherent()
    model = Model(CATrustRegionMethod.Optimizer)
    set_silent(model)
    @variable(model, x, start = 0.0)
    @NLobjective(
        model,
        Min,
        -1.500000003 * x^4 + 2.000000004 * x^3 + 0.5 * x^2 - x,
    )

    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test value(x) ≈ 1.0 atol = 1.0e-12
    @test objective_value(model) ≈ 1.0e-9 atol = 1.0e-12
    optimizer = backend(model).optimizer.model
    @test optimizer.inner.x ≈ [1.0] atol = 1.0e-12
    @test optimizer.inner.obj_val ≈ 1.0e-9 atol = 1.0e-12
    @test optimizer.inner.grad_val <=
          optimizer.inner.termination_criteria.gradient_termination_tolerance
end

function optimizeHardCaseUsingSimpleBivariateConvexProblem()
    model = createHardCaseUsingSimpleBivariateConvexProblemJuMP()

    #Test using JUMP
    optimize!(model)
    x = JuMP.value.(model[:x])
    y = JuMP.value.(model[:y])
    status = MOI.get(model, MOI.TerminationStatus())
    @test status == MOI.OPTIMAL
    # Retrieve the solver instance
    optimizer = backend(model).optimizer.model

    nlp = MathOptNLPModel(model)
    termination_criteria = CATrustRegionMethod.TerminationCriteria()
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters()

    x_k, status, iteration_stats, algorithm_counter, itr =
        CATrustRegionMethod.optimize(nlp, termination_criteria, algorithm_params, nlp.meta.x0, 0.0)

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

    @test algorithm_counter.total_function_evaluation ==
          computation_stats["total_function_evaluation"]
    @test algorithm_counter.total_gradient_evaluation ==
          computation_stats["total_gradient_evaluation"]
    @test algorithm_counter.total_hessian_evaluation ==
          computation_stats["total_hessian_evaluation"]
    @test algorithm_counter.total_number_factorizations ==
          computation_stats["total_number_factorizations"]
    @test algorithm_counter.total_number_factorizations_findinterval ==
          computation_stats["total_number_factorizations_findinterval"]
    @test algorithm_counter.total_number_factorizations_bisection ==
          computation_stats["total_number_factorizations_bisection"]
    @test algorithm_counter.total_number_factorizations_compute_search_direction ==
          computation_stats["total_number_factorizations_compute_search_direction"]
    @test algorithm_counter.total_number_factorizations_inverse_power_iteration ==
          computation_stats["total_number_factorizations_inverse_power_iteration"]


    @test x_k == [x, y]
    @test itr == optimizer.inner.itr
    @test x_k == optimizer.inner.x
    @test status == CATrustRegionMethod.TerminationStatusCode.OPTIMAL
    @test iteration_stats == optimizer.inner.iteration_stats

    @test algorithm_counter.total_function_evaluation ==
          optimizer.inner.algorithm_counter.total_function_evaluation
    @test algorithm_counter.total_gradient_evaluation ==
          optimizer.inner.algorithm_counter.total_gradient_evaluation
    @test algorithm_counter.total_hessian_evaluation ==
          optimizer.inner.algorithm_counter.total_hessian_evaluation
    @test algorithm_counter.total_number_factorizations ==
          optimizer.inner.algorithm_counter.total_number_factorizations
    @test algorithm_counter.total_number_factorizations_findinterval ==
          optimizer.inner.algorithm_counter.total_number_factorizations_findinterval
    @test algorithm_counter.total_number_factorizations_bisection ==
          optimizer.inner.algorithm_counter.total_number_factorizations_bisection
    @test algorithm_counter.total_number_factorizations_compute_search_direction ==
          optimizer.inner.algorithm_counter.total_number_factorizations_compute_search_direction
    @test algorithm_counter.total_number_factorizations_inverse_power_iteration ==
          optimizer.inner.algorithm_counter.total_number_factorizations_inverse_power_iteration
end

function optimize_rosenbrook1_model_MOI_wrapper_with_user_specified_attributes()
    # time_limit and output_flag
    β = 0.2
    ω_2 = 8.0
    r_1 = 100.0
    print_level = -1
    MAX_ITERATIONS = 100
    MAX_TIME = 5 * 60.0
    gradient_termination_tolerance = 1e-3
    options = Dict{String,Any}(
        "algorithm_params!r_1" => r_1,
        "algorithm_params!β" => β,
        "algorithm_params!ω_2" => ω_2,
        "algorithm_params!print_level" => print_level,
        "termination_criteria!MAX_ITERATIONS" => MAX_ITERATIONS,
        "termination_criteria!MAX_TIME" => MAX_TIME,
        "termination_criteria!gradient_termination_tolerance" =>
            gradient_termination_tolerance,
    )
    model = rosenbrook1()
    attachSolverWithAttributesToJuMPModel(model, options)

    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.TimeLimitSec(), MAX_TIME)
    @test MOI.get(model, MOI.Silent()) == true
    @test MOI.get(model, MOI.TimeLimitSec()) == MAX_TIME
    @test MOI.get(model, MOI.RawOptimizerAttribute("time_limit")) == MAX_TIME

    optimize!(model)

    @test MOI.get(model, MOI.Silent()) == true
    @test MOI.get(model, MOI.TimeLimitSec()) == MAX_TIME
    @test MOI.get(model, MOI.RawOptimizerAttribute("time_limit")) == MAX_TIME

    MOI.set(model, MOI.Silent(), false)
    @test MOI.get(model, MOI.Silent()) == false
    MOI.set(model, MOI.RawOptimizerAttribute("time_limit"), 2 * MAX_TIME)
    @test MOI.get(model, MOI.TimeLimitSec()) == 2 * MAX_TIME
    @test MOI.get(model, MOI.RawOptimizerAttribute("time_limit")) == 2 * MAX_TIME
    MOI.set(model, MOI.RawOptimizerAttribute("algorithm_params!r_1"), 2 * r_1)

    # Retrieve the solver instance
    optimizer = backend(model).optimizer.model
    @test optimizer.inner.termination_criteria.MAX_TIME == MAX_TIME
    @test optimizer.inner.algorithm_params.print_level == -1
    @test optimizer.inner.algorithm_params.r_1 == r_1

    # call optimize again (the previous modifications should be reflected in the optimizer)
    optimize!(model)
    # Retrieve the solver instance
    optimizer = backend(model).optimizer.model
    @test optimizer.inner.termination_criteria.MAX_TIME == 2 * MAX_TIME
    @test optimizer.inner.algorithm_params.print_level == 0
    @test optimizer.inner.algorithm_params.r_1 == 2 * r_1

    MOI.set(model, MOI.TimeLimitSec(), nothing)
    @test MOI.get(model, MOI.TimeLimitSec()) === nothing
    @test_throws ArgumentError MOI.set(model, MOI.TimeLimitSec(), -1.0)
    @test_throws ArgumentError MOI.set(model, MOI.TimeLimitSec(), Inf)
end

function nonfinite_termination_attributes_are_rejected()
    for field in (
        "gradient_termination_tolerance",
        "MAX_TIME",
        "STEP_SIZE_LIMIT",
        "MINIMUM_OBJECTIVE_FUNCTION",
    ), invalid_value in (NaN, Inf, -Inf)
        model = CATrustRegionMethod.Optimizer()
        @test_throws AssertionError MOI.set(
            model,
            MOI.RawOptimizerAttribute("termination_criteria!$field"),
            invalid_value,
        )
        @test isempty(model.options)
    end
end

function optimize_models_MOI_wrapper()
    optimize_rosenbrook1_model_MOI_wrapper_with_default_arguments()
    optimize_rosenbrook1_model_MOI_wrapper_with_user_specified_arguments()
    optimize_model_with_constraints_failure_expected()
    unsupported_nlp_constraints_and_max_sense_are_rejected()
    direct_scalar_objectives_are_optimized()
    direct_objective_storage_obeys_moi()
    rejected_trial_moi_result_is_coherent()
    optimizeHardCaseUsingSimpleBivariateConvexProblem()
    optimize_rosenbrook1_model_MOI_wrapper_with_user_specified_attributes()
    nonfinite_termination_attributes_are_rejected()
end

@testset "optimization_using_MOI_wrapper" begin
    optimize_models_MOI_wrapper()
end
