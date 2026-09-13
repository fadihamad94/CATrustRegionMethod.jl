using JuMP
import MathOptInterface as MOI

@testset "UTR MathOptInterface integration" begin
    optimizer = UTR.Optimizer()
    @test MOI.get(optimizer, MOI.SolverName()) == "UTROptimizer"
    @test MOI.get(optimizer, MOI.TerminationStatus()) ==
          MOI.OPTIMIZE_NOT_CALLED
    @test MOI.get(optimizer, MOI.ResultCount()) == 0
    @test MOI.get(
        optimizer,
        MOI.RawOptimizerAttribute("algorithm_params!ρ_0"),
    ) == 1.0

    MOI.set(
        optimizer,
        MOI.RawOptimizerAttribute("algorithm_params!ρ_0"),
        2,
    )
    @test MOI.get(
        optimizer,
        MOI.RawOptimizerAttribute("algorithm_params!ρ_0"),
    ) == 2.0
    @test_throws AssertionError MOI.set(
        optimizer,
        MOI.RawOptimizerAttribute("algorithm_params!η"),
        1 / 32,
    )
    @test_throws ArgumentError MOI.set(
        optimizer,
        MOI.RawOptimizerAttribute("unknown!option"),
        1,
    )
    @test_throws AssertionError MOI.set(
        optimizer,
        MOI.RawOptimizerAttribute("algorithm_params!trust_region_subproblem_solver"),
        "unsupported",
    )
    @test !MOI.get(optimizer, MOI.Silent())
    MOI.set(optimizer, MOI.Silent(), true)
    @test MOI.get(optimizer, MOI.Silent())
    @test MOI.get(optimizer, MOI.TimeLimitSec()) === nothing
    MOI.set(optimizer, MOI.TimeLimitSec(), 2.5)
    @test MOI.get(optimizer, MOI.TimeLimitSec()) == 2.5
    MOI.set(optimizer, MOI.TimeLimitSec(), nothing)
    @test MOI.get(optimizer, MOI.TimeLimitSec()) === nothing
    @test_throws ArgumentError MOI.set(optimizer, MOI.TimeLimitSec(), -1.0)

    getter_optimizer = UTR.Optimizer()
    getter_variable = MOI.add_variable(getter_optimizer)
    quadratic_objective = MOI.ScalarQuadraticFunction(
        MOI.ScalarQuadraticTerm{Float64}[],
        [MOI.ScalarAffineTerm(2.0, getter_variable)],
        1.0,
    )
    MOI.set(
        getter_optimizer,
        MOI.ObjectiveFunction{typeof(quadratic_objective)}(),
        quadratic_objective,
    )
    converted_objective = MOI.get(
        getter_optimizer,
        MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}(),
    )
    @test converted_objective isa MOI.ScalarNonlinearFunction
    @test_throws InexactError MOI.get(
        getter_optimizer,
        MOI.ObjectiveFunction{MOI.VariableIndex}(),
    )

    quadratic_model = Model(UTR.Optimizer)
    set_silent(quadratic_model)
    @variable(quadratic_model, x, start = 0.0)
    @objective(quadratic_model, Min, (x - 1.0)^2)
    optimize!(quadratic_model)
    @test termination_status(quadratic_model) == MOI.LOCALLY_SOLVED
    @test primal_status(quadratic_model) == MOI.FEASIBLE_POINT
    @test result_count(quadratic_model) == 1
    @test value(x) ≈ 1.0 atol = 1e-8
    @test objective_value(quadratic_model) ≈ 0.0 atol = 1e-14
    @test solve_time(quadratic_model) >= 0.0

    # Repeated solves retain the model, while a model mutation invalidates the
    # previous result until optimize! is called again.
    optimize!(quadratic_model)
    @test termination_status(quadratic_model) == MOI.LOCALLY_SOLVED
    set_start_value(x, 2.0)
    @test termination_status(quadratic_model) == MOI.OPTIMIZE_NOT_CALLED
    @test result_count(quadratic_model) == 0
    optimize!(quadratic_model)
    @test termination_status(quadratic_model) == MOI.LOCALLY_SOLVED
    @test value(x) ≈ 1.0 atol = 1e-8

    nonlinear_model = Model(UTR.Optimizer)
    set_silent(nonlinear_model)
    @variable(nonlinear_model, y, start = 0.5)
    @objective(nonlinear_model, Min, (y^2 - 1.0)^2)
    optimize!(nonlinear_model)
    @test termination_status(nonlinear_model) == MOI.LOCALLY_SOLVED
    @test abs(abs(value(y)) - 1.0) <= 1e-6

    legacy_nlp_model = Model(UTR.Optimizer)
    set_silent(legacy_nlp_model)
    @variable(legacy_nlp_model, z, start = 0.0)
    @NLobjective(legacy_nlp_model, Min, (z - 1.0)^2)
    optimize!(legacy_nlp_model)
    @test termination_status(legacy_nlp_model) == MOI.LOCALLY_SOLVED
    @test value(z) ≈ 1.0 atol = 1e-8

    maximization_model = Model(UTR.Optimizer)
    set_silent(maximization_model)
    @variable(maximization_model, w, start = 0.0)
    @objective(maximization_model, Max, -(w - 1.0)^2)
    optimize!(maximization_model)
    @test termination_status(maximization_model) == MOI.INVALID_MODEL
    @test result_count(maximization_model) == 0

    constrained_optimizer = UTR.Optimizer()
    variable = MOI.add_variable(constrained_optimizer)
    evaluator = MOI.Nonlinear.Evaluator(
        let nonlinear = MOI.Nonlinear.Model()
            MOI.Nonlinear.set_objective(
                nonlinear,
                MOI.ScalarNonlinearFunction(
                    :^,
                    Any[variable, 2.0],
                ),
            )
            nonlinear
        end,
        MOI.Nonlinear.SparseReverseMode(),
        [variable],
    )
    MOI.set(
        constrained_optimizer,
        MOI.NLPBlock(),
        MOI.NLPBlockData(
            [MOI.NLPBoundsPair(0.0, Inf)],
            evaluator,
            true,
        ),
    )
    MOI.set(
        constrained_optimizer,
        MOI.ObjectiveSense(),
        MOI.MIN_SENSE,
    )
    MOI.optimize!(constrained_optimizer)
    @test MOI.get(constrained_optimizer, MOI.TerminationStatus()) ==
          MOI.INVALID_MODEL
    @test MOI.get(constrained_optimizer, MOI.ResultCount()) == 0

    time_limited_model = Model(UTR.Optimizer)
    set_silent(time_limited_model)
    set_time_limit_sec(time_limited_model, 0.0)
    @variable(time_limited_model, t, start = 0.0)
    @objective(time_limited_model, Min, (t - 1.0)^2)
    optimize!(time_limited_model)
    @test termination_status(time_limited_model) == MOI.TIME_LIMIT
    @test result_count(time_limited_model) == 0

    step_limited_model = Model(UTR.Optimizer)
    set_silent(step_limited_model)
    set_optimizer_attribute(
        step_limited_model,
        "termination_criteria!STEP_SIZE_LIMIT",
        1.0,
    )
    @variable(step_limited_model, q, start = 0.0)
    @objective(step_limited_model, Min, (q - 1.0)^2)
    optimize!(step_limited_model)
    @test termination_status(step_limited_model) == MOI.SLOW_PROGRESS
    @test result_count(step_limited_model) == 0

    inner_limited_model = Model(UTR.Optimizer)
    set_silent(inner_limited_model)
    set_optimizer_attribute(
        inner_limited_model,
        "termination_criteria!MAX_INNER_ITERATIONS",
        1,
    )
    @variable(inner_limited_model, r, start = 0.5)
    @objective(inner_limited_model, Min, (r^2 - 1.0)^2)
    optimize!(inner_limited_model)
    @test termination_status(inner_limited_model) == MOI.ITERATION_LIMIT
    @test raw_status(inner_limited_model) == "InnerIterationLimit"
    @test primal_status(inner_limited_model) == MOI.FEASIBLE_POINT
    @test result_count(inner_limited_model) == 1

    objective_limited_model = Model(UTR.Optimizer)
    set_silent(objective_limited_model)
    set_optimizer_attribute(
        objective_limited_model,
        "termination_criteria!MINIMUM_OBJECTIVE_FUNCTION",
        1.0,
    )
    @variable(objective_limited_model, p, start = 0.0)
    @objective(objective_limited_model, Min, (p - 1.0)^2)
    optimize!(objective_limited_model)
    @test termination_status(objective_limited_model) == MOI.OBJECTIVE_LIMIT
    @test result_count(objective_limited_model) == 1
    @test objective_value(objective_limited_model) == 1.0

    invalid_objective_limit_model = Model(UTR.Optimizer)
    for invalid_value in (NaN, Inf, -Inf)
        @test_throws AssertionError set_optimizer_attribute(
            invalid_objective_limit_model,
            "termination_criteria!MINIMUM_OBJECTIVE_FUNCTION",
            invalid_value,
        )
    end
end
