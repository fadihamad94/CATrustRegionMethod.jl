using JuMP
using LinearAlgebra
using NLPModels
using NLPModelsJuMP
using CATrustRegionShared

include(
    joinpath(
        pkgdir(CATrustRegionShared),
        "test_support",
        "SharedNLPTestModels.jl",
    ),
)
using .SharedNLPTestModels: quadratic_nlp

function quartic_nlp(start::Float64)
    model = Model()
    @variable(model, x, start = start)
    @NLobjective(model, Min, (x^2 - 1.0)^2)
    return MathOptNLPModel(model)
end

function quiet_parameters()
    parameters = UTR.AlgorithmicParameters()
    parameters.print_level = -1
    return parameters
end

@testset "UTR direct integration" begin
    nlp = quadratic_nlp(0.0)
    solution,
    status,
    history,
    counter,
    iterations,
    elapsed = UTR.UTR_solve(
        nlp,
        UTR.TerminationCriteria(100, 100, 1e-8),
        quiet_parameters(),
    )
    @test status == UTR.TerminationStatusCode.OPTIMAL
    @test solution ≈ [1.0] atol = 1e-8
    @test history[end, :fval] ≈ 0.0 atol = 1e-14
    @test history[end, :gradnorm] < 1e-8
    @test names(history) == [
        "k",
        "fval",
        "gradnorm",
        "min_gradnorm_fval",
        "min_gradnorm",
    ]
    @test history[end, :min_gradnorm_fval] ≈ 0.0 atol = 1e-14
    @test history[end, :min_gradnorm] < 1e-8
    @test iterations == size(history, 1)
    @test elapsed >= 0.0
    @test counter.total_function_evaluation == nlp.counters.neval_obj
    @test counter.total_gradient_evaluation == nlp.counters.neval_grad
    @test counter.total_hessian_evaluation == nlp.counters.neval_hess
    @test counter.total_number_subproblem_solves >= iterations
    @test UTR.assertFactorizationAccounting(counter) === counter

    stationary_nlp = quadratic_nlp(1.0)
    stationary_result = UTR.UTR_solve(
        stationary_nlp,
        UTR.TerminationCriteria(),
        quiet_parameters(),
    )
    @test stationary_result[2] == UTR.TerminationStatusCode.OPTIMAL
    @test stationary_result[1] == [1.0]
    @test stationary_result[4].total_hessian_evaluation == 0
    @test stationary_nlp.counters.neval_hess == 0
    @test size(stationary_result[3], 1) == 1

    equality_nlp = quadratic_nlp(0.5)
    equality_result = UTR.UTR_solve(
        equality_nlp,
        UTR.TerminationCriteria(10, 100, 1.0),
        quiet_parameters(),
    )
    @test equality_result[2] == UTR.TerminationStatusCode.OPTIMAL
    @test equality_result[4].total_hessian_evaluation > 0

    nonconvex_nlp = quartic_nlp(0.5)
    nonconvex_result = UTR.UTR_solve(
        nonconvex_nlp,
        UTR.TerminationCriteria(100, 100, 1e-7),
        quiet_parameters(),
    )
    @test nonconvex_result[2] == UTR.TerminationStatusCode.OPTIMAL
    @test abs(abs(only(nonconvex_result[1])) - 1.0) <= 1e-6
    @test nonconvex_result[4].total_number_rejected_trials > 0
    @test nonconvex_result[4].total_number_subproblem_solves >
          nonconvex_result[5]

    inner_limited = UTR.UTR_solve(
        quartic_nlp(0.5),
        UTR.TerminationCriteria(100, 1, 1e-7),
        quiet_parameters(),
    )
    @test inner_limited[2] ==
          UTR.TerminationStatusCode.INNER_ITERATION_LIMIT
    @test inner_limited[4].total_number_rejected_trials == 1
    @test inner_limited[4].total_number_subproblem_solves == 2
    @test inner_limited[5] == 2
    @test size(inner_limited[3], 1) == 2
    @test inner_limited[3][end, :fval] ==
          inner_limited[3][end - 1, :fval]
    @test inner_limited[3][end, :gradnorm] ==
          inner_limited[3][end - 1, :gradnorm]
    @test inner_limited[3][end, :min_gradnorm] <=
          inner_limited[3][end, :gradnorm]

    rejected_best_parameters = quiet_parameters()
    rejected_best_parameters.ρ_0 = 1.0e-4
    rejected_best_parameters.η = 0.001
    rejected_best_parameters.ξ = 0.26
    rejected_best = UTR.UTR_solve(
        quartic_nlp(-2.0),
        UTR.TerminationCriteria(1, 1, 1.0e-12),
        rejected_best_parameters,
    )
    @test rejected_best[2] ==
          UTR.TerminationStatusCode.INNER_ITERATION_LIMIT
    @test size(rejected_best[3], 1) == 1
    @test rejected_best[3][end, :fval] == 9.0
    @test rejected_best[3][end, :gradnorm] == 24.0
    @test rejected_best[3][end, :min_gradnorm_fval] <
          rejected_best[3][end, :fval]
    @test rejected_best[3][end, :min_gradnorm] <
          rejected_best[3][end, :gradnorm]

    outer_limited = UTR.UTR_solve(
        quadratic_nlp(0.0),
        UTR.TerminationCriteria(1, 100, 1e-12),
        quiet_parameters(),
    )
    @test outer_limited[2] == UTR.TerminationStatusCode.ITERATION_LIMIT
    @test outer_limited[5] == 1

    step_limited = UTR.UTR_solve(
        quadratic_nlp(0.0),
        UTR.TerminationCriteria(100, 100, 1e-8, 18_000.0, 0.8),
        quiet_parameters(),
    )
    @test step_limited[2] == UTR.TerminationStatusCode.STEP_SIZE_LIMIT
    @test step_limited[4].total_number_subproblem_solves == 0

    time_limited = UTR.UTR_solve(
        quadratic_nlp(0.0),
        UTR.TerminationCriteria(100, 100, 1e-8, eps(Float64)),
        quiet_parameters(),
    )
    @test time_limited[2] == UTR.TerminationStatusCode.TIME_LIMIT

    tiny_penalty_parameters = quiet_parameters()
    tiny_penalty_parameters.ρ_0 = nextfloat(0.0)
    nonfinite_radius = UTR.UTR_solve(
        quadratic_nlp(0.0),
        UTR.TerminationCriteria(),
        tiny_penalty_parameters,
    )
    @test nonfinite_radius[2] == UTR.TerminationStatusCode.NUMERICAL_ERROR
    @test nonfinite_radius[4].total_number_subproblem_solves == 0

    constrained_model = Model()
    @variable(constrained_model, x, start = 0.0)
    @NLobjective(constrained_model, Min, x^2)
    @constraint(constrained_model, x >= -1.0)
    constrained_nlp = MathOptNLPModel(constrained_model)
    @test_throws ArgumentError UTR.UTR_solve(constrained_nlp)
end
