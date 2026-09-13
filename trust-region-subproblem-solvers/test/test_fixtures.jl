using CATrustRegionShared
using JuMP
using NLPModelsJuMP

include(
    joinpath(
        pkgdir(CATrustRegionShared),
        "test_support",
        "SharedNLPTestModels.jl",
    ),
)
using .SharedNLPTestModels

mutable struct TestAlgorithmicParameters
    r_1::Float64
end

function old_solver_test_problem(nlp)
    return nlp, nothing, TestAlgorithmicParameters(0.5)
end

test_create_dummy_problem() = old_solver_test_problem(createDummyNLPModel())
test_create_dummy_problem2() = old_solver_test_problem(createDummyNLPModel2())
test_create_simple_convex_nlp_model() =
    old_solver_test_problem(createSimpleConvexNLPModeL())

function create_hard_case_nlp(objective)
    model = Model()
    @variable(model, x)
    @variable(model, y)
    objective(model, x, y)
    return MathOptNLPModel(model)
end

function test_create_simple_univariate_convex_model()
    nlp, _, _ = old_solver_test_problem(createSimpleUnivariateConvexProblem())
    return nlp, nothing, TestAlgorithmicParameters(0.5)
end

function test_create_simple_univariate_convex_model_solved_same_as_Newton()
    nlp, _, _ = old_solver_test_problem(createSimpleUnivariateConvexProblem())
    return nlp, nothing, TestAlgorithmicParameters(2.0)
end

function test_create_hard_case_using_simple_univariate_convex_model()
    model = Model()
    @variable(model, x)
    @NLobjective(model, Min, -x^2)
    return MathOptNLPModel(model), nothing, TestAlgorithmicParameters(1.0)
end

function test_create_hard_case_using_simple_bivariate_convex_model()
    nlp = create_hard_case_nlp() do model, x, y
        @NLobjective(model, Min, -x^2 - y^2)
    end
    return nlp, nothing, TestAlgorithmicParameters(1.0)
end

function test_create_hard_case_using_bivariate_convex_model_1()
    nlp = create_hard_case_nlp() do model, x, y
        @NLobjective(model, Min, -x^2 - 2 * y^2)
    end
    return nlp, nothing, TestAlgorithmicParameters(1.0)
end

function test_create_hard_case_using_bivariate_convex_model_2()
    nlp = create_hard_case_nlp() do model, x, y
        @NLobjective(model, Min, x^2 + 0.01 * x - y^2)
    end
    return nlp, nothing, TestAlgorithmicParameters(1.0)
end

function test_create_hard_case_using_bivariate_convex_model_3()
    return (
        createHardCaseUsingSimpleBivariateConvexProblem3(),
        nothing,
        TestAlgorithmicParameters(1.0),
    )
end

function createHardCaseUsingSimpleBivariateConvexProblem3()
    return create_hard_case_nlp() do model, x, y
        @NLobjective(model, Min, x^2 - 10 * x * y + y^2)
    end
end
