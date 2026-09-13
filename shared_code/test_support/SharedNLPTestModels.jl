module SharedNLPTestModels

using JuMP
using NLPModelsJuMP
using Test

export assert_counter_matches,
    assert_oracle_counters_match,
    createComplexConvexNLPModeL1,
    createComplexNLPModeL1,
    createDummyNLPModel,
    createDummyNLPModel2,
    createSimpleConvexNLPModeL,
    createSimpleUnivariateConvexProblem,
    createSinCosNLPModeL1,
    createSinCosNLPModeL2,
    quadratic_nlp

function assert_oracle_counters_match(counter, nlp)
    @test counter.total_function_evaluation == nlp.counters.neval_obj
    @test counter.total_gradient_evaluation == nlp.counters.neval_grad
    @test counter.total_hessian_evaluation == nlp.counters.neval_hess
    return counter
end

function assert_counter_matches(counter, expected::AbstractDict)
    for (field_name, expected_value) in expected
        @test getfield(counter, Symbol(field_name)) == expected_value
    end
    return counter
end

function createDummyNLPModel()
    model = Model()
    starts = [-1.2, 1.0]
    @variable(model, x[index = 1:2], start = starts[index])
    @NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)
    return MathOptNLPModel(model)
end

function createDummyNLPModel2()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (2 * x + y - 1)^2 + x + y + (x^2 - 2 * y^2)^3)
    return MathOptNLPModel(model)
end

function createSimpleUnivariateConvexProblem(
    start::Union{Nothing,Float64} = nothing,
)
    model = Model()
    if start === nothing
        @variable(model, x)
    else
        @variable(model, x, start = start)
    end
    @NLobjective(model, Min, (x - 1)^2)
    return MathOptNLPModel(model)
end

quadratic_nlp(start::Float64) = createSimpleUnivariateConvexProblem(start)

function createSimpleConvexNLPModeL()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (x + y - 1)^2)
    return MathOptNLPModel(model)
end

function createComplexConvexNLPModeL1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (x + y - 1)^2 + x + y + (x - 2 * y)^2)
    return MathOptNLPModel(model)
end

function createComplexNLPModeL1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (2 * x + y - 1)^2 + x + y + (x^2 - 2 * y^2)^2)
    return MathOptNLPModel(model)
end

function createSinCosNLPModeL1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, sin(x) * cos(y))
    return MathOptNLPModel(model)
end

function createSinCosNLPModeL2()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, sin(x) + cos(y))
    return MathOptNLPModel(model)
end

end
