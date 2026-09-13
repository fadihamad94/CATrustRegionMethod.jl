function evaluate_function(nlp, x::Vector{Float64}, counter)
    increment!(counter, :total_function_evaluation)
    if nlp isa NLPModels.AbstractNLPModel
        return NLPModels.obj(nlp, x)
    end
    return MOI.eval_objective(nlp.evaluator, x)
end

function evaluate_gradient(nlp, x::Vector{Float64}, counter)
    increment!(counter, :total_gradient_evaluation)
    if nlp isa NLPModels.AbstractNLPModel
        return NLPModels.grad(nlp, x)
    end
    :Grad in MOI.features_available(nlp.evaluator) ||
        throw(ErrorException("Gradient computation is not supported."))
    gradient = zeros(Float64, length(x))
    MOI.eval_objective_gradient(nlp.evaluator, gradient, x)
    return gradient
end

function evaluate_hessian(
    nlp,
    x::Vector{Float64},
    counter;
    constraint_count::Integer = length(x),
    normalize_nlp = identity,
    normalize_moi = identity,
)
    increment!(counter, :total_hessian_evaluation)
    if nlp isa NLPModels.AbstractNLPModel
        return normalize_nlp(NLPModels.hess(nlp, x))
    end
    :Hess in MOI.features_available(nlp.evaluator) ||
        throw(ErrorException("Hessian computation is not supported."))
    structure = MOI.hessian_lagrangian_structure(nlp.evaluator)
    values = zeros(Float64, length(structure))
    MOI.eval_hessian_lagrangian(
        nlp.evaluator,
        values,
        x,
        1.0,
        zeros(Float64, constraint_count),
    )
    hessian = spzeros(Float64, length(x), length(x))
    for (index, (row, column)) in enumerate(structure)
        hessian[row, column] += values[index]
    end
    return normalize_moi(hessian)
end
