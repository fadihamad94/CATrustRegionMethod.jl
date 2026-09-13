mutable struct VariableInfo
    lower_bound::Float64
    has_lower_bound::Bool
    lower_bound_dual_start::Union{Nothing,Float64}
    upper_bound::Float64
    has_upper_bound::Bool
    upper_bound_dual_start::Union{Nothing,Float64}
    is_fixed::Bool
    start::Union{Nothing,Float64}
end

VariableInfo() =
    VariableInfo(-Inf, false, nothing, Inf, false, nothing, false, nothing)

struct EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end

MOI.features_available(::EmptyNLPEvaluator) = [:Grad, :Hess]
MOI.initialize(::EmptyNLPEvaluator, ::Vector{Symbol}) = nothing
MOI.eval_objective(::EmptyNLPEvaluator, ::AbstractVector) = 0.0
MOI.eval_objective_gradient(::EmptyNLPEvaluator, gradient, ::AbstractVector) =
    fill!(gradient, 0.0)
MOI.hessian_lagrangian_structure(::EmptyNLPEvaluator) =
    Tuple{Int64,Int64}[]

function MOI.eval_hessian_lagrangian(
    ::EmptyNLPEvaluator,
    values,
    ::AbstractVector,
    ::Real,
    ::AbstractVector,
)
    @assert isempty(values)
    return
end

empty_nlp_data() = MOI.NLPBlockData([], EmptyNLPEvaluator(), false)

abstract type AbstractUnconstrainedOptimizer <: MOI.AbstractOptimizer end

model_changed!(::AbstractUnconstrainedOptimizer) = nothing

MOI.supports_incremental_interface(::AbstractUnconstrainedOptimizer) = true
MOI.get(model::AbstractUnconstrainedOptimizer, ::MOI.RawSolver) = model
MOI.supports(::AbstractUnconstrainedOptimizer, ::MOI.Name) = true
MOI.get(model::AbstractUnconstrainedOptimizer, ::MOI.Name) = model.name
MOI.set(model::AbstractUnconstrainedOptimizer, ::MOI.Name, name::String) =
    (model.name = name)
MOI.get(model::AbstractUnconstrainedOptimizer, ::MOI.NumberOfVariables) =
    length(model.variable_info)

function MOI.add_variable(model::AbstractUnconstrainedOptimizer)
    push!(model.variable_info, VariableInfo())
    model_changed!(model)
    return MOI.VariableIndex(length(model.variable_info))
end

function MOI.add_variables(model::AbstractUnconstrainedOptimizer, count::Int)
    return [MOI.add_variable(model) for _ in 1:count]
end

function MOI.get(
    model::AbstractUnconstrainedOptimizer,
    ::MOI.ListOfVariableIndices,
)
    return [MOI.VariableIndex(i) for i in eachindex(model.variable_info)]
end

function MOI.is_valid(
    model::AbstractUnconstrainedOptimizer,
    variable::MOI.VariableIndex,
)
    return variable.value in eachindex(model.variable_info)
end
