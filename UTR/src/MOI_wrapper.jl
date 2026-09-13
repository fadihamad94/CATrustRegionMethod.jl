########################################################
# MathOptInterface/JuMP wrapper for the UTR solver.
#
# The storage layout follows the CAT optimizer, but result
# lifecycle and model validation follow the MOI conventions.
########################################################

export Optimizer

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

const SupportedObjectiveFunction = Union{
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarNonlinearFunction,
}

"""
    UTRProblem

The result and the effective configuration of the most recent solve.
"""
mutable struct UTRProblem
    status::Symbol
    x::Vector{Float64}
    grad_val::Float64
    obj_val::Float64
    solve_time::Float64
    itr::Int64
    iteration_stats::DataFrame
    algorithm_counter::AlgorithmCounter
    termination_criteria::TerminationCriteria
    algorithm_params::AlgorithmicParameters
    has_result::Bool
end

mutable struct Optimizer <: AbstractUnconstrainedOptimizer
    inner::Union{Nothing,UTRProblem}

    # Model data.
    name::String
    variable_info::Vector{VariableInfo}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
    objective::Union{Nothing,SupportedObjectiveFunction}

    # Optimizer attributes.
    options::Dict{String,Any}
end

function Optimizer(; options...)
    model = Optimizer(
        nothing,
        "",
        VariableInfo[],
        empty_nlp_data(),
        MOI.FEASIBILITY_SENSE,
        nothing,
        Dict{String,Any}(),
    )
    set_options(model, options)
    return model
end

function set_options(model::Optimizer, options)
    candidate = copy(model.options)
    for (name, value) in options
        _set_raw_option!(candidate, string(name), value)
    end
    # Validate all options together before committing any of them.
    _create_parameters(candidate)
    model.options = candidate
    return
end

_invalidate_results!(model::Optimizer) = (model.inner = nothing)
model_changed!(model::Optimizer) = _invalidate_results!(model)

function _empty_iteration_stats()
    return DataFrame(
        k = Int64[],
        fval = Float64[],
        gradnorm = Float64[],
        min_gradnorm_fval = Float64[],
        min_gradnorm = Float64[],
    )
end

function _starting_point(model::Optimizer)
    x = Vector{Float64}(undef, length(model.variable_info))
    for (i, variable) in enumerate(model.variable_info)
        x[i] = something(variable.start, 0.0)
    end
    return x
end

function _field_value(record, field::Symbol, value)
    hasfield(typeof(record), field) ||
        throw(ArgumentError("Unknown optimizer parameter field: $field"))
    field_type = fieldtype(typeof(record), field)
    try
        return convert(field_type, value)
    catch
        throw(
            ArgumentError(
                "Value $(repr(value)) cannot be converted to $field_type for parameter $field",
            ),
        )
    end
end

function _apply_parameter_option!(
    termination_criteria::TerminationCriteria,
    algorithm_params::AlgorithmicParameters,
    name::String,
    value,
)
    parts = split(name, '!'; limit = 2)
    length(parts) == 2 || throw(
        ArgumentError(
            "Raw optimizer attributes must use `termination_criteria!FIELD` " *
            "or `algorithm_params!FIELD`.",
        ),
    )
    group, field_name = parts
    isempty(field_name) &&
        throw(ArgumentError("A raw optimizer attribute field cannot be empty."))
    field = Symbol(field_name)
    if group == "termination_criteria"
        setfield!(
            termination_criteria,
            field,
            _field_value(termination_criteria, field, value),
        )
    elseif group == "algorithm_params"
        setfield!(
            algorithm_params,
            field,
            _field_value(algorithm_params, field, value),
        )
    else
        throw(
            ArgumentError(
                "Unknown raw optimizer attribute group `$group`; expected " *
                "`termination_criteria` or `algorithm_params`.",
            ),
        )
    end
    return
end

function _create_parameters(options::AbstractDict{String,<:Any})
    termination_criteria = TerminationCriteria()
    algorithm_params = AlgorithmicParameters()
    for (name, value) in options
        name in ("time_limit", "output_flag") && continue
        _apply_parameter_option!(
            termination_criteria,
            algorithm_params,
            name,
            value,
        )
    end
    if haskey(options, "time_limit") && options["time_limit"] !== nothing
        termination_criteria.MAX_TIME =
            options["time_limit"] == 0.0 ?
            nextfloat(0.0) :
            _field_value(
                termination_criteria,
                :MAX_TIME,
                options["time_limit"],
            )
    end
    if get(options, "output_flag", false)
        algorithm_params.print_level = -1
    end
    validateTerminationCriteria(termination_criteria)
    validateAlgorithmicParameters(algorithm_params)
    return termination_criteria, algorithm_params
end

function _set_raw_option!(
    options::Dict{String,Any},
    name::String,
    value,
)
    if name == "time_limit"
        if value === nothing
            options[name] = nothing
        elseif value isa Real && isfinite(value) && value >= 0
            options[name] = Float64(value)
        else
            throw(
                ArgumentError(
                    "`time_limit` must be `nothing` or a nonnegative finite number.",
                ),
            )
        end
    elseif name == "output_flag"
        value isa Bool ||
            throw(ArgumentError("`output_flag` must be a Boolean."))
        options[name] = value
    else
        # Validate the spelling and the value type before storing the option.
        termination_criteria = TerminationCriteria()
        algorithm_params = AlgorithmicParameters()
        _apply_parameter_option!(
            termination_criteria,
            algorithm_params,
            name,
            value,
        )
        options[name] = if startswith(name, "termination_criteria!")
            field = Symbol(split(name, '!'; limit = 2)[2])
            getfield(termination_criteria, field)
        else
            field = Symbol(split(name, '!'; limit = 2)[2])
            getfield(algorithm_params, field)
        end
    end
    return
end

function _raw_default(name::String)
    if name == "time_limit"
        return nothing
    elseif name == "output_flag"
        return false
    end
    termination_criteria = TerminationCriteria()
    algorithm_params = AlgorithmicParameters()
    parts = split(name, '!'; limit = 2)
    length(parts) == 2 ||
        throw(ArgumentError("Unknown raw optimizer attribute `$name`."))
    field = Symbol(parts[2])
    if parts[1] == "termination_criteria" &&
       hasfield(TerminationCriteria, field)
        return getfield(termination_criteria, field)
    elseif parts[1] == "algorithm_params" &&
           hasfield(AlgorithmicParameters, field)
        return getfield(algorithm_params, field)
    end
    throw(ArgumentError("Unknown raw optimizer attribute `$name`."))
end

##################################################
# Basic model storage
##################################################

function MOI.is_empty(model::Optimizer)
    return isempty(model.name) &&
           isempty(model.variable_info) &&
           model.nlp_data.evaluator isa EmptyNLPEvaluator &&
           model.sense == MOI.FEASIBILITY_SENSE &&
           model.objective === nothing
end

function MOI.empty!(model::Optimizer)
    model.inner = nothing
    model.name = ""
    empty!(model.variable_info)
    model.nlp_data = empty_nlp_data()
    model.sense = MOI.FEASIBILITY_SENSE
    model.objective = nothing
    return
end

function _check_objective_indices(model::Optimizer, objective)
    MOIU.map_indices(objective) do vi
        MOI.throw_if_not_valid(model, vi)
        return vi
    end
    return
end

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveFunction{F},
    objective::F,
) where {F<:SupportedObjectiveFunction}
    _check_objective_indices(model, objective)
    model.objective = objective
    _invalidate_results!(model)
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ObjectiveFunction{F},
) where {F<:MOI.AbstractScalarFunction}
    model.objective === nothing && throw(
        MOI.GetAttributeNotAllowed(
            MOI.ObjectiveFunction{F}(),
            "No scalar objective function has been set.",
        ),
    )
    try
        return convert(F, model.objective)
    catch error
        error isa OutOfMemoryError && rethrow()
        error isa InterruptException && rethrow()
        throw(InexactError(:get, F, model.objective))
    end
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveFunctionType)
    model.objective === nothing && throw(
        MOI.GetAttributeNotAllowed(
            MOI.ObjectiveFunctionType(),
            "No scalar objective function has been set.",
        ),
    )
    return typeof(model.objective)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    model.sense = sense
    sense == MOI.FEASIBILITY_SENSE && (model.objective = nothing)
    _invalidate_results!(model)
    return
end

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

function MOI.set(
    model::Optimizer,
    ::MOI.NLPBlock,
    nlp_data::MOI.NLPBlockData,
)
    model.nlp_data = nlp_data
    _invalidate_results!(model)
    return
end

MOI.get(model::Optimizer, ::MOI.NLPBlock) = model.nlp_data

function MOI.get(model::Optimizer, ::MOI.ListOfModelAttributesSet)
    attributes = MOI.AbstractModelAttribute[]
    !isempty(model.name) && push!(attributes, MOI.Name())
    model.sense != MOI.FEASIBILITY_SENSE &&
        push!(attributes, MOI.ObjectiveSense())
    if model.objective !== nothing
        push!(
            attributes,
            MOI.ObjectiveFunction{typeof(model.objective)}(),
        )
    end
    !(model.nlp_data.evaluator isa EmptyNLPEvaluator) &&
        push!(attributes, MOI.NLPBlock())
    return attributes
end

# UTR is deliberately an unconstrained solver. Caching optimizers and JuMP
# receive the standard MOI.UnsupportedConstraint exception while copying
# explicit constraints into this optimizer.
MOI.supports_constraint(
    ::Optimizer,
    ::Type{F},
    ::Type{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet} = false

MOI.get(::Optimizer, ::MOI.ListOfConstraintTypesPresent) =
    Tuple{Type,Type}[]

function MOI.get(model::Optimizer, ::MOI.ListOfVariableAttributesSet)
    return any(variable -> variable.start !== nothing, model.variable_info) ?
           MOI.AbstractVariableAttribute[MOI.VariablePrimalStart()] :
           MOI.AbstractVariableAttribute[]
end

##################################################
# Optimizer and model attributes
##################################################

MOI.get(::Optimizer, ::MOI.SolverName) = "UTROptimizer"
MOI.get(::Optimizer, ::MOI.SolverVersion) = "v0.1.0"

function MOI.get(model::Optimizer, ::MOI.ListOfOptimizerAttributesSet)
    names = sort!(collect(keys(model.options)))
    return MOI.AbstractOptimizerAttribute[
        MOI.RawOptimizerAttribute(name) for name in names
    ]
end

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{F},
) where {F<:SupportedObjectiveFunction} = true

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.set(
    model::Optimizer,
    attribute::MOI.RawOptimizerAttribute,
    value,
)
    candidate = copy(model.options)
    _set_raw_option!(candidate, attribute.name, value)
    # setfield! on a mutable configuration bypasses its inner constructor, so
    # validate the complete effective configuration before committing.
    _create_parameters(candidate)
    model.options = candidate
    return
end

function MOI.get(
    model::Optimizer,
    attribute::MOI.RawOptimizerAttribute,
)
    return get(model.options, attribute.name) do
        _raw_default(attribute.name)
    end
end

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, flag::Bool)
    MOI.set(model, MOI.RawOptimizerAttribute("output_flag"), flag)
    return
end

function MOI.get(model::Optimizer, ::MOI.Silent)
    return MOI.get(model, MOI.RawOptimizerAttribute("output_flag"))
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, limit::Nothing)
    MOI.set(model, MOI.RawOptimizerAttribute("time_limit"), limit)
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, limit::Real)
    MOI.set(
        model,
        MOI.RawOptimizerAttribute("time_limit"),
        Float64(limit),
    )
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return MOI.get(model, MOI.RawOptimizerAttribute("time_limit"))
end

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, vi)
    model.variable_info[vi.value].start =
        value === nothing ? nothing : Float64(value)
    _invalidate_results!(model)
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
)
    MOI.throw_if_not_valid(model, vi)
    return model.variable_info[vi.value].start
end

MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimal,
    ::Type{MOI.VariableIndex},
) = true

MOI.supports(::Optimizer, ::MOI.SolveTimeSec) = true

const SupportedModelAttribute = Union{
    MOI.ListOfModelAttributesSet,
    MOI.NumberOfVariables,
    MOI.ListOfVariableIndices,
    MOI.ObjectiveFunctionType,
    MOI.ObjectiveValue,
    MOI.RawSolver,
    MOI.RawStatusString,
    MOI.ResultCount,
    MOI.TerminationStatus,
    MOI.PrimalStatus,
    MOI.DualStatus,
    MOI.SolveTimeSec,
}

MOI.supports(::Optimizer, ::SupportedModelAttribute) = true

MOI.supports(::Optimizer, ::MOI.ObjectiveLimit) = false
MOI.supports(::Optimizer, ::MOI.AbsoluteGapTolerance) = false
MOI.supports(::Optimizer, ::MOI.RelativeGapTolerance) = false
MOI.supports(::Optimizer, ::MOI.SolutionLimit) = false

##################################################
# Solve preparation and execution
##################################################

function _direct_objective_nlp_data(model::Optimizer)
    nonlinear_model = MOI.Nonlinear.Model()
    objective = if model.objective isa MOI.ScalarNonlinearFunction
        model.objective
    else
        convert(MOI.ScalarNonlinearFunction, model.objective)
    end
    MOI.Nonlinear.set_objective(nonlinear_model, objective)
    variables = MOI.get(model, MOI.ListOfVariableIndices())
    evaluator = MOI.Nonlinear.Evaluator(
        nonlinear_model,
        MOI.Nonlinear.SparseReverseMode(),
        variables,
    )
    return MOI.NLPBlockData(evaluator)
end

function _nlp_data_for_solve(model::Optimizer)
    if model.sense == MOI.MAX_SENSE
        return nothing
    elseif model.sense != MOI.MIN_SENSE
        return nothing
    elseif !isempty(model.nlp_data.constraint_bounds)
        return nothing
    end

    has_nlp_block = !(model.nlp_data.evaluator isa EmptyNLPEvaluator)
    if has_nlp_block && model.objective !== nothing
        # Two objective representations would have ambiguous precedence.
        return nothing
    elseif has_nlp_block
        model.nlp_data.has_objective || return nothing
        return model.nlp_data
    elseif model.objective !== nothing
        return _direct_objective_nlp_data(model)
    end
    return nothing
end

function _initialize_evaluator(nlp_data::MOI.NLPBlockData)
    features = MOI.features_available(nlp_data.evaluator)
    (:Grad in features && :Hess in features) || return false
    MOI.initialize(nlp_data.evaluator, [:Grad, :Hess])
    return true
end

function _invalid_model!(
    model::Optimizer,
    termination_criteria::TerminationCriteria,
    algorithm_params::AlgorithmicParameters,
    start_time::Float64,
)
    elapsed = time() - start_time
    model.inner = UTRProblem(
        :InvalidModel,
        _starting_point(model),
        NaN,
        NaN,
        elapsed,
        0,
        _empty_iteration_stats(),
        AlgorithmCounter(),
        termination_criteria,
        algorithm_params,
        false,
    )
    return
end

function _status_symbol(status)
    status == TerminationStatusCode.OPTIMAL && return :Optimal
    status == TerminationStatusCode.UNBOUNDED && return :Unbounded
    status == TerminationStatusCode.ITERATION_LIMIT && return :IterationLimit
    status == TerminationStatusCode.INNER_ITERATION_LIMIT &&
        return :InnerIterationLimit
    status == TerminationStatusCode.TIME_LIMIT && return :TimeLimit
    status == TerminationStatusCode.MEMORY_LIMIT && return :MemoryLimit
    status == TerminationStatusCode.STEP_SIZE_LIMIT && return :StepSizeLimit
    status == TerminationStatusCode.NUMERICAL_ERROR && return :NumericalError
    status == TerminationStatusCode.TRUST_REGION_SUBPROBLEM_ERROR &&
        return :TrustRegionSubproblemError
    status == TerminationStatusCode.INVALID_MODEL && return :InvalidModel
    return :OtherError
end

function _final_history_values(iteration_stats::DataFrame)
    if nrow(iteration_stats) == 0
        return NaN, NaN
    end
    obj_val =
        hasproperty(iteration_stats, :fval) ?
        Float64(iteration_stats[end, :fval]) : NaN
    grad_val =
        hasproperty(iteration_stats, :gradnorm) ?
        Float64(iteration_stats[end, :gradnorm]) : NaN
    return obj_val, grad_val
end

function MOI.optimize!(model::Optimizer)
    start_time = time()
    termination_criteria, algorithm_params =
        _create_parameters(model.options)
    nlp_data = _nlp_data_for_solve(model)
    if nlp_data === nothing
        _invalid_model!(
            model,
            termination_criteria,
            algorithm_params,
            start_time,
        )
        return
    end

    evaluator_initialized = try
        _initialize_evaluator(nlp_data)
    catch
        false
    end
    if !evaluator_initialized
        _invalid_model!(
            model,
            termination_criteria,
            algorithm_params,
            start_time,
        )
        return
    end

    # `UTR_solve(::Optimizer, ...)` intentionally consumes `nlp_data`. A
    # directly supplied MOI scalar objective is converted to a temporary NLP
    # block, while the original model representation remains available for
    # repeated solves and objective replacement.
    original_nlp_data = model.nlp_data
    model.nlp_data = nlp_data
    result = try
        UTR_solve(model, termination_criteria, algorithm_params)
    finally
        model.nlp_data = original_nlp_data
    end

    x,
    status,
    iteration_stats,
    algorithm_counter,
    outer_iteration,
    elapsed_time = result
    obj_val, grad_val = _final_history_values(iteration_stats)
    status_symbol = _status_symbol(status)
    has_result =
        status_symbol != :InvalidModel &&
        length(x) == length(model.variable_info) &&
        nrow(iteration_stats) > 0 &&
        isfinite(obj_val) &&
        isfinite(grad_val)
    model.inner = UTRProblem(
        status_symbol,
        Vector{Float64}(x),
        grad_val,
        obj_val,
        Float64(elapsed_time),
        Int64(outer_iteration),
        iteration_stats,
        algorithm_counter,
        termination_criteria,
        algorithm_params,
        has_result,
    )
    return
end

##################################################
# Result attributes
##################################################

function _termination_status(status::Symbol)
    status == :Optimal && return MOI.LOCALLY_SOLVED
    status == :Unbounded && return MOI.OBJECTIVE_LIMIT
    status == :IterationLimit && return MOI.ITERATION_LIMIT
    status == :InnerIterationLimit && return MOI.ITERATION_LIMIT
    status == :TimeLimit && return MOI.TIME_LIMIT
    status == :MemoryLimit && return MOI.MEMORY_LIMIT
    status == :StepSizeLimit && return MOI.SLOW_PROGRESS
    status == :NumericalError && return MOI.NUMERICAL_ERROR
    status == :InvalidModel && return MOI.INVALID_MODEL
    return MOI.OTHER_ERROR
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    model.inner === nothing && return MOI.OPTIMIZE_NOT_CALLED
    return _termination_status(model.inner.status)
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    model.inner === nothing && return string(MOI.OPTIMIZE_NOT_CALLED)
    return string(model.inner.status)
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.inner !== nothing && model.inner.has_result ? 1 : 0
end

function MOI.get(model::Optimizer, attribute::MOI.PrimalStatus)
    if attribute.result_index != 1 ||
       model.inner === nothing ||
       !model.inner.has_result
        return MOI.NO_SOLUTION
    elseif model.inner.status in (
        :Optimal,
        :Unbounded,
        :IterationLimit,
        :InnerIterationLimit,
        :TimeLimit,
        :MemoryLimit,
        :StepSizeLimit,
    )
        return MOI.FEASIBLE_POINT
    end
    return MOI.UNKNOWN_RESULT_STATUS
end

MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

function MOI.get(
    model::Optimizer,
    attribute::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attribute)
    MOI.throw_if_not_valid(model, vi)
    return model.inner.x[vi.value]
end

function MOI.get(model::Optimizer, attribute::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attribute)
    return model.inner.obj_val
end

function MOI.get(model::Optimizer, ::MOI.SolveTimeSec)
    model.inner === nothing && return NaN
    return model.inner.solve_time
end
