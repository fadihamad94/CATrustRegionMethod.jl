########################################################
## this code is based on ModelReader in NLPModels
## and KNITRO.jl
########################################################

export Optimizer

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

const SupportedObjectiveFunction = Union{
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarNonlinearFunction,
}

mutable struct CATProblem
    status::Symbol  # Final status
    x::Vector{Float64}  # Starting and final solution
    grad_val::Float64  # Final objective gradient
    obj_val::Float64  # (length 1) Final objective
    solve_time::Float64
    itr::Int64 #Total number of iterations

    # Custom attributes of the Optimizer
    iteration_stats::DataFrame
    algorithm_counter::AlgorithmCounter
    termination_criteria::TerminationCriteria
    algorithm_params::AlgorithmicParameters

    function CATProblem()
        return new()
    end
end

mutable struct Optimizer <: AbstractUnconstrainedOptimizer
    inner::Union{CATProblem,Nothing}

    # Storage for `MOI.Name`.
    name::String

    # Problem data.
    variable_info::Vector{VariableInfo}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
    objective::Union{Nothing,SupportedObjectiveFunction}

    nlp_dual_start::Union{Nothing,Vector{Float64}}

    # Parameters.
    options::Dict{String,Any}

    # Solution attributes.
    solve_time::Float64
end

function Optimizer(; options...)
    options_dict = Dict{String,Any}()

    for (name, value) in options
        options_dict[string(name)] = value
    end

    OptimizerModel = Optimizer(
        CATProblem(),
        "",
        [],
        empty_nlp_data(),
        MOI.FEASIBILITY_SENSE,
        nothing,
        nothing,
        options_dict,
        NaN,
    )
    set_options(OptimizerModel, options)

    return OptimizerModel
end

function set_options(model::Optimizer, options)
    for (name, value) in options
        sname = string(name)
        MOI.set(model, MOI.RawOptimizerAttribute(sname), value)
    end
    return
end

function MOI.get(model::Optimizer, ::MOI.ListOfModelAttributesSet)
    attributes = MOI.AbstractModelAttribute[]
    if model.sense != MOI.FEASIBILITY_SENSE
        push!(attributes, MOI.ObjectiveSense())
    end
    if model.objective != nothing
        F = MOI.get(model, MOI.ObjectiveFunctionType())
        push!(attributes, MOI.ObjectiveFunction{F}())
    end
    !(model.nlp_data.evaluator isa EmptyNLPEvaluator) &&
        push!(attributes, MOI.NLPBlock())
    if !isempty(model.name)
        push!(attributes, MOI.Name())
    end
    return attributes
end

###
### MOI.Silent
###

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.get(model::Optimizer, ::MOI.Silent)
    return MOI.get(model, MOI.RawOptimizerAttribute("output_flag"))
end

function MOI.set(model::Optimizer, ::MOI.Silent, flag::Bool)
    MOI.set(model, MOI.RawOptimizerAttribute("output_flag"), flag)
    return
end

###
MOI.get(::Optimizer, ::MOI.SolverName) = "CATOptimizer"

function MOI.get(::Optimizer, ::MOI.SolverVersion)
    X, Y, Z = 1, 0, 0
    return "v$X.$Y.$Z"
end


###
### MOI.TimeLimitSec
###

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    return MOI.set(model, MOI.RawOptimizerAttribute("time_limit"), nothing)
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, limit::Real)
    return MOI.set(model, MOI.RawOptimizerAttribute("time_limit"), Float64(limit))
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    value = MOI.get(model, MOI.RawOptimizerAttribute("time_limit"))
    return value
end

"""
    MOI.is_empty(model::Optimizer )
"""

function MOI.is_empty(model::Optimizer)
    return isempty(model.variable_info) &&
           model.nlp_data.evaluator isa EmptyNLPEvaluator &&
           model.sense == MOI.FEASIBILITY_SENSE &&
           model.objective === nothing
end

function MOI.empty!(model::Optimizer)
    model.inner = CATProblem()
    model.name = ""
    empty!(model.variable_info)
    model.nlp_data = empty_nlp_data()
    model.sense = MOI.FEASIBILITY_SENSE
    model.objective = nothing
    model.nlp_dual_start = nothing
end

function has_upper_bound(model::Optimizer, vi::MOI.VariableIndex)
    return model.variable_info[vi.value].has_upper_bound
end

function has_lower_bound(model::Optimizer, vi::MOI.VariableIndex)
    return model.variable_info[vi.value].has_lower_bound
end

function is_fixed(model::Optimizer, vi::MOI.VariableIndex)
    return model.variable_info[vi.value].is_fixed
end

function _check_objective_indices(model::Optimizer, objective)
    MOIU.map_indices(objective) do variable
        MOI.throw_if_not_valid(model, variable)
        return variable
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

############################
## END ModelReader CODE
############################

function create_pars_JuMP(options)
    termination_criteria = TerminationCriteria()
    algorithm_params = AlgorithmicParameters()
    for (param, value) in options
        what = split(String(param), "!") # we represent a parameter such as termination_criteria.MAX_ITERATIONS as termination_criteria!MAX_ITERATIONS because we cannot pass termination_criteria.MAX_ITERATIONS as a parameter
        node = nothing
        if what[1] == "termination_criteria"
            node = termination_criteria
        elseif what[1] == "algorithm_params"
            node = algorithm_params
        else
            if param ∉ ["time_limit", "output_flag"]
                error("Unkown argument.")
            end
        end
        if param ∉ ["time_limit", "output_flag"]
            field = what[2]
            setfield!(node, Symbol(field), value)
        end
    end

    validateTerminationCriteria(termination_criteria)
    validateTrustRegionSubproblemSolverParameters(
        algorithm_params.trust_region_subproblem_solver,
        algorithm_params.γ_1,
        algorithm_params.γ_2,
        algorithm_params.γ_3,
    )

    return termination_criteria, algorithm_params
end

function _direct_objective_nlp_data(model::Optimizer)
    nonlinear_model = MOI.Nonlinear.Model()
    objective =
        model.objective isa MOI.ScalarNonlinearFunction ?
        model.objective :
        convert(MOI.ScalarNonlinearFunction, model.objective)
    MOI.Nonlinear.set_objective(nonlinear_model, objective)
    evaluator = MOI.Nonlinear.Evaluator(
        nonlinear_model,
        MOI.Nonlinear.SparseReverseMode(),
        MOI.get(model, MOI.ListOfVariableIndices()),
    )
    return MOI.NLPBlockData(evaluator)
end

function _nlp_data_for_solve(model::Optimizer)
    if model.sense == MOI.FEASIBILITY_SENSE
        return empty_nlp_data()
    elseif model.sense != MOI.MIN_SENSE
        return nothing
    end
    isempty(model.nlp_data.constraint_bounds) || return nothing

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
    return Float64[
        something(variable.start, 0.0) for variable in model.variable_info if
        !variable.is_fixed
    ]
end

function _invalid_model!(
    model::Optimizer,
    termination_criteria::TerminationCriteria,
    algorithm_params::AlgorithmicParameters,
    start_time::Float64,
)
    model.inner = CATProblem()
    model.inner.status = :InvalidModel
    model.inner.x = _starting_point(model)
    model.inner.grad_val = NaN
    model.inner.obj_val = NaN
    model.inner.solve_time = time() - start_time
    model.inner.itr = 0
    model.inner.iteration_stats = _empty_iteration_stats()
    model.inner.algorithm_counter = AlgorithmCounter()
    model.inner.termination_criteria = termination_criteria
    model.inner.algorithm_params = algorithm_params
    return
end

function MOI.optimize!(solver::Optimizer)
    t = time()

    termination_criteria, algorithm_params = create_pars_JuMP(solver.options)
    try
        if MOI.get(solver, MOI.RawOptimizerAttribute("time_limit")) != nothing
            time_limit =
                MOI.get(solver, MOI.RawOptimizerAttribute("time_limit"))
            termination_criteria.MAX_TIME =
                time_limit == 0.0 ? nextfloat(0.0) : time_limit
        end
    catch
        MOI.set(
            solver,
            MOI.RawOptimizerAttribute("time_limit"),
            termination_criteria.MAX_TIME,
        )
    end

    try
        if MOI.get(solver, MOI.Silent())
            algorithm_params.print_level = -1
        end
        if !MOI.get(solver, MOI.Silent()) && algorithm_params.print_level == -1
            algorithm_params.print_level = 0
        end
    catch
        MOI.set(
            solver,
            MOI.RawOptimizerAttribute("output_flag"),
            algorithm_params.print_level == 0,
        )
    end

    nlp_data = _nlp_data_for_solve(solver)
    if nlp_data === nothing
        _invalid_model!(solver, termination_criteria, algorithm_params, t)
        return
    end
    evaluator_initialized = try
        _initialize_evaluator(nlp_data)
    catch error
        error isa OutOfMemoryError && rethrow()
        error isa InterruptException && rethrow()
        false
    end
    if !evaluator_initialized
        _invalid_model!(solver, termination_criteria, algorithm_params, t)
        return
    end

    # A directly supplied scalar objective is converted to a temporary NLP block.
    # Preserve the stored MOI representation for getters, repeated solves, and
    # objective replacement.
    original_nlp_data = solver.nlp_data
    solver.nlp_data = nlp_data
    result = try
        CAT_solve(solver, termination_criteria, algorithm_params)
    finally
        solver.nlp_data = original_nlp_data
    end

    x,
    status,
    iteration_stats,
    algorithm_counter,
    k,
    _,
    best_gradient_x = result

    status_str = convertStatusCodeToStatusString(status)

    solver.inner = CATProblem()
    solver.inner.status = status_CAT_To_JuMP(status_str)
    solver.inner.x =
        status == TerminationStatusCode.OPTIMAL ? best_gradient_x : x

    function_value = NaN
    gradient_value = NaN
    if size(last(iteration_stats, 1))[1] > 0
        final_row = last(iteration_stats, 1)
        if status == TerminationStatusCode.OPTIMAL
            function_value = final_row[!, "min_gradnorm_fval"][1]
            gradient_value = final_row[!, "min_gradnorm"][1]
        else
            function_value = final_row[!, "fval"][1]
            gradient_value = final_row[!, "gradnorm"][1]
        end
    end

    solver.inner.obj_val = function_value
    solver.inner.grad_val = gradient_value
    solver.inner.itr = k
    solver.inner.solve_time = time() - t

    # custom CAT features
    solver.inner.termination_criteria = termination_criteria
    solver.inner.algorithm_params = algorithm_params
    solver.inner.iteration_stats = iteration_stats
    solver.inner.algorithm_counter = algorithm_counter
end

function convertStatusCodeToStatusString(status)
    return canonical_status_string(status)
end

function convertStatusToJuMPStatusCode_TerminationStatus(status)
    dict_status_code = Dict(
        :Optimal => MOI.OPTIMAL,
        :Unbounded => MOI.INFEASIBLE_OR_UNBOUNDED,
        :IterationLimit => MOI.ITERATION_LIMIT,
        :TimeLimit => MOI.TIME_LIMIT,
        :UserLimit => MOI.OTHER_LIMIT,
        :InvalidModel => MOI.INVALID_MODEL,
        :Error => MOI.OTHER_ERROR,
    )
    return dict_status_code[status]
end

function convertStatusToJuMPStatusCode(status)
    dict_status_code = Dict(
        :Optimal => MOI.FEASIBLE_POINT,
        :Unbounded => MOI.INFEASIBLE_POINT,
        :IterationLimit => MOI.INFEASIBILITY_CERTIFICATE,
        :TimeLimit => MOI.INFEASIBILITY_CERTIFICATE,
        :UserLimit => MOI.INFEASIBILITY_CERTIFICATE,
        :InvalidModel => MOI.INVALID_MODEL,
        :Error => MOI.NO_SOLUTION,
    )
    return dict_status_code[status]
end

function status_CAT_To_JuMP(status::String)
    # since our status are not equal to JuMPs we need to do a conversion
    if status == "OPTIMAL"
        return :Optimal
    elseif status == "UNBOUNDED"
        return :Unbounded
    elseif status == "ITERATION_LIMIT" ||
           status == "TIME_LIMIT" ||
           status == "STEP_SIZE_LIMIT" ||
           status == "MEMORY_LIMIT"
        return :UserLimit
    elseif status ==  "INVALID_MODEL"
        return :InvalidModel
    else
        return :Error
    end
end

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{F},
) where {F<:SupportedObjectiveFunction}
    return true
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

const SUPPORTED_MODEL_ATTR = Union{
    MOI.ObjectiveSense,
    MOI.NumberOfVariables,
    MOI.ListOfVariableIndices,
    MOI.ObjectiveFunctionType,
    MOI.ObjectiveValue,
    MOI.DualObjectiveValue,
    MOI.RawSolver,
    MOI.RawStatusString,
    MOI.ResultCount,
    MOI.TerminationStatus,
    MOI.PrimalStatus,
    MOI.DualStatus
}

MOI.supports(::Optimizer, ::SUPPORTED_MODEL_ATTR) = true

MOI.supports(::Optimizer, ::MOI.ObjectiveLimit) = false

MOI.supports(::Optimizer, ::MOI.AbsoluteGapTolerance) = false

MOI.supports(::Optimizer, ::MOI.RelativeGapTolerance) = false

MOI.supports(::Optimizer, ::MOI.SolutionLimit) = false

MOI.supports(::Optimizer, ::MOI.SolveTimeSec) = true

function MOI.get(model::Optimizer, ::MOI.SolveTimeSec)
    return model.inner.solve_time;
end

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    if !isempty(nlp_data.constraint_bounds)
        throw(
            MOI.SetAttributeNotAllowed(
                MOI.NLPBlock(),
                "CAT supports only unconstrained nonlinear models.",
            ),
        )
    end
    model.nlp_data = nlp_data
    return
end

MOI.get(model::Optimizer, ::MOI.NLPBlock) = model.nlp_data

function MOI.set(model::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    if sense == MOI.MAX_SENSE
        throw(
            MOI.SetAttributeNotAllowed(
                MOI.ObjectiveSense(),
                "CAT supports minimization only.",
            ),
        )
    end
    model.sense = sense
    sense == MOI.FEASIBILITY_SENSE && (model.objective = nothing)
    return
end

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, vi)
    model.variable_info[vi.value].start = value
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
)
    return model.variable_info[vi.value].start
end

function MOI.get(model::MOIU.CachingOptimizer, args...)
    return MOI.get(model.model.optimizer, args)
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, v::VariableRef)
    return MOI.get(model, attr, v.variable)
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    return model.inner.x[vi.value]
end

function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, value)
    candidate = copy(model.options)
    if p.name == "time_limit"
        if value === nothing
            candidate[p.name] = nothing
        elseif value isa Real && isfinite(value) && value >= 0
            candidate[p.name] = Float64(value)
        else
            throw(
                ArgumentError(
                    "`time_limit` must be `nothing` or a nonnegative finite number.",
                ),
            )
        end
    elseif p.name == "output_flag"
        value isa Bool ||
            throw(ArgumentError("`output_flag` must be a Boolean."))
        candidate[p.name] = value
    else
        candidate[p.name] = value
    end
    # Mutable parameter records are populated with setfield!, so validate the
    # complete effective configuration before committing the option.
    create_pars_JuMP(candidate)
    model.options = candidate
    return
end

function MOI.get(model::Optimizer, p::MOI.RawOptimizerAttribute)
    if haskey(model.options, p.name)
        return model.options[p.name]
    end
    error("RawParameter with name $(p.name) is not set.")
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    try
        model.inner.status
    catch
        return MOI.OPTIMIZE_NOT_CALLED
    end
    status_ = convertStatusToJuMPStatusCode_TerminationStatus(model.inner.status)
    return status_
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    try
        model.inner.status
    catch
        return string(MOI.OPTIMIZE_NOT_CALLED)
    end
    status_ = model.inner.status
    return string(status_)
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return (model.inner !== nothing) ? 1 : 0
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    try
        model.inner.status
    catch
        return MOI.NO_SOLUTION
    end

    status_ = convertStatusToJuMPStatusCode(model.inner.status)
    return status_
end

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    try
        model.inner.status
    catch
        return MOI.NO_SOLUTION
    end

    status_ = convertStatusToJuMPStatusCode(model.inner.status)
    return status_
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.inner.obj_val
end
