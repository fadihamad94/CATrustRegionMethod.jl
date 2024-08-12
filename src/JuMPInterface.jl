########################################################
## this code is based on ModelReader in NLPModels
## and KNITRO.jl
########################################################

export CATSolver

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

# TODO
const SF = Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}}
const VAF = MOI.VectorAffineFunction{Float64}
const VOV = MOI.VectorOfVariables

# ScalarAffineFunctions and VectorAffineFunctions
const SAF = MOI.ScalarAffineFunction{Float64}
const AF = Union{SAF,VAF}

const SS = Union{
    MOI.EqualTo{Float64},
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.Interval{Float64},
}
# LinSets
const LS = Union{MOI.EqualTo{Float64},MOI.GreaterThan{Float64},MOI.LessThan{Float64}}
# VecLinSets
const VLS = Union{MOI.Nonnegatives,MOI.Nonpositives,MOI.Zeros}
##################################################
mutable struct VariableInfo
    lower_bound::Float64  # May be -Inf even if has_lower_bound == true
    has_lower_bound::Bool # Implies lower_bound == Inf
    lower_bound_dual_start::Union{Nothing,Float64}
    upper_bound::Float64  # May be Inf even if has_upper_bound == true
    has_upper_bound::Bool # Implies upper_bound == Inf
    upper_bound_dual_start::Union{Nothing,Float64}
    is_fixed::Bool        # Implies lower_bound == upper_bound and !has_lower_bound and !has_upper_bound.
    start::Union{Nothing,Float64}
end
VariableInfo() = VariableInfo(-Inf, false, nothing, Inf, false, nothing, false, nothing)

mutable struct CATProblem
    status::Symbol  # Final status
    x::Vector{Float64}  # Starting and final solution
    grad_val::Float64  # Final objective gradient
    obj_val::Float64  # (length 1) Final objective
    solve_time::Float64
    itr::Int64 #Total number of iterations

    # Custom attributes of the CATSolver
    iteration_stats::DataFrame
    computation_stats::Dict{String,Int64}
    termination_criteria::TerminationCriteria
    algorithm_params::AlgorithmicParameters

    function CATProblem()
        return new()
    end
end

##################################################
# EmptyNLPEvaluator for non-NLP problems.
struct EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end
MOI.features_available(::EmptyNLPEvaluator) = [:Grad, :Hess]
MOI.initialize(::EmptyNLPEvaluator, features) = nothing
MOI.eval_objective(::EmptyNLPEvaluator, x) = NaN


MOI.eval_objective_gradient(::EmptyNLPEvaluator, g, x) = nothing
MOI.hessian_lagrangian_structure(::EmptyNLPEvaluator) = Tuple{Int64,Int64}[]

function MOI.eval_hessian_lagrangian(::EmptyNLPEvaluator, H, x, s, mu)
    @assert length(H) == 0
    return
end
function MOI.eval_hessian_lagrangian(::Nothing, H, x, s, mu)
    @assert length(H) == 0
    return
end

empty_nlp_data() = MOI.NLPBlockData([], EmptyNLPEvaluator(), false)



mutable struct CATSolver <: MOI.AbstractOptimizer
    #inner::CATProblem
    inner::Union{CATProblem,Nothing}

    # Problem data.
    variable_info::Vector{VariableInfo}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
    objective::Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
        Nothing,
    }

    nlp_dual_start::Union{Nothing,Vector{Float64}}

    # Parameters.
    silent::Bool
    options::Dict{String,Any}

    # Solution attributes.
    solve_time::Float64
end

function CATSolver(; options...)
    options_dict = Dict{String,Any}()

    for (name, value) in options
        options_dict[string(name)] = value
    end

    CATSolverModel = CATSolver(
        CATProblem(),
        [],
        empty_nlp_data(),
        MOI.FEASIBILITY_SENSE,
        nothing,
        nothing,
        false,
        options_dict,
        NaN,
    )
    set_options(CATSolverModel, options)

    return CATSolverModel
end

function set_options(model::CATSolver, options)
    for (name, value) in options
        sname = string(name)
        MOI.set(model, MOI.RawOptimizerAttribute(sname), value)
    end
    return
end

MOI.get(::CATSolver, ::MOI.SolverName) = "CATSolver"

"""
    MOI.is_empty(model::CATSolver )
"""

function MOI.is_empty(model::CATSolver)
    return isempty(model.variable_info) &&
           model.nlp_data.evaluator isa EmptyNLPEvaluator &&
           model.sense == MOI.FEASIBILITY_SENSE
end

function MOI.empty!(model::CATSolver)
    model.inner = nothing
    empty!(model.variable_info)
    model.nlp_data = empty_nlp_data()
    model.sense = MOI.FEASIBILITY_SENSE
    model.objective = nothing
    model.nlp_dual_start = nothing
end

function MOI.get(model::CATSolver, ::MOI.ListOfVariableIndices)
    return [MOI.VariableIndex(i) for i = 1:length(model.variable_info)]
end

function MOI.add_variable(model::CATSolver)
    push!(model.variable_info, VariableInfo())
    return MOI.VariableIndex(length(model.variable_info))
end

function MOI.add_variables(model::CATSolver, n::Int)
    return [MOI.add_variable(model) for i = 1:n]
end

function MOI.is_valid(model::CATSolver, vi::MOI.VariableIndex)
    return vi.value in eachindex(model.variable_info)
end

function MOI.copy_to(model::CATSolver, src::MOI.ModelLike; copy_names = false)
    return MOI.Utilities.default_copy_to(model, src, copy_names)
end

function has_upper_bound(model::CATSolver, vi::MOI.VariableIndex)
    return model.variable_info[vi.value].has_upper_bound
end

function has_lower_bound(model::CATSolver, vi::MOI.VariableIndex)
    return model.variable_info[vi.value].has_lower_bound
end

function is_fixed(model::CATSolver, vi::MOI.VariableIndex)
    return model.variable_info[vi.value].is_fixed
end

function MOI.set(
    model::CATSolver,
    ::MOI.ObjectiveFunction,
    func::Union{MOI.VariableIndex,MOI.ScalarAffineFunction,MOI.ScalarQuadraticFunction},
)
    check_inbounds(model, func)
    model.objective = func
    return
end

function NLPModels.obj(nlp::MathOptNLPModel, x::AbstractVector)
    NLPModels.increment!(nlp, :neval_obj)
    if nlp.obj.type == "LINEAR"
        res = dot(nlp.obj.gradient, x) + nlp.obj.constant
    end
    if nlp.obj.type == "QUADRATIC"
        res =
            0.5 * coo_sym_dot(
                nlp.obj.hessian.rows,
                nlp.obj.hessian.cols,
                nlp.obj.hessian.vals,
                x,
                x,
            ) +
            dot(nlp.obj.gradient, x) +
            nlp.obj.constant
    end
    if nlp.obj.type == "NONLINEAR"
        res = MOI.eval_objective(nlp.eval, x)
    end
    return res
end

function append_to_hessian_sparsity!(
    ::Any,
    ::Union{MOI.VariableIndex,MOI.ScalarAffineFunction},
)
    return nothing
end

function append_to_hessian_sparsity!(hessian_sparsity, quad::MOI.ScalarQuadraticFunction)
    for term in quad.quadratic_terms
        push!(hessian_sparsity, (term.variable_1.value, term.variable_2.value))
    end
end

function hessian_lagrangian_structure(model::CATSolver, nlp::MathOptNLPModel)
    hessian_sparsity = Tuple{Int64,Int64}[]
    @assert model.objective !== nothing
    append_to_hessian_sparsity!(hessian_sparsity, model.obj)

    nlp_hessian_sparsity = MOI.hessian_lagrangian_structure(nlp.eval)
    append!(hessian_sparsity, nlp_hessian_sparsity)
    return hessian_sparsity
end

function hess_coord(
    nlp::MathOptNLPModel,
    x::Array{Float64};
    obj_weight::Float64 = 1.0,
    y::Array{Float64} = zeros(nlp.meta.ncon),
)
    NLPModels.increment!(nlp, :neval_hess)
    MOI.eval_hessian_lagrangian(nlp.eval, nlp.obj.hessian.vals, x, obj_weight, y)

    return (
        NLPModels.hess_structure(nlp)[1],
        NLPModels.hess_structure(nlp)[2],
        NLPModels.hess_coord(nlp, x, y, obj_weight = obj_weight),
    )
end

function NLPModels.hess_structure!(
    nlp::MathOptNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    if nlp.obj.type == "QUADRATIC"
        for index = 1:(nlp.obj.nnzh)
            rows[index] = nlp.obj.hessian.rows[index]
            cols[index] = nlp.obj.hessian.cols[index]
        end
    end
    if (nlp.obj.type == "NONLINEAR") || (nlp.meta.nnln > 0)
        hesslag_struct = MOI.hessian_lagrangian_structure(nlp.eval)
        for index = (nlp.obj.nnzh+1):(nlp.meta.nnzh)
            shift_index = index - nlp.obj.nnzh
            rows[index] = hesslag_struct[shift_index][1]
            cols[index] = hesslag_struct[shift_index][2]
        end
    end
    return rows, cols
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
            error("Unkown argument.")
        end
        field = what[2]
        setfield!(node, Symbol(field), value)
    end

    return termination_criteria, algorithm_params
end

MOI.get(model::CATSolver, ::MOI.RawSolver) = model

# copy
MOI.supports_incremental_interface(solver::CATSolver) = true
function MOI.copy_to(solver::CATSolver, src::MOI.ModelLike)
    return MOI.Utilities.automatic_copy_to(model, src)
end

function MOI.optimize!(solver::CATSolver)
    t = time()

    has_nlp_objective = false
    if !isa(solver.nlp_data.evaluator, EmptyNLPEvaluator)
        features = MOI.features_available(solver.nlp_data.evaluator)::Vector{Symbol}
        has_hessian = (:Hess in features)
        has_hessvec = (:HessVec in features)
        has_nlp_objective = solver.nlp_data.has_objective

        init_feat = Symbol[]
        has_nlp_objective && push!(init_feat, :Grad)
        if has_hessian
            push!(init_feat, :Hess)
        elseif has_hessvec
            push!(init_feat, :HessVec)
        end

        MOI.initialize(solver.nlp_data.evaluator, init_feat)
    end

    termination_criteria, algorithm_params = create_pars_JuMP(solver.options)
    x, status, iteration_stats, computation_stats, k =
        CAT_solve(solver, termination_criteria, algorithm_params)

    status_str = convertStatusCodeToStatusString(status)

    solver.inner = CATProblem()
    solver.inner.status = status_CAT_To_JuMP(status_str)
    solver.inner.x = x

    function_value = NaN
    gradient_value = NaN
    if size(last(iteration_stats, 1))[1] > 0
        function_value = last(iteration_stats, 1)[!, "fval"][1]
        gradient_value = last(iteration_stats, 1)[!, "gradval"][1]
    end

    solver.inner.obj_val = function_value
    solver.inner.grad_val = gradient_value
    solver.inner.itr = k
    solver.inner.solve_time = time() - t

    # custom CAT features
    solver.inner.termination_criteria = termination_criteria
    solver.inner.algorithm_params = algorithm_params
    solver.inner.iteration_stats = iteration_stats
    solver.inner.computation_stats = computation_stats
end

function convertStatusCodeToStatusString(status)
    dict_status_code = Dict(
        consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL =>
            "OPTIMAL",
        consistently_adaptive_trust_region_method.TerminationStatusCode.UNBOUNDED =>
            "UNBOUNDED",
        consistently_adaptive_trust_region_method.TerminationStatusCode.ITERATION_LIMIT =>
            "ITERATION_LIMIT",
        consistently_adaptive_trust_region_method.TerminationStatusCode.TIME_LIMIT =>
            "TIME_LIMIT",
        consistently_adaptive_trust_region_method.TerminationStatusCode.MEMORY_LIMIT =>
            "MEMORY_LIMIT",
        consistently_adaptive_trust_region_method.TerminationStatusCode.STEP_SIZE_LIMIT =>
            "STEP_SIZE_LIMIT",
        consistently_adaptive_trust_region_method.TerminationStatusCode.NUMERICAL_ERROR =>
            "NUMERICAL_ERROR",
        consistently_adaptive_trust_region_method.TerminationStatusCode.OTHER_ERROR =>
            "OTHER_ERROR",
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
           status == "STEP_SIZE_LIMIT"
        return :UserLimit
    else
        return :Error
    end
end

function MOI.optimize!(solver::CATSolver, jumpModel::Model)
    t = time()
    nlp = MathOptNLPModel(jumpModel)
    solver.nlp_data = nlp
    termination_criteria = TerminationCriteria()
    algorithm_params = AlgorithmicParameters()
    x, status, iteration_stats, computation_stats, k =
        CAT_solve(solver, termination_criteria, algorithm_params)

    status_str = convertStatusCodeToStatusString(status)

    solver.inner = CATProblem()
    solver.inner.status = status_CAT_To_JuMP(status_str)
    solver.inner.x = x

    function_value = NaN
    gradient_value = NaN
    if size(last(iteration_stats, 1))[1] > 0
        function_value = last(iteration_stats, 1)[!, "fval"][1]
        gradient_value = last(iteration_stats, 1)[!, "gradval"][1]
    end

    solver.inner.obj_val = function_value
    solver.inner.grad_val = gradient_value
    solver.inner.itr = k
    solver.inner.solve_time = time() - t

    # custom CAT features
    solver.inner.termination_criteria = termination_criteria
    solver.inner.algorithm_params = algorithm_params
    solver.inner.iteration_stats = iteration_stats
    solver.inner.computation_stats = computation_stats
end

function MOI.get(
    model::MathOptInterface.Utilities.CachingOptimizer,
    attr::MOI.TerminationStatus,
)
    return MOI.get(model.optimizer, attr)
end

function check_inbounds(model::CATSolver, vi::MOI.VariableIndex)
    return MOI.throw_if_not_valid(model, vi)
end

function check_inbounds(model::CATSolver, aff::MOI.ScalarAffineFunction)
    for term in aff.terms
        MOI.throw_if_not_valid(model, term.variable)
    end
end

function check_inbounds(model::CATSolver, quad::MOI.ScalarQuadraticFunction)
    for term in quad.affine_terms
        MOI.throw_if_not_valid(model, term.variable)
    end
    for term in quad.quadratic_terms
        MOI.throw_if_not_valid(model, term.variable_1)
        MOI.throw_if_not_valid(model, term.variable_2)
    end
end

MOI.supports(::CATSolver, ::MOI.NLPBlock) = true

function MOI.supports(::CATSolver, SF)
    return true
end

function MOI.supports(::CATSolver, ::MOI.ObjectiveFunction{MOI.VariableIndex})
    return true
end

function MOI.supports(
    ::CATSolver,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

function MOI.supports(
    ::CATSolver,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}},
)
    return true
end

MOI.supports(::CATSolver, ::MOI.ObjectiveSense) = true

MOI.supports(::CATSolver, ::MOI.Silent) = true

MOI.supports(::CATSolver, ::MOI.RawOptimizerAttribute) = true

function MOI.get(model::CATSolver, ::MOI.ObjectiveFunction)
    return model.objective
end

function MOI.set(model::CATSolver, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    model.nlp_data = nlp_data
    return
end

function MOI.set(model::CATSolver, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.sense = sense
    return
end

MOI.get(model::CATSolver, ::MOI.ObjectiveSense) = model.sense

function MOI.set(model::CATSolver, ::MOI.Silent, value)
    model.silent = value
    return
end

MOI.get(model::CATSolver, ::MOI.Silent) = model.silent

function MOI.supports(::CATSolver, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

function MOI.set(
    model::CATSolver,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, vi)
    model.variable_info[vi.value].start = value
    return
end

function MOI.get(model::MOIU.CachingOptimizer, args...)
    return MOI.get(model.model.optimizer, args)
end

function MOI.get(model::CATSolver, attr::MOI.VariablePrimal, v::VariableRef)
    return MOI.get(model, attr, v.variable)
end

function MOI.get(model::CATSolver, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    return model.inner.x[vi.value]
end

function MOI.set(model::CATSolver, ::MOI.TimeLimitSec, value::Real)
    MOI.set(model, MOI.RawOptimizerAttribute(TIME_LIMIT), Float64(value))
end

function MOI.set(model::CATSolver, p::MOI.RawOptimizerAttribute, value)
    model.options[p.name] = value
    return
end

function MOI.get(model::CATSolver, p::MOI.RawOptimizerAttribute)
    if haskey(model.options, p.name)
        return model.options[p.name]
    end
    error("RawParameter with name $(p.name) is not set.")
end

function MOI.get(model::CATSolver, ::MOI.TerminationStatus)
    if model.inner === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    status = model.inner.status
    return status
end

function MOI.get(model::CATSolver, ::MOI.RawStatusString)
    return string(model.inner.status)
end

function MOI.get(model::CATSolver, ::MOI.ResultCount)
    return (model.inner !== nothing) ? 1 : 0
end

function MOI.get(model::CATSolver, attr::MOI.PrimalStatus)
    if !(1 <= attr.N <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end

    status = model.inner.status
    return status
end

function MOI.get(model::CATSolver, attr::MOI.DualStatus)
    if !(1 <= attr.N <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = model.inner.status
    return status
end

function MOI.get(model::CATSolver, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.inner.obj_val
end
