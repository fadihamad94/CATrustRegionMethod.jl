__precompile__()

module CATrustRegionShared

using EnumX
using JSON
using MathOptInterface
using NLPModels
using Printf
using SparseArrays

const MOI = MathOptInterface

export AbstractUnconstrainedOptimizer,
    EmptyNLPEvaluator,
    VariableInfo,
    @define_algorithm_counter,
    @define_termination_criteria,
    @define_termination_status,
    assert_factorization_accounting,
    canonical_status_string,
    empty_nlp_data,
    evaluate_function,
    evaluate_gradient,
    evaluate_hessian,
    format_to_six_decimals,
    increment!,
    model_changed!,
    record_factorization_stats!,
    record_iterative_stats!

include("defaults.jl")
include("types.jl")
include("counters.jl")
include("evaluation.jl")
include("moi_storage.jl")
include("formatting.jl")

end
