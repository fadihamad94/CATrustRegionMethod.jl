__precompile__()

module UniversalTrustRegionMethod

import CATrustRegionShared
import CATrustRegionShared: model_changed!
using CATrustRegionShared:
    AbstractUnconstrainedOptimizer,
    EmptyNLPEvaluator,
    VariableInfo,
    @define_algorithm_counter,
    @define_termination_criteria,
    assert_factorization_accounting,
    empty_nlp_data,
    evaluate_function,
    evaluate_gradient,
    evaluate_hessian,
    increment!,
    record_factorization_stats!
using DataFrames
using EnumX
using JSON
using JuMP
using LinearAlgebra
using MathOptInterface
using NLPModels
using NLPModelsJuMP
using Random
using SparseArrays

import TrustRegionSubproblemSolvers
using TrustRegionSubproblemSolvers:
    DIRECT_NEW_SOLVER,
    FactorizationStats,
    HessianMatrix,
    OptionalSparseCholeskyWorkspace,
    SparseCholeskyWorkspace,
    TrustRegionSubproblemError,
    factorizeShiftedHessian!,
    restoreFullMatrix,
    selectHessianRepresentation,
    setSubproblemFailureDiagnosticContext,
    solveTrustRegionSubproblem

export AlgorithmCounter, AlgorithmicParameters, Optimizer, TerminationCriteria, UTR_solve

include("common.jl")
include("utils.jl")
include("MOI_wrapper.jl")
include("main.jl")

end
