__precompile__()

module CATrustRegionMethod
import CATrustRegionShared
using CATrustRegionShared:
    AbstractUnconstrainedOptimizer,
    EmptyNLPEvaluator,
    VariableInfo,
    @define_algorithm_counter,
    @define_termination_criteria,
    @define_termination_status,
    canonical_status_string,
    empty_nlp_data,
    evaluate_function,
    evaluate_gradient,
    evaluate_hessian,
    record_factorization_stats!,
    record_iterative_stats!
using NLPModels,
    LinearAlgebra,
    DataFrames,
    Dates,
    SparseArrays,
    EnumX,
    JuMP,
    NLPModelsJuMP,
    MathOptInterface,
    Random,
    CUTEst,
    JSON

import TrustRegionSubproblemSolvers
using TrustRegionSubproblemSolvers:
    DIRECT_NEW_SOLVER,
    HessianMatrix,
    OptionalSparseCholeskyWorkspace,
    SparseCholeskyWorkspace,
    TrustRegionSubproblemError,
    matrix_l2_norm,
    restoreFullMatrix,
    selectHessianRepresentation,
    setSubproblemFailureDiagnosticContext,
    solveTrustRegionSubproblemOldApproach

export TerminationCriteria, AlgorithmicParameters, AlgorithmCounter
export computeSecondOrderModel,
    compute_ρ_hat,
    CAT,
    CAT_solve

include("common.jl")
include("MOI_wrapper.jl")
include("main.jl")

end
