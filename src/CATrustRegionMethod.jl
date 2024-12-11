__precompile__()

module CATrustRegionMethod
using NLPModels,
    LinearAlgebra,
    DataFrames,
    SparseArrays,
    EnumX,
    JuMP,
    NLPModelsJuMP,
    MathOptInterface,
    Random,
    CUTEst,
    CSV

export TerminationCriteria, AlgorithmicParameters, AlgorithmCounter
export phi,
    findinterval,
    bisection,
    computeSecondOrderModel,
    optimizeSecondOrderModel,
    compute_ρ_hat,
    CAT,
    CAT_solve

include("./common.jl")
include("./utils.jl")
include("./old_trust_region_subproblem_solver.jl")
include("./trust_region_subproblem_solver.jl")
include("./MOI_wrapper.jl")
include("./main.jl")

end
