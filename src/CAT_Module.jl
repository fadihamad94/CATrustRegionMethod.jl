__precompile__()

module consistently_adaptive_trust_region_method
using NLPModels,
    LinearAlgebra,
    DataFrames,
    SparseArrays,
    EnumX,
    JuMP,
    NLPModelsJuMP,
    MathOptInterface,
    Random,
    CUTEst

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
include("./trust_region_subproblem_solver.jl")
include("./JuMPInterface.jl")
include("./CAT.jl")

end
