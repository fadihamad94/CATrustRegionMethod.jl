__precompile__()
#Main algorithm code goes here
module consistently_adaptive_trust_region_method
using NLPModels, LinearAlgebra, DataFrames, SparseArrays, EnumX, JuMP, NLPModelsJuMP, MathOptInterface, Random

export TerminationConditions, INITIAL_RADIUS_STRUCT, Problem_Data
export phi, findinterval, bisection, computeSecondOrderModel, optimizeSecondOrderModel, compute_ρ_hat, CAT, CAT_solve

include("./trust_region_subproblem_solver.jl")
include("./common.jl")
include("./JuMPInterface.jl")
include("./CAT.jl")

end
