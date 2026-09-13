__precompile__()

module TrustRegionSubproblemSolvers

using CSV
using DataFrames
using Dates
using DelimitedFiles
using JSON
using LinearAlgebra
using Random
using SparseArrays

export DIRECT_NEW_SOLVER,
    FactorizationStats,
    HessianMatrix,
    IterativeStats,
    OptionalSparseCholeskyWorkspace,
    SparseCholeskyWorkspace,
    TrustRegionSubproblemError,
    TrustRegionSubproblemResult
export solveTrustRegionSubproblem, solveTrustRegionSubproblemOldApproach
export checkTrustRegionSubproblemTerminationCriteria,
    effectiveHessianDensity,
    matrix_l2_norm,
    selectHessianRepresentation,
    setSubproblemFailureDetailsDirectory,
    validateTrustRegionSubproblemTerminationCriteria

include("common.jl")
include("shared_trust_region_solver.jl")
include("utils.jl")
include("validate_region_solution.jl")
include("old_trust_region_subproblem_solver.jl")
include("trust_region_subproblem_solver.jl")

end
