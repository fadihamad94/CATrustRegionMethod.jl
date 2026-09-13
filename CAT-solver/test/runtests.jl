using NLPModels, LinearAlgebra, DataFrames, SparseArrays

include("test_package_split.jl")
include("test_fixtures.jl")
include("test_CAT_model_and_parameters.jl")
include("run_CAT_tests.jl")
include("run_MOI_wrapper_tests.jl")
include("MOI_Wrapper.jl")
include("test_CUTEst_integration.jl")
