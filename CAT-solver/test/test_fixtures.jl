using JuMP
using NLPModelsJuMP
using CATrustRegionShared

import CATrustRegionMethod
include(
    joinpath(
        pkgdir(CATrustRegionShared),
        "test_support",
        "SharedNLPTestModels.jl",
    ),
)
using .SharedNLPTestModels

function cat_test_problem(nlp)
    termination_criteria = CATrustRegionMethod.TerminationCriteria(100, 1e-4)
    algorithm_params = CATrustRegionMethod.AlgorithmicParameters(0.25, 0.5, 2.0, 2.0)
    algorithm_params.r_1 = 0.5
    return nlp, termination_criteria, algorithm_params
end

test_create_dummy_problem() = cat_test_problem(createDummyNLPModel())
test_create_dummy_problem2() = cat_test_problem(createDummyNLPModel2())
test_create_simple_convex_nlp_model() = cat_test_problem(createSimpleConvexNLPModeL())
test_create_complex_convex_nlp1_model() = cat_test_problem(createComplexConvexNLPModeL1())
test_create_complex_nlp_modeL1() = cat_test_problem(createComplexNLPModeL1())
test_create_problem_sin_cos_mode_nlp1() = cat_test_problem(createSinCosNLPModeL1())
test_create_problem_sin_cos_mode_nlp2() = cat_test_problem(createSinCosNLPModeL2())
