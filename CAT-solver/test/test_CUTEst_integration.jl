using CUTEst
using Test

import CATrustRegionMethod

@testset "CUTEstModel optimization does not return OTHER_ERROR" begin
    nlp = CUTEstModel{Float64}("ARGLINA")
    try
        termination_criteria = CATrustRegionMethod.TerminationCriteria(1, 1.0e-5, 30.0)
        algorithm_params = CATrustRegionMethod.AlgorithmicParameters()
        algorithm_params.print_level = -1

        _, status, _, algorithm_counter = CATrustRegionMethod.optimize(
            nlp,
            termination_criteria,
            algorithm_params,
            nlp.meta.x0,
            0.0,
        )

        @test status != CATrustRegionMethod.TerminationStatusCode.OTHER_ERROR
        @test algorithm_counter.total_function_evaluation > 0
        @test algorithm_counter.total_gradient_evaluation > 0
        @test algorithm_counter.total_hessian_evaluation > 0
    finally
        finalize(nlp)
    end
end
