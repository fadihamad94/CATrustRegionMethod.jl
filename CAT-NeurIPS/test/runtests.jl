using CATNeurIPS
using CATrustRegionShared
using Test

include(
    joinpath(
        pkgdir(CATrustRegionShared),
        "test_support",
        "SharedNLPTestModels.jl",
    ),
)
using .SharedNLPTestModels

@testset "CATNeurIPS" begin
    @testset "parameter validation" begin
        @test CATNeurIPS.AlgorithmicParameters().omega == 8.0
        @test_throws ArgumentError CATNeurIPS.AlgorithmicParameters(0.0)
        @test_throws ArgumentError CATNeurIPS.AlgorithmicParameters(
            0.1,
            0.1,
            1.0,
        )
        @test_throws ArgumentError CATNeurIPS.TerminationCriteria(0, 1.0e-5, 1.0)
    end

    @testset "initial optimum" begin
        nlp = createSimpleUnivariateConvexProblem(1.0)
        result = CATNeurIPS.solve(
            nlp,
            CATNeurIPS.TerminationCriteria(10, 1.0e-8, 10.0),
        )
        @test result.status == "OPTIMAL"
        @test result.solution == [1.0]
        @test result.objective == 0.0
        @test result.gradient_norm == 0.0
        @test result.subproblem_solves == 0
    end

    @testset "conference outer method uses maintained old subproblem" begin
        nlp = createSimpleUnivariateConvexProblem(0.0)
        parameters = CATNeurIPS.AlgorithmicParameters(
            0.1,
            0.1,
            8.0,
            2.0,
            0.0,
            0.8,
        )
        result = CATNeurIPS.solve(
            nlp,
            CATNeurIPS.TerminationCriteria(10, 1.0e-8, 10.0),
            parameters,
        )
        @test result.status == "OPTIMAL"
        @test result.solution ≈ [1.0]
        @test result.objective ≈ 0.0 atol = 1.0e-12
        @test result.gradient_norm ≈ 0.0 atol = 1.0e-12
        @test result.subproblem_solves == 1
    end
end
