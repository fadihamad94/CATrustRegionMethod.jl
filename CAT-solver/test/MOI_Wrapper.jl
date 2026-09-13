# ============================ /test/MOI_wrapper.jl ============================
module TestCAT

using Test
import CATrustRegionMethod
import MathOptInterface as MOI

const OPTIMIZER = MOI.instantiate(
    MOI.OptimizerWithAttributes(CATrustRegionMethod.Optimizer, MOI.Silent() => true),
)

const BRIDGED = MOI.instantiate(
    MOI.OptimizerWithAttributes(CATrustRegionMethod.Optimizer, MOI.Silent() => true),
    with_bridge_type = Float64,
)

# See the docstring of MOI.Test.Config for other arguments.
const CONFIG = MOI.Test.Config(
    # Modify tolerances as necessary.
    atol = 1e-6,
    rtol = 1e-6,
    # Use MOI.LOCALLY_SOLVED for local solvers.
    optimal_status = MOI.OPTIMAL,
    # Pass attributes or MOI functions to `exclude` to skip tests that
    # rely on this functionality.
    exclude = Any[MOI.VariableName, MOI.delete],
)

"""
    runtests()

This function runs all functions in the this Module starting with `test_`.
"""
function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", r"^test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

"""
    test_runtests()

This function runs all the tests in MathOptInterface.Test.

Pass arguments to `exclude` to skip tests for functionality that is not
implemented or that your solver doesn't support.
"""
function test_runtests()
    MOI.Test.runtests(
        BRIDGED,
        CONFIG,
        exclude = [
            r"^test_attribute_NumberThreads$",
            # CAT is unconstrained; every `test_basic_` case requires a constraint.
            r"^test_basic_",
            # CAT supports one scalar objective, not vector-valued multiobjective models.
            r"^test_multiobjective_",
            # CAT is not compliant with the MOI.ListOfModelAttributesSet attribute
            r"^test_objective_get_ObjectiveFunction_ScalarAffineFunction$", # TODO SUPPORT
            # This to test an unbounded linear program. CAT works only with functions that are at least twice differentiable.
            r"^test_solve_TerminationStatus_DUAL_INFEASIBLE$", # TODO support
            # CAT is not compliant with the MOI.ListOfModelAttributesSet attribute
            "_in_ListOfModelAttributesSet",  # TODO support
        ],
        # Keep the compliance surface pinned to the MOI version in Manifest.toml so
        # dependency upgrades cannot silently add new interface requirements.
        exclude_tests_after = v"1.51.1",
    )
    return
end

"""
    test_SolverName()

You can also write new tests for solver-specific functionality. Write each new
test as a function with a name beginning with `test_`.
"""
function test_SolverName()
    @test MOI.get(CATrustRegionMethod.Optimizer(), MOI.SolverName()) == "CATOptimizer"
    return
end

end # module TestCAT

# This line at tne end of the file runs all the tests!
@testset "MOI_Validation_Tests" begin
    TestCAT.runtests()
end
