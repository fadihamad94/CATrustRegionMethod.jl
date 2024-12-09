# ============================ /test/MOI_wrapper.jl ============================
module TestCAT
include("../src/CAT.jl")
using Test
import MathOptInterface as MOI

const OPTIMIZER = MOI.instantiate(
    MOI.OptimizerWithAttributes(CAT.CATSolver, MOI.Silent() => true),
)

const BRIDGED = MOI.instantiate(
    MOI.OptimizerWithAttributes(CAT.CATSolver, MOI.Silent() => true),
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
            # r"^test",
            r"^test_attribute_NumberThreads$",
            # r"^test_add_parameter$",
            # r"^test_attribute_ObjectiveLimit$",
            # r"^test_attribute_AbsoluteGapTolerance$",
            # r"^test_attribute_RelativeGapTolerance$",
            # r"^test_attribute_SolutionLimit$",
            # r"^test_attribute_SolveTimeSec$",
            # r"^test_attribute_TimeLimitSec$",
            # r"^test_multiobjective_vector_affine_function$",
            # r"^test_multiobjective_vector_affine_function_delete$",
            # r"^test_multiobjective_vector_affine_function_delete_vector$",
            # r"^test_multiobjective_vector_affine_function_modify$",
            # r"^test_multiobjective_vector_nonlinear$",
            # r"^test_multiobjective_vector_nonlinear_delete$",
            # r"^test_multiobjective_vector_nonlinear_delete_vector$",
            # r"^test_multiobjective_vector_nonlinear_modify$",
            # r"^test_multiobjective_vector_of_variables$",
            # r"^test_multiobjective_vector_of_variables_delete$",
            # r"^test_multiobjective_vector_of_variables_delete_all$",
            # r"^test_multiobjective_vector_of_variables_delete_vector$",
            # r"^test_multiobjective_vector_quadratic_function$",
            # r"^test_multiobjective_vector_quadratic_function_delete$",
            # r"^test_multiobjective_vector_quadratic_function_delete_vector$",
            # r"^test_multiobjective_vector_quadratic_function_modify$",
            # r"^test_nonlinear_duals$",
            # r"^test_nonlinear_expression_hs071$",
            # r"^test_nonlinear_expression_hs071_epigraph$",
            # r"^test_nonlinear_expression_hs109$",
            # r"^test_nonlinear_expression_hs110$",
            # r"^test_nonlinear_expression_multivariate_function$",
            # r"^test_nonlinear_expression_overrides_objective$",
            # r"^test_nonlinear_expression_quartic$",
            # r"^test_nonlinear_expression_univariate_function$",
            # r"^test_nonlinear_hs071_no_hessian$",
            # CAT only supports unconstrained problems.
            # r"^test_nonlinear_objective$",
            # r"^test_nonlinear_objective_and_moi_objective_test$",
            # r"^test_nonlinear_without_objective$",
            # r"^test_objective_ObjectiveFunction_blank$",
            # r"^test_objective_ObjectiveSense_in_ListOfModelAttributesSet$",
            # r"^test_objective_ScalarAffineFunction_in_ListOfModelAttributesSet$",
            # r"^test_objective_ScalarQuadraticFunction_in_ListOfModelAttributesSet$",
            # r"^test_objective_VariableIndex_in_ListOfModelAttributesSet$",
            r"^test_objective_get_ObjectiveFunction_ScalarAffineFunction$", # TODO check
            r"^test_objective_set_via_modify$", #TODO check
            r"^test_solve_TerminationStatus_DUAL_INFEASIBLE$", # TODO check
            # r"^test_variable_add_variable$",
            # r"^test_variable_add_variables$",
            # r"^test_variable_delete$",
            # r"^test_variable_delete_variables$",
            r"^test_nonlinear_invalid$", # TODO check
            # r"^test_model_default_TerminationStatus$",
            # r"^test_model_default_PrimalStatus$",
            # r"^test_model_default_DualStatus$",
            # r"^test_model_VariablePrimalStart$",
            r"^test_model_Name$",
            r"^test_quadratic$",
            # CAT is not compliant with the MOI.ListOfModelAttributesSet attribute
            "_in_ListOfModelAttributesSet",
        ],
        # This argument is useful to prevent tests from failing on future
        # releases of MOI that add new tests. Don't let this number get too far
        # behind the current MOI release though. You should periodically check
        # for new tests to fix bugs and implement new features.
        exclude_tests_after = v"0.10.5",
    )
    return
end

# function test_runtests()
#     MOI.Test.runtests(
#         OPTIMIZER, CONFIG,
#         exclude=[
#             # behaviour to implement: list of model, constraint attributes set
#             "test_model_ListOfConstraintAttributesSet",
#             "test_model_ModelFilter_AbstractModelAttribute",
#             "test_model_ModelFilter_ListOfConstraintIndices",
#             "test_model_ModelFilter_ListOfConstraintTypesPresent",
#             "test_model_Name",
#             "test_objective_set_via_modify",
#             # requires get quadratic objective
#             "test_objective_get_ObjectiveFunction_ScalarAffineFunction",
#             # Tulip not compliant with MOI convention for primal/dual infeasible models
#             # See expected behavior at https://jump.dev/MathOptInterface.jl/dev/background/infeasibility_certificates/
#             "test_unbounded",
#             # Tulip is not compliant with the MOI.ListOfModelAttributesSet attribute
#             "_in_ListOfModelAttributesSet",
#         ]
#     )
# end

"""
    test_SolverName()

You can also write new tests for solver-specific functionality. Write each new
test as a function with a name beginning with `test_`.
"""
function test_SolverName()
    @test MOI.get(CAT.CATSolver(), MOI.SolverName()) == "CATSolver"
    return
end

end # module TestCAT

# This line at tne end of the file runs all the tests!
TestCAT.runtests()
