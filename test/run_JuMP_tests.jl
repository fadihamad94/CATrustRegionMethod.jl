######################
##### ROSENBROOK #####
######################
function rosenbrook1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (2.0 - x)^2 + 100 * (y - x^2)^2)
    return model
end

######################
####Utility Method####
######################
function attachSolverWithAttributesToJuMPModel(model:: Model, options::Dict{String, Any})
	set_optimizer(model, consistently_adaptive_trust_region_method.CATSolver)
	for (name, value) in options
        sname = string(name)
	set_optimizer_attribute(model, sname, value)
    end
end

function optimize_rosenbrook1_model_JuMPInterface()
    model = rosenbrook1()
    set_optimizer(model, consistently_adaptive_trust_region_method.CATSolver)

    #Test using JUMP
    optimize!(model)
    x = JuMP.value.(model[:x])
    y = JuMP.value.(model[:y])
    status = MOI.get(model, MOI.TerminationStatus())
    @test status == :Optimal
    # Retrieve the solver instance
    optimizer = backend(model).optimizer.model

    nlp = MathOptNLPModel(model)
    termination_conditions_struct_default = consistently_adaptive_trust_region_method.TerminationConditions()
    initial_radius_struct_default = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct_default, initial_radius_struct_default)
    x_k, status, iteration_stats, computation_stats, itr =  consistently_adaptive_trust_region_method.CAT(problem, nlp.meta.x0, 0.0)

    @test x_k == [x, y]
    @test itr == optimizer.inner.itr
    @test x_k == optimizer.inner.x
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL
    @test iteration_stats == optimizer.inner.iteration_stats
    @test computation_stats == optimizer.inner.computation_stats
end

function optimize_rosenbrook1_model_JuMPInterface_with_arguments()
	β_1 = 0.2
	ω_2 = 8.0
	r_1 = 100.0
	print_level = 1
	MAX_ITERATIONS = 10
	gradient_termination_tolerance = 1e-3
    options = Dict{String, Any}("initial_radius_struct!r_1"=>r_1,
    	"β_1"=>β_1,
		"ω_2"=>ω_2,
        "print_level"=>print_level,
        "termination_conditions_struct!MAX_ITERATIONS"=>MAX_ITERATIONS,
        "termination_conditions_struct!gradient_termination_tolerance"=>gradient_termination_tolerance)
	model = rosenbrook1()
	attachSolverWithAttributesToJuMPModel(model, options)

	#Test using JUMP (UserLimit due to MAX_ITERATIONS = 10)
    optimize!(model)
    x = JuMP.value.(model[:x])
    y = JuMP.value.(model[:y])
    status = MOI.get(model, MOI.TerminationStatus())
	@test status == :UserLimit

	# Retrieve the solver instance
    optimizer = backend(model).optimizer.model

    nlp = MathOptNLPModel(model)
    termination_conditions_struct_default = consistently_adaptive_trust_region_method.TerminationConditions()
    initial_radius_struct_default = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct_default, initial_radius_struct_default)
	problem.β_1 = β_1
	problem.ω_2 = ω_2
	problem.initial_radius_struct.r_1 = r_1
	problem.print_level = print_level
	problem.termination_conditions_struct.MAX_ITERATIONS = MAX_ITERATIONS
	problem.termination_conditions_struct.gradient_termination_tolerance = gradient_termination_tolerance
    x_k, status, iteration_stats, computation_stats, itr =  consistently_adaptive_trust_region_method.CAT(problem, nlp.meta.x0, 0.0)

    @test x_k == [x, y]
    @test itr == optimizer.inner.itr
    @test x_k == optimizer.inner.x
    @test status == consistently_adaptive_trust_region_method.TerminationStatusCode.ITERATION_LIMIT
    @test iteration_stats == optimizer.inner.iteration_stats
    @test computation_stats == optimizer.inner.computation_stats
end

function optimize_models_JuMPInterface()
    optimize_rosenbrook1_model_JuMPInterface()
    optimize_rosenbrook1_model_JuMPInterface_with_arguments()
end

@testset "optimization_using_JUMP_interface" begin
    optimize_models_JuMPInterface()
end
