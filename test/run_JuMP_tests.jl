using MathOptInterface
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

function rosenbrook2()
    model = Model()
    @variable(model, x >= 0.0)
    @variable(model, y >= 0.0)
    @NLobjective(model, Min, (2.0 - x)^2 + 100 * (y - x^2)^2)
    @constraint(model, x + y >= 0.1)
	@NLconstraint(model, x* y + x >= 0.1)
    return model
end

function createHardCaseUsingSimpleBivariateConvexProblem()
    model = Model(consistently_adaptive_trust_region_method.CATSolver)
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x ^ 2 - 10 * x * y + y ^ 2)
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

function optimize_rosenbrook1_model_JuMPInterface_with_default_arguments()
	default_β_1 = 0.1
	default_θ = 0.1
	default_ω_1 = 8.0
	default_ω_2 = 20.0
	default_γ_1 = 1e-2
	default_γ_2 = 0.8
	default_r_1 = 0.0
	default_print_level = 0
	default_max_iterations = 100000
	default_gradient_termination_tolerance = 1e-5
	default_max_time = 5 * 60 * 60.0
	default_step_size_limit = 2.0e-16
	options = Dict{String, Any}("initial_radius_struct!r_1"=>default_r_1,
    	"β_1"=>default_β_1,
		"ω_2"=>default_ω_2,
        "print_level"=>default_print_level,
        "termination_conditions_struct!MAX_ITERATIONS"=>default_max_iterations,
        "termination_conditions_struct!gradient_termination_tolerance"=>default_gradient_termination_tolerance)
	model = rosenbrook1()
	attachSolverWithAttributesToJuMPModel(model, options)

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

	@test optimizer.inner.pars.β_1 == default_β_1
	@test optimizer.inner.pars.θ == default_θ
	@test optimizer.inner.pars.ω_1 == default_ω_1
	@test optimizer.inner.pars.ω_2 == default_ω_2
	@test optimizer.inner.pars.γ_1 == default_γ_1
	@test optimizer.inner.pars.γ_2 == default_γ_2
	@test optimizer.inner.pars.initial_radius_struct.r_1 == default_r_1
	@test optimizer.inner.pars.print_level == default_print_level
	@test optimizer.inner.pars.termination_conditions_struct.MAX_ITERATIONS == default_max_iterations
	@test optimizer.inner.pars.termination_conditions_struct.gradient_termination_tolerance == default_gradient_termination_tolerance
	@test optimizer.inner.pars.termination_conditions_struct.MAX_TIME == default_max_time
	@test optimizer.inner.pars.termination_conditions_struct.STEP_SIZE_LIMIT == default_step_size_limit
end

function optimize_rosenbrook1_model_JuMPInterface_with_user_specified_arguments()
	β_1 = 0.2
	ω_2 = 8.0
	r_1 = 100.0
	print_level = -1
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

	@test optimizer.inner.pars.β_1 == β_1
	@test optimizer.inner.pars.ω_2 == ω_2
	@test optimizer.inner.pars.initial_radius_struct.r_1 == r_1
	@test optimizer.inner.pars.print_level == print_level
	@test optimizer.inner.pars.termination_conditions_struct.MAX_ITERATIONS == MAX_ITERATIONS
	@test optimizer.inner.pars.termination_conditions_struct.gradient_termination_tolerance == gradient_termination_tolerance
end

function optimize_model_with_constraints_failure_expected()
	model = rosenbrook2()
    set_optimizer(model, consistently_adaptive_trust_region_method.CATSolver)
	try
		optimize!(model)
	catch e
		@test e == MOI.UnsupportedConstraint{MathOptInterface.VariableIndex, MathOptInterface.GreaterThan{Float64}}("")
	end
end

function optimizeHardCaseUsingSimpleBivariateConvexProblem()
	model = createHardCaseUsingSimpleBivariateConvexProblem()

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

function optimize_models_JuMPInterface()
    optimize_rosenbrook1_model_JuMPInterface_with_default_arguments()
    optimize_rosenbrook1_model_JuMPInterface_with_user_specified_arguments()
	optimize_model_with_constraints_failure_expected()
	optimizeHardCaseUsingSimpleBivariateConvexProblem()
end

@testset "optimization_using_JUMP_interface" begin
    optimize_models_JuMPInterface()
end
