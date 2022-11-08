using JuMP, NLPModels, NLPModelsJuMP, Random, Distributions, LinearAlgebra, Test, Optim, DataFrames, StatsBase, CSV
include("../src/CAT.jl")
include("../src/tru.jl")
include("../src/arc.jl")
Random.seed!(0)

const CAT_SOLVER = "CAT"
const NEWTON_TRUST_REGION_SOLVER = "NewtonTrustRegion"
const ARC_SOLVER = "ARC_SOLVER"
const TRU_FACTORIZATION = "StandardTrustRegionMethodFactorization"
const TRU_ITERATIVE = "StandardTrustRegionMethodIterative"

function formulateLogissticRegressionProblem(x::Matrix{Float64}, y::Matrix{Int64})
    model = Model()
    n = size(x)[1]
    d = size(x)[2]

    @variable(model, θ[i=1:d])
    @NLobjective(model, Min, sum(log(1 + exp(sum(-y[i, j] * θ[j] * x[i, j] for j in 1:d))) for i in 1:n))
    return model
end

x = rand(10, 8)
y = ones(Int64, 10, 8)

formulateLogissticRegressionProblem(x, y)

function f(x::Vector)
	obj(nlp, x)
end

function g!(storage::Vector, x::Vector)
	storage[:] = grad(nlp, x)
end

function fg!(g::Vector, x::Vector)
	g[:] = grad(nlp, x)
	obj(nlp, x)
end

function h!(storage::Matrix, x::Vector)
	storage[:, :] = hess(nlp, x)
end

function hv!(Hv::Vector, x::Vector, v::Vector)
	H = hess(nlp, x)
    Hv[:] = H * v
end

function validateResults(
	nlp::AbstractNLPModel,
	solution::Vector,
	x::Matrix,
	y::Matrix,
	θ::Vector,
	n::Int64,
	d::Int64
	)

	objective_function_nlp_model = obj(nlp, solution)
    objective_function_formulation =  sum(log(1 + exp(sum(-y[i, j] * θ[j] * x[i, j] for j in 1:d))) for i in 1:n)
    @test norm(objective_function_nlp_model - objective_function_formulation, 2) <= 1e-8
end

function solveLogisticRegression(
	max_it::Int64,
	max_time::Float64,
	tol_opt::Float64,
	x::Matrix{Float64},
	y::Matrix{Int64}
	)
	n = size(x)[1]
    d = size(x)[2]

	all_results = DataFrame(solver = [], itr = [], total_function_evaluation = [],  total_gradient_evaluation = [])

    model = formulateLogissticRegressionProblem(x, y)
    global nlp = MathOptNLPModel(model)
    x0 = nlp.meta.x0
	GRADIENT_TOLERANCE = tol_opt
	ITERATION_LIMIT = max_it

	println("------------------Solving Using CAT-----------------")
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.1, 0.1, 8.0, 1.0, ITERATION_LIMIT, GRADIENT_TOLERANCE)
    δ = 0.0
    # solution, status, iteration_stats, computation_stats, itr = consistently_adaptive_trust_region_method.CAT(problem, x0, δ, "GALAHAD_TRS")
	solution, status, iteration_stats, computation_stats, itr = consistently_adaptive_trust_region_method.CAT(problem, x0, δ)
	@show norm(solution, 2)
    θ = solution
	validateResults(nlp, solution, x, y, θ, n, d)
	push!(all_results, (CAT_SOLVER, itr, computation_stats["total_function_evaluation"], computation_stats["total_gradient_evaluation"]))

	println("------------------Solving Using NewtonTrustRegion------------------")
	d_ = Optim.TwiceDifferentiable(f, g!, h!, nlp.meta.x0)
	results = optimize(d_, nlp.meta.x0, Optim.NewtonTrustRegion(), Optim.Options(show_trace=false, iterations = ITERATION_LIMIT, f_calls_limit = ITERATION_LIMIT, g_abstol = GRADIENT_TOLERANCE))
	if  !Optim.converged(results)
		push!(all_results, (NEWTON_TRUST_REGION_SOLVER, ITERATION_LIMIT, ITERATION_LIMIT, ITERATION_LIMIT))
	else
		solution = Optim.minimizer(results)
		@show norm(solution, 2)
		θ = solution
		validateResults(nlp, solution, x, y, θ, n, d)
		push!(all_results, (NEWTON_TRUST_REGION_SOLVER, Optim.iterations(results), Optim.f_calls(results), Optim.g_calls(results)))
	end

	println("------------------Solving Using ARC-----------------")
	print_level = 0
	initial_weight = 1.0
	userdata, solution = arc(length(nlp.meta.x0), nlp.meta.x0, grad(nlp, nlp.meta.x0), print_level, ITERATION_LIMIT, initial_weight)
	status = userdata.status
	iter = userdata.iter
	total_function_evaluation = userdata.total_function_evaluation
	total_gradient_evaluation = userdata.total_gradient_evaluation
	total_hessian_evaluation = userdata.total_hessian_evaluation
	@show status, iter, total_function_evaluation, total_gradient_evaluation, total_hessian_evaluation
	@show norm(solution, 2)
	θ = solution
	validateResults(nlp, solution, x, y, θ, n, d)
	push!(all_results, (ARC_SOLVER, iter, total_function_evaluation, total_gradient_evaluation))

	println("------------------Solving Using TRU Factorization-----------------")
	print_level = 0
	initial_radius = 1.0
	subproblem_direct = true
	initial_x = zeros(length(nlp.meta.x0))
	userdata, solution = tru(length(initial_x), initial_x, grad(nlp, initial_x), print_level, ITERATION_LIMIT, initial_radius, subproblem_direct)
	status = userdata.status
	iter = userdata.iter
	total_function_evaluation = userdata.total_function_evaluation
	total_gradient_evaluation = userdata.total_gradient_evaluation
	total_hessian_evaluation = userdata.total_hessian_evaluation
	@show status, iter, total_function_evaluation, total_gradient_evaluation, total_hessian_evaluation
	@show norm(solution, 2)
	θ = solution
	validateResults(nlp, solution, x, y, θ, n, d)
	push!(all_results, (TRU_FACTORIZATION, iter, total_function_evaluation, total_gradient_evaluation))

	println("------------------Solving Using TRU Iterative-----------------")
	print_level = 0
	initial_radius = 1.0
	subproblem_direct = false
	initial_x = zeros(length(initial_x))
	userdata, solution = tru(length(initial_x), initial_x, grad(nlp, initial_x), print_level, ITERATION_LIMIT, initial_radius, subproblem_direct)
	status = userdata.status
	iter = userdata.iter
	total_function_evaluation = userdata.total_function_evaluation
	total_gradient_evaluation = userdata.total_gradient_evaluation
	total_hessian_evaluation = userdata.total_hessian_evaluation
	@show status, iter, total_function_evaluation, total_gradient_evaluation, total_hessian_evaluation
	@show norm(solution, 2)
	θ = solution
	validateResults(nlp, solution, x, y, θ, n, d)
	push!(all_results, (TRU_ITERATIVE, iter, total_function_evaluation, total_gradient_evaluation))

	return all_results
end

function filterRows(solver::String, iterations_vector::Vector{Int64})
    return filter!(x->x == solver, iterations_vector)
end

equals_method(name::String, solver::String) = name == solver

function saveResults(dirrectoryName::String, all_instances_results::Dict{String, Vector{Any}})
	fileNameCAT = "all_instances_results_CAT.csv"
	fileNameNewton = "all_instances_results_newton.csv"
	fileNameARC = "all_instances_results_arc.csv"
	fileNameTRU_F = "all_instances_results_tru_factorization.csv"
	fileNameTRU_I = "all_instances_results_tru_iterative.csv"
	fullPathCAT = string(dirrectoryName, "/", fileNameCAT)
	fullPathNewton = string(dirrectoryName, "/", fileNameNewton)
	fullPathARC = string(dirrectoryName, "/", fileNameARC)
	fullPathTRU_F = string(dirrectoryName, "/", fileNameTRU_F)
	fullPathTRU_I = string(dirrectoryName, "/", fileNameTRU_I)

	all_instances_results_CAT = all_instances_results[CAT_SOLVER]
	all_instances_results_newton = all_instances_results[NEWTON_TRUST_REGION_SOLVER]
	all_instances_results_arc = all_instances_results[ARC_SOLVER]
	all_instances_results_tru_f = all_instances_results[TRU_FACTORIZATION]
	all_instances_results_tru_i = all_instances_results[TRU_ITERATIVE]

	df_CAT = DataFrame(iter=[], fct=[], gradient=[])
	for i in 1:length(all_instances_results_CAT)
		push!(df_CAT, (all_instances_results_CAT[i][1], all_instances_results_CAT[i][2], all_instances_results_CAT[i][3]))
	end

	df_newton = DataFrame(iter=[], fct=[], gradient=[])
	for i in 1:length(all_instances_results_newton)
		push!(df_newton, (all_instances_results_newton[i][1], all_instances_results_newton[i][2], all_instances_results_newton[i][3]))
	end

	df_arc = DataFrame(iter=[], fct=[], gradient=[])
	for i in 1:length(all_instances_results_arc)
		push!(df_arc, (all_instances_results_arc[i][1], all_instances_results_arc[i][2], all_instances_results_arc[i][3]))
	end

	df_tru_f = DataFrame(iter=[], fct=[], gradient=[])
	for i in 1:length(all_instances_results_tru_f)
		push!(df_tru_f, (all_instances_results_tru_f[i][1], all_instances_results_tru_f[i][2], all_instances_results_tru_f[i][3]))
	end

	df_tru_i = DataFrame(iter=[], fct=[], gradient=[])
	for i in 1:length(all_instances_results_tru_i)
		push!(df_tru_i, (all_instances_results_tru_i[i][1], all_instances_results_tru_i[i][2], all_instances_results_tru_i[i][3]))
	end

	CSV.write(fullPathCAT, df_CAT)
	CSV.write(fullPathNewton, df_newton)
	CSV.write(fullPathARC, df_arc)
	CSV.write(fullPathTRU_F, df_tru_f)
	CSV.write(fullPathTRU_I, df_tru_i)

	return df_CAT, df_newton
end

function computeGeometricMeans(dirrectoryName::String, all_instances_results::Dict{String, Vector{Any}}, paper_results::DataFrame)
	results_geomean = DataFrame(solver = [], itr = [], total_function_evaluation = [],  total_gradient_evaluation = [])
	for key in keys(all_instances_results)
		temp = all_instances_results[key]
		temp_matrix = zeros(length(temp), length(temp[1]))
		count = 1
		for element in temp
			temp_matrix[count, :] = element
			count = count + 1
		end
		temp_geomean = geomean.(eachcol(temp_matrix))
		push!(results_geomean, (key, temp_geomean[1], temp_geomean[2], temp_geomean[3]))
	end
	fileName = "geomean_results.csv"
	fullPath = string(dirrectoryName, "/", fileName)
	CSV.write(fullPath, results_geomean)

	paper_results = results_geomean

	return paper_results
end


function comutePairedTtest(dirrectoryName::String, all_instances_results::Dict{String, Vector{Any}}, paper_results::DataFrame)
	all_instances_results_CAT = all_instances_results[CAT_SOLVER]
	all_instances_results_newton = all_instances_results[NEWTON_TRUST_REGION_SOLVER]
	n = length(all_instances_results_CAT)

	all_runs_iterations = []
	all_runs_function = []
	all_runs_gradient = []
	for i in 1:n
		push!(all_runs_iterations, log(all_instances_results_newton[i][1]) - log(all_instances_results_CAT[i][1]))
		push!(all_runs_function, log(all_instances_results_newton[i][2]) - log(all_instances_results_CAT[i][2]))
		push!(all_runs_gradient, log(all_instances_results_newton[i][3]) - log(all_instances_results_CAT[i][3]))
	end

	mean_iterations = mean(all_runs_iterations)
	mean_function = mean(all_runs_function)
	mean_gradient = mean(all_runs_gradient)

	standard_deviation_iterations = std(all_runs_iterations)
	standard_deviation_function = std(all_runs_function)
	standard_deviation_gradient = std(all_runs_gradient)

	standard_error_iterations = standard_deviation_iterations / sqrt(n)
	standard_error_function = standard_deviation_function / sqrt(n)
	standard_error_gradient = standard_deviation_gradient / sqrt(n)

	ratio_iterations = mean_iterations / standard_error_iterations
	ratio_function = mean_function / standard_error_function
	ratio_gradient = mean_gradient / standard_error_gradient

	CI_iterations = (exp(mean_iterations - standard_error_iterations), exp(mean_iterations + standard_error_iterations))
	CI_function = (exp(mean_function - standard_error_function), exp(mean_function + standard_error_function))
	CI_gradient = (exp(mean_gradient - standard_error_gradient), exp(mean_gradient + standard_error_gradient))

	CI_df = DataFrame(ratio=[], lower=[], upper=[])
	push!(CI_df, ("iterations", CI_iterations[1], CI_iterations[2]))
	push!(CI_df, ("function_competitions", CI_function[1], CI_function[2]))
	push!(CI_df, ("gradient_competitions", CI_gradient[1], CI_gradient[2]))

	fileName = "confidence_interval_paired_test.csv"
	fullPath = string(dirrectoryName, "/", fileName)
	CSV.write(fullPath, CI_df)
	push!(paper_results, ("95 % CI for ratio", [CI_iterations[1], CI_iterations[2]], [CI_function[1], CI_function[2]], [CI_gradient[1], CI_gradient[2]]))
	return paper_results
end

function computeCI(dirrectoryName::String, all_instances_results::Dict{String, Vector{Any}})
	all_instances_results_CAT = all_instances_results[CAT_SOLVER]
	all_instances_results_newton = all_instances_results[NEWTON_TRUST_REGION_SOLVER]
	n = length(all_instances_results_CAT)

	all_runs_CAT_iterations = []
	all_runs_CAT_function = []
	all_runs_CAT_gradient = []
	for i in 1:n
		push!(all_runs_CAT_iterations, log(all_instances_results_CAT[i][1]))
		push!(all_runs_CAT_function,log(all_instances_results_CAT[i][2]))
		push!(all_runs_CAT_gradient, log(all_instances_results_CAT[i][3]))
	end

	all_runs_newton_iterations = []
	all_runs_newton_function = []
	all_runs_newton_gradient = []
	for i in 1:n
		push!(all_runs_newton_iterations, log(all_instances_results_newton[i][1]))
		push!(all_runs_newton_function, log(all_instances_results_newton[i][2]))
		push!(all_runs_newton_gradient, log(all_instances_results_newton[i][3]))
	end

	mean_CAT_iterations = mean(all_runs_CAT_iterations)
	mean_CAT_function = mean(all_runs_CAT_function)
	mean_CAT_gradient = mean(all_runs_CAT_gradient)

	standard_deviation_CAT_iterations = std(all_runs_CAT_iterations)
	standard_deviation_CAT_function = std(all_runs_CAT_function)
	standard_deviation_CAT_gradient = std(all_runs_CAT_gradient)

	standard_error_CAT_iterations = standard_deviation_CAT_iterations / sqrt(n)
	standard_error_CAT_function = standard_deviation_CAT_function / sqrt(n)
	standard_error_CAT_gradient = standard_deviation_CAT_gradient / sqrt(n)

	mean_newton_iterations = mean(all_runs_newton_iterations)
	mean_newton_function = mean(all_runs_newton_function)
	mean_newton_gradient = mean(all_runs_newton_gradient)

	standard_deviation_newton_iterations = std(all_runs_newton_iterations)
	standard_deviation_newton_function = std(all_runs_newton_function)
	standard_deviation_newton_gradient = std(all_runs_newton_gradient)

	standard_error_newton_iterations = standard_deviation_newton_iterations / sqrt(n)
	standard_error_newton_function = standard_deviation_newton_function / sqrt(n)
	standard_error_newton_gradient = standard_deviation_newton_gradient / sqrt(n)

	CI_CAT_iterations = (exp(mean_CAT_iterations - standard_error_CAT_iterations), exp(mean_CAT_iterations + standard_error_CAT_iterations))
	CI_CAT_function = (exp(mean_CAT_function - standard_error_CAT_function), exp(mean_CAT_function + standard_error_CAT_function))
	CI_falt_gradient = (exp(mean_CAT_gradient - standard_deviation_CAT_gradient), exp(mean_CAT_gradient + standard_deviation_CAT_gradient))

	CI_newton_iterations = (exp(mean_newton_iterations - standard_error_newton_iterations), exp(mean_newton_iterations + standard_error_newton_iterations))
	CI_newton_function = (exp(mean_newton_function - standard_error_newton_function), exp(mean_newton_function + standard_error_newton_function))
	CI_newton_gradient = (exp(mean_newton_gradient - standard_error_newton_gradient), exp(mean_newton_gradient + standard_error_newton_gradient))

	CI_geomean_CAT = DataFrame(criteria=[], lower=[], upper=[])
	push!(CI_geomean_CAT, ("iterations", CI_CAT_iterations[1], CI_CAT_iterations[2]))
	push!(CI_geomean_CAT, ("function_competitions", CI_CAT_function[1], CI_CAT_function[2]))
	push!(CI_geomean_CAT, ("gradient_competitions", CI_falt_gradient[1], CI_falt_gradient[2]))

	fileName = "confidence_interval_geomean_CAT.csv"
	fullPath = string(dirrectoryName, "/", fileName)
	CSV.write(fullPath, CI_geomean_CAT)

	CI_geomean_newton = DataFrame(criteria=[], lower=[], upper=[])
	push!(CI_geomean_newton, ("iterations", CI_newton_iterations[1], CI_newton_iterations[2]))
	push!(CI_geomean_newton, ("function_competitions", CI_newton_function[1], CI_newton_function[2]))
	push!(CI_geomean_newton, ("gradient_competitions", CI_newton_gradient[1], CI_newton_gradient[2]))

	fileName = "confidence_interval_geomean_newton.csv"
	fullPath = string(dirrectoryName, "/", fileName)
	CSV.write(fullPath, CI_geomean_newton)

	return CI_geomean_CAT, CI_geomean_newton
end

function solveLogisticRegressionMultipleTimes(
		folder_name::String,
		max_it::Int64, max_time::Float64,
		tol_opt::Float64,
		instances::Int64
	)
	@show VERSION
	all_instances_results = Dict(CAT_SOLVER => [], NEWTON_TRUST_REGION_SOLVER => [], ARC_SOLVER => [], TRU_FACTORIZATION => [], TRU_ITERATIVE => [])
	for i in 1:instances
		df = solveLogisticRegression(max_it, max_time, tol_opt, x, y)
		for key in keys(all_instances_results)
			temp = all_instances_results[key]
			temp_vector = filter(:solver => ==(key), df)
			push!(temp, [temp_vector[1, 2], temp_vector[1, 3], temp_vector[1, 4]])
			all_instances_results[key] = temp
        end
	end
	file_name_paper_results = "full_paper_results_geomean.txt"
	full_path_paper_results =  string(folder_name, "/", file_name_paper_results)
	paper_results = DataFrame(solver = [], itr = [], total_function_evaluation = [],  total_gradient_evaluation = [])
	saveResults(folder_name, all_instances_results)
	paper_results = computeGeometricMeans(folder_name, all_instances_results, paper_results)
	paper_results = comutePairedTtest(folder_name, all_instances_results, paper_results)
	computeCI(folder_name, all_instances_results)
	@show paper_results
end
