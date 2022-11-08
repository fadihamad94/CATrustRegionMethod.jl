using JuMP, NLPModels, NLPModelsJuMP, Random, Distributions, LinearAlgebra, Test, Optim, DataFrames, StatsBase, CSV
include("../src/CAT.jl")
include("../src/tru.jl")
include("../src/arc.jl")
Random.seed!(0)
using Ipopt
const CAT_SOLVER = "CAT"
const NEWTON_TRUST_REGION_SOLVER = "NewtonTrustRegion"
const ARC_SOLVER = "ARC_SOLVER"
const TRU_FACTORIZATION = "StandardTrustRegionMethodFactorization"
const TRU_ITERATIVE = "StandardTrustRegionMethodIterative"

function formulateMatrixCompletionProblem(M::Matrix, Ω::Matrix{Int64}, r::Int64, λ::Float64)
	# @show "Creating model"
	model = Model()
    m = size(M)[1]
	n = size(M)[2]

	temp_M = Ω .* M

	F = svd(temp_M)
    @variable(model, X[i=1:m, j=1:r], start = F.U[i, j])
	@variable(model, Y[i=1:n, j=1:r], start = F.Vt[i, j])

	@NLexpression(model, square_loss,  0.5 * (sum(sum(Ω[i, j] * (M[i, j] - (sum(X[i, k] * transpose(Y)[k, j] for k in 1:r))) ^ 2 for j in 1:n) for i in 1:m)))
	@NLexpression(model, frobeniusNorm_X, sum(sum(X[i, j] ^ 2 for j in 1:size(X)[2]) for i in 1:size(X)[1]))
	@NLexpression(model, frobeniusNorm_Y, sum(sum(Y[i, j] ^ 2 for j in 1:size(Y)[2]) for i in 1:size(Y)[1]))
	@NLobjective(model, Min, square_loss + λ * (frobeniusNorm_X + frobeniusNorm_Y))

	return model
end

function getMinMaxofMatrix(M::Matrix)
	minR = Inf
	maxR = -Inf
	for i in 1:size(M)[1]
		temp_min = minimum(M[i, :])
		temp_max = maximum(M[i, :])
		minR = min(minR, temp_min)
		maxR = max(maxR, temp_max)
	end
	return minR, maxR
end

function formulateMatrixCompletionProblem(M::Matrix, Ω::Matrix{Int64}, r::Int64, λ_1::Float64, λ_2::Float64)
	# @show "Creating model"
	model = Model()

	D = transpose(M)
    n_1 = size(D)[1]
	n_2 = size(D)[2]
	Ω = transpose(Ω)
	temp_min, temp_max = getMinMaxofMatrix(M)
	temp_D = Ω .* D
	μ = mean(temp_D)
	# F = svd(temp_D)
	# @show "Creating variables"
	# A = ones(n_1, r)
	# B = ones(n_2, r)
	A = rand(Uniform(temp_min, temp_max), n_1, r)
	B = rand(Uniform(temp_min, temp_max), n_2, r)
	@variable(model, P[i=1:n_1, j=1:r], start = A[i, j])
	@variable(model, Q[i=1:n_2, j=1:r], start = B[i, j])
	# @show size(F.U)
    # @variable(model, P[i=1:n_1, j=1:r], start = F.U[i, j])
	# @show size(F.Vt)
	# @show (n_1, n_2, r)
	# @variable(model, Q[i=1:n_2, j=1:r], start = F.Vt[i, j])
#=
	@show "Defining functions"
	computePQTranspose(i, j) = sum(P[i, k] * transpose(Q)[k, j] for k in 1:r)
	register(model, :computePQTranspose, 2, computePQTranspose; autodiff = true)

	computeObservDeviationRow(i) = (1 / n_2) * sum(computePQTranspose(i, j) for j in 1:n_2) - μ
	register(model, :computeObservDeviationRow, 1, computeObservDeviationRow; autodiff = true)

	computeObservDeviationColumn(j) = (1 / n_1) * sum(computePQTranspose(i, j) for i in 1:n_1) - μ
	register(model, :computeObservDeviationColumn, 1, computeObservDeviationColumn; autodiff = true)

	@show "Defining sum_observer_deviation_rows_squared expression"
	# @NLexpression(model, sum_observer_deviation_rows_squared, sum(computeObservDeviationRow(i) ^ 2 for i in 1:n_1))
	@NLexpression(model, sum_observer_deviation_rows_squared, sum(((1 / n_2) * sum(sum(P[i, k] * transpose(Q)[k, j] for k in 1:r) for j in 1:n_2) - μ) ^ 2 for i in 1:n_1))
	@show "Defining sum_observer_deviation_columns_squared expression"
	@NLexpression(model, sum_observer_deviation_columns_squared, sum(computeObservDeviationColumn(j) ^ 2 for j in 1:n_2))

	@show "Defining square_loss expression"
	@NLexpression(model, square_loss,  0.5 * (sum(sum(Ω[i, j] * (D[i, j] - μ - computeObservDeviationRow(i) - computeObservDeviationColumn(j) - computePQTranspose(i, j)) ^ 2 for j in 1:n_2) for i in 1:n_1)))



	@show "Defining objective function"
	@show sum_observer_deviation_rows_squared
	# @NLobjective(model, Min, square_loss + λ_1 * (sum_observer_deviation_rows_squared + sum_observer_deviation_columns_squared) + λ_2 * (frobeniusNorm_P + frobeniusNorm_Q))
	# @NLobjective(model, Min, frobeniusNorm_P + frobeniusNorm_Q)
=#
	# @show "Defining sum_observer_deviation_rows_squared expression"
	@NLexpression(model, sum_observer_deviation_rows_squared, sum(((1 / n_2) * sum(sum(P[i, k] * transpose(Q)[k, j] for k in 1:r) for j in 1:n_2) - μ) ^ 2 for i in 1:n_1))
	# @show "Defining sum_observer_deviation_columns_squared expression"
	@NLexpression(model, sum_observer_deviation_columns_squared, sum(((1 / n_1) * sum(sum(P[i, k] * transpose(Q)[k, j] for k in 1:r) for i in 1:n_1) - μ) ^ 2 for j in 1:n_2))

	# @show "Defining frobeniusNorm_P expression"
	@NLexpression(model, frobeniusNorm_P, sum(sum(P[i, j] ^ 2 for j in 1:r) for i in 1:n_1))
	# @show "Defining frobeniusNorm_Q expression"
	@NLexpression(model, frobeniusNorm_Q, sum(sum(Q[i, j] ^ 2 for j in 1:r) for i in 1:n_2))

	# @show "Defining square_loss expression"
	@NLexpression(model, square_loss,  0.5 * (sum(sum(Ω[i, j] * (D[i, j] - μ - ((1 / n_2) * sum(sum(P[i, k] * transpose(Q)[k, j] for k in 1:r) for j in 1:n_2) - μ) - ((1 / n_1) * sum(sum(P[i, k] * transpose(Q)[k, j] for k in 1:r) for i in 1:n_1) - μ) - sum(P[i, k] * transpose(Q)[k, j] for k in 1:r)) ^ 2 for j in 1:n_2) for i in 1:n_1)))

	# @show "Defining objective function"
	@NLobjective(model, Min, square_loss + λ_1 * (sum_observer_deviation_rows_squared + sum_observer_deviation_columns_squared) + λ_2 * (frobeniusNorm_P + frobeniusNorm_Q))
	return model
end

function formulateMatrixCompletionProblemTrial(M::Matrix, Ω::Matrix{Int64}, r::Int64, λ_1::Float64, λ_2::Float64)
	# @show "Creating model"
	model = Model()

	D = transpose(M)
    n_1 = size(D)[1]
	n_2 = size(D)[2]
	Ω = transpose(Ω)
	temp_min, temp_max = getMinMaxofMatrix(M)
	temp_D = Ω .* D
	μ = mean(temp_D)
	A = rand(Uniform(temp_min, temp_max), n_1, r)
	B = rand(Uniform(temp_min, temp_max), n_2, r)
	@variable(model, P[i=1:n_1, j=1:r], start = A[i, j])
	@variable(model, Q[i=1:n_2, j=1:r], start = B[i, j])

	@NLobjective(model, Min, sum(sum(Ω[i, j] *
	(D[i, j]
	- μ
	- ((1 / n_2) * sum(sum(P[i, k] * transpose(Q)[k, t] for k in 1:r) for t in 1:n_2) - μ)
	- ((1 / n_1) * sum(sum(P[t, k] * transpose(Q)[k, j] for k in 1:r) for t in 1:n_1) - μ)
	- sum(P[i, k] * transpose(Q)[k, j] for k in 1:r)) ^ 2
	+ Ω[i, j] * λ_1 * (((1 / n_2) * sum(sum(P[i, k] * transpose(Q)[k, t] for k in 1:r) for t in 1:n_2) - μ) ^ 2
	+ ((1 / n_1) * sum(sum(P[t, k] * transpose(Q)[k, j] for k in 1:r) for t in 1:n_1) - μ) ^ 2)
	+ Ω[i, j] * λ_2 * (sum(P[i, t] ^ 2 for t in 1:r)
	+ sum(Q[j, t] ^ 2 for t in 1:r))
	for j in 1:n_2)
	for i in 1:n_1))
	return model
end

function prepareData(directoryName::String, fileName::String, rows::Int64, columns::Int64)
	# @show "Reading file: $fileName"
	filePath = string(directoryName, "/", fileName)
	df = DataFrame(CSV.File(filePath))
	for i in 1:size(df)[1]
		for j in 1:size(df)[2]
			if typeof(df[i, j]) == Missing
				df[i, j] = 1.0
			end
		end
	end

	df = Missings.replace(df, 1.0)
	df = df[2:(2 + rows - 1),5:(5 + columns - 1)]
	M = Matrix(df)
	return transpose(M)
end

function prepareData(directoryName::String, fileName::String, rows::Int64, columns::Int64, i::Int64, j::Int64)
	@show "Reading file: $fileName"
	filePath = string(directoryName, "/", fileName)
	df = DataFrame(CSV.File(filePath))
	@show "Replacing missing values"
	for i in 1:size(df)[1]
		for j in 1:size(df)[2]
			if typeof(df[i, j]) == Missing
				df[i, j] = 1.0
			end
		end
	end

	df = Missings.replace(df, 1.0)
	@show "Creating Matrix M"
	df = df[2:size(df)[1], 5:size(df)[2]]
	df = df[1 + rows * (i - 1):rows * i,1 + columns * (j - 1):columns * j]
	M = Matrix(df)
	return M
end

function sampleData(M::Matrix)
	rows = size(M)[1]
	columns = size(M)[2]
	T = rows * columns
	Ω = rand(DiscreteUniform(0,1), rows, columns)
	return Ω
end

directoryName = "/Users/fah33/PhD_Research/CAT/benchmark/data"
# fileName = "electricity_data.csv"
fileName = "Aberdeen 66_11kV FY2021.csv"
# rows = 70
# columns = 50
# rows = 7 #m
# columns = 4 #n
# M = prepareData(directoryName, fileName, rows, columns)
# @show M
# Ω = sampleData(M)
# r = Int64(columns / 2)
# λ = 1.0

n_1 = 48
n_2 = 56
r = 12
# n_1 = 2
# n_2 = 2
# r = 2
# λ_1 = 0.001
# λ_2 = 0.001
# max_it = 10000
max_it = 1000
# max_time = 5 * 60.0 * 60.0
max_time = 1 * 60.0 * 60.0
tol_opt = 1e-5
# rows = n_2
# columns = n_1
# D = prepareData(directoryName, fileName, rows, columns, 1, 1)
# Ω = sampleData(D)
λ_1 = 0.001
λ_2 = 0.001
# model = formulateMatrixCompletionProblem(D, Ω, r, λ_1, λ_2)
# nlp = MathOptNLPModel(model)
# @show obj(nlp, nlp.meta.x0)
# @show norm(grad(nlp, nlp.meta.x0), 2)

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
	M::Matrix,
	Ω::Matrix{Int64},
	X::Matrix{Float64},
	Y::Matrix{Float64},
	m::Int64,
	n::Int64,
	r::Int64,
	λ::Float64
	)
	objective_function_nlp_model = obj(nlp, solution)
	# @show objective_function_nlp_model
	square_loss = 0.5 * (sum(sum(Ω[i, j] * (M[i, j] - (sum(X[i, k] * transpose(Y)[k, j] for k in 1:r))) ^ 2 for j in 1:n) for i in 1:m))
	frobeniusNorm_X = sum(sum(X[i, j] ^ 2 for j in 1:size(X)[2]) for i in 1:size(X)[1])
	frobeniusNorm_Y = sum(sum(Y[i, j] ^ 2 for j in 1:size(Y)[2]) for i in 1:size(Y)[1])
	objective_function_formulation = square_loss + λ * (frobeniusNorm_X + frobeniusNorm_Y)
    @test norm(objective_function_nlp_model - objective_function_formulation, 2) <= 1e-8
end

# function solveMatricCompletion(
# 	max_it::Int64,
# 	max_time::Float64,
# 	tol_opt::Float64,
# 	M::Matrix,
# 	Ω::Matrix{Int64},
# 	r::Int64,
# 	λ::Float64
# 	)
function solveMatricCompletion(
	max_it::Int64,
	max_time::Float64,
	tol_opt::Float64,
	M::Matrix,
	Ω::Matrix{Int64},
	r::Int64,
	λ_1::Float64,
	λ_2::Float64
	)
	m = size(M)[1]
	n = size(M)[2]
	all_results = DataFrame(solver = [], itr = [], total_function_evaluation = [],  total_gradient_evaluation = [])
    @time begin
		model = formulateMatrixCompletionProblem(M, Ω, r, λ_1, λ_2)
		# model = formulateMatrixCompletionProblemTrial(M, Ω, r, λ_1, λ_2)
	end
    global nlp = MathOptNLPModel(model)
    x0 = nlp.meta.x0
	GRADIENT_TOLERANCE = tol_opt
	ITERATION_LIMIT = max_it

	println("------------------Solving Using CAT-----------------")
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.1, 0.1, 8.0, 1.0, ITERATION_LIMIT, GRADIENT_TOLERANCE)
    δ = 0.0
	@time begin
		solution, status, iteration_stats, computation_stats, itr = consistently_adaptive_trust_region_method.CAT(problem, x0, δ)
	end
    X = reshape(solution[1:(m * r)], m, r)
	Y = reshape(solution[(m * r) + 1:end], n, r)
	# validateResults(nlp, solution, M, Ω, X, Y, m, n, r, λ)
	# square_loss = 0.5 * (sum(sum((1 - Ω[i, j]) * (M[i, j] - (sum(X[i, k] * transpose(Y)[k, j] for k in 1:r))) ^ 2 for j in 1:n) for i in 1:m))
	# @show square_loss
	push!(all_results, (CAT_SOLVER, itr, computation_stats["total_function_evaluation"], computation_stats["total_gradient_evaluation"]))

	println("------------------Solving Using NewtonTrustRegion------------------")
	# max_time = 60.0 * 5
	d_ = Optim.TwiceDifferentiable(f, g!, h!, nlp.meta.x0)
	@time begin
		results = optimize(d_, nlp.meta.x0, Optim.NewtonTrustRegion(), Optim.Options(show_trace=false, iterations = ITERATION_LIMIT, f_calls_limit = ITERATION_LIMIT, g_abstol = GRADIENT_TOLERANCE, time_limit = max_time))
	end
	if  !Optim.converged(results)
		@show Optim.iterations(results)
		@show Optim.f_calls(results)
		@show Optim.g_calls(results)
		push!(all_results, (NEWTON_TRUST_REGION_SOLVER, ITERATION_LIMIT, ITERATION_LIMIT, ITERATION_LIMIT))
	else
		solution = Optim.minimizer(results)
		# @show norm(solution, 2)
		X = reshape(solution[1:(m * r)], m, r)
		Y = reshape(solution[(m * r) + 1:end], n, r)
		# validateResults(nlp, solution, M, Ω, X, Y, m, n, r, λ)
		@show Optim.iterations(results)
		# square_loss = 0.5 * (sum(sum((1 - Ω[i, j]) * (M[i, j] - (sum(X[i, k] * transpose(Y)[k, j] for k in 1:r))) ^ 2 for j in 1:n) for i in 1:m))
		# @show square_loss
		push!(all_results, (NEWTON_TRUST_REGION_SOLVER, Optim.iterations(results), Optim.f_calls(results), Optim.g_calls(results)))
	end

	return all_results
end

# solveMatricCompletion(max_it, max_time, tol_opt, M, Ω, r, λ)

function getData(directoryName::String, fileName::String, rows::Int64, columns::Int64)
	M = prepareData(directoryName, fileName, rows, columns)
	Ω = sampleData(M)
	return M, Ω
end

function getData(directoryName::String, fileName::String, rows::Int64, columns::Int64, i::Int64, j::Int64)
	M = prepareData(directoryName, fileName, rows, columns, i, j)
	Ω = sampleData(M)
	return M, Ω
end

function filterRows(solver::String, iterations_vector::Vector{Int64})
    return filter!(x->x == solver, iterations_vector)
end

equals_method(name::String, solver::String) = name == solver

function saveResults(dirrectoryName::String, all_instances_results::Dict{String, Vector{Any}})
	fileNameCAT = "all_instances_results_CAT.csv"
	fileNameNewton = "all_instances_results_newton.csv"
	fullPathCAT = string(dirrectoryName, "/", fileNameCAT)
	fullPathNewton = string(dirrectoryName, "/", fileNameNewton)

	all_instances_results_CAT = all_instances_results[CAT_SOLVER]
	all_instances_results_newton = all_instances_results[NEWTON_TRUST_REGION_SOLVER]

	df_CAT = DataFrame(iter=[], fct=[], gradient=[])
	for i in 1:length(all_instances_results_CAT)
		push!(df_CAT, (all_instances_results_CAT[i][1], all_instances_results_CAT[i][2], all_instances_results_CAT[i][3]))
	end

	df_newton = DataFrame(iter=[], fct=[], gradient=[])
	for i in 1:length(all_instances_results_newton)
		push!(df_newton, (all_instances_results_newton[i][1], all_instances_results_newton[i][2], all_instances_results_newton[i][3]))
	end

	CSV.write(fullPathCAT, df_CAT)
	CSV.write(fullPathNewton, df_newton)

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

function solveMatrixCompletionMultipleTimes(
		folder_name::String,
		max_it::Int64, max_time::Float64,
		tol_opt::Float64
	)
	@show VERSION
	all_instances_results = Dict(CAT_SOLVER => [], NEWTON_TRUST_REGION_SOLVER => [])
	fileNames = readdir(string(folder_name, "/", "data"); join=false)
	λ = 1.0
	# rows = 70
	# columns = 50
	# rows = 60
	# columns = 48
	# r = Int64(columns / 2)

	count = 1
	# r = 24
	# rows = 75
	# columns = 48
	rows = 30
	columns = 24
	# rows = 15
	# columns = 12
	r = Int64(columns / 2)
	total_rows = 300
	total_columns = 96
	rows = 96
	columns = 96
	# total_i = Int64(total_rows / rows)
	# total_j = Int64(total_columns / columns)
	total_i = 1
	total_j = 1
	#columns 48
	#Need to transpose the matrix am constructing to match their problem
	#Paper :  Matrix n_1 * n_2 where n_1 = 48 and n_2 = 56 and r = 12
	#n_1 or rows is number of days
	#n_2 or columns is number of time intervals
	# rows = 48
	# columns = 56
	# r = 12

	rows = 48
	columns = 56
	r = 12
	fileNames = ["Adamstown 132_11kV FY2021.csv"]
	# fileNames = ["Aberdeen 66_11kV FY2021.csv"]
	for fileName in fileNames
		# rows = 16
		# columns = 18
		# r = 4
		# rows = 24
	    # columns = 28
	    # r = 6
		rows = 36
		columns = 42
		r = 9
		i = 1
		j = 1
		# @show fileName
		rows = 30
		columns = 48
		r = 9

		D, Ω = getData(string(folder_name, "/", "data"), fileName, rows, columns, i, j)
		for k in 1:10
			try
				df = solveMatricCompletion(max_it, max_time, tol_opt, D, Ω, r, λ_1, λ_2)
				for key in keys(all_instances_results)
					temp = all_instances_results[key]
					temp_vector = filter(:solver => ==(key), df)
					push!(temp, [temp_vector[1, 2], temp_vector[1, 3], temp_vector[1, 4]])
					all_instances_results[key] = temp
				end
				Ω = sampleData(D)
			catch e
				@show e
			end
		end
		# for i in 1:total_i
		# 	for j in 1:total_j
		# 		try
		# 			@time begin
		# 				D, Ω = getData(string(folder_name, "/", "data"), fileName, rows, columns, i, j)
		# 			end
		# 			df = solveMatricCompletion(max_it, max_time, tol_opt, D, Ω, r, λ_1, λ_2)
		# 			for key in keys(all_instances_results)
		# 				temp = all_instances_results[key]
		# 				temp_vector = filter(:solver => ==(key), df)
		# 				push!(temp, [temp_vector[1, 2], temp_vector[1, 3], temp_vector[1, 4]])
		# 				all_instances_results[key] = temp
		# 	        end
		# 			count = count + 1
		# 		catch e
		# 			#do nothing just to complete the run
		# 			@show e
		# 		end
		# 	end
		# end
	end

	# for fileName in fileNames
	# 	rows = 24
	#     columns = 28
	#     r = 6
	# 	@show fileName
	# 	if count > 1
	# 		break
	# 	end
	# 	count = count + 1
	# 	for i in 1:total_i
	# 		for j in 1:total_j
	# 			try
	# 				@time begin
	# 					D, Ω = getData(string(folder_name, "/", "data"), fileName, rows, columns, i, j)
	# 				end
	# 				df = solveMatricCompletion(max_it, max_time, tol_opt, D, Ω, r, λ_1, λ_2)
	# 				for key in keys(all_instances_results)
	# 					temp = all_instances_results[key]
	# 					temp_vector = filter(:solver => ==(key), df)
	# 					push!(temp, [temp_vector[1, 2], temp_vector[1, 3], temp_vector[1, 4]])
	# 					all_instances_results[key] = temp
	# 		        end
	# 				count = count + 1
	# 			catch e
	# 				#do nothing just to complete the run
	# 				@show e
	# 			end
	# 		end
	# 	end
	# end

	# #TRial 1
	# rows = 60
	# columns = 48
	# r = 24
	# # for i in 1:6
	# rows = 75
	# columns = 48
	# for i in 1:4
	# 	for j in 1:2
	# 		fileName = "Aberdeen 66_11kV FY2021.csv"
	# 		M, Ω = getData(string(folder_name, "/", "data"), fileName, rows, columns, i, j)
	# 		df = solveMatricCompletion(max_it, max_time, tol_opt, M, Ω, r, λ)
	# 		for key in keys(all_instances_results)
	# 			temp = all_instances_results[key]
	# 			temp_vector = filter(:solver => ==(key), df)
	# 			push!(temp, [temp_vector[1, 2], temp_vector[1, 3], temp_vector[1, 4]])
	# 			all_instances_results[key] = temp
	#         end
	# 	end
	# end
	# for fileName in fileNames
	# 	@show fileName
	# 	if count > 5
	# 		break
	# 	end
	# 	M, Ω = getData(string(folder_name, "/", "data"), fileName, rows, columns)
	# 	df = solveMatricCompletion(max_it, max_time, tol_opt, M, Ω, r, λ)
	# 	for key in keys(all_instances_results)
	# 		temp = all_instances_results[key]
	# 		temp_vector = filter(:solver => ==(key), df)
	# 		push!(temp, [temp_vector[1, 2], temp_vector[1, 3], temp_vector[1, 4]])
	# 		all_instances_results[key] = temp
    #     end
	# 	count = count + 1
	# end
	file_name_paper_results = "full_paper_results_geomean.txt"
	full_path_paper_results =  string(folder_name, "/", file_name_paper_results)
	paper_results = DataFrame(solver = [], itr = [], total_function_evaluation = [],  total_gradient_evaluation = [])
	saveResults(folder_name, all_instances_results)
	paper_results = computeGeometricMeans(folder_name, all_instances_results, paper_results)
	paper_results = comutePairedTtest(folder_name, all_instances_results, paper_results)
	computeCI(folder_name, all_instances_results)
	@show paper_results
end

folder_name = "/Users/fah33/PhD_Research/CAT/benchmark"
solveMatrixCompletionMultipleTimes(
		folder_name,
		max_it,
		max_time,
		tol_opt
	)

#Results
#=
Adamstown 132_11kV FY2021.csv
10 instances
rows = 24
columns = 28
r = 6

Row │ solver             itr                 total_function_evaluation  total_gradient_evaluation
	│ Any                Any                 Any                        Any
─────┼─────────────────────────────────────────────────────────────────────────────────────────────
  1 │ CAT               65.0174             66.0367                    66.0367
  2 │ NewtonTrustRegion  3856.72             3869.85                    3869.85
  3 │ 95 % CI for ratio  [31.9986, 109.963]  [31.673, 108.425]          [31.673, 108.425]

  10 instances
  rows = 16
  columns = 18
  r = 4

  Row │ solver             itr               total_function_evaluation  total_gradient_evaluation
      │ Any                Any               Any                        Any
 ─────┼───────────────────────────────────────────────────────────────────────────────────────────
    1 │ CAT               142.353           143.357                    143.357
    2 │ NewtonTrustRegion  10000.0           10000.0                    10000.0
    3 │ 95 % CI for ratio  [68.143, 72.418]  [67.6802, 71.8948]         [67.6802, 71.8948]


	rows = 30
	columns = 48
	r = 9

	Row │ solver             itr                 total_function_evaluation  total_gradient_evaluation
	     │ Any                Any                 Any                        Any
	─────┼─────────────────────────────────────────────────────────────────────────────────────────────
	   1 │ CAT               218.182             219.185                    219.185
	   2 │ NewtonTrustRegion  1000.0              1000.0                     1000.0
	   3 │ 95 % CI for ratio  [4.48189, 4.68704]  [4.46185, 4.66513]         [4.46185, 4.66513]
 =#
