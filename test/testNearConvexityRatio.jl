using JuMP, NLPModels, NLPModelsJuMP, LinearAlgebra, DataFrames
include("../src/CAT.jl")

polynomial_exponential_1_d = "polynomial_exponential_1_d"
polynomial_exponential_2_d = "polynomial_exponential_2_d"
polynomial_n_d             = "polynomial_n_d"
exponential_n_d            = "exponential_n_d"
polynomial_exponential_n_d = "polynomial_exponential_n_d"

β = 0.1
θ = 0.1
ω = 8.0
r_1 = 1.0
max_it = 10000
tol_opt = 1e-5
max_time = 30 * 60.0
γ_2 = 0.8
δ = 0.0

trust_region_method_subproblem_solver = "GALAHAD_TRS"
optimization_method = "CAT_GALAHAD_FACTORIZATION"

directory_name = "/afs/pitt.edu/home/f/a/fah33/CAT/test/nearConvexity"

n = 5

function problem_polynomial_exponential_1_d()
	model = Model()
	@variable(model, x)
	@NLobjective(model, Min, exp(x) + x ^ 4 + 2 * x + (x ^ 2  + 3 * x - 2) ^ 2)
	return model
end

function problem_polynomial_exponential_2_d()
	model = Model()
	@variable(model, x)
	@variable(model, y)
	@NLobjective(model, Min, exp(x + y) + x ^ 4 + (y - 2) ^ 2 + x * y ^ 2)
	return model
end

function problem_polynomial_n_d(n::Int64)
	model = Model()
	@variable(model, x[i=1:n])
	@NLobjective(model, Min, sum((x[i] + i * (-1) ^ i) ^ (2 * i) for i in 1:n))
	return model
end

function problem_exponential_n_d(n::Int64)
	model = Model()
	@variable(model, x[i=1:n])
	@NLobjective(model, Min, sum(exp(x[i] ^ 2) for i in 1:n))
	return model
end

function problem_polynomial_exponential_n_d(n::Int64)
	model = Model()
	@variable(model, x[i=1:n])
	@NLobjective(model, Min, sum(exp(x[i] ^ (2 * i)) + (x[i] + i * (-1) ^ i) ^ (2 * i) for i in 1:n))
	return model
end

function optimizeAndValidateConvexity_problem_polynomial_exponential_1_d()
	model = problem_polynomial_exponential_1_d()
	global nlp = MathOptNLPModel(model)
	problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, β, θ, ω, r_1, max_it, tol_opt, max_time, γ_2)
	x_1 = [-50.0]
	x, status, iteration_stats, computation_stats, total_iterations_count = consistently_adaptive_trust_region_method.CAT(problem, x_1, δ, trust_region_method_subproblem_solver)
	outputHowOftenNearConvexityConditionHolds(string(directory_name, "/$optimization_method"),
		polynomial_exponential_1_d,
		status,
		optimization_method,
		iteration_stats
		)
end

function optimizeAndValidateConvexity_problem_polynomial_exponential_2_d()
	model = problem_polynomial_exponential_2_d()
	global nlp = MathOptNLPModel(model)
	problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, β, θ, ω, r_1, max_it, tol_opt, max_time, γ_2)
	x_1 = [-6.0, 600.0]
	x, status, iteration_stats, computation_stats, total_iterations_count = consistently_adaptive_trust_region_method.CAT(problem, x_1, δ, trust_region_method_subproblem_solver)
	outputHowOftenNearConvexityConditionHolds(string(directory_name, "/$optimization_method"),
		polynomial_exponential_2_d,
		status,
		optimization_method,
		iteration_stats
		)
end

function optimizeAndValidateConvexity_problem_polynomial_n_d(n::Int64)
	model = problem_polynomial_n_d(n)
	global nlp = MathOptNLPModel(model)
	problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, β, θ, ω, r_1, max_it, tol_opt, max_time, γ_2)
	x_1 = zeros(n)
	x, status, iteration_stats, computation_stats, total_iterations_count = consistently_adaptive_trust_region_method.CAT(problem, x_1, δ, trust_region_method_subproblem_solver)
	outputHowOftenNearConvexityConditionHolds(string(directory_name, "/$optimization_method"),
		polynomial_exponential_2_d,
		status,
		optimization_method,
		iteration_stats
		)
end

function optimizeAndValidateConvexity_problem_exponential_n_d(n::Int64)
	model = problem_exponential_n_d(n)
	global nlp = MathOptNLPModel(model)
	problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, β, θ, ω, r_1, max_it, tol_opt, max_time, γ_2)
	x_1 = ones(n)
	x, status, iteration_stats, computation_stats, total_iterations_count = consistently_adaptive_trust_region_method.CAT(problem, x_1, δ, trust_region_method_subproblem_solver)
	outputHowOftenNearConvexityConditionHolds(string(directory_name, "/$optimization_method"),
		polynomial_exponential_2_d,
		status,
		optimization_method,
		iteration_stats
		)
end

function optimizeAndValidateConvexity_problem_polynomial_exponential_n_d(n::Int64)
	model = problem_polynomial_exponential_n_d(n)
	global nlp = MathOptNLPModel(model)
	problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, β, θ, ω, r_1, max_it, tol_opt, max_time, γ_2)
	x_1 = zeros(n)
	x, status, iteration_stats, computation_stats, total_iterations_count = consistently_adaptive_trust_region_method.CAT(problem, x_1, δ, trust_region_method_subproblem_solver)
	outputHowOftenNearConvexityConditionHolds(string(directory_name, "/$optimization_method"),
		polynomial_exponential_2_d,
		status,
		optimization_method,
		iteration_stats
		)
end

function outputHowOftenNearConvexityConditionHolds(directory_name::String,
	cutest_problem::String,
	status::String,
	optimization_method::String,
	iteration_stats::DataFrame
	)
	numberOfRow = size(iteration_stats)[1]
	numberOfCol = size(iteration_stats)[2]
	count = 0
	θ = 0.5
	for k in 1:numberOfRow
		f_T = last(iteration_stats, 1)[!, "fval"][1]
		g_T = last(iteration_stats, 1)[!, "gradval"][1]
		x_T = last(iteration_stats, 1)[!, "x"][1]

		f_k = iteration_stats[k,"fval"]
		x_k = iteration_stats[k,"x"]
		g_k = grad(nlp, x_k)
		if f_T <= f_k + θ * transpose(x_T - x_k) * g_k
			count = count + 1
		end
	end
	rate = 0.0
	if numberOfRow > 0
		rate = count / numberOfRow
	end
	near_convexity_rate_csv_file_name = "table_near_convexity_rate_$optimization_method.csv"
	near_convexity_rate_csv_file_path = string(directory_name, "/", near_convexity_rate_csv_file_name)
	open(near_convexity_rate_csv_file_path,"a") do near_convexity_rate_csv_file
		write(near_convexity_rate_csv_file, "$cutest_problem,$status,$rate\n")
    end
end


function main()

	total_results_output_directory =  string(directory_name, "/$optimization_method")
	near_convexity_rate_csv_file_name = "table_near_convexity_rate_$optimization_method.csv"
	near_convexity_rate_csv_file_path = string(total_results_output_directory, "/", near_convexity_rate_csv_file_name)
	rm(near_convexity_rate_csv_file_path, force=true)
    mkpath(total_results_output_directory);

	open(near_convexity_rate_csv_file_path,"a") do near_convexity_rate_csv_file
		write(near_convexity_rate_csv_file, "problem_name,status,rate\n");
	end

	optimizeAndValidateConvexity_problem_polynomial_exponential_1_d()
	optimizeAndValidateConvexity_problem_polynomial_exponential_2_d()
	optimizeAndValidateConvexity_problem_polynomial_n_d(n)
	optimizeAndValidateConvexity_problem_exponential_n_d(n)
	optimizeAndValidateConvexity_problem_polynomial_exponential_n_d(n)
end


# main()
