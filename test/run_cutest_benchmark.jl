using JuMP, NLPModels, NLPModelsJuMP, LinearAlgebra, Optim, CUTEst, CSV, Test, DataFrames, SparseArrays, StatsBase, Random
include("../src/CAT.jl")
include("../src/tru.jl")
include("../src/arc.jl")

const problems_paper_list =  ["ALLINITU", "ARGLINA", "BARD", "BEALE", "BIGGS6", "BOX3", "BRKMCC", "BROWNAL", "BROWNBS", "BROWNDEN", "CHNROSNB", "CLIFF", "CUBE", "DENSCHNA", "DENSCHNB", "DENSCHNC", "DENSCHND", "DENSCHNE", "DENSCHNF", "DJTL", "ENGVAL2", "ERRINROS", "EXPFIT", "GENROSEB", "GROWTHLS", "GULF", "HAIRY", "HATFLDD", "HATFLDE", "HEART6LS", "HEART8LS", "HELIX", "HIMMELBB", "HUMPS", "HYDC20LS", "JENSMP", "KOWOSB", "LOGHAIRY", "MANCINO", "MEXHAT", "MEYER3", "OSBORNEA", "OSBORNEB", "PALMER5C", "PALMER6C", "PALMER7C", "PALMER8C", "PARKCH", "PENALTY2", "PENALTY3", "PFIT1LS", "PFIT2LS", "PFIT3LS", "PFIT4LS", "ROSENBR", "S308", "SENSORS", "SINEVAL", "SISSER", "SNAIL", "STREG", "TOINTGOR", "TOINTPSP", "VARDIM", "VIBRBEAM", "WATSON", "YFITU"]
#const problems_paper_list =  ["ALLINITU", "ARGLINA"]
const optimization_method_CAT = "CAT"
const optimization_method_CAT_theta_0 = "CAT_THETA_ZERO"
const optimization_metnod_newton_trust_region = "NewtonTrustRegion"

const optimization_method_CAT_galahad_factorization = "CAT_GALAHAD_FACTORIZATION"
const optimization_method_CAT_galahad_iterative = "CAT_GALAHAD_ITERATIVE"
const optimization_method_arc_galahad = "ARC"
const optimization_method_tru_galahd_factorization = "TRU_GALAHAD_FACTORIZATION"
const optimization_method_tru_galahd_iterative = "TRU_GALAHAD_ITERATIVE"

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

function get_problem_list(min_nvar, max_nvar)
	return CUTEst.select(min_var = min_nvar, max_var = max_nvar, max_con = 0, only_free_var = true)
end


function run_cutest_with_CAT(
    folder_name::String,
    default_problems::Bool,
    max_it::Int64,
    max_time::Float64,
    tol_opt::Float64,
    θ::Float64,
    β::Float64,
	ω::Float64,
    γ_2::Float64,
    r_1::Float64,
	δ::Float64,
    min_nvar::Int64,
    max_nvar::Int64,
	optimization_method::String
    )
    cutest_problems = problems_paper_list
    if !default_problems
        cutest_problems = get_problem_list(min_nvar, max_nvar)
    end
	trust_region_method_subproblem_solver = optimization_method == optimization_method_CAT ? consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_DEFAULT : (optimization_method == optimization_method_CAT_galahad_factorization ? consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_TRS : consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_GLTR)
	if θ == 0.0
		optimization_method = optimization_method_CAT_theta_0
	end
    train_size = 0.8
    number_of_problems = length(cutest_problems)
    #=
    train_cutest_problems = cutest_problems[1:Int(round(0.8 * number_of_problems)) + 1]
    executeCUTEST_Models_benchmark(train_cutest_problems, string(folder_name, "_train"), optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1, δ, trust_region_method_subproblem_solver)
    test_cutest_problems = cutest_problems[Int(round(0.8 * number_of_problems)) + 1 : number_of_problems] 
    executeCUTEST_Models_benchmark(test_cutest_problems, string(folder_name, "_test"), optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1, δ, trust_region_method_subproblem_solver)
    =#

    #This code is for trying some syntax and logic
    #=
    Random.seed!(0)
    cutest_problem_indixes = collect(1:number_of_problems)
    cutest_problem_indixes = shuffle(cutest_problem_indixes)
    train_cutest_problems = cutest_problems[cutest_problem_indixes[1:floor(Int, 0.8 * number_of_problems) + 1]]
    executeCUTEST_Models_benchmark(train_cutest_problems, string(folder_name, "_train"), optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1, δ, trust_region_method_subproblem_solver)
    test_cutest_problems = cutest_problems[cutest_problem_indixes[Int(round(0.8 * number_of_problems)) + 1 : number_of_problems]]
    executeCUTEST_Models_benchmark(test_cutest_problems, string(folder_name, "_test"), optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1, δ, trust_region_method_subproblem_solver)
    =#

    Random.seed!(0)
    cutest_problem_indixes = collect(1:number_of_problems)
    cutest_problem_indixes = shuffle(cutest_problem_indixes)
    train_cutest_problems = cutest_problems[cutest_problem_indixes[1:floor(Int, 0.8 * number_of_problems) + 1]]
    train_cutest_problems = cutest_problems
    executeCUTEST_Models_benchmark(train_cutest_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1, δ, trust_region_method_subproblem_solver)
    #test_cutest_problems = cutest_problems[cutest_problem_indixes[Int(round(0.8 * number_of_problems)) + 1 : number_of_problems]]
    #executeCUTEST_Models_benchmark(test_cutest_problems, string(folder_name, "_test"), optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1, δ, trust_region_method_subproblem_solver)
end

#OLD_CODE
#=
function run_cutest_with_CAT(
    folder_name::String,
    default_problems::Bool,
    max_it::Int64,
    max_time::Float64,
    tol_opt::Float64,
    θ::Float64,
    β::Float64,
	ω::Float64,
    γ_2::Float64,
    r_1::Float64,
	δ::Float64,
    min_nvar::Int64,
    max_nvar::Int64,
	optimization_method::String
    )
    cutest_problems = problems_paper_list
    if !default_problems
        cutest_problems = get_problem_list(min_nvar, max_nvar)
    end
	trust_region_method_subproblem_solver = optimization_method == optimization_method_CAT ? consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_DEFAULT : (optimization_method == optimization_method_CAT_galahad_factorization ? consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_TRS : consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_GLTR)
	if θ == 0.0
		optimization_method = optimization_method_CAT_theta_0
	end
	executeCUTEST_Models_benchmark(cutest_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1, δ, trust_region_method_subproblem_solver)
end
=#

function run_cutest_with_newton_trust_region(
    folder_name::String,
    default_problems::Bool,
    max_it::Int64,
    max_time::Float64,
    tol_opt::Float64,
    r_1::Float64,
    min_nvar::Int64,
    max_nvar::Int64
    )
    cutest_problems = problems_paper_list
    if !default_problems
        cutest_problems = get_problem_list(min_nvar, max_nvar)
    end
    optimization_method = optimization_metnod_newton_trust_region
	θ = β = ω = γ_2 = 0.0
	executeCUTEST_Models_benchmark(cutest_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1)
end

function run_cutest_with_arc(
    folder_name::String,
    default_problems::Bool,
    max_it::Int64,
    max_time::Float64,
    tol_opt::Float64,
    σ_1::Float64,
    min_nvar::Int64,
    max_nvar::Int64
    )
	cutest_problems = problems_paper_list
    if !default_problems
        cutest_problems = get_problem_list(min_nvar, max_nvar)
    end
    optimization_method = optimization_method_arc_galahad
	θ = β = ω = γ_2 = 0.0
	executeCUTEST_Models_benchmark(cutest_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, σ_1)
end

function run_cutest_with_tru(
    folder_name::String,
    default_problems::Bool,
    max_it::Int64,
    max_time::Float64,
    tol_opt::Float64,
    r_1::Float64,
    min_nvar::Int64,
    max_nvar::Int64,
	optimization_method::String
    )
	cutest_problems = problems_paper_list
    if !default_problems
        cutest_problems = get_problem_list(min_nvar, max_nvar)
    end
	θ = β = ω = γ_2 = 0.0
	executeCUTEST_Models_benchmark(cutest_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1)
end

function runModelFromProblem(
	cutest_problem::String,
	folder_name::String,
	optimization_method::String,
	max_it::Int64,
    max_time::Float64,
    tol_opt::Float64,
    θ::Float64,
    β::Float64,
	ω::Float64,
    γ_2::Float64,
    r_1::Float64,
	δ::Float64,
	trust_region_method_subproblem_solver::String=consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_DEFAULT
	)
    global nlp = nothing
    try
        println("-----------EXECUTING PROBLEM----------", cutest_problem)
        nlp = CUTEstModel(cutest_problem)
		if optimization_method == optimization_method_CAT || optimization_method == optimization_method_CAT_theta_0 || optimization_method == optimization_method_CAT_galahad_factorization || optimization_method == optimization_method_CAT_galahad_iterative
			problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, β, θ, ω, r_1, max_it, tol_opt, max_time, γ_2)
	        x_1 = problem.nlp.meta.x0
	        x, status, iteration_stats, computation_stats, total_iterations_count = consistently_adaptive_trust_region_method.CAT(problem, x_1, δ, trust_region_method_subproblem_solver)
			function_value = NaN
			gradient_value = NaN
			if size(last(iteration_stats, 1))[1] > 0
				function_value = last(iteration_stats, 1)[!, "fval"][1]
			    gradient_value = last(iteration_stats, 1)[!, "gradval"][1]
			end
			computation_stats_modified = Dict("function_value" => function_value, "gradient_value" => gradient_value)
			for key in keys(computation_stats)
				computation_stats_modified[key] = computation_stats[key]
			end
			println("------------------------MODEL SOLVED WITH STATUS: ", status)
			directory_name = string(folder_name, "/", "$optimization_method")
			outputResultsToCSVFile(directory_name, cutest_problem, iteration_stats)
			total_number_factorizations = Int64(computation_stats_modified["total_number_factorizations"])
			outputIterationsStatusToCSVFile(directory_name, cutest_problem, status, computation_stats_modified, total_iterations_count, optimization_method, total_number_factorizations)
			# outputHowOftenNearConvexityConditionHolds(directory_name, cutest_problem, status, optimization_method, iteration_stats)
		elseif optimization_method == optimization_metnod_newton_trust_region
			d = Optim.TwiceDifferentiable(f, g!, h!, nlp.meta.x0)
			results = optimize(d, nlp.meta.x0, Optim.NewtonTrustRegion(initial_delta=r_1), Optim.Options(show_trace=false, iterations = max_it, time_limit = max_time, g_abstol = tol_opt))
			x = Optim.minimizer(results)
			total_iterations_count = Optim.iterations(results)
			total_function_evaluation = Optim.f_calls(results)
			total_gradient_evaluation = Optim.g_calls(results)
			total_hessian_evaluation = Optim.h_calls(results)
			function_value = obj(nlp, x)
			graient_value = norm(grad(nlp, x), 2)
			computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "function_value" => function_value, "gradient_value" => graient_value)
			status = results.g_converged ? "OPTIMAL" : (Optim.iteration_limit_reached(results) ? "ITERATION_LIMIT" : "FAILURE")
			if status == "ITERATION_LIMIT"
				total_iterations_count = max_it + 1
			end
			println("------------------------MODEL SOLVED WITH STATUS: ", status)
			directory_name = string(folder_name, "/", "$optimization_method")
			outputIterationsStatusToCSVFile(directory_name, cutest_problem, status, computation_stats, total_iterations_count, optimization_method)
		elseif optimization_method == optimization_method_arc_galahad
			initial_weight = r_1
			print_level = 0
			userdata, solution = arc(length(nlp.meta.x0), nlp.meta.x0, grad(nlp, nlp.meta.x0), print_level, max_it, initial_weight)
			status = userdata.status == 0 ? "OPTIMAL" : (userdata.status == -18 ? "ITERATION_LIMIT" : "FAILURE")
			iter = userdata.iter
			total_iterations_count = iter
			total_function_evaluation = userdata.total_function_evaluation
			total_gradient_evaluation = userdata.total_gradient_evaluation
			total_hessian_evaluation = userdata.total_hessian_evaluation
			function_value = obj(nlp, solution)
			graient_value = norm(grad(nlp, solution), 2)
			if status != 0 || graient_value > tol_opt
				iter = max_it + 1
				total_function_evaluation = max_it + 1
				total_gradient_evaluation = max_it + 1
			end
			computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "function_value" => function_value, "gradient_value" => graient_value)
			println("------------------------MODEL SOLVED WITH STATUS: ", status)
			directory_name = string(folder_name, "/", "$optimization_method")
			outputIterationsStatusToCSVFile(directory_name, cutest_problem, status, computation_stats, total_iterations_count, optimization_method)
		elseif optimization_method == optimization_method_tru_galahd_factorization || optimization_method == optimization_method_tru_galahd_iterative
			subproblem_direct = optimization_method == optimization_method_tru_galahd_factorization ? true : false
			initial_x = nlp.meta.x0
			print_level = 0
			userdata, solution = tru(length(initial_x), initial_x, grad(nlp, initial_x), print_level, max_it, r_1, subproblem_direct)
			status = userdata.status == 0 ? "OPTIMAL" : (userdata.status == -18 ? "ITERATION_LIMIT" : "FAILURE")
			iter = userdata.iter
			total_iterations_count = iter
			total_function_evaluation = userdata.total_function_evaluation
			total_gradient_evaluation = userdata.total_gradient_evaluation
			total_hessian_evaluation = userdata.total_hessian_evaluation
			function_value = obj(nlp, solution)
			graient_value = norm(grad(nlp, solution), 2)
			if status != 0 || graient_value > tol_opt
				iter = max_it + 1
				total_function_evaluation = max_it + 1
				total_gradient_evaluation = max_it + 1
			end
			computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "function_value" => function_value, "gradient_value" => graient_value)
			println("------------------------MODEL SOLVED WITH STATUS: ", status)
			directory_name = string(folder_name, "/", "$optimization_method")
			outputIterationsStatusToCSVFile(directory_name, cutest_problem, status, computation_stats, total_iterations_count, optimization_method)
		end
    finally
        if nlp != nothing
            finalize(nlp)
        end
    end
end

function executeCUTEST_Models_benchmark(
	cutest_problems::Vector{String},
	folder_name::String,
	optimization_method::String,
	max_it::Int64=10000,
    max_time::Float64=30*60,
    tol_opt::Float64=1e-5,
    θ::Float64=0.1,
    β::Float64=0.1,
	ω::Float64=8.0,
    γ_2::Float64=0.8,
    r_1::Float64=1.0,
	δ::Float64=0.0,
	trust_region_method_subproblem_solver::String=consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_DEFAULT
	)
	#=
	all_runs_results_output_file_path = string(folder_name, "/all_runs_results.csv")
	open(all_runs_results_output_file_path,"a") do all_runs_status_csv_file
		write(all_runs_status_csv_file, "optimization_method,beta,theta,geomean_total_iterations_count,geomean_count_factorization\n");
    	end
	
	optimization_method_possible_values = ["CAT", "CAT_GALAHAD_FACTORIZATION"]
	beta_possible_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
	theta_possible_values = [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
	possible_scenarios = [[optimization_method_, beta, theta] for optimization_method_ in optimization_method_possible_values for beta in beta_possible_values for theta in theta_possible_values]
	=#
	#for possible_scenario in possible_scenarios
	#	optimization_method = possible_scenario[1]
	#	θ = possible_scenario[3]
	#	β = possible_scenario[2]
		#Old Code
		#=
		total_results_output_directory =  string(folder_name, "/$optimization_method")
		total_results_output_file_name = "table_cutest_$optimization_method.csv"
		total_results_output_file_path = string(total_results_output_directory, "/", total_results_output_file_name)
    		rm(total_results_output_file_path, force=true)
	    	mkpath(total_results_output_directory);
    		open(total_results_output_file_path,"a") do iteration_status_csv_file
			write(iteration_status_csv_file, "problem_name,status,total_iterations_count,function_value,graient_value,total_function_evaluation,total_gradient_evaluation,total_hessian_evaluation,count_factorization\n");
    		end
		=#
		total_results_output_directory =  string(folder_name, "/$optimization_method")
		total_results_output_file_name = "table_cutest_$optimization_method.csv"
		total_results_output_file_path = string(total_results_output_directory, "/", total_results_output_file_name)
		if !isfile(total_results_output_file_path)
			mkpath(total_results_output_directory);
    			open(total_results_output_file_path,"a") do iteration_status_csv_file
				write(iteration_status_csv_file, "problem_name,status,total_iterations_count,function_value,graient_value,total_function_evaluation,total_gradient_evaluation,total_hessian_evaluation,count_factorization\n");
	    		end
		end
		
	# if occursin(optimization_method_CAT, optimization_method)
	# 	near_convexity_rate_csv_file_name = "table_near_convexity_rate_$optimization_method.csv"
	# 	near_convexity_rate_csv_file_path = string(total_results_output_directory, "/", near_convexity_rate_csv_file_name)
	# 	open(near_convexity_rate_csv_file_path,"a") do near_convexity_rate_csv_file
	# 		write(near_convexity_rate_csv_file, "problem_name,status,rate\n");
	#     end
	# end
		
		for problem in cutest_problems
			problem_output_file_path = string(total_results_output_directory, "/", problem, ".csv")
			if isfile(problem_output_file_path) || problem in DataFrame(CSV.File(total_results_output_file_path)).problem_name
				@show problem
				continue
			else
        			runModelFromProblem(problem, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1, δ, trust_region_method_subproblem_solver)
			end
    		end
		df = DataFrame(CSV.File(total_results_output_file_path))
		geomean_total_iterations_count = geomean(df.total_iterations_count)
		geomean_count_factorization = geomean(df.count_factorization)


    		geomean_total_function_evaluation = geomean(df.total_function_evaluation)
    		geomean_total_gradient_evaluation = geomean(df.total_gradient_evaluation)
		geomean_total_hessian_evaluation  = geomean(df.total_hessian_evaluation)


		@show geomean_total_iterations_count
		@show geomean_count_factorization
		@show geomean_total_function_evaluation
		@show geomean_total_gradient_evaluation
		@show geomean_total_hessian_evaluation
	#	open(all_runs_results_output_file_path,"a") do all_runs_status_csv_file
	#		write(all_runs_status_csv_file, "$optimization_method,$β,$θ,$geomean_total_iterations_count,$geomean_count_factorization\n")
    	#	end
	#end
end

function outputResultsToCSVFile(directory_name::String, cutest_problem::String, results::DataFrame)
	cutest_problem_file_name = string(directory_name, "/$cutest_problem.csv")
    CSV.write(cutest_problem_file_name, results, header = true)
end

function outputIterationsStatusToCSVFile(
	directory_name::String,
	cutest_problem::String,
	status::String,
	computation_stats::Dict,
	total_iterations_count::Int32,
	optimization_method::String,
	count_factorization::Int64=0
	)
    total_function_evaluation = Int(computation_stats["total_function_evaluation"])
    total_gradient_evaluation = Int(computation_stats["total_gradient_evaluation"])
    total_hessian_evaluation  = Int(computation_stats["total_hessian_evaluation"])

    function_value = computation_stats["function_value"]
    gradient_value = computation_stats["gradient_value"]
	file_name = string(directory_name, "/", "table_cutest_$optimization_method.csv")
    open(file_name,"a") do iteration_status_csv_file
		write(iteration_status_csv_file, "$cutest_problem,$status,$total_iterations_count,$function_value,$gradient_value,$total_function_evaluation,$total_gradient_evaluation,$total_hessian_evaluation,$count_factorization\n")
    end
end

function outputIterationsStatusToCSVFile(
	directory_name::String,
	cutest_problem::String,
	status::String,
	computation_stats::Dict,
	total_iterations_count::Int64,
	optimization_method::String,
	count_factorization::Int64=0
	)
    total_function_evaluation = Int(computation_stats["total_function_evaluation"])
    total_gradient_evaluation = Int(computation_stats["total_gradient_evaluation"])
    total_hessian_evaluation  = Int(computation_stats["total_hessian_evaluation"])

    function_value = computation_stats["function_value"]
    gradient_value = computation_stats["gradient_value"]
	file_name = string(directory_name, "/", "table_cutest_$optimization_method.csv")
    open(file_name,"a") do iteration_status_csv_file
		write(iteration_status_csv_file, "$cutest_problem,$status,$total_iterations_count,$function_value,$gradient_value,$total_function_evaluation,$total_gradient_evaluation,$total_hessian_evaluation,$count_factorization\n")
    end
end

# function outputHowOftenNearConvexityConditionHolds(directory_name::String,
# 	cutest_problem::String,
# 	status::String,
# 	optimization_method::String,
# 	iteration_stats::DataFrame
# 	)
# 	numberOfRow = size(iteration_stats)[1]
# 	numberOfCol = size(iteration_stats)[2]
# 	count = 0
# 	theta = 1.0
#     # theta = 10.0
# 	for k in 1:numberOfRow
# 		f_T = last(iteration_stats, 1)[!, "fval"][1]
# 		g_T = last(iteration_stats, 1)[!, "gradval"][1]
# 		x_T = last(iteration_stats, 1)[!, "x"][1]
#
# 		f_k = iteration_stats[k,"fval"]
# 		x_k = iteration_stats[k,"x"]
# 		g_k = grad(nlp, x_k)
# 		if (f_T >= f_k + transpose(g_k) * (x_T - x_k))  && !(f_T >= f_k + theta * (transpose(g_k) * (x_T - x_k)))
# 			@show f_T
# 			@show f_k
# 			@show transpose(g_k) * (x_T - x_k)
# 			@show theta *(transpose(g_k) * (x_T - x_k))
# 			@show f_k + transpose(g_k) * (x_T - x_k)
# 			@show f_k + theta * (transpose(g_k) * (x_T - x_k))
# 			throw(error("Calculation buggy"))
# 		end
# 		if f_T >= f_k + theta * (transpose(g_k) * (x_T - x_k))
# 			count = count + 1
# 		end
# 	end
# 	rate = 0.0
# 	if numberOfRow > 0
# 		rate = count / numberOfRow
# 	end
# 	near_convexity_rate_csv_file_name = "table_near_convexity_rate_$optimization_method.csv"
# 	near_convexity_rate_csv_file_path = string(directory_name, "/", near_convexity_rate_csv_file_name)
# 	open(near_convexity_rate_csv_file_path,"a") do near_convexity_rate_csv_file
# 		write(near_convexity_rate_csv_file, "$cutest_problem,$status,$rate\n")
#     end
# end
