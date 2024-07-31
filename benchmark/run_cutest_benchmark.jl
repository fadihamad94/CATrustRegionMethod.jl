using JuMP, NLPModels, NLPModelsJuMP, LinearAlgebra, CUTEst, CSV, Test, DataFrames, SparseArrays, StatsBase, Random, Dates
include("../src/CAT_Module.jl")

const optimization_method_CAT = "CAT"
const optimization_method_CAT_theta_0 = "CAT_THETA_ZERO"

const optimization_method_CAT_galahad_factorization = "CAT_GALAHAD_FACTORIZATION"
const optimization_method_CAT_galahad_iterative = "CAT_GALAHAD_ITERATIVE"

const skip_list = ["YATP1LS", "YATP2CLS", "YATP2LS", "YATP1CLS"]

const default_problems_list = ["ARGLINA", "ARGLINB", "ARGLINC", "ARGTRIGLS", "ARWHEAD", "BA-L16LS", "BA-L21LS", "BA-L49LS", "BA-L52LS", "BA-L73LS", "BDQRTIC", "BOX", "BOXPOWER", "BROWNAL", "BROYDN3DLS", "BROYDN7D", "BROYDNBDLS", "BRYBND", "CHAINWOO", "COATING", "COATINGNE", "COSINE", "CRAGGLVY", "CURLY10", "CURLY20", "CURLY30", "CYCLOOCFLS", "DIXMAANA", "DIXMAANB", "DIXMAANC", "DIXMAAND", "DIXMAANE", "DIXMAANF", "DIXMAANG", "DIXMAANH", "DIXMAANI", "DIXMAANJ", "DIXMAANK", "DIXMAANL", "DIXMAANM", "DIXMAANN", "DIXMAANO", "DIXMAANP", "DIXON3DQ", "DQDRTIC", "DQRTIC", "EDENSCH", "EG2", "EIGENALS", "EIGENBLS", "EIGENCLS", "ENGVAL1", "EXTROSNB", "FLETBV3M", "FLETCBV2", "FLETCBV3", "FLETCHBV", "FLETCHCR", "FMINSRF2", "FMINSURF", "FREUROTH", "GENHUMPS", "GENROSE", "INDEF", "INDEFM", "INTEQNELS", "JIMACK", "KSSLS", "LIARWHD", "LUKSAN11LS", "LUKSAN15LS", "LUKSAN16LS", "LUKSAN17LS", "LUKSAN21LS", "LUKSAN22LS", "MANCINO", "MNISTS0LS", "MNISTS5LS", "MODBEALE", "MOREBV", "MSQRTALS", "MSQRTBLS", "NCB20", "NCB20B", "NONCVXU2", "NONCVXUN", "NONDIA", "NONDQUAR", "NONMSQRT", "OSCIGRAD", "OSCIPATH", "PENALTY1", "PENALTY2", "PENALTY3", "POWELLSG", "POWER", "QING", "QUARTC", "SBRYBND", "SCHMVETT", "SCOSINE", "SCURLY10", "SCURLY20", "SCURLY30", "SENSORS", "SINQUAD", "SPARSINE", "SPARSQUR", "SPIN2LS", "SPINLS", "SPMSRTLS", "SROSENBR", "SSBRYBND", "SSCOSINE", "TESTQUAD", "TOINTGSS", "TQUARTIC", "TRIDIA", "VARDIM", "VAREIGVL", "WOODS", "YATP1CLS", "YATP1LS", "YATP2CLS", "YATP2LS"]

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
	ω_1::Float64,
	ω_2::Float64,
	γ_1::Float64,
    γ_2::Float64,
    r_1::Float64,
	δ::Float64,
    min_nvar::Int64,
    max_nvar::Int64,
    print_level::Int64,
    optimization_method::String
    )

    cutest_problems = []
    if default_problems
		cutest_problems = default_problems_list
    else
		cutest_problems = get_problem_list(min_nvar, max_nvar)
    end

	trust_region_method_subproblem_solver = optimization_method == optimization_method_CAT ? consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_DEFAULT : (optimization_method == optimization_method_CAT_galahad_factorization ? consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_TRS : consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_GLTR)
	if θ == 0.0
		optimization_method = optimization_method_CAT_theta_0
	end

	cutest_problems = filter!(e->e ∉ skip_list, cutest_problems)
    executeCUTEST_Models_benchmark(cutest_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω_1, ω_2, γ_1, γ_2, r_1, print_level, δ, trust_region_method_subproblem_solver)
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
	ω_1::Float64,
	ω_2::Float64,
	γ_1::Float64,
    γ_2::Float64,
    r_1::Float64,
	δ::Float64,
	print_level::Int64,
	trust_region_method_subproblem_solver::String=consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_DEFAULT
	)
    global nlp = nothing
    try
		dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
        println("$dates_format-----------EXECUTING PROBLEM----------", cutest_problem)
		@info "$dates_format-----------EXECUTING PROBLEM----------$cutest_problem"
        nlp = CUTEstModel(cutest_problem)
		if optimization_method == optimization_method_CAT || optimization_method == optimization_method_CAT_theta_0 || optimization_method == optimization_method_CAT_galahad_factorization || optimization_method == optimization_method_CAT_galahad_iterative
			termination_conditions_struct = consistently_adaptive_trust_region_method.TerminationConditions(max_it, tol_opt, max_time)
		    initial_radius_struct = consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(r_1)
			problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, termination_conditions_struct, initial_radius_struct, β, θ, ω_1, ω_2, γ_1, γ_2,print_level)
	        x_1 = problem.nlp.meta.x0
	        x, status, iteration_stats, computation_stats, total_iterations_count, total_execution_time = consistently_adaptive_trust_region_method.CAT(problem, x_1, δ, trust_region_method_subproblem_solver)
			status_string = convertStatusCodeToStatusString(status)
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
			dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
			println("$dates_format------------------------MODEL SOLVED WITH STATUS: ", status)
			@info "$dates_format------------------------MODEL SOLVED WITH STATUS: $status"
			directory_name = string(folder_name, "/", "$optimization_method")
			# outputResultsToCSVFile(directory_name, cutest_problem, iteration_stats)
			outputIterationsStatusToCSVFile(directory_name, cutest_problem, status_string, total_execution_time, computation_stats_modified, optimization_method)
		end
	catch e
		@show e
		status = "INCOMPLETE"
		computation_stats = Dict("total_function_evaluation" => 2 * max_it + 1, "total_gradient_evaluation" => 2 * max_it + 1, "total_hessian_evaluation" => 2 * max_it + 1, "total_number_factorizations" =>  2 * max_it + 1, "function_value" => NaN, "gradient_value" => NaN)
		dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
		println("$dates_format------------------------MODEL SOLVED WITH STATUS: ", status)
		@info "$dates_format------------------------MODEL SOLVED WITH STATUS: $status"
		directory_name = string(folder_name, "/", "$optimization_method")
		total_execution_time = max_time
		outputIterationsStatusToCSVFile(directory_name, cutest_problem, status, total_execution_time, computation_stats, optimization_method)
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
	max_it::Int64=100000,
    max_time::Float64=5 * 60 * 60.0,
    tol_opt::Float64=1e-5,
    θ::Float64=0.1,
	β::Float64=0.1,
	ω_1::Float64=8.0,
	ω_2::Float64=20.0,
	γ_1::Float64=0.01,
    γ_2::Float64=0.8,
    r_1::Float64=0.0,
	print_level::Int64=0,
	δ::Float64=0.0,
	trust_region_method_subproblem_solver::String=consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_DEFAULT
	)
	println("CUTEst Problems are: $cutest_problems")
	geomean_results_file_path = string(folder_name, "/", "geomean_total_results.csv")

	if !isfile(geomean_results_file_path)
		open(geomean_results_file_path, "w") do file
			write(file, "optimization_method,total_failure,geomean_total_function_evaluation,geomean_total_gradient_evaluation,geomean_total_hessian_evaluation,geomean_count_factorization\n");
		end
	end

	total_results_output_directory =  string(folder_name, "/", "$optimization_method")
	total_results_output_file_name = "table_cutest_$optimization_method.csv"
	total_results_output_file_path = string(total_results_output_directory, "/", total_results_output_file_name)
	if !isfile(total_results_output_file_path)
		mkpath(total_results_output_directory);
		open(total_results_output_file_path,"a") do iteration_status_csv_file
			write(iteration_status_csv_file, "problem_name,status,total_execution_time,function_value,gradient_value,total_function_evaluation,total_gradient_evaluation,total_hessian_evaluation,total_factorization_evaluation\n");
    	end
	end

	for problem in cutest_problems
		problem_output_file_path = string(total_results_output_directory, "/", problem, ".csv")
		if isfile(problem_output_file_path) || problem in DataFrame(CSV.File(total_results_output_file_path)).problem_name || problem ∈ skip_list
			@show problem
			dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
			@info "$dates_format Skipping Problem $problem."
			continue
		else
        	runModelFromProblem(problem, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω_1, ω_2, γ_1, γ_2, r_1, δ, print_level, trust_region_method_subproblem_solver)
		end
    end
	df = DataFrame(CSV.File(total_results_output_file_path))
	df = filter(:problem_name => p_n -> p_n in cutest_problems, df)

	@show "Computing Normal Geometric Means"
	shift = 0
	geomean_total_function_evaluation, geomean_total_gradient_evaluation, geomean_total_hessian_evaluation, geomean_count_factorization = computeNormalGeomeans(df, shift, tol_opt, max_time / 3600, max_it)

	counts = countmap(df.status)
	total_failure = length(df.status) - get(counts, "SUCCESS", 0) - get(counts, "OPTIMAL", 0)
	open(geomean_results_file_path, "a") do file
		write(file, "$optimization_method,$total_failure,$geomean_total_function_evaluation,$geomean_total_gradient_evaluation,$geomean_total_hessian_evaluation,$geomean_count_factorization\n");
	end
end

function computeNormalGeomeans(df::DataFrame, shift::Int64, ϵ::Float64, time_limit::Float64, max_it::Int64)
	ϕ(θ) = (max_it + 1)/time_limit
	return computeShiftedAndCorrectedGeomeans(ϕ, df, shift, ϵ, time_limit, max_it)
end

function computeShiftedGeomeans(df::DataFrame, shift::Int64, ϵ::Float64, time_limit::Float64, max_it::Int64)
	ϕ(θ) = (max_it + 1)/time_limit
	return computeShiftedAndCorrectedGeomeans(ϕ, df, shift, ϵ, time_limit, max_it)
end

function computeShiftedAndCorrectedGeomeans(ϕ::Function, df::DataFrame, shift::Int64, ϵ::Float64, time_limit::Float64, max_it::Int64)
	total_factorization_count_vec = Vector{Float64}()
	total_function_evaluation_vec = Vector{Float64}()
	total_gradient_evaluation_vec = Vector{Float64}()
	total_hessian_evaluation_vec = Vector{Float64}()
	for i in 1:size(df)[1]
		if df[i, :].status == "SUCCESS" || df[i, :].status == "OPTIMAL"
			push!(total_factorization_count_vec, df[i, :].total_factorization_evaluation)
			push!(total_function_evaluation_vec, df[i, :].total_function_evaluation)
			push!(total_gradient_evaluation_vec, df[i, :].total_gradient_evaluation)
			push!(total_hessian_evaluation_vec, df[i, :].total_hessian_evaluation)
		else
			push!(total_factorization_count_vec, 2 * max_it + 1)
			push!(total_function_evaluation_vec, 2 * max_it + 1)
			push!(total_gradient_evaluation_vec, 2 * max_it + 1)
			push!(total_hessian_evaluation_vec, 2 * max_it + 1)
		end
	end

	df_results_new = DataFrame()
	df_results_new.problem_name = df.problem_name
	df_results_new.total_factorization_evaluation = total_factorization_count_vec
	df_results_new.total_function_evaluation = total_function_evaluation_vec
	df_results_new.total_gradient_evaluation = total_gradient_evaluation_vec
	df_results_new.total_hessian_evaluation = total_hessian_evaluation_vec

	return computeShiftedGeomeans(df_results_new, shift)
end

function computeShiftedGeomeans(df::DataFrame, shift::Int64)
	geomean_count_factorization = geomean(df.total_factorization_evaluation .+ shift) - shift
	geomean_total_function_evaluation = geomean(df.total_function_evaluation .+ shift) - shift
	geomean_total_gradient_evaluation = geomean(df.total_gradient_evaluation .+ shift) - shift
	geomean_total_hessian_evaluation  = geomean(df.total_hessian_evaluation .+ shift) - shift

	return (geomean_total_function_evaluation, geomean_total_gradient_evaluation, geomean_total_hessian_evaluation, geomean_count_factorization)
end

function outputResultsToCSVFile(directory_name::String, cutest_problem::String, results::DataFrame)
	cutest_problem_file_name = string(directory_name, "/$cutest_problem.csv")
    CSV.write(cutest_problem_file_name, results, header = true)
end

function convertStatusCodeToStatusString(status)
    dict_status_code = Dict(consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL => "OPTIMAL",
    consistently_adaptive_trust_region_method.TerminationStatusCode.UNBOUNDED => "UNBOUNDED",
    consistently_adaptive_trust_region_method.TerminationStatusCode.ITERATION_LIMIT => "ITERATION_LIMIT",
    consistently_adaptive_trust_region_method.TerminationStatusCode.TIME_LIMIT => "TIME_LIMIT",
    consistently_adaptive_trust_region_method.TerminationStatusCode.MEMORY_LIMIT => "MEMORY_LIMIT",
    consistently_adaptive_trust_region_method.TerminationStatusCode.STEP_SIZE_LIMIT => "STEP_SIZE_LIMIT",
    consistently_adaptive_trust_region_method.TerminationStatusCode.NUMERICAL_ERROR => "NUMERICAL_ERROR",
    consistently_adaptive_trust_region_method.TerminationStatusCode.OTHER_ERROR => "OTHER_ERROR")
    return dict_status_code[status]
end

function outputIterationsStatusToCSVFile(
	directory_name::String,
	cutest_problem::String,
	status::String,
	total_execution_time::Float64,
	computation_stats::Dict,
	optimization_method::String
	)
    total_function_evaluation = Int(computation_stats["total_function_evaluation"])
    total_gradient_evaluation = Int(computation_stats["total_gradient_evaluation"])
    total_hessian_evaluation  = Int(computation_stats["total_hessian_evaluation"])
	total_number_factorizations = Int(computation_stats["total_number_factorizations"])
    function_value = computation_stats["function_value"]
    gradient_value = computation_stats["gradient_value"]
	file_name = string(directory_name, "/", "table_cutest_$optimization_method.csv")
    open(file_name,"a") do iteration_status_csv_file
		write(iteration_status_csv_file, "$cutest_problem,$status,$total_execution_time,$function_value,$gradient_value,$total_function_evaluation,$total_gradient_evaluation,$total_hessian_evaluation,$total_number_factorizations\n")
    end
end
