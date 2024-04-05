import ArgParse
using JuMP, CUTEst, CSV, DataFrames, StatsBase, Dates, Statistics, Plots
include("../src/CAT.jl")

"""
Defines parses and args.
# Returns
A dictionary with the values of the command-line arguments.
"""

const skip_list = ["DMN15333LS", "DIAMON2DLS", "DMN37142LS", "BA-L49LS", "NONCVXU2", "DMN15102LS", "DMN15332LS", "DMN37143LS", "EIGENCLS", "YATP1LS", "YATP2CLS", "YATP2LS", "YATP1CLS"]

function if_mkpath(dir::String)
  if !isdir(dir)
     mkpath(dir)
  end
end

function get_problem_list(min_nvar, max_nvar)
	return CUTEst.select(min_var = min_nvar, max_var = max_nvar, max_con = 0, only_free_var = true)
end

function parse_command_line()
  arg_parse = ArgParse.ArgParseSettings()

  ArgParse.@add_arg_table! arg_parse begin
    "--output_dir"
    help = "The directory for output files."
    arg_type = String
    required = true

    "--default_problems"
    help = "Specify weither to use the same list of CUTEst tests used in the paper or not. IF not, you can specify the size of the problems."
    arg_type = Bool
    required = true

    "--max_it"
    help = "The maximum number of iterations to run"
    arg_type = Int64
    default = 10000

    "--max_time"
    help = "The maximum time to run in seconds"
    arg_type = Float64
    default = 30 * 60.0

    "--tol_opt"
    help = "The tolerance for optimality"
    arg_type = Float64
    default = 1e-5

    "--θ"
    help = "θ parameter for CAT"
    arg_type = Float64
    default = 0.1

    "--β_1"
    help = "β_1 parameter for CAT"
    arg_type = Float64
    default = 0.1

    "--β_2"
    help = "β parameter for CAT"
    arg_type = Float64
    default = 0.8

    "--ω_1"
    help = "ω_1 parameter for CAT"
    arg_type = Float64
    default = 4.0

    "--ω_2"
    help = "ω_2 parameter for CAT"
    arg_type = Float64
    default = 20.0

    "--γ_2"
    help = "γ_2 parameter for CAT"
    arg_type = Float64
    default = 0.2

    "--r_1"
    help = "Initial trust region radius. Negative values indicates using our default radius of value 10 * \frac{|g(x_1)||}{||H(x_1)||}"
    arg_type = Float64
    default = 1.0

    "--min_nvar"
    help = "The minimum number of variables for CUTEst model"
    arg_type = Int64
    default =  1

    "--max_nvar"
    help = "The maximum number of variables for CUTEst model"
    arg_type = Int64
    default =  500

    "--δ"
    help = "Starting δ for CAT"
    arg_type = Float64
    default = 0.0

	"--print_level"
	help = "Print level. If < 0, nothing to print, 0 for info and > 0 for debugging."
    arg_type = Int64
    default = 0

    "--criteria"
    help = "The ordering of criteria separated by commas. Allowed values are `ρ_hat_rule`, `GALAHAD_TRS`, `initial_radius`, `radius_update_rule`."
    arg_type = String
    # default = "ρ_hat_rule,GALAHAD_TRS,initial_radius,radius_update_rule"
	default = "ρ_hat_rule,radius_update_rule,initial_radius"
  end

  return ArgParse.parse_args(arg_parse)
end

function createProblemData(
	criteria::Vector{String},
	max_it::Int64,
	max_time::Float64,
	tol_opt::Float64,
	θ::Float64,
	β_1::Float64,
	β_2::Float64,
	ω_1::Float64,
	ω_2::Float64,
	γ_2::Float64,
	r_1::Float64)
		problem_data_vec = []
		solver = consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_DEFAULT
		compute_ρ_hat_approach = "NOT DEFAULT"
		problem_data = (β_1, β_1, θ, ω_1, ω_1, r_1, max_it, tol_opt, max_time, γ_2, solver, compute_ρ_hat_approach)
		for crt in criteria
			if crt == "original"
				push!(problem_data_vec, problem_data)
			elseif crt == "ρ_hat_rule"
				compute_ρ_hat_approach = "DEFAULT"
				# Create a new tuple with the specified element overridden
				index_to_override = 12
				new_problem_data = (problem_data[1:index_to_override-1]..., compute_ρ_hat_approach, problem_data[index_to_override+1:end]...)
				problem_data = new_problem_data
				push!(problem_data_vec, problem_data)
			elseif crt == "GALAHAD_TRS"
				solver = consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_TRS
				index_to_override = 11
				new_problem_data = (problem_data[1:index_to_override-1]..., solver, problem_data[index_to_override+1:end]...)
				problem_data = new_problem_data
				push!(problem_data_vec, problem_data)
			elseif crt == "initial_radius"
				r_1 = 0.0
				index_to_override = 6
				new_problem_data = (problem_data[1:index_to_override-1]..., r_1, problem_data[index_to_override+1:end]...)
				problem_data = new_problem_data
				push!(problem_data_vec, problem_data)
			else
				index_to_override = 2
				new_problem_data = (problem_data[1:index_to_override-1]..., β_2, problem_data[index_to_override+1:end]...)
				problem_data = new_problem_data
				index_to_override = 5
				new_problem_data = (problem_data[1:index_to_override-1]..., ω_2, problem_data[index_to_override+1:end]...)
				problem_data = new_problem_data
				push!(problem_data_vec, problem_data)
			end
		end
		return problem_data_vec
end

function outputIterationsStatusToCSVFile(
	start_time::String,
	end_time::String,
	cutest_problem::String,
	status::String,
	computation_stats::Dict,
	total_results_output_file_path::String,
	total_iterations_count::Integer,
	count_factorization::Integer
	)
    total_function_evaluation = Int(computation_stats["total_function_evaluation"])
    total_gradient_evaluation = Int(computation_stats["total_gradient_evaluation"])
    total_hessian_evaluation  = Int(computation_stats["total_hessian_evaluation"])

    function_value = computation_stats["function_value"]
    gradient_value = computation_stats["gradient_value"]

    open(total_results_output_file_path,"a") do iteration_status_csv_file
			write(iteration_status_csv_file, "$start_time,$end_time,$cutest_problem,$status,$total_iterations_count,$function_value,$gradient_value,$total_function_evaluation,$total_gradient_evaluation,$total_hessian_evaluation,$count_factorization\n")
    end
end

function runModelFromProblem(
	cutest_problem::String,
	criteria::String,
	problem_data,
	δ::Float64,
	print_level::Int64,
	total_results_output_file_path::String
	)

	global nlp = nothing
	β_1, β_2, θ, ω_1, ω_2, r_1, max_it, tol_opt, max_time, γ_2, solver, compute_ρ_hat_approach = problem_data
	start_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
	try
		dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
		println("$dates_format-----------EXECUTING PROBLEM----------", cutest_problem)
		@info "$dates_format-----------EXECUTING PROBLEM----------$cutest_problem"
		nlp = CUTEstModel(cutest_problem)
		problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, β_1, β_2, θ, ω_1, ω_2, r_1, max_it, tol_opt, γ_2, max_time, print_level, compute_ρ_hat_approach)
		x_1 = problem.nlp.meta.x0
		start_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
		x = x_1
		status = Nothing
		iteration_stats = Nothing
		total_iterations_count = 0
		if criteria == "original" || criteria == "CAT"
			x, status, iteration_stats, computation_stats, total_iterations_count = consistently_adaptive_trust_region_method.CAT_original_alg(problem, x_1, δ, solver)
	    else
			x, status, iteration_stats, computation_stats, total_iterations_count = consistently_adaptive_trust_region_method.CAT(problem, x_1, δ, solver)
		end
		end_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
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

		total_number_factorizations = Int64(computation_stats_modified["total_number_factorizations"])
		outputIterationsStatusToCSVFile(start_time, end_time, cutest_problem, status, computation_stats_modified, total_results_output_file_path, total_iterations_count, total_number_factorizations)
	catch e
		@show e
		status = "INCOMPLETE"
		computation_stats = Dict("total_function_evaluation" => max_it + 1, "total_gradient_evaluation" => max_it + 1, "total_hessian_evaluation" => max_it + 1, "function_value" => NaN, "gradient_value" => NaN)
		dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
		end_time = dates_format
		println("$dates_format------------------------MODEL SOLVED WITH STATUS: ", status)
		@info "$dates_format------------------------MODEL SOLVED WITH STATUS: $status"
		outputIterationsStatusToCSVFile(start_time, end_time, cutest_problem, status, computation_stats, total_results_output_file_path, max_it + 1, max_it + 1)
  	finally
	  	if nlp != nothing
	    	finalize(nlp)
	    end
  	end
end

function computeGeomeans(df::DataFrame, max_it::Int64)
	total_iterations_count_vec = Vector{Float64}()
	total_factorization_count_vec = Vector{Float64}()
	total_function_evaluation_vec = Vector{Float64}()
	total_gradient_evaluation_vec = Vector{Float64}()
	total_hessian_evaluation_vec = Vector{Float64}()
	for i in 1:size(df)[1]
		if df[i, :].status == "SUCCESS" || df[i, :].status == "OPTIMAL"
			push!(total_iterations_count_vec, df[i, :].total_iterations_count)
			push!(total_factorization_count_vec, df[i, :].count_factorization)
			push!(total_function_evaluation_vec, df[i, :].total_function_evaluation)
			push!(total_gradient_evaluation_vec, df[i, :].total_gradient_evaluation)
			push!(total_hessian_evaluation_vec, df[i, :].total_hessian_evaluation)
		else
			push!(total_iterations_count_vec, max_it + 1)
			push!(total_factorization_count_vec, max(df[i, :].count_factorization, max_it + 1))
			push!(total_function_evaluation_vec, max_it + 1)
			push!(total_gradient_evaluation_vec, max_it + 1)
			push!(total_hessian_evaluation_vec, max_it + 1)
		end
	end

	df_results_new = DataFrame()
	df_results_new.problem_name = df.problem_name
	df_results_new.total_iterations_count = total_iterations_count_vec
	df_results_new.count_factorization = total_factorization_count_vec
	df_results_new.total_function_evaluation = total_function_evaluation_vec
	df_results_new.total_gradient_evaluation = total_gradient_evaluation_vec
	df_results_new.total_hessian_evaluation = total_hessian_evaluation_vec

	geomean_total_iterations_count = geomean(df_results_new.total_iterations_count)
	geomean_count_factorization = geomean(df_results_new.count_factorization)
	geomean_total_function_evaluation = geomean(df_results_new.total_function_evaluation)
	geomean_total_gradient_evaluation = geomean(df_results_new.total_gradient_evaluation)
	geomean_total_hessian_evaluation  = geomean(df_results_new.total_hessian_evaluation)

	return (geomean_total_iterations_count, geomean_total_function_evaluation, geomean_total_gradient_evaluation, geomean_total_hessian_evaluation, geomean_count_factorization)
end

function runProblems(
	criteria::Vector{String},
	problem_data_vec::Vector{Any},
	δ::Float64,
	folder_name::String,
	default_problems::Bool,
	min_nvar::Int64,
	max_nvar::Int64,
	print_level::Int64)

	cutest_problems = []
	if default_problems
		cutest_problems = CUTEst.select(contype="unc")
	else
		cutest_problems = get_problem_list(min_nvar, max_nvar)
	end

	cutest_problems = filter!(e->e ∉ skip_list, cutest_problems)

	# default_test_problems = ["DIXMAANI", "LIARWHD", "SCHMVETT", "VAREIGVL", "CYCLOOCFLS", "DIXMAANJ", "SBRYBND", "ARGLINC", "TOINTGOR", "DIXMAANC", "WAYSEA2", "DMN37142LS", "EIGENALS", "YATP1LS", "GENHUMPS", "OSCIPATH", "FLETCBV2", "DIXMAAND", "S308NE", "SPMSRTLS", "NONCVXUN", "BRYBND", "DIXMAANM", "DQRTIC", "MISRA1ALS", "BOX", "DMN37143LS", "TOINTGSS", "SPARSINE", "VESUVIALS", "INTEQNELS", "COATINGNE", "HAIRY", "PALMER1C", "BA-L49LS", "SSCOSINE", "NONCVXU2", "BROYDN7D", "COSINE", "DIXMAANO", "SPINLS", "BROYDN3DLS", "PALMER2C", "CURLY20", "NONMSQRT", "CURLY30", "FREUROTH", "PALMER8C", "FMINSRF2", "YATP2CLS", "DMN15332LS", "SCURLY20", "NONDQUAR", "SCURLY30", "LUKSAN21LS", "ECKERLE4LS", "HAHN1LS", "DMN15102LS", "WOODS", "JIMACK", "HIMMELBF", "VARDIM", "BROYDNBDLS", "FLETBV3M", "DIXMAANA", "CHWIRUT2LS", "POWER", "DEVGLA2NE", "BA-L1SPLS", "BA-L73LS", "DIAMON2DLS", "THURBERLS", "GAUSS1LS", "PENALTY3", "MODBEALE", "PALMER6C", "DIXMAANN", "LUKSAN17LS", "EIGENCLS", "INDEFM", "SROSENBR", "MNISTS5LS", "INDEF", "ARWHEAD", "DIXON3DQ", "MGH09LS", "BDQRTIC", "DIXMAANH", "DIXMAANB", "BA-L21LS", "GULF", "POWELLSG", "TRIDIA", "DMN15333LS", "NCB20", "FLETCBV3", "CHAINWOO", "HATFLDE", "DIXMAANP", "FLETCHCR", "ERRINRSM", "FMINSURF", "DIXMAANE", "MSQRTALS", "CURLY10", "BOXPOWER", "DQDRTIC", "OSCIGRAD", "FLETCHBV", "ARGLINB", "DIXMAANF", "BA-L16LS", "MOREBV", "NONDIA", "MSQRTBLS", "SCURLY10", "TQUARTIC", "CHWIRUT1LS", "YATP2LS", "ENGVAL1", "DIXMAANG", "HILBERTB", "DIXMAANK", "QUARTC", "EDENSCH", "YATP1CLS", "TOINTPSP", "SSBRYBND", "CRAGGLVY", "SPARSQUR", "DMN15103LS", "NCB20B", "BA-L52LS", "POWERSUM", "ROSENBRTU", "SINQUAD", "DIXMAANL", "PALMER4C", "TESTQUAD", "EIGENBLS", "TRIGON2", "POWELLSQLS", "SCOSINE", "EXP2", "METHANB8LS", "LANCZOS3LS", "DIAMON3DLS"]
	#
	# default_test_problems = filter!(e->e ∉ skip_list, default_test_problems)

	# cutest_problems = default_test_problems

	println("CUTEst Problems are: $cutest_problems")
	@info length(cutest_problems)

	geomean_results_file_path = string(folder_name, "/", "geomean_results_ablation_study.csv")

	if isfile(geomean_results_file_path)
        rm(geomean_results_file_path)  # Delete the file if it already exists
    end

    open(geomean_results_file_path, "w") do file
        write(file, "criteria,total_failure,geomean_total_iterations_count,geomean_total_function_evaluation,geomean_total_gradient_evaluation,geomean_total_hessian_evaluation,geomean_count_factorization\n");
    end

	for index in 1:length(criteria)
		crt = criteria[index]
		total_results_output_directory =  string(folder_name, "/$crt")
		total_results_output_file_name = "table_cutest_$crt.csv"
		total_results_output_file_path = string(total_results_output_directory, "/", total_results_output_file_name)

		if !isfile(total_results_output_file_path)
			mkpath(total_results_output_directory);
			open(total_results_output_file_path,"a") do iteration_status_csv_file
				write(iteration_status_csv_file, "start_time,end_time,problem_name,status,total_iterations_count,function_value,gradient_value,total_function_evaluation,total_gradient_evaluation,total_hessian_evaluation,count_factorization\n");
		    end
		end

		for cutest_problem in cutest_problems
			if cutest_problem in DataFrame(CSV.File(total_results_output_file_path)).problem_name
				@show cutest_problem
				dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
				@info "$dates_format Skipping Problem $cutest_problem."
				continue
			else
				runModelFromProblem(cutest_problem, crt, problem_data_vec[index], δ, print_level, total_results_output_file_path)
			end
		end

		df = DataFrame(CSV.File(total_results_output_file_path))
		df = filter(:problem_name => p_n -> p_n in cutest_problems, df)
		max_it = problem_data_vec[index][7]
		geomean_total_iterations_count, geomean_total_function_evaluation, geomean_total_gradient_evaluation, geomean_total_hessian_evaluation, geomean_count_factorization = computeGeomeans(df, max_it)
		counts = countmap(df.status)
		total_failure = length(df.status) - get(counts, "SUCCESS", 0) - get(counts, "OPTIMAL", 0)
		open(geomean_results_file_path, "a") do file
	    	write(file, "$crt,$total_failure,$geomean_total_iterations_count,$geomean_total_function_evaluation,$geomean_total_gradient_evaluation,$geomean_total_hessian_evaluation,$geomean_count_factorization\n");
	  	end
	end

	df_geomean_results = DataFrame(CSV.File(geomean_results_file_path))
	return df_geomean_results
end

function plotFigure(
	values_x,
	values_y,
	ylabel::String,
	xlabel::Matrix{String},
	folder_name::String,
	plot_name::String
	)
	plot(
		values_x,
		values_y,
      	color = :blue,
      	ylabel=ylabel,
		legend=false
		#size=(900, 500)
    )
    fullPath = string(folder_name, "/", plot_name)
	@show fullPath
    png(fullPath)
end

function plotFigures(
	df::DataFrame,
	folder_name::String,
	)

	criteria = df.criteria
	total_failure = df.total_failure
	geomean_total_iterations_count = df.geomean_total_iterations_count
	geomean_total_function_evaluation = df.geomean_total_function_evaluation
	geomean_total_gradient_evaluation = df.geomean_total_gradient_evaluation
	geomean_total_hessian_evaluation = df.geomean_total_hessian_evaluation
	geomean_count_factorization = df.geomean_count_factorization

	new_criteria = [String.("CAT")]
	for i in 2:length(criteria)
		push!(new_criteria, string("+", criteria[i]))
	end
	xlabel = hcat(new_criteria...)
	@show xlabel
	#Plot how fraction evaluation changes based on each criteria
	ylabel = "GEOMEAN for total # of function evaluations"
	plot_name = "ablation_study_geomean_fct_evaluations.png"
	plotFigure(new_criteria, geomean_total_function_evaluation, ylabel, xlabel, folder_name, plot_name)

	#Plot how gradient evaluation changes based on each criteria
	ylabel = "GEOMEAN for total # of gradient evaluations"
	plot_name = "ablation_study_geomean_grad_evaluations.png"
	plotFigure(new_criteria, geomean_total_gradient_evaluation, ylabel, xlabel, folder_name, plot_name)

	#Plot number of failures based on each criteria
	ylabel = "Total number of failures"
	plot_name = "ablation_study_total_number_failures.png"
	plotFigure(new_criteria, total_failure, ylabel, xlabel, folder_name, plot_name)
end

function main()
	parsed_args = parse_command_line()
	folder_name = parsed_args["output_dir"]

  	if_mkpath("$folder_name")
  	default_problems = parsed_args["default_problems"]
  	min_nvar = 0
  	max_nvar = 0
  	if !default_problems
    	min_nvar = parsed_args["min_nvar"]
    	max_nvar = parsed_args["max_nvar"]
  	end
  	max_it = parsed_args["max_it"]
  	max_time = parsed_args["max_time"]
  	tol_opt = parsed_args["tol_opt"]
  	r_1 = parsed_args["r_1"]

  	θ = parsed_args["θ"]
  	β_1 = parsed_args["β_1"]
  	β_2 = parsed_args["β_2"]
  	ω_1 = parsed_args["ω_1"]
  	ω_2 = parsed_args["ω_2"]
  	γ_2 = parsed_args["γ_2"]
  	δ = parsed_args["δ"]
	print_level = parsed_args["print_level"]

  	# default_criteria = ["ρ_hat_rule", "GALAHAD_TRS", "initial_radius", "radius_update_rule"]
	default_criteria = ["ρ_hat_rule", "initial_radius", "radius_update_rule"]
  	criteria = split(parsed_args["criteria"], ",")
  	for val in criteria
    	if val ∉ default_criteria
      		error("`criteria` allowed values are `ρ_hat_rule`, `GALAHAD_TRS`, `initial_radius`, `radius_update_rule`.")
    	end
  	end
	criteria = vcat("original", criteria)
	criteria = String.(criteria)
	# criteria = criteria[1:2]
	# criteria = criteria[3:4]
  	# run_cutest_with_CAT(folder_name, default_problems, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1, δ, min_nvar, max_nvar, train_batch_count, train_batch_index, parsed_args["solver"])
	problem_data_vec = createProblemData(criteria, max_it, max_time, tol_opt, θ, β_1, β_2, ω_1, ω_2, γ_2, r_1)
	@info criteria
	@info problem_data_vec
	df_geomean_results = runProblems(criteria, problem_data_vec, δ, folder_name, default_problems, min_nvar, max_nvar, print_level)
	plotFigures(df_geomean_results, folder_name)
end

main()
