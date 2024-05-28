
using JuMP, NLPModels, NLPModelsJuMP, LinearAlgebra, Optim, CUTEst, CSV, Test, DataFrames, SparseArrays, StatsBase, Random, Dates
include("../src/CAT.jl")
include("../src/tru.jl")
include("../src/arc.jl")

const optimization_method_CAT = "CAT"
const optimization_method_CAT_theta_0 = "CAT_THETA_ZERO"
const optimization_metnod_newton_trust_region = "NewtonTrustRegion"

const optimization_method_CAT_galahad_factorization = "CAT_GALAHAD_FACTORIZATION"
const optimization_method_CAT_galahad_iterative = "CAT_GALAHAD_ITERATIVE"
const optimization_method_arc_galahad = "ARC"
const optimization_method_tru_galahd_factorization = "TRU_GALAHAD_FACTORIZATION"
const optimization_method_tru_galahd_iterative = "TRU_GALAHAD_ITERATIVE"

# const skip_list = ["ARGLINB", "DIAMON2DLS", "DIAMON3DLS", "DMN15102LS", "DMN15103LS", "DMN15332LS", "DMN15333LS", "DMN37142LS", "DMN37143LS", "FLETCHCR", "MNISTS5LS"]
const skip_list = []

const default_train_problems = ["AKIVA", "ALLINITU", "ARGLINA", "ARGTRIGLS", "BA-L1LS", "BARD", "BEALE", "BENNETT5LS", "BIGGS6", "BOX3", "BOXBODLS", "BRKMCC", "BROWNAL", "BROWNBS", "BROWNDEN", "CERI651ALS", "CERI651BLS", "CERI651CLS", "CERI651DLS", "CERI651ELS", "CHNROSNB", "CHNRSNBM", "CLIFF", "CLUSTERLS", "COATING", "COOLHANSLS", "CUBE", "DANIWOODLS", "DANWOODLS", "DENSCHNA", "DENSCHNB", "DENSCHNC", "DENSCHND", "DENSCHNE", "DENSCHNF", "DEVGLA1", "DEVGLA2", "DJTL", "EG2", "EGGCRATE", "ELATVIDU", "ENGVAL2", "ENSOLS", "ERRINROS", "EXPFIT", "EXTROSNB", "FBRAIN3LS", "GAUSS2LS", "GAUSS3LS", "GAUSSIAN", "GBRAINLS", "GENROSE", "GROWTHLS", "HATFLDD", "HATFLDFL", "HATFLDFLS", "HATFLDGLS", "HEART6LS", "HEART8LS", "HELIX", "HIELOW", "HILBERTA", "HIMMELBB", "HIMMELBCLS", "HIMMELBG", "HIMMELBH", "HUMPS", "HYDC20LS", "HYDCAR6LS", "JENSMP", "JUDGE", "KIRBY2LS", "KOWOSB", "KSSLS", "LANCZOS1LS", "LANCZOS2LS", "LOGHAIRY", "LSC1LS", "LSC2LS", "LUKSAN11LS", "LUKSAN12LS", "LUKSAN13LS", "LUKSAN14LS", "LUKSAN15LS", "LUKSAN16LS", "LUKSAN22LS", "MANCINO", "MARATOSB", "METHANL8LS", "MEXHAT", "MEYER3", "MGH10LS", "MGH10SLS", "MGH17LS", "MISRA1BLS", "MISRA1CLS", "MISRA1DLS", "MNISTS0LS", "MUONSINELS", "NELSONLS", "OSBORNEA", "OSBORNEB", "PALMER1D", "PALMER3C", "PALMER5C", "PALMER5D", "PALMER7C", "PARKCH", "PENALTY1", "PENALTY2", "POWELLBSLS", "PRICE3", "PRICE4", "QING", "RAT42LS", "RAT43LS", "RECIPELS", "ROSENBR", "ROSZMAN1LS", "S308", "SENSORS", "SINEVAL", "SISSER", "SNAIL", "SPIN2LS", "SSI", "STRATEC", "STREG", "STRTCHDV", "TOINTQOR", "TRIGON1", "VANDANMSLS", "VESUVIOLS", "VESUVIOULS", "VIBRBEAM", "WATSON", "WAYSEA1", "YFITU", "ZANGWIL2"]

const default_test_problems = ["DIXMAANI", "LIARWHD", "SCHMVETT", "VAREIGVL", "CYCLOOCFLS", "DIXMAANJ", "SBRYBND", "ARGLINC", "TOINTGOR", "DIXMAANC", "WAYSEA2", "DMN37142LS", "EIGENALS", "YATP1LS", "GENHUMPS", "OSCIPATH", "FLETCBV2", "DIXMAAND", "S308NE", "SPMSRTLS", "NONCVXUN", "BRYBND", "DIXMAANM", "DQRTIC", "MISRA1ALS", "BOX", "DMN37143LS", "TOINTGSS", "SPARSINE", "VESUVIALS", "INTEQNELS", "COATINGNE", "HAIRY", "PALMER1C", "BA-L49LS", "SSCOSINE", "NONCVXU2", "BROYDN7D", "COSINE", "DIXMAANO", "SPINLS", "BROYDN3DLS", "PALMER2C", "CURLY20", "NONMSQRT", "CURLY30", "FREUROTH", "PALMER8C", "FMINSRF2", "YATP2CLS", "DMN15332LS", "SCURLY20", "NONDQUAR", "SCURLY30", "LUKSAN21LS", "ECKERLE4LS", "HAHN1LS", "DMN15102LS", "WOODS", "JIMACK", "HIMMELBF", "VARDIM", "BROYDNBDLS", "FLETBV3M", "DIXMAANA", "CHWIRUT2LS", "POWER", "DEVGLA2NE", "BA-L1SPLS", "BA-L73LS", "DIAMON2DLS", "THURBERLS", "GAUSS1LS", "PENALTY3", "MODBEALE", "PALMER6C", "DIXMAANN", "LUKSAN17LS", "EIGENCLS", "INDEFM", "SROSENBR", "MNISTS5LS", "INDEF", "ARWHEAD", "DIXON3DQ", "MGH09LS", "BDQRTIC", "DIXMAANH", "DIXMAANB", "BA-L21LS", "GULF", "POWELLSG", "TRIDIA", "DMN15333LS", "NCB20", "FLETCBV3", "CHAINWOO", "HATFLDE", "DIXMAANP", "FLETCHCR", "ERRINRSM", "FMINSURF", "DIXMAANE", "MSQRTALS", "CURLY10", "BOXPOWER", "DQDRTIC", "OSCIGRAD", "FLETCHBV", "ARGLINB", "DIXMAANF", "BA-L16LS", "MOREBV", "NONDIA", "MSQRTBLS", "SCURLY10", "TQUARTIC", "CHWIRUT1LS", "YATP2LS", "ENGVAL1", "DIXMAANG", "HILBERTB", "DIXMAANK", "QUARTC", "EDENSCH", "YATP1CLS", "TOINTPSP", "SSBRYBND", "CRAGGLVY", "SPARSQUR", "DMN15103LS", "NCB20B", "BA-L52LS", "POWERSUM", "ROSENBRTU", "SINQUAD", "DIXMAANL", "PALMER4C", "TESTQUAD", "EIGENBLS", "TRIGON2", "POWELLSQLS", "SCOSINE", "EXP2", "METHANB8LS", "LANCZOS3LS", "DIAMON3DLS"]

const NON_SUCCESS_STATUSES = ["FAILURE", "ITERATION_LIMIT", "INCOMPLETE", "LINRARY_STOP", "KILLED", "FAILURE_SMALL_RADIUS", "FAILURE_WRONG_PREDICTED_REDUCTION", "FAILURE_UNBOUNDED_OBJECTIVE"]

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

function get_problems_test_train_split(cutest_problems::Vector{String})
	train_problems = Vector{String}()
	test_problems = Vector{String}()
	for cutest_problem in cutest_problems
		if cutest_problem ∈ default_train_problems
			push!(train_problems, cutest_problem)
		else
			push!(test_problems, cutest_problem)
		end
	end
	return train_problems, test_problems
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
    if !default_problems
		cutest_problems = get_problem_list(min_nvar, max_nvar)
    else
		cutest_problems = CUTEst.select(contype="unc")
    end

	trust_region_method_subproblem_solver = optimization_method == optimization_method_CAT ? consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_DEFAULT : (optimization_method == optimization_method_CAT_galahad_factorization ? consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_TRS : consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_GLTR)
	if θ == 0.0
		optimization_method = optimization_method_CAT_theta_0
	end

	train_problems, test_problems = get_problems_test_train_split(cutest_problems)
    executeCUTEST_Models_benchmark("train", train_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω_1, ω_2, γ_1, γ_2, r_1, print_level, δ, trust_region_method_subproblem_solver)
	executeCUTEST_Models_benchmark("test", test_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω_1, ω_2, γ_1, γ_2, r_1, print_level, δ, trust_region_method_subproblem_solver)
end

function run_cutest_with_newton_trust_region(
    folder_name::String,
    default_problems::Bool,
    max_it::Int64,
    max_time::Float64,
    tol_opt::Float64,
    r_1::Float64,
    min_nvar::Int64,
    max_nvar::Int64,
	print_level::Int64
    )
	cutest_problems = []
    if !default_problems
		cutest_problems = get_problem_list(min_nvar, max_nvar)
    else
		cutest_problems = CUTEst.select(contype="unc")
    end

    optimization_method = optimization_metnod_newton_trust_region
	θ = β = ω_1 = ω_2 = γ_1 = γ_2 = 0.0

	train_problems, test_problems = get_problems_test_train_split(cutest_problems)
    executeCUTEST_Models_benchmark("train", train_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω_1, ω_2, γ_1, γ_2, r_1, print_level)
	executeCUTEST_Models_benchmark("test", test_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω_1, ω_2, γ_1, γ_2, r_1, print_level)
end

function run_cutest_with_arc(
    folder_name::String,
    default_problems::Bool,
    max_it::Int64,
    max_time::Float64,
    tol_opt::Float64,
    σ_1::Float64,
    min_nvar::Int64,
    max_nvar::Int64,
	print_level::Int64
    )
	cutest_problems = []
    if !default_problems
		cutest_problems = get_problem_list(min_nvar, max_nvar)
    else
		cutest_problems = CUTEst.select(contype="unc")
    end

    optimization_method = optimization_method_arc_galahad
	θ = β = ω_1 = ω_2 = γ_1 = γ_2 = 0.0

	train_problems, test_problems = get_problems_test_train_split(cutest_problems)
    executeCUTEST_Models_benchmark("train", train_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω_1, ω_2, γ_1, γ_2, σ_1, print_level)
    executeCUTEST_Models_benchmark("test", test_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω_1, ω_2, γ_1, γ_2, σ_1, print_level)
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
	print_level::Int64,
	optimization_method::String
    )
	cutest_problems = []
    if !default_problems
		cutest_problems = get_problem_list(min_nvar, max_nvar)
    else
		cutest_problems = CUTEst.select(contype="unc")
    end

    θ = β = ω_1 = ω_2 = γ_1 = γ_2 = 0.0

    train_problems, test_problems = get_problems_test_train_split(cutest_problems)
    executeCUTEST_Models_benchmark("train", train_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω_1, ω_2, γ_1, γ_2, r_1, print_level)
	executeCUTEST_Models_benchmark("test", test_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω_1, ω_2, γ_1, γ_2, r_1, print_level)
end

function runModelFromProblem(
	prefix::String, #Specify if we are running train or test problems
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
	start_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
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
			start_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
	        x, status, iteration_stats, computation_stats, total_iterations_count = consistently_adaptive_trust_region_method.CAT(problem, x_1, δ, trust_region_method_subproblem_solver)
			status_string = convertSsatusCodeToStatusString(status)
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
			directory_name = string(folder_name, "/", prefix, "_$optimization_method")
			# outputResultsToCSVFile(directory_name, cutest_problem, iteration_stats)
			total_number_factorizations = Int64(computation_stats_modified["total_number_factorizations"])
			outputIterationsStatusToCSVFile(start_time, end_time, directory_name, cutest_problem, status_string, computation_stats_modified, total_iterations_count, optimization_method, total_number_factorizations)
		elseif optimization_method == optimization_metnod_newton_trust_region
			d = Optim.TwiceDifferentiable(f, g!, h!, nlp.meta.x0)
			start_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
			results = optimize(d, nlp.meta.x0, Optim.NewtonTrustRegion(initial_delta=r_1), Optim.Options(show_trace=false, iterations = max_it, time_limit = max_time, g_abstol = tol_opt))
			end_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
			x = Optim.minimizer(results)
			total_iterations_count = Optim.iterations(results)
			total_function_evaluation = Optim.f_calls(results)
			total_gradient_evaluation = Optim.g_calls(results)
			total_hessian_evaluation = Optim.h_calls(results)
			function_value = obj(nlp, x)
			gradient_value = norm(grad(nlp, x), 2)
			computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "function_value" => function_value, "gradient_value" => gradient_value)
			status = results.g_converged ? "OPTIMAL" : (Optim.iteration_limit_reached(results) ? "ITERATION_LIMIT" : "FAILURE")
			if status == "ITERATION_LIMIT"
				total_iterations_count = max_it + 1
			end
			dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
			println("$dates_format------------------------MODEL SOLVED WITH STATUS: ", status)
			@info "$dates_format------------------------MODEL SOLVED WITH STATUS: $status"
			directory_name = string(folder_name, "/", prefix, "_$optimization_method")
			outputIterationsStatusToCSVFile(start_time, end_time, directory_name, cutest_problem, status, computation_stats, total_iterations_count, optimization_method)
		elseif optimization_method == optimization_method_arc_galahad
			initial_weight = r_1
			max_inner_iterations_or_factorizations = 10000
			start_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
			userdata, solution = arc(length(nlp.meta.x0), nlp.meta.x0, grad(nlp, nlp.meta.x0), print_level, max_it, initial_weight, max_inner_iterations_or_factorizations, max_time)
			end_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
			status = userdata.status == 0 ? "OPTIMAL" : userdata.status == -18 ? "ITERATION_LIMIT" : userdata.status == -19 ? "MAX_TIME" : "FAILURE"
			iter = max(userdata.iter, 1) #Safety check as ARC returns 0 iter count if already on the optimal solution
			total_iterations_count = iter
			total_function_evaluation = max(userdata.total_function_evaluation, 1)
			total_gradient_evaluation = max(userdata.total_gradient_evaluation, 1)
			total_hessian_evaluation = max(userdata.total_hessian_evaluation, 1)
			total_inner_iterations_or_factorizations = max(userdata.total_inner_iterations_or_factorizations, 1)
			function_value = obj(nlp, solution)
			gradient_value = norm(grad(nlp, solution), 2)
			if status == "OPTIMAL" && gradient_value > tol_opt
				status = "FAILURE"
			end
			computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "function_value" => function_value, "gradient_value" => gradient_value)
			dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
			println("$dates_format------------------------MODEL SOLVED WITH STATUS: ", status)
			@info "$dates_format------------------------MODEL SOLVED WITH STATUS: $status"
			directory_name = string(folder_name, "/", prefix, "_$optimization_method")
			outputIterationsStatusToCSVFile(start_time, end_time, directory_name, cutest_problem, status, computation_stats, total_iterations_count, optimization_method, total_inner_iterations_or_factorizations)
		elseif optimization_method == optimization_method_tru_galahd_factorization || optimization_method == optimization_method_tru_galahd_iterative
			subproblem_direct = optimization_method == optimization_method_tru_galahd_factorization ? true : false
			initial_x = nlp.meta.x0
			max_inner_iterations_or_factorizations = 10000
			start_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
			userdata, solution = tru(length(initial_x), initial_x, grad(nlp, initial_x), print_level, max_it, r_1, subproblem_direct, max_inner_iterations_or_factorizations, max_time)
			end_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
			status = userdata.status == 0 ? "OPTIMAL" : userdata.status == -18 ? "ITERATION_LIMIT" : userdata.status == -19 ? "MAX_TIME" : "FAILURE"
			iter = max(userdata.iter, 1) #Safety check as TRM returns 0 iter count if already on the optimal solution
			total_iterations_count = iter
			total_function_evaluation = max(userdata.total_function_evaluation, 1)
			total_gradient_evaluation = max(userdata.total_gradient_evaluation, 1)
			total_hessian_evaluation = max(userdata.total_hessian_evaluation, 1)
			total_inner_iterations_or_factorizations = max(userdata.total_inner_iterations_or_factorizations, 1)
			function_value = obj(nlp, solution)
			gradient_value = norm(grad(nlp, solution), 2)
			if status == "OPTIMAL" && gradient_value > tol_opt
				status = "FAILURE"
			end
			computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "function_value" => function_value, "gradient_value" => gradient_value)
			dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
			println("$dates_format------------------------MODEL SOLVED WITH STATUS: ", status)
			@info "$dates_format------------------------MODEL SOLVED WITH STATUS: $status"
			directory_name = string(folder_name, "/", prefix, "_$optimization_method")
			outputIterationsStatusToCSVFile(start_time, end_time, directory_name, cutest_problem, status, computation_stats, total_iterations_count, optimization_method, total_inner_iterations_or_factorizations)
		end
	catch e
		@show e
		status = "INCOMPLETE"
		computation_stats = Dict("total_function_evaluation" => max_it + 1, "total_gradient_evaluation" => max_it + 1, "total_hessian_evaluation" => max_it + 1, "function_value" => NaN, "gradient_value" => NaN)
		dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
		println("$dates_format------------------------MODEL SOLVED WITH STATUS: ", status)
		@info "$dates_format------------------------MODEL SOLVED WITH STATUS: $status"
		directory_name = string(folder_name, "/", prefix, "_$optimization_method")
		end_time = dates_format
		outputIterationsStatusToCSVFile(start_time, end_time, directory_name, cutest_problem, status, computation_stats, max_it + 1, optimization_method, max_it + 1)
        finally
            if nlp != nothing
                finalize(nlp)
        end
    end
end

function executeCUTEST_Models_benchmark(
	prefix::String, #Specify if we are running train or test problems
	cutest_problems::Vector{String},
	folder_name::String,
	optimization_method::String,
	max_it::Int64=10000,
    max_time::Float64=30*60,
    tol_opt::Float64=1e-5,
    θ::Float64=0.1,
	β::Float64=0.1,
	ω_1::Float64=4.0,
	ω_2::Float64=20.0,
	γ_1::Float64=0.01,
    γ_2::Float64=0.2,
    r_1::Float64=1.0,
	print_level::Int64=0,
	δ::Float64=0.0,
	trust_region_method_subproblem_solver::String=consistently_adaptive_trust_region_method.OPTIMIZATION_METHOD_DEFAULT
	)
	println("CUTEst Problems are: $cutest_problems")
	geomean_results_file_path = string(folder_name, "/", prefix, "_geomean_total_results.csv")

	if !isfile(geomean_results_file_path)
		open(geomean_results_file_path, "w") do file
			write(file, "optimization_method,total_failure,geomean_total_iterations_count,geomean_total_function_evaluation,geomean_total_gradient_evaluation,geomean_total_hessian_evaluation,geomean_count_factorization\n");
		end
	end

	total_results_output_directory =  string(folder_name, "/", prefix, "_$optimization_method")
	total_results_output_file_name = "table_cutest_$optimization_method.csv"
	total_results_output_file_path = string(total_results_output_directory, "/", total_results_output_file_name)
	if !isfile(total_results_output_file_path)
		mkpath(total_results_output_directory);
		open(total_results_output_file_path,"a") do iteration_status_csv_file
			write(iteration_status_csv_file, "start_time,end_time,problem_name,status,total_iterations_count,function_value,gradient_value,total_function_evaluation,total_gradient_evaluation,total_hessian_evaluation,total_factorization_evaluation\n");
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
        	runModelFromProblem(prefix, problem, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω_1, ω_2, γ_1, γ_2, r_1, δ, print_level, trust_region_method_subproblem_solver)
		end
    end
	df = DataFrame(CSV.File(total_results_output_file_path))
	df = filter(:problem_name => p_n -> p_n in cutest_problems, df)

	@show "Computing Normal Geometric Means"
	shift = 0
	geomean_total_iterations_count, geomean_total_function_evaluation, geomean_total_gradient_evaluation, geomean_total_hessian_evaluation, geomean_count_factorization = computeNormalGeomeans(df, shift, tol_opt, max_time / 3600, max_it)

	counts = countmap(df.status)
	total_failure = length(df.status) - get(counts, "SUCCESS", 0) - get(counts, "OPTIMAL", 0)
	open(geomean_results_file_path, "a") do file
		write(file, "$optimization_method,$total_failure,$geomean_total_iterations_count,$geomean_total_function_evaluation,$geomean_total_gradient_evaluation,$geomean_total_hessian_evaluation,$geomean_count_factorization\n");
	end

	# shift = 10
	# @show "Computing Shifted Geometric Means with Shift = 10"
	# computeShiftedGeomeans(df, shift, tol_opt, max_time / 3600, max_it)
	#
	# @show "Computing Shifted & Corrected Geomeans with ϕ = θ^{3/2} shift = 10"
	# ϕ(θ) = θ ^ (3/ 2)
	# computeShiftedAndCorrectedGeomeans(ϕ, df, shift, tol_opt, max_time / 3600, max_it)
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
	total_iterations_count_vec = Vector{Float64}()
	total_factorization_count_vec = Vector{Float64}()
	total_function_evaluation_vec = Vector{Float64}()
	total_gradient_evaluation_vec = Vector{Float64}()
	total_hessian_evaluation_vec = Vector{Float64}()
	for i in 1:size(df)[1]
		if df[i, :].status == "SUCCESS" || df[i, :].status == "OPTIMAL"
			push!(total_iterations_count_vec, df[i, :].total_iterations_count)
			push!(total_factorization_count_vec, df[i, :].total_factorization_evaluation)
			push!(total_function_evaluation_vec, df[i, :].total_function_evaluation)
			push!(total_gradient_evaluation_vec, df[i, :].total_gradient_evaluation)
			push!(total_hessian_evaluation_vec, df[i, :].total_hessian_evaluation)
		elseif df[i, :].status == "MAX_TIME"
			temp_ = ϕ(df[i, :].gradient_value / ϵ) * time_limit
			push!(total_iterations_count_vec, temp_)
			push!(total_factorization_count_vec, temp_ == max_it + 1 ? max(temp_, df[i, :].total_factorization_evaluation) : temp_)
			push!(total_function_evaluation_vec, temp_)
			push!(total_gradient_evaluation_vec, temp_)
			push!(total_hessian_evaluation_vec, temp_)
		else
			push!(total_iterations_count_vec, max_it + 1)
			push!(total_factorization_count_vec, max(df[i, :].total_factorization_evaluation, max_it + 1))
			push!(total_function_evaluation_vec, max_it + 1)
			push!(total_gradient_evaluation_vec, max_it + 1)
			push!(total_hessian_evaluation_vec, max_it + 1)
		end
	end

	df_results_new = DataFrame()
	df_results_new.problem_name = df.problem_name
	df_results_new.total_iterations_count = total_iterations_count_vec
	df_results_new.total_factorization_evaluation = total_factorization_count_vec
	df_results_new.total_function_evaluation = total_function_evaluation_vec
	df_results_new.total_gradient_evaluation = total_gradient_evaluation_vec
	df_results_new.total_hessian_evaluation = total_hessian_evaluation_vec

	return computeShiftedGeomeans(df_results_new, shift)
end

function computeShiftedGeomeans(df::DataFrame, shift::Int64)
	geomean_total_iterations_count = geomean(df.total_iterations_count .+ shift) - shift
	geomean_count_factorization = geomean(df.total_factorization_evaluation .+ shift) - shift
	geomean_total_function_evaluation = geomean(df.total_function_evaluation .+ shift) - shift
	geomean_total_gradient_evaluation = geomean(df.total_gradient_evaluation .+ shift) - shift
	geomean_total_hessian_evaluation  = geomean(df.total_hessian_evaluation .+ shift) - shift
	if shift == 0
		@info geomean_total_iterations_count
		@info geomean_count_factorization
		@info geomean_total_function_evaluation
		@info geomean_total_gradient_evaluation
		@info geomean_total_hessian_evaluation
	end
	@show geomean_total_iterations_count
	@show geomean_count_factorization
	@show geomean_total_function_evaluation
	@show geomean_total_gradient_evaluation
	@show geomean_total_hessian_evaluation

	return (geomean_total_iterations_count, geomean_total_function_evaluation, geomean_total_gradient_evaluation, geomean_total_hessian_evaluation, geomean_count_factorization)
end

function outputResultsToCSVFile(directory_name::String, cutest_problem::String, results::DataFrame)
	cutest_problem_file_name = string(directory_name, "/$cutest_problem.csv")
    CSV.write(cutest_problem_file_name, results, header = true)
end

function convertSsatusCodeToStatusString(status)
    dict_status_code = Dict(consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL => "OPTIMAL",
    consistently_adaptive_trust_region_method.TerminationStatusCode.UNBOUNDED => "UNBOUNDED",
    consistently_adaptive_trust_region_method.TerminationStatusCode.ITERATION_LIMIT => "ITERATION_LIMIT",
    consistently_adaptive_trust_region_method.TerminationStatusCode.TIME_LIMIT => "TIME_LIMIT",
    consistently_adaptive_trust_region_method.TerminationStatusCode.MEMORY_LIMIT => "MEMORY_LIMIT",
    consistently_adaptive_trust_region_method.TerminationStatusCode.TRUST_REGION_RADIUS_LIMIT => "TRUST_REGION_RADIUS_LIMIT",
    consistently_adaptive_trust_region_method.TerminationStatusCode.NUMERICAL_ERROR => "NUMERICAL_ERROR",
    consistently_adaptive_trust_region_method.TerminationStatusCode.OTHER_ERROR => "OTHER_ERROR")
    return dict_status_code[status]
end

function outputIterationsStatusToCSVFile(
	start_time::String,
	end_time::String,
	directory_name::String,
	cutest_problem::String,
	status::String,
	computation_stats::Dict,
	total_iterations_count::Integer,
	optimization_method::String,
	count_factorization::Integer=0
	)
    total_function_evaluation = Int(computation_stats["total_function_evaluation"])
    total_gradient_evaluation = Int(computation_stats["total_gradient_evaluation"])
    total_hessian_evaluation  = Int(computation_stats["total_hessian_evaluation"])

    function_value = computation_stats["function_value"]
    gradient_value = computation_stats["gradient_value"]
	file_name = string(directory_name, "/", "table_cutest_$optimization_method.csv")
    open(file_name,"a") do iteration_status_csv_file
		write(iteration_status_csv_file, "$start_time,$end_time,$cutest_problem,$status,$total_iterations_count,$function_value,$gradient_value,$total_function_evaluation,$total_gradient_evaluation,$total_hessian_evaluation,$count_factorization\n")
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
