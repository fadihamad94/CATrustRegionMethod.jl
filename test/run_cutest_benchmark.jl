using JuMP, NLPModels, NLPModelsJuMP, LinearAlgebra, Optim, CUTEst, CSV, Test, DataFrames, SparseArrays, StatsBase, Random
include("../src/CAT.jl")
include("../src/tru.jl")
include("../src/arc.jl")

#const problems_paper_list =  ["ALLINITU", "ARGLINA", "BARD", "BEALE", "BIGGS6", "BOX3", "BRKMCC", "BROWNAL", "BROWNBS", "BROWNDEN", "CHNROSNB", "CLIFF", "CUBE", "DENSCHNA", "DENSCHNB", "DENSCHNC", "DENSCHND", "DENSCHNE", "DENSCHNF", "DJTL", "ENGVAL2", "ERRINROS", "EXPFIT", "GENROSEB", "GROWTHLS", "GULF", "HAIRY", "HATFLDD", "HATFLDE", "HEART6LS", "HEART8LS", "HELIX", "HIMMELBB", "HUMPS", "HYDC20LS", "JENSMP", "KOWOSB", "LOGHAIRY", "MANCINO", "MEXHAT", "MEYER3", "OSBORNEA", "OSBORNEB", "PALMER5C", "PALMER6C", "PALMER7C", "PALMER8C", "PARKCH", "PENALTY2", "PENALTY3", "PFIT1LS", "PFIT2LS", "PFIT3LS", "PFIT4LS", "ROSENBR", "S308", "SENSORS", "SINEVAL", "SISSER", "SNAIL", "STREG", "TOINTGOR", "TOINTPSP", "VARDIM", "VIBRBEAM", "WATSON", "YFITU"]
#const problems_paper_list =  ["DIXMAANI", "LIARWHD", "SCHMVETT", "LUKSAN13LS", "VAREIGVL", "JUDGE", "CYCLOOCFLS", "DIXMAANJ", "FBRAIN3LS", "SPIN2LS", "SBRYBND", "ARGLINC", "TOINTGOR", "DIXMAANC", "WAYSEA2", "BROWNDEN", "HILBERTA", "DMN37142LS", "PALMER5D", "BOXBODLS", "HIMMELBB", "ENGVAL2", "MUONSINELS", "ENSOLS", "PRICE4", "EIGENALS", "YATP1LS", "CERI651ELS", "GENHUMPS", "OSCIPATH", "FLETCBV2", "DIXMAAND", "SISSER", "TRIGON1", "MGH17SLS", "PENALTY1", "SPMSRTLS", "NONCVXUN", "BRYBND", "DIXMAANM", "GROWTHLS", "SINEVAL", "GAUSS2LS", "STRATEC", "NELSONLS", "HYDCAR6LS", "DQRTIC", "MISRA1ALS", "WAYSEA1", "BOX", "DMN37143LS", "TOINTGSS", "GAUSS3LS", "SPARSINE", "VESUVIALS", "INTEQNELS", "HAIRY", "YFITU", "CHNRSNBM", "HIMMELBCLS", "CYCLIC3LS", "MARATOSB", "LSC2LS", "PALMER1C", "BA-L49LS", "SSCOSINE", "POWELLBSLS", "NONCVXU2", "HIMMELBG", "BROYDN7D", "COSINE", "DIXMAANO", "DENSCHNF", "COOLHANSLS", "PRICE3", "VESUVIOULS", "KOWOSB", "LUKSAN12LS", "SPINLS", "HIMMELBH", "ZANGWIL2", "BROYDN3DLS", "PALMER2C", "HEART8LS", "CURLY20", "VANDANMSLS", "HATFLDD", "NONMSQRT", "MISRA1DLS", "BRKMCC", "CURLY30", "FREUROTH", "PALMER8C", "FMINSRF2", "DENSCHNA", "YATP2CLS", "DMN15332LS", "METHANL8LS", "SCURLY20", "MISRA1BLS", "DENSCHNC", "NONDQUAR", "S308", "SNAIL", "SCURLY30", "LUKSAN21LS", "MANCINO", "EXPFIT", "BOX3", "ECKERLE4LS", "HAHN1LS", "MGH17LS", "LUKSAN22LS", "DMN15102LS", "PALMER1D", "WOODS", "JIMACK", "HIMMELBF", "VARDIM", "JENSMP", "CERI651DLS", "BROYDNBDLS", "GBRAINLS", "FLETBV3M", "DIXMAANA", "CHWIRUT2LS", "POWER", "PENALTY2", "BA-L1SPLS", "BA-L73LS", "ALLINITU", "VESUVIOLS", "DIAMON2DLS", "THURBERLS", "CERI651ALS", "VIBRBEAM", "GAUSS1LS", "PENALTY3", "DJTL", "LSC1LS", "MODBEALE", "PALMER6C", "DIXMAANN", "CERI651CLS", "LUKSAN17LS", "PALMER5C", "EIGENCLS", "INDEFM", "OSBORNEA", "BIGGS6", "PALMER7C", "BEALE", "SROSENBR", "MNISTS5LS", "MGH10SLS", "ARGLINA", "INDEF", "SENSORS", "ARWHEAD", "RAT43LS", "CLUSTERLS", "HELIX", "DIXON3DQ", "MEXHAT", "DENSCHNB", "MNISTS0LS", "MGH09LS", "BDQRTIC", "DIXMAANH", "DIXMAANB", "MEYER3", "BA-L21LS", "GULF", "POWELLSG", "TRIDIA", "DMN15333LS", "DENSCHND", "BROWNBS", "SSI", "NCB20", "FLETCBV3", "KSSLS", "CHAINWOO", "HATFLDE", "LUKSAN11LS", "KIRBY2LS", "LUKSAN16LS", "DIXMAANP", "COATING", "FLETCHCR", "ERRINRSM", "FMINSURF", "DIXMAANE", "BENNETT5LS", "MSQRTALS", "CURLY10", "DANWOODLS", "ARGTRIGLS", "BOXPOWER", "DANIWOODLS", "RAT42LS", "DQDRTIC", "DEVGLA1", "HATFLDFL", "OSCIGRAD", "STREG", "FLETCHBV", "AKIVA", "QING", "LANCZOS2LS", "ARGLINB", "DIXMAANF", "BA-L16LS", "PARKCH", "MOREBV", "ROSENBR", "NONDIA", "HYDC20LS", "HATFLDFLS", "ERRINROS", "LUKSAN14LS", "HIELOW", "MSQRTBLS", "BROWNAL", "HUMPS", "BARD", "HATFLDGLS", "SCURLY10", "MGH10LS", "TQUARTIC", "EXTROSNB", "DEVGLA2", "CHWIRUT1LS", "YATP2LS", "ENGVAL1", "LUKSAN15LS", "DIXMAANG", "EGGCRATE", "HILBERTB", "BA-L1LS", "DIXMAANK", "QUARTC", "RECIPELS", "EDENSCH", "CHNROSNB", "YATP1CLS", "TOINTPSP", "LANCZOS1LS", "PALMER3C", "SSBRYBND", "ELATVIDU", "CRAGGLVY", "SPARSQUR", "DMN15103LS", "TOINTQOR", "ROSZMAN1LS", "NCB20B", "BA-L52LS", "POWERSUM", "ROSENBRTU", "WATSON", "GAUSSIAN", "SINQUAD", "EG2", "DIXMAANL", "GENROSE", "PALMER4C", "TESTQUAD", "EIGENBLS", "MISRA1CLS", "DENSCHNE", "OSBORNEB", "CLIFF", "STRTCHDV", "TRIGON2", "HEART6LS", "POWELLSQLS", "SCOSINE", "EXP2", "METHANB8LS", "LANCZOS3LS", "DIAMON3DLS", "LOGHAIRY", "CERI651BLS", "CUBE"]

const problems_paper_list = CUTEst.select(max_con=0, only_free_var=true)

const optimization_method_CAT = "CAT"
const optimization_method_CAT_theta_0 = "CAT_THETA_ZERO"
const optimization_metnod_newton_trust_region = "NewtonTrustRegion"

const optimization_method_CAT_galahad_factorization = "CAT_GALAHAD_FACTORIZATION"
const optimization_method_CAT_galahad_iterative = "CAT_GALAHAD_ITERATIVE"
const optimization_method_arc_galahad = "ARC"
const optimization_method_tru_galahd_factorization = "TRU_GALAHAD_FACTORIZATION"
const optimization_method_tru_galahd_iterative = "TRU_GALAHAD_ITERATIVE"

const train_test_split = 0.8

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
	train_batch_count::Int64,
	train_batch_index::Int64,
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
    number_of_problems = length(cutest_problems)
    Random.seed!(0)
    cutest_problem_indixes = collect(1:number_of_problems)
    cutest_problem_indixes = shuffle(cutest_problem_indixes)
    train_batch_size = floor(Int, train_test_split * length(cutest_problem_indixes) / train_batch_count)
    start_index = (train_batch_index - 1) * train_batch_size + 1
    end_index = train_batch_index == train_batch_count ? floor(Int, train_test_split * number_of_problems) : end_index
    train_cutest_problems = cutest_problems[cutest_problem_indixes[1:floor(Int, train_test_split * number_of_problems) + 1]][start_index:end_index]
    #train_cutest_problems = cutest_problems
    executeCUTEST_Models_benchmark(train_cutest_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1, δ, trust_region_method_subproblem_solver)
    #test_cutest_problems = cutest_problems[cutest_problem_indixes[Int(round(train_test_split * number_of_problems)) + 1 : number_of_problems]]
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
    max_nvar::Int64,
	train_batch_count::Int64,
	train_batch_index::Int64
    )
	cutest_problems = problems_paper_list
    if !default_problems
        cutest_problems = get_problem_list(min_nvar, max_nvar)
    end
    optimization_method = optimization_method_arc_galahad
	θ = β = ω = γ_2 = 0.0
	Random.seed!(0)
	number_of_problems = length(cutest_problems)
	cutest_problem_indixes = collect(1:number_of_problems)
    cutest_problem_indixes = shuffle(cutest_problem_indixes)
	train_batch_size = floor(Int, train_test_split * length(cutest_problem_indixes) / train_batch_count)
	start_index = (train_batch_index - 1) * train_batch_size + 1
	end_index = train_batch_index * train_batch_size
    train_cutest_problems = cutest_problems[cutest_problem_indixes[1:floor(Int, train_test_split * number_of_problems) + 1]][start_index:end_index]
	executeCUTEST_Models_benchmark(train_cutest_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, σ_1)
	#test_cutest_problems = cutest_problems[cutest_problem_indixes[Int(round(train_test_split * number_of_problems)) + 1 : number_of_problems]]
	# executeCUTEST_Models_benchmark(test_cutest_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, σ_1)
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
	train_batch_count::Int64,
	train_batch_index::Int64,
	optimization_method::String
    )
	cutest_problems = problems_paper_list
    if !default_problems
        cutest_problems = get_problem_list(min_nvar, max_nvar)
    end
    θ = β = ω = γ_2 = 0.0

    Random.seed!(0)
    number_of_problems = length(cutest_problems)
    cutest_problem_indixes = collect(1:number_of_problems)
    cutest_problem_indixes = shuffle(cutest_problem_indixes)
	train_batch_size = floor(Int, train_test_split * length(cutest_problem_indixes) / train_batch_count)
	start_index = (train_batch_index - 1) * train_batch_size + 1
	end_index = train_batch_index * train_batch_size
    train_cutest_problems = cutest_problems[cutest_problem_indixes[1:floor(Int, train_test_split * number_of_problems) + 1]][start_index:end_index]
    executeCUTEST_Models_benchmark(train_cutest_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1)
    #test_cutest_problems = cutest_problems[cutest_problem_indixes[Int(round(train_test_split * number_of_problems)) + 1 : number_of_problems]]
    #executeCUTEST_Models_benchmark(test_cutest_problems, folder_name, optimization_method, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1)
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
			gradient_value = norm(grad(nlp, x), 2)
			computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "function_value" => function_value, "gradient_value" => gradient_value)
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
			max_inner_iterations_or_factorizations = 10000
			userdata, solution = arc(length(nlp.meta.x0), nlp.meta.x0, grad(nlp, nlp.meta.x0), print_level, max_it, initial_weight, max_inner_iterations_or_factorizations, max_time)
			status = userdata.status == 0 ? "OPTIMAL" : userdata.status == -18 ? "ITERATION_LIMIT" : userdata.status == -19 ? "MAX_TIME" : "FAILURE"
			iter = userdata.iter
			total_iterations_count = iter
			total_function_evaluation = userdata.total_function_evaluation
			total_gradient_evaluation = userdata.total_gradient_evaluation
			total_hessian_evaluation = userdata.total_hessian_evaluation
			total_inner_iterations_or_factorizations = userdata.total_inner_iterations_or_factorizations
			function_value = obj(nlp, solution)
			gradient_value = norm(grad(nlp, solution), 2)
			if userdata.status != 0 || gradient_value > tol_opt
				iter = max_it + 1
				total_iterations_count = iter
				total_function_evaluation = max_it + 1
				total_gradient_evaluation = max_it + 1
				total_hessian_evaluation = max_it + 1
			end
			computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "function_value" => function_value, "gradient_value" => gradient_value)
			println("------------------------MODEL SOLVED WITH STATUS: ", status)
			directory_name = string(folder_name, "/", "$optimization_method")
			outputIterationsStatusToCSVFile(directory_name, cutest_problem, status, computation_stats, total_iterations_count, optimization_method, total_inner_iterations_or_factorizations)
		elseif optimization_method == optimization_method_tru_galahd_factorization || optimization_method == optimization_method_tru_galahd_iterative
			subproblem_direct = optimization_method == optimization_method_tru_galahd_factorization ? true : false
			initial_x = nlp.meta.x0
			print_level = 0
			max_inner_iterations_or_factorizations = 10000
			userdata, solution = tru(length(initial_x), initial_x, grad(nlp, initial_x), print_level, max_it, r_1, subproblem_direct, max_inner_iterations_or_factorizations, max_time)
			status = userdata.status == 0 ? "OPTIMAL" : userdata.status == -18 ? "ITERATION_LIMIT" : userdata.status == -19 ? "MAX_TIME" : "FAILURE"
			iter = userdata.iter
			total_iterations_count = iter
			total_function_evaluation = userdata.total_function_evaluation
			total_gradient_evaluation = userdata.total_gradient_evaluation
			total_hessian_evaluation = userdata.total_hessian_evaluation
			total_inner_iterations_or_factorizations = userdata.total_inner_iterations_or_factorizations
			function_value = obj(nlp, solution)
			gradient_value = norm(grad(nlp, solution), 2)
			#=
			@show status
			@show total_iterations_count
			@show total_function_evaluation
			@show total_gradient_evaluation
			@show total_hessian_evaluation
			@show gradient_value > tol_opt
			@show gradient_value
			@show tol_opt
			=#
			if userdata.status != 0 || gradient_value > tol_opt
				iter = max_it + 1
				total_iterations_count = iter
				total_function_evaluation = max_it + 1
				total_gradient_evaluation = max_it + 1
				total_hessian_evaluation = max_it + 1
			end
			computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation, "function_value" => function_value, "gradient_value" => gradient_value)
			println("------------------------MODEL SOLVED WITH STATUS: ", status)
			directory_name = string(folder_name, "/", "$optimization_method")
			outputIterationsStatusToCSVFile(directory_name, cutest_problem, status, computation_stats, total_iterations_count, optimization_method, total_inner_iterations_or_factorizations)
		end
	catch e
		computation_stats = Dict("total_function_evaluation" => max_it + 1, "total_gradient_evaluation" => max_it + 1, "total_hessian_evaluation" => max_it + 1, "function_value" => NaN, "gradient_value" => NaN)
		println("------------------------MODEL SOLVED WITH STATUS: ", status)
		directory_name = string(folder_name, "/", "$optimization_method")
		outputIterationsStatusToCSVFile(directory_name, cutest_problem, "INCOMPLETE", computation_stats, total_iterations_count, optimization_method, max_it + 1)
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

	total_results_output_directory =  string(folder_name, "/$optimization_method")
	total_results_output_file_name = "table_cutest_$optimization_method.csv"
	total_results_output_file_path = string(total_results_output_directory, "/", total_results_output_file_name)
	if !isfile(total_results_output_file_path)
		mkpath(total_results_output_directory);
			open(total_results_output_file_path,"a") do iteration_status_csv_file
			write(iteration_status_csv_file, "problem_name,status,total_iterations_count,function_value,gradient_value,total_function_evaluation,total_gradient_evaluation,total_hessian_evaluation,count_factorization\n");
    		end
	end

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
	df = filter(:problem_name => p_n -> p_n in cutest_problems, df)

	@show "Computing Normal Geometric Means"
	shift = 0
	computeNormalGeomeans(df, shift, tol_opt, max_time / 3600, max_it)

	shift = 10
	@show "Computing Shifted Geometric Means with Shift = 10"
	computeShiftedGeomeans(df, shift, tol_opt, max_time / 3600, max_it)

	@show "Computing Shifted & Corrected Geomeans with ϕ = θ^{3/2} shift = 10"
	ϕ(θ) = θ ^ (3/ 2)
	computeShiftedAndCorrectedGeomeans(ϕ, df, shift, tol_opt, max_time / 3600, max_it)
end

function computeNormalGeomeans(df::DataFrame, shift::Int64, ϵ::Float64, time_limit::Float64, max_it::Int64)
	ϕ(θ) = max_it/time_limit
	computeShiftedAndCorrectedGeomeans(ϕ, df, shift, ϵ, time_limit, max_it)
end

function computeShiftedGeomeans(df::DataFrame, shift::Int64, ϵ::Float64, time_limit::Float64, max_it::Int64)
	ϕ(θ) = max_it/time_limit
	computeShiftedAndCorrectedGeomeans(ϕ, df, shift, ϵ, time_limit, max_it)
end

function computeShiftedAndCorrectedGeomeans(ϕ::Function, df::DataFrame, shift::Int64, ϵ::Float64, time_limit::Float64, max_it::Int64)
	total_iterations_count_vec = Vector{Float64}()
	total_factorization_count_vec = Vector{Float64}()
	total_function_evaluation_vec = Vector{Float64}()
	total_gradient_evaluation_vec = Vector{Float64}()
	total_hessian_evaluation_vec = Vector{Float64}()
	non_success_statuses = ["FAILURE", "ITERARION_LIMIT", "INCOMPLETE"]
	for i in 1:size(df)[1]
		if df[i, :].status == "SUCCESS"
			push!(total_iterations_count_vec, df[i, :].total_iterations_count)
			push!(total_factorization_count_vec, df[i, :].count_factorization)
			push!(total_function_evaluation_vec, df[i, :].total_function_evaluation)
			push!(total_gradient_evaluation_vec, df[i, :].total_gradient_evaluation)
			push!(total_hessian_evaluation_vec, df[i, :].total_hessian_evaluation)
		elseif df[i, :].status ∈ non_success_statuses
			push!(total_iterations_count_vec, max_it)
			push!(total_factorization_count_vec, max_it)
			push!(total_function_evaluation_vec, max_it)
			push!(total_gradient_evaluation_vec, max_it)
			push!(total_hessian_evaluation_vec, max_it)
		else
			temp_ = ϕ(df[i, :].gradient_value / ϵ) * time_limit
			push!(total_iterations_count_vec, temp_)
			push!(total_factorization_count_vec, temp_)
			push!(total_function_evaluation_vec, temp_)
			push!(total_gradient_evaluation_vec, temp_)
			push!(total_hessian_evaluation_vec, temp_)
		end
	end

	df_results_new = DataFrame()
	df_results_new.problem_name = df.problem_name
	df_results_new.total_iterations_count = total_iterations_count_vec
	df_results_new.count_factorization = total_factorization_count_vec
	df_results_new.total_function_evaluation = total_function_evaluation_vec
	df_results_new.total_gradient_evaluation = total_gradient_evaluation_vec
	df_results_new.total_hessian_evaluation = total_hessian_evaluation_vec

	computeShiftedGeomeans(df_results_new, shift)
end

function computeShiftedGeomeans(df::DataFrame, shift::Int64)
	geomean_total_iterations_count = geomean(df.total_iterations_count .+ shift) - shift
	geomean_count_factorization = geomean(df.count_factorization .+ shift) - shift
	geomean_total_function_evaluation = geomean(df.total_function_evaluation .+ shift) - shift
	geomean_total_gradient_evaluation = geomean(df.total_gradient_evaluation .+ shift) - shift
	geomean_total_hessian_evaluation  = geomean(df.total_hessian_evaluation .+ shift) - shift

	@show geomean_total_iterations_count
	@show geomean_count_factorization
	@show geomean_total_function_evaluation
	@show geomean_total_gradient_evaluation
	@show geomean_total_hessian_evaluation
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

function outputIterationsStatusToCSVFile(
	directory_name::String,
	cutest_problem::String,
	status::String,
	computation_stats::Dict,
	total_iterations_count::Int64,
	optimization_method::String,
	count_factorization::Int32=0
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
	total_iterations_count::Int32,
	optimization_method::String,
	count_factorization::Int32=0
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
