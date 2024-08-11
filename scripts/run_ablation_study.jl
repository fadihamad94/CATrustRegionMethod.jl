import ArgParse
using JuMP, CUTEst, CSV, DataFrames, StatsBase, Dates, Statistics

using Random
include("../src/CAT_Module.jl")

"""
Defines parses and args.
# Returns
A dictionary with the values of the command-line arguments.
"""

const skip_list = ["YATP1LS", "YATP2CLS", "YATP2LS", "YATP1CLS"]

const default_problems_list = [
    "ARGLINA",
    "ARGLINB",
    "ARGLINC",
    "ARGTRIGLS",
    "ARWHEAD",
    "BA-L16LS",
    "BA-L21LS",
    "BA-L49LS",
    "BA-L52LS",
    "BA-L73LS",
    "BDQRTIC",
    "BOX",
    "BOXPOWER",
    "BROWNAL",
    "BROYDN3DLS",
    "BROYDN7D",
    "BROYDNBDLS",
    "BRYBND",
    "CHAINWOO",
    "COATING",
    "COATINGNE",
    "COSINE",
    "CRAGGLVY",
    "CURLY10",
    "CURLY20",
    "CURLY30",
    "CYCLOOCFLS",
    "DIXMAANA",
    "DIXMAANB",
    "DIXMAANC",
    "DIXMAAND",
    "DIXMAANE",
    "DIXMAANF",
    "DIXMAANG",
    "DIXMAANH",
    "DIXMAANI",
    "DIXMAANJ",
    "DIXMAANK",
    "DIXMAANL",
    "DIXMAANM",
    "DIXMAANN",
    "DIXMAANO",
    "DIXMAANP",
    "DIXON3DQ",
    "DQDRTIC",
    "DQRTIC",
    "EDENSCH",
    "EG2",
    "EIGENALS",
    "EIGENBLS",
    "EIGENCLS",
    "ENGVAL1",
    "EXTROSNB",
    "FLETBV3M",
    "FLETCBV2",
    "FLETCBV3",
    "FLETCHBV",
    "FLETCHCR",
    "FMINSRF2",
    "FMINSURF",
    "FREUROTH",
    "GENHUMPS",
    "GENROSE",
    "INDEF",
    "INDEFM",
    "INTEQNELS",
    "JIMACK",
    "KSSLS",
    "LIARWHD",
    "LUKSAN11LS",
    "LUKSAN15LS",
    "LUKSAN16LS",
    "LUKSAN17LS",
    "LUKSAN21LS",
    "LUKSAN22LS",
    "MANCINO",
    "MNISTS0LS",
    "MNISTS5LS",
    "MODBEALE",
    "MOREBV",
    "MSQRTALS",
    "MSQRTBLS",
    "NCB20",
    "NCB20B",
    "NONCVXU2",
    "NONCVXUN",
    "NONDIA",
    "NONDQUAR",
    "NONMSQRT",
    "OSCIGRAD",
    "OSCIPATH",
    "PENALTY1",
    "PENALTY2",
    "PENALTY3",
    "POWELLSG",
    "POWER",
    "QING",
    "QUARTC",
    "SBRYBND",
    "SCHMVETT",
    "SCOSINE",
    "SCURLY10",
    "SCURLY20",
    "SCURLY30",
    "SENSORS",
    "SINQUAD",
    "SPARSINE",
    "SPARSQUR",
    "SPIN2LS",
    "SPINLS",
    "SPMSRTLS",
    "SROSENBR",
    "SSBRYBND",
    "SSCOSINE",
    "TESTQUAD",
    "TOINTGSS",
    "TQUARTIC",
    "TRIDIA",
    "VARDIM",
    "VAREIGVL",
    "WOODS",
    "YATP1CLS",
    "YATP1LS",
    "YATP2CLS",
    "YATP2LS",
]

function if_mkpath(dir::String)
    if !isdir(dir)
        mkpath(dir)
    end
end

function readFile(filePath::String)
    df = DataFrame(CSV.File(filePath))
    return df
end

function get_problem_list(min_nvar, max_nvar)
    return CUTEst.select(
        min_var = min_nvar,
        max_var = max_nvar,
        max_con = 0,
        only_free_var = true,
    )
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
        default = 100000

        "--max_time"
        help = "The maximum time to run in seconds"
        arg_type = Float64
        default = 5 * 60 * 60.0

        "--tol_opt"
        help = "The tolerance for optimality"
        arg_type = Float64
        default = 1e-5

        "--θ"
        help = "θ parameter for CAT"
        arg_type = Float64
        default = 0.1

        "--β"
        help = "β parameter for CAT"
        arg_type = Float64
        default = 0.1

        "--ω_1"
        help = "ω_1 parameter for CAT"
        arg_type = Float64
        default = 8.0

        "--ω_2"
        help = "ω_2 parameter for CAT"
        arg_type = Float64
        default = 20.0

        "--γ_1"
        help = "γ_1 parameter for CAT"
        arg_type = Float64
        default = 0.01

        "--γ_2"
        help = "γ_2 parameter for CAT"
        arg_type = Float64
        default = 0.8

        "--γ_3"
        help = "γ_3 parameter for CAT"
        arg_type = Float64
        default = 1.0

        "--ξ"
        help = "ξ parameter for CAT"
        arg_type = Float64
        default = 0.1

        "--r_1"
        help = "Initial trust region radius. Negative values indicates using our default radius of value 10 * \frac{|g(x_1)||}{||H(x_1)||}"
        arg_type = Float64
        default = 0.0

        "--min_nvar"
        help = "The minimum number of variables for CUTEst model"
        arg_type = Int64
        default = 1

        "--max_nvar"
        help = "The maximum number of variables for CUTEst model"
        arg_type = Int64
        default = 500

        "--δ"
        help = "Starting δ for CAT"
        arg_type = Float64
        default = 0.0

        "--print_level"
        help = "Print level. If < 0, nothing to print, 0 for info and > 0 for debugging."
        arg_type = Int64
        default = 0

        "--seed"
        help = "Specify seed level for randomness."
        arg_type = Int64
        default = 1

        "--criteria"
        help = "The ordering of criteria separated by commas. Allowed values are `ρ_hat_rule`, `initial_radius`, `radius_update_rule`."
        arg_type = String
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
    β::Float64,
    ω_1::Float64,
    ω_2::Float64,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    ξ::Float64,
    r_1::Float64,
)
    problem_data_vec = []
    radius_update_rule_approach = "DEFAULT"
    problem_data_original = (
        β,
        θ,
        ω_1,
        ω_2,
        r_1,
        max_it,
        tol_opt,
        max_time,
        γ_1,
        γ_2,
        γ_3,
        ξ,
        radius_update_rule_approach,
    )
    for crt in criteria
        if crt == "original"
            problem_data = problem_data_original
            push!(problem_data_vec, problem_data)
        elseif crt == "ρ_hat_rule"
            problem_data = problem_data_original
            index_to_override = 2
            new_problem_data = (
                problem_data[1:index_to_override-1]...,
                0.0,
                problem_data[index_to_override+1:end]...,
            )
            problem_data = new_problem_data
            problem_data = new_problem_data
            push!(problem_data_vec, problem_data)
        elseif crt == "initial_radius"
            problem_data = problem_data_original
            r_1 = 1.0
            index_to_override = 5
            new_problem_data = (
                problem_data[1:index_to_override-1]...,
                r_1,
                problem_data[index_to_override+1:end]...,
            )
            problem_data = new_problem_data
            push!(problem_data_vec, problem_data)
            # radius_update_rule
        else
            problem_data = problem_data_original
            index_to_override = 4
            new_problem_data = (
                problem_data[1:index_to_override-1]...,
                ω_1,
                problem_data[index_to_override+1:end]...,
            )
            problem_data = new_problem_data
            index_to_override = 13
            radius_update_rule_approach = "NOT DEFAULT"
            new_problem_data = (
                problem_data[1:index_to_override-1]...,
                radius_update_rule_approach,
                problem_data[index_to_override+1:end]...,
            )
            problem_data = new_problem_data
            push!(problem_data_vec, problem_data)
        end
    end
    return problem_data_vec
end

function outputIterationsStatusToCSVFile(
    cutest_problem::String,
    status::String,
    total_execution_time::Float64,
    computation_stats::Dict,
    total_results_output_file_path::String,
)
    total_function_evaluation = Int(computation_stats["total_function_evaluation"])
    total_gradient_evaluation = Int(computation_stats["total_gradient_evaluation"])
    total_hessian_evaluation = Int(computation_stats["total_hessian_evaluation"])
    total_number_factorizations = Int(computation_stats["total_number_factorizations"])
    function_value = computation_stats["function_value"]
    gradient_value = computation_stats["gradient_value"]

    open(total_results_output_file_path, "a") do iteration_status_csv_file
        write(
            iteration_status_csv_file,
            "$cutest_problem,$status,$total_execution_time,$function_value,$gradient_value,$total_function_evaluation,$total_gradient_evaluation,$total_hessian_evaluation,$total_number_factorizations\n",
        )
    end
end

function runModelFromProblem(
    cutest_problem::String,
    criteria::String,
    problem_data,
    δ::Float64,
    print_level::Int64,
    seed::Int64,
    total_results_output_file_path::String,
)

    global nlp = nothing
    β,
    θ,
    ω_1,
    ω_2,
    r_1,
    max_it,
    tol_opt,
    max_time,
    γ_1,
    γ_2,
    γ_3,
    ξ,
    radius_update_rule_approach = problem_data
    start_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
    try
        dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
        println("$dates_format-----------EXECUTING PROBLEM----------", cutest_problem)
        @info "$dates_format-----------EXECUTING PROBLEM----------$cutest_problem"
        nlp = CUTEstModel(cutest_problem)
        termination_conditions_struct =
            consistently_adaptive_trust_region_method.TerminationConditions(
                max_it,
                tol_opt,
                max_time,
            )
        initial_radius_struct =
            consistently_adaptive_trust_region_method.INITIAL_RADIUS_STRUCT(r_1)
        problem = consistently_adaptive_trust_region_method.Problem_Data(
            nlp,
            termination_conditions_struct,
            initial_radius_struct,
            β,
            θ,
            ω_1,
            ω_2,
            γ_1,
            γ_2,
            γ_3,
            ξ,
            seed,
            print_level,
            radius_update_rule_approach,
        )
        x_1 = problem.nlp.meta.x0
        x = x_1
        status = Nothing
        iteration_stats = Nothing
        x,
        status,
        iteration_stats,
        computation_stats,
        total_iterations_count,
        total_execution_time =
            consistently_adaptive_trust_region_method.CAT(problem, x_1, δ)
        function_value = NaN
        gradient_value = NaN
        if size(last(iteration_stats, 1))[1] > 0
            function_value = last(iteration_stats, 1)[!, "fval"][1]
            gradient_value = last(iteration_stats, 1)[!, "gradval"][1]
        end
        computation_stats_modified =
            Dict("function_value" => function_value, "gradient_value" => gradient_value)
        for key in keys(computation_stats)
            computation_stats_modified[key] = computation_stats[key]
        end
        dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
        println("$dates_format------------------------MODEL SOLVED WITH STATUS: ", status)
        @info "$dates_format------------------------MODEL SOLVED WITH STATUS: $status"

        status_string = convertStatusCodeToStatusString(status)
        outputIterationsStatusToCSVFile(
            cutest_problem,
            status_string,
            total_execution_time,
            computation_stats_modified,
            total_results_output_file_path,
        )
    catch e
        @show e
        status = "INCOMPLETE"
        computation_stats = Dict(
            "total_function_evaluation" => 2 * max_it + 1,
            "total_gradient_evaluation" => 2 * max_it + 1,
            "total_hessian_evaluation" => 2 * max_it + 1,
            "total_number_factorizations" => 2 * max_it + 1,
            "function_value" => NaN,
            "gradient_value" => NaN,
        )
        dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
        end_time = dates_format
        println("$dates_format------------------------MODEL SOLVED WITH STATUS: ", status)
        @info "$dates_format------------------------MODEL SOLVED WITH STATUS: $status"
        total_execution_time = 18000.0
        outputIterationsStatusToCSVFile(
            cutest_problem,
            status,
            total_execution_time,
            computation_stats,
            total_results_output_file_path,
        )
    finally
        if nlp != nothing
            finalize(nlp)
        end
    end
end

function computeGeomeans(df::DataFrame, max_it::Int64)
    total_factorization_count_vec = Vector{Float64}()
    total_function_evaluation_vec = Vector{Float64}()
    total_gradient_evaluation_vec = Vector{Float64}()
    total_hessian_evaluation_vec = Vector{Float64}()
    for i = 1:size(df)[1]
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

    geomean_count_factorization = geomean(df_results_new.total_factorization_evaluation)
    geomean_total_function_evaluation = geomean(df_results_new.total_function_evaluation)
    geomean_total_gradient_evaluation = geomean(df_results_new.total_gradient_evaluation)
    geomean_total_hessian_evaluation = geomean(df_results_new.total_hessian_evaluation)

    return (
        geomean_total_function_evaluation,
        geomean_total_gradient_evaluation,
        geomean_total_hessian_evaluation,
        geomean_count_factorization,
    )
end

function runProblems(
    criteria::Vector{String},
    problem_data_vec::Vector{Any},
    δ::Float64,
    folder_name::String,
    default_problems::Bool,
    min_nvar::Int64,
    max_nvar::Int64,
    print_level::Int64,
    seed::Int64,
)

    cutest_problems = []
    if default_problems
        cutest_problems = default_problems_list
    else
        cutest_problems = get_problem_list(min_nvar, max_nvar)
    end
    cutest_problems = filter!(e -> e ∉ skip_list, cutest_problems)

    geomean_results_file_path =
        string(folder_name, "/", "geomean_results_ablation_study.csv")

    if isfile(geomean_results_file_path)
        rm(geomean_results_file_path)  # Delete the file if it already exists
    end

    open(geomean_results_file_path, "w") do file
        write(
            file,
            "criteria,total_failure,geomean_total_function_evaluation,geomean_total_gradient_evaluation,geomean_total_hessian_evaluation,geomean_count_factorization\n",
        )
    end

    for index = 1:length(criteria)
        crt = criteria[index]
        total_results_output_directory = string(folder_name, "/$crt")
        total_results_output_file_name = "table_cutest_$crt.csv"
        total_results_output_file_path =
            string(total_results_output_directory, "/", total_results_output_file_name)

        if !isfile(total_results_output_file_path)
            mkpath(total_results_output_directory)
            open(total_results_output_file_path, "a") do iteration_status_csv_file
                write(
                    iteration_status_csv_file,
                    "problem_name,status,total_execution_time,function_value,gradient_value,total_function_evaluation,total_gradient_evaluation,total_hessian_evaluation,total_factorization_evaluation\n",
                )
            end
        end

        for cutest_problem in cutest_problems
            if cutest_problem in
               DataFrame(CSV.File(total_results_output_file_path)).problem_name
                @show cutest_problem
                dates_format = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
                @info "$dates_format Skipping Problem $cutest_problem."
                continue
            else
                runModelFromProblem(
                    cutest_problem,
                    crt,
                    problem_data_vec[index],
                    δ,
                    print_level,
                    seed,
                    total_results_output_file_path,
                )
            end
        end

        df = DataFrame(CSV.File(total_results_output_file_path))
        df = filter(:problem_name => p_n -> p_n in cutest_problems, df)
        max_it = problem_data_vec[index][6]
        geomean_total_function_evaluation,
        geomean_total_gradient_evaluation,
        geomean_total_hessian_evaluation,
        geomean_count_factorization = computeGeomeans(df, max_it)
        counts = countmap(df.status)
        total_failure =
            length(df.status) - get(counts, "SUCCESS", 0) - get(counts, "OPTIMAL", 0)
        open(geomean_results_file_path, "a") do file
            write(
                file,
                "$crt,$total_failure,$geomean_total_function_evaluation,$geomean_total_gradient_evaluation,$geomean_total_hessian_evaluation,$geomean_count_factorization\n",
            )
        end
    end

    df_geomean_results = DataFrame(CSV.File(geomean_results_file_path))
    return df_geomean_results
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
    β = parsed_args["β"]
    ω_1 = parsed_args["ω_1"]
    ω_2 = parsed_args["ω_2"]
    γ_1 = parsed_args["γ_1"]
    γ_2 = parsed_args["γ_2"]
    γ_3 = parsed_args["γ_3"]
    ξ = parsed_args["ξ"]
    δ = parsed_args["δ"]
    print_level = parsed_args["print_level"]
    seed = parsed_args["seed"]

    default_criteria = ["ρ_hat_rule", "initial_radius", "radius_update_rule"]
    criteria = split(parsed_args["criteria"], ",")
    for val in criteria
        if val ∉ default_criteria
            error(
                "`criteria` allowed values are `ρ_hat_rule`, `GALAHAD_TRS`, `initial_radius`, `radius_update_rule`.",
            )
        end
    end
    criteria = vcat("original", criteria)
    criteria = String.(criteria)

    problem_data_vec = createProblemData(
        criteria,
        max_it,
        max_time,
        tol_opt,
        θ,
        β,
        ω_1,
        ω_2,
        γ_1,
        γ_2,
        γ_3,
        ξ,
        r_1,
    )
    @info criteria
    @info problem_data_vec
    runProblems(
        criteria,
        problem_data_vec,
        δ,
        folder_name,
        default_problems,
        min_nvar,
        max_nvar,
        print_level,
        seed,
    )
end

function convertStatusCodeToStatusString(status)
    dict_status_code = Dict(
        consistently_adaptive_trust_region_method.TerminationStatusCode.OPTIMAL =>
            "OPTIMAL",
        consistently_adaptive_trust_region_method.TerminationStatusCode.UNBOUNDED =>
            "UNBOUNDED",
        consistently_adaptive_trust_region_method.TerminationStatusCode.ITERATION_LIMIT =>
            "ITERATION_LIMIT",
        consistently_adaptive_trust_region_method.TerminationStatusCode.TIME_LIMIT =>
            "TIME_LIMIT",
        consistently_adaptive_trust_region_method.TerminationStatusCode.MEMORY_LIMIT =>
            "MEMORY_LIMIT",
        consistently_adaptive_trust_region_method.TerminationStatusCode.STEP_SIZE_LIMIT =>
            "STEP_SIZE_LIMIT",
        consistently_adaptive_trust_region_method.TerminationStatusCode.NUMERICAL_ERROR =>
            "NUMERICAL_ERROR",
        consistently_adaptive_trust_region_method.TerminationStatusCode.OTHER_ERROR =>
            "OTHER_ERROR",
    )
    return dict_status_code[status]
end

main()
