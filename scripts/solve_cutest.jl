import ArgParse
include("../benchmark/run_cutest_benchmark.jl")

"""
Defines parses and args.
# Returns
A dictionary with the values of the command-line arguments.
"""

function if_mkpath(dir::String)
    if !isdir(dir)
        mkpath(dir)
    end
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
        help = "Initial trust region radius"
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
    end

    return ArgParse.parse_args(arg_parse)
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

    print_level = parsed_args["print_level"]
    seed = parsed_args["seed"]

    θ = parsed_args["θ"]
    β = parsed_args["β"]
    ω_1 = parsed_args["ω_1"]
    ω_2 = parsed_args["ω_2"]
    γ_1 = parsed_args["γ_1"]
    γ_2 = parsed_args["γ_2"]
    γ_3 = parsed_args["γ_3"]
    ξ = parsed_args["ξ"]
    δ = parsed_args["δ"]
    run_cutest_with_CAT(
        folder_name,
        default_problems,
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
        δ,
        min_nvar,
        max_nvar,
        print_level,
        seed,
    )
end

main()
