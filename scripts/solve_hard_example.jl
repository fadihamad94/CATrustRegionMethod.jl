import ArgParse
include("../src/hard_example_complexity/plot_hard_example.jl")

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

    "--max_it"
    help = "The maximum number of iterations to run"
    arg_type = Int64
    default = 1000000

    "--max_time"
    help = "The maximum time to run in seconds"
    arg_type = Float64
    default = 30 * 60.0

    "--tol_opt"
    help = "The tolerance for optimality"
    arg_type = Float64
    default = 1e-3

    "--θ"
    help = "θ parameter"
    arg_type = Float64
    default = 0.1

    "--r_1"
    help = "Initial trust region radius"
    arg_type = Float64
    default = 1.5
  end

  return ArgParse.parse_args(arg_parse)
end

function main()
  parsed_args = parse_command_line()

  folder_name = parsed_args["output_dir"]
  max_it = parsed_args["max_it"]
  max_time = parsed_args["max_time"]
  tol_opt = parsed_args["tol_opt"]
  θ = parsed_args["θ"]
  r_1 = parsed_args["r_1"]
  if_mkpath("$folder_name")

  solveHardComplexityExample(folder_name, max_it, max_time, tol_opt, θ, r_1)
end

main()
