import ArgParse
include("../benchmark/learningLinearDynamicalSystem.jl")

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
    default = 10000

    "--max_time"
    help = "The maximum time to run in seconds"
    arg_type = Float64
    default = 30 * 60.0

    "--tol_opt"
    help = "The tolerance for optimality"
    arg_type = Float64
    default = 1e-5

    "--d"
    help = "The dimension of the vector h"
    arg_type = Int64
    default = 4

    "--T"
    help="The time horizon"
    arg_type = Int64
    default = 50

    "--σ"
    help="Standard deviation for the gaussian noise in the evolution of the system"
    arg_type = Float64
    default = 0.01

    "--instances"
    help="Number of randomly generated instances to solve"
    arg_type = Int64
    default = 60
  end

  return ArgParse.parse_args(arg_parse)
end

function main()
  parsed_args = parse_command_line()

  folder_name = parsed_args["output_dir"]
  max_it = parsed_args["max_it"]
  max_time = parsed_args["max_time"]
  tol_opt = parsed_args["tol_opt"]
  d = parsed_args["d"]
  T = parsed_args["T"]
  σ = parsed_args["σ"]
  instances = parsed_args["instances"]
  if_mkpath("$folder_name")

  solveLinearDynamicalSystemMultipleTimes(folder_name, max_it, max_time, tol_opt, d, T, σ, instances)
end

main()
