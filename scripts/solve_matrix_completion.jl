import ArgParse
include("../benchmark/matrixCompletion.jl")

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
    default = 1000

    "--max_time"
    help = "The maximum time to run in seconds"
    arg_type = Float64
    default = 30 * 60.0

    "--tol_opt"
    help = "The tolerance for optimality"
    arg_type = Float64
    default = 1e-5

    "--λ_1"
    help = "The regularization parameter for observed deviations"
    arg_type = Float64
    default = 0.001

    "--λ_2"
    help = "The regularization parameter for the decomposition of the input matrix"
    arg_type = Float64
    default = 0.001

    "--instances"
    help="Number of randomly generated instances to solve"
    arg_type = Int64
    default = 10
  end

  return ArgParse.parse_args(arg_parse)
end

function main()
  parsed_args = parse_command_line()

  folder_name = parsed_args["output_dir"]
  max_it = parsed_args["max_it"]
  max_time = parsed_args["max_time"]
  tol_opt = parsed_args["tol_opt"]
  λ_1 = parsed_args["λ_1"]
  λ_2 = parsed_args["λ_2"]
  instances = parsed_args["instances"]
  if_mkpath("$folder_name")

  solveMatrixCompletionMultipleTimes(folder_name, max_it, max_time, tol_opt, λ_1, λ_2, instances)
end

main()
