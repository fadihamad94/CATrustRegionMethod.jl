import ArgParse
include("../test/run_cutest_benchmark.jl")

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

    "--solver"
    help = "The optimization method to use, must be `CAT`, `CAT_GALAHAD_FACTORIZATION`, `CAT_GALAHAD_ITERATIVE`, `NewtonTrustRegion`, `ARC`, `TRU_GALAHAD_FACTORIZATION`, or `TRU_GALAHAD_ITERATIVE`."
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

    "--θ"
    help = "θ parameter for CAT"
    arg_type = Float64
    default = 0.1

    "--β"
    help = "β parameter for CAT"
    arg_type = Float64
    default = 0.1

    "--ω"
    help = "ω parameter for CAT"
    arg_type = Float64
    default = 8.0

    "--γ_2"
    help = "γ_2 parameter for CAT"
    arg_type = Float64
    default = 0.8

    "--r_1"
    help = "Initial trust region radius"
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
  end

  return ArgParse.parse_args(arg_parse)
end

function restoreParsed_argsAsString(parsed_args::Dict{String, Any})
  parsed_args_str = ""
  for (key, value) in parsed_args
    parsed_args_str = string(parsed_args_str, " --", key, " ", value)
  end
  return parsed_args_str
end
function main()
  # parsed_args = parse_command_line()
  # parsed_args_str = strip(restoreParsed_argsAsString(parsed_args))
  # train_batch_count = Threads.nthreads()
  # run(Cmd(`julia ../CAt-Journal/scripts/solve_cutest.jl $parsed_args_str`));
  # Threads.@threads for train_batch_index in 1:1
  #   #run(Cmd(`julia ./src/test_julia_script.jl $parsed_args_str --train_batch_index $train_batch_index --train_batch_count $train_batch_count`));
  #   @show "$parsed_args_str --train_batch_index $train_batch_index --train_batch_count $train_batch_count"
  #   run(Cmd(`julia ../CAt-Journal/scripts/solve_cutest.jl $parsed_args_str`));
  #   # run(Cmd(`julia ../CAt-Journal/scripts/solve_cutest.jl --ω 8.0 --output_dir test --δ 0.0 --max_it 10000 --min_nvar 1 --solver CAT --tol_opt 1.0e-5 --max_time 1800.0 --default_problems true --r_1 1.0 --γ_2 0.8 --θ 0.1 --β 0.1 --max_nvar 500`))
  # end

  # folder_name = parsed_args["output_dir"]
  # if_mkpath("$folder_name")
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

  Threads.@threads for train_batch_index in 1:train_batch_count
    folder_name = str(parsed_args["output_dir"], "_$train_batch_index")
    if_mkpath("$folder_name")
    if parsed_args["solver"] == "CAT" || parsed_args["solver"] == "CAT_GALAHAD_FACTORIZATION" || parsed_args["solver"] == "CAT_GALAHAD_ITERATIVE"
      θ = parsed_args["θ"]
      β = parsed_args["β"]
      ω = parsed_args["ω"]
      γ_2 = parsed_args["γ_2"]
      δ = parsed_args["δ"]
      run_cutest_with_CAT(folder_name, default_problems, max_it, max_time, tol_opt, θ, β, ω, γ_2, r_1, δ, min_nvar, max_nvar, train_batch_count, train_batch_index, parsed_args["solver"])
    elseif parsed_args["solver"] == "NewtonTrustRegion"
      run_cutest_with_newton_trust_region(folder_name, default_problems, max_it, max_time, tol_opt, r_1, min_nvar, max_nvar)
    elseif parsed_args["solver"] == "ARC"
      run_cutest_with_arc(folder_name, default_problems, max_it, max_time, tol_opt, r_1, min_nvar, max_nvar, train_batch_count, train_batch_index)
    elseif parsed_args["solver"] == "TRU_GALAHAD_FACTORIZATION" || parsed_args["solver"] == "TRU_GALAHAD_ITERATIVE"
      run_cutest_with_tru(folder_name, default_problems, max_it, max_time, tol_opt, r_1, min_nvar, max_nvar, train_batch_count, train_batch_index, parsed_args["solver"])
    else
      error("`solver` arg must be `CAT`, `CAT_GALAHAD_FACTORIZATION`, `CAT_GALAHAD_ITERATIVE`, `NewtonTrustRegion`, `ARC`, `TRU_GALAHAD_FACTORIZATION`, or `TRU_GALAHAD_ITERATIVE`.")
    end
  end
end

main()
