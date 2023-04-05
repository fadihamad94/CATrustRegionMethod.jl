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

    "--execute"
    help = "This parameter is needed to determine if the scripts to run different batches at the same time should be only created or created and executed."
    arg_type = Bool
    default = false
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

function getParsedArgsAsArray(parsed_args::Dict{String, Any}, train_batch_count::Int64, train_batch_index::Int64)
  parsed_args_str_vector = Vector{String}()
  for (key, value) in parsed_args
    if key == "execute"
      continue
    end
    push!(parsed_args_str_vector, "--$key")
    if key == "output_dir"
      push!(parsed_args_str_vector, "$(value)_$train_batch_index")
    else
      push!(parsed_args_str_vector, "$value")
    end
  end
  push!(parsed_args_str_vector, "--train_batch_count")
  push!(parsed_args_str_vector, "$train_batch_count")
  push!(parsed_args_str_vector, "--train_batch_index")
  push!(parsed_args_str_vector, "$train_batch_index")
  return parsed_args_str_vector
end

function main()
  initialization_lock = ReentrantLock()
  parsed_args = parse_command_line()
  execute = parsed_args["execute"]
  # parsed_args_str = strip(restoreParsed_argsAsString(parsed_args))
  train_batch_count = Threads.nthreads()
  Threads.@threads for train_batch_index in 1:train_batch_count
    parsedArgsAsStringArray = getParsedArgsAsArray(parsed_args, train_batch_count, train_batch_index)
    if execute
      #This lockiing mechanism was added to give enough time for each process to setup and do the initializations as CUTEst fails if multiple threads run at the same time when reading the list of CUTEst problems and their size
      #Another option would be later to be implemented is to make this thread reads the problems and then pass each batch to a different thread 
      lock(initialization_lock)
      string_command_arguments = join(parsedArgsAsStringArray, " ")
      @show "Executing command julia -project=../CAT/CAT/scripts ../CAT-Journal/scripts/solve_cutest.jl $string_command_arguments"
      run(Cmd(`julia --project=../CAT/CAT/scripts ../CAT-Journal/scripts/solve_cutest.jl $parsedArgsAsStringArray`, detach=true));
      @show "Reach here"
      sleep(5 * 60.0)
      unlock(initialization_lock)
    else
      open("./scripts/solve_cutest_batch_$(train_batch_index).sh", "w") do file
        string_command_arguments = join(parsedArgsAsStringArray, " ")
    	write(file, "julia --project=../CAT/CAT/scripts ../CAT-Journal/scripts/solve_cutest.jl $string_command_arguments")
      end
    end
  end
end

main()
