import ArgParse
include("../benchmark/collect_CUTEst_results.jl")
include("../benchmark/plot_figures.jl")

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
  end

  return ArgParse.parse_args(arg_parse)
end

function main()
  parsed_args = parse_command_line()

  folder_name = parsed_args["output_dir"]
  if_mkpath("$folder_name")

  collectAllResults(folder_name)
  plotAllFigures(folder_name)
end

main()
