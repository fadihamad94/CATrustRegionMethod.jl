module BenchmarkDefaults

using JSON

const DEFAULTS_FILE = joinpath(@__DIR__, "defaults.json")

function default_num_threads(path::String = DEFAULTS_FILE)::Int64
    defaults = JSON.parsefile(path)
    execution = get(defaults, "benchmark_execution", nothing)
    execution isa AbstractDict ||
        error("Missing object `benchmark_execution` in $path.")
    value = get(execution, "num_threads", nothing)
    value isa Integer && !(value isa Bool) ||
        error("`benchmark_execution.num_threads` in $path must be an integer.")
    value > 0 ||
        error("`benchmark_execution.num_threads` in $path must be positive.")
    return Int64(value)
end

function main(arguments::Vector{String} = ARGS)::Nothing
    arguments == ["num_threads"] ||
        error("Usage: benchmark_defaults.jl num_threads")
    println(default_num_threads())
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

export DEFAULTS_FILE, default_num_threads

end
