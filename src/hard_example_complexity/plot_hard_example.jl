include("./hard_example_implementation_code.jl")
include("./CAT_complexity_modified.jl")

using NLPModels, LinearAlgebra, DataFrames, SparseArrays, Test

function solveHardComplexityExample(folder_name::String, max_it::Int64, max_time::Float64, tol_opt::Float64, θ::Float64, r_1::Float64)
    @test max_time > 0
    δ = 0.0
    problem = Problem_Data(0.1, θ, 8.0, r_1, max_it, max_time, tol_opt)
    x_default, status_default, iteration_stats_default, computation_stats_default, total_iterations_count_default = CAT(problem, x_0, η, α_k, δ)
    @show status_default

    problem = Problem_Data(0.1, 0.0, 8.0, r_1, max_it, max_time, tol_opt)
    x_theta_0, status_theta_0, iteration_stats_theta_0, computation_stats_theta_0, total_iterations_count_theta_0 = CAT(problem, x_0, η, α_k, δ)
    @show status_theta_0

    iteration_stats_default.k = iteration_stats_default.k .+ 1
    iteration_stats_theta_0.k = iteration_stats_theta_0.k .+ 1
    plot!(iteration_stats_default.k,
        iteration_stats_default.gradval,
        label="Our method default (θ = $θ)",
        ylabel="Gradient norm",
        xlabel="Total number of iterations",
        xlims=(1, max_it),
        c=:black,
        legend=:topright,
        xaxis=:log10,
        yaxis=:log10,
    )
    plot!(iteration_stats_theta_0.k,
        iteration_stats_theta_0.gradval,
        label="Our method (θ = 0.0)",
        ylabel="Gradient norm",
        xlabel="Total number of iterations",
        xlims=(1, max_it),
        c=:red,
        legend=:topright,
        xaxis=:log10,
        yaxis=:log10
    )

    file_name = "hard_example_complexity_plot.png"
    full_path = string(folder_name, "/", file_name)
    png(full_path)
end
