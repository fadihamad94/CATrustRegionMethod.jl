using CSV, Plots, DataFrames

const CAT_FACTORIZATION = "CAT_FACTORIZATION"
const CAT_THETA_ZERO_FACTORIZATION = "CAT_THETA_ZERO_FACTORIZATION"
const ARC_FACTORIZATION = "ARC_FACTORIZATION"
const TRU_FACTORIZATION = "TRU_FACTORIZATION"
const NewtonTrustRegion = "TRU_FACTORIZATION"

const CAT_FACTORIZATION_COLOR = :black
const CAT_THETA_ZERO_FACTORIZATION_COLOR = :red
const ARC_FACTORIZATION_COLOR = :orange
const NewtonTrustRegion_COLOR = :purple

ITR_LIMIT = 1000000

const TOTAL_ITERATIONS = [Int(10 * i) for i in 1:(ITR_LIMIT/10)]
const TOTAL_GRADIENTS  = [Int(10 * i) for i in 1:(ITR_LIMIT/10)]

function readFile(fileName::String)
    df = DataFrame(CSV.File(fileName))
    return df
end

function filterRows(total_iterations_max::Int64, iterations_vector::Vector{Int64})
    return filter!(x->x < total_iterations_max, iterations_vector)
end

function computeFraction(df::DataFrame, TOTAL::Vector{Int64}, criteria::String)
    total_number_problems = size(df)[1]

    if criteria == "Iterations"
        results_fraction = DataFrame(Iterations=Int[], CAT_FACTORIZATION=Float64[], CAT_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        results_total = DataFrame(Iterations=Int[], CAT_FACTORIZATION=Int[], CAT_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
    else
        results_fraction = DataFrame(Gradients=Int[], CAT_FACTORIZATION=Float64[], CAT_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        results_total = DataFrame(Gradients=Int[], CAT_FACTORIZATION=Int[], CAT_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
    end

    for total in TOTAL
        total_problems_CAT_FACTORIZATION = length(filterRows(total, df[:, CAT_FACTORIZATION]))
        total_problems_CAT_THETA_ZERO_FACTORIZATION = length(filterRows(total, df[:, CAT_THETA_ZERO_FACTORIZATION]))
        # total_problems_CAT_THETA_ZERO_FACTORIZATION = length(filterRows(total, df[:, CAT_FACTORIZATION]))
        total_problems_ARC_FACTORIZATION = length(filterRows(total, df[:, ARC_FACTORIZATION]))
        total_problems_NewtonTrustRegion = length(filterRows(total, df[:, TRU_FACTORIZATION]))
        push!(results_fraction, (total, total_problems_CAT_FACTORIZATION / total_number_problems, total_problems_CAT_THETA_ZERO_FACTORIZATION / total_number_problems, total_problems_ARC_FACTORIZATION / total_number_problems, total_problems_NewtonTrustRegion / total_number_problems))
        push!(results_total, (total, total_problems_CAT_FACTORIZATION, total_problems_CAT_THETA_ZERO_FACTORIZATION, total_problems_ARC_FACTORIZATION, total_problems_NewtonTrustRegion))
    end

    return results_fraction
end

function plotFigureComparisonCAT(df::DataFrame, criteria::String, dirrectoryName::String, plot_name::String)
    # @show first(df, 100)
    # @show df.CAT_FACTORIZATION == df.CAT_THETA_ZERO_FACTORIZATION
    data = Matrix(df[!, Not(criteria)])
    # criteria_keyrword = criteria == "Iterations" ? "iterations" : "gradient evaluations"
    criteria_keyrword = criteria == "Iterations" ? "function evaluations" : "gradient evaluations"
    plot(df[!, criteria],
        data,
        label=["Our method default (θ = 0.1)" "Our method (θ = 0.0)"],
        color = [CAT_FACTORIZATION_COLOR CAT_THETA_ZERO_FACTORIZATION_COLOR],
        ylabel="Fraction of problems solved",
        xlabel=string("Total number of ", criteria_keyrword),
        legend=:bottomright,
        xlims=(10, ITR_LIMIT),
        xaxis=:log10
    )
    fullPath = string(dirrectoryName, "/", plot_name)
    png(fullPath)
end

function generateFiguresComparisonCAT(dirrectoryName::String)
    fileName = "all_algorithm_results_iterations.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Iterations")
    results = results[:, filter(x -> (x in ["Iterations", CAT_FACTORIZATION,CAT_THETA_ZERO_FACTORIZATION]), names(results))]
    # @show first(results, 10)
    plot_name = "fraction_of_problems_solved_versus_total_iterations_count_comparison_CAT.png"
    plotFigureComparisonCAT(results, "Iterations", dirrectoryName, plot_name)
end

function plotFiguresComparisonFinal(df::DataFrame, criteria::String, dirrectoryName::String, plot_name::String)
    data = Matrix(df[!, Not(criteria)])
    # criteria_keyrword = criteria == "Iterations" ? "iterations" : "gradient evaluations"
    criteria_keyrword = criteria == "Iterations" ? "function evaluations" : "gradient evaluations"
    plot(df[!, criteria],
        data,
        # label=["Our method" "ARC with g-rule" "Newton trust region"],
        label=["CAT" "ARC" "TRU"],
        color = [CAT_FACTORIZATION_COLOR ARC_FACTORIZATION_COLOR NewtonTrustRegion_COLOR],
        ylabel="Fraction of problems solved",
        xlabel=string("Total number of ", criteria_keyrword),
        legend=:bottomright,
        xlims=(10, ITR_LIMIT),
        xaxis=:log10
    )
    fullPath = string(dirrectoryName, "/", plot_name)
    @show fullPath
    # @show show(names(df))
    png(fullPath)
end

function generateFiguresIterationsComparisonFinal(dirrectoryName::String)
    fileName = "all_algorithm_results_iterations.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Iterations")
    results = results[:, filter(x -> (x in ["Iterations", CAT_FACTORIZATION,ARC_FACTORIZATION,NewtonTrustRegion]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_iterations_count_final.png"
    plotFiguresComparisonFinal(results, "Iterations", dirrectoryName, plot_name)
end

function generateFiguresGradientsComparisonFinal(dirrectoryName::String)
    fileName = "all_algorithm_results_gradients.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_GRADIENTS, "Gradients")
    # @show names(results)
    results = results[:, filter(x -> (x in ["Gradients", CAT_FACTORIZATION,ARC_FACTORIZATION,NewtonTrustRegion]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_gradients_count_final.png"
    plotFiguresComparisonFinal(results, "Gradients", dirrectoryName, plot_name)
end

function plotAllFigures(dirrectoryName::String)
    generateFiguresComparisonCAT(dirrectoryName)
    generateFiguresIterationsComparisonFinal(dirrectoryName)
    generateFiguresGradientsComparisonFinal(dirrectoryName)
end

plotAllFigures("/Users/fah33/PhD_Research/CAT_RESULTS_BENCHMARK")
