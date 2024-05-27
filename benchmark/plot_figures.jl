using CSV, Plots, DataFrames

const CAT_I_FACTORIZATION = "CAT_I_FACTORIZATION"
const CAT_II_FACTORIZATION = "CAT_II_FACTORIZATION"
const CAT_II_THETA_ZERO_FACTORIZATION = "CAT_II_THETA_ZERO_FACTORIZATION"
const ARC_FACTORIZATION = "ARC_FACTORIZATION"
const TRU_FACTORIZATION = "TRU_FACTORIZATION"
const TRU_FACTORIZATION = "TRU_FACTORIZATION"

const CAT_I_FACTORIZATION_COLOR = :blue
const CAT_II_FACTORIZATION_COLOR = :black
const CAT_II_THETA_ZERO_FACTORIZATION_COLOR = :red
const ARC_FACTORIZATION_COLOR = :orange
const TRU_FACTORIZATION_COLOR = :purple

ITR_LIMIT = 100000
TIME_LIMIT = 18000

const TOTAL_ITERATIONS = [Int(10 * i) for i in 1:(ITR_LIMIT/10)]
const TOTAL_TIME = [Int(5 * i) for i in 1:(TIME_LIMIT/5)]
const TOTAL_TIME_2 = vcat([0.1, 0.2, 0.3, 0.4, 1, 2, 3, 4], [Int(5 * i) for i in 1:(TIME_LIMIT/5)])

function readFile(fileName::String)
    df = DataFrame(CSV.File(fileName))
    return df
end

function filterRows(total_iterations_max::Int64, iterations_vector::Vector{Int64})
    return filter!(x->x < total_iterations_max, iterations_vector)
end
function filterRows(total_iterations_max::Int64, iterations_vector::Vector{Float64})
    return filter!(x->x < total_iterations_max, iterations_vector)
end

function computeFraction(df::DataFrame, TOTAL::Vector{Int64}, criteria::String)
    total_number_problems = size(df)[1]

    if criteria == "Iterations"
        results_fraction = DataFrame(Iterations=Int[], CAT_I_FACTORIZATION=Float64[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        results_total = DataFrame(Iterations=Int[], CAT_I_FACTORIZATION=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
    elseif criteria == "Gradients"
        results_fraction = DataFrame(Gradients=Int[], CAT_I_FACTORIZATION=Float64[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        results_total = DataFrame(Gradients=Int[], CAT_I_FACTORIZATION=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
    elseif criteria == "Hessian"
        results_fraction = DataFrame(Hessian=Int[], CAT_I_FACTORIZATION=Float64[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        results_total = DataFrame(Hessian=Int[], CAT_I_FACTORIZATION=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
    elseif criteria == "Factorization"
        results_fraction = DataFrame(Factorization=Int[], CAT_I_FACTORIZATION=Float64[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        results_total = DataFrame(Factorization=Int[], CAT_I_FACTORIZATION=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
    else
        results_fraction = DataFrame(Time=Int[], CAT_I_FACTORIZATION=Float64[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        results_total = DataFrame(Time=Int[], CAT_I_FACTORIZATION=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
    end

    for total in TOTAL
        total_problems_CAT_I_FACTORIZATION = length(filterRows(total, df[:, CAT_I_FACTORIZATION]))
        total_problems_CAT_II_FACTORIZATION = length(filterRows(total, df[:, CAT_II_FACTORIZATION]))
        total_problems_CAT_II_THETA_ZERO_FACTORIZATION = length(filterRows(total, df[:, CAT_II_THETA_ZERO_FACTORIZATION]))
        # total_problems_CAT_THETA_ZERO_FACTORIZATION = length(filterRows(total, df[:, CAT_FACTORIZATION]))
        total_problems_ARC_FACTORIZATION = length(filterRows(total, df[:, ARC_FACTORIZATION]))
        total_problems_TRU_FACTORIZATION = length(filterRows(total, df[:, TRU_FACTORIZATION]))
        push!(results_fraction, (total, total_problems_CAT_I_FACTORIZATION / total_number_problems, total_problems_CAT_II_FACTORIZATION / total_number_problems, total_problems_CAT_II_THETA_ZERO_FACTORIZATION / total_number_problems, total_problems_ARC_FACTORIZATION / total_number_problems, total_problems_TRU_FACTORIZATION / total_number_problems))
        push!(results_total, (total, total_problems_CAT_I_FACTORIZATION, total_problems_CAT_II_FACTORIZATION, total_problems_CAT_II_THETA_ZERO_FACTORIZATION, total_problems_ARC_FACTORIZATION, total_problems_TRU_FACTORIZATION))
    end

    return results_fraction
end

function computeFraction_CAT(df::DataFrame, TOTAL::Vector{Int64}, criteria::String)
    total_number_problems = size(df)[1]

    if criteria == "Iterations"
        results_fraction = DataFrame(Iterations=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[])
        results_total = DataFrame(Iterations=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[])
    elseif criteria == "Gradients"
        results_fraction = DataFrame(Gradients=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[])
        results_total = DataFrame(Gradients=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[])
    elseif criteria == "Hessian"
        results_fraction = DataFrame(Hessian=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[])
        results_total = DataFrame(Hessian=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[])
    elseif criteria == "Factorization"
        results_fraction = DataFrame(Factorization=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[])
        results_total = DataFrame(Factorization=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[])
    else
        results_fraction = DataFrame(Time=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[])
        results_total = DataFrame(Time=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[])
    end

    for total in TOTAL

        total_problems_CAT_II_FACTORIZATION = length(filterRows(total, df[:, CAT_II_FACTORIZATION ]))
        total_problems_CAT_II_THETA_ZERO_FACTORIZATION = length(filterRows(total, df[:, CAT_II_THETA_ZERO_FACTORIZATION]))

        push!(results_fraction, (total, total_problems_CAT_II_FACTORIZATION / total_number_problems, total_problems_CAT_II_THETA_ZERO_FACTORIZATION / total_number_problems))
        push!(results_total, (total, total_problems_CAT_II_FACTORIZATION, total_problems_CAT_II_THETA_ZERO_FACTORIZATION))
    end

    return results_fraction
end

function computeFraction(df::DataFrame, TOTAL::Vector{Float64}, criteria::String)
    total_number_problems = size(df)[1]

    if criteria == "Iterations"
        results_fraction = DataFrame(Iterations=Int[], CAT_I_FACTORIZATION=Float64[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        results_total = DataFrame(Iterations=Int[], CAT_I_FACTORIZATION=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
    elseif criteria == "Gradients"
        results_fraction = DataFrame(Gradients=Int[], CAT_I_FACTORIZATION=Float64[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        results_total = DataFrame(Gradients=Int[], CAT_I_FACTORIZATION=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
    elseif criteria == "Hessian"
        results_fraction = DataFrame(Hessian=Int[], CAT_I_FACTORIZATION=Float64[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        results_total = DataFrame(Hessian=Int[], CAT_I_FACTORIZATION=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
    elseif criteria == "Factorization"
        results_fraction = DataFrame(Factorization=Int[], CAT_I_FACTORIZATION=Float64[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        results_total = DataFrame(Factorization=Int[], CAT_I_FACTORIZATION=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
    else
        results_fraction = DataFrame(Time=Float64[], CAT_I_FACTORIZATION=Float64[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        results_total = DataFrame(Time=Float64[], CAT_I_FACTORIZATION=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
        # results_fraction = DataFrame(Time=Int[], CAT_I_FACTORIZATION=Float64[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Float64[], ARC_FACTORIZATION=Float64[], TRU_FACTORIZATION=Float64[])
        # results_total = DataFrame(Time=Int[], CAT_I_FACTORIZATION=Int[], CAT_II_FACTORIZATION=Float64[], CAT_II_THETA_ZERO_FACTORIZATION=Int[], ARC_FACTORIZATION=Int[], TRU_FACTORIZATION=Int[])
    end

    # @show first(df[:, CAT_I_FACTORIZATION ], 5)
    # @show first(df[:, CAT_II_FACTORIZATION ], 5)
    for total in TOTAL
        total_problems_CAT_I_FACTORIZATION = length(filterRows(total, df[:, CAT_I_FACTORIZATION]))
        total_problems_CAT_II_FACTORIZATION = length(filterRows(total, df[:, CAT_II_FACTORIZATION ]))
        total_problems_CAT_II_THETA_ZERO_FACTORIZATION = length(filterRows(total, df[:, CAT_II_THETA_ZERO_FACTORIZATION]))
        # total_problems_CAT_THETA_ZERO_FACTORIZATION = length(filterRows(total, df[:, CAT_FACTORIZATION]))
        total_problems_ARC_FACTORIZATION = length(filterRows(total, df[:, ARC_FACTORIZATION]))
        total_problems_TRU_FACTORIZATION = length(filterRows(total, df[:, TRU_FACTORIZATION]))
        push!(results_fraction, (total, total_problems_CAT_I_FACTORIZATION / total_number_problems, total_problems_CAT_II_FACTORIZATION / total_number_problems, total_problems_CAT_II_THETA_ZERO_FACTORIZATION / total_number_problems, total_problems_ARC_FACTORIZATION / total_number_problems, total_problems_TRU_FACTORIZATION / total_number_problems))
        push!(results_total, (total, total_problems_CAT_I_FACTORIZATION, total_problems_CAT_II_FACTORIZATION, total_problems_CAT_II_THETA_ZERO_FACTORIZATION, total_problems_ARC_FACTORIZATION, total_problems_TRU_FACTORIZATION))
    end
    @show length(TOTAL)
    # @show first(df[:, CAT_II_FACTORIZATION ], 5)
    # @show first(df[:, ARC_FACTORIZATION ], 5)
    # @show first(df[:, TRU_FACTORIZATION ], 5)

    return results_fraction
end

function plotFigureComparisonCAT(df::DataFrame, criteria::String, dirrectoryName::String, plot_name::String)
    # @show first(df, 100)
    # @show df.CAT_FACTORIZATION == df.CAT_THETA_ZERO_FACTORIZATION
    data = Matrix(df[!, Not(criteria)])
    # criteria_keyrword = criteria == "Iterations" ? "iterations" : "gradient evaluations"
    criteria_keyrword = criteria == "Iterations" ? "function evaluations" : "gradient evaluations"
    # @show first(df, 5)
    # @show last(df, 5)
    plot(df[!, criteria],
        data,
        label=["Our method default (θ = 0.1)" "Our method (θ = 0.0)"],
        color = [CAT_II_FACTORIZATION_COLOR CAT_II_THETA_ZERO_FACTORIZATION_COLOR],
        ylabel="Fraction of problems solved",
        xlabel=string("Total number of ", criteria_keyrword),
        legend=:bottomright,
        xlims=(10, ITR_LIMIT),
        xaxis=:log10
    )
    yaxis!((0, 1.0), 0.1:0.1:1.0)
    fullPath = string(dirrectoryName, "/", plot_name)
    png(fullPath)
end

function generateFiguresComparisonCAT(dirrectoryName::String)
    fileName = "all_algorithm_results_functions_CAT.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction_CAT(df, TOTAL_ITERATIONS, "Iterations")
    results = results[:, filter(x -> (x in ["Iterations", CAT_II_FACTORIZATION,CAT_II_THETA_ZERO_FACTORIZATION]), names(results))]
    # @show first(results, 10)
    plot_name = "fraction_of_problems_solved_versus_total_functions_count_comparison_CAT.png"
    plotFigureComparisonCAT(results, "Iterations", dirrectoryName, plot_name)

    fileName = "all_algorithm_results_gradients_CAT.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction_CAT(df, TOTAL_ITERATIONS, "Gradients")
    results = results[:, filter(x -> (x in ["Gradients", CAT_II_FACTORIZATION,CAT_II_THETA_ZERO_FACTORIZATION]), names(results))]
    # @show first(results, 10)
    plot_name = "fraction_of_problems_solved_versus_total_gradients_count_comparison_CAT.png"
    plotFigureComparisonCAT(results, "Gradients", dirrectoryName, plot_name)
end

function plotFiguresComparisonFinal(df::DataFrame, criteria::String, dirrectoryName::String, plot_name::String)
    @show last(df, 10)
    data = Matrix(df[!, Not(criteria)])
    # criteria_keyrword = criteria == "Iterations" ? "iterations" : "gradient evaluations"
    dict_ = Dict("Iterations" => "function evaluations", "Gradients" => "gradient evaluations",
    "Hessian" => "hessian evaluations", "Factorization" => "factorization", "Time" => "seconds")
    criteria_keyrword = dict_[criteria]
    LIMIT = criteria == "Time" ? TIME_LIMIT : ITR_LIMIT
    plot(df[!, criteria],
        data,
        # label=["Our method" "ARC with g-rule" "Newton trust region"],
        label=["CAT I" "CAT II" "ARC" "TRU"],
        color = [CAT_I_FACTORIZATION_COLOR CAT_II_FACTORIZATION_COLOR ARC_FACTORIZATION_COLOR TRU_FACTORIZATION_COLOR],
        # label=["CAT II" "ARC" "TRU"],
        # color = [CAT_II_FACTORIZATION_COLOR ARC_FACTORIZATION_COLOR TRU_FACTORIZATION_COLOR],
        ylabel="Fraction of problems solved",
        # xlabel="Wall clock time (secs)",
        xlabel="Total number of $criteria_keyrword",
        legend=:bottomright,
        xlims=(10, LIMIT),
        # xlims=(1, LIMIT), For time
        xaxis=:log10
    )
    yaxis!((0, 0.9), 0.1:0.1:0.9)
    # xticks!([0.1, 10, 100, 1000])
    fullPath = string(dirrectoryName, "/", plot_name)
    @show fullPath
    # @show show(names(df))
    png(fullPath)
end

function generateFiguresIterationsComparisonFinal(dirrectoryName::String)
    # fileName = "all_algorithm_results_iterations.csv"
    fileName = "all_algorithm_results_functions.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Iterations")
    results = results[:, filter(x -> (x in ["Iterations", CAT_I_FACTORIZATION,CAT_II_FACTORIZATION,ARC_FACTORIZATION,TRU_FACTORIZATION]), names(results))]
    # results = results[:, filter(x -> (x in ["Iterations", CAT_II_FACTORIZATION,ARC_FACTORIZATION,TRU_FACTORIZATION]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_functions_count_final.png"
    plotFiguresComparisonFinal(results, "Iterations", dirrectoryName, plot_name)
end

function generateFiguresGradientsComparisonFinal(dirrectoryName::String)
    fileName = "all_algorithm_results_gradients.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Gradients")
    # @show names(results)
    results = results[:, filter(x -> (x in ["Gradients", CAT_I_FACTORIZATION,CAT_II_FACTORIZATION,ARC_FACTORIZATION,TRU_FACTORIZATION]), names(results))]
    # results = results[:, filter(x -> (x in ["Gradients" ,CAT_II_FACTORIZATION,ARC_FACTORIZATION,TRU_FACTORIZATION]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_gradients_count_final.png"
    plotFiguresComparisonFinal(results, "Gradients", dirrectoryName, plot_name)
end

function generateFiguresHessianComparisonFinal(dirrectoryName::String)
    fileName = "all_algorithm_results_hessian.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Hessian")
    # @show names(results)
    results = results[:, filter(x -> (x in ["Hessian", CAT_I_FACTORIZATION,CAT_II_FACTORIZATION,ARC_FACTORIZATION,TRU_FACTORIZATION]), names(results))]
    # results = results[:, filter(x -> (x in ["Hessian" ,CAT_II_FACTORIZATION,ARC_FACTORIZATION,TRU_FACTORIZATION]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_hessian_count_final.png"
    plotFiguresComparisonFinal(results, "Hessian", dirrectoryName, plot_name)
end

function generateFiguresFactorizationComparisonFinal(dirrectoryName::String)
    fileName = "all_algorithm_results_factorization.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Factorization")
    # @show names(results)
    results = results[:, filter(x -> (x in ["Factorization", CAT_I_FACTORIZATION,CAT_II_FACTORIZATION,ARC_FACTORIZATION,TRU_FACTORIZATION]), names(results))]
    # results = results[:, filter(x -> (x in ["Factorization" ,CAT_II_FACTORIZATION,ARC_FACTORIZATION,TRU_FACTORIZATION]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_factorization_count_final.png"
    plotFiguresComparisonFinal(results, "Factorization", dirrectoryName, plot_name)
end

function generateFiguresTimeComparisonFinal(dirrectoryName::String)
    fileName = "all_algorithms_results_time.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    # results = computeFraction(df, TOTAL_TIME, "Time")
    results = computeFraction(df, TOTAL_TIME_2, "Time")
    # @show names(results)
    # results = results[:, filter(x -> (x in ["Time", CAT_I_FACTORIZATION,CAT_II_FACTORIZATION,ARC_FACTORIZATION,TRU_FACTORIZATION]), names(results))]
    results = results[:, filter(x -> (x in ["Time" ,CAT_II_FACTORIZATION,ARC_FACTORIZATION,TRU_FACTORIZATION]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_time_final.png"
    plotFiguresComparisonFinal(results, "Time", dirrectoryName, plot_name)
end

function plotAllFigures(dirrectoryName::String)
    generateFiguresComparisonCAT(dirrectoryName)
    generateFiguresIterationsComparisonFinal(dirrectoryName)
    generateFiguresGradientsComparisonFinal(dirrectoryName)
    generateFiguresHessianComparisonFinal(dirrectoryName)
    generateFiguresFactorizationComparisonFinal(dirrectoryName)
    # generateFiguresTimeComparisonFinal(dirrectoryName)
end

# plotAllFigures("/Users/fah33/PhD_Research/CAT_RESULTS_BENCHMARK/FINAL_VERSION")
# This code to debug the implementation
# dir_ = "/Users/fah33/PhD_Research/CAT_RESULTS_BENCHMARK/results_debug_collect_results_script"
# plotAllFigures(dir_)

dir_ = "/Users/fah33/PhD_Research/CAT_RESULTS_BENCHMARK/results_final_all_algorithms/CUTEST"
plotAllFigures(dir_)
