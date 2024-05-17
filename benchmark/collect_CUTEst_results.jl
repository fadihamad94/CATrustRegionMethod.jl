using CSV, DataFrames

const CAT_FACTORIZATION_I = "CAT_I_FACTORIZATION"
const CAT_FACTORIZATION_II = "CAT_II_FACTORIZATION"
const CAT_THETA_ZERO_FACTORIZATION = "CAT_II_THETA_ZERO_FACTORIZATION"
const ARC_FACTORIZATION = "ARC_FACTORIZATION"
const TRU_FACTORIZATION= "TRU_FACTORIZATION"

const CAT_OPTIMIZATION_METHOD_I = "CAT_I"
const CAT_OPTIMIZATION_METHOD_II = "CAT_II"
const CAT_II_THETA_ZERO_OPTIMIZATION_METHOD = "CAT_II_THETA_ZERO"
const ARC_FACTORIZATION_OPTIMIZATION_METHOD = "ARC_FACTORIZATION"
const TRU_FACTORIZATION_OPTIMIZATION_METHOD = "TRU_FACTORIZATION"

const CURRENT_DIRECTORY = dirname(Base.source_path())
const RESULTS_DEFAULT_DIRECTORY_NAME = string(CURRENT_DIRECTORY, "/../results")

const PROBLEM_NAME_COLUMN = "problem_name"
const TOTAL_ITERATIONS_COUNT_COLUMN = "total_iterations_count"
const TOTAL_FUNCTION_EVALUATION_COLUMN = "total_function_evaluation"
const TOTAL_GRADIENT_EVALUATION_COLUMN = "total_gradient_evaluation"
const TOTAL_HESSIAN_EVALUATION_COLUMN = "total_hessian_evaluation"
const TOTAL_FACTORIZATION_EVALUATION_COLUMN = "total_factorization_evaluation"

function readFile(filePath::String)
    df = DataFrame(CSV.File(filePath))
    return df
end

function if_mkpath(dir::String)
  if !isdir(dir)
     mkpath(dir)
  end
end

function buildFilePath(directoryName::String, optimization_method::String)
    total_results_output_directory =  string(directoryName, "/$optimization_method")
    optimization_method = replace(optimization_method, "train_" => "")
    optimization_method = replace(optimization_method, "test_" => "")
    total_results_output_file_name = "table_cutest_$optimization_method.csv"
    total_results_file_path = string(total_results_output_directory, "/", total_results_output_file_name)
    return total_results_file_path
end

function collectResultsPerSolver(directoryName::String, optimization_method::String)
    total_results_file_path_train = buildFilePath(directoryName, "train_$optimization_method")
    total_results_file_path_test  = buildFilePath(directoryName, "test_$optimization_method")
    if !isfile(total_results_file_path_train) || !isfile(total_results_file_path_test)
        total_results_file_path_train = buildFilePath(RESULTS_DEFAULT_DIRECTORY_NAME, "train_$optimization_method")
        total_results_file_path_test  = buildFilePath(RESULTS_DEFAULT_DIRECTORY_NAME, "test_$optimization_method")
    end
    df_train = readFile(total_results_file_path_train)
    df_test = readFile(total_results_file_path_test)
    df = vcat(df_train, df_test)
    sorted_df = sort(df, PROBLEM_NAME_COLUMN)
    return sorted_df
end

function mergeDataFrames(df_results_CAT_I_FACTORIZATION_CRITERIA::DataFrame, df_results_CAT_II_FACTORIZATION_CRITERIA::DataFrame, df_results_CAT_II_THETA_ZERO_FACTORIZATION_CRITERIA::DataFrame, df_results_ARC_FACTORIZATION_OPTIMIZATION_CRITERIA::DataFrame, df_results_TRUST_REGION_OPTIMIZATION_CRITERIA::DataFrame)
    df = DataFrame(PROBLEM_NAME = [], CAT_I_FACTORIZATION = [], CAT_II_FACTORIZATION = [], CAT_II_THETA_ZERO_FACTORIZATION = [], ARC_FACTORIZATION = [], TRU_FACTORIZATION = [])
    matrix_results_CAT_I_FACTORIZATION_CRITERIA = Matrix(df_results_CAT_I_FACTORIZATION_CRITERIA)
    matrix_results_CAT_II_FACTORIZATION_CRITERIA = Matrix(df_results_CAT_II_FACTORIZATION_CRITERIA)
    matrix_results_CAT_II_THETA_ZERO_FACTORIZATION_CRITERIA = Matrix(df_results_CAT_II_THETA_ZERO_FACTORIZATION_CRITERIA)
    matrix_results_ARC_FACTORIZATION_OPTIMIZATION_CRITERIA = Matrix(df_results_ARC_FACTORIZATION_OPTIMIZATION_CRITERIA)
    matrix_results_TRUST_REGION_OPTIMIZATION_CRITERIA = Matrix(df_results_TRUST_REGION_OPTIMIZATION_CRITERIA)

    for row in 1:size(matrix_results_CAT_I_FACTORIZATION_CRITERIA)[1]
        merged_vector = vcat(matrix_results_CAT_I_FACTORIZATION_CRITERIA[row, :], matrix_results_CAT_II_FACTORIZATION_CRITERIA[row, :], matrix_results_CAT_II_THETA_ZERO_FACTORIZATION_CRITERIA[row, :], matrix_results_ARC_FACTORIZATION_OPTIMIZATION_CRITERIA[row, :], matrix_results_TRUST_REGION_OPTIMIZATION_CRITERIA[row, :])
        push!(df, merged_vector)
    end

    return df
end

function mergeDataFrames_CAT_II(df_results_CAT_II_FACTORIZATION_CRITERIA::DataFrame, df_results_CAT_II_THETA_ZERO_FACTORIZATION_CRITERIA::DataFrame)
    df = DataFrame(PROBLEM_NAME = [], CAT_II_FACTORIZATION = [], CAT_II_THETA_ZERO_FACTORIZATION = [])
    matrix_results_CAT_II_FACTORIZATION_CRITERIA = Matrix(df_results_CAT_II_FACTORIZATION_CRITERIA)
    matrix_results_CAT_II_THETA_ZERO_FACTORIZATION_CRITERIA = Matrix(df_results_CAT_II_THETA_ZERO_FACTORIZATION_CRITERIA)

    for row in 1:size(matrix_results_CAT_II_FACTORIZATION_CRITERIA)[1]
        merged_vector = vcat(matrix_results_CAT_II_FACTORIZATION_CRITERIA[row, :], matrix_results_CAT_II_THETA_ZERO_FACTORIZATION_CRITERIA[row, :])
        push!(df, merged_vector)
    end

    return df
end

function saveCSVFile(directoryName::String, criteria::String, results::DataFrame)
    results_file_path = string(directoryName, "/", "all_algorithm_results_$criteria.csv")
    CSV.write(results_file_path, results, header = true)
end

function generateALLResultsCSVFile(directoryName::String, df_results_CAT_I_FACTORIZATION::DataFrame, df_results_CAT_II_FACTORIZATION::DataFrame, df_results_CAT_II_THETA_ZERO_FACTORIZATION::DataFrame, df_results_ARC_FACTORIZATION_OPTIMIZATION::DataFrame, df_results_TRUST_REGION_OPTIMIZATION::DataFrame)
    results = DataFrame(PROBLEM_NAME = [], CAT_I_FACTORIZATION_ITR = [], CAT_I_FACTORIZATION_F = [], CAT_I_FACTORIZATION_G = [], CAT_I_FACTORIZATION_H = [],CAT_I_FACTORIZATION_FCT = [], CAT_II_FACTORIZATION_ITR = [], CAT_II_FACTORIZATION_F = [], CAT_II_FACTORIZATION_G = [], CAT_II_FACTORIZATION_H = [], CAT_II_FACTORIZATION_FCT = [],
    CAT_II_THETA_ZERO_FACTORIZATION_ITR = [], CAT_II_THETA_ZERO_FACTORIZATION_F = [], CAT_II_THETA_ZERO_FACTORIZATION_G = [], CAT_II_THETA_ZERO_FACTORIZATION_H = [], CAT_II_THETA_ZERO_FACTORIZATION_FCT = [], ARC_FACTORIZATION_ITR = [], ARC_FACTORIZATION_F = [], ARC_FACTORIZATION_G = [], ARC_FACTORIZATION_H = [],
    ARC_FACTORIZATION_FCT = [], TRU_FACTORIZATION_ITR = [], TRU_FACTORIZATION_F = [], TRU_FACTORIZATION_G = [], TRU_FACTORIZATION_H = [], TRU_FACTORIZATION_FCT = [])
    matrix_results_CAT_I_FACTORIZATION = Matrix(df_results_CAT_I_FACTORIZATION)
    matrix_results_CAT_II_FACTORIZATION = Matrix(df_results_CAT_II_FACTORIZATION)
    matrix_results_CAT_II_THETA_ZERO_FACTORIZATION = Matrix(df_results_CAT_II_THETA_ZERO_FACTORIZATION)
    matrix_results_ARC_FACTORIZATION_OPTIMIZATION = Matrix(df_results_ARC_FACTORIZATION_OPTIMIZATION)
    matrix_results_TRUST_REGION_OPTIMIZATION = Matrix(df_results_TRUST_REGION_OPTIMIZATION)

    for row in 1:size(matrix_results_CAT_I_FACTORIZATION)[1]
        merged_vector = vcat(matrix_results_CAT_I_FACTORIZATION[row, :], matrix_results_CAT_II_FACTORIZATION[row, :], matrix_results_CAT_II_THETA_ZERO_FACTORIZATION[row, :], matrix_results_ARC_FACTORIZATION_OPTIMIZATION[row, :], matrix_results_TRUST_REGION_OPTIMIZATION[row, :])
        push!(results, merged_vector)
    end
    if_mkpath(directoryName)
    all_results_file_path = string(directoryName, "/", "all_algorithm_all_results.csv")
    CSV.write(all_results_file_path, results, header = true)
end

#=
Collect results for each optimization method
#If directory for specific optimization method  doesn't exist which means user didn't execute
this optimization method, then get the results from what we saved based on our experiments
=#
function collectAllResults(directoryName::String)
    #Collect results for CAT I
    optimization_method = CAT_OPTIMIZATION_METHOD_I
    df_results_CAT_I_FACTORIZATION = collectResultsPerSolver(directoryName, optimization_method)

    #Collect results for CAT II
    optimization_method = CAT_OPTIMIZATION_METHOD_II
    df_results_CAT_II_FACTORIZATION = collectResultsPerSolver(directoryName, optimization_method)

    #Collect results for CAT II θ = 0.0 Factorizarion
    optimization_method = CAT_II_THETA_ZERO_OPTIMIZATION_METHOD
    df_results_CAT_II_THETA_ZERO_FACTORIZATION = collectResultsPerSolver(directoryName, optimization_method)

    #Collect results for ARC
    optimization_method = ARC_FACTORIZATION_OPTIMIZATION_METHOD
    df_results_ARC_FACTORIZATION_OPTIMIZATION = collectResultsPerSolver(directoryName, optimization_method)

    #Collect results for TRU
    optimization_method = TRU_FACTORIZATION_OPTIMIZATION_METHOD
    df_results_TRUST_REGION_OPTIMIZATION = collectResultsPerSolver(directoryName, optimization_method)

    #Validate all df has the same problems
    @assert df_results_CAT_I_FACTORIZATION.problem_name == df_results_CAT_II_FACTORIZATION.problem_name
    @assert df_results_CAT_I_FACTORIZATION.problem_name == df_results_CAT_II_THETA_ZERO_FACTORIZATION.problem_name
    @assert df_results_CAT_I_FACTORIZATION.problem_name == df_results_ARC_FACTORIZATION_OPTIMIZATION.problem_name
    @assert df_results_CAT_I_FACTORIZATION.problem_name == df_results_TRUST_REGION_OPTIMIZATION.problem_name

    #Generate results for all algorithm  for total number of iterations,  total number of function evaluations,
    #and total number of gradient evaluations, total number of hessian evaluation, and total number
    #of factorizations  in same file
    df_results_CAT_I_FACTORIZATION_ALL = df_results_CAT_I_FACTORIZATION[:, filter(x -> (x in [PROBLEM_NAME_COLUMN, TOTAL_ITERATIONS_COUNT_COLUMN, TOTAL_FUNCTION_EVALUATION_COLUMN, TOTAL_GRADIENT_EVALUATION_COLUMN, TOTAL_HESSIAN_EVALUATION_COLUMN, TOTAL_FACTORIZATION_EVALUATION_COLUMN]), names(df_results_CAT_I_FACTORIZATION))]
    df_results_CAT_II_FACTORIZATION_ALL = df_results_CAT_II_FACTORIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN, TOTAL_FUNCTION_EVALUATION_COLUMN, TOTAL_GRADIENT_EVALUATION_COLUMN, TOTAL_HESSIAN_EVALUATION_COLUMN, TOTAL_FACTORIZATION_EVALUATION_COLUMN]), names(df_results_CAT_II_FACTORIZATION))]
    df_results_CAT_II_THETA_ZERO_FACTORIZATION_ALL = df_results_CAT_II_THETA_ZERO_FACTORIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN, TOTAL_FUNCTION_EVALUATION_COLUMN, TOTAL_GRADIENT_EVALUATION_COLUMN, TOTAL_HESSIAN_EVALUATION_COLUMN, TOTAL_FACTORIZATION_EVALUATION_COLUMN]), names(df_results_CAT_II_THETA_ZERO_FACTORIZATION))]
    df_results_ARC_FACTORIZATION_OPTIMIZATION_ALL = df_results_ARC_FACTORIZATION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN, TOTAL_FUNCTION_EVALUATION_COLUMN, TOTAL_GRADIENT_EVALUATION_COLUMN, TOTAL_HESSIAN_EVALUATION_COLUMN, TOTAL_FACTORIZATION_EVALUATION_COLUMN]), names(df_results_ARC_FACTORIZATION_OPTIMIZATION))]
    df_results_TRUST_REGION_OPTIMIZATION_ALL = df_results_TRUST_REGION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN, TOTAL_FUNCTION_EVALUATION_COLUMN, TOTAL_GRADIENT_EVALUATION_COLUMN, TOTAL_HESSIAN_EVALUATION_COLUMN, TOTAL_FACTORIZATION_EVALUATION_COLUMN]), names(df_results_TRUST_REGION_OPTIMIZATION))]
    generateALLResultsCSVFile(directoryName, df_results_CAT_I_FACTORIZATION_ALL, df_results_CAT_II_FACTORIZATION_ALL, df_results_CAT_II_THETA_ZERO_FACTORIZATION_ALL, df_results_ARC_FACTORIZATION_OPTIMIZATION_ALL, df_results_TRUST_REGION_OPTIMIZATION_ALL)

    #Generate results for all algorithm for total number of iterations in a separate file
    df_results_CAT_I_FACTORIZATION_ITERATIONS = df_results_CAT_I_FACTORIZATION[:, filter(x -> (x in [PROBLEM_NAME_COLUMN, TOTAL_ITERATIONS_COUNT_COLUMN]), names(df_results_CAT_I_FACTORIZATION))]
    df_results_CAT_II_FACTORIZATION_ITERATIONS = df_results_CAT_II_FACTORIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN]), names(df_results_CAT_II_FACTORIZATION))]
    df_results_CAT_II_THETA_ZERO_FACTORIZATION_ITERATIONS  = df_results_CAT_II_THETA_ZERO_FACTORIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN]), names(df_results_CAT_II_THETA_ZERO_FACTORIZATION))]
    df_results_ARC_FACTORIZATION_OPTIMIZATION_ITERATIONS = df_results_ARC_FACTORIZATION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN]), names(df_results_ARC_FACTORIZATION_OPTIMIZATION))]
    df_results_TRUST_REGION_OPTIMIZATION_ITERATIONS = df_results_TRUST_REGION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN]), names(df_results_TRUST_REGION_OPTIMIZATION))]
    df_results_ITERATIONS = mergeDataFrames(df_results_CAT_I_FACTORIZATION_ITERATIONS, df_results_CAT_II_FACTORIZATION_ITERATIONS, df_results_CAT_II_THETA_ZERO_FACTORIZATION_ITERATIONS , df_results_ARC_FACTORIZATION_OPTIMIZATION_ITERATIONS, df_results_TRUST_REGION_OPTIMIZATION_ITERATIONS)
    saveCSVFile(directoryName, "iterations", df_results_ITERATIONS)

    #Generate results for all algorithm for total number of function evaluations in a separate file
    df_results_CAT_I_FACTORIZATION_FUNCTION = df_results_CAT_I_FACTORIZATION[:, filter(x -> (x in [PROBLEM_NAME_COLUMN, TOTAL_FUNCTION_EVALUATION_COLUMN]), names(df_results_CAT_I_FACTORIZATION))]
    df_results_CAT_II_FACTORIZATION_FUNCTION = df_results_CAT_II_FACTORIZATION[:, filter(x -> (x in [TOTAL_FUNCTION_EVALUATION_COLUMN]), names(df_results_CAT_II_FACTORIZATION))]
    df_results_CAT_II_THETA_ZERO_FACTORIZATION_FUNCTION  = df_results_CAT_II_THETA_ZERO_FACTORIZATION[:, filter(x -> (x in [TOTAL_FUNCTION_EVALUATION_COLUMN]), names(df_results_CAT_II_THETA_ZERO_FACTORIZATION))]
    df_results_ARC_FACTORIZATION_OPTIMIZATION_FUNCTION = df_results_ARC_FACTORIZATION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_FUNCTION_EVALUATION_COLUMN]), names(df_results_ARC_FACTORIZATION_OPTIMIZATION))]
    df_results_TRUST_REGION_OPTIMIZATION_FUNCTION = df_results_TRUST_REGION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_FUNCTION_EVALUATION_COLUMN]), names(df_results_TRUST_REGION_OPTIMIZATION))]
    df_results_FUNCTION = mergeDataFrames(df_results_CAT_I_FACTORIZATION_FUNCTION, df_results_CAT_II_FACTORIZATION_FUNCTION, df_results_CAT_II_THETA_ZERO_FACTORIZATION_FUNCTION , df_results_ARC_FACTORIZATION_OPTIMIZATION_FUNCTION, df_results_TRUST_REGION_OPTIMIZATION_FUNCTION)
    saveCSVFile(directoryName, "functions", df_results_FUNCTION)

    #Generate results for all algorithm for total number of gradient evaluations in a separate file
    df_results_CAT_I_FACTORIZATION_GRADIENT = df_results_CAT_I_FACTORIZATION[:, filter(x -> (x in [PROBLEM_NAME_COLUMN, TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_CAT_I_FACTORIZATION))]
    df_results_CAT_II_FACTORIZATION_GRADIENT = df_results_CAT_II_FACTORIZATION[:, filter(x -> (x in [TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_CAT_II_FACTORIZATION))]
    df_results_CAT_II_THETA_ZERO_FACTORIZATION_GRADIENT = df_results_CAT_II_THETA_ZERO_FACTORIZATION[:, filter(x -> (x in [TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_CAT_II_THETA_ZERO_FACTORIZATION))]
    df_results_ARC_FACTORIZATION_OPTIMIZATION_GRADIENT = df_results_ARC_FACTORIZATION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_ARC_FACTORIZATION_OPTIMIZATION))]
    df_results_TRUST_REGION_OPTIMIZATION_GRADIENT = df_results_TRUST_REGION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_TRUST_REGION_OPTIMIZATION))]
    df_results_GRADIENT = mergeDataFrames(df_results_CAT_I_FACTORIZATION_GRADIENT, df_results_CAT_II_FACTORIZATION_GRADIENT, df_results_CAT_II_THETA_ZERO_FACTORIZATION_GRADIENT, df_results_ARC_FACTORIZATION_OPTIMIZATION_GRADIENT, df_results_TRUST_REGION_OPTIMIZATION_GRADIENT)
    saveCSVFile(directoryName, "gradients", df_results_GRADIENT)

    #Generate results for all algorithm for total number of hessian evaluations in a separate file
    df_results_CAT_I_FACTORIZATION_HESSIAN = df_results_CAT_I_FACTORIZATION[:, filter(x -> (x in [PROBLEM_NAME_COLUMN, TOTAL_HESSIAN_EVALUATION_COLUMN]), names(df_results_CAT_I_FACTORIZATION))]
    df_results_CAT_II_FACTORIZATION_HESSIAN = df_results_CAT_II_FACTORIZATION[:, filter(x -> (x in [TOTAL_HESSIAN_EVALUATION_COLUMN]), names(df_results_CAT_II_FACTORIZATION))]
    df_results_CAT_II_THETA_ZERO_FACTORIZATION_HESSIAN= df_results_CAT_II_THETA_ZERO_FACTORIZATION[:, filter(x -> (x in [TOTAL_HESSIAN_EVALUATION_COLUMN]), names(df_results_CAT_II_THETA_ZERO_FACTORIZATION))]
    df_results_ARC_FACTORIZATION_OPTIMIZATION_HESSIAN = df_results_ARC_FACTORIZATION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_HESSIAN_EVALUATION_COLUMN]), names(df_results_ARC_FACTORIZATION_OPTIMIZATION))]
    df_results_TRUST_REGION_OPTIMIZATION_HESSIAN = df_results_TRUST_REGION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_HESSIAN_EVALUATION_COLUMN]), names(df_results_TRUST_REGION_OPTIMIZATION))]
    df_results_HESSIAN = mergeDataFrames(df_results_CAT_I_FACTORIZATION_HESSIAN, df_results_CAT_II_FACTORIZATION_HESSIAN, df_results_CAT_II_THETA_ZERO_FACTORIZATION_HESSIAN, df_results_ARC_FACTORIZATION_OPTIMIZATION_HESSIAN, df_results_TRUST_REGION_OPTIMIZATION_HESSIAN)
    saveCSVFile(directoryName, "hessian", df_results_HESSIAN)

    #Generate results for all algorithm for total number of factorization evaluations in a separate file
    df_results_CAT_I_FACTORIZATION_TOTAL = df_results_CAT_I_FACTORIZATION[:, filter(x -> (x in [PROBLEM_NAME_COLUMN, TOTAL_FACTORIZATION_EVALUATION_COLUMN]), names(df_results_CAT_I_FACTORIZATION))]
    df_results_CAT_II_FACTORIZATION_TOTAL = df_results_CAT_II_FACTORIZATION[:, filter(x -> (x in [TOTAL_FACTORIZATION_EVALUATION_COLUMN]), names(df_results_CAT_II_FACTORIZATION))]
    df_results_CAT_II_THETA_ZERO_FACTORIZATION_TOTAL = df_results_CAT_II_THETA_ZERO_FACTORIZATION[:, filter(x -> (x in [TOTAL_FACTORIZATION_EVALUATION_COLUMN]), names(df_results_CAT_II_THETA_ZERO_FACTORIZATION))]
    df_results_ARC_FACTORIZATION_OPTIMIZATION_TOTAL = df_results_ARC_FACTORIZATION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_FACTORIZATION_EVALUATION_COLUMN]), names(df_results_ARC_FACTORIZATION_OPTIMIZATION))]
    df_results_TRUST_REGION_OPTIMIZATION_TOTAL= df_results_TRUST_REGION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_FACTORIZATION_EVALUATION_COLUMN]), names(df_results_TRUST_REGION_OPTIMIZATION))]
    df_results_FACTORIZATION = mergeDataFrames(df_results_CAT_I_FACTORIZATION_TOTAL, df_results_CAT_II_FACTORIZATION_TOTAL, df_results_CAT_II_THETA_ZERO_FACTORIZATION_TOTAL, df_results_ARC_FACTORIZATION_OPTIMIZATION_TOTAL, df_results_TRUST_REGION_OPTIMIZATION_TOTAL)
    saveCSVFile(directoryName, "factorization", df_results_FACTORIZATION)

    #Generate results for comparison CAT II for total number of function evaluations in a separate file
    df_results_CAT_II_FACTORIZATION_FUNCTION = df_results_CAT_II_FACTORIZATION[:, filter(x -> (x in [PROBLEM_NAME_COLUMN, TOTAL_FUNCTION_EVALUATION_COLUMN]), names(df_results_CAT_II_FACTORIZATION))]
    df_results_FUNCTION_CAT_II = mergeDataFrames_CAT_II(df_results_CAT_II_FACTORIZATION_FUNCTION, df_results_CAT_II_THETA_ZERO_FACTORIZATION_FUNCTION)
    saveCSVFile(directoryName, "functions_CAT", df_results_FUNCTION_CAT_II)

    #Generate results for comparison CAT II for total number of gradient evaluations in a separate file
    df_results_CAT_II_FACTORIZATION_GRADIENT = df_results_CAT_II_FACTORIZATION[:, filter(x -> (x in [PROBLEM_NAME_COLUMN, TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_CAT_II_FACTORIZATION))]
    df_results_GRADIENT_CAT_II = mergeDataFrames_CAT_II(df_results_CAT_II_FACTORIZATION_GRADIENT, df_results_CAT_II_THETA_ZERO_FACTORIZATION_GRADIENT)
    saveCSVFile(directoryName, "gradients_CAT", df_results_GRADIENT_CAT_II)
end

dir_ = "/Users/fah33/PhD_Research/CAT_RESULTS_BENCHMARK/results_debug_collect_results_script"
collectAllResults(dir_)
# using Plots
#
# # Example data
# solvers = ["CAT II", "ARC", "TRU"]
# fractions = [0.87, 0.82, 0.51]  # Example fractions (replace with actual data)
#
# # Plot the bar chart
# bar(solvers, fractions,  legend=false)
#
# # Add labels and title
# xlabel!("Solver")
# ylabel!("Fraction within 1e-2 of best")
# # title!("Fraction of instances within 1e-2 of best objective value by solver")
#
# # Show the plot
