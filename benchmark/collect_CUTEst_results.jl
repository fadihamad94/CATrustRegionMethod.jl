using CSV, DataFrames

const CAT_FACTORIZATION = "CAT_FACTORIZATION"
const CAT_THETA_ZERO_FACTORIZATION = "CAT_THETA_ZERO_FACTORIZATION"
const ARC_FACTORIZATION = "ARC_FACTORIZATION"
const NEWTON_TRUST_REGION= "NewtonTrustRegion"

const CAT_OPTIMIZATION_METHOD = "CAT"
const CAT_THETA_ZERO_OPTIMIZATION_METHOD = "CAT_THETA_ZERO"
const ARC_FACTORIZATION_OPTIMIZATION_METHOD = "ARC_FACTORIZATION"
const NEWTON_TRUST_REGION_OPTIMIZATION_METHOD = "NewtonTrustRegion"

const CURRENT_DIRECTORY = dirname(Base.source_path())
const RESULTS_DEFAULT_DIRECTORY_NAME = string(CURRENT_DIRECTORY, "/../results")

const PROBLEM_NAME_COLUMN = "problem_name"
const TOTAL_ITERATIONS_COUNT_COLUMN = "total_iterations_count"
const TOTAL_FUNCTION_EVALUATION_COLUMN = "total_function_evaluation"
const TOTAL_GRADIENT_EVALUATION_COLUMN = "total_gradient_evaluation"

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
    total_results_output_file_name = "table_cutest_$optimization_method.csv"
    total_results_file_path = string(total_results_output_directory, "/", total_results_output_file_name)
    return total_results_file_path
end

function collectResultsPerSolver(directoryName::String, optimization_method::String)
    total_results_file_path = buildFilePath(directoryName, optimization_method)
    if isfile(total_results_file_path)
        return readFile(total_results_file_path)
    else
        total_results_file_path = buildFilePath(RESULTS_DEFAULT_DIRECTORY_NAME, optimization_method)
        return readFile(total_results_file_path)
    end
end

function mergeDataFrames(df_results_CAT_FACTORIZATION_CRITERIA::DataFrame, df_results_CAT_THETA_ZERO_FACTORIZATION_CRITERIA::DataFrame, df_results_ARC_FACTORIZATION_OPTIMIZATION_CRITERIA::DataFrame, df_results_TRUST_REGION_OPTIMIZATION_CRITERIA::DataFrame)
    df = DataFrame(PROBLEM_NAME = [], CAT_FACTORIZATION = [], CAT_THETA_ZERO_FACTORIZATION = [], ARC_FACTORIZATION = [], NEWTON_TRUST_REGION = [])
    matrix_results_CAT_FACTORIZATION_CRITERIA = Matrix(df_results_CAT_FACTORIZATION_CRITERIA)
    matrix_results_CAT_THETA_ZERO_FACTORIZATION_CRITERIA = Matrix(df_results_CAT_THETA_ZERO_FACTORIZATION_CRITERIA)
    matrix_results_ARC_FACTORIZATION_OPTIMIZATION_CRITERIA = Matrix(df_results_ARC_FACTORIZATION_OPTIMIZATION_CRITERIA)
    matrix_results_TRUST_REGION_OPTIMIZATION_CRITERIA = Matrix(df_results_TRUST_REGION_OPTIMIZATION_CRITERIA)

    for row in 1:size(matrix_results_CAT_FACTORIZATION_CRITERIA)[1]
        merged_vector = vcat(matrix_results_CAT_FACTORIZATION_CRITERIA[row, :], matrix_results_CAT_THETA_ZERO_FACTORIZATION_CRITERIA[row, :], matrix_results_ARC_FACTORIZATION_OPTIMIZATION_CRITERIA[row, :], matrix_results_TRUST_REGION_OPTIMIZATION_CRITERIA[row, :])
        push!(df, merged_vector)
    end

    return df
end

function saveCSVFile(directoryName::String, criteria::String, results::DataFrame)
    results_file_path = string(directoryName, "/", "all_algorithm_results_$criteria.csv")
    CSV.write(results_file_path, results, header = true)
end

function generateALLResultsCSVFile(directoryName::String, df_results_CAT_FACTORIZATION::DataFrame, df_results_CAT_THETA_ZERO_FACTORIZATION::DataFrame, df_results_ARC_FACTORIZATION_OPTIMIZATION::DataFrame, df_results_TRUST_REGION_OPTIMIZATION::DataFrame)
    results = DataFrame(PROBLEM_NAME = [], CAT_FACTORIZATION_ITR = [], CAT_FACTORIZATION_F = [], CAT_FACTORIZATION_G = [], CAT_THETA_ZERO_FACTORIZATION_ITR = [], CAT_THETA_ZERO_FACTORIZATION_F = [], CAT_THETA_ZERO_FACTORIZATION_G= [], ARC_FACTORIZATION_ITR = [], ARC_FACTORIZATION_F = [], ARC_FACTORIZATION_G = [], NEWTON_TRUST_REGION_ITR = [], NEWTON_TRUST_REGION_F = [], NEWTON_TRUST_REGION_G = [])
    matrix_results_CAT_FACTORIZATION = Matrix(df_results_CAT_FACTORIZATION)
    matrix_results_CAT_THETA_ZERO_FACTORIZATION = Matrix(df_results_CAT_THETA_ZERO_FACTORIZATION)
    matrix_results_ARC_FACTORIZATION_OPTIMIZATION = Matrix(df_results_ARC_FACTORIZATION_OPTIMIZATION)
    matrix_results_TRUST_REGION_OPTIMIZATION = Matrix(df_results_TRUST_REGION_OPTIMIZATION)

    for row in 1:size(matrix_results_CAT_FACTORIZATION)[1]
        merged_vector = vcat(matrix_results_CAT_FACTORIZATION[row, :], matrix_results_CAT_THETA_ZERO_FACTORIZATION[row, :], matrix_results_ARC_FACTORIZATION_OPTIMIZATION[row, :], matrix_results_TRUST_REGION_OPTIMIZATION[row, :])
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
    #Collect results for CAT
    optimization_method = CAT_OPTIMIZATION_METHOD
    df_results_CAT_FACTORIZATION = collectResultsPerSolver(directoryName, optimization_method)

    #Collect results for CAT θ = 0.0 Factorizarion
    optimization_method = CAT_THETA_ZERO_OPTIMIZATION_METHOD
    df_results_CAT_THETA_ZERO_FACTORIZATION = collectResultsPerSolver(directoryName, optimization_method)

    #Collect results for ARC g-rule
    optimization_method = ARC_FACTORIZATION_OPTIMIZATION_METHOD
    df_results_ARC_FACTORIZATION_OPTIMIZATION = collectResultsPerSolver(directoryName, optimization_method)

    #Collect results for Newton Trust Region
    optimization_method = NEWTON_TRUST_REGION_OPTIMIZATION_METHOD
    df_results_TRUST_REGION_OPTIMIZATION = collectResultsPerSolver(directoryName, optimization_method)

    #Generate results for all algorithm  for total number of iterations,  total number of function evaluations,
    #and total number of gradient evaluations in same file
    df_results_CAT_FACTORIZATION_ALL = df_results_CAT_FACTORIZATION[:, filter(x -> (x in [PROBLEM_NAME_COLUMN, TOTAL_ITERATIONS_COUNT_COLUMN, TOTAL_FUNCTION_EVALUATION_COLUMN, TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_CAT_FACTORIZATION))]
    df_results_CAT_THETA_ZERO_FACTORIZATION_ALL = df_results_CAT_THETA_ZERO_FACTORIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN, TOTAL_FUNCTION_EVALUATION_COLUMN, TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_CAT_THETA_ZERO_FACTORIZATION))]
    df_results_ARC_FACTORIZATION_OPTIMIZATION_ALL = df_results_ARC_FACTORIZATION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN, TOTAL_FUNCTION_EVALUATION_COLUMN, TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_ARC_FACTORIZATION_OPTIMIZATION))]
    df_results_TRUST_REGION_OPTIMIZATION_ALL = df_results_TRUST_REGION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN, TOTAL_FUNCTION_EVALUATION_COLUMN, TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_TRUST_REGION_OPTIMIZATION))]
    generateALLResultsCSVFile(directoryName, df_results_CAT_FACTORIZATION_ALL, df_results_CAT_THETA_ZERO_FACTORIZATION_ALL, df_results_ARC_FACTORIZATION_OPTIMIZATION_ALL, df_results_TRUST_REGION_OPTIMIZATION_ALL)

    #Generate results for all algorithm for total number of iterations in a separate file
    df_results_CAT_FACTORIZATION_ITERATIONS = df_results_CAT_FACTORIZATION[:, filter(x -> (x in [PROBLEM_NAME_COLUMN, TOTAL_ITERATIONS_COUNT_COLUMN]), names(df_results_CAT_FACTORIZATION))]
    df_results_CAT_THETA_ZERO_FACTORIZATION_ITERATIONS = df_results_CAT_THETA_ZERO_FACTORIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN]), names(df_results_CAT_THETA_ZERO_FACTORIZATION))]
    df_results_ARC_FACTORIZATION_OPTIMIZATION_ITERATIONS = df_results_ARC_FACTORIZATION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN]), names(df_results_ARC_FACTORIZATION_OPTIMIZATION))]
    df_results_TRUST_REGION_OPTIMIZATION_ITERATIONS = df_results_TRUST_REGION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_ITERATIONS_COUNT_COLUMN]), names(df_results_TRUST_REGION_OPTIMIZATION))]
    df_results_ITERATIONS = mergeDataFrames(df_results_CAT_FACTORIZATION_ITERATIONS, df_results_CAT_THETA_ZERO_FACTORIZATION_ITERATIONS, df_results_ARC_FACTORIZATION_OPTIMIZATION_ITERATIONS, df_results_TRUST_REGION_OPTIMIZATION_ITERATIONS)
    saveCSVFile(directoryName, "iterations", df_results_ITERATIONS)

    #Generate results for all algorithm for total number of function evaluations in a separate file
    df_results_CAT_FACTORIZATION_FUNCTION = df_results_CAT_FACTORIZATION[:, filter(x -> (x in [PROBLEM_NAME_COLUMN, TOTAL_FUNCTION_EVALUATION_COLUMN]), names(df_results_CAT_FACTORIZATION))]
    df_results_CAT_THETA_ZERO_FACTORIZATION_FUNCTION = df_results_CAT_THETA_ZERO_FACTORIZATION[:, filter(x -> (x in [TOTAL_FUNCTION_EVALUATION_COLUMN]), names(df_results_CAT_THETA_ZERO_FACTORIZATION))]
    df_results_ARC_FACTORIZATION_OPTIMIZATION_FUNCTION = df_results_ARC_FACTORIZATION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_FUNCTION_EVALUATION_COLUMN]), names(df_results_ARC_FACTORIZATION_OPTIMIZATION))]
    df_results_TRUST_REGION_OPTIMIZATION_FUNCTION = df_results_TRUST_REGION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_FUNCTION_EVALUATION_COLUMN]), names(df_results_TRUST_REGION_OPTIMIZATION))]
    df_results_FUNCTION = mergeDataFrames(df_results_CAT_FACTORIZATION_FUNCTION, df_results_CAT_THETA_ZERO_FACTORIZATION_FUNCTION, df_results_ARC_FACTORIZATION_OPTIMIZATION_FUNCTION, df_results_TRUST_REGION_OPTIMIZATION_FUNCTION)
    saveCSVFile(directoryName, "functions", df_results_FUNCTION)

    #Generate results for all algorithm for total number of gradient evaluations in a separate file
    df_results_CAT_FACTORIZATION_GRADIENT = df_results_CAT_FACTORIZATION[:, filter(x -> (x in [PROBLEM_NAME_COLUMN, TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_CAT_FACTORIZATION))]
    df_results_CAT_THETA_ZERO_FACTORIZATION_GRADIENT = df_results_CAT_THETA_ZERO_FACTORIZATION[:, filter(x -> (x in [TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_CAT_THETA_ZERO_FACTORIZATION))]
    df_results_ARC_FACTORIZATION_OPTIMIZATION_GRADIENT = df_results_ARC_FACTORIZATION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_ARC_FACTORIZATION_OPTIMIZATION))]
    df_results_TRUST_REGION_OPTIMIZATION_GRADIENT = df_results_TRUST_REGION_OPTIMIZATION[:, filter(x -> (x in [TOTAL_GRADIENT_EVALUATION_COLUMN]), names(df_results_TRUST_REGION_OPTIMIZATION))]
    df_results_GRADIENT = mergeDataFrames(df_results_CAT_FACTORIZATION_GRADIENT, df_results_CAT_THETA_ZERO_FACTORIZATION_GRADIENT, df_results_ARC_FACTORIZATION_OPTIMIZATION_GRADIENT, df_results_TRUST_REGION_OPTIMIZATION_GRADIENT)
    saveCSVFile(directoryName, "gradients", df_results_GRADIENT)
end
