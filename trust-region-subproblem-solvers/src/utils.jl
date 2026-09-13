"""
  matrix_l2_norm(H, num_iter, tol)
  Estimate the l2 norm (spectral norm) of a matrix by applying
  power iteration to `H' * H` without forming that product.

  # Inputs:
	- `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
		The Hessian at the current iterate x_k.
	- `num_iter::Int64`. Maximum number of iterations to run the power method.
	- `tol::Float64`. The tolerance for the accuracy of the power method.
  # Output:
   A nonnegative estimate of the spectral norm of `H`.
"""
function maximumAbsoluteEntry(H::Matrix{Float64})::Float64
    return maximum(abs, H)
end

function maximumAbsoluteEntry(H::SparseMatrixCSC{Float64,Int})::Float64
    values = nonzeros(H)
    return isempty(values) ? 0.0 : maximum(abs, values)
end

function maximumAbsoluteEntry(
    H::Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
)::Float64
    data = parent(H)
    values = nonzeros(data)
    rows = rowvals(data)
    use_upper = H.uplo == 'U'
    scale = 0.0
    for column in axes(data, 2)
        for index in nzrange(data, column)
            row = rows[index]
            if use_upper ? row <= column : row >= column
                scale = max(scale, abs(values[index]))
            end
        end
    end
    return scale
end

function matrix_l2_norm(
    H::HessianMatrix;
    num_iter::Int64 = DEFAULT_POWER_ITERATION_MAX_ITERATIONS,
    tol::Float64 = DEFAULT_POWER_ITERATION_TOLERANCE,
)
    size(H, 1) == size(H, 2) || throw(DimensionMismatch("H must be square."))
    num_iter > 0 || throw(ArgumentError("num_iter must be positive."))
    isfinite(tol) && tol >= 0.0 ||
        throw(ArgumentError("tol must be finite and nonnegative."))
    finiteHessian(H) || throw(ArgumentError("H must be finite."))
    n = size(H, 1)
    n == 0 && return 0.0
    matrix_scale = maximumAbsoluteEntry(H)
    matrix_scale == 0.0 && return 0.0
    scaled_H =
        H isa Symmetric ?
        Symmetric(
            parent(H) / matrix_scale,
            H.uplo == 'L' ? :L : :U,
        ) :
        H / matrix_scale

    v = randn(n)
    v_norm = norm(v)
    while v_norm == 0.0
        randn!(v)
        v_norm = norm(v)
    end
    v ./= v_norm

    previous_estimate = Inf
    for _ = 1:num_iter
        Hv = scaled_H * v
        estimate = norm(Hv)
        estimate == 0.0 && return 0.0
        isfinite(estimate) ||
            throw(OverflowError("The spectral-norm iteration produced a nonfinite value."))
        if isfinite(previous_estimate) &&
           abs(estimate - previous_estimate) <=
           tol * max(estimate, previous_estimate)
            result = matrix_scale * estimate
            isfinite(result) || throw(
                OverflowError("The matrix spectral norm is not representable as Float64."),
            )
            return result
        end

        # Alternating normalized products by H and H' is power iteration on
        # H' * H without forming or squaring the matrix. Scaling H first avoids
        # overflowing the normal product for large but finite matrices.
        Hv ./= estimate
        normal_product_v = transpose(scaled_H) * Hv
        right_norm = norm(normal_product_v)
        isfinite(right_norm) ||
            throw(OverflowError("The spectral-norm iteration produced a nonfinite value."))
        right_norm == 0.0 && return matrix_scale * estimate
        v .= normal_product_v ./ right_norm
        previous_estimate = estimate
    end

    final_estimate = norm(scaled_H * v)
    isfinite(final_estimate) ||
        throw(OverflowError("The spectral-norm iteration produced a nonfinite value."))
    result = matrix_scale * final_estimate
    isfinite(result) ||
        throw(OverflowError("The matrix spectral norm is not representable as Float64."))
    return result
end

function effectiveHessianDensity(H::SparseMatrixCSC{Float64,Int})
    isempty(H) && return 0.0
    return nnz(H) / length(H)
end


function effectiveHessianDensity(
    H::Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
)
    isempty(H) && return 0.0
    data = parent(H)
    rows = rowvals(data)
    effective_nnz = 0
    use_upper = H.uplo == 'U'
    for column in axes(data, 2)
        for index in nzrange(data, column)
            row = rows[index]
            active = use_upper ? row <= column : row >= column
            if active
                effective_nnz += row == column ? 1 : 2
            end
        end
    end
    return effective_nnz / length(H)
end


function selectHessianRepresentation(H::Matrix{Float64}, threshold::Float64)
    0.0 <= threshold <= 1.0 ||
        throw(ArgumentError("Hessian density threshold must be in [0, 1]."))
    return H
end


function selectHessianRepresentation(
    H::SparseMatrixCSC{Float64,Int},
    threshold::Float64,
)
    0.0 <= threshold <= 1.0 ||
        throw(ArgumentError("Hessian density threshold must be in [0, 1]."))
    return effectiveHessianDensity(H) > threshold ? Matrix(H) : H
end


function selectHessianRepresentation(
    H::Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
    threshold::Float64,
)
    0.0 <= threshold <= 1.0 ||
        throw(ArgumentError("Hessian density threshold must be in [0, 1]."))
    2 * nnz(parent(H)) <= threshold * length(H) && return H
    return effectiveHessianDensity(H) > threshold ? Matrix(H) : H
end

const SUBPROBLEM_FAILURE_DETAILS_DIRECTORY = Ref{Union{Nothing,String}}(nothing)
const SUBPROBLEM_FAILURE_DIAGNOSTIC_CONTEXT = Ref{Dict{String,Any}}(Dict{String,Any}())

function setSubproblemFailureDetailsDirectory(path::Union{Nothing,String})
    SUBPROBLEM_FAILURE_DETAILS_DIRECTORY[] = path
    return nothing
end

function hasSubproblemFailureDetailsDirectory()
    return SUBPROBLEM_FAILURE_DETAILS_DIRECTORY[] !== nothing
end

function getSubproblemFailureDetailsDirectory()
    return something(SUBPROBLEM_FAILURE_DETAILS_DIRECTORY[], ".")
end

function subproblemFailureDetailsPath(file_name::String)
    directory = getSubproblemFailureDetailsDirectory()
    mkpath(directory)
    return joinpath(directory, file_name)
end

function counterValue(algorithm_counter, field::Symbol)
    return hasproperty(algorithm_counter, field) ? getproperty(algorithm_counter, field) : 0
end

function algorithmCounterSnapshot(algorithm_counter)
    return Dict{String,Any}(
        "total_function_evaluation" =>
            counterValue(algorithm_counter, :total_function_evaluation),
        "total_gradient_evaluation" =>
            counterValue(algorithm_counter, :total_gradient_evaluation),
        "total_hessian_evaluation" =>
            counterValue(algorithm_counter, :total_hessian_evaluation),
        "total_number_subproblem_iterations" =>
            counterValue(algorithm_counter, :total_number_subproblem_iterations),
        "total_number_hessian_vector_products" =>
            counterValue(
                algorithm_counter,
                :total_number_hessian_vector_products,
            ),
        "total_number_factorizations" =>
            counterValue(algorithm_counter, :total_number_factorizations),
        "total_number_factorizations_findinterval" =>
            counterValue(algorithm_counter, :total_number_factorizations_findinterval),
        "total_number_factorizations_bisection" =>
            counterValue(algorithm_counter, :total_number_factorizations_bisection),
        "total_number_factorizations_compute_search_direction" =>
            counterValue(
                algorithm_counter,
                :total_number_factorizations_compute_search_direction,
            ),
        "total_number_factorizations_inverse_power_iteration" =>
            counterValue(
                algorithm_counter,
                :total_number_factorizations_inverse_power_iteration,
            ),
    )
end

function setSubproblemFailureDiagnosticContext(;
    iteration::Union{Nothing,Int64} = nothing,
    source_optimizer::Union{Nothing,String} = nothing,
    inner_trial::Union{Nothing,Int64} = nothing,
    algorithm_counter = nothing,
)
    context = Dict{String,Any}(
        "iteration" => iteration,
        "source_optimizer" => source_optimizer,
        "inner_trial" => inner_trial,
    )
    if algorithm_counter !== nothing
        context["algorithm_counter_before_validation"] =
            algorithmCounterSnapshot(algorithm_counter)
    end
    SUBPROBLEM_FAILURE_DIAGNOSTIC_CONTEXT[] = context
    return nothing
end

function getSubproblemFailureDiagnosticContext()
    return copy(SUBPROBLEM_FAILURE_DIAGNOSTIC_CONTEXT[])
end

function nextSubproblemFailureDetailsStem(problem_name::String)
    directory = getSubproblemFailureDetailsDirectory()
    mkpath(directory)
    prefix = "$(problem_name)_failure_"
    suffix = "_info.json"
    last_index = 0
    for file_name in readdir(directory)
        if startswith(file_name, prefix) && endswith(file_name, suffix)
            start_index = ncodeunits(prefix) + 1
            end_index = ncodeunits(file_name) - ncodeunits(suffix)
            failure_index = tryparse(Int, file_name[start_index:end_index])
            if failure_index !== nothing
                last_index = max(last_index, failure_index)
            end
        end
    end
    failure_index = last_index + 1
    return failure_index, joinpath(directory, "$(prefix)$(failure_index)")
end

function writeSparseMatrixTriplet(file_path::String, H)
    sparse_H = sparse(H)
    rows, cols, vals = findnz(sparse_H)
    open(file_path, "w") do io
        println(io, "row,col,value")
        for (row, col, val) in zip(rows, cols, vals)
            println(io, "$row,$col,$val")
        end
    end
    return nnz(sparse_H)
end

function printFailures(
    problem_name::String,
    failure_reason_6a::Bool,
    failure_reason_6b::Bool,
    failure_reason_6c::Bool,
    failure_reason_6d::Bool,
)
    # Create a DataFrame with the specified columns
    df = DataFrame(
        problem_name = problem_name,
        failure_reason_6a = failure_reason_6a,
        failure_reason_6b = failure_reason_6b,
        failure_reason_6c = failure_reason_6c,
        failure_reason_6d = failure_reason_6d,
    )

    file_path = subproblemFailureDetailsPath("error_reason.csv")

    # Check if file exists to decide whether to append or create a new file
    if isfile(file_path)
        CSV.write(file_path, df; append = true, header = false)
    else
        CSV.write(file_path, df)
    end
end


function restoreFullMatrix(
    H::Union{
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    },
)
    nmbRows = size(H)[1]
    numbColumns = size(H)[2]
    for i = 1:nmbRows
        for j = i:numbColumns
            H[i, j] = H[j, i]
        end
    end
    return H
end
