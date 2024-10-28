"""
  power_iteration(H, num_iter, tol)
  Computes the maximum eigenvalue iteratively for the matrix H.

  # Inputs:
	- `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
		The Hessian at the current iterate x_k.
	- `num_iter::Int64`. Maximum number of iterations to run the power method.
	- `print_level::Float64`. The tolerance for the accuracy of the power method.
  # Output:
   Scalar that has the value of the l2 norm of the matrix H.
"""
function power_iteration(
    H::Union{
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    };
    num_iter::Int64 = 20,
    tol::Float64 = 1e-6,
)
    n = size(H, 1)
    v = rand(n)  # Initialize a random vector
    v /= norm(v)  # Normalize the vector

    λ = 0.0  # Initialize the largest eigenvalue
    for i = 1:num_iter
        Hv = H * v
        v_new = Hv / norm(Hv)  # Normalize the new vector
        λ_new = dot(v_new, H * v_new)  # Rayleigh quotient

        # Check for convergence
        if abs(λ_new - λ) < tol
            break
        end

        v = v_new
        λ = λ_new
    end
    return λ, v
end

"""
  matrix_l2_norm(H, num_iter, tol)
  Computes the l2 norm (the spectral norm) of a matrix.
  The spectral norm of a matrix H is the largest singular value of H (i.e., the square root of the largest eigenvalue of
  the matrix H ^ T * H. However, since the matrix H is symmetrix. We can compute the maximum eigenvalue of H and this
  will be the l2 norm. The computation of the maximum eigenvalue is done iteratively using power iteration method.

  # Inputs:
	- `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
		The Hessian at the current iterate x_k.
	- `num_iter::Int64`. Maximum number of iterations to run the power method.
	- `print_level::Float64`. The tolerance for the accuracy of the power method.
  # Output:
   Scalar that has the value of the l2 norm of the matrix H.
"""
function matrix_l2_norm(
    H::Union{
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    };
    num_iter::Int64 = 20,
    tol::Float64 = 1e-6,
)
    λ_max, _ = power_iteration(H; num_iter = num_iter, tol = tol)  # Largest eigenvalue of A^T A
    return abs(λ_max)  # Largest singular value (spectral norm)
end

@inline function increment!(algorithm_counter::AlgorithmCounter, s::Symbol)
    increment!(algorithm_counter, s, 1)
end

@inline function increment!(algorithm_counter::AlgorithmCounter, s::Symbol, count::Int64)
    increment!(algorithm_counter, Val(s), count)
end

for filed_name in fieldnames(AlgorithmCounter)
    @eval increment!(
        algorithm_counter::AlgorithmCounter,
        ::Val{$(Meta.quot(filed_name))},
        count::Int64,
    ) = algorithm_counter.$filed_name += count
end

function printFailures(problem_name::String, failure_reason_6a::Bool, failure_reason_6b::Bool, failure_reason_6c::Bool, failure_reason_6d::Bool)
    # Create a DataFrame with the specified columns
    df = DataFrame(
        problem_name = problem_name,
        failure_reason_6a = failure_reason_6a,
        failure_reason_6b = failure_reason_6b,
        failure_reason_6c = failure_reason_6c,
        failure_reason_6d = failure_reason_6d
    )

    # Check if file exists to decide whether to append or create a new file
    if isfile("error_reason.csv")
        CSV.write("error_reason.csv", df; append = true, header = false)
    else
        CSV.write("error_reason.csv", df)
    end
end
