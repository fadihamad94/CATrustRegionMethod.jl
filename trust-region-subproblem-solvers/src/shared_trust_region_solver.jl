function sparseMatrixAndTriangle(H::SparseMatrixCSC{Float64,Int})
    return H, :full
end

function sparseMatrixAndTriangle(H::Symmetric{Float64,SparseMatrixCSC{Float64,Int}})
    return parent(H), H.uplo == 'L' ? :lower : :upper
end

function resetSparseCholeskyWorkspace!(workspace::SparseCholeskyWorkspace)
    workspace.factor = nothing
    workspace.matrix = nothing
    workspace.triangle = :none
    workspace.colptr = Int[]
    workspace.rowval = Int[]
    return workspace
end

function prepareSparseCholeskyWorkspace!(
    workspace::SparseCholeskyWorkspace,
    H::Union{SparseMatrixCSC{Float64,Int},Symmetric{Float64,SparseMatrixCSC{Float64,Int}}},
)
    matrix, triangle = sparseMatrixAndTriangle(H)
    if workspace.matrix === matrix && workspace.triangle == triangle
        return workspace
    end

    same_pattern =
        workspace.factor !== nothing &&
        workspace.triangle == triangle &&
        workspace.colptr == matrix.colptr &&
        workspace.rowval == matrix.rowval
    if !same_pattern
        workspace.factor = nothing
    end
    workspace.matrix = matrix
    workspace.triangle = triangle
    # Outer iterations replace Hessian objects, so retain their structure without copying it.
    workspace.colptr = matrix.colptr
    workspace.rowval = matrix.rowval
    return workspace
end

function factorizeShiftedHessian!(::Nothing, H::Matrix{Float64}, shift::Float64)
    return cholesky(H + shift * I; check = false)
end

function factorizeShiftedHessian!(
    workspace::SparseCholeskyWorkspace,
    H::Matrix{Float64},
    shift::Float64,
)
    if workspace.dense_buffer === nothing || size(workspace.dense_buffer) != size(H)
        workspace.dense_buffer = similar(H)
    end
    buffer = workspace.dense_buffer
    copyto!(buffer, H)
    @inbounds for index in axes(buffer, 1)
        buffer[index, index] += shift
    end
    return cholesky!(Hermitian(buffer, :U); check = false)
end

function factorizeShiftedHessian!(
    ::Nothing,
    H::Union{SparseMatrixCSC{Float64,Int},Symmetric{Float64,SparseMatrixCSC{Float64,Int}}},
    shift::Float64,
)
    return cholesky(H; shift = shift, check = false)
end

function factorizeShiftedHessian!(
    workspace::SparseCholeskyWorkspace,
    H::Union{SparseMatrixCSC{Float64,Int},Symmetric{Float64,SparseMatrixCSC{Float64,Int}}},
    shift::Float64,
)
    if !workspace.reuse_sparse_symbolic_factorization
        return cholesky(H; shift = shift, check = false)
    end
    prepareSparseCholeskyWorkspace!(workspace, H)
    if workspace.factor === nothing
        workspace.factor = cholesky(H; shift = shift, check = false)
        workspace.symbolic_factorizations += 1
    else
        cholesky!(workspace.factor, H; shift = shift, check = false)
    end
    return workspace.factor
end

function negativeGradient!(workspace::Nothing, g::Vector{Float64})
    return -g
end

function negativeGradient!(workspace::SparseCholeskyWorkspace, g::Vector{Float64})
    resize!(workspace.negative_gradient, length(g))
    @. workspace.negative_gradient = -g
    return workspace.negative_gradient
end

function solveFactorizedSystem(
    factorization::Factorization{Float64},
    g::Vector{Float64},
    workspace::OptionalSparseCholeskyWorkspace,
)
    direction = similar(g)
    ldiv!(direction, factorization, negativeGradient!(workspace, g))
    return direction
end

function shiftedResidualBuffer(workspace::Nothing, g::Vector{Float64})
    return similar(g)
end

function shiftedResidualBuffer(workspace::SparseCholeskyWorkspace, g::Vector{Float64})
    resize!(workspace.shifted_residual, length(g))
    return workspace.shifted_residual
end

function iterativeRefinementBuffers(workspace::Nothing, g::Vector{Float64})
    return shiftedResidualBuffer(workspace, g), similar(g)
end

function iterativeRefinementBuffers(
    workspace::SparseCholeskyWorkspace,
    g::Vector{Float64},
)
    resize!(workspace.refinement_correction, length(g))
    return shiftedResidualBuffer(workspace, g), workspace.refinement_correction
end

iterativeRefinementMaxIterations(::Nothing) =
    DEFAULT_ITERATIVE_REFINEMENT_MAX_ITERATIONS
iterativeRefinementMaxIterations(workspace::SparseCholeskyWorkspace) =
    workspace.iterative_refinement_max_iterations

function floatingPointSafeBallProjection!(d::Vector{Float64}, radius::Float64)
    if radius < 0.0
        throw(ArgumentError("Trust-region radius must be nonnegative."))
    end

    d_norm = norm(d)
    if d_norm <= radius
        return d
    end

    d .*= radius / d_norm
    projected_norm = norm(d)
    if projected_norm > radius
        # Scaling by radius / ||d|| can still round just outside the ball. Use the
        # previous representable Float64 scale factor to guarantee the returned step is feasible.
        d .*= prevfloat(radius / projected_norm)
    end
    return d
end

function floatingPointSafeBallProjection(d::Vector{Float64}, radius::Float64)
    return floatingPointSafeBallProjection!(copy(d), radius)
end
