const UTRNLPData = Union{NLPModels.AbstractNLPModel,MathOptInterface.NLPBlockData}

struct InvalidHessianError <: Exception
    message::String
end

Base.showerror(io::IO, error::InvalidHessianError) =
    print(io, error.message)

function _inverseRadiusCoefficient(ρ::Float64, denominator::Float64)
    return ρ > 1.0 ? inv(ρ) / denominator : inv(denominator * ρ)
end

function _scaledRadius(
    gradient_scale::Float64,
    ρ::Float64,
    denominator::Float64,
)
    return ρ > 1.0 ?
           (gradient_scale / ρ) / denominator :
           gradient_scale / (denominator * ρ)
end

"""
    selectUTRSubproblemParameters(H, gradient_norm, ρ, counter, workspace)

Apply the shifted-Cholesky rule from `docs/adaptive_utr_implementation.tex`.
The returned `σ` and `r` are coefficients. The actual Hessian shift is
`σ * sqrt(gradient_norm)`, and the actual trust-region radius is
`r * sqrt(gradient_norm)`.
"""
function selectUTRSubproblemParameters(
    H::HessianMatrix,
    gradient_norm::Float64,
    ρ::Float64,
    algorithm_counter::AlgorithmCounter = AlgorithmCounter(),
    workspace::OptionalSparseCholeskyWorkspace = SparseCholeskyWorkspace(),
)
    isfinite(gradient_norm) && gradient_norm >= 0.0 ||
        throw(ArgumentError("The gradient norm must be finite and nonnegative."))
    isfinite(ρ) && ρ > 0.0 ||
        throw(ArgumentError("The penalty parameter must be finite and positive."))

    gradient_scale = sqrt(gradient_norm)
    τ = ρ * gradient_scale
    isfinite(τ) ||
        throw(OverflowError("The UTR shifted-Cholesky threshold overflowed."))

    recordParameterSelectionFactorizations!(algorithm_counter)
    plus_factorization = factorizeShiftedHessian!(workspace, H, τ)
    plus_is_positive_definite = issuccess(plus_factorization)

    selection_case = 1
    if plus_is_positive_definite
        recordParameterSelectionFactorizations!(algorithm_counter)
        minus_factorization = factorizeShiftedHessian!(workspace, H, -τ)
        selection_case = issuccess(minus_factorization) ? 1 : 2
    end

    σ = selection_case == 1 ? 0.0 : ρ
    denominator = selection_case == 1 ? 2.0 : 4.0
    r = _inverseRadiusCoefficient(ρ, denominator)
    hessian_shift = selection_case == 1 ? 0.0 : τ
    radius = _scaledRadius(gradient_scale, ρ, denominator)
    return (
        selection_case = selection_case,
        σ = σ,
        r = r,
        gradient_scale = gradient_scale,
        τ = τ,
        hessian_shift = hessian_shift,
        radius = radius,
    )
end

"""
    regularizeHessian(H, shift)

Return the Hessian used by the shared trust-region subproblem. `shift` is the
already-scaled value `σ * sqrt(norm(g))`.
"""
function regularizeHessian(H::HessianMatrix, shift::Float64)
    isfinite(shift) && shift >= 0.0 ||
        throw(
            ArgumentError(
                "The Hessian regularization must be finite and nonnegative.",
            ),
        )
    shift == 0.0 && return H
    return H + shift * I
end

function _hasSufficientDecrease(
    objective_change::Float64,
    gradient_norm::Float64,
    ρ::Float64,
    η::Float64,
)
    objective_change > 0.0 && return false
    gradient_norm == 0.0 && return true

    reduction = -objective_change
    reduction > 0.0 || return false
    isinf(reduction) && return true

    # This ordering handles the common case without forming G^(3/2). If an
    # intermediate overflows or underflows, compare logarithms instead.
    threshold =
        η * (gradient_norm / ρ) * sqrt(gradient_norm)
    if isfinite(threshold) && threshold > 0.0
        return reduction >= threshold
    end
    return log2(reduction) >=
           log2(η) - log2(ρ) + 1.5 * log2(gradient_norm)
end

"""
    trialAccepted(f_current, f_trial, gradient_norm, trial_gradient_norm, ρ, η, ξ)

Evaluate the monotonicity and first-order acceptance test from the local UTR
implementation description.
"""
function trialAccepted(
    f_current::Float64,
    f_trial::Float64,
    gradient_norm::Float64,
    trial_gradient_norm::Float64,
    ρ::Float64,
    η::Float64,
    ξ::Float64,
)
    objective_change = f_trial - f_current
    monotone = objective_change <= 0.0
    sufficient_decrease =
        _hasSufficientDecrease(objective_change, gradient_norm, ρ, η)
    gradient_contraction = trial_gradient_norm <= ξ * gradient_norm
    return monotone && (sufficient_decrease || gradient_contraction)
end

function rejectedPenalty(ρ::Float64, μ_1::Float64)
    next_ρ = ρ * μ_1
    isfinite(next_ρ) ||
        throw(OverflowError("The UTR penalty parameter overflowed."))
    return next_ρ
end

function acceptedPenalty(ρ::Float64, ρ_min::Float64, μ_2::Float64)
    return max(ρ_min, ρ / μ_2)
end

function _finite_scalar(value)
    return value isa Real && isfinite(value)
end

function _finite_vector(value)
    return value isa AbstractVector && all(isfinite, value)
end

_finite_matrix(H::SparseMatrixCSC) = all(isfinite, nonzeros(H))
_finite_matrix(H::Symmetric{<:Any,<:SparseMatrixCSC}) =
    all(isfinite, nonzeros(parent(H)))
_finite_matrix(H::AbstractMatrix) = all(isfinite, H)

function _normalizeHessian(H, dimension::Int)
    H isa AbstractMatrix ||
        throw(InvalidHessianError("The Hessian must be a real matrix."))
    size(H) == (dimension, dimension) ||
        throw(
            InvalidHessianError(
                "The Hessian dimension does not match the number of variables.",
            ),
        )
    eltype(H) <: Real ||
        throw(InvalidHessianError("The Hessian must be real-valued."))

    normalized = if H isa Matrix{Float64} ||
                    H isa SparseMatrixCSC{Float64,Int} ||
                    H isa Symmetric{
                        Float64,
                        SparseMatrixCSC{Float64,Int},
                    }
        H
    elseif H isa SparseMatrixCSC
        SparseMatrixCSC{Float64,Int}(H)
    elseif H isa Symmetric && parent(H) isa SparseMatrixCSC
        Symmetric(
            SparseMatrixCSC{Float64,Int}(parent(H)),
            H.uplo,
        )
    else
        Matrix{Float64}(H)
    end

    _finite_matrix(normalized) ||
        throw(DomainError("Hessian", "The Hessian must be finite."))
    issymmetric(normalized) ||
        throw(InvalidHessianError("The Hessian must be symmetric."))
    return normalized
end

function evalFunction(
    nlp::UTRNLPData,
    x::Vector{Float64},
    algorithm_counter::AlgorithmCounter,
)
    return Float64(evaluate_function(nlp, x, algorithm_counter))
end

function evalGradient(
    nlp::UTRNLPData,
    x::Vector{Float64},
    algorithm_counter::AlgorithmCounter,
)
    return Vector{Float64}(evaluate_gradient(nlp, x, algorithm_counter))
end

function evalHessian(
    nlp::UTRNLPData,
    x::Vector{Float64},
    algorithm_counter::AlgorithmCounter,
)
    constraint_count =
        nlp isa NLPModels.AbstractNLPModel ? 0 : length(nlp.constraint_bounds)
    normalize_hessian = H -> _normalizeHessian(restoreFullMatrix(H), length(x))
    return evaluate_hessian(
        nlp,
        x,
        algorithm_counter;
        constraint_count,
        normalize_nlp = H -> _normalizeHessian(H, length(x)),
        normalize_moi = normalize_hessian,
    )
end

function _problemName(nlp::UTRNLPData)
    if nlp isa NLPModels.AbstractNLPModel
        return isempty(nlp.meta.name) ? "Generic" : nlp.meta.name
    end
    return "Generic"
end

function solveUTRSubproblem(
    nlp::UTRNLPData,
    g::Vector{Float64},
    H::HessianMatrix,
    warm_shift::Float64,
    radius::Float64,
    gradient_norm::Float64,
    algorithm_params::AlgorithmicParameters,
    algorithm_counter::AlgorithmCounter,
    workspace::OptionalSparseCholeskyWorkspace,
)
    increment!(algorithm_counter, :total_number_subproblem_solves)
    result = solveTrustRegionSubproblem(
        _problemName(nlp),
        g,
        H,
        warm_shift,
        algorithm_params.γ_1,
        algorithm_params.γ_2,
        algorithm_params.γ_3,
        radius,
        gradient_norm;
        print_level = algorithm_params.print_level,
        use_backup_trust_region_subproblem_solver =
            algorithm_params.use_backup_trust_region_subproblem_solver,
        sparse_cholesky_workspace = workspace,
        handle_hard_case = algorithm_params.handle_hard_case,
    )
    recordFactorizationStats!(algorithm_counter, result.factorizations)
    return result
end

function UTR_solve(model::JuMP.Model)
    return UTR_solve(
        model,
        TerminationCriteria(),
        AlgorithmicParameters(),
    )
end

function UTR_solve(
    model::JuMP.Model,
    termination_criteria::TerminationCriteria,
    algorithm_params::AlgorithmicParameters,
)
    return UTR_solve(
        MathOptNLPModel(model),
        termination_criteria,
        algorithm_params,
    )
end

function UTR_solve(nlp::NLPModels.AbstractNLPModel)
    return UTR_solve(
        nlp,
        TerminationCriteria(),
        AlgorithmicParameters(),
    )
end

function UTR_solve(
    nlp::NLPModels.AbstractNLPModel,
    termination_criteria::TerminationCriteria,
    algorithm_params::AlgorithmicParameters,
)
    unconstrained(nlp) ||
        throw(ArgumentError("Constrained minimization problems are unsupported."))
    return optimize(
        nlp,
        termination_criteria,
        algorithm_params,
        Vector{Float64}(nlp.meta.x0),
        0.0,
    )
end

function UTR_solve(
    solver::Optimizer,
    termination_criteria::TerminationCriteria,
    algorithm_params::AlgorithmicParameters,
)
    return optimize(
        solver.nlp_data,
        termination_criteria,
        algorithm_params,
        _starting_point(solver),
        0.0,
    )
end

function _terminationResult(
    x::Vector{Float64},
    status,
    iteration_stats::DataFrame,
    algorithm_counter::AlgorithmCounter,
    outer_iteration::Int,
    start_time::Float64,
)
    assertFactorizationAccounting(algorithm_counter)
    return (
        x,
        status,
        iteration_stats,
        algorithm_counter,
        Int64(outer_iteration),
        time() - start_time,
    )
end

function _statusForException(error)
    if error isa OutOfMemoryError
        return TerminationStatusCode.MEMORY_LIMIT
    elseif error isa TrustRegionSubproblemError
        return TerminationStatusCode.TRUST_REGION_SUBPROBLEM_ERROR
    elseif error isa InvalidHessianError
        return TerminationStatusCode.INVALID_MODEL
    elseif error isa OverflowError ||
           error isa DomainError ||
           error isa PosDefException
        return TerminationStatusCode.NUMERICAL_ERROR
    elseif error isa ErrorException &&
           error.msg in (
               "Hessian computation is not supported.",
               "Gradient computation is not supported.",
           )
        return TerminationStatusCode.INVALID_MODEL
    end
    return TerminationStatusCode.OTHER_ERROR
end

"""
    optimize(nlp, termination_criteria, algorithm_params, x, warm_shift)

Run the first-order adaptive UTR method described in
`docs/adaptive_utr_implementation.tex`.
"""
function optimize(
    nlp::UTRNLPData,
    termination_criteria::TerminationCriteria,
    algorithm_params::AlgorithmicParameters,
    x::Vector{Float64},
    warm_shift::Float64 = 0.0,
)
    start_time = time()
    iteration_stats = DataFrame(
        k = Int64[],
        fval = Float64[],
        gradnorm = Float64[],
        min_gradnorm_fval = Float64[],
        min_gradnorm = Float64[],
    )
    algorithm_counter = AlgorithmCounter()
    outer_iteration = 1
    x_k = copy(x)

    try
        validateTerminationCriteria(termination_criteria)
        validateAlgorithmicParameters(algorithm_params)
        isfinite(warm_shift) && warm_shift >= 0.0 ||
            throw(ArgumentError("The shared-solver warm shift must be nonnegative."))
        nlp === nothing && throw(ArgumentError("The optimization model is required."))

        Random.seed!(algorithm_params.seed)
        parameter_selection_workspace = SparseCholeskyWorkspace(
            algorithm_params.reuse_sparse_symbolic_factorization,
            termination_criteria.iterative_refinement_max_iterations,
        )
        subproblem_workspace = SparseCholeskyWorkspace(
            algorithm_params.reuse_sparse_symbolic_factorization,
            termination_criteria.iterative_refinement_max_iterations,
        )

        if !_finite_vector(x_k)
            return _terminationResult(
                x_k,
                TerminationStatusCode.NUMERICAL_ERROR,
                iteration_stats,
                algorithm_counter,
                outer_iteration,
                start_time,
            )
        end
        f_current = evalFunction(nlp, x_k, algorithm_counter)
        g_current = evalGradient(nlp, x_k, algorithm_counter)
        if !_finite_scalar(f_current) || !_finite_vector(g_current)
            return _terminationResult(
                x_k,
                TerminationStatusCode.NUMERICAL_ERROR,
                iteration_stats,
                algorithm_counter,
                outer_iteration,
                start_time,
            )
        end
        gradient_norm = norm(g_current, 2)
        if !isfinite(gradient_norm)
            return _terminationResult(
                x_k,
                TerminationStatusCode.NUMERICAL_ERROR,
                iteration_stats,
                algorithm_counter,
                outer_iteration,
                start_time,
            )
        end
        min_grad_norm = gradient_norm
        min_gnorm_obj = f_current

        # The TeX requires the first-order check before computing the Hessian.
        if gradient_norm < termination_criteria.gradient_termination_tolerance
            push!(
                iteration_stats,
                (1, f_current, gradient_norm, min_gnorm_obj, min_grad_norm),
            )
            return _terminationResult(
                x_k,
                TerminationStatusCode.OPTIMAL,
                iteration_stats,
                algorithm_counter,
                1,
                start_time,
            )
        end
        if f_current <= termination_criteria.MINIMUM_OBJECTIVE_FUNCTION
            push!(
                iteration_stats,
                (1, f_current, gradient_norm, min_gnorm_obj, min_grad_norm),
            )
            return _terminationResult(
                x_k,
                TerminationStatusCode.UNBOUNDED,
                iteration_stats,
                algorithm_counter,
                1,
                start_time,
            )
        end

        hessian_current = selectHessianRepresentation(
            evalHessian(nlp, x_k, algorithm_counter),
            algorithm_params.dense_hessian_threshold,
        )
        ρ_k = algorithm_params.ρ_0

        for k in 1:termination_criteria.MAX_ITERATIONS
            outer_iteration = k
            ρ_trial = ρ_k
            accepted = false

            for j in 1:termination_criteria.MAX_INNER_ITERATIONS
                if time() - start_time >= termination_criteria.MAX_TIME
                    return _terminationResult(
                        x_k,
                        TerminationStatusCode.TIME_LIMIT,
                        iteration_stats,
                        algorithm_counter,
                        k,
                        start_time,
                    )
                end

                if algorithm_params.print_level >= 1
                    println(
                        "UTR outer iteration $k, inner trial $j, penalty $ρ_trial.",
                    )
                end

                selection = selectUTRSubproblemParameters(
                    hessian_current,
                    gradient_norm,
                    ρ_trial,
                    algorithm_counter,
                    parameter_selection_workspace,
                )
                if !isfinite(selection.radius)
                    return _terminationResult(
                        x_k,
                        TerminationStatusCode.NUMERICAL_ERROR,
                        iteration_stats,
                        algorithm_counter,
                        k,
                        start_time,
                    )
                elseif selection.radius <=
                       termination_criteria.STEP_SIZE_LIMIT
                    return _terminationResult(
                        x_k,
                        TerminationStatusCode.STEP_SIZE_LIMIT,
                        iteration_stats,
                        algorithm_counter,
                        k,
                        start_time,
                    )
                end
                regularized_hessian =
                    regularizeHessian(hessian_current, selection.hessian_shift)

                setSubproblemFailureDiagnosticContext(
                    iteration = k,
                    source_optimizer = "UTR",
                    inner_trial = j,
                    algorithm_counter = algorithm_counter,
                )
                subproblem_result = solveUTRSubproblem(
                    nlp,
                    g_current,
                    regularized_hessian,
                    warm_shift,
                    selection.radius,
                    gradient_norm,
                    algorithm_params,
                    algorithm_counter,
                    subproblem_workspace,
                )
                warm_shift = subproblem_result.delta
                if !isfinite(warm_shift) || warm_shift < 0.0
                    return _terminationResult(
                        x_k,
                        TerminationStatusCode.NUMERICAL_ERROR,
                        iteration_stats,
                        algorithm_counter,
                        k,
                        start_time,
                    )
                end

                if !subproblem_result.success
                    increment!(
                        algorithm_counter,
                        :total_number_rejected_trials,
                    )
                    ρ_trial =
                        rejectedPenalty(ρ_trial, algorithm_params.μ_1)
                    continue
                end

                direction = subproblem_result.direction
                if !_finite_vector(direction)
                    return _terminationResult(
                        x_k,
                        TerminationStatusCode.NUMERICAL_ERROR,
                        iteration_stats,
                        algorithm_counter,
                        k,
                        start_time,
                    )
                end
                direction_norm = norm(direction, 2)
                if !isfinite(direction_norm)
                    return _terminationResult(
                        x_k,
                        TerminationStatusCode.NUMERICAL_ERROR,
                        iteration_stats,
                        algorithm_counter,
                        k,
                        start_time,
                    )
                elseif direction_norm <=
                       termination_criteria.STEP_SIZE_LIMIT
                    return _terminationResult(
                        x_k,
                        TerminationStatusCode.STEP_SIZE_LIMIT,
                        iteration_stats,
                        algorithm_counter,
                        k,
                        start_time,
                    )
                end

                trial_x = x_k + direction
                if !_finite_vector(trial_x)
                    return _terminationResult(
                        x_k,
                        TerminationStatusCode.NUMERICAL_ERROR,
                        iteration_stats,
                        algorithm_counter,
                        k,
                        start_time,
                    )
                end
                f_trial = evalFunction(nlp, trial_x, algorithm_counter)
                if !_finite_scalar(f_trial)
                    return _terminationResult(
                        x_k,
                        TerminationStatusCode.NUMERICAL_ERROR,
                        iteration_stats,
                        algorithm_counter,
                        k,
                        start_time,
                    )
                end

                trial_gradient = g_current
                trial_gradient_norm = Inf
                if f_trial <= f_current
                    trial_gradient =
                        evalGradient(nlp, trial_x, algorithm_counter)
                    if !_finite_vector(trial_gradient)
                        return _terminationResult(
                            x_k,
                            TerminationStatusCode.NUMERICAL_ERROR,
                            iteration_stats,
                            algorithm_counter,
                            k,
                            start_time,
                        )
                    end
                    trial_gradient_norm = norm(trial_gradient, 2)
                    if !isfinite(trial_gradient_norm)
                        return _terminationResult(
                            x_k,
                            TerminationStatusCode.NUMERICAL_ERROR,
                            iteration_stats,
                            algorithm_counter,
                            k,
                            start_time,
                        )
                    end
                    if trial_gradient_norm < min_grad_norm
                        min_grad_norm = trial_gradient_norm
                        min_gnorm_obj = f_trial
                    end
                end

                if trialAccepted(
                    f_current,
                    f_trial,
                    gradient_norm,
                    trial_gradient_norm,
                    ρ_trial,
                    algorithm_params.η,
                    algorithm_params.ξ,
                )
                    x_k = trial_x
                    f_current = f_trial
                    g_current = trial_gradient
                    gradient_norm = trial_gradient_norm
                    ρ_k = acceptedPenalty(
                        ρ_trial,
                        algorithm_params.ρ_min,
                        algorithm_params.μ_2,
                    )
                    accepted = true
                    push!(
                        iteration_stats,
                        (
                            k,
                            f_current,
                            gradient_norm,
                            min_gnorm_obj,
                            min_grad_norm,
                        ),
                    )

                    if algorithm_params.print_level >= 1
                        println(
                            "UTR accepted outer iteration $k with " *
                            "||g|| = $gradient_norm and penalty $ρ_k.",
                        )
                    end
                    if gradient_norm <
                       termination_criteria.gradient_termination_tolerance
                        return _terminationResult(
                            x_k,
                            TerminationStatusCode.OPTIMAL,
                            iteration_stats,
                            algorithm_counter,
                            k,
                            start_time,
                        )
                    elseif f_current <=
                           termination_criteria.MINIMUM_OBJECTIVE_FUNCTION
                        return _terminationResult(
                            x_k,
                            TerminationStatusCode.UNBOUNDED,
                            iteration_stats,
                            algorithm_counter,
                            k,
                            start_time,
                        )
                    elseif time() - start_time >= termination_criteria.MAX_TIME
                        return _terminationResult(
                            x_k,
                            TerminationStatusCode.TIME_LIMIT,
                            iteration_stats,
                            algorithm_counter,
                            k,
                            start_time,
                        )
                    end
                    break
                end

                increment!(algorithm_counter, :total_number_rejected_trials)
                ρ_trial = rejectedPenalty(ρ_trial, algorithm_params.μ_1)
            end

            if !accepted
                # Preserve the best-gradient objective/gradient pair found by
                # rejected trials in this exhausted outer iteration. The current
                # iterate itself is unchanged.
                push!(
                    iteration_stats,
                    (
                        k,
                        f_current,
                        gradient_norm,
                        min_gnorm_obj,
                        min_grad_norm,
                    ),
                )
                return _terminationResult(
                    x_k,
                    TerminationStatusCode.INNER_ITERATION_LIMIT,
                    iteration_stats,
                    algorithm_counter,
                    k,
                    start_time,
                )
            end
            if k < termination_criteria.MAX_ITERATIONS
                hessian_current = selectHessianRepresentation(
                    evalHessian(nlp, x_k, algorithm_counter),
                    algorithm_params.dense_hessian_threshold,
                )
            end
        end

        return _terminationResult(
            x_k,
            TerminationStatusCode.ITERATION_LIMIT,
            iteration_stats,
            algorithm_counter,
            termination_criteria.MAX_ITERATIONS,
            start_time,
        )
    catch error
        status = _statusForException(error)
        if algorithm_params.print_level >= 0
            @warn "UTR terminated after an exception." exception = (
                error,
                catch_backtrace(),
            )
        end
        return _terminationResult(
            x_k,
            status,
            iteration_stats,
            algorithm_counter,
            outer_iteration,
            start_time,
        )
    end
end
