@enum PhiValue::Int8 begin
    PHI_NEGATIVE = -1
    PHI_ZERO = 0
    PHI_POSITIVE = 1
    PHI_UNCERTIFIED = 2
end

struct PhiEvaluation
    value::PhiValue
    shift::Float64
    accepted_shift::Float64
    direction::Vector{Float64}
    positive_definite::Bool
    certified::Bool
    direction_norm::Float64
    shifted_residual_norm::Float64
    unshifted_residual_norm::Float64
    model_value::Float64
end

mutable struct FactorizationCounter
    findinterval::Int
    bisection::Int
    compute_search_direction::Int
    inverse_power_iteration::Int
end

FactorizationCounter() = FactorizationCounter(0, 0, 0, 0)

function incrementFactorization!(
    counter::FactorizationCounter,
    category::Symbol,
)
    if category === :findinterval
        counter.findinterval += 1
    elseif category === :bisection
        counter.bisection += 1
    elseif category === :compute_search_direction
        counter.compute_search_direction += 1
    elseif category === :inverse_power_iteration
        counter.inverse_power_iteration += 1
    else
        throw(ArgumentError("Unknown factorization category: $category"))
    end
    return counter
end

function factorizationStats(counter::FactorizationCounter)
    total =
        counter.findinterval +
        counter.bisection +
        counter.compute_search_direction +
        counter.inverse_power_iteration
    return FactorizationStats(
        total,
        counter.findinterval,
        counter.bisection,
        counter.compute_search_direction,
        counter.inverse_power_iteration,
    )
end

struct SearchResult
    status::Symbol
    candidate::PhiEvaluation
    lower::PhiEvaluation
    upper::PhiEvaluation
end

function uncertifiedEvaluation(
    g::Vector{Float64},
    shift::Float64;
    direction::Vector{Float64} = zeros(length(g)),
    positive_definite::Bool = false,
    shifted_residual_norm::Float64 = Inf,
    unshifted_residual_norm::Float64 = Inf,
    model_value::Float64 = Inf,
)
    return PhiEvaluation(
        PHI_UNCERTIFIED,
        shift,
        shift,
        direction,
        positive_definite,
        false,
        norm(direction),
        shifted_residual_norm,
        unshifted_residual_norm,
        model_value,
    )
end

function nonPositiveDefiniteEvaluation(g::Vector{Float64}, shift::Float64)
    return PhiEvaluation(
        PHI_NEGATIVE,
        shift,
        shift,
        zeros(length(g)),
        false,
        true,
        Inf,
        Inf,
        Inf,
        Inf,
    )
end

function refinementTolerance(
    γ_1::Float64,
    γ_3::Float64,
    min_grad::Float64,
)
    return γ_1 * min_grad * min(1.0 / 3.0, (1.0 - γ_3) / (1.0 + γ_3))
end

function hardCaseTolerance(
    γ_1::Float64,
    γ_3::Float64,
    min_grad::Float64,
    radius::Float64,
)
    radius > 0.0 || return 0.0
    return γ_1 * min_grad / radius * min(1.0 / 6.0, (1.0 - γ_3) / 3.0)
end

function certifiedShiftedSolve(
    factorization::Factorization{Float64},
    g::Vector{Float64},
    H::HessianMatrix,
    shift::Float64,
    tolerance::Float64,
    workspace::OptionalSparseCholeskyWorkspace,
)
    direction = solveFactorizedSystem(factorization, g, workspace)
    residual, correction = iterativeRefinementBuffers(workspace, g)
    hessian_times_direction = similar(g)
    max_iterations = iterativeRefinementMaxIterations(workspace)
    iterations = 0

    residual_norm = Inf
    unshifted_residual_norm = Inf
    model_value = Inf
    while true
        mul!(hessian_times_direction, H, direction)
        @. residual = hessian_times_direction + g
        unshifted_residual_norm = norm(residual)
        model_value = dot(g, direction) + 0.5 * dot(direction, hessian_times_direction)
        @. residual += shift * direction
        residual_norm = norm(residual)
        certified =
            isfinite(residual_norm) &&
            isfinite(unshifted_residual_norm) &&
            isfinite(model_value) &&
            residual_norm <= tolerance &&
            model_value <= 0.0
        if certified || iterations == max_iterations
            return (
                direction = direction,
                residual_norm = residual_norm,
                unshifted_residual_norm = unshifted_residual_norm,
                model_value = model_value,
                iterations = iterations,
                certified = certified,
            )
        end

        residual .*= -1.0
        ldiv!(correction, factorization, residual)
        direction .+= correction
        iterations += 1
    end
end

function phi(
    g::Vector{Float64},
    H::HessianMatrix,
    shift::Float64,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    radius::Float64,
    min_grad::Float64,
    counter::FactorizationCounter,
    category::Symbol,
    workspace::OptionalSparseCholeskyWorkspace,
)
    incrementFactorization!(counter, category)
    factorization = try
        factorizeShiftedHessian!(workspace, H, shift)
    catch
        return uncertifiedEvaluation(g, shift)
    end
    if !issuccess(factorization)
        return nonPositiveDefiniteEvaluation(g, shift)
    end

    tolerance = refinementTolerance(γ_1, γ_3, min_grad)
    solve = try
        certifiedShiftedSolve(
            factorization,
            g,
            H,
            shift,
            tolerance,
            workspace,
        )
    catch
        return uncertifiedEvaluation(
            g,
            shift;
            positive_definite = true,
        )
    end
    direction_norm = norm(solve.direction)
    value = PHI_POSITIVE
    accepted_shift = shift
    if !isfinite(direction_norm)
        value = PHI_UNCERTIFIED
    elseif direction_norm > radius
        value = PHI_NEGATIVE
    elseif γ_2 * radius <= direction_norm
        value = PHI_ZERO
    elseif solve.unshifted_residual_norm <= γ_1 * min_grad
        value = PHI_ZERO
        accepted_shift = 0.0
    end
    return PhiEvaluation(
        value,
        shift,
        accepted_shift,
        solve.direction,
        true,
        solve.certified,
        direction_norm,
        solve.residual_norm,
        solve.unshifted_residual_norm,
        solve.model_value,
    )
end

function scaledShift(base_shift::Float64, exponent::Int)
    scaled = ldexp(base_shift, exponent)
    return isfinite(scaled) && scaled >= 0.0 ? scaled : Inf
end

function findinterval(
    g::Vector{Float64},
    H::HessianMatrix,
    warm_shift::Float64,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    radius::Float64,
    min_grad::Float64,
    counter::FactorizationCounter,
    workspace::OptionalSparseCholeskyWorkspace;
    zero_evaluation::Union{Nothing,PhiEvaluation} = nothing,
    max_iterations::Int = DEFAULT_FIND_INTERVAL_MAX_ITERATIONS,
)
    if warm_shift == 0.0 && zero_evaluation !== nothing
        if zero_evaluation.value == PHI_ZERO
            return SearchResult(
                :root,
                zero_evaluation,
                zero_evaluation,
                zero_evaluation,
            )
        end
    elseif warm_shift != 0.0
        warm_evaluation = phi(
            g,
            H,
            warm_shift,
            γ_1,
            γ_2,
            γ_3,
            radius,
            min_grad,
            counter,
            :findinterval,
            workspace,
        )
        if warm_evaluation.value == PHI_ZERO
            return SearchResult(
                :root,
                warm_evaluation,
                warm_evaluation,
                warm_evaluation,
            )
        elseif warm_evaluation.value == PHI_UNCERTIFIED
            return SearchResult(
                :failure,
                warm_evaluation,
                warm_evaluation,
                warm_evaluation,
            )
        end
    end

    base_shift = warm_shift == 0.0 ? 1.0 : warm_shift
    base_evaluation =
        warm_shift == 0.0 ?
        phi(
            g,
            H,
            base_shift,
            γ_1,
            γ_2,
            γ_3,
            radius,
            min_grad,
            counter,
            :findinterval,
            workspace,
        ) : warm_evaluation
    if base_evaluation.value == PHI_ZERO
        return SearchResult(
            :root,
            base_evaluation,
            base_evaluation,
            base_evaluation,
        )
    elseif base_evaluation.value == PHI_UNCERTIFIED
        return SearchResult(
            :failure,
            base_evaluation,
            base_evaluation,
            base_evaluation,
        )
    end

    direction_sign = -Int(base_evaluation.value)
    x_evaluation = base_evaluation
    for iteration = 1:max_iterations
        y_shift = scaledShift(base_shift, direction_sign * iteration^2)
        if !isfinite(y_shift) || y_shift == x_evaluation.shift
            return SearchResult(
                :failure,
                x_evaluation,
                x_evaluation,
                x_evaluation,
            )
        end
        y_evaluation = phi(
            g,
            H,
            y_shift,
            γ_1,
            γ_2,
            γ_3,
            radius,
            min_grad,
            counter,
            :findinterval,
            workspace,
        )
        if y_evaluation.value == PHI_ZERO
            return SearchResult(
                :root,
                y_evaluation,
                y_evaluation,
                y_evaluation,
            )
        elseif y_evaluation.value == PHI_UNCERTIFIED
            return SearchResult(
                :failure,
                y_evaluation,
                x_evaluation,
                y_evaluation,
            )
        elseif Int(x_evaluation.value) * Int(y_evaluation.value) < 0
            lower, upper =
                x_evaluation.shift < y_evaluation.shift ?
                (x_evaluation, y_evaluation) : (y_evaluation, x_evaluation)
            if lower.value != PHI_NEGATIVE ||
               upper.value != PHI_POSITIVE
                return SearchResult(:failure, y_evaluation, lower, upper)
            end
            return SearchResult(:interval, upper, lower, upper)
        end
        x_evaluation = y_evaluation
    end
    return SearchResult(
        :failure,
        x_evaluation,
        x_evaluation,
        x_evaluation,
    )
end

function bisection(
    g::Vector{Float64},
    H::HessianMatrix,
    interval::SearchResult,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    radius::Float64,
    min_grad::Float64,
    counter::FactorizationCounter,
    workspace::OptionalSparseCholeskyWorkspace;
    max_iterations::Int = DEFAULT_BISECTION_MAX_ITERATIONS,
)
    interval.status === :root && return interval
    interval.status === :interval ||
        return SearchResult(
            :failure,
            interval.candidate,
            interval.lower,
            interval.upper,
        )

    lower = interval.lower
    upper = interval.upper
    interval_tolerance =
        hardCaseTolerance(γ_1, γ_3, min_grad, radius)
    for _ = 1:max_iterations
        midpoint = lower.shift + (upper.shift - lower.shift) / 2.0
        if midpoint == lower.shift || midpoint == upper.shift
            break
        end
        evaluation = phi(
            g,
            H,
            midpoint,
            γ_1,
            γ_2,
            γ_3,
            radius,
            min_grad,
            counter,
            :bisection,
            workspace,
        )
        if evaluation.value == PHI_ZERO
            return SearchResult(:root, evaluation, lower, upper)
        elseif evaluation.value == PHI_UNCERTIFIED
            return SearchResult(:failure, evaluation, lower, upper)
        elseif evaluation.value == PHI_NEGATIVE
            lower = evaluation
        else
            upper = evaluation
        end

        if upper.shift - lower.shift <= interval_tolerance &&
           !lower.positive_definite &&
           upper.value == PHI_POSITIVE
            return SearchResult(:hard_case, upper, lower, upper)
        end
    end
    return SearchResult(:failure, upper, lower, upper)
end

function checkCandidateTerminationCriteria(
    direction::Vector{Float64},
    g::Vector{Float64},
    H::HessianMatrix,
    shift::Float64,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    radius::Float64,
    min_grad::Float64,
)
    length(direction) == length(g) ||
        throw(DimensionMismatch("Direction and gradient dimensions must agree."))
    all(isfinite, direction) ||
        throw(ArgumentError("The direction must be finite."))
    validateTrustRegionInputs(g, H, shift, radius, min_grad)
    validateGammaParameters(γ_1, γ_2, γ_3)
    direction_norm, unshifted_residual_norm, shifted_residual_norm, model_value =
        terminationMetrics(direction, g, H, shift)
    stationarity_tolerance = γ_1 * min_grad
    boundary_lower_bound = γ_2 * radius
    model_decrease_threshold = -γ_3 * 0.5 * shift * direction_norm^2
    all(
        isfinite,
        (
            direction_norm,
            unshifted_residual_norm,
            shifted_residual_norm,
            model_value,
            stationarity_tolerance,
            boundary_lower_bound,
            model_decrease_threshold,
        ),
    ) || throw(OverflowError("The candidate termination metrics must be finite."))
    failure_reason_6a = shifted_residual_norm > stationarity_tolerance
    failure_reason_6b = γ_2 * shift * radius > shift * direction_norm
    failure_reason_6c = direction_norm > radius
    failure_reason_6d = model_value > model_decrease_threshold
    valid =
        !(
            failure_reason_6a ||
            failure_reason_6b ||
            failure_reason_6c ||
            failure_reason_6d
        )
    return TrustRegionTerminationCheck(
        valid,
        direction_norm == 0.0,
        valid && shift == 0.0,
        shift,
        direction_norm,
        unshifted_residual_norm,
        shifted_residual_norm,
        stationarity_tolerance,
        boundary_lower_bound,
        model_value,
        model_decrease_threshold,
        model_value <= 0.0,
        failure_reason_6a,
        failure_reason_6b,
        failure_reason_6c,
        failure_reason_6d,
    )
end

function boundaryCandidates(
    base_direction::Vector{Float64},
    eigenvector::Vector{Float64},
    g::Vector{Float64},
    H::HessianMatrix,
    radius::Float64,
)
    direction_dot_eigenvector = dot(base_direction, eigenvector)
    radicand =
        direction_dot_eigenvector^2 +
        radius^2 -
        sum(abs2, base_direction)
    roundoff_scale =
        max(1.0, direction_dot_eigenvector^2, radius^2, sum(abs2, base_direction))
    if radicand < -16.0 * eps(roundoff_scale)
        return nothing
    end
    boundary_root = sqrt(max(0.0, radicand))
    α_1 = -direction_dot_eigenvector + boundary_root
    α_2 = -direction_dot_eigenvector - boundary_root
    candidate_1 = floatingPointSafeBallProjection(
        base_direction + α_1 * eigenvector,
        radius,
    )
    candidate_2 = floatingPointSafeBallProjection(
        base_direction + α_2 * eigenvector,
        radius,
    )
    model_1 = dot(g, candidate_1) + 0.5 * dot(candidate_1, H * candidate_1)
    model_2 = dot(g, candidate_2) + 0.5 * dot(candidate_2, H * candidate_2)
    return model_1 <= model_2 ?
           ((candidate_1, model_1), (candidate_2, model_2)) :
           ((candidate_2, model_2), (candidate_1, model_1))
end

function inversePowerIteration(
    g::Vector{Float64},
    H::HessianMatrix,
    upper::PhiEvaluation,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    radius::Float64,
    min_grad::Float64,
    counter::FactorizationCounter,
    workspace::OptionalSparseCholeskyWorkspace;
    max_iterations::Int = DEFAULT_INVERSE_POWER_MAX_ITERATIONS,
)
    incrementFactorization!(counter, :inverse_power_iteration)
    factorization = try
        factorizeShiftedHessian!(workspace, H, upper.shift)
    catch
        return false, upper
    end
    issuccess(factorization) || return false, upper

    eigenvector = randn(length(g))
    eigenvector_norm = norm(eigenvector)
    if !isfinite(eigenvector_norm) || eigenvector_norm == 0.0
        return false, upper
    end
    eigenvector ./= eigenvector_norm
    next_eigenvector = similar(eigenvector)
    hessian_times_eigenvector = similar(eigenvector)
    eigen_residual = similar(eigenvector)
    eigen_tolerance =
        hardCaseTolerance(γ_1, γ_3, min_grad, radius)
    last_evaluation = upper

    for _ = 1:max_iterations
        try
            ldiv!(next_eigenvector, factorization, eigenvector)
        catch
            return false, last_evaluation
        end
        next_norm = norm(next_eigenvector)
        if !isfinite(next_norm) || next_norm == 0.0
            return false, last_evaluation
        end
        next_eigenvector ./= next_norm
        eigenvector, next_eigenvector = next_eigenvector, eigenvector

        mul!(hessian_times_eigenvector, H, eigenvector)
        eigenvalue = dot(eigenvector, hessian_times_eigenvector)
        @. eigen_residual = hessian_times_eigenvector - eigenvalue * eigenvector
        norm(eigen_residual) <= eigen_tolerance || continue

        candidates =
            boundaryCandidates(upper.direction, eigenvector, g, H, radius)
        candidates === nothing && return false, last_evaluation
        for (candidate, model_value) in candidates
            check = checkCandidateTerminationCriteria(
                candidate,
                g,
                H,
                upper.shift,
                γ_1,
                γ_2,
                γ_3,
                radius,
                min_grad,
            )
            last_evaluation = PhiEvaluation(
                PHI_ZERO,
                upper.shift,
                upper.shift,
                candidate,
                true,
                check.valid,
                check.d_norm,
                check.shifted_residual_norm,
                check.unshifted_residual_norm,
                model_value,
            )
            check.valid && return true, last_evaluation
        end
    end
    return false, last_evaluation
end

function searchShift(
    g::Vector{Float64},
    H::HessianMatrix,
    warm_shift::Float64,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    radius::Float64,
    min_grad::Float64,
    counter::FactorizationCounter,
    workspace::OptionalSparseCholeskyWorkspace;
    zero_evaluation::Union{Nothing,PhiEvaluation} = nothing,
)
    interval = findinterval(
        g,
        H,
        warm_shift,
        γ_1,
        γ_2,
        γ_3,
        radius,
        min_grad,
        counter,
        workspace,
        zero_evaluation = zero_evaluation,
    )
    return interval.status === :interval ?
           bisection(
        g,
        H,
        interval,
        γ_1,
        γ_2,
        γ_3,
        radius,
        min_grad,
        counter,
        workspace,
    ) : interval
end

function randomUnitVector(n::Int)
    vector = randn(n)
    vector_norm = norm(vector)
    return isfinite(vector_norm) && vector_norm > 0.0 ?
           vector / vector_norm : nothing
end

function subproblemResult(
    success::Bool,
    evaluation::PhiEvaluation,
    delta_prime::Float64,
    hard_case::Bool,
    counter::FactorizationCounter,
)
    return TrustRegionSubproblemResult(
        success,
        evaluation.accepted_shift,
        delta_prime,
        evaluation.direction,
        hard_case,
        factorizationStats(counter),
    )
end

function optimizeSecondOrderModel(
    problem_name::String,
    g::Vector{Float64},
    H::HessianMatrix,
    warm_shift::Float64,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    radius::Float64,
    min_grad::Float64;
    print_level::Int = DEFAULT_PRINT_LEVEL,
    use_backup_trust_region_subproblem_solver::Bool =
        DEFAULT_USE_BACKUP_TRUST_REGION_SUBPROBLEM_SOLVER,
    sparse_cholesky_workspace::OptionalSparseCholeskyWorkspace =
        SparseCholeskyWorkspace(),
    handle_hard_case::Bool = DEFAULT_HANDLE_HARD_CASE,
    inverse_power_max_iterations::Int = DEFAULT_INVERSE_POWER_MAX_ITERATIONS,
)
    counter = FactorizationCounter()
    zero_evaluation = phi(
        g,
        H,
        0.0,
        γ_1,
        γ_2,
        γ_3,
        radius,
        min_grad,
        counter,
        :compute_search_direction,
        sparse_cholesky_workspace,
    )
    if zero_evaluation.value == PHI_ZERO
        zero_check = checkCandidateTerminationCriteria(
            zero_evaluation.direction,
            g,
            H,
            zero_evaluation.accepted_shift,
            γ_1,
            γ_2,
            γ_3,
            radius,
            min_grad,
        )
        if zero_check.valid
            return subproblemResult(true, zero_evaluation, 0.0, false, counter)
        end
        # A finite but uncertified solve can be classified as a root from its
        # norm alone. Do not let the interval search reuse that rejected root.
        zero_evaluation = nothing
    end

    search = searchShift(
        g,
        H,
        warm_shift,
        γ_1,
        γ_2,
        γ_3,
        radius,
        min_grad,
        counter,
        sparse_cholesky_workspace,
        zero_evaluation = zero_evaluation,
    )
    if search.status === :root
        check = checkCandidateTerminationCriteria(
            search.candidate.direction,
            g,
            H,
            search.candidate.accepted_shift,
            γ_1,
            γ_2,
            γ_3,
            radius,
            min_grad,
        )
        if check.valid
            return subproblemResult(
                true,
                search.candidate,
                search.upper.shift,
                false,
                counter,
            )
        end
        search = SearchResult(
            :failure,
            search.candidate,
            search.lower,
            search.upper,
        )
    end

    hard_case = search.status === :hard_case
    # Intentional lazy-ablation behavior: disabling hard-case handling returns the
    # available upper-endpoint candidate without inverse-power correction or final
    # certification. Production configurations keep `handle_hard_case=true`; the
    # benchmark disables it only for the lazy subproblem-solver ablations.
    # Here `success=true` means that the ablation produced a step, not that the step
    # satisfies the standard Equation (6) termination contract.
    if hard_case && !handle_hard_case
        return subproblemResult(true, search.upper, search.upper.shift, true, counter)
    elseif hard_case
        inverse_success, inverse_candidate = inversePowerIteration(
            g,
            H,
            search.upper,
            γ_1,
            γ_2,
            γ_3,
            radius,
            min_grad,
            counter,
            sparse_cholesky_workspace,
            max_iterations = inverse_power_max_iterations,
        )
        if inverse_success
            return subproblemResult(
                true,
                inverse_candidate,
                search.upper.shift,
                true,
                counter,
            )
        end
        search = SearchResult(
            :failure,
            inverse_candidate,
            search.lower,
            search.upper,
        )
    end

    if use_backup_trust_region_subproblem_solver
        unit_vector = randomUnitVector(length(g))
        if unit_vector !== nothing
            perturbed_g = g + 0.5 * γ_1 * min_grad * unit_vector
            backup = searchShift(
                perturbed_g,
                H,
                warm_shift,
                0.5 * γ_1,
                γ_2,
                γ_3,
                radius,
                min_grad,
                counter,
                sparse_cholesky_workspace,
            )
            backup_hard_case = backup.status === :hard_case
            # Preserve the same intentional lazy-ablation contract for a hard case
            # encountered by the perturbed-gradient backup search.
            if backup_hard_case && !handle_hard_case
                return subproblemResult(
                    true,
                    backup.upper,
                    backup.upper.shift,
                    true,
                    counter,
                )
            elseif backup_hard_case
                inverse_success, inverse_candidate =
                    inversePowerIteration(
                        perturbed_g,
                        H,
                        backup.upper,
                        0.5 * γ_1,
                        γ_2,
                        γ_3,
                        radius,
                        min_grad,
                        counter,
                        sparse_cholesky_workspace,
                        max_iterations = inverse_power_max_iterations,
                    )
                backup = SearchResult(
                    inverse_success ? :root : :failure,
                    inverse_candidate,
                    backup.lower,
                    backup.upper,
                )
            end
            if backup.status === :root
                original_check = checkCandidateTerminationCriteria(
                    backup.candidate.direction,
                    g,
                    H,
                    backup.candidate.accepted_shift,
                    γ_1,
                    γ_2,
                    γ_3,
                    radius,
                    min_grad,
                )
                if original_check.valid
                    return subproblemResult(
                        true,
                        backup.candidate,
                        backup.upper.shift,
                        hard_case || backup_hard_case,
                        counter,
                    )
                end
            end
            search = backup
        end
    end

    print_level >= 1 &&
        println("Trust-region solver did not obtain a certified candidate.")
    return subproblemResult(
        false,
        search.candidate,
        search.upper.shift,
        hard_case,
        counter,
    )
end

function validateResult(
    problem_name::String,
    result::TrustRegionSubproblemResult,
    g::Vector{Float64},
    H::HessianMatrix,
    original_shift::Float64,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    radius::Float64,
    min_grad::Float64,
    print_level::Int,
)
    check = checkCandidateTerminationCriteria(
        result.direction,
        g,
        H,
        result.delta,
        γ_1,
        γ_2,
        γ_3,
        radius,
        min_grad,
    )
    return validateTrustRegionSubproblemTerminationCriteria(
        problem_name,
        result.direction,
        g,
        H,
        original_shift,
        result.delta,
        result.delta_prime,
        γ_1,
        γ_2,
        γ_3,
        radius,
        min_grad,
        result.hard_case,
        print_level,
        validation_context = "final_subproblem_solution",
        local_factorization_counts = Dict{String,Any}(
            "total" => result.factorizations.total,
            "findinterval" => result.factorizations.findinterval,
            "bisection" => result.factorizations.bisection,
            "compute_search_direction" =>
                result.factorizations.compute_search_direction,
            "inverse_power_iteration" =>
                result.factorizations.inverse_power_iteration,
        ),
        write_failure_details = true,
        termination_check = check,
    )
end

"""
    solveTrustRegionSubproblem(
        problem_name, g, H, δ, γ_1, γ_2, γ_3, r, min_grad; kwargs...
    )

Solve a quadratic trust-region subproblem using the certified interval and
hard-case algorithm.
"""
function solveTrustRegionSubproblem(
    problem_name::String,
    g::Vector{Float64},
    H::HessianMatrix,
    δ::Float64,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    r::Float64,
    min_grad::Float64;
    print_level::Int = DEFAULT_PRINT_LEVEL,
    use_backup_trust_region_subproblem_solver::Bool =
        DEFAULT_USE_BACKUP_TRUST_REGION_SUBPROBLEM_SOLVER,
    sparse_cholesky_workspace::OptionalSparseCholeskyWorkspace =
        SparseCholeskyWorkspace(),
    handle_hard_case::Bool = DEFAULT_HANDLE_HARD_CASE,
)
    validateTrustRegionInputs(g, H, δ, r, min_grad)
    validateGammaParameters(γ_1, γ_2, γ_3)

    if H isa Matrix{Float64} && sparse_cholesky_workspace !== nothing
        resetSparseCholeskyWorkspace!(sparse_cholesky_workspace)
    end
    result = optimizeSecondOrderModel(
        problem_name,
        g,
        H,
        δ,
        γ_1,
        γ_2,
        γ_3,
        r,
        min_grad,
        print_level = print_level,
        use_backup_trust_region_subproblem_solver =
            use_backup_trust_region_subproblem_solver,
        sparse_cholesky_workspace = sparse_cholesky_workspace,
        handle_hard_case = handle_hard_case,
    )
    # Skip certification only for the intentional lazy hard-case ablation. Default
    # hard-case handling is always validated.
    if !(result.hard_case && !handle_hard_case)
        validateResult(
            problem_name,
            result,
            g,
            H,
            δ,
            γ_1,
            γ_2,
            γ_3,
            r,
            min_grad,
            print_level,
        )
    end
    return result
end
