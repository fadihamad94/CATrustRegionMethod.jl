module CATNeurIPS

using LinearAlgebra
using NLPModels

import TrustRegionSubproblemSolvers

const DEFAULT_BETA = 0.1
const DEFAULT_THETA = 0.1
const DEFAULT_OMEGA = 8.0
const DEFAULT_INITIAL_RADIUS = 1.0
const DEFAULT_DELTA = 0.0
const DEFAULT_GAMMA_2 = 0.8
const SOURCE_REPOSITORY_COMMIT = "905e4976c0d6384456ebf522b871559c30a0e878"

"Parameters used by the CAT implementation published at NeurIPS 2022."
struct AlgorithmicParameters
    beta::Float64
    theta::Float64
    omega::Float64
    initial_radius::Float64
    delta::Float64
    gamma_2::Float64

    function AlgorithmicParameters(
        beta::Real = DEFAULT_BETA,
        theta::Real = DEFAULT_THETA,
        omega::Real = DEFAULT_OMEGA,
        initial_radius::Real = DEFAULT_INITIAL_RADIUS,
        delta::Real = DEFAULT_DELTA,
        gamma_2::Real = DEFAULT_GAMMA_2,
    )
        beta = Float64(beta)
        theta = Float64(theta)
        omega = Float64(omega)
        initial_radius = Float64(initial_radius)
        delta = Float64(delta)
        gamma_2 = Float64(gamma_2)
        0.0 < beta < 1.0 || throw(ArgumentError("beta must lie in (0, 1)."))
        0.0 <= theta < 1.0 || throw(ArgumentError("theta must lie in [0, 1)."))
        beta * theta < 1.0 - beta ||
            throw(ArgumentError("beta * theta must be less than 1 - beta."))
        omega > 1.0 || throw(ArgumentError("omega must be greater than one."))
        initial_radius > 0.0 ||
            throw(ArgumentError("The initial radius must be positive."))
        delta >= 0.0 || throw(ArgumentError("delta must be nonnegative."))
        1.0 / omega < gamma_2 <= 1.0 ||
            throw(ArgumentError("gamma_2 must lie in (1 / omega, 1]."))
        return new(beta, theta, omega, initial_radius, delta, gamma_2)
    end
end

"Stopping limits used by a NeurIPS CAT solve."
struct TerminationCriteria
    max_iterations::Int64
    gradient_tolerance::Float64
    max_time_seconds::Float64

    function TerminationCriteria(
        max_iterations::Integer,
        gradient_tolerance::Real,
        max_time_seconds::Real,
    )
        max_iterations > 0 ||
            throw(ArgumentError("The iteration limit must be positive."))
        gradient_tolerance >= 0.0 ||
            throw(ArgumentError("The gradient tolerance must be nonnegative."))
        max_time_seconds > 0.0 ||
            throw(ArgumentError("The time limit must be positive."))
        return new(
            Int64(max_iterations),
            Float64(gradient_tolerance),
            Float64(max_time_seconds),
        )
    end
end

"Result and benchmark accounting returned by `solve`."
struct SolverResult
    solution::Vector{Float64}
    status::String
    objective::Float64
    gradient_norm::Float64
    iterations::Int64
    execution_time::Float64
    subproblem_solves::Int64
end

function compute_second_order_model(
    objective::Float64,
    gradient::Vector{Float64},
    hessian,
    direction::Vector{Float64},
)::Float64
    return objective + dot(gradient, direction) +
           0.5 * dot(direction, hessian * direction)
end

function compute_rho(
    objective::Float64,
    next_objective::Float64,
    gradient::Vector{Float64},
    next_gradient::Vector{Float64},
    hessian,
    direction::Vector{Float64},
    theta::Float64,
)::Float64
    model_value =
        compute_second_order_model(objective, gradient, hessian, direction)
    guarantee = 0.5 * theta * norm(next_gradient) * norm(direction)
    return (objective - next_objective) /
           (objective - model_value + guarantee)
end

function solver_result(
    solution::Vector{Float64},
    status::String,
    objective::Float64,
    gradient::Vector{Float64},
    iterations::Int64,
    start_time::Float64,
    subproblem_solves::Int64,
)::SolverResult
    return SolverResult(
        copy(solution),
        status,
        objective,
        norm(gradient),
        iterations,
        time() - start_time,
        subproblem_solves,
    )
end

"""
    solve(nlp, termination, parameters=AlgorithmicParameters(); x=nlp.meta.x0)

Run the NeurIPS 2022 CAT outer iteration. The original repository's duplicated
trust-region implementation is intentionally not included: every subproblem is
delegated to `TrustRegionSubproblemSolvers.solveTrustRegionSubproblemOldApproach`.
"""
function solve(
    nlp::NLPModels.AbstractNLPModel,
    termination::TerminationCriteria,
    parameters::AlgorithmicParameters = AlgorithmicParameters();
    x::AbstractVector{<:Real} = nlp.meta.x0,
)::SolverResult
    start_time = time()
    x_k = Float64.(x)
    delta_k = parameters.delta
    radius = parameters.initial_radius
    subproblem_solves = Int64(0)

    gradient = Vector{Float64}(grad(nlp, x_k))
    objective = Float64(obj(nlp, x_k))
    if norm(gradient) <= termination.gradient_tolerance
        return solver_result(
            x_k,
            "OPTIMAL",
            objective,
            gradient,
            1,
            start_time,
            subproblem_solves,
        )
    end

    hessian = nothing
    compute_hessian = true
    for iteration = 1:termination.max_iterations
        if compute_hessian
            hessian = hess(nlp, x_k)
        end

        success, delta_k, direction, _ =
            TrustRegionSubproblemSolvers.solveTrustRegionSubproblemOldApproach(
                objective,
                gradient,
                hessian,
                x_k,
                delta_k,
                parameters.gamma_2,
                radius,
            )
        subproblem_solves += 1
        success || return solver_result(
            x_k,
            "TRUST_REGION_SUBPROBLEM_ERROR",
            objective,
            gradient,
            Int64(iteration),
            start_time,
            subproblem_solves,
        )

        candidate = x_k + direction
        next_objective = Float64(obj(nlp, candidate))
        next_gradient = Vector{Float64}(grad(nlp, candidate))
        rho = compute_rho(
            objective,
            next_objective,
            gradient,
            next_gradient,
            hessian,
            direction,
            parameters.theta,
        )

        if next_objective <= objective
            x_k = candidate
            objective = next_objective
            gradient = next_gradient
            compute_hessian = true
        else
            compute_hessian = false
        end

        if rho <= parameters.beta
            radius = norm(direction) / parameters.omega
        else
            radius = parameters.omega * norm(direction)
        end

        # This intentionally follows the published repository, which tests the
        # candidate gradient even when the objective-based acceptance test rejects
        # the candidate. The benchmark's independent optimality check will expose
        # any resulting false success claim.
        if norm(next_gradient) <= termination.gradient_tolerance
            return solver_result(
                x_k,
                "OPTIMAL",
                objective,
                gradient,
                Int64(iteration),
                start_time,
                subproblem_solves,
            )
        end
        if time() - start_time > termination.max_time_seconds
            return solver_result(
                x_k,
                "TIME_LIMIT",
                objective,
                gradient,
                Int64(iteration),
                start_time,
                subproblem_solves,
            )
        end
    end

    return solver_result(
        x_k,
        "ITERATION_LIMIT",
        objective,
        gradient,
        termination.max_iterations,
        start_time,
        subproblem_solves,
    )
end

export AlgorithmicParameters,
    SOURCE_REPOSITORY_COMMIT,
    SolverResult,
    TerminationCriteria,
    compute_rho,
    compute_second_order_model,
    solve

end
