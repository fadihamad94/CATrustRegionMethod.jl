struct TrustRegionTerminationCheck
    valid::Bool
    zero_step::Bool
    interior_accepted::Bool
    effective_delta::Float64
    d_norm::Float64
    unshifted_residual_norm::Float64
    shifted_residual_norm::Float64
    stationarity_tolerance::Float64
    boundary_lower_bound::Float64
    model_value::Float64
    model_decrease_threshold::Float64
    interior_model_decrease::Bool
    failure_reason_6a::Bool
    failure_reason_6b::Bool
    failure_reason_6c::Bool
    failure_reason_6d::Bool
end

function validateTrustRegionInputs(
    g::Vector{Float64},
    H::HessianMatrix,
    δ_k::Float64,
    r::Float64,
    min_grad::Float64,
)
    length(g) == size(H, 1) == size(H, 2) ||
        throw(DimensionMismatch("Gradient and Hessian dimensions must agree."))
    all(isfinite, g) || throw(ArgumentError("The gradient must be finite."))
    finiteHessian(H) || throw(ArgumentError("The Hessian must be finite."))
    issymmetric(H) || throw(ArgumentError("The Hessian must be symmetric."))
    isfinite(δ_k) && δ_k >= 0.0 ||
        throw(ArgumentError("The shift must be finite and nonnegative."))
    isfinite(r) && r >= 0.0 ||
        throw(ArgumentError("The trust-region radius must be finite and nonnegative."))
    isfinite(min_grad) && min_grad >= 0.0 ||
        throw(ArgumentError("The stationarity scale must be finite and nonnegative."))
    return nothing
end

function terminationMetrics(
    d::Vector{Float64},
    g::Vector{Float64},
    H::HessianMatrix,
    δ_k::Float64,
    hessian_times_d::Union{Nothing,Vector{Float64}} = nothing,
    d_norm::Union{Nothing,Float64} = nothing,
)
    d_norm === nothing && (d_norm = norm(d))
    hessian_times_d === nothing && (hessian_times_d = H * d)
    unshifted_residual = similar(g)
    @. unshifted_residual = hessian_times_d + g
    unshifted_residual_norm = norm(unshifted_residual)
    model_value = dot(g, d) + 0.5 * dot(d, hessian_times_d)
    @. unshifted_residual += δ_k * d
    shifted_residual_norm = norm(unshifted_residual)
    return d_norm, unshifted_residual_norm, shifted_residual_norm, model_value
end

function checkTrustRegionSubproblemTerminationCriteria(
    d::Vector{Float64},
    g::Vector{Float64},
    H::HessianMatrix,
    δ_k::Float64,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    r::Float64,
    min_grad::Float64,
    hessian_times_d::Union{Nothing,Vector{Float64}} = nothing,
    precomputed_d_norm::Union{Nothing,Float64} = nothing,
)
    length(d) == length(g) ||
        throw(DimensionMismatch("Direction and gradient dimensions must agree."))
    all(isfinite, d) || throw(ArgumentError("The direction must be finite."))
    validateTrustRegionInputs(g, H, δ_k, r, min_grad)
    if hessian_times_d !== nothing
        length(hessian_times_d) == length(d) ||
            throw(DimensionMismatch("The Hessian product and direction dimensions must agree."))
        all(isfinite, hessian_times_d) ||
            throw(ArgumentError("The Hessian product must be finite."))
    end
    if precomputed_d_norm !== nothing
        isfinite(precomputed_d_norm) && precomputed_d_norm >= 0.0 ||
            throw(ArgumentError("The direction norm must be finite and nonnegative."))
    end
    validateGammaParameters(γ_1, γ_2, γ_3)
    d_norm = precomputed_d_norm === nothing ? norm(d) : precomputed_d_norm
    d_norm, unshifted_residual_norm, shifted_residual_norm, model_value =
        terminationMetrics(d, g, H, δ_k, hessian_times_d, d_norm)
    stationarity_tolerance = γ_1 * min_grad
    boundary_lower_bound = γ_2 * r
    all(
        isfinite,
        (
            d_norm,
            unshifted_residual_norm,
            shifted_residual_norm,
            model_value,
            stationarity_tolerance,
            boundary_lower_bound,
        ),
    ) || throw(OverflowError("The termination metrics must be finite."))
    condition_6a = unshifted_residual_norm <= stationarity_tolerance
    condition_6c = d_norm <= r
    interior_model_decrease = model_value < 0.0
    interior_accepted = condition_6a && condition_6c && interior_model_decrease
    if interior_accepted
        return TrustRegionTerminationCheck(
            true,
            false,
            true,
            δ_k,
            d_norm,
            unshifted_residual_norm,
            shifted_residual_norm,
            stationarity_tolerance,
            boundary_lower_bound,
            model_value,
            0.0,
            interior_model_decrease,
            false,
            false,
            false,
            false,
        )
    end

    effective_delta = δ_k
    failure_reason_6a = shifted_residual_norm > stationarity_tolerance
    failure_reason_6b = false
    if δ_k > 0.0 && boundary_lower_bound > d_norm
        if condition_6a
            effective_delta = 0.0
        else
            failure_reason_6b = true
        end
    end
    failure_reason_6c = !condition_6c
    model_decrease_threshold = -γ_3 * 0.5 * effective_delta * d_norm^2
    isfinite(model_decrease_threshold) ||
        throw(OverflowError("The model-decrease threshold must be finite."))
    failure_reason_6d = model_value > model_decrease_threshold
    valid =
        !(failure_reason_6a || failure_reason_6b || failure_reason_6c || failure_reason_6d)
    return TrustRegionTerminationCheck(
        valid,
        d_norm == 0.0,
        false,
        effective_delta,
        d_norm,
        unshifted_residual_norm,
        shifted_residual_norm,
        stationarity_tolerance,
        boundary_lower_bound,
        model_value,
        model_decrease_threshold,
        interior_model_decrease,
        failure_reason_6a,
        failure_reason_6b,
        failure_reason_6c,
        failure_reason_6d,
    )
end


"""
  validateTrustRegionSubproblemTerminationCriteria(problem_name, d_k, g, H, δ_original, δ, δ_prime, γ_1, γ_2, γ_3, r, min_grad, hard_case, print_level)
  validate the trust-region subproblem termination conditions.

  # Inputs:
    - `problem_name::String`. Name of the problem being optimized for example a CUTEst benchamrk problem SCURLY10.
	- 'd_k:Vector{Float64}`. See (1). The search direction which is the solution of (1).
	- `g::Vector{Float64}`. See (1). The gradient at the current iterate x.
	- `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
	See (1). The Hessian at the current iterate x.
	- `δ_original::Float64`. See (1). A warm start value for solving the above system (2).
	- 'δ::Float64'. See (3). The lower bound of the interval [δ, δ_prime] such that ϕ(δ) >= 0.
	- 'δ_prime::Float64'. See (3). The upper bound of the interval [δ, δ_prime] such that ϕ(δ) <= 0.
	- `γ_1::Float64`. See (2). Specify how much the step d_k should be close from the exact solution.
	- `γ_2::Float64`. See (2). Specify how close the step d_k should be close from the trust-region boundary when δ > 0.
	- `γ_3::Float64`. See (2). Specify upper bound on the Model value.
	- `r::Float64`. See (1). The trsut-region radius.
	- `min_grad::Float64`. See (2). The minumum gradient over all iterates.
	- 'hard_case::Bool'. See (2). It specifies if δ_m = -λ_1(H) where λ_1 is the minimum eigenvalue of the matrix H.
	- `print_level::Float64`. The verbosity level of logs.

  # Outputs:
	If subproblem termination conditions are satisfied or not.
"""
function validateTrustRegionSubproblemTerminationCriteria(
    problem_name::String,
    d_k::Vector{Float64},
    g::Vector{Float64},
    H::Union{
        Matrix{Float64},
        SparseMatrixCSC{Float64,Int},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int}},
    },
    original_δ::Float64,
    δ_k::Float64,
    δ_prime_k::Float64,
    γ_1::Float64,
    γ_2::Float64,
    γ_3::Float64,
    r::Float64,
    min_grad::Float64,
    hard_case::Bool,
    print_level::Int64 = DEFAULT_PRINT_LEVEL;
    validation_context::String = "",
    local_factorization_counts::Union{Nothing,Dict{String,Any}} = nothing,
    write_failure_details::Bool = false,
    termination_check::Union{Nothing,TrustRegionTerminationCheck} = nothing,
)
    check =
        termination_check === nothing ?
        checkTrustRegionSubproblemTerminationCriteria(
            d_k,
            g,
            H,
            δ_k,
            γ_1,
            γ_2,
            γ_3,
            r,
            min_grad,
        ) : termination_check
    input_δ_k = δ_k
    δ_k = check.effective_delta
    d_norm = check.d_norm
    unshifted_residual_norm = check.unshifted_residual_norm
    shifted_residual_norm = check.shifted_residual_norm
    stationarity_tolerance = check.stationarity_tolerance
    boundary_lower_bound = check.boundary_lower_bound
    model_val = check.model_value
    model_decrease_threshold = check.model_decrease_threshold
    condition_6a = unshifted_residual_norm <= stationarity_tolerance
    condition_6d = check.interior_model_decrease
    if check.valid
        if print_level >= 2
            if check.interior_accepted
                println("==========ACCEPTING STEP==============")
            else
                if input_δ_k != δ_k
                    println(
                        "Treating delta_k as 0.0 because the unshifted residual is already small.",
                    )
                end
                println("##########ACCEPTING STEP##########")
            end
        end
        return check.interior_accepted ? true : nothing
    end

    error_message = "Trust-region subproblem failure."
    failure_reason_6a = check.failure_reason_6a
    failure_reason_6b = check.failure_reason_6b
    failure_reason_6c = check.failure_reason_6c
    failure_reason_6d = check.failure_reason_6d
    message = "HARD CASE is $hard_case."
    if failure_reason_6a
        temp_1 = shifted_residual_norm
        temp_2 = stationarity_tolerance
        message = string(
            message,
            ". Value of norm(H * d + g + δ * d) is $temp_1, value of γ_1 * min_grad is $temp_2, and value of min_grad is $min_grad.",
        )
        error_message = string(error_message, " Reason (6a) failed to be satisfied.")
    end
    if input_δ_k != δ_k && print_level >= 2
        println("Treating delta_k as 0.0 because the unshifted residual is already small.")
    elseif failure_reason_6b
        message = string(
            message,
            ". Value of γ_2 * r is $boundary_lower_bound, value of δ_k is $δ_k, and value of ||d_k|| = $d_norm",
        )
        error_message = string(error_message, " Reason (6b) failed to be satisfied.")
    end
    if failure_reason_6c
        message = string(message, ". Value of ||d_k|| = $d_norm and value of r = $r")
        error_message = string(error_message, " Reason (6c) failed to be satisfied.")
    end
    if failure_reason_6d
        message = string(
            message,
            ". Value of dot(g, d) + 0.5 * dot(d, H * d) is $model_val, value of -γ_3 * 0.5 * δ * (norm(d)) ^ 2 is $model_decrease_threshold, and value of δ_k is $δ_k.",
        )
        error_message = string(error_message, " Reason (6d) failed to be satisfied.")
    end

    # print to file also to add as a unit test case
    if !check.valid
        if print_level >= 2
            println("=============================================")
            println(message)
            println("=============================================")
            println(
                "hard_case, original_δ, δ_k, δ_prime_k, γ_1, γ_2, γ_3, r, min_grad are $hard_case, $original_δ, $δ_k, $δ_prime_k, $γ_1, $γ_2, $γ_3, $r, $min_grad.",
            )
            println("=============================================")
        end
        if write_failure_details && print_level >= 2 && problem_name != "problem_name"
            failure_index, failure_stem = nextSubproblemFailureDetailsStem(problem_name)
            gradient_file = "$(failure_stem)_gradient.txt"
            hessian_file = "$(failure_stem)_hessian.txt"
            info_file = "$(failure_stem)_info.json"
            writedlm(gradient_file, g, ",")
            hessian_nnz = writeSparseMatrixTriplet(hessian_file, H)
            diagnostic_context = getSubproblemFailureDiagnosticContext()
            algorithm_counter_info =
                get(diagnostic_context, "algorithm_counter_before_validation", nothing)
            total_factorizations_before_validation =
                algorithm_counter_info === nothing ? nothing :
                get(algorithm_counter_info, "total_number_factorizations", nothing)
            local_total_factorizations =
                local_factorization_counts === nothing ? nothing :
                get(local_factorization_counts, "total", nothing)
            total_factorizations_estimate =
                total_factorizations_before_validation === nothing ||
                local_total_factorizations === nothing ? nothing :
                total_factorizations_before_validation + local_total_factorizations
            failure_reasons = String[]
            if failure_reason_6a
                push!(failure_reasons, "6a_stationarity")
            end
            if failure_reason_6b
                push!(failure_reasons, "6b_boundary")
            end
            if failure_reason_6c
                push!(failure_reasons, "6c_trust_region_radius")
            end
            if failure_reason_6d
                push!(failure_reasons, "6d_model_decrease")
            end
            info = Dict{String,Any}(
                "problem_name" => problem_name,
                "failure_index" => failure_index,
                "timestamp" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                "validation_context" => validation_context,
                "iteration" => get(diagnostic_context, "iteration", nothing),
                "failure_message" => error_message,
                "failure_details" => message,
                "failure_reasons" => failure_reasons,
                "failure_reason_6a" => failure_reason_6a,
                "failure_reason_6b" => failure_reason_6b,
                "failure_reason_6c" => failure_reason_6c,
                "failure_reason_6d" => failure_reason_6d,
                "hard_case" => hard_case,
                "original_delta" => original_δ,
                "input_delta_k" => input_δ_k,
                "effective_delta_k" => δ_k,
                "delta_prime_k" => δ_prime_k,
                "gamma_1" => γ_1,
                "gamma_2" => γ_2,
                "gamma_3" => γ_3,
                "radius" => r,
                "min_grad" => min_grad,
                "step_norm" => d_norm,
                "boundary_lower_bound_gamma_2_radius" => boundary_lower_bound,
                "stationarity_tolerance_gamma_1_min_grad" => stationarity_tolerance,
                "unshifted_residual_norm" => unshifted_residual_norm,
                "shifted_residual_norm" => shifted_residual_norm,
                "delta_times_step_norm" => input_δ_k * d_norm,
                "model_value" => model_val,
                "model_decrease_threshold" => model_decrease_threshold,
                "conditions" => Dict{String,Any}(
                    "unshifted_stationarity" => condition_6a,
                    "shifted_stationarity" => !failure_reason_6a,
                    "boundary" => !failure_reason_6b,
                    "trust_region_radius" => !failure_reason_6c,
                    "model_decrease" => !failure_reason_6d,
                    "interior_acceptance_model_decrease" => condition_6d,
                ),
                "algorithm_counter_before_validation" => algorithm_counter_info,
                "local_factorization_counts" => local_factorization_counts,
                "total_factorizations_so_far_estimate" => total_factorizations_estimate,
                "gradient_file" => basename(gradient_file),
                "hessian_file" => basename(hessian_file),
                "hessian_format" => "sparse_triplet_csv",
                "hessian_nnz" => hessian_nnz,
                "hessian_size" => collect(size(H)),
            )
            open(info_file, "w") do io
                JSON.print(io, info, 4)
                println(io)
            end
        end
        throw(
            TrustRegionSubproblemError(
                error_message,
                failure_reason_6a,
                failure_reason_6b,
                failure_reason_6c,
                failure_reason_6d,
            ),
        )
    end
    if print_level >= 2
        println("##########ACCEPTING STEP##########")
    end
end
