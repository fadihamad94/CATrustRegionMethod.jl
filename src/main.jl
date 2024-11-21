"""
  computeSecondOrderModel(g, hessian_current,d)
  Computes the second-order Taylor series expansion around the current iterate:
  	1/2 d ^ T * H * d + g ^ T  * d (1)

  # Inputs:
	- `g::Vector{Float64}`. The gradient at the current iterate x.
	- `H::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
		The Hessian at the current iterate x.
	- `d::Vector{Float64}`. The serach direction.
  # Output:
   Scalar that represents the value of the second-order Taylor series expansion around the current iterate.
"""
function computeSecondOrderModel(
    g::Vector{Float64},
    H::Union{
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    },
    d::Vector{Float64},
)
    return transpose(g) * d + 0.5 * transpose(d) * H * d
end

"""
  compute_ρ_hat(fval_current, fval_next, gval_current, gval_next, hessian_current, d_k, θ, print_level)

  Computes ρ_hat based on our approach: the ratio of the actual reduction in the objective function to the predicted reduction based
  on the second-order Taylor series expansion.
  ρ_hat = (fval_next - fval_current) / (-M_k + 0.5 * θ * min(||gval_current||, ||gval_next||) * ||d_k||)
  where M_k = 1/2 d ^ T * H * d + g ^ T  * d

  # Inputs:
    - `fval_current::Float64`. The function value at the current iterate x_k.
    - `fval_next::Float64`. The function value at the next candidate iterate x_k + d_k.
	- `gval_current::Vector{Float64}`. The gradient at the current iterate x_k.
	- `gval_next::Vector{Float64}`. The gradient at the next candidate iterate x_k + d_k.
	- `hessian_current::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
		The Hessian at the current iterate x_k.
	- `d_k::Vector{Float64}`. The serach direction.
	- `θ::Float64`. Param used for the preducted reduction.
	- `print_level::Float64`. The verbosity level of logs.
  # Output:
   Scalar that represents the value of the ratio of the actual reduction in the objective function to the predicted
   reduction based on the second-order Taylor series expansion.
"""
function compute_ρ_hat(
    fval_current::Float64,
    fval_next::Float64,
    gval_current::Vector{Float64},
    gval_next::Vector{Float64},
    hessian_current::Union{
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    },
    d_k::Vector{Float64},
    θ::Float64,
    print_level::Int64 = 0,
)
    second_order_model_value_current_iterate =
        computeSecondOrderModel(gval_current, hessian_current, d_k)
    guarantee_factor =
        θ * 0.5 * min(norm(gval_current, 2), norm(gval_next, 2)) * norm(d_k, 2)
    actual_fct_decrease = fval_current - fval_next
    predicted_fct_decrease = -second_order_model_value_current_iterate
    ρ_hat = actual_fct_decrease / (predicted_fct_decrease + guarantee_factor)
    if print_level >= 1 && ρ_hat == -Inf || isnan(ρ_hat)
        println(
            "ρ_hat is $ρ_hat. actual_fct_decrease is $actual_fct_decrease, predicted_fct_decrease is $predicted_fct_decrease, and guarantee_factor is $guarantee_factor.",
        )
    end
    return ρ_hat, actual_fct_decrease, predicted_fct_decrease, guarantee_factor
end

"""
  compute_ρ_standard_trust_region_method(fval_current, fval_next, gval_current, gval_next, hessian_current, d_k, print_level)

  Computes ρ based on standard trust-region methods: the ratio of the actual reduction in the objective function to the
  predicted reduction based on the second-order Taylor series expansion.
  ρ_hat = (fval_next - fval_current) / (-M_k)
  where M_k = 1/2 d ^ T * H * d + g ^ T  * d

  # Inputs:
    - `fval_current::Float64`. The function value at the current iterate x_k.
    - `fval_next::Float64`. The function value at the next candidate iterate x_k + d_k.
	- `gval_current::Vector{Float64}`. The gradient at the current iterate x_k.
	- `gval_next::Vector{Float64}`. The gradient at the next candidate iterate x_k + d_k.
	- `hessian_current::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}}`.
		The Hessian at the current iterate x_k.
	- `d_k::Vector{Float64}`. The serach direction.
	- `print_level::Float64`. The verbosity level of logs.
  # Output:
   Scalar that represents the value of the ratio of the actual reduction in the objective function to the predicted
   reduction based on the second-order Taylor series expansion.
"""
function compute_ρ_standard_trust_region_method(
    fval_current::Float64,
    fval_next::Float64,
    gval_current::Vector{Float64},
    hessian_current::Union{
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    },
    d_k::Vector{Float64},
    print_level::Int64 = 0,
)
    second_order_model_value_current_iterate =
        computeSecondOrderModel(gval_current, hessian_current, d_k)
    actual_fct_decrease = fval_current - fval_next
    predicted_fct_decrease = -second_order_model_value_current_iterate
    ρ = actual_fct_decrease / predicted_fct_decrease
    if print_level >= 1 && ρ == -Inf || isnan(ρ)
        println(
            "ρ is $ρ. actual_fct_decrease is $actual_fct_decrease and predicted_fct_decrease is $predicted_fct_decrease.",
        )
    end
    return ρ, actual_fct_decrease, predicted_fct_decrease
end

"""
  sub_routine_trust_region_sub_problem_solver(fval_current, gval_current, hessian_current, x_k, δ_k, γ_1, γ_2, r_k, min_gval_norm, nlp, print_level, trust_region_subproblem_solver)
  The method computes a search direction that solves the trust-region subproblem. The method checks for numerical erros to
  validate that the trust-region subproblem was solved correctly. If the second order model (1) is positive, then we
  failed to solve the trusrt-region subproblem.

  See optimizeSecondOrderModel in ./trust_region_subproblem_solver.jl and ./trust_region_subproblem_solver.jl for more details on the implementation, on the
  inputs description, and the outputs description.
"""
function sub_routine_trust_region_sub_problem_solver(
    fval_current::Float64,
    gval_current::Vector{Float64},
    hessian_current::Union{
        SparseMatrixCSC{Float64,Int64},
        Symmetric{Float64,SparseMatrixCSC{Float64,Int64}},
    },
    x_k::Vector{Float64},
    δ_k::Float64,
    γ_1::Float64,
    γ_2::Float64,
    r_k::Float64,
    min_gval_norm::Float64,
    nlp::Union{AbstractNLPModel,MathOptNLPModel,MathOptInterface.NLPBlockData,CUTEstModel},
    algorithm_counter::AlgorithmCounter,
    print_level::Int64,
    trust_region_subproblem_solver::String = "NEW",
)
    fval_next = fval_current
    gval_next_temp = gval_current
    start_time_temp = time()

    # Solve the trust-region subproblem to generate the search direction d_k
    success_subproblem_solve, d_k, hard_case = nothing, nothing, nothing

    if trust_region_subproblem_solver == "OLD"
        success_subproblem_solve, δ_k, d_k, hard_case =
            solveTrustRegionSubproblemOldApproach(
                fval_current,
                gval_current,
                hessian_current,
                x_k,
                δ_k,
                γ_2,
                r_k,
            )
    else
        problem_name = "Generic"
        if typeof(nlp) == AbstractNLPModel ||
           typeof(nlp) == MathOptNLPModel ||
           typeof(nlp) == CUTEstModel
            problem_name = nlp.meta.name
        end
        success_subproblem_solve, δ_k, d_k, hard_case = solveTrustRegionSubproblem(
            problem_name,
            fval_current,
            gval_current,
            hessian_current,
            x_k,
            δ_k,
            γ_1,
            γ_2,
            r_k,
            min_gval_norm,
            algorithm_counter,
            print_level,
        )
    end
    end_time_temp = time()
    total_time_temp = end_time_temp - start_time_temp
    if print_level >= 2
        println("solveTrustRegionSubproblem operation took $total_time_temp.")
    end

    if !success_subproblem_solve
        throw(
            TrustRegionSubproblemError(
                "Trust-region subproblem failure.",
                true,
                true,
                true,
                true,
            ),
        )
    end

    start_time_temp = time()
    second_order_model_value_current_iterate =
        computeSecondOrderModel(gval_current, hessian_current, d_k)
    end_time_temp = time()
    total_time_temp = end_time_temp - start_time_temp
    if print_level >= 2
        println("computeSecondOrderModel operation took $total_time_temp.")
    end

    # When we are able to solve the trust-region subproblem, we check for numerical error
    # in computing the predicted reduction from the second order model M_k. If no numerical
    # errors, we compute the objective function for the candidate solution to check if
    # we will accept the step in case this leads to reduction in the function value.
    if second_order_model_value_current_iterate < 0
        start_time_temp = time()
        fval_next = evalFunction(nlp, x_k + d_k, algorithm_counter)
        end_time_temp = time()
        total_time_temp = end_time_temp - start_time_temp
        if print_level >= 2
            println("fval_next operation took $total_time_temp.")
        end
    end

    return fval_next, success_subproblem_solve, δ_k, d_k, hard_case
end

function CAT_solve(m::JuMP.Model)
    nlp_raw = MathOptNLPModel(m)
    return CAT_solve(nlp_raw)
end

function CAT_solve(nlp_raw::NLPModels.AbstractNLPModel)
    termination_criteria = TerminationCriteria()
    algorithm_params = AlgorithmicParameters()
    return CAT_solve(nlp_raw, termination_criteria, algorithm_params)
end

function CAT_solve(
    m::JuMP.Model,
    termination_criteria::TerminationCriteria,
    algorithm_params::AlgorithmicParameters,
)
    nlp_raw = MathOptNLPModel(m)
    return CAT_solve(nlp_raw, termination_criteria, algorithm_params)
end

function _i_not_fixed(variable_info::Vector{VariableInfo})
    vector_indixes = zeros(Int64, 0)
    for i = 1:length(variable_info)
        if !variable_info[i].is_fixed
            push!(vector_indixes, i)
        end
    end
    return vector_indixes
end

function CAT_solve(
    solver::CATSolver,
    termination_criteria::TerminationCriteria,
    algorithm_params::AlgorithmicParameters,
)
    ind = _i_not_fixed(solver.variable_info)
    non_fixed_variables = solver.variable_info[ind]
    starting_points_vector = zeros(0)
    for variable in non_fixed_variables
        push!(starting_points_vector, variable.start == nothing ? 0.0 : variable.start)
    end
    x = deepcopy(starting_points_vector)
    δ = 0.0
    return optimize(solver.nlp_data, termination_criteria, algorithm_params, x, δ)
end

function CAT_solve(
    nlp_raw::NLPModels.AbstractNLPModel,
    termination_criteria::TerminationCriteria,
    algorithm_params::AlgorithmicParameters,
)
    if ncon(nlp_raw) > 0
        throw(ErrorException("Constrained minimization problems are unsupported"))
    end
    δ = 0.0
    x = nlp_raw.meta.x0
    return optimize(nlp_raw, termination_criteria, algorithm_params, x, δ)
end

"""
  optimize(uconstrainedMinimizationProblem, termination_criteria, algorithm_params, x, δ)
  This is the main algorithm code.

  # Inputs:
    - `problem::Union{AbstractNLPModel,MathOptNLPModel, MathOptInterface.NLPBlockData,CUTEstModel}`. The utility to compute function, gradient, and Hessian. See Problem_Data struct in ./common.jl for more details.
    - `termination_criteria::TerminationCriteria. This contains all the termination conditions such as tolerance for gradient norm. See TerminationCriteria struct in ./common.jl for more details.``
    - `algorithm_params::AlgorithmicParameters. This contains all the necessary algorithm params such as initial radius, the θ param that is used for computing ρ_hat and so forth. See AlgorithmicParameters struct in ./common.jl for more details.`
    - `x::Vector{Float64}`. The initial iterate.
	- `δ::Float64`. The initial warm start value for δ. See optimizeSecondOrderModel in ./trust_region_subproblem_solver.jl for more details.

  # Outputs:
    - `x_k::Vector{Float64}`. The final iterate.
	- `status::TerminationStatusCode`. The algorithm termination status. Check TerminationStatusCode struct in ./common.jl for more details.
	- `iteration_stats::DataFrame`. It constains the algorithm history for the function and gradient value at each iteration.
	- `algorithm_counter::AlgorithmCounter`. It contains the total number of function evaluations, gradient evaluations , Hessian evaluations,
	and total number of factorizations.
	- `k::Int64`. The total number of iterations.
	- `total_execution_time::Float64.` The total Wall clock time (seconds).
"""
function optimize(
    nlp::Union{AbstractNLPModel,MathOptNLPModel,MathOptInterface.NLPBlockData,CUTEstModel},
    termination_criteria::TerminationCriteria,
    algorithm_params::AlgorithmicParameters,
    x::Vector{Float64},
    δ::Float64,
)
    start_time_ = time()
    @assert(δ >= 0)
    @assert nlp != nothing

    #Termination conditions
    MAX_ITERATIONS = termination_criteria.MAX_ITERATIONS
    MAX_TIME = termination_criteria.MAX_TIME
    gradient_termination_tolerance = termination_criteria.gradient_termination_tolerance
    STEP_SIZE_LIMIT = termination_criteria.STEP_SIZE_LIMIT
    MINIMUM_OBJECTIVE_FUNCTION = termination_criteria.MINIMUM_OBJECTIVE_FUNCTION

    #Algorithm parameters
    β = algorithm_params.β
    ω_1 = algorithm_params.ω_1
    ω_2 = algorithm_params.ω_2
    γ_1 = algorithm_params.γ_1
    γ_2 = algorithm_params.γ_2
    γ_3 = algorithm_params.γ_3
    θ = algorithm_params.θ
    ξ = algorithm_params.ξ
    eval_offset = algorithm_params.eval_offset
    seed = algorithm_params.seed

    #Initial radius
    r_1 = algorithm_params.r_1
    INITIAL_RADIUS_MULTIPLICATIVE_RULE = algorithm_params.INITIAL_RADIUS_MULTIPLICATIVE_RULE

    #Initial conditions
    x_k = x
    δ_k = δ
    r_k = r_1

    print_level = algorithm_params.print_level
    radius_update_rule_approach = algorithm_params.radius_update_rule_approach

    #Algorithm history
    iteration_stats = DataFrame(k = [], fval = [], gradval = [])

    #Algorithm stats
    algorithm_counter = AlgorithmCounter()

    # Trust-region subproblem method
    trust_region_subproblem_solver = algorithm_params.trust_region_subproblem_solver

    #Specify the seed for reproducibility of results
    Random.seed!(seed)

    k = 1
    try
        gval_current = evalGradient(nlp, x_k, algorithm_counter)
        fval_current = evalFunction(nlp, x_k, algorithm_counter)
        hessian_current = evalHessian(nlp, x_k, algorithm_counter)

        #If user doesn't change the starting radius, we select the radius as described in the paper:
        #Initial radius heuristic selection rule : r_1 = INITIAL_RADIUS_MULTIPLICATIVE_RULE * ||gval_current|| / ||hessian_current||
        if r_k <= 0.0
            matrix_l2_norm_val = matrix_l2_norm(hessian_current, num_iter = 20)
            if matrix_l2_norm_val > 0
                r_k =
                    INITIAL_RADIUS_MULTIPLICATIVE_RULE * norm(gval_current, 2) /
                    matrix_l2_norm_val
            else
                r_k = 1.0
            end
        end

        compute_hessian = false
        if norm(gval_current, 2) <= gradient_termination_tolerance
            if print_level >= 0
                println("*********************************Iteration Count: ", 1)
            end
            push!(iteration_stats, (1, fval_current, norm(gval_current, 2)))
            end_time_ = time()
            total_execution_time = end_time_ - start_time_
            return x_k,
            TerminationStatusCode.OPTIMAL,
            iteration_stats,
            algorithm_counter,
            1,
            total_execution_time
        end

        start_time = time()
        min_gval_norm = norm(gval_current, 2)
        while k <= MAX_ITERATIONS
            temp_grad = gval_current
            if print_level >= 1
                start_time_str = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
                println("$start_time. Iteration $k with radius $r_k.")
            end
            if compute_hessian
                start_time_temp = time()
                hessian_current = evalHessian(nlp, x_k, algorithm_counter)
                end_time_temp = time()
                total_time_temp = end_time_temp - start_time_temp
                if print_level >= 2
                    println("hessian_current operation took $total_time_temp.")
                end
            end

            # Solve the trsut-region subproblem and generate the search direction d_k
            fval_next, success_subproblem_solve, δ_k, d_k, hard_case =
                sub_routine_trust_region_sub_problem_solver(
                    fval_current,
                    gval_current,
                    hessian_current,
                    x_k,
                    δ_k,
                    γ_1,
                    γ_2,
                    r_k,
                    min_gval_norm,
                    nlp,
                    algorithm_counter,
                    print_level,
                    trust_region_subproblem_solver,
                )

            gval_next = gval_current
            # When we are able to solve the trust-region subproblem, we compute ρ_k to check if the
            # candidate solution has a reduction in the function value so that we accept the step
            if success_subproblem_solve
                if fval_next <=
                   fval_current +
                   ξ * min_gval_norm * norm(d_k) +
                   (1 + abs(fval_current)) * eval_offset
                    temp_grad = evalGradient(nlp, x_k + d_k, algorithm_counter)
                    temp_norm = norm(temp_grad, 2)
                    if isnan(temp_norm)
                        if print_level >= 0
                            println("$k. grad(nlp, x_k + d_k) is NaN.")
                        end
                        @warn "$k grad(nlp, x_k + d_k) is NaN."
                    else
                        min_gval_norm = min(min_gval_norm, temp_norm)
                    end
                end

                start_time_temp = time()
                ρ_k, actual_fct_decrease, predicted_fct_decrease =
                    compute_ρ_standard_trust_region_method(
                        fval_current,
                        fval_next,
                        gval_current,
                        hessian_current,
                        d_k,
                        print_level,
                    )
                end_time_temp = time()
                total_time_temp = end_time_temp - start_time_temp
                if print_level >= 2
                    println(
                        "compute_ρ_standard_trust_region_method operation took $total_time_temp.",
                    )
                end
            else
                # In case we fail to solve the trust-region subproblem, we mark that as a failure
                # by setting ρ_k to a negative default value (-1.0) and  the search direction d_k to 0 vector
                ρ_k = -1.0
                actual_fct_decrease = 0.0
                predicted_fct_decrease = 0.0
                d_k = zeros(length(x_k))
            end
            if print_level >= 1
                println(
                    "Iteration $k with fval_next is $fval_next and fval_current is $fval_current.",
                )
                println(
                    "Iteration $k with actual_fct_decrease is $actual_fct_decrease and predicted_fct_decrease is $predicted_fct_decrease.",
                )
            end
            if fval_next > fval_current
                ρ_hat_k = -1.0
            else
                ρ_hat_k = ρ_k
            end
            norm_gval_current = norm(gval_current, 2)
            norm_gval_next = norm_gval_current
            # Accept the generated search direction d_k when ρ_k is positive
            # and compute ρ_hat_k for the radius update rule
            if ρ_k >= 0.0 && (fval_next <= fval_current)
                if print_level >= 1
                    println(
                        "$k. =======STEP IS ACCEPTED========== $ρ_k =========fval_next is $fval_next and fval_current is $fval_current.",
                    )
                end
                if !success_subproblem_solve && norm(d_k) > 0
                    @warn(
                        "Warning: accepting search direction even with TRS failure. Search direction is a descent direction."
                    )
                    println(
                        "Warning: accepting search direction even with TRS failure. Search direction is a descent direction.",
                    )
                end
                x_k = x_k + d_k
                start_time_temp = time()
                gval_next = temp_grad
                if isnan(min_gval_norm)
                    if print_level >= 0
                        println("$k. min_gval_norm is NaN")
                    end
                    @warn "$k min_gval_norm is NaN."
                end
                end_time_temp = time()
                total_time_temp = end_time_temp - start_time_temp
                if print_level >= 2
                    println("gval_next operation took $total_time_temp.")
                end
                start_time_temp = time()
                ρ_hat_k, actual_fct_decrease, predicted_fct_decrease, guarantee_factor =
                    compute_ρ_hat(
                        fval_current,
                        fval_next,
                        gval_current,
                        gval_next,
                        hessian_current,
                        d_k,
                        θ,
                        print_level,
                    )
                end_time_temp = time()
                total_time_temp = end_time_temp - start_time_temp
                if print_level >= 2
                    println("compute_ρ_hat operation took $total_time_temp.")
                end

                fval_current = fval_next
                gval_current = gval_next
                compute_hessian = true
            else
                #else x_k+1 = x_k, fval_current, gval_current, hessian_current will not change
                compute_hessian = false
            end
            if print_level >= 1
                println("$k. ρ_hat_k is $ρ_hat_k.")
                println("$k. hard_case is $hard_case")
                norm_d_k = norm(d_k, 2)
                println("$k. r_k is $r_k and ||d_k|| is $norm_d_k.")
            end
            # Radius update
            if radius_update_rule_approach == "DEFAULT"
                if !success_subproblem_solve || isnan(ρ_hat_k) || ρ_hat_k < β
                    r_k = r_k / ω_1
                else
                    r_k = max(ω_2 * norm(d_k, 2), r_k)
                end
                # This to be able to test the performance of the algorithm for the ablation study
            else
                if !success_subproblem_solve
                    r_k = r_k / ω_1
                else
                    if isnan(ρ_hat_k) || ρ_hat_k < β
                        r_k = norm(d_k, 2) / ω_1
                    else
                        r_k = ω_1 * norm(d_k, 2)
                    end
                end
            end

            push!(iteration_stats, (k, fval_current, norm(gval_current, 2)))
            # Check termination condition for gradient
            if norm(gval_next, 2) <= gradient_termination_tolerance ||
               min_gval_norm <= gradient_termination_tolerance
                push!(iteration_stats, (k, fval_next, min_gval_norm))
                if print_level >= 0
                    println("*********************************Iteration Count: ", k)
                end
                if print_level >= 2
                    try
                        cholesky(Matrix(hessian_current))
                        println("==============Local Minimizer=============")
                    catch e
                        println("==============Saddle Point=============")
                    end
                end
                end_time_ = time()
                total_execution_time = end_time_ - start_time_
                return x_k,
                TerminationStatusCode.OPTIMAL,
                iteration_stats,
                algorithm_counter,
                k,
                total_execution_time
            end

            # Check termination condition for step size if it becomes too small
            if r_k <= STEP_SIZE_LIMIT || 0 < norm(d_k) <= STEP_SIZE_LIMIT
                if print_level >= 0
                    println("$k. Step size is too small.")
                end
                end_time_ = time()
                total_execution_time = end_time_ - start_time_
                return x_k,
                TerminationStatusCode.STEP_SIZE_LIMIT,
                iteration_stats,
                algorithm_counter,
                k,
                total_execution_time
            end

            # Check termination condition for function value if the objective function is unbounded (safety check)
            if fval_current <= MINIMUM_OBJECTIVE_FUNCTION ||
               fval_next <= MINIMUM_OBJECTIVE_FUNCTION
                if print_level >= 0
                    println(
                        "$k. Function values ($fval_current, $fval_next) are too small.",
                    )
                end
                end_time_ = time()
                total_execution_time = end_time_ - start_time_
                return x_k,
                TerminationStatusCode.UNBOUNDED,
                iteration_stats,
                algorithm_counter,
                k,
                total_execution_time
            end

            # Check termination condition for time if we exceeded the time limit
            if time() - start_time > MAX_TIME
                end_time_ = time()
                total_execution_time = end_time_ - start_time_
                return x_k,
                TerminationStatusCode.TIME_LIMIT,
                iteration_stats,
                algorithm_counter,
                k,
                total_execution_time
            end
            k += 1
        end
        # Handle exceptions
    catch e
        @error e
        status = TerminationStatusCode.OTHER_ERROR
        if isa(e, OutOfMemoryError)
            status = TerminationStatusCode.MEMORY_LIMIT
        end
        if isa(e, TrustRegionSubproblemError)
            status = TerminationStatusCode.TRUST_REGION_SUBPROBLEM_ERROR
            problem_name = "Generic"
            if typeof(nlp) == AbstractNLPModel ||
               typeof(nlp) == MathOptNLPModel ||
               typeof(nlp) == CUTEstModel
                problem_name = nlp.meta.name
            end
            failure_reason_6a = e.failure_reason_6a
            failure_reason_6b = e.failure_reason_6b
            failure_reason_6c = e.failure_reason_6c
            failure_reason_6d = e.failure_reason_6d
            printFailures(
                problem_name,
                failure_reason_6a,
                failure_reason_6b,
                failure_reason_6c,
                failure_reason_6d,
            )
        end
        end_time_ = time()
        total_execution_time = end_time_ - start_time_
        return x_k, status, iteration_stats, algorithm_counter, k, total_execution_time
    end

    end_time_ = time()
    total_execution_time = end_time_ - start_time_
    return x_k,
    TerminationStatusCode.ITERATION_LIMIT,
    iteration_stats,
    algorithm_counter,
    k,
    total_execution_time
end

function evalFunction(
    nlp::Union{
        AbstractNLPModel,
        MathOptNLPModel,
        MathOptInterface.NLPBlockData,
        CUTEstModel,
        Nothing,
    },
    x::Vector{Float64},
    algorithm_counter::AlgorithmCounter,
)
    increment!(algorithm_counter, :total_function_evaluation)
    if typeof(nlp) == AbstractNLPModel ||
       typeof(nlp) == MathOptNLPModel ||
       typeof(nlp) == CUTEstModel
        return obj(nlp, x)
    else
        return MOI.eval_objective(nlp.evaluator, x)
    end
end

function evalGradient(
    nlp::Union{AbstractNLPModel,MathOptNLPModel,MathOptInterface.NLPBlockData,CUTEstModel},
    x::Vector{Float64},
    algorithm_counter::AlgorithmCounter,
)
    increment!(algorithm_counter, :total_gradient_evaluation)
    if typeof(nlp) == AbstractNLPModel ||
       typeof(nlp) == MathOptNLPModel ||
       typeof(nlp) == CUTEstModel
        return grad(nlp, x)
    else
        gval = zeros(length(x))
        MOI.eval_objective_gradient(nlp.evaluator, gval, x)
        return gval
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

function evalHessian(
    nlp::Union{AbstractNLPModel,MathOptNLPModel,MathOptInterface.NLPBlockData,CUTEstModel},
    x::Vector{Float64},
    algorithm_counter::AlgorithmCounter,
)
    increment!(algorithm_counter, :total_hessian_evaluation)
    if typeof(nlp) == AbstractNLPModel ||
       typeof(nlp) == MathOptNLPModel ||
       typeof(nlp) == CUTEstModel
        return hess(nlp, x)
    else
        hessian = spzeros(Float64, (length(x), length(x)))
        d = nlp.evaluator
        hessian_sparsity = MOI.hessian_lagrangian_structure(d)
        #Sparse hessian entry vector
        H_vec = zeros(Float64, length(hessian_sparsity))
        MOI.eval_hessian_lagrangian(d, H_vec, x, 1.0, zeros(0))
        if !isempty(hessian_sparsity)
            index = 1
            if length(hessian_sparsity) < 3
                index = length(hessian_sparsity)
            elseif length(x) > 1
                index = length(H_vec)
            end
            for i = 1:index
                hessian[hessian_sparsity[i][1], hessian_sparsity[i][2]] += H_vec[i]
            end
        end
        return restoreFullMatrix(hessian)
    end
end
