using LinearAlgebra, SparseArrays, DataFrames, Plots
include("./hard_example_implementation_code.jl")
export computeNewtonSteps
function computeNewtonSteps(k::Int64, η::Float64, x_k::Vector{Float64})
    d_k = compute_d_k(k, η)
    x_k = x_k + d_k
    return x_k, d_k
end

function computeNewtonStepsModified(g::Vector{Float64}, H::SparseMatrixCSC{Float64,Int64})
    d_k = H \ (-g)
    return d_k
end

#The way the example is build is that for standard trust region, we will only make Newton's steps.
function optimize_modified(x_0::Vector{Float64}, η::Float64, α_k::Float64, gradient_termination_tolerance::Float64, MAX_ITERATION::Int64)
    #The way this example was built is to have always ρ = 1 and so if we want implement standard trust-region, it
    #will behave exactly similar to Newtons' method. Especially that the Hessian is always positive semidefinite
    #Also the function value is always decreasing
    k = 0
    x_k = x_0
    d_k = [0.0, 0.0]
    f2_1_0 = 25000.3
    f2_2_0 = (π ^ 2) / 6
    g_k = 0
    f1_k = f2_1_0
    f2_2_comma_k = f2_2_0
    g_k = compute_gradient2_at_kth_iterate(k, η)
    H = compute_hessian2_at_kth_iterate(k)
    iteration_stats = DataFrame(k = [], xval = [], stepval = [], fval = [], gval = [])
    for k in 0:MAX_ITERATION
        g_k = compute_g2_close_form(x_k, x_k, η, k, k, α_k)
        H = compute_H2_close_form(x_k, x_k, η, k, k, α_k)
        d_k = computeNewtonStepsModified(g_k, H)
        x_k_2, d_k_2 = computeNewtonSteps(k, η, x_k)

        f1_k = compute_f1_close_form_k(x_k[1] + d_k[1], x_k[1], η, k, α_k, f2_1_0)
        f2_2_comma_k = compute_f2_2coma_close_form_k(x_k[2] + d_k[2], x_k[2], η, k, f2_2_0)
        f_k = f1_k + f2_2_comma_k
        push!(iteration_stats, (k, x_k + d_k, d_k, f_k, norm(g_k, 2)))
        if (norm(g_k, 2) <= gradient_termination_tolerance)
            H = compute_hessian2_at_kth_iterate(k)
            cholesky(Matrix(H))
            return x_k, "Optimal", iteration_stats
        end
        x_k = x_k + d_k
    end
    return x_k, "MAX_ITERATION", iteration_stats
end


function optimize(x_0::Vector{Float64}, η::Float64, α_k::Float64, gradient_termination_tolerance::Float64, MAX_ITERATION::Int64)
    #The way this example was built is to have always ρ = 1 and so if we want implement standard trust-region, it
    #will behave exactly similar to Newtons' method. Especially that the Hessian is always positive semidefinite
    #Also the function value is always decreasing
    k = 0
    x_k = x_0
    d_k = [0.0, 0.0]
    f2_1_0 = 25000.3
    f2_2_0 = (π ^ 2) / 6
    g_k = 0
    f1_k = f2_1_0
    f2_2_comma_k = f2_2_0
    f1_k = coumpute_f1_next_iterate_value(f1_k, η, k, α_k)
    f2_2_comma_k = coumpute_f2_2_next_iterate_value(f2_2_comma_k, η, k)
    f_k = compute_f2_close_form(x_k, x_k - d_k, η, k, α_k, f1_k, f2_2_comma_k)
    g_k = compute_gradient2_at_kth_iterate(k, η)
    iteration_stats = DataFrame(k = [], xval = [], stepval = [], fval = [], gval = [])
    for k in 0:MAX_ITERATION
        g_k = compute_gradient2_at_kth_iterate(k, η)
        x_k, d_k = computeNewtonSteps(k, η, x_k)
        f1_k = coumpute_f1_next_iterate_value(f1_k, η, k, α_k)
        f2_2_comma_k = coumpute_f2_2_next_iterate_value(f2_2_comma_k, η, k)
        f_k = compute_f2_close_form(x_k, x_k - d_k, η, k, α_k, f1_k, f2_2_comma_k)
        push!(iteration_stats, (k, x_k - d_k, d_k, f_k, norm(g_k, 2)))
        if (norm(g_k, 2) <= gradient_termination_tolerance)
            H = compute_hessian2_at_kth_iterate(k)
            cholesky(Matrix(H))
            return x_k, "Optimal", iteration_stats
        end
    end
    return x_k, "MAX_ITERATION", iteration_stats
end
