using LinearAlgebra, DataFrames, SparseArrays, Plots
include("./hard_example_implementation_code.jl")
include("./complexity_standard_trust_region_example.jl")

function plotf1(η::Float64, α_k::Float64, K::Int64)
    f1_0 = 25000.3
    fval_current = f1_0
    function_value_evaluation_history= DataFrame(k = [], fval = [])
    push!(function_value_evaluation_history, (0, fval_current))
    for k in 1:K
        fval_current = coumpute_f1_next_iterate_value(fval_current, η, k, α_k)
        push!(function_value_evaluation_history, (k, fval_current))
    end
    x = [0, 20 , 40, 60, 80, 100, 120, 140, 160]
    index = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    plot!(function_value_evaluation_history.k,
        function_value_evaluation_history.fval,
        title = "Function Value",
        #label="f-val",
        ylabel="fval",
        xlabel="k",
        xticks = (x, index)
        #xticks = (0:K, string.(0:K))
        #legend=:bottomright,
    )

    png("f1_function_value_example_paper_plot")
    return function_value_evaluation_history
end

function plotf1_contineous(η::Float64, α_k::Float64, K::Int64)
    f1_0 = 25000.3
    function_value_evaluation_history= DataFrame(k = [], x_k = [], fval = [])
    x_k = 0.0
    f1_k_plus_1 = f1_0
    push!(function_value_evaluation_history, (0, x_k, f1_k_plus_1))
    temp_x_k = x_k
    for k in 0:(K - 1)
        # f1_k_plus_1 = coumpute_f1_next_iterate_value(f1_k_plus_1, η, k, α_k)
        d_k  = compute_d_k(k, η)
        increment = d_k[1] / 10
        for i in 1:10
            fval_current = compute_f1_close_form_k(x_k + i * increment, x_k, η, k, α_k, f1_0)
            # fval_current = compute_f1_close_form(x_k + i * increment, x_k, η, k, α_k, f1_k_plus_1)
            push!(function_value_evaluation_history, (i + (10 * (k)), x_k + i * increment, fval_current))
            # x_k = x_k + i * increment
            temp_x_k = x_k + i * increment
        end
        x_k = x_k + d_k[1]
        if x_k != temp_x_k
            print("!!!!!!!!!")
        end
    end
    x = [0, 20 , 40, 60, 80, 100, 120, 140, 160]
    index = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    plot!(function_value_evaluation_history.k,
        function_value_evaluation_history.fval,
        title = "Function Value",
        label="f-val",
        ylabel="fval",
        xlabel="k",
        xticks = (x, index),
        #xticks = (0:K, string.(0:K))
        legend=:topright,
    )

    png("f1_function_value_example_paper_plot_contineous")
    return function_value_evaluation_history
end

function plotg1_contineous(η::Float64, α_k::Float64, K::Int64)
    gradient_value_evaluation_history= DataFrame(k = [], x_k = [], gval = [])
    x_k = 0.0
    gval_current = compute_g1_close_form(x_k, x_k, η, 0, α_k)
    push!(gradient_value_evaluation_history, (0, x_k, gval_current[1]))
    temp_x_k = x_k
    for k in 0:(K-1)
        d_k  = compute_d_k(k, η)
        increment = d_k[1] / 10
        for i in 1:10
            gval_current = compute_g1_close_form(x_k + i * increment, x_k, η, k, α_k)
            push!(gradient_value_evaluation_history, (i + (10 * (k)), x_k + i * increment, gval_current[1]))
            # x_k = x_k + i * increment
            temp_x_k = x_k + i * increment
        end
        x_k = x_k + d_k[1]
        if x_k != temp_x_k
            print("!!!!!!!!!")
        end
    end
    x = [0, 20 , 40, 60, 80, 100, 120, 140, 160]
    index = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    plot!(gradient_value_evaluation_history.k,
        gradient_value_evaluation_history.gval,
        title = "Gradient Value",
        label="g-val",
        ylabel="gval",
        xlabel="k",
        xticks = (x, index),
        yticks = -1:0.1:0,
        #xticks = (0:K, string.(0:K))
        legend=:bottomright,
    )

    png("g1_gradient_value_example_paper_plot_contineous")
    return gradient_value_evaluation_history
end

function ploth1_contineous(η::Float64, α_k::Float64, K::Int64)
    hessian_value_evaluation_history= DataFrame(k = [], x_k = [], hval = [])
    x_k = 0.0
    temp_x_k = x_k
    hval_current = compute_H1_close_form(x_k, x_k, η, 0, α_k)
    push!(hessian_value_evaluation_history, (0, x_k, hval_current[1]))
    for k in 0:(K - 1)
        d_k  = compute_d_k(k, η)
        increment = d_k[1] / 10
        for i in 1:10
            hval_current = compute_H1_close_form(x_k + i * increment, x_k, η, k, α_k)
            push!(hessian_value_evaluation_history, (i + (10 * (k)), x_k + i * increment, hval_current[1]))
            # x_k = x_k + i * increment
            temp_x_k = x_k + i * increment
        end
        x_k = x_k + d_k[1]
        if x_k != temp_x_k
            print("!!!!!!!!!")
        end
    end
    x = [0, 20 , 40, 60, 80, 100, 120, 140, 160]
    index = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    plot!(hessian_value_evaluation_history.k,
        hessian_value_evaluation_history.hval,
        title = "Hessian Value",
        label="h-val",
        ylabel="hval",
        xlabel="k",
        xticks = (x, index),
        #xticks = (0:K, string.(0:K))
        legend=:bottomright
    )

    png("h1_hessian_value_example_paper_plot_contineous")
    return hessian_value_evaluation_history
end

function plotj1_contineous(η::Float64, α_k::Float64, K::Int64)
    jerk_value_evaluation_history= DataFrame(k = [], x_k = [], jval = [])
    x_k = 0.0
    temp_x_k = x_k
    jerk_current = compute_j1_close_form(x_k, x_k, η, 0, α_k)
    push!(jerk_value_evaluation_history, (0, x_k, jerk_current[1]))
    for k in 0:(K - 1)
        d_k  = compute_d_k(k, η)
        increment = d_k[1] / 10
        for i in 1:10
            jerk_current = compute_j1_close_form(x_k + i * increment, x_k, η, k, α_k)
            push!(jerk_value_evaluation_history, (i + (10 * (k)), x_k + i * increment, jerk_current[1]))
            # x_k = x_k + i * increment
            temp_x_k = x_k + i * increment
        end
        x_k = x_k + d_k[1]
        if x_k != temp_x_k
            print("!!!!!!!!!")
        end
    end
    x = [0, 20 , 40, 60, 80, 100, 120, 140, 160]
    index = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    # x = [0, 10 , 20, 30, 40, 50, 60, 70]
    # index = [0, 1, 2, 3, 4, 5, 6, 7]
    plot!(jerk_value_evaluation_history.k,
        jerk_value_evaluation_history.jval,
        title = "Jerk Value",
        label="j-val",
        ylabel="jval",
        xlabel="k",
        xticks = (x, index),
        yticks = -60:20:160,
        #xticks = (0:K, string.(0:K))
        legend=:topleft,
    )

    png("j1_jerk_value_example_paper_plot_contineous")
    return jerk_value_evaluation_history
end

η = 1e-4
α_k = 1.0
K = 16
#plotf1(η, α_k, K)

function plotf2(η::Float64, K::Int64)
    f2_0 = (π ^ 2) / 6
    fval_current = f2_0
    function_value_evaluation_history= DataFrame(k = [], fval = [])
    push!(function_value_evaluation_history, (0, fval_current))
    for k in 1:K
        fval_current = coumpute_f2_2_next_iterate_value(fval_current, η, k)
        push!(function_value_evaluation_history, (k, fval_current))
    end
    x = [0, 20 , 40, 60, 80, 100, 120, 140, 160]
    index = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    plot!(function_value_evaluation_history.k,
        function_value_evaluation_history.fval,
        title = "Function Value",
        #label="f-val",
        ylabel="fval",
        xlabel="k",
        xticks = (x, index)
        #xticks = (0:K, string.(0:K))
        #legend=:bottomright,
    )

    png("f2_function_value_example_paper_plot")
    return function_value_evaluation_history
end

#plotf2(η, K)

function plotf2comma2(η::Float64, K::Int64)
    x_k = Vector{Float64}([0])
    f2_0 = (π ^ 2) / 6
    f2_2_comma_k_next = f2_0
    function_value_evaluation_history= DataFrame(k = [], fval = [])
    d_k = Vector{Float64}([1])
    for k in 0:K
        f2_2_comma_k_next = coumpute_f2_2_next_iterate_value(f2_2_comma_k_next, η, k + 1)
        fval_current = compute_f2_2coma_close_form(x_k[1] + d_k[1], x_k[1], η, k + 1, f2_2_comma_k_next)
        x_k = x_k + d_k
        push!(function_value_evaluation_history, (k, fval_current))
    end
    x = [0, 20 , 40, 60, 80, 100, 120, 140, 160]
    index = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    plot!(function_value_evaluation_history.k,
        function_value_evaluation_history.fval,
        title = "Function Value",
        #label="f-val",
        ylabel="fval",
        xlabel="k",
        xticks = (x, index)
        #xticks = (0:K, string.(0:K))
        #legend=:bottomright,
    )

    png("f2comma2_function_value_example_paper_plot")
    return function_value_evaluation_history
end


function plotf2_contineous(η::Float64, K::Int64)
    # f2_0 = (π ^ 2) / 6
    f2_0 = 0.8
    function_value_evaluation_history= DataFrame(k = [], x_k = [], fval = [])
    x_k = 0.0
    temp_x_k = x_k
    f2_k_plus_1 = f2_0
    push!(function_value_evaluation_history, (0, x_k, f2_k_plus_1))
    for k in 0:(K - 1)
        d_k  = compute_d_k(k, η)
        increment = d_k[2] / 10
        for i in 1:10
            fval_current = compute_f2_2coma_close_form_k(x_k + i * increment, x_k, η, k, f2_0)
            push!(function_value_evaluation_history, (i + (10 * (k)), x_k + i * increment, fval_current))
            # x_k = x_k + i * increment
            temp_x_k = x_k + i * increment
        end
        x_k = x_k + d_k[2]
        if x_k != temp_x_k
            print("!!!!!!!!!")
        end
    end
    x = [0, 20 , 40, 60, 80, 100, 120, 140, 160]
    index = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    plot!(function_value_evaluation_history.k,
        function_value_evaluation_history.fval,
        title = "Function Value",
        label="f-val",
        ylabel="fval",
        xlabel="k",
        xticks = (x, index),
        #xticks = (0:K, string.(0:K))
        legend=:topright
    )

    png("f2_function_value_example_paper_plot_contineous")
    return function_value_evaluation_history
end

function plotg2_contineous(η::Float64, K::Int64)
    gradient_value_evaluation_history= DataFrame(k = [], x_k = [], gval = [])
    x_k = 0.0
    temp_x_k = x_k
    gval_current = compute_g2_2_close_form(x_k, x_k, η, 0)
    push!(gradient_value_evaluation_history, (0, x_k, gval_current))
    for k in 1:(K - 1)
        d_k  = compute_d_k(k, η)
        increment = d_k[2] / 10
        for i in 1:10
            gval_current = compute_g2_2_close_form(x_k + i * increment, x_k, η, k)
            push!(gradient_value_evaluation_history, (i + (10 * (k)), x_k + i * increment, gval_current))
            # x_k = x_k + i * increment
            temp_x_k = x_k + i * increment
        end
        x_k = x_k + d_k[2]
        if x_k != temp_x_k
            print("!!!!!!!!!")
        end
    end
    x = [0, 20 , 40, 60, 80, 100, 120, 140, 160]
    index = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    plot!(gradient_value_evaluation_history.k,
        gradient_value_evaluation_history.gval,
        title = "Gradient Value",
        label="g-val",
        ylabel="gval",
        xlabel="k",
        xticks = (x, index),
        yticks = -1:0.1:0,
        #xticks = (0:K, string.(0:K))
        legend=:bottomright
    )

    png("g2_gradient_value_example_paper_plot_contineous")
    return gradient_value_evaluation_history
end

function ploth2_contineous(η::Float64, K::Int64)
    hessian_value_evaluation_history= DataFrame(k = [], x_k = [], hval = [])
    x_k = 0.0
    temp_x_k = x_k
    hval_current = compute_h2_2_close_form(x_k, x_k, η, 0)
    push!(hessian_value_evaluation_history, (0, x_k, hval_current))
    for k in 0:(K - 1)
        d_k  = compute_d_k(k, η)
        increment = d_k[2] / 10
        for i in 1:10
            hval_current = compute_h2_2_close_form(x_k + i * increment, x_k, η, k)
            push!(hessian_value_evaluation_history, (i + (10 * (k)), x_k + i * increment, hval_current))
            # x_k = x_k + i * increment
            temp_x_k = x_k + i * increment
        end
        x_k = x_k + d_k[2]
        if x_k != temp_x_k
            print("!!!!!!!!!")
        end
    end
    x = [0, 20 , 40, 60, 80, 100, 120, 140, 160]
    index = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    plot!(hessian_value_evaluation_history.k,
        hessian_value_evaluation_history.hval,
        title = "Hessian Value",
        label="h-val",
        ylabel="hval",
        xlabel="k",
        xticks = (x, index),
        yticks = -0.2:0.2:1.4,
        #xticks = (0:K, string.(0:K))
        legend=:topright
    )

    png("h2_hessian_value_example_paper_plot_contineous")
    return hessian_value_evaluation_history
end

function plotj2_contineous(η::Float64, K::Int64)
    f2_0 = (π ^ 2) / 6
    jerk_value_evaluation_history= DataFrame(k = [], x_k = [], jval = [])
    x_k = 0.0
    temp_x_k = x_k
    jval_current = compute_j2_2_close_form(x_k, x_k, η, 0)
    push!(jerk_value_evaluation_history, (0, x_k, jval_current))
    for k in 0:(K - 1)
        d_k  = compute_d_k(k, η)
        increment = d_k[2] / 10
        for i in 1:10
            jval_current = compute_j2_2_close_form(x_k + i * increment, x_k, η, k)
            push!(jerk_value_evaluation_history, (i + (10 * (k)), x_k + i * increment, jval_current))
            # x_k = x_k + i * increment
            temp_x_k = x_k + i * increment
        end
        x_k = x_k + d_k[2]
        if x_k != temp_x_k
            print("!!!!!!!!!")
        end
    end
    x = [0, 20 , 40, 60, 80, 100, 120, 140, 160]
    index = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    plot!(jerk_value_evaluation_history.k,
        jerk_value_evaluation_history.jval,
        title = "Jerk Value",
        label="j-val",
        ylabel="jval",
        xlabel="k",
        xticks = (x, index),
        yticks = -3:1:4,
        #xticks_spacing = 2
        legend=:bottomright,
    )

    png("j2_jerk_value_example_paper_plot_contineous")
    return jerk_value_evaluation_history
end

#println(plotf1(η, α_k, K))
#println(plotf2(η, K))
#println(plotf2comma2(η, K))

# print(plotf1_contineous(η, α_k, K))
# print(plotg1_contineous(η, α_k, K))
print(ploth1_contineous(η, α_k, K))
# print(plotj1_contineous(η, α_k, K))

# print(plotf2_contineous(η, K))
# println(plotg2_contineous(η, K))
# println(ploth2_contineous(η, K))
# println(plotj2_contineous(η, K))
