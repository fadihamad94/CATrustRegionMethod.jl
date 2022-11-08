using NLPModels, LinearAlgebra, DataFrames, SparseArrays
export compute_gradient2_at_kth_iterate, coumpute_f1_next_iterate_value, coumpute_f2_2_next_iterate_value, compute_f2_close_form, compute_hessian2_at_kth_iterate

x_0 = Vector{Float64}([0, 0])
η = 1e-4
τ = 4e-4
α_k = 1.0

f1_0 = 25000.3
f2_0 = π ^ 2 / 6

#Gradient Descent Section
#f_k = f ^ {(1)} (x_k) (2.4)
#f_0 = f ^ {(1)} (x_0) is given (2.8) (value = 25000.3)
#f_{k+1} = f ^ {(1)} (x_{k + 1}) = f_k - α_k (1 - 0.5 * α_k) * (1 \ (k + 1)) ^ (1 + 2 * η) (2.8)
#f1_k(x) = f ^ {(1)} (x)  = p_k(x - x_k) + f_{k+1} (2.11) for x ∈ [x_k, x_{k + 1}]

#η = τ / (4 - 2 * τ) (2.10)
function compute_η(τ::Float64)
    η = τ / (4 - 2 * τ)
    return η
end

#def Ϛ(t)
function compute_rienmann_function_value(t::Float64, K::Int64)
    rienmann_sum = sum((1 / (k + 1)) ^ t for k in 1:K)
    return rienmann_sum
end

#f ^ {(1)} (x_0) = f_0 is given (2.8) (value = 25000.3)
function compute_f1_0_value(η::Float64, K::Int64)
    f1_0 = 0.5 * (compute_rienmann_function_value(1 + 2 * η, K))
    return f1_0
end

function compute_gradient1_at_kth_iterate(η::Float64, k::Int64)
    g_k = - (1 / (k + 1)) ^ (0.5 + η)
    return Vector{Float64}([g_k])
end

#f ^ {(1)} (x_k) = f_k (2.4)
#f ^ {(1)} (x_{k + 1}) = f_{k+1} = f_k - α_k (1 - 0.5 * α_k) * (1 \ (k + 1)) ^ (1 + 2 * η) (2.8)
function coumpute_f1_next_iterate_value(f1_k::Float64, η::Float64, k::Int64, α_k::Float64)
    f1_k_plus_1 = f1_k - α_k * (1 - 0.5 * α_k) * (1 / (k + 1)) ^ (1 + 2 * η)
    return f1_k_plus_1
end

function coumpute_f1_next_iterate_value_k(f1_comma_0::Float64, η::Float64, k::Int64, α_k::Float64)
    f1_comma_k = f1_comma_0
    f1_comma_kplus1 = 0
    for i in 0:k
        f1_comma_kplus1 = coumpute_f1_next_iterate_value(f1_comma_k, η, i, α_k)
        f1_comma_k = f1_comma_kplus1
    end
    return f1_comma_kplus1
end

#p_k(t) = c_0,k + c_1,k * t + c_2,k * t ^ 2 + c_3,k * t ^ 3 + c_4,k * t ^ 4 + c_5,k * t ^ 5 (2.12)
function compute_p_k(t::Float64, η::Float64, k::Int64, α_k::Float64)
    c_0_k = α_k * (1 - 0.5 * α_k) * (1 / (k + 1)) ^ (1 + 2 * η)
    c_1_k = - (1 / (k + 1)) ^ (0.5 + η)
    c_2_k = 0.5
    s_k = α_k * (1 / (k + 1)) ^ (0.5 + η)
    ψ_k = ((k + 1) / (k + 2)) ^ (0.5 + η)
    ϕ_k = (1 / α_k) * (1 - α_k - ψ_k)
    c_3_k = -4 * (ϕ_k / s_k)
    c_4_k = 7 * (ϕ_k / s_k ^ 2)
    c_5_k = -3 * (ϕ_k / s_k ^ 3)
    p_k = c_0_k + c_1_k * t + c_2_k * t ^ 2 + c_3_k * t ^ 3 + c_4_k * t ^ 4 + c_5_k * t ^ 5
    return p_k
end

#f ^ {(1)}(x) = p_k(x - x_k) + f_{k + 1} (2.11)
function compute_f1_close_form(x::Float64, x_k::Float64, η::Float64, k::Int64, α_k::Float64, f1_k_plus_1::Float64)
    p_k = compute_p_k(x - x_k, η, k, α_k)
    return p_k + f1_k_plus_1
end

function compute_f1_close_form_k(x::Float64, x_k::Float64, η::Float64, k::Int64, α_k::Float64, f1_comma_0::Float64)
    f1_k_plus_1 = coumpute_f1_next_iterate_value_k(f1_comma_0, η, k, α_k)
    p_k = compute_p_k(x - x_k, η, k, α_k)
    return p_k + f1_k_plus_1, f1_k_plus_1
end

function compute_p_prime_k(t::Float64, η::Float64, k::Int64, α_k::Float64)
    c_0_k = α_k * (1 - 0.5 * α_k) * (1 / (k + 1)) ^ (1 + 2 * η)
    c_1_k = - (1 / (k + 1)) ^ (0.5 + η)
    c_2_k = 0.5
    s_k = α_k * (1 / (k + 1)) ^ (0.5 + η)
    ψ_k = ((k + 1) / (k + 2)) ^ (0.5 + η)
    ϕ_k = (1 / α_k) * (1 - α_k - ψ_k)
    c_3_k = -4 * (ϕ_k / s_k)
    c_4_k = 7 * (ϕ_k / s_k ^ 2)
    c_5_k = -3 * (ϕ_k / s_k ^ 3)
    p_prime_k = c_1_k + 2 * c_2_k * t + 3 * c_3_k * t ^ 2 + 4 * c_4_k * t ^ 3 + 5 * c_5_k * t ^ 4
    return Vector{Float64}([p_prime_k])
end

function compute_p_double_prime_k(t::Float64, η::Float64, k::Int64, α_k::Float64)
    c_0_k = α_k * (1 - 0.5 * α_k) * (1 / (k + 1)) ^ (1 + 2 * η)
    c_1_k = - (1 / (k + 1)) ^ (0.5 + η)
    c_2_k = 0.5
    s_k = α_k * (1 / (k + 1)) ^ (0.5 + η)
    ψ_k = ((k + 1) / (k + 2)) ^ (0.5 + η)
    ϕ_k = (1 / α_k) * (1 - α_k - ψ_k)
    c_3_k = -4 * (ϕ_k / s_k)
    c_4_k = 7 * (ϕ_k / s_k ^ 2)
    c_5_k = -3 * (ϕ_k / s_k ^ 3)
    p_double_prime_k = 2 * c_2_k + 6 * c_3_k * t + 12 * c_4_k * t ^ 2 + 20 * c_5_k * t ^ 3
    return Vector{Float64}([p_double_prime_k])
end

function compute_p_triple_prime_k(t::Float64, η::Float64, k::Int64, α_k::Float64)
    c_0_k = α_k * (1 - 0.5 * α_k) * (1 / (k + 1)) ^ (1 + 2 * η)
    c_1_k = - (1 / (k + 1)) ^ (0.5 + η)
    c_2_k = 0.5
    s_k = α_k * (1 / (k + 1)) ^ (0.5 + η)
    ψ_k = ((k + 1) / (k + 2)) ^ (0.5 + η)
    ϕ_k = (1 / α_k) * (1 - α_k - ψ_k)
    c_3_k = -4 * (ϕ_k / s_k)
    c_4_k = 7 * (ϕ_k / s_k ^ 2)
    c_5_k = -3 * (ϕ_k / s_k ^ 3)
    p_double_prime_k = 6 * c_3_k + 24 * c_4_k * t + 60 * c_5_k * t ^ 2
    return Vector{Float64}([p_double_prime_k])
end

function compute_g1_close_form(x::Float64, x_k::Float64, η::Float64, k::Int64, α_k::Float64)
    gradient_at_f1_k_plus_1 = compute_gradient1_at_kth_iterate(η, k + 1)
    gradient_at_p_k = compute_p_prime_k(x - x_k, η, k, α_k)
    # return gradient_at_p_k + gradient_at_f1_k_plus_1
    return gradient_at_p_k
end


function compute_H1_close_form(x::Float64, x_k::Float64, η::Float64, k::Int64, α_k::Float64)
    hessian_at_f1_k_plus_1 = Vector{Float64}([1.0])
    hessian_at_p_k = compute_p_double_prime_k(x - x_k, η, k, α_k)
    # return hessian_at_f1_k_plus_1 + hessian_at_p_k
    return hessian_at_p_k
end

function compute_j1_close_form(x::Float64, x_k::Float64, η::Float64, k::Int64, α_k::Float64)
    jerk_at_p_k = compute_p_triple_prime_k(x - x_k, η, k, α_k)
    return jerk_at_p_k
end

#Newton's method section
#f_k = f ^{(2)} (x_k) (3.1)
#f_k for k = 0 is given (3.1) (value = π ^ 2 \ 6)
#f_k+1 = f_k - 0.5 * ((1 / (k + 1)) ^ (1 + 2 * η) + (1 / (k + 1)) ^ 2) (3.1)

#d_k_1 = (1 / (k + 1)) ^ (0.5 + η) (3.3)
function compute_d_k(k::Int64, η::Float64)
    d_k_1 = (1 / (k + 1)) ^ (0.5 + η)
    d_k_2 = 1
    d_k = Vector{Float64}([d_k_1, d_k_2])
    return d_k
end

#f ^ {(2)} (x_0) = f_0 is given (3.1) (value = π ^ 2 \ 6)
function compute_f2_0_value(η::Float64, K::Int64)
    f2_0 = 0.5 * (compute_rienmann_function_value(1 + 2 * η, K) + compute_rienmann_function_value(2.0, K))
    return f2_0
end

#f_k+1 = f_k - 0.5 * ((1 / (k + 1)) ^ (1 + 2 * η) + (1 / (k + 1)) ^ 2) (3.1)
function compute_actual_f2_value_reduction(k::Int64, η::Float64)
    actual_reduction = 0.5 * ((1 / (k + 1)) ^ (1 + 2 * η) + (1 / (k + 1)) ^ 2)
    return actual_reduction
end

#f ^ {(2)} (x_k) = f_k (3.1)
#f ^ {(2)} (x_{k + 1}) = f_{k+1} = f_k - 0.5 * ((1 / (k + 1)) ^ (1 + 2 * η) + (1 / (k + 1)) ^ 2) (3.1)
function coumpute_f2_next_iterate_value(f2_k::Float64, η::Float64, k::Int64)
    actual_reduction = compute_actual_f2_value_reduction(k, η)
    f2_k_plus_1 = f2_k - actual_reduction
    return f2_k_plus_1
end

#g_k (3.4)
function compute_gradient2_at_kth_iterate(k::Int64, η::Float64)
    g_k_1 = (1 / (k + 1)) ^ (0.5 + η)
    g_k_2 = (1 / (k + 1)) ^ 2
    g_k = -Vector{Float64}([g_k_1, g_k_2])
    return g_k
end

#H_k (3.4)
function compute_hessian2_at_kth_iterate(k::Int64)
    H = SparseArrays.sparse([[1.0 0.0]; [0.0 (1.0 / (k + 1.0)) ^ 2]])
    return H
end

#m_k
function computeSecondOrderModel(f::Float64, g::Vector{Float64}, H::SparseMatrixCSC{Float64,Int64}, d_k::Vector{Float64})
    #the way the function is constructed here is that the next function value equals to the model always
    return f + transpose(g) * d_k + 0.5 * transpose(d_k) * H * d_k
end

#f^{(2)}_{1, 0} = f_{1, 0} (3.9) (2.8)(given value 2500.3)
function compute_f2_1comma0_value(η::Float64, K::Int64)
    return compute_f1_0_value(η, K)
end

#f^{(2)}_{1, k + 1} = f_{1, k + 1} = f_{1, k} - 0.5 * (1/ (k + 1)) ^ (1 + 2 * η) (3.9) (2.8)
function coumpute_f2_1_next_iterate_value(f2_1_comma_k::Float64, η::Float64, k::Int64, α_k::Float64)
    return coumpute_f1_next_iterate_value(f2_1_comma_k, η, k, α_k)
end

function coumpute_f2_1_next_iterate_value_k(f2_1_comma_0::Float64, η::Float64, k::Int64, α_k::Float64)
     return coumpute_f1_next_iterate_value_k(f2_1_comma_0, η, k, α_k)
end

#f^{(2)}_{2, 0} = f_{2, 0} (3.10) (given value π ^ 2 \ 6)
function compute_f2_2comma0_value(K::Int64)
    f2_1comma0 = 0.5 * compute_rienmann_function_value(2.0, K)
    return f2_1comma0
end

#f^{(2)}_{1, k + 1} = f_{1, k + 1} = f_{1, k} - 0.5 * (1/ (k + 1)) ^ 2 (3.10)
function coumpute_f2_2_next_iterate_value(f2_2_comma_k::Float64, η::Float64, k::Int64)
    f2_2_comma_kplus1 = f2_2_comma_k - 0.5 * (1/ (k + 1)) ^ 2
    return f2_2_comma_kplus1
end

function coumpute_f2_2_next_iterate_value_k(f2_2_comma_0::Float64, η::Float64, k::Int64)
    f2_2_comma_k = f2_2_comma_0
    f2_2_comma_kplus1 = 0
    for i in 0:k
        f2_2_comma_kplus1 = coumpute_f2_2_next_iterate_value(f2_2_comma_k, η, i)
        f2_2_comma_k = f2_2_comma_kplus1
    end
    return f2_2_comma_kplus1
end

#q_k(t) = d_0,k + d_1,k * t + d_2,k * t ^ 2 + d_3,k * t ^ 3 + d_4,k * t ^ 4 + d_5,k * t ^ 5 (3.10)
function compute_q_k(t::Float64, η::Float64, k::Int64)
    d_0_k = 0.5 * (1 / (k + 1)) ^ 2
    d_1_k = - (1 / (k + 1)) ^ 2
    d_2_k = 0.5 * (1 / (k + 1)) ^ 2
    d_3_k = 0.5 * (9 * (1 / (k + 2)) ^ 2 - (1 / (k + 1)) ^ 2)
    d_4_k = 0.5 * (-16 * (1 / (k + 2)) ^ 2 + 2 * (1 / (k + 1)) ^ 2)
    d_5_k = 0.5 * (7 * (1 / (k + 2)) ^ 2 - (1 / (k + 1)) ^ 2)
    q_k = d_0_k + d_1_k * t + d_2_k * t ^ 2 + d_3_k * t ^ 3 + d_4_k * t ^ 4 + d_5_k * t ^ 5
    return q_k
end

#f ^ {(2,2)}(x) = q_k(x - x_k) + f_{2, k + 1} (3.10)
function compute_f2_2coma_close_form(x::Float64, x_k::Float64, η::Float64, k::Int64, f2_2_comma_k_plus_1::Float64)
    q_k = compute_q_k(x - x_k, η, k)
    return q_k + f2_2_comma_k_plus_1
end

function compute_f2_2coma_close_form_k(x::Float64, x_k::Float64, η::Float64, k::Int64, f2_2_comma_0::Float64)
    f2_2_comma_k_plus_1 = coumpute_f2_2_next_iterate_value_k(f2_2_comma_0, η, k)
    q_k = compute_q_k(x - x_k, η, k)
    return q_k + f2_2_comma_k_plus_1, f2_2_comma_k_plus_1
end

#f ^ {(2)}(x) = f ^ {(2,1)}([x]_1) + f ^ {(2,2)}([x]_2) = f ^ {(1)}([x]_1) + f ^ {(2,2)}([x]_2)
#f ^ {(2)}(x) = p_k([x - x_k]_1) + f1_{k + 1} +  q_k([x - x_k]_2) + f2_{2, k + 1} (2.11) (3.10)
function compute_f2_close_form(x::Vector{Float64}, x_k::Vector{Float64}, η::Float64, k::Int64, α_k::Float64, f1_k_plus_1::Float64, f2_2_comma_k_plus_1::Float64)
    f2comma1 = compute_f1_close_form(x[1], x_k[1], η, k, α_k, f1_k_plus_1)
    f2comma2 = compute_f2_2coma_close_form(x[2], x_k[2], η, k, f2_2_comma_k_plus_1)
    return f2comma1 + f2comma2
end

function compute_f2_close_form_modified(x::Vector{Float64}, x_k::Vector{Float64}, η::Float64, k::Int64, α_k::Float64, f1_k_plus_1::Float64, f2_2_comma_k_plus_1::Float64)
    f2comma1 = compute_f1_close_form(x[1], x_k[1], η, k, α_k, f1_k_plus_1)
    f2comma2 = compute_f2_2coma_close_form(x[2], x_k[2], η, k, f2_2_comma_k_plus_1)
    return f2comma1, f2comma2, f2comma1 + f2comma2
end

function compute_f2_close_form(x::Vector{Float64}, x_k::Vector{Float64}, η::Float64, k_1::Int64, k_2::Int64, α_k::Float64, f1_k_plus_1::Float64, f2_2_comma_k_plus_1::Float64)
    f2comma1 = compute_f1_close_form(x[1], x_k[1], η, k_1, α_k, f1_k_plus_1)
    f2comma2 = compute_f2_2coma_close_form(x[2], x_k[2], η, k_2, f2_2_comma_k_plus_1)
    return f2comma1 + f2comma2
end

function compute_f2_close_form_modified(x::Vector{Float64}, x_k::Vector{Float64}, η::Float64, k_1::Int64, k_2::Int64, α_k::Float64, f1_k_plus_1::Float64, f2_2_comma_k_plus_1::Float64)
    f2comma1 = compute_f1_close_form(x[1], x_k[1], η, k_1, α_k, f1_k_plus_1)
    f2comma2 = compute_f2_2coma_close_form(x[2], x_k[2], η, k_2, f2_2_comma_k_plus_1)
    return f2comma1, f2comma2, f2comma1 + f2comma2
end

function compute_f2_close_form_k(x::Vector{Float64}, x_k::Vector{Float64}, η::Float64, k::Int64, α_k::Float64, f1_comma_0::Float64, f2_2_comma_0::Float64)
    f2comma1 = compute_f1_close_form_k(x[1], x_k[1], η, k, α_k, f1_comma_0)
    f2comma2 = compute_f2_2coma_close_form_k(x[2], x_k[2], η, k, f2_2_comma_0)
    return f2comma1 + f2comma2
end

function compute_q_prime_k(t::Float64, η::Float64, k::Int64)
    d_0_k = 0.5 * (1 / (k + 1)) ^ 2
    d_1_k = - (1 / (k + 1)) ^ 2
    d_2_k = 0.5 * (1 / (k + 1)) ^ 2
    d_3_k = 0.5 * (9 * (1 / (k + 2)) ^ 2 - (1 / (k + 1)) ^ 2)
    d_4_k = 0.5 * (-16 * (1 / (k + 2)) ^ 2 + 2 * (1 / (k + 1)) ^ 2)
    d_5_k = 0.5 * (7 * (1 / (k + 2)) ^ 2 - (1 / (k + 1)) ^ 2)
    q_prime_k = d_1_k + 2 * d_2_k * t ^ 1 + 3 * d_3_k * t ^ 2 + 4 * d_4_k * t ^ 3 + 5 * d_5_k * t ^ 4
    return Vector{Float64}([q_prime_k])
end

function compute_q_double_prime_k(t::Float64, η::Float64, k::Int64)
    d_0_k = 0.5 * (1 / (k + 1)) ^ 2
    d_1_k = - (1 / (k + 1)) ^ 2
    d_2_k = 0.5 * (1 / (k + 1)) ^ 2
    d_3_k = 0.5 * (9 * (1 / (k + 2)) ^ 2 - (1 / (k + 1)) ^ 2)
    d_4_k = 0.5 * (-16 * (1 / (k + 2)) ^ 2 + 2 * (1 / (k + 1)) ^ 2)
    d_5_k = 0.5 * (7 * (1 / (k + 2)) ^ 2 - (1 / (k + 1)) ^ 2)
    q_double_prime_k = 2 * d_2_k + 6 * d_3_k * t + 12 * d_4_k * t ^ 2 + 20 * d_5_k * t ^ 3
    return Vector{Float64}([q_double_prime_k])
end

function compute_q_triple_prime_k(t::Float64, η::Float64, k::Int64)
    d_0_k = 0.5 * (1 / (k + 1)) ^ 2
    d_1_k = - (1 / (k + 1)) ^ 2
    d_2_k = 0.5 * (1 / (k + 1)) ^ 2
    d_3_k = 0.5 * (9 * (1 / (k + 2)) ^ 2 - (1 / (k + 1)) ^ 2)
    d_4_k = 0.5 * (-16 * (1 / (k + 2)) ^ 2 + 2 * (1 / (k + 1)) ^ 2)
    d_5_k = 0.5 * (7 * (1 / (k + 2)) ^ 2 - (1 / (k + 1)) ^ 2)
    q_double_prime_k = 6 * d_3_k + 24 * d_4_k * t + 60 * d_5_k * t ^ 2
    return Vector{Float64}([q_double_prime_k])
end

function compute_g2_2_close_form(x, x_k, η::Float64, k::Int64)
    #gradient f2_2
    gradient_at_f2_k_plus_1 = compute_gradient2_at_kth_iterate(k + 1, η)
    gradient_at_q_k = compute_q_prime_k(x - x_k, η, k)
    g_2_2 = gradient_at_q_k[1]
    return g_2_2
end

function compute_h2_2_close_form(x, x_k, η::Float64, k::Int64)
    #hessian f2_2
    hessian_at_f2_k_plus_1 = compute_hessian2_at_kth_iterate(k + 1)
    hessian_at_q_k = compute_q_double_prime_k(x - x_k, η, k)
    h_2_2 = hessian_at_q_k[1]
    return h_2_2
end

function compute_j2_2_close_form(x, x_k, η::Float64, k::Int64)
    #hessian f2_2
    jerk_at_q_k = compute_q_triple_prime_k(x - x_k, η, k)
    j_2_2 = jerk_at_q_k[1]
    return j_2_2
end

function compute_g2_close_form(x::Vector{Float64}, x_k::Vector{Float64}, η::Float64, k::Int64, α_k::Float64)
    #gradient f2_1
    gradient_f2_1 = compute_g1_close_form(x[1], x_k[1], η, k, α_k)
    #gradient f2_2
    gradient_f2_2 = compute_g2_2_close_form(x[2], x_k[2], η, k)

    g_2_1 = gradient_f2_1[1]
    g_2_2 = gradient_f2_2

    return  Vector{Float64}([g_2_1, g_2_2])
end

function compute_g2_close_form(x::Vector{Float64}, x_k::Vector{Float64}, η::Float64, k_1::Int64, k_2::Int64, α_k::Float64)
    #gradient f2_1
    gradient_f2_1 = compute_g1_close_form(x[1], x_k[1], η, k_1, α_k)
    #gradient f2_2
    gradient_f2_2 = compute_g2_2_close_form(x[2], x_k[2], η, k_2)

    g_2_1 = gradient_f2_1[1]
    g_2_2 = gradient_f2_2

    return  Vector{Float64}([g_2_1, g_2_2])
end

function compute_H2_close_form(x::Vector{Float64}, x_k::Vector{Float64}, η::Float64, k_1::Int64, k_2::Int64, α_k::Float64)
    #hessian f2_1
    hessian_f2_1 = compute_H1_close_form(x[1], x_k[1], η, k_1, α_k)
    #hessian f2_2
    hessian_f2_2 = compute_h2_2_close_form(x[2], x_k[2], η, k_2)
    #hessian f2
    H_2 = SparseArrays.sparse([[hessian_f2_1[1] 0.0]; [0.0 hessian_f2_2]])
    return H_2
end
