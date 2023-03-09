export phi, findinterval, bisection, restoreFullMatrix
using LinearAlgebra
#=
The big picture idea here is to optimize the trust region subproblem using a factorization method based
on the optimality conditions:
H d_k + g + δ d_k = 0
H + δ I ≥ 0
δ(r -  ||d_k ||) = 0

That is why we defined the below phi to solve that using bisection logic.
=#

const OPTIMIZATION_METHOD_TRS = "GALAHAD_TRS"
const OPTIMIZATION_METHOD_GLTR = "GALAHAD_GLTR"
const OPTIMIZATION_METHOD_DEFAULT = "OUR_APPROACH"

const LIBRARY_PATH_TRS = string(@__DIR__ ,"/../lib/trs.so")
const LIBRARY_PATH_GLTR = string(@__DIR__ ,"/../lib/gltr.so")

mutable struct Subproblem_Solver_Methods
    OPTIMIZATION_METHOD_TRS::String
    OPTIMIZATION_METHOD_GLTR::String
    OPTIMIZATION_METHOD_DEFAULT::String
    function Subproblem_Solver_Methods()
        return new(OPTIMIZATION_METHOD_TRS, OPTIMIZATION_METHOD_GLTR, OPTIMIZATION_METHOD_DEFAULT)
    end
end

const subproblem_solver_methods = Subproblem_Solver_Methods()

#Data returned by calling the GALAHAD library in case we solve trust region subproblem
#using their factorization approach
struct userdata_type_trs
	status::Cint
	factorizations::Cint
	obj::Cdouble
	solution::Ptr{Cdouble}
	hard_case::Cuchar
	multiplier::Cdouble
	x_norm::Cdouble
end

#Data returned by calling the GALAHAD library in case we solve trust region subproblem
#using their GLTR approach
struct userdata_type_gltr
	status::Cint
	iter::Cint
	obj::Cdouble
	hard_case::Cuchar
	multiplier::Cdouble
	mnormx::Cdouble
end

function getHessianLowerTriangularPart(H)
	h_vec = Vector{Float64}()
	for i in 1:size(H)[1]
		for j in 1:i
			push!(h_vec, H[i, j])
		end
	end
	return h_vec
end

function solveTrustRegionSubproblem(f::Float64, g::Vector{Float64}, H, x_k::Vector{Float64}, δ::Float64, ϵ::Float64, r::Float64, min_grad::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
	if subproblem_solver_method == OPTIMIZATION_METHOD_DEFAULT
		return optimizeSecondOrderModel(g, H, δ, ϵ, r)
	end

	if subproblem_solver_method == OPTIMIZATION_METHOD_TRS
		return trs(f, g, H, δ, ϵ, r)
	end

	if subproblem_solver_method == OPTIMIZATION_METHOD_GLTR
		return gltr(f, g, H, r, min_grad)
	end

	return optimizeSecondOrderModel(g, H, δ, ϵ, r)
end

function trs(f::Float64, g::Vector{Float64}, H, δ::Float64, ϵ::Float64, r::Float64)
	print_level = 0
    max_factorizations = 1000
	H_dense = getHessianLowerTriangularPart(H)
	d = zeros(length(g))
	full_Path = string(@__DIR__ ,"/test")
	userdata = ccall((:trs, LIBRARY_PATH_TRS), userdata_type_trs, (Cint, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Cdouble, Cint, Cint), length(g), f, d, g, H_dense, r, print_level, max_factorizations)
	if userdata.status != 0
		#=try
			#@show norm(H, 2) * norm(inv(H), 2)
			#inv(H)
			#@show H
		catch e
			@show H
			#@show e
			@show "H is invertible"
		end=#
		#throw(error("Failed to solve trust region subproblem using TRS factorization method from GALAHAD. Status is $(userdata.status)."))
		return optimizeSecondOrderModel(g, H, δ, ϵ, r)
	end
	return userdata.multiplier, d, userdata.factorizations
end

function gltr(f::Float64, g::Vector{Float64}, H, r::Float64, min_grad::Float64)
	#@show "---------------------------------------"
	#@show "---------------------------------------"
	print_level = 0
    	iter = 10000
	H_dense = getHessianLowerTriangularPart(H)
	d = zeros(length(g))
	stop_relative = 1.5e-8
	#stop_relative = 1.5e-7
	#=if (1e-5 * min_grad) <= 1e-6
		@show "**************************************************************************************************+++++++++++++++++++++++++++++++++++++++"
	end=#
	stop_relative = min(1e-6 * min_grad, 1e-6)
	stop_absolute = 0.0
	#stop_relative = 1e-10 * min_grad
	#stop_relative *= min_grad
	stop_absolute = -1.0
	#stop_relative = -1.0
	steihaug_toint = false
	userdata = ccall((:gltr, LIBRARY_PATH_GLTR), userdata_type_gltr, (Cint, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Cdouble, Cint, Cint, Cdouble, Cdouble, Cuchar), length(g), f, d, g, H_dense, r, print_level, iter, stop_relative, stop_absolute, steihaug_toint)
	if userdata.status < 0
		@show "ENTERED HERE TRYING AGAIN"			
		steihaug_toint = true
		stop_relative = min(0.1 * min_grad, 0.1)
		d = zeros(length(g))
		userdata = ccall((:gltr, LIBRARY_PATH_GLTR), userdata_type_gltr, (Cint, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Cdouble, Cint, Cint, Cdouble, Cdouble, Cuchar), length(g), f, d, g, H_dense, r, print_level, iter, stop_relative, stop_absolute, steihaug_toint)
	end
	if userdata.status != 0
		throw(error("Failed to solve trust region subproblem using GLTR iterative method from GALAHAD. Status is $(userdata.status)."))
	end
	#=if norm(H * d + userdata.multiplier * I * d + g, 2) > 1e-1
		@show norm(H * d + userdata.multiplier * I * d + g, 2)
	end=#
	#=if norm(H * d + userdata.multiplier * I * d + g, 2) - max(stop_relative * norm(g), stop_absolute) > 1e-1
		@show norm(H * d + userdata.multiplier * I * d + g, 2)
		@show max(stop_relative * norm(g), stop_absolute)
		@show norm(d) - r		
		@show userdata.iter
	end
	@assert(norm(H * d + userdata.multiplier * I * d + g, 2) - max(stop_relative * norm(g), stop_absolute) <= 1e-1)=#
	#@show "---------------------------------------"
	#@show userdata.iter
	#@show userdata.multiplier
	#@show norm(d, 2)
	#@show norm(H * d + userdata.multiplier * I * d + g, 2)
	#@show "---------------------------------------"
	return userdata.multiplier, d, userdata.iter
end

#Based on Theorem 4.3 in Numerical Optimization by Wright
function optimizeSecondOrderModel(g::Vector{Float64}, H, δ::Float64, ϵ::Float64, r::Float64)
    #When δ is 0 and the Hessian is positive semidefinite, we can directly compute the direction
    total_number_factorizations = 0
    try
	total_number_factorizations += 1
        cholesky(Matrix(H))
        d_k = H \ (-g)
        if norm(d_k, 2) <= r
            return 0.0, d_k, total_number_factorizations
        end
    catch e
		#Do nothing
    end

    try
	δ, δ_prime, temp_total_number_factorizations = findinterval(g, H, δ, ϵ, r)
	total_number_factorizations += temp_total_number_factorizations
        δ_m, temp_total_number_factorizations = bisection(g, H, δ, ϵ, δ_prime, r)
	total_number_factorizations += temp_total_number_factorizations
        sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
	total_number_factorizations  += 1
        d_k = (cholesky(H + δ_m * sparse_identity) \ (-g))
        return δ_m, d_k, total_number_factorizations
    catch e
        if e == ErrorException("Bisection logic failed to find a root for the phi function")
	    	δ, d_k = solveHardCaseLogic(g, H, r)
            return δ, d_k, total_number_factorizations
        elseif e == ErrorException("Bisection logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0.")
            δ, d_k = solveHardCaseLogic(g, H, r)
	    	return δ, d_k, total_number_factorizations
        else
            throw(e)
        end
    end
end

function phi(g::Vector{Float64}, H, δ::Float64, ϵ::Float64, r::Float64)
    sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
    shifted_hessian = H + δ * sparse_identity
    #cholesky factorization only works on positive definite matrices
    try
        cholesky(shifted_hessian)
        computed_norm = norm(shifted_hessian \ g, 2)
	if computed_norm < ϵ * r
        return 1
	elseif computed_norm <= r
        return 0
    else
        return -1
    end
    catch e
        return -1
    end
end

function findinterval(g::Vector{Float64}, H, δ::Float64, ϵ::Float64, r::Float64)
    Φ_δ = phi(g, H, 0.0, ϵ, r)

    if Φ_δ == 0
        δ = 0.0
        δ_prime = 0.0
        return δ, δ_prime, 1
    end

    Φ_δ = phi(g, H, δ, ϵ, r)

    if Φ_δ == 0
        δ_prime = δ
        return δ, δ_prime, 2
    end

    δ_prime = δ == 0.0 ? 1.0 : δ * 2
    Φ_δ_prime = 0.0

    k = 1
    while k < 100
        Φ_δ_prime = phi(g, H, δ_prime, ϵ, r)
        if Φ_δ_prime == 0
            δ = δ_prime
            return δ, δ_prime, k + 2
        end

        if ((Φ_δ * Φ_δ_prime) < 0)
            break
        end
        if Φ_δ_prime > 0
            δ_prime = δ_prime / 2
        elseif Φ_δ_prime < 0
            δ_prime = δ_prime * 2
        end
        k = k + 1
    end

    #switch so that δ for ϕ_δ >= 0 and δ_prime for ϕ_δ_prime <= 0
    if Φ_δ_prime > 0 && Φ_δ < 0
        δ_temp = δ
        Φ_δ_temp = Φ_δ
        δ = δ_prime
        δ_prime = δ_temp
        Φ_δ = Φ_δ_prime
        Φ_δ_prime = Φ_δ_temp
    end

    if (Φ_δ  * Φ_δ_prime > 0)
        throw(error("Bisection logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0."))
    end
    return δ, δ_prime, min(k, 100) + 2
end

function bisection(g::Vector{Float64}, H, δ::Float64, ϵ::Float64, δ_prime::Float64, r::Float64)
    # the input of the function is the two end of the interval (δ,δ_prime)
    # our goal here is to find the approximate δ using classic bisection method

    #Bisection logic
    k = 1
    δ_m = (δ + δ_prime) / 2
    Φ_δ_m = phi(g, H, δ_m, ϵ, r)

    while (Φ_δ_m != 0) && k <= 100
        if Φ_δ_m > 0
            δ = δ_m
        else
            δ_prime = δ_m
        end
        δ_m = (δ + δ_prime) / 2
        Φ_δ_m = phi(g, H, δ_m, ϵ, r)
        k = k + 1
    end

    if (Φ_δ_m != 0)
        throw(error("Bisection logic failed to find a root for the phi function"))
    end
    return δ_m, min(k, 100) + 1
end

#Based on 'THE HARD CASE' section from Numerical Optimization by Wright
function solveHardCaseLogic(g::Vector{Float64}, H, r::Float64)
    minimumEigenValue = eigmin(Matrix(H))
    δ = -minimumEigenValue
    z =  eigvecs(Matrix(H))[:,1]
    Q = eigvecs(Matrix(H))
    eigenvaluesVector = eigvals(Matrix(H))

    D = zeros(size(Q))
    for i in 1:size(D)[1]
        D[i, i] = eigenvaluesVector[i]
    end

    norm_d_k_squared_without_τ_squared = 0.0

    for i in 1:length(eigenvaluesVector)
        if eigenvaluesVector[i] != minimumEigenValue
            norm_d_k_squared_without_τ_squared = norm_d_k_squared_without_τ_squared + ((Q[:, i]' * g) ^ 2 / (eigenvaluesVector[i] + δ) ^ 2)
        end
    end

    norm_d_k_squared = r ^ 2
    τ = sqrt(norm_d_k_squared - norm_d_k_squared_without_τ_squared)
    d_k = τ .* z

    for i in 1:length(eigenvaluesVector)
        if eigenvaluesVector[i] != minimumEigenValue
            d_k = d_k .- ((Q[:, i]' * g) / (eigenvaluesVector[i] + δ)) * Q[:, i]
        end
    end

    return δ, d_k
end

function restoreFullMatrix(A)
	#Old NLPModelsJuMP package used to return only a lower traingular matrix
    # nmbRows = size(A)[1]
    # numbColumns = size(A)[2]
    # for i in 1:nmbRows
    #     for j in i:numbColumns
    #         A[i, j] = A[j, i]
    #     end
    # end
    return A
end
