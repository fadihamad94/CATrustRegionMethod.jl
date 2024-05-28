export phi, findinterval, bisection
using LinearAlgebra
using Dates

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

function if_mkpath(dir::String)
  if !isdir(dir)
     mkpath(dir)
  end
end

struct userdata_type_trs
	status::Cint
	factorizations::Cint
	hard_case::Cuchar
	multiplier::Cdouble
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

function getHessianDenseLowerTriangularPart(H)
	h_vec = Vector{Float64}()
	for i in 1:size(H)[1]
		for j in 1:i
			push!(h_vec, H[i, j])
		end
	end
	return h_vec
end


function getHessianSparseLowerTriangularPart(H)
	H_ne = 0
	H_val = Vector{Float64}()
	H_row = Vector{Int32}()
	H_col= Vector{Int32}()
	H_ptr = Vector{Int32}()
	temp = 0
	if H[1, 1] != 0
		push!(H_ptr, 0)
	end
	for i in 1:size(H)[1]
		for j in 1:i
			if H[i, j] != 0.0
				if temp == 0
					temp = H_ne
				end
				H_ne += 1
				push!(H_val, H[i, j])
				push!(H_row, i - 1)
				push!(H_col, j - 1)
			end
		end
		if temp != 0
			push!(H_ptr, temp)
		end
		temp = 0
	end
	push!(H_ptr, H_ne)
	return H_ne, H_val, H_row, H_col, H_ptr
end


function solveTrustRegionSubproblem(f::Float64, g::Vector{Float64}, H, x_k::Vector{Float64}, δ::Float64, γ_2::Float64, r::Float64, min_grad::Float64, problem_name::String, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT, print_level::Int64=0)
	if subproblem_solver_method == OPTIMIZATION_METHOD_DEFAULT
		return optimizeSecondOrderModel(g, H, δ, γ_2, r, min_grad, print_level)
	end

	if subproblem_solver_method == OPTIMIZATION_METHOD_TRS
		return trs(f, g, H, δ, γ_2, r, problem_name, min_grad, print_level)
	end

	if subproblem_solver_method == OPTIMIZATION_METHOD_GLTR
		return gltr(f, g, H, r, min_grad, print_level)
	end

	return optimizeSecondOrderModel(g, H, δ, γ_2, r, min_grad, print_level)
end

function trs(f::Float64, g::Vector{Float64}, H, δ::Float64, γ_2::Float64, r::Float64, problem_name::String, min_grad::Float64, print_level::Int64=0)
    max_factorizations = 1000
	H_type = "sparse_by_rows"
	#H_type = "dense"
	#H_type = "coordinate"
	H_ne = 0
	H_val = Nothing
	H_row = Nothing
	H_col = Nothing
	H_ptr = Nothing
	if H_type == "dense"
		H_val = getHessianDenseLowerTriangularPart(H)
		H_ne = length(H_val)
		H_row = [Int32(0)]
		H_col = [Int32(0)]
		H_ptr = [Int32(0)]
	else
		start_time_temp = time()
		H_ne, H_val, H_row, H_col, H_ptr = getHessianSparseLowerTriangularPart(H)
		end_time_temp = time()
		total_time_temp = end_time_temp - start_time_temp
		if print_level >= 2
			@info "getHessianSparseLowerTriangularPart operation took $total_time_temp."
			println("getHessianSparseLowerTriangularPart operation took $total_time_temp.")
		end
	end
	d = zeros(length(g))
	full_Path = string(@__DIR__ ,"/test")
	use_initial_multiplier = true
	initial_multiplier = δ
	use_stop_args = true
	stop_normal = 1e-5
    stop_hard = 1e-5
	if H_type == "sparse_by_rows" && length(H_ptr) != length(g) + 1
		@warn "Weired case detected."
		H_type = "coordinate"
	end
	# Convert the Julia string to a C-compatible representation (Cstring)
	string_problem_name = string(@__DIR__ ,"/../DEBUG_TRS/$problem_name.csv")
	if print_level >= 0
		if_mkpath(string(@__DIR__ ,"/../DEBUG_TRS"))
		if !isfile(string_problem_name)
			open(string_problem_name,"a") do iteration_status_csv_file
				write(iteration_status_csv_file, "status,hard_case,x_norm,radius,multiplier,lambda,len_history,factorizations\n");
	    	end
		end
	end

	start_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
	start_time_temp = time()
	userdata = ccall((:trs, LIBRARY_PATH_TRS), userdata_type_trs, (Cint, Cint, Cstring, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cint}, Ref{Cint}, Ref{Cint}, Cdouble, Cint, Cint, Cuchar, Cdouble, Cuchar, Cdouble, Cdouble, Cstring), length(g), H_ne, H_type, f, d, g, H_val, H_row, H_col, H_ptr, r, print_level, max_factorizations, use_initial_multiplier, initial_multiplier, use_stop_args, stop_normal, stop_hard, string_problem_name)
	end_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
	end_time_temp = time()
	total_time_temp = end_time_temp - start_time_temp
	if print_level >= 2
		@info "calling GALAHAD operation took $total_time_temp."
		println("calling GALAHAD operation took $total_time_temp.")
	end

	tol = 1e-1
	condition_success = norm(d, 2) - r <= tol || abs(norm(d, 2) - r) <= stop_normal * r + tol || abs(norm(d, 2) - r) <= stop_normal + tol
	total_number_factorizations = userdata.factorizations
	if userdata.status != 0 || !condition_success
		if print_level >= 1
			println("Failed to solve trust region subproblem using TRS factorization method from GALAHAD. Status is $(userdata.status).")
		end
		if userdata.status == 0
			norm_d = norm(d, 2)
			@warn "Solution isn't inside the trust-region. ||d_k|| = $norm_d but radius is $r."
			if print_level >= 1
				println("Solution isn't inside the trust-region. ||d_k|| = $norm_d but radius is $r.")
			end
		else
			if print_level >= 0
				@warn "Failed to solve trust region subproblem using TRS factorization method from GALAHAD. Status is $(userdata.status)."
			end
		end
		# This code was used when getting the preliminary results. Maybe we need it later
		# start_time_temp = time()
		# δ = max(δ, abs(eigmin(Matrix(H))))
		# end_time_temp = time()
		# total_time_temp = end_time_temp - start_time_temp
		# println("eigmin operation took $total_time_temp.")
		try
			start_time_temp = time()
			success, δ, d_k, temp_total_number_factorizations, hard_case = optimizeSecondOrderModel(g, H, δ, stop_normal, r, min_grad, print_level)
			total_number_factorizations += temp_total_number_factorizations
			end_time_temp = time()
			total_time_temp = end_time_temp - start_time_temp
			if print_level >= 2
				@info "$success. optimizeSecondOrderModel operation took $total_time_temp."
				println("optimizeSecondOrderModel operation took $total_time_temp.")
			end
			return success, δ, d_k, total_number_factorizations, hard_case
		catch e
			@error e
			throw(e)
		end
	end

    multiplier = userdata.multiplier
	hard_case = Bool(userdata.hard_case != 0)
    return true, multiplier, d, total_number_factorizations, hard_case
end

function gltr(f::Float64, g::Vector{Float64}, H, r::Float64, min_grad::Float64, print_level::Int64=0)
    iter = 10000
	H_dense = getHessianDenseLowerTriangularPart(H)
	d = zeros(length(g))
	stop_relative = 1.5e-8
	stop_relative = min(1e-6 * min_grad, 1e-6)
	stop_absolute = 0.0
	steihaug_toint = false
	stop_absolute = 0.0
	stop_relative = 0.0
	userdata = ccall((:gltr, LIBRARY_PATH_GLTR), userdata_type_gltr, (Cint, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Cdouble, Cint, Cint, Cdouble, Cdouble, Cuchar), length(g), f, d, g, H_dense, r, print_level, iter, stop_relative, stop_absolute, steihaug_toint)
	if userdata.status < 0
		steihaug_toint = true
		stop_relative = min(0.1 * min_grad, 0.1)
		d = zeros(length(g))
		userdata = ccall((:gltr, LIBRARY_PATH_GLTR), userdata_type_gltr, (Cint, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Cdouble, Cint, Cint, Cdouble, Cdouble, Cuchar), length(g), f, d, g, H_dense, r, print_level, iter, stop_relative, stop_absolute, steihaug_toint)
	end
	if userdata.status != 0
		throw(error("Failed to solve trust region subproblem using GLTR iterative method from GALAHAD. Status is $(userdata.status)."))
	end
	return true, userdata.multiplier, d, userdata.iter, false
end

#Based on Theorem 4.3 in Numerical Optimization by Wright

# function computeSearchDirection(g::Vector{Float64}, H, δ::Float64, γ_2::Float64, r::Float64, total_number_factorizations::Int64, min_grad::Float64, print_level::Int64=0) (Old)
function computeSearchDirection(g::Vector{Float64}, H, δ::Float64, γ_2::Float64, r::Float64, min_grad::Float64, print_level::Int64=0)
	temp_total_number_factorizations_bisection = 0
	temp_total_number_factorizations_findinterval = 0
	temp_total_number_factorizations_compute_search_direction = 0
	temp_total_number_factorizations_ = 0
	start_time_temp = time()
	if print_level >= 2
		println("Starting Find Interval")
	end
	success, δ, δ_prime, temp_total_number_factorizations_findinterval = findinterval(g, H, δ, γ_2, r, print_level)
	temp_total_number_factorizations_ += temp_total_number_factorizations_findinterval
	end_time_temp = time()
	total_time_temp = end_time_temp - start_time_temp
	if print_level >= 2
		println("findinterval operation finished with (δ, δ_prime) = ($δ, $δ_prime) and took $total_time_temp.")
	end

	if !success
		@assert temp_total_number_factorizations_ == temp_total_number_factorizations_findinterval + temp_total_number_factorizations_bisection + temp_total_number_factorizations_compute_search_direction
		# return false, false, δ, δ, δ_prime, zeros(length(g)), total_number_factorizations, false old code
		return false, false, δ, δ, δ_prime, zeros(length(g)), temp_total_number_factorizations_, false, temp_total_number_factorizations_findinterval, temp_total_number_factorizations_bisection, temp_total_number_factorizations_compute_search_direction
	end

	start_time_temp = time()
	success, δ_m, δ, δ_prime, temp_total_number_factorizations_bisection = bisection(g, H, δ, γ_2, δ_prime, r, min_grad, print_level)
	temp_total_number_factorizations_ += temp_total_number_factorizations_bisection
	end_time_temp = time()
	total_time_temp = end_time_temp - start_time_temp
	if print_level >= 2
		println("$success. bisection operation took $total_time_temp.")
	end

	if !success
		@assert temp_total_number_factorizations_ == temp_total_number_factorizations_findinterval + temp_total_number_factorizations_bisection + temp_total_number_factorizations_compute_search_direction
		# return true, false, δ_m, δ, δ_prime, zeros(length(g)), total_number_factorizations, false Old code
		return true, false, δ_m, δ, δ_prime, zeros(length(g)), temp_total_number_factorizations_, false, temp_total_number_factorizations_findinterval, temp_total_number_factorizations_bisection, temp_total_number_factorizations_compute_search_direction
	end

	@assert δ <= δ_m <= δ_prime

	sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
	temp_total_number_factorizations_compute_search_direction += 1
	temp_total_number_factorizations_ += temp_total_number_factorizations_compute_search_direction

	start_time_temp = time()
	d_k = cholesky(H + δ_m * sparse_identity) \ (-g)
	end_time_temp = time()
	total_time_temp = end_time_temp - start_time_temp
	if print_level >= 2
		println("d_k operation took $total_time_temp.")
	end
	# return true, true, δ_m, δ, δ_prime, d_k, total_number_factorizations, false Old code
	@assert temp_total_number_factorizations_ == temp_total_number_factorizations_findinterval + temp_total_number_factorizations_bisection + temp_total_number_factorizations_compute_search_direction
	return true, true, δ_m, δ, δ_prime, d_k, temp_total_number_factorizations_, false, temp_total_number_factorizations_findinterval, temp_total_number_factorizations_bisection, temp_total_number_factorizations_compute_search_direction
end

function optimizeSecondOrderModel(g::Vector{Float64}, H, δ::Float64, γ_2::Float64, r::Float64, min_grad::Float64, print_level::Int64=0)
    #When δ is 0 and the Hessian is positive semidefinite, we can directly compute the direction
	total_number_factorizations = 0
	temp_total_number_factorizations_findinterval = 0
	temp_total_number_factorizations_bisection = 0
	temp_total_number_factorizations_compute_search_direction = 0
	temp_total_number_factorizations_inverse_power_iteration = 0
	temp_total_number_factorizations_ = 0
    try
		# total_number_factorizations += 1
		temp_total_number_factorizations_compute_search_direction += 1
		temp_total_number_factorizations_ += temp_total_number_factorizations_compute_search_direction
        d_k = cholesky(H) \ (-g)
		if norm(d_k, 2) <= r
        	# return true, 0.0, d_k, total_number_factorizations, false
			@assert temp_total_number_factorizations_ == temp_total_number_factorizations_findinterval + temp_total_number_factorizations_bisection + temp_total_number_factorizations_compute_search_direction + temp_total_number_factorizations_inverse_power_iteration
			total_number_factorizations += temp_total_number_factorizations_
			return true, 0.0, d_k, total_number_factorizations, false, temp_total_number_factorizations_findinterval, temp_total_number_factorizations_bisection, temp_total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration
        end
    catch e
		#Do nothing
    end
	δ_m = δ
	δ_prime = δ
    try
		# success_find_interval, success_bisection, δ_m, δ, δ_prime, d_k, temp_total_number_factorizations, hard_case = computeSearchDirection(g, H, δ, γ_2, r, total_number_factorizations, min_grad, print_level)
		success_find_interval, success_bisection, δ_m, δ, δ_prime, d_k, temp_total_number_factorizations, hard_case, temp_total_number_factorizations_findinterval, temp_total_number_factorizations_bisection, total_number_factorizations_compute_search_direction = computeSearchDirection(g, H, δ, γ_2, r, min_grad, print_level)
		@assert temp_total_number_factorizations == temp_total_number_factorizations_findinterval + temp_total_number_factorizations_bisection + total_number_factorizations_compute_search_direction
		temp_total_number_factorizations_compute_search_direction += total_number_factorizations_compute_search_direction # TO ACCOUNT FOR THE FIRST ATTEMP WITH d_k = cholesky(H) \ (-g)
		temp_total_number_factorizations_ += temp_total_number_factorizations
		success = success_find_interval && success_bisection
		if success
			# return true, δ_m, d_k, total_number_factorizations, hard_case
			@assert temp_total_number_factorizations_ == temp_total_number_factorizations_findinterval + temp_total_number_factorizations_bisection + temp_total_number_factorizations_compute_search_direction + temp_total_number_factorizations_inverse_power_iteration
			total_number_factorizations += temp_total_number_factorizations_
			return true, δ_m, d_k, total_number_factorizations, hard_case, temp_total_number_factorizations_findinterval, temp_total_number_factorizations_bisection, temp_total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration
		end
		if success_find_interval
			throw(error("Bisection logic failed to find a root for the phi function"))
		else
			throw(error("Bisection logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0."))
		end
    catch e
		println("Error: ", e)
        if e == ErrorException("Bisection logic failed to find a root for the phi function")
			start_time_temp = time()
			# success, δ, d_k, temp_total_number_factorizations = solveHardCaseLogic(g, H, γ_2, r, δ, δ_prime, min_grad, print_level)
			# total_number_factorizations += temp_total_number_factorizations
			success, δ, d_k, temp_total_number_factorizations, total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration = solveHardCaseLogic(g, H, γ_2, r, δ, δ_prime, min_grad, print_level)
			@assert temp_total_number_factorizations == total_number_factorizations_compute_search_direction + temp_total_number_factorizations_inverse_power_iteration
			temp_total_number_factorizations_compute_search_direction += total_number_factorizations_compute_search_direction
			temp_total_number_factorizations_ += temp_total_number_factorizations
			end_time_temp = time()
			total_time_temp = end_time_temp - start_time_temp
			if print_level >= 2
				@info "$success. 1.solveHardCaseLogic operation took $total_time_temp."
				println("$success. 1.solveHardCaseLogic operation took $total_time_temp.")
			end
            # return success, δ, d_k, total_number_factorizations, true
			@assert temp_total_number_factorizations_ == temp_total_number_factorizations_findinterval + temp_total_number_factorizations_bisection + temp_total_number_factorizations_compute_search_direction + temp_total_number_factorizations_inverse_power_iteration
			total_number_factorizations += temp_total_number_factorizations_
            return success, δ, d_k, total_number_factorizations, true, temp_total_number_factorizations_findinterval, temp_total_number_factorizations_bisection, temp_total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration
        elseif e == ErrorException("Bisection logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0.")
			@error e
			start_time_temp = time()
			# success, δ, d_k, temp_total_number_factorizations = solveHardCaseLogic(g, H, γ_2, r, δ, δ_prime, min_grad, print_level)
			# total_number_factorizations += temp_total_number_factorizations
			success, δ, d_k, temp_total_number_factorizations, total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration = solveHardCaseLogic(g, H, γ_2, r, δ, δ_prime, min_grad, print_level)
			@assert temp_total_number_factorizations == total_number_factorizations_compute_search_direction + temp_total_number_factorizations_inverse_power_iteration
			temp_total_number_factorizations_compute_search_direction += total_number_factorizations_compute_search_direction
			temp_total_number_factorizations_ += temp_total_number_factorizations
			end_time_temp = time()
			total_time_temp = end_time_temp - start_time_temp
			if print_level >= 2
				@info "$success. 2.solveHardCaseLogic operation took $total_time_temp."
				println("$success. 2.solveHardCaseLogic operation took $total_time_temp.")
			end
	    	# return success, δ, d_k, total_number_factorizations, true
			@assert temp_total_number_factorizations_ == temp_total_number_factorizations_findinterval + temp_total_number_factorizations_bisection + temp_total_number_factorizations_compute_search_direction + temp_total_number_factorizations_inverse_power_iteration
			total_number_factorizations += temp_total_number_factorizations_
	    	return success, δ, d_k, total_number_factorizations, true, temp_total_number_factorizations_findinterval, temp_total_number_factorizations_bisection, temp_total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration
        else
			@error e
            throw(e)
        end
    end
end


function phi(g::Vector{Float64}, H, δ::Float64, γ_2::Float64, r::Float64, print_level::Int64=0)
    sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
    shifted_hessian = H + δ * sparse_identity
	temp_d = zeros(length(g))
	positive_definite = true
    try
		start_time_temp = time()
        shifted_hessian_fact = cholesky(shifted_hessian)
		end_time_temp = time()
		total_time_temp = end_time_temp - start_time_temp
		if print_level >= 2
			println("cholesky inside phi function took $total_time_temp.")
		end

		start_time_temp = time()
		temp_d = shifted_hessian_fact \ (-g)
		computed_norm = norm(temp_d, 2)
		end_time_temp = time()
		total_time_temp = end_time_temp - start_time_temp
		if print_level >= 2
			println("computed_norm opertion took $total_time_temp.")
		end

		if (δ <= 1e-6 && computed_norm <= r)
			return 0, temp_d, positive_definite
		elseif computed_norm < γ_2 * r
	        return -1, temp_d, positive_definite
		elseif computed_norm <= r
	        return 0, temp_d, positive_definite
	    else
	        return 1, temp_d, positive_definite
	    end
    catch e
		positive_definite = false
        return 1, temp_d, positive_definite
    end
end

function findinterval(g::Vector{Float64}, H, δ::Float64, γ_2::Float64, r::Float64, print_level::Int64=0)
	@assert δ >= 0
	if print_level >= 1
		println("STARTING WITH δ = $δ.")
	end
    Φ_δ, temp_d, positive_definite = phi(g, H, 0.0, γ_2, r)

    if Φ_δ == 0
        δ = 0.0
        δ_prime = 0.0
        return true, δ, δ_prime, 1
    end

	δ_original = δ

    Φ_δ, temp_d, positive_definite = phi(g, H, δ, γ_2, r)

    if Φ_δ == 0
        δ_prime = δ
        return true, δ, δ_prime, 2
    end

	δ_prime = δ
	Φ_δ_prime = Φ_δ
	search_δ_prime = true

	if Φ_δ > 0
		δ_prime = δ == 0.0 ? 1.0 : δ * 2
		search_δ_prime = true
	else
		# Here ϕ(δ) < 0 and we need to find new δ' >= 0 such that ϕ(δ') >= 0 and δ' < δ which is not possible
		# in case δ == 0
		@assert δ > 0
		search_δ_prime = false
		# The aim is to find [δ, δ'] such that ϕ(δ) ∈ {0, 1}, ϕ(δ') ∈ {0, -1}, and  ϕ(δ) * ϕ(δ') <= ∈ {0, -1}
		# since here ϕ(δ) < 0, we set δ' = δ and we search for δ < δ'such that ϕ(δ) ∈ {0, 1}
		δ_prime = δ
		Φ_δ_prime = -1
		δ = δ / 2
	end

	max_iterations = 50
    k = 1
	while k < max_iterations
		if search_δ_prime
        	Φ_δ_prime, temp_d, positive_definite = phi(g, H, δ_prime, γ_2, r)
	        if Φ_δ_prime == 0
	            δ = δ_prime
	            return true, δ, δ_prime, k + 2
	        end
		else
			Φ_δ, temp_d, positive_definite = phi(g, H, δ, γ_2, r)
			if Φ_δ == 0
	            δ_prime = δ
	            return true, δ, δ_prime, k + 2
	        end
		end

        if ((Φ_δ * Φ_δ_prime) < 0)
			if print_level >= 1
				println("ENDING WITH ϕ(δ) = $Φ_δ and Φ_δ_prime = $Φ_δ_prime.")
				println("ENDING WITH δ = $δ and δ_prime = $δ_prime.")
			end
			@assert δ_prime > δ
			@assert ((δ == 0.0) & (δ_prime == 1.0)) || ((δ_prime / δ) == 2 ^ (2 ^ (k - 1))) || ((δ_prime / δ) - 2 ^ (2 ^ (k - 1)) <= 1e-3)
			factor = δ_prime / δ
			return true, δ, δ_prime, k + 2
        end
		if search_δ_prime
			# Here Φ_δ_prime is still 1 and we are continue searching for δ',
			# but we can update δ to give it larger values which is the current value of δ'
			@assert Φ_δ_prime > 0
			δ = δ_prime
			δ_prime = δ_prime * (2 ^ (2 ^ k))
		else
			# Here Φ_δ is still -1 and we are continue searching for δ,
			# but we can update δ' to give it smaller value which is the current value of δ
			@assert Φ_δ < 0
			δ_prime = δ
			δ = δ / (2 ^ (2 ^ k))
		end

        k = k + 1
    end

    if (Φ_δ  * Φ_δ_prime > 0)
		if print_level >= 1
			println("Φ_δ is $Φ_δ and Φ_δ_prime is $Φ_δ_prime. δ is $δ and δ_prime is $δ_prime.")
		end
		return false, δ, δ_prime, max_iterations + 2
    end
	factor = δ_prime / δ
    return true, δ, δ_prime, max_iterations + 2
end

function bisection(g::Vector{Float64}, H, δ::Float64, γ_2::Float64, δ_prime::Float64, r::Float64, min_grad::Float64, print_level::Int64=0)
    # the input of the function is the two end of the interval (δ,δ_prime)
    # our goal here is to find the approximate δ using classic bisection method
	initial_δ = δ
	initial_δ_prime = δ_prime
	if print_level >= 1
		println("****************************STARTING BISECTION with (δ, δ_prime) = ($δ, $δ_prime)**************")
	end
    #Bisection logic
    k = 1
    δ_m = (δ + δ_prime) / 2
    Φ_δ_m, temp_d, positive_definite = phi(g, H, δ_m, γ_2, r)
	max_iterations = 100
    while (Φ_δ_m != 0) && k <= max_iterations
		start_time_str = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
		if print_level >= 2
			println("$start_time_str. Bisection iteration $k.")
		end
        if Φ_δ_m > 0
            δ = δ_m
        else
            δ_prime = δ_m
        end
		δ_m = (δ + δ_prime) / 2
        Φ_δ_m, temp_d, positive_definite = phi(g, H, δ_m, γ_2, r)
        k = k + 1
        γ_1 = 100
		if Φ_δ_m != 0
			ϕ_δ_prime, d_temp_δ_prime, positive_definite_δ_prime = phi(g, H, δ_prime, γ_2, r)
			ϕ_δ, d_temp_δ, positive_definite_δ = phi(g, H, δ, γ_2, r)
			q_1 = norm(H * d_temp_δ_prime + g + δ_prime * d_temp_δ_prime)
			q_2 = min_grad / γ_1
			if print_level >= 2
				println("$k===============Bisection entered here=================")
			end

			if (abs(δ_prime - δ) <= (min_grad / (1000 * r))) && q_1 <= q_2 && !positive_definite_δ
			# if (abs(δ_prime - δ) <= (min_grad / (1000 * r))) && q_1 <= q_2 && !positive_definite_δ
				if print_level >= 2
					println("$k===================norm(H * d_temp_δ_prime + g + δ_prime * d_temp_δ_prime) is $q_1.============")
					println("$k===================min_grad / (100 r) is $q_2.============")
					println("$k===================ϕ_δ_prime is $ϕ_δ_prime.============")

					println("$k===============Bisection entered here=================")
					mimimum_eigenvalue = eigmin(Matrix(H))
					mimimum_eigenvalue_abs = abs(mimimum_eigenvalue)
					@info "$k=============1Bisection Failure New Logic==============$initial_δ,$δ,$mimimum_eigenvalue,$mimimum_eigenvalue_abs."
					println("$k=============1Bisection Failure New Logic==============$initial_δ,$δ,$mimimum_eigenvalue,$mimimum_eigenvalue_abs.")
				end
				break
			end
		end
    end

    if (Φ_δ_m != 0)
		if print_level >= 1
			println("Φ_δ_m is $Φ_δ_m.")
			println("δ, δ_prime, and δ_m are $δ, $δ_prime, and $δ_m. γ_2 is $γ_2.")
		end
		return false, δ_m, δ, δ_prime, min(k, max_iterations) + 1
    end
	if print_level >= 1
		println("****************************ENDING BISECTION with δ_m = $δ_m**************")
	end
    return true, δ_m, δ, δ_prime, min(k, max_iterations) + 1
end

function solveHardCaseLogic(g::Vector{Float64}, H, γ_2::Float64, r::Float64, δ::Float64, δ_prime::Float64, min_grad::Float64, print_level::Int64=0)
	sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
	total_number_factorizations = 0
	temp_total_number_factorizations_compute_search_direction = 0
	temp_total_number_factorizations_inverse_power_iteration = 0
	temp_total_number_factorizations_ = 0

	temp_eigenvalue = 0
	try
		start_time_temp = time()
		success, eigenvalue, eigenvector, temp_total_number_factorizations_inverse_power_iteration, temp_d_k = inverse_power_iteration(g, H, min_grad, δ, δ_prime, r, γ_2)
		temp_eigenvalue = eigenvalue
		end_time_temp = time()
	    total_time_temp = end_time_temp - start_time_temp
		if print_level >= 2
	    	@info "inverse_power_iteration operation took $total_time_temp."
		end
		eigenvalue = abs(eigenvalue)
		temp_total_number_factorizations_ += temp_total_number_factorizations_inverse_power_iteration
		norm_temp_d_k = norm(temp_d_k)

		if norm_temp_d_k == 0
			@assert temp_total_number_factorizations_ == temp_total_number_factorizations_compute_search_direction + temp_total_number_factorizations_inverse_power_iteration
			total_number_factorizations += temp_total_number_factorizations_
			return false, eigenvalue, zeros(length(g)), total_number_factorizations, temp_total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration
		end

		if print_level >= 2
			@info "candidate search direction norm is $norm_temp_d_k. r is $r. γ_2 is $γ_2"
		end
		if γ_2 * r <= norm(temp_d_k) <= r
			@assert temp_total_number_factorizations_ == temp_total_number_factorizations_compute_search_direction + temp_total_number_factorizations_inverse_power_iteration
			total_number_factorizations += temp_total_number_factorizations_
			return true, eigenvalue, temp_d_k, total_number_factorizations, temp_total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration
		end
		if norm(temp_d_k) > r
			if print_level >= 1
				println("This is noit a hard case. FAILURE======candidate search direction norm is $norm_temp_d_k. r is $r. γ_2 is $γ_2")
				@warn "This is noit a hard case. candidate search direction norm is $norm_temp_d_k. r is $r. γ_2 is $γ_2"
			end
		end

		@assert temp_total_number_factorizations_ == temp_total_number_factorizations_compute_search_direction + temp_total_number_factorizations_inverse_power_iteration
		total_number_factorizations += temp_total_number_factorizations_
		return false, eigenvalue, zeros(length(g)), total_number_factorizations, temp_total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration
	catch e
		@error e
		if print_level >= 2
			matrix_H = Matrix(H)
			mimimum_eigenvalue = eigmin(Matrix(H))
			println("FAILURE+++++++inverse_power_iteration operation returned non positive matrix. retunred_eigen_value is $temp_eigenvalue and mimimum_eigenvalue is $mimimum_eigenvalue.")
		end
		@assert temp_total_number_factorizations_ ==  temp_total_number_factorizations_compute_search_direction + temp_total_number_factorizations_inverse_power_iteration
		total_number_factorizations += temp_total_number_factorizations_
		return false, δ_prime, zeros(length(g)), total_number_factorizations, temp_total_number_factorizations_compute_search_direction, temp_total_number_factorizations_inverse_power_iteration
	end
end

function inverse_power_iteration(g, H, min_grad, δ, δ_prime, r, γ_2; max_iter=1000, ϵ=1e-3, print_level=2)
   sigma = δ_prime
   start_time_temp = time()
   n = size(H, 1)
   x = ones(n)
   y = ones(n)
   sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
   y_original_fact = cholesky(H + sigma * sparse_identity)
   temp_factorization = 1
   for k in 1:max_iter
       y = y_original_fact \ x
       y /= norm(y)
	   eigenvalue = dot(y, H * y)

	   if norm(H * y + δ_prime * y) <= abs(δ_prime - δ) + (min_grad / (10 ^ 2 * r))
		   try
			   temp_factorization += 1
			   temp_d_k =  cholesky(H + (abs(eigenvalue) + 1e-1) * sparse_identity) \ (-g)
       		   return true, eigenvalue, y, temp_factorization, temp_d_k
		   catch
			   #DO NOTHING
		   end
	   end

	   #Keep as a safety check. This a sign that we can't solve thr trust region subprobelm
       if norm(x + y) <= ϵ || norm(x - y) <= ϵ
		   eigenvalue = dot(y, H * y)
		   try
			   temp_factorization += 1
			   temp_d_k =  cholesky(H + (abs(eigenvalue) + 1e-1) * sparse_identity) \ (-g)
			   return true, eigenvalue, y, temp_factorization, temp_d_k
		   catch
			   #DO NOTHING
		   end
       end

       x = y
   end
   temp_ = dot(y, H * y)

   temp_1 = norm(x + y)
   temp_2 = norm(x - y)
   if print_level >= 2
	   @error ("Inverse power iteration did not converge. computed eigenValue is $temp_. norm(x + y) = $temp_1 and norm(x - y) = $temp_2.")
   end

   if print_level >= 2
	   end_time_temp = time()
	   total_time_temp = end_time_temp - start_time_temp
	   @info "inverse_power_iteration operation took $total_time_temp."
	   println("inverse_power_iteration operation took $total_time_temp.")
   end

   temp_d_k = zeros(length(g))
   return false, temp_, y, temp_factorization, temp_d_k
end

#Based on 'THE HARD CASE' section from Numerical Optimization by Wright
function solveHardCaseLogic(g::Vector{Float64}, H, γ_2::Float64, r::Float64, print_level::Int64=0)
    minimumEigenValue = eigmin(Matrix(H))
	if minimumEigenValue >= 0
		Q = eigvecs(Matrix(H))
		eigenvaluesVector = eigvals(Matrix(H))

		temp_d_0 = zeros(length(g))
		for i in 1:length(eigenvaluesVector)
			temp_d_0 = temp_d_0 .- ((Q[:, i]' * g) / (eigenvaluesVector[i] + 0)) * Q[:, i]
	    end

		temp_d_0_norm = norm(temp_d_0, 2)
		less_than_radius = temp_d_0_norm <= r
		if print_level >= 1
			println("temp_d_0_norm is $temp_d_0_norm and ||d(0)|| <= r is $less_than_radius.")
		end
		if less_than_radius
			return true, 0.0, temp_d_0, 0
		end
		if print_level >= 1
			println("minimumEigenValue is $minimumEigenValue")
			println("r is $r")
			println("g is $g")
			H_matrix = Matrix(H)
			println("H is $H_matrix")
		end
		return false, minimumEigenValue, zeros(length(g)), 0
	end
    δ = -minimumEigenValue
	try
		Q = eigvecs(Matrix(H))
		z =  Q[:,1]
		temp_ = dot(z', g)
		if print_level >= 1
			println("Q_1 ^ T g = $temp_.")
			println("minimumEigenValue = $minimumEigenValue.")
		end
	    eigenvaluesVector = eigvals(Matrix(H))

		temp_d = zeros(length(g))
		for i in 1:length(eigenvaluesVector)
			if eigenvaluesVector[i] != minimumEigenValue
	            temp_d = temp_d .- ((Q[:, i]' * g) / (eigenvaluesVector[i] + δ)) * Q[:, i]
	        end
	    end

		temp_d_norm = norm(temp_d, 2)
		less_than_radius_ = temp_d_norm < r
		if print_level >= 1
			println("temp_d_norm is $temp_d_norm and ||d(-λ_1)|| < r is $less_than_radius_.")
		end

		if !less_than_radius_
			if print_level >= 0
				println("This is not a hard case sub-problem.")
			end
			@error "This is not a hard case sub-problem."
			try
				success_find_interval, success_bisection, δ_m, d_k, total_number_factorizations, temp_hard_case  = computeSearchDirection(g, H, δ, γ_2, r, 0, print_level)
				temp_success = success_find_interval && success_bisection
				return temp_success, δ_m, d_k, total_number_factorizations
			catch e
				@error e
			end
		end

	    norm_d_k_squared_without_τ_squared = 0.0

	    for i in 1:length(eigenvaluesVector)
	        if eigenvaluesVector[i] != minimumEigenValue
	            norm_d_k_squared_without_τ_squared = norm_d_k_squared_without_τ_squared + ((Q[:, i]' * g) ^ 2 / (eigenvaluesVector[i] + δ) ^ 2)
	        end
	    end

	    norm_d_k_squared = r ^ 2
		if norm_d_k_squared < norm_d_k_squared_without_τ_squared && print_level >= 1
			println("norm_d_k_squared is $norm_d_k_squared and norm_d_k_squared_without_τ_squared is $norm_d_k_squared_without_τ_squared.")
		end

		if norm_d_k_squared < norm_d_k_squared_without_τ_squared
			if less_than_radius
				if print_level >= 1
					println("HAD CASE LOGIC: δ, d_k and r are $δ, $temp_d_norm, and $r.")
				end
				return true, δ, temp_d, 0
			end
			if print_level >= 1
				println("minimumEigenValue is $minimumEigenValue")
				println("r is $r")
				println("g is $g")
				H_matrix = Matrix(H)
				println("H is $H_matrix")
			end
			return false, δ, zeros(length(g)), 0
		end

	    τ = sqrt(norm_d_k_squared - norm_d_k_squared_without_τ_squared)
	    d_k = τ .* z

	    for i in 1:length(eigenvaluesVector)
	        if eigenvaluesVector[i] != minimumEigenValue
	            d_k = d_k .- ((Q[:, i]' * g) / (eigenvaluesVector[i] + δ)) * Q[:, i]
	        end
	    end
		temp_norm_d_k = norm(d_k, 2)
		if print_level >= 1
			println("HAD CASE LOGIC: δ, d_k and r are $δ, $temp_norm_d_k, and $r.")
		end
	    return true, δ, d_k, 0
	catch e
		@show e
		if print_level >= 1
			println("minimumEigenValue is $minimumEigenValue")
			println("r is $r")
			println("g is $g")
			H_matrix = Matrix(H)
			println("H is $H_matrix")
		end
		return false, δ, zeros(length(g)), 0
	end

end
