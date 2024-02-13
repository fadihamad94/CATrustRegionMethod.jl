#= tru.jl
TRU Julia interface using C sparse matrix indexing =#

const LIBRARY_PATH_TRU = string(@__DIR__ ,"/../lib/tru.so")

struct userdata_type_tru
    n::Cint
    eval_f::Ptr{Cvoid}
	eval_g::Ptr{Cvoid}
	eval_h::Ptr{Cvoid}
	status::Cint
	iter::Cint
	total_function_evaluation::Cint
	total_gradient_evaluation::Cint
	total_hessian_evaluation::Cint
	total_inner_iterations_or_factorizations::Cint
	solution::Ptr{Cdouble}
end

function eval_f(x::Ref{Cdouble})
	temp_vec = Vector{Float64}()
	for i in 1:(length(nlp.meta.x0))
		push!(temp_vec, unsafe_load(x, i))
	end
	return obj(nlp, temp_vec)
end

function eval_g(x::Ref{Cdouble})
	temp_vec = Vector{Float64}()
	for i in 1:(length(nlp.meta.x0))
		push!(temp_vec, unsafe_load(x, i))
	end
	temp_result = grad(nlp, temp_vec)
	p = Libc.malloc(Core.sizeof(Cdouble) * length(nlp.meta.x0))
	p = convert(Ptr{Float64}, p)
    for i in 1:(length(nlp.meta.x0))
		unsafe_store!(p, temp_result[i] , i)
	end
	return p
end

function eval_h(x::Ref{Cdouble})
	temp_vec = Vector{Float64}()
	for i in 1:(length(nlp.meta.x0))
		push!(temp_vec, unsafe_load(x, i))
	end
	h_matrix = hess(nlp, temp_vec)
	h_vec = []
	for i in 1:length(nlp.meta.x0)
		for j in 1:i
			push!(h_vec, h_matrix[i, j])
		end
	end
	p = Libc.malloc(Core.sizeof(Cdouble) * length(h_vec))
	p = convert(Ptr{Float64}, p)
	for i in 1:(length(h_vec))
		unsafe_store!(p, h_vec[i] , i)
	end
	return p
end

function eval_h_sparse(x::Ref{Cdouble})
	temp_vec = Vector{Float64}()
	for i in 1:(length(nlp.meta.x0))
		push!(temp_vec, unsafe_load(x, i))
	end
	h_matrix = hess(nlp, temp_vec)
	h_vec = []
	for i in 1:length(nlp.meta.x0)
		for j in 1:i
			if h_matrix[i, j] != 0.0
				push!(h_vec, h_matrix[i, j])
			end
		end
	end
	current_time = Dates.format(now(), "mm/dd/yyyy HH:MM:SS")
	length_h_vec = length(h_vec)
	# @info "Computing Hessian at $current_time with $length_h_vec."
	p = Libc.malloc(Core.sizeof(Cdouble) * length(h_vec))
	p = convert(Ptr{Float64}, p)
	for i in 1:(length(h_vec))
		unsafe_store!(p, h_vec[i] , i)
	end
	return p
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
	#push!(H_ptr, size(H)[1] + 1)
	push!(H_ptr, H_ne)
	return H_ne, H_val, H_row, H_col, H_ptr
end

function tru(n::Int64, x::Vector{Float64}, g::Vector{Float64}, H, print_level::Int64, maxit::Int64, initial_radius::Float64, subproblem_direct::Bool, max_inner_iterations_or_factorizations::Int64, max_time::Float64=30*60.0)
	H_type = "sparse_by_rows"
	# if length(x) < 3000
	# 	H_type = "dense"
	# end
	# H_type = "dense"
	# H_type = "coordinate"
	eval_f_c = @cfunction(eval_f, Cdouble, (Ptr{Cdouble},));
	eval_g_c = @cfunction(eval_g, Ptr{Cdouble}, (Ptr{Cdouble},));
	eval_h_c = @cfunction(eval_h_sparse, Ptr{Cdouble}, (Ptr{Cdouble},));
	if H_type == "dense"
		eval_h_c = @cfunction(eval_h, Ptr{Cdouble}, (Ptr{Cdouble},));
	end

	H_ne = 0
	H_val = Nothing
	H_row = Nothing
	H_col = Nothing
	H_ptr = Nothing
	if H_type == "dense"
		# H_val = getHessianDenseLowerTriangularPart(H)
		H_row = [Int32(0)]
		H_col = [Int32(0)]
		H_ptr = [Int32(0)]
		# H_ne = length(H_dense)
		H_ne = Int(n * (n + 1) / 2)
	else
		H_ne, H_val, H_row, H_col, H_ptr = getHessianSparseLowerTriangularPart(H)
	end

	if H_type == "sparse_by_rows" && length(H_ptr) != length(g) + 1
		@warn "Weired case detected."
		H_type = "coordinate"
	end
	@info "H_type is $H_type."
	@info "H_ne is $H_ne."

	p = Libc.malloc(Core.sizeof(Cdouble) * length(nlp.meta.x0))
	p = convert(Ptr{Float64}, p)
	for i in 1:n
		unsafe_store!(p, x[i] , i)
	end
	userdata = userdata_type_tru(n, eval_f_c, eval_g_c, eval_h_c, 0, 0, 0, 0, 0, 0,p)
	stop_g_absolute = 1e-5
	stop_g_relative = stop_g_absolute / norm(g, 2)
	clock_time_limit = max_time
	#stop_s = 1e-8
	stop_s = -1.0
	# non_monotone = 25
	non_monotone = 0
    userdata = ccall((:tru, LIBRARY_PATH_TRU), userdata_type_tru, (Cint, Cstring, Ref{Cdouble}, Ref{Cdouble}, Ref{Cint}, Ref{Cint}, Ref{Cint}, userdata_type_tru, Cint, Cint, Cdouble, Cdouble, Cdouble, Cdouble, Cuchar, Cint, Cdouble, Cint), H_ne, H_type, x, g, H_row, H_col, H_ptr, userdata, print_level, maxit, initial_radius, stop_g_absolute, stop_g_relative, stop_s, subproblem_direct, max_inner_iterations_or_factorizations, clock_time_limit, non_monotone)
	solution = Vector{Float64}()
	for i in 1:length(nlp.meta.x0)
		push!(solution, unsafe_load(userdata.solution, i))
	end
	@show userdata.total_inner_iterations_or_factorizations
	return userdata, solution
end
