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

function tru(n::Int64, x::Vector{Float64}, g::Vector{Float64}, print_level::Int64, maxit::Int64, initial_radius::Float64, subproblem_direct::Bool, max_inner_iterations_or_factorizations::Int64)
	eval_f_c = @cfunction(eval_f, Cdouble, (Ptr{Cdouble},));
	eval_g_c = @cfunction(eval_g, Ptr{Cdouble}, (Ptr{Cdouble},));
	eval_h_c = @cfunction(eval_h, Ptr{Cdouble}, (Ptr{Cdouble},));
	p = Libc.malloc(Core.sizeof(Cdouble) * length(nlp.meta.x0))
	p = convert(Ptr{Float64}, p)
	for i in 1:n
		unsafe_store!(p, x[i] , i)
	end
	userdata = userdata_type_tru(n, eval_f_c, eval_g_c, eval_h_c, 0, 0, 0, 0, 0, 0,p)
	stop_g_absolute = 1e-5
	stop_g_relative = stop_g_absolute / norm(g, 2)
	clock_time_limit = 30 * 60.0
	#stop_s = 1e-8
	stop_s = -1.0
    userdata = ccall((:tru, LIBRARY_PATH_TRU), userdata_type_tru, (Ref{Cdouble}, Ref{Cdouble}, userdata_type_tru, Cint, Cint, Cdouble, Cdouble, Cdouble, Cdouble, Cuchar, Cint, Cdouble), x, g, userdata, print_level, maxit, initial_radius, stop_g_absolute, stop_g_relative, stop_s, subproblem_direct, max_inner_iterations_or_factorizations, clock_time_limit)
	solution = Vector{Float64}()
	for i in 1:length(nlp.meta.x0)
		push!(solution, unsafe_load(userdata.solution, i))
	end
	return userdata, solution
end
