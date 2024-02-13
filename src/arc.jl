#= arc.jl
ARC Julia interface using C sparse matrix indexing =#

const LIBRARY_PATH_ARC = string(@__DIR__ ,"/../lib/arc.so")

struct userdata_type_arc
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

function arc(n::Int64, x::Vector{Float64}, g::Vector{Float64}, print_level::Int64, maxit::Int64, initial_weight::Float64, max_inner_iterations_or_factorizations::Int64, clock_time_limit::Float64)
	eval_f_c = @cfunction(eval_f, Cdouble, (Ptr{Cdouble},));
	eval_g_c = @cfunction(eval_g, Ptr{Cdouble}, (Ptr{Cdouble},));
	eval_h_c = @cfunction(eval_h, Ptr{Cdouble}, (Ptr{Cdouble},));
	p = Libc.malloc(Core.sizeof(Cdouble) * length(nlp.meta.x0))
	p = convert(Ptr{Float64}, p)
	for i in 1:n
		unsafe_store!(p, x[i] , i)
	end
	#default
	eta_too_successful = 2.0
    eta_1 = 1e-8
	eta_2 = 0.9
	subproblem_direct = false
	stop_g_absolute = 1e-5
	stop_g_relative = stop_g_absolute / norm(g, 2)
	ϵ_machine = eps(Float64)
	stop_s = ϵ_machine
	userdata = userdata_type_arc(n, eval_f_c, eval_g_c, eval_h_c, 0, 0, 0, 0, 0, 0, p)
	@time begin
  	 userdata = ccall((:arc, LIBRARY_PATH_ARC), userdata_type_arc, (Ref{Cdouble}, Ref{Cdouble}, userdata_type_arc, Cint, Cint, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cuchar, Cint, Cdouble), x, g, userdata, print_level, maxit, initial_weight, stop_g_absolute, stop_g_relative, stop_s, eta_too_successful, eta_1, eta_2, subproblem_direct, max_inner_iterations_or_factorizations, clock_time_limit)
	end
	solution = Vector{Float64}()
	for i in 1:n
		push!(solution, unsafe_load(userdata.solution, i))
	end
	return userdata, solution
end
