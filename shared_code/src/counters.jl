macro define_algorithm_counter(variant)
    variant_value = variant isa QuoteNode ? variant.value : variant
    common_fields = [
        :(total_function_evaluation::Int64),
        :(total_gradient_evaluation::Int64),
        :(total_hessian_evaluation::Int64),
        :(total_number_factorizations::Int64),
        :(total_number_subproblem_iterations::Int64),
        :(total_number_hessian_vector_products::Int64),
    ]
    factor_fields = [
        :(total_number_factorizations_findinterval::Int64),
        :(total_number_factorizations_bisection::Int64),
        :(total_number_factorizations_compute_search_direction::Int64),
        :(total_number_factorizations_inverse_power_iteration::Int64),
    ]
    fields = if variant_value == :cat
        [common_fields; factor_fields]
    elseif variant_value == :utr
        [
            common_fields
            :(total_number_factorizations_parameter_selection::Int64)
            factor_fields
            :(total_number_subproblem_solves::Int64)
            :(total_number_rejected_trials::Int64)
        ]
    else
        error("@define_algorithm_counter expects :cat or :utr")
    end
    zeros = fill(0, length(fields))
    return esc(
        quote
            mutable struct AlgorithmCounter
                $(fields...)
                AlgorithmCounter() = new($(zeros...))
            end
        end,
    )
end

@inline increment!(counter, field::Symbol) = increment!(counter, field, 1)

@inline function increment!(counter, field::Symbol, count::Integer)
    hasfield(typeof(counter), field) ||
        throw(ArgumentError("Unknown counter field: $field"))
    value = getfield(counter, field) + Int64(count)
    setfield!(counter, field, value)
    return value
end

function assert_factorization_accounting(counter)
    parameter_selection =
        hasfield(typeof(counter), :total_number_factorizations_parameter_selection) ?
        counter.total_number_factorizations_parameter_selection : 0
    @assert counter.total_number_factorizations ==
            parameter_selection +
            counter.total_number_factorizations_findinterval +
            counter.total_number_factorizations_bisection +
            counter.total_number_factorizations_compute_search_direction +
            counter.total_number_factorizations_inverse_power_iteration
    return counter
end

function record_factorization_stats!(counter, stats)
    increment!(counter, :total_number_factorizations, stats.total)
    increment!(
        counter,
        :total_number_factorizations_findinterval,
        stats.findinterval,
    )
    increment!(
        counter,
        :total_number_factorizations_bisection,
        stats.bisection,
    )
    increment!(
        counter,
        :total_number_factorizations_compute_search_direction,
        stats.compute_search_direction,
    )
    increment!(
        counter,
        :total_number_factorizations_inverse_power_iteration,
        stats.inverse_power_iteration,
    )
    return assert_factorization_accounting(counter)
end

function record_iterative_stats!(counter, stats)
    stats.iterations >= 0 ||
        throw(ArgumentError("The subproblem iteration count must be nonnegative."))
    stats.hessian_vector_products >= 0 ||
        throw(ArgumentError("The Hessian-vector product count must be nonnegative."))
    increment!(
        counter,
        :total_number_subproblem_iterations,
        stats.iterations,
    )
    increment!(
        counter,
        :total_number_hessian_vector_products,
        stats.hessian_vector_products,
    )
    return counter
end
