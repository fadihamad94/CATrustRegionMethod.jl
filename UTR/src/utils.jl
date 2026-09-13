function assertFactorizationAccounting(algorithm_counter::AlgorithmCounter)
    return assert_factorization_accounting(algorithm_counter)
end

"""Record the shifted Cholesky tests used to select UTR's `(σ, r)` case."""
function recordParameterSelectionFactorizations!(
    algorithm_counter::AlgorithmCounter,
    count::Integer = 1,
)
    @assert count >= 0
    increment!(
        algorithm_counter,
        :total_number_factorizations_parameter_selection,
        count,
    )
    increment!(algorithm_counter, :total_number_factorizations, count)
    return assertFactorizationAccounting(algorithm_counter)
end

"""Add the factorization statistics returned by one shared subproblem solve."""
function recordFactorizationStats!(
    algorithm_counter::AlgorithmCounter,
    stats::FactorizationStats,
)
    return record_factorization_stats!(algorithm_counter, stats)
end
