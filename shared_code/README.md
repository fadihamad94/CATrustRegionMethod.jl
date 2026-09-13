# CATrustRegionShared

`CATrustRegionShared` is a private package for code shared by the CAT and UTR
solvers. It contains common termination defaults, solver-local type generators,
algorithm-counter operations, NLP/MOI evaluation adapters, basic MOI storage
types, and benchmark formatting helpers.

The package deliberately does not define a shared solver-parameter object.
CAT and UTR retain separate `AlgorithmicParameters` types and source identical
trust-region subproblem defaults from `TrustRegionSubproblemSolvers`.

Reusable test-only NLP model builders live in `test_support` and are not loaded
by the runtime module.
