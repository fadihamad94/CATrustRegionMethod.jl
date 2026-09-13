# Trust-region subproblem solvers

`TrustRegionSubproblemSolvers` solves quadratic trust-region subproblems of the
form

```math
\min_d \; g^T d + \tfrac12 d^T H d \quad \text{subject to } \lVert d \rVert \le r.
```

It is a standalone Julia package and the local subproblem dependency of
`CATrustRegionMethod`.

## Installation and tests

The package requires Julia 1.12 or later. From this directory:

```shell
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Usage

```julia
using LinearAlgebra
using TrustRegionSubproblemSolvers

g = [-2.0]
H = reshape([2.0], 1, 1)
result = solveTrustRegionSubproblem(
    "example",
    g,
    H,
    0.0,  # initial shift
    0.01, # gamma_1
    0.8,  # gamma_2
    0.5,  # gamma_3
    0.5,  # radius
    norm(g);
    print_level = 0,
)

result.direction
result.delta
result.factorizations.total
```

The implementation enforces all four termination inequalities, including
accepting a valid zero step. It attempts to
certify each shifted solve with iterative refinement, but continues the
interval search using the computed direction when that stronger certificate is
not obtained. The final direction must still satisfy all four termination
inequalities. On numerical failure, it reruns the complete algorithm with the
configured perturbed-gradient backup.

The result also reports the final bracket endpoint, whether hard-case logic was
used, and factorization counts for interval finding, bisection, direction
computation, and inverse iteration. The conference-version solver remains
available as `solveTrustRegionSubproblemOldApproach` for CAT-NeurIPS and the
corresponding ablation.
