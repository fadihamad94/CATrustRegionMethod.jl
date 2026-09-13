# Universal trust-region solver (`UniversalTrustRegionMethod`)

This package implements the first-order adaptive universal trust-region (UTR)
method specified in
[`adaptive_utr_implementation.tex`](docs/adaptive_utr_implementation.tex) for
unconstrained minimization:

```math
\min_{x \in \mathbb{R}^n} f(x).
```

At an inner trial, let

```math
s_k = \sqrt{\lVert \nabla f(x_k) \rVert_2}.
```

The coefficients selected by the adaptive strategy are converted into the
actual regularized Hessian and trust-region radius as

```math
H_{\mathrm{reg}} =
\nabla^2 f(x_k) + \sigma_k s_k I,
\qquad
\Delta_k = r_k s_k.
```

Thus `σ_k` is a Hessian-shift coefficient rather than the final shift, and
`r_k` is a radius coefficient rather than the final radius.

With `τ_k = ρ_k s_k`, shifted Cholesky tests choose:

- `σ_k = 0` and `r_k = 1 / (2ρ_k)` if `H_k + τ_k I` is not positive
  definite or `H_k - τ_k I` is positive definite.
- `σ_k = ρ_k` and `r_k = 1 / (4ρ_k)` otherwise.

The tests use strict positive definiteness, so the exact positive boundary
`λ_min(H_k) = τ_k` uses the second choice.

## Installation and tests

The package requires Julia 1.12 or later. From the monorepo root:

```shell
julia --project=UTR -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The local dependency on `TrustRegionSubproblemSolvers` is declared through
Julia's `[sources]` table.

## Usage

Use the solver directly with an `NLPModels.AbstractNLPModel`:

```julia
using UniversalTrustRegionMethod

solution, status, history, counters, iterations, elapsed =
    UTR_solve(nlp)
```

Or use the MathOptInterface optimizer through JuMP:

```julia
using JuMP
using UniversalTrustRegionMethod

model = Model(UniversalTrustRegionMethod.Optimizer)
@variable(model, x, start = -1.2)
@variable(model, y, start = 1.0)
@NLobjective(model, Min, (1 - x)^2 + 100 * (y - x^2)^2)
set_silent(model)
optimize!(model)
```

Algorithm and termination settings use the same raw-attribute convention as
the CAT solver:

```julia
set_attribute(model, "algorithm_params!ρ_0", 2.0)
set_attribute(model, "termination_criteria!gradient_termination_tolerance", 1e-6)
```

Each outer iteration attempts at most `MAX_INNER_ITERATIONS` trial steps
(default: 100). If none is accepted, the direct API returns
`TerminationStatusCode.INNER_ITERATION_LIMIT`; the MathOptInterface wrapper
reports `MOI.ITERATION_LIMIT` and retains `InnerIterationLimit` as its raw
status so this case remains distinguishable from the outer-iteration limit.

UTR always uses the certified direct trust-region subproblem solver. Its
adaptive rule first performs shifted Cholesky tests to decide whether
regularization is needed, so UTR does not expose an iterative-only
configuration. Counters retain the common subproblem-iteration and
Hessian-vector-product fields, but these remain zero for UTR.

The solver supports unconstrained minimization only. Benchmarking is maintained
outside this package. Objective gradients and Hessians must both be available
from the supplied model.
