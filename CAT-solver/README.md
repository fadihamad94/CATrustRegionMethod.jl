# CAT solver (`CATrustRegionMethod`)

This package implements a trust-region method for unconstrained optimization: 

$$\min_{x \in \mathbb{R}^n} f(x).$$

The method finds stationary points, specifically points with $|| \nabla f(x) || \leq \epsilon$. In particular, in our paper, we show that the method achieves the best possible convergence bound up to an additive logarithmic factor, for finding an $\epsilon$-approximate stationary point, namely $O( \Delta_f L^{1/2} \epsilon^{-3/2}) + \tilde{O}(1)$ iterations, where $L$ is the Lipschitz constant of the Hessian, $\Delta_f$ is the optimality gap, and $\epsilon$ is the termination tolerance for the gradient norm."

Consistently adaptive (CA) in the package name refers to the method achieving the best possible convergence bound without requiring knowledge of the Lipschitz constant ($L$) of the Hessian.

## License

CATrustRegionMethod.jl is licensed under the [MIT License](LICENSE).

## Installation

CATrustRegionMethod requires Julia 1.12 or later.

From the monorepo root, instantiate the CAT package and its sibling subproblem
dependency as follows:

```shell
cd CAT-solver
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The local dependency is declared in `Project.toml` through Julia's `[sources]`
table. In code, the package remains `CATrustRegionMethod`, preserving the
existing public API.

That source declaration is local to this monorepo. In an external development
environment, develop `trust-region-subproblem-solvers` before developing
`CAT-solver`; a separately distributed CAT package will require the subproblem
package to be registered or pinned by URL.

The commands below assume `CAT-solver` is the current directory.

## Running

### Use with JuMP

To use CATrustRegionMethod with JuMP, use `CATrustRegionMethod.Optimizer`:

```julia
using CATrustRegionMethod, JuMP
model = Model(CATrustRegionMethod.Optimizer)
@variable(model, x)
@variable(model, y)
@NLobjective(model, Min, (2.0 - x)^2 + 100 * (y - x^2)^2)
set_attribute(model, "time_limit", 1800.0)
set_attribute(model, "algorithm_params!r_1", 100.0)
optimize!(model)
status = termination_status(model)
# Retrieve the solver instance
optimizer = unsafe_backend(model)
# Algorithm stats (total function evalation, ...)
algorithm_counter = optimizer.inner.algorithm_counter
```

Both JuMP's current `@objective` representation (including direct quadratic and
nonlinear scalar objectives) and the legacy `@NLobjective` representation are
supported. CAT is a minimization solver and rejects maximization objectives.

CAT uses the certified `DIRECT-NEW` trust-region subproblem solver. The `OLD`
implementation remains available only for reproducing the conference-version
ablation:

```julia
set_attribute(
    model,
    "algorithm_params!trust_region_subproblem_solver",
    "OLD",
)
```

The algorithm counter reports factorization counts and retains zero-valued
subproblem-iteration and Hessian-vector-product fields for a consistent result
interface.

### Direct API

`CAT_solve(model[, termination_criteria, algorithm_params])` returns
`(x, status, iteration_stats, algorithm_counter, iterations, execution_time,
best_gradient_x)`. The first six entries retain their historical positions.
`x` is the current accepted iterate; `best_gradient_x` is the evaluated point
paired with the history's `min_gradnorm_fval` and `min_gradnorm` values.

### Benchmarks

The CUTEst benchmark pipeline, frozen profiles, and ablation studies live in the
standalone
[`benchmark-unconstrained-optimization-solvers`](../benchmark-unconstrained-optimization-solvers/README.md)
project.

The trust-region subproblem implementation and its tests live in
[`trust-region-subproblem-solvers`](../trust-region-subproblem-solvers/README.md).
CAT calls that package through a one-way dependency.

### Examples

Examples can be found under the [test directory](test).

## References

* [Hamad, Fadi, and Oliver Hinder. "A simple and practical adaptive trust-region method."](https://arxiv.org/abs/2412.02079)
* [Hamad, Fadi, and Oliver Hinder. "A consistently adaptive trust-region method."](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2c19666cbb2c14d45d39e2dcf6ab0b99-Abstract-Conference.html)

## Citing

If you use our method in your research, you are kindly asked to cite the relevant papers:

```raw
@article{hamad2024simple,
  title={A simple and practical adaptive trust-region method},
  author={Hamad, Fadi and Hinder, Oliver},
  journal={arXiv preprint arXiv:2412.02079},
  year={2024}
}

@article{hamad2022consistently,
  title={A consistently adaptive trust-region method},
  author={Hamad, Fadi and Hinder, Oliver},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={6640--6653},
  year={2022}
}
```
