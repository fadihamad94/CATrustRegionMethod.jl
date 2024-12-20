# CATrustRegionMethod

This package implements a trust-region method for unconstrained optimization: 

$$\min_{x \in \mathbb{R}^n} f(x).$$

The method finds stationary points, specifically points with $|| \nabla f(x) || \leq \epsilon$. In particular, in our paper, we show that the method achieves the best possible convergence bound up to an additive logarithmic factor, for finding an $\epsilon$-approximate stationary point, namely $O( \Delta_f L^{1/2} \epsilon^{-3/2}) + \tilde{O}(1)$ iterations, where $L$ is the Lipschitz constant of the Hessian, $\Delta_f$ is the optimality gap, and $\epsilon$ is the termination tolerance for the gradient norm."

Consistently adaptive (CA) in the package name refers to the method achieving the best possible convergence bound without requiring knowledge of the Lipschitz constant ($L$) of the Hessian.

## License

CATrustRegionMethod.jl is licensed under the [MIT License](https://github.com/fadihamad94/CAT-Journal/blob/master/LICENSE).

## Installation

Install CATrustRegionMethod as follows:

```julia
import Pkg
Pkg.add("CATrustRegionMethod")
```

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

### CUTEst test set

To test our solver on CUTEst test set, please use the script:

```julia
solve_cutest.jl
```

To see the meaning of each argument:

```shell
$ julia --project=. scripts/solve_cutest.jl --help
```

Here is a simple example:

```shell
$ julia --project=. scripts/solve_cutest.jl --output_dir ./scripts/benchmark/results/cutest --default_problems true
```

### Plots for CUTEst test set

```shell
$ julia --project=. scripts/plot_CUTEst_results.jl --output_dir ./scripts/benchmark/results/cutest
```

## Instructions for reproducing our experiments

### CUTEst test set

```shell
$ julia --project=. scripts/solve_cutest.jl --output_dir ./scripts/benchmark/results/cutest --default_problems true
```

```shell
$ julia --project=. scripts/solve_cutest.jl --output_dir ./scripts/benchmark/results/cutest --default_problems true --θ 0.0
```

```shell
$ julia --project=. scripts/run_ablation_study.jl --output_dir ./scripts/benchmark/results_ablation_study/cutest --default_problems true
```

### Examples

Examples can be found under the [test directory](https://github.com/fadihamad94/CAT-Journal/tree/master/test)

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
