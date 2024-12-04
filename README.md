# CAT
This package is the implementation of a simple and practical adaptive trust-region method for finding stationary points of nonconvex functions with L-Lipschitz Hessians and bounded optimality gap.

## License
CAT.jl is licensed under the [MIT License](https://github.com/fadihamad94/CAT-Journal/blob/master/LICENSE).

## Installation
Installing the CAT solver can be done in two different ways:

### Install CAT as a package
Install Julia 1.10.4 or later. CAT can be installed and tested through the Julia package manager:

```julia
julia> ]
pkg> add https://github.com/fadihamad94/CAT-Journal.git
pkg> test CAT
```

### One-time setup
Install Julia 1.10.4 or later. From the root directory of the repository, run:

```console
$ julia --project=scripts -e 'import Pkg; Pkg.instantiate()'
```

Validate setup by running the unit tests:

```console
$ julia --project=scripts test/runtests.jl
```

## Running

### How to use with JuMP
Here is a simple example where a JuMP model is passed to the CAT solver
```julia
using CAT, JuMP
model = Model()
@variable(model, x)
@variable(model, y)
@NLobjective(model, Min, (2.0 - x)^2 + 100 * (y - x^2)^2)
set_optimizer(model, CAT.CATSolver)
MOI.set(model, MOI.RawOptimizerAttribute("time_limit"), 1800.0)
MOI.set(model, MOI.RawOptimizerAttribute("algorithm_params!r_1"), 100.0)
optimize!(model)
status = MOI.get(model, MOI.TerminationStatus())
# Retrieve the solver instance
optimizer = backend(model).optimizer.model
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
$ julia --project=scripts scripts/solve_cutest.jl --help
```

Here is a simple example:

```shell
$ julia --project=scripts scripts/solve_cutest.jl --output_dir ./scripts/benchmark/results/cutest --default_problems true
```

### Plots for CUTEst test set
```shell
$ julia --project=scripts scripts/plot_CUTEst_results.jl --output_dir ./scripts/benchmark/results/cutest
```

## Instructions for reproducing our experiments

### CUTEst test set

```shell
$ julia --project=scripts scripts/solve_cutest.jl --output_dir ./scripts/benchmark/results/cutest --default_problems true
```

```shell
$ julia --project=scripts scripts/solve_cutest.jl --output_dir ./scripts/benchmark/results/cutest --default_problems true --θ 0.0
```

```shell
$ julia --project=scripts scripts/run_ablation_study.jl --output_dir ./scripts/benchmark/results_ablation_study/cutest --default_problems true
```

### Examples
Examples can be found under the [test directory](https://github.com/fadihamad94/CAT-Journal/tree/master/test)

## References
* [Hamad, Fadi, and Oliver Hinder. "A simple and practical adaptive trust-region method."](https://arxiv.org/abs/2412.02079)
* [Hamad, Fadi, and Oliver Hinder. "A consistently adaptive trust-region method."](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2c19666cbb2c14d45d39e2dcf6ab0b99-Abstract-Conference.html)
