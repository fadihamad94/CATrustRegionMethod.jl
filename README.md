# CAT
This package is the implementation of a consistently **a**daptive **t**rust-region method (called CAT) for finding stationary points of nonconvex functions with L-Lipschitz Hessians and bounded optimality gap.

## One-time setup
Install Julia 1.6.0 or later. From the root directory of the repository, run:

```console
$ julia --project=scripts -e 'import Pkg; Pkg.instantiate()'
```

Validate setup by running the unit tests:

```console
$ julia --project=scripts test/run_tests.jl
```

## Running
### Learning linear dynamical systems
To test our solver on learning linear dynamical system, please use the script:

```julia
solve_learning_problem.jl
```

To see the meaning of each argument:

```console
$ julia --project=scripts scripts/solve_learning_problem.jl --help
```

Here is a simple example:

```console
$ julia --project=scripts scripts/solve_learning_problem.jl --output_dir ./scripts/benchmark/results/learning_problem --d 3 --T 5 --σ 0.1 --instances 5
```

## Matrix completion problem
To test our solver on matrix completion problem, please use the script:

```julia
solve_matrix_completion.jl
```

To see the meaning of each argument:

```shell
$ julia --project=scripts scripts/solve_matrix_completion.jl --help
```

Here is a simple example:

```shell
$ julia --project=scripts scripts/solve_matrix_completion.jl --output_dir ./scripts/benchmark/results/matrix_completion --λ1 0.001 --λ2 0.001 --instances 5
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
$ julia --project=scripts scripts/solve_cutest.jl --output_dir ./scripts/benchmark/results/cutest --default_problems true --solver CAT
```

### Complexity hard example

To test our solver on the complexity hard example from Cartis et.al paper "On the complexity of steepest descent, newton’s and regularized newton’s methods for nonconvex unconstrained optimization problems", please use the script:

```shell
solve_hard_example.jl
```

To see the meaning of each argument:

```shell
$ julia --project=scripts scripts/solve_hard_example.jl --help
```

Here is a simple example:

```shell
$ julia --project=scripts scripts/solve_hard_example.jl --output_dir ./scripts/benchmark/results/hard_complexity_example --tol_opt 1e-3 --r_1 1.5
```

### Plots for CUTEst test set
```shell
$ julia --project=scripts scripts/plot_CUTEst_results.jl --output_dir ./scripts/benchmark/results/cutest
```

## Instructions for reproducing our experiments
### Learning linear dynamical systems

```console
$ julia --project=scripts scripts/solve_learning_problem.jl --output_dir ./scripts/benchmark/results/learning_problem --d 4 --T 50 --σ 0.01 --instances 60
```

## Matrix completion problem

```shell
$ julia --project=scripts scripts/solve_matrix_completion.jl --output_dir ./scripts/benchmark/results/matrix_completion --λ1 0.01 --λ2 0.01 --instances 10
```

### CUTEst test set

```shell
$ julia --project=scripts scripts/solve_cutest.jl --output_dir ./scripts/benchmark/results/cutest --default_problems true --solver CAT
```

```shell
$ julia --project=scripts scripts/solve_cutest.jl --output_dir ./scripts/benchmark/results/cutest --default_problems true --solver CAT --θ 0.0
```

```shell
$ julia --project=scripts scripts/solve_cutest.jl --output_dir ./scripts/benchmark/results/cutest --default_problems true --solver NewtonTrustRegion
```

### Complexity hard example

```shell
$ julia --project=scripts scripts/solve_hard_example.jl --output_dir ./scripts/benchmark/results/hard_complexity_example --tol_opt 1e-3 --r_1 1.5
```
