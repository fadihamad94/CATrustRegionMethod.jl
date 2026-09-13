# CAT paper replication code

This repository contains the code to reproduce `A simple and practical adaptive trust-region method by Fadi Hamad and Oliver Hinder'
https://arxiv.org/abs/2412.02079

## Repository layout

- `CAT-solver/`: the current CAT method and its JuMP/MathOptInterface wrapper.
- `CAT-NeurIPS/`: the conference-version CAT outer method.
- `UTR/`: the universal trust-region method.
- `trust-region-subproblem-solvers/`: the current direct subproblem solver and
  the conference-version `OLD` implementation.
- `shared_code/`: shared solver infrastructure.
- `benchmark-unconstrained-optimization-solvers/`: the pinned CUTEst problem
  profiles and serial experiment runner.

The benchmark project depends on the other five local projects. Julia's
`[sources]` entries resolve those dependencies directly from this checkout.

## Requirements and installation

Install Julia 1.12 or later, clone this repository, and instantiate each
environment from the repository root:

```shell
julia --project=shared_code -e 'using Pkg; Pkg.instantiate()'
julia --project=trust-region-subproblem-solvers -e 'using Pkg; Pkg.instantiate()'
julia --project=CAT-solver -e 'using Pkg; Pkg.instantiate()'
julia --project=CAT-NeurIPS -e 'using Pkg; Pkg.instantiate()'
julia --project=UTR -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark-unconstrained-optimization-solvers \
  -e 'using Pkg; Pkg.instantiate()'
```

The benchmark environment installs `CUTEst.jl`, SIFDecode, and their Julia
artifacts. The first CUTEst run downloads and compiles the selected problem,
so a Fortran/C toolchain may be required by the platform's CUTEst setup.

The retained CAT, CAT-NeurIPS, and UTR implementations do not require GALAHAD
or HSL. To reproduce the paper's external GALAHAD TRU/ARC comparisons, obtain
GALAHAD separately and obtain the licensed HSL MA57 sources from the
[HSL website](https://www.hsl.rl.ac.uk/). Follow the HSL licence terms, install
MA57 where the GALAHAD build can find it, and use the configuration recorded in
[`modify_GALAHAD.md`](modify_GALAHAD.md). Neither package is redistributed here.

## Tests

Run the permanent suites serially:

```shell
julia --project=shared_code -e 'using Pkg; Pkg.test()'
julia --project=trust-region-subproblem-solvers -e 'using Pkg; Pkg.test()'
julia --project=CAT-solver -e 'using Pkg; Pkg.test()'
julia --project=CAT-NeurIPS -e 'using Pkg; Pkg.test()'
julia --project=UTR -e 'using Pkg; Pkg.test()'
julia --project=benchmark-unconstrained-optimization-solvers \
  benchmark-unconstrained-optimization-solvers/test/runtests.jl
```

## Reproducing the numerical results

Each command runs CUTEst problems serially in fresh Julia processes. Start
with `--super_fast`, then use `--full` for the pinned 125-problem paper set:

```shell
cd benchmark-unconstrained-optimization-solvers
./scripts/run_benchmark.sh --super_fast --solver CAT \
  --results results/cat-smoke

./scripts/run_benchmark.sh --full --solver CAT \
  --results results/paper/cat
./scripts/run_benchmark.sh --full --solver CAT-NeurIPS \
  --results results/paper/cat-neurips
./scripts/run_benchmark.sh --full --solver UTR \
  --results results/paper/utr
```

Run the seven paper ablations separately:

```shell
for variant in original rho-hat-rule radius-update-rule initial-radius \
  conference-subproblem-solver xi-zero b-k-zero; do
  ./scripts/run_benchmark.sh --full --ablation \
    --ablation-variant "$variant" \
    --results "results/paper/ablation-$variant"
done
```

The default seed is `1`; the full profile uses at most 100,000 outer
iterations and 18,000 seconds per problem. Use `--threads N`, `--seed N`, or
`--manual_skip NAME...` when needed. A resumed command reuses valid raw JSON
files only when its recorded environment and numerical settings match.

Each results directory contains `run_settings.json`, per-problem JSON under
`raw/`, logs, an aggregate CSV under the method directory, and either
`run_summary.json` or `ablation_summary.json`. Successful solver claims are
checked with a fresh CUTEst gradient evaluation at tolerance `1e-5` outside
the recorded solver time.
