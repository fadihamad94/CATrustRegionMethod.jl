# CUTEst replication benchmark

This project reproduces the paper's numerical results for CAT, CAT-NeurIPS,
UTR, and the seven retained CAT ablations. It contains a frozen 125-problem
CUTEst profile and smaller development profiles.

## Install

Use Julia 1.12 or later. From this directory:

```shell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
```

`CUTEst.jl` and SIFDecode are installed through the Julia environment. A local
compiler toolchain may be needed when a problem is decoded for the first time.
The CAT, CAT-NeurIPS, and UTR experiments in this repository do not require
GALAHAD or HSL. External comparison setup is documented in
[`../modify_GALAHAD.md`](../modify_GALAHAD.md).

## Run a solver

Each problem runs serially in a fresh Julia process. The output directory may
be resumed with the identical command and environment.

```shell
./scripts/run_benchmark.sh --super_fast --solver CAT \
  --results results/cat-smoke

./scripts/run_benchmark.sh --full --solver CAT \
  --results results/paper/cat
./scripts/run_benchmark.sh --full --solver CAT-NeurIPS \
  --results results/paper/cat-neurips
./scripts/run_benchmark.sh --full --solver UTR \
  --results results/paper/utr
```

Available profiles are `super_fast` (30 problems), `very_fast` (94), `fast`
(102), and `full` (125). The full profile sets a limit of 100,000 outer
iterations and 18,000 seconds per problem. The default seed and thread count
are both `1`. See `./scripts/run_benchmark.sh --help` for overrides and manual
problem exclusions.

## Run the ablations

The retained ablation slugs are:

| Slug | Change from the current CAT configuration |
| --- | --- |
| `original` | no change; reference CAT configuration |
| `rho-hat-rule` | conference acceptance-ratio rule |
| `radius-update-rule` | conference radius-update rule |
| `initial-radius` | fixed initial radius of 1 |
| `conference-subproblem-solver` | conference-version subproblem solver |
| `xi-zero` | set xi to zero |
| `b-k-zero` | set xi and the evaluation offset to zero |

Example:

```shell
./scripts/run_benchmark.sh --full --ablation \
  --ablation-variant conference-subproblem-solver \
  --results results/paper/ablation-conference-subproblem-solver
```

## Outputs and validation

The runner writes:

- `run_settings.json`: numerical settings, environment provenance, and a
  source fingerprint;
- `raw/<variant>/<problem>.json`: one record per problem;
- `logs/<variant>/<problem>.log`: solver output;
- `<variant>/table_cutest_<variant>.csv`: aggregate rows;
- `run_summary.json` or `ablation_summary.json`: status counts and summary
  statistics.

After a solver claims `OPTIMAL`, the pipeline opens a fresh CUTEst model and
requires the Euclidean gradient norm to be at most `1e-5`. Validation time and
evaluations are excluded from solver timing and counters. Failed observations
receive uniform penalties of twice the configured iteration or time limit in
the aggregate statistics.
