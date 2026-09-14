# Data used by the paper figures

This directory contains only the numerical data used to produce the ten figures
included in `CAT-Journal_Paper/SIAM/CAT.tex`: six performance profiles and
four ablation plots. The selection was traced against
`benchmark/report-configs/paper-main.json` and
`benchmark/report-configs/paper-combined-ablation.json` on commit
`6e74e175d6ca73c85c3758722a0d56a4137ee425` of the `siam-review` branch.

The source experiments are collected in
`revision-full-2026-v8-arc-ma57-corrected`. They use the same 125-problem
unconstrained CUTEst set, random seed 1, a limit of 100,000 outer iterations,
and a solver time limit of 18,000 seconds per problem. Claimed successful
solutions were checked independently with a fresh CUTEst gradient evaluation
at tolerance `1e-5`. Validation time and evaluations are excluded from the
recorded solver measurements.

## Performance-profile data

The five files in `benchmarks/` provide the series used by the six
performance-profile figures:

| File | Figure label | Configuration |
| --- | --- | --- |
| `cat-direct-new.csv` | This paper | CAT with the repository's `DIRECT-NEW` subproblem solver |
| `cat-neurips.csv` | Conference version | Conference-version CAT with the `OLD` subproblem solver |
| `utr-direct-new.csv` | UTR | UTR with the repository's `DIRECT-NEW` solver |
| `arc-direct.csv` | ARC | GALAHAD ARC with its direct RQS subproblem solver and HSL MA57 |
| `tru-direct.csv` | TRU | GALAHAD TRU with the direct TRS solver and MA57 |

All five series are used for the function-evaluation, gradient-evaluation,
Hessian-evaluation, wall-clock-time, and terminal-objective profiles. The
factorization profile uses every series except the conference version, which
does not expose a compatible factorization count.

Each CSV contains one header and 125 problem rows with these columns:

| Column | Meaning |
| --- | --- |
| `problem_name` | CUTEst problem name |
| `status` | Normalized final solver status |
| `total_execution_time` | Solver time in seconds |
| `terminal_objective_value` | Objective at the final accepted iterate, used by the paper's objective profile |
| `total_function_evaluation` | Objective evaluations |
| `total_gradient_evaluation` | Gradient evaluations |
| `total_hessian_evaluation` | Hessian evaluations |
| `total_factorization_evaluation` | Direct factorizations |

The resource profiles plot, for each budget on the horizontal axis, the
fraction of the 125 problems solved within that budget. The objective profile
uses the terminal objective value for every method.

## Ablation-figure data

The six per-problem CSVs in `ablations/` contain exactly the configurations
plotted in the four ablation panels. They have the same 125 rows and eight
columns as the performance-profile CSVs.

| File | Figure label |
| --- | --- |
| `original.csv` | This paper |
| `rho-hat-rule.csv` | Classical rho rule |
| `radius-update-rule.csv` | Conference radius update |
| `b-k-zero.csv` | Set `b_k = 0` |
| `initial-radius.csv` | Fixed initial radius |
| `conference-subproblem-solver.csv` | Conference subproblem solver |

`ablations/summary.csv` has one row per configuration. Its four
`shifted_geomean_*` columns correspond directly to the paper figures for
wall-clock time and function, gradient, and Hessian evaluations. Each value is
the shifted geometric mean over all 125 problems, with shift 1:

```math
\exp\left(\frac{1}{n}\sum_{i=1}^n \log(x_i+1)\right)-1.
```

A non-successful observation receives a penalty of 36,000 seconds for runtime
or 200,000 for an evaluation count before the statistic is computed.

## Statuses and missing values

`OPTIMAL` means that the solver reported success and passed the independent
gradient check. Other statuses identify iteration, time, step-size, subproblem,
or numerical termination. `NaN` means that a value was not exposed by the
corresponding solver interface. Runtime comparisons are machine-dependent and
should be interpreted together with evaluation counts.

## What was omitted

Configurations not referenced by the manuscript figures were removed:
iterative ARC, CAT with GALAHAD direct, CAT with iterative GLTR, iterative TRU,
the GALAHAD-direct ablation, and the `xi-zero` ablation. The last of these
appears in a reviewer-response table, but not in a figure.

GALAHAD and the licensed HSL MA57 sources are not distributed in this
repository. See [`modify_GALAHAD.md`](../modify_GALAHAD.md) for the external
comparison setup.
