# CATNeurIPS

This package preserves the outer CAT algorithm from the public
[`CAT-NeurIPS`](https://github.com/fadihamad94/CAT-NeurIPS) repository for
same-protocol numerical comparisons with the current solver.
The adaptation is based on commit
`905e4976c0d6384456ebf522b871559c30a0e878` from its `main` branch.

The original repository bundled its own trust-region subproblem code. This
adaptation deliberately delegates every subproblem to the maintained
`TrustRegionSubproblemSolvers.solveTrustRegionSubproblemOldApproach` backend,
so the conference outer method and the `conference-subproblem-solver` ablation
share one implementation of the old subproblem algorithm.

The benchmark selector is `CAT-NeurIPS`.
