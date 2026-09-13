# Reproducing the GALAHAD comparisons

GALAHAD and HSL are not distributed with this repository. The paper's
external comparison runs used GALAHAD 5.5.1 at commit
`e5b872c27f2f8d597b91ef75896e8f02c0550ddc` and HSL 4.0.6 through
`HSL_jll` 4.0.6+0. That HSL build came from `libHSL` commit
`def3910679b2940eabcebdd64ce126554059df95`; the Julia wrapper tree recorded
in the original manifest was `4c7f8674babdebebb99ba9b20686711c5f48a395`.
Direct TRU used the licensed HSL MA57 sparse linear solver.

## Install GALAHAD and HSL

1. Clone GALAHAD and check out the exact paper revision:

   ```shell
   git clone https://github.com/ralna/GALAHAD.git
   git -C GALAHAD checkout --detach e5b872c27f2f8d597b91ef75896e8f02c0550ddc
   ```

   This commit is the GALAHAD 5.5.1 release.
2. Request and download HSL MA57 from the
   [HSL website](https://www.hsl.rl.ac.uk/download/hsl-galahad/latest/).
   Obtain HSL 4.0.6 (the version and source revision pinned above), choose the
   double-precision HSL subset, accept its licence, and unpack it outside this
   repository. The `latest` URL is only the download entry point; do not
   substitute a newer release without recording the resulting deviation. HSL
   is separately licensed and cannot be redistributed with this replication
   package.
3. Obtain ARCHDefs, SIFDecode, CUTEst, and MASTSIF as described in the
   [CUTEst installation guide](https://github.com/ralna/CUTEst/wiki). Set
   `ARCHDEFS`, `SIFDECODE`, and `CUTEST` to those source directories and set
   `GALAHAD` to the GALAHAD checkout.
4. Copy `GALAHAD/src/makedefs/packages.default` to
   `GALAHAD/src/makedefs/packages`. In the copy, set `HSLSUBSET` to the
   extracted HSL subset's `src` directory and set `HSLARCHIVESUBSET` to
   `$(HSLSUBSET)`. With a real HSL subset configured, that file selects
   `LINEARSOLVER = ma57`.
5. From the GALAHAD checkout, run `$ARCHDEFS/install_optrove`, select a
   double-precision build with the C interfaces and CUTEst support, and answer
   its compiler/platform prompts. The command prints the generated
   `machine.os.compiler` identifier and the environment settings for the
   installed libraries.
6. From `GALAHAD/src`, verify the C interfaces with:

   ```shell
   make -f "$GALAHAD/makefiles/machine.os.compiler" test_all_ciface
   ```

   Replace `machine.os.compiler` with the identifier produced by the install.
   Confirm from GALAHAD's information structure that a direct TRU run resolves
   both requested symmetric and definite linear solvers to `ma57`; do not
   silently substitute another solver.
7. Use the same problem names and starting points as the frozen `full` profile
   in `benchmark-unconstrained-optimization-solvers/benchmark/run-profiles.json`.

The retained Julia experiments do not depend on this installation. A separate
GALAHAD driver is required because this repository intentionally omits all
GALAHAD adapter and bridge code.

## Driver behavior used for the paper

Implement the GALAHAD TRU and ARC comparisons through their C
reverse-communication interfaces. For direct variants, obtain the lower
triangle of each CUTEst Hessian in sparse coordinate form and set
`hessian_available = true` and `subproblem_direct = true`. For iterative
variants, set both controls to `false` and answer Hessian-vector requests with
CUTEst products. The paper's direct TRU configuration explicitly requested
MA57; direct ARC retained GALAHAD's initialized sparse linear-solver control.

Use the following common stopping and resource settings:

| Setting | Value |
| --- | ---: |
| absolute gradient tolerance (`stop_g_absolute`) | `1e-5` |
| relative gradient tolerance (`stop_g_relative`) | `0` |
| step tolerance (`stop_s`) | `2e-16` |
| norm control (`norm`) | `1` |
| maximum outer iterations (`maxit`) | `100000` |
| wall-clock limit (`clock_time_limit`) | `18000` seconds |

Disable the CPU-time limit, use the wall-clock limit above, and retain
GALAHAD's other initialized controls. Run every CUTEst problem in a fresh
process so problem state and resource measurements are isolated.

After a solver reports success, create a fresh CUTEst model at the returned
point and accept the observation only if the Euclidean gradient norm is finite
and at most `1e-5`. Perform this check after taking the timed resource snapshot;
its time and evaluations are not part of the solver's measurements.

Record objective, gradient norm, final status, iterations, elapsed time, and
CUTEst function, gradient, Hessian, and Hessian-vector evaluation counts. Use
CUTEst's report for evaluation counts. For direct TRU and ARC, reproduce the
paper's factorization metric by rounding
`inform.factorization_average * inform.iter` to the nearest integer. For
iterative variants, report zero direct factorizations and record GALAHAD's
cumulative Krylov/CG iterations as subproblem iterations.