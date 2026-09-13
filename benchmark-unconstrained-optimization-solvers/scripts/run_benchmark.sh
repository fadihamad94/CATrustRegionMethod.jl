#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_benchmark.sh PROFILE --solver CAT|CAT-NeurIPS|UTR --results PATH [OPTIONS]
  scripts/run_benchmark.sh PROFILE --ablation --ablation-variant SLUG --results PATH [OPTIONS]

Profiles (choose exactly one):
  --full | --fast | --very_fast | --super_fast

Options:
  --solver NAME              Run CAT, CAT-NeurIPS, or UTR
  --subproblem-solver NAME   DIRECT-NEW (default); CAT-NeurIPS uses OLD
  --gamma-1 FLOAT            Override gamma_1 with a value in (0, 1)
  --ablation                 Run one CAT ablation variant
  --ablation-variant SLUG    original, rho-hat-rule, radius-update-rule,
                             initial-radius, conference-subproblem-solver,
                             xi-zero, or b-k-zero
  --seed INTEGER             Nonnegative random seed (default: 1)
  --print-level INTEGER      Solver print level; -1 suppresses output (default: 0)
  --threads INTEGER          Positive thread count (default: benchmark/defaults.json)
  --manual_skip PROBLEM...   Explicitly omit problems; may be repeated
  --results PATH             Output directory (required)
  --help                     Show this help
EOF
}

problem_set=""
kind="benchmark"
solver=""
ablation_variant=""
subproblem_solver="DIRECT-NEW"
subproblem_solver_seen=0
gamma_1=""
seed=1
print_level=0
number_of_threads=""
results_folder=""
manual_skips=()

select_problem_set() {
    [[ -z "$problem_set" ]] || { echo "Choose exactly one profile." >&2; exit 2; }
    problem_set="$1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full|--fast|--very_fast|--super_fast)
            select_problem_set "${1#--}"; shift ;;
        --solver)
            [[ $# -ge 2 ]] || { echo "Missing value for --solver." >&2; exit 2; }
            solver="$2"; shift 2 ;;
        --subproblem-solver)
            [[ $# -ge 2 ]] || { echo "Missing value for --subproblem-solver." >&2; exit 2; }
            subproblem_solver="$2"; subproblem_solver_seen=1; shift 2 ;;
        --gamma-1)
            [[ $# -ge 2 ]] || { echo "Missing value for --gamma-1." >&2; exit 2; }
            gamma_1="$2"; shift 2 ;;
        --ablation)
            kind="ablation"; shift ;;
        --ablation-variant)
            [[ $# -ge 2 ]] || { echo "Missing value for --ablation-variant." >&2; exit 2; }
            ablation_variant="$2"; shift 2 ;;
        --seed)
            [[ $# -ge 2 ]] || { echo "Missing value for --seed." >&2; exit 2; }
            seed="$2"; shift 2 ;;
        --print-level)
            [[ $# -ge 2 ]] || { echo "Missing value for --print-level." >&2; exit 2; }
            print_level="$2"; shift 2 ;;
        --threads)
            [[ $# -ge 2 ]] || { echo "Missing value for --threads." >&2; exit 2; }
            number_of_threads="$2"; shift 2 ;;
        --manual_skip)
            shift
            [[ $# -gt 0 && "$1" != --* ]] || {
                echo "--manual_skip requires at least one problem." >&2; exit 2;
            }
            while [[ $# -gt 0 && "$1" != --* ]]; do
                manual_skips+=("$1"); shift
            done ;;
        --results)
            [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for --results." >&2; exit 2; }
            results_folder="$2"; shift 2 ;;
        --help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$problem_set" ]] || { echo "Choose exactly one profile." >&2; exit 2; }
[[ -n "$results_folder" ]] || { echo "--results is required." >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "--seed must be nonnegative." >&2; exit 2; }
[[ "$print_level" =~ ^-1$|^[0-9]+$ ]] || {
    echo "--print-level must be -1 or nonnegative." >&2; exit 2;
}
if [[ -n "$number_of_threads" ]]; then
    [[ "$number_of_threads" =~ ^[1-9][0-9]*$ ]] || {
        echo "--threads must be positive." >&2; exit 2;
    }
fi
[[ "$subproblem_solver" == "DIRECT-NEW" || "$subproblem_solver" == "OLD" ]] || {
    echo "--subproblem-solver must be DIRECT-NEW or OLD." >&2; exit 2;
}
if [[ -n "$gamma_1" ]]; then
    [[ "$gamma_1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] &&
        awk -v value="$gamma_1" 'BEGIN { exit !(value > 0 && value < 1) }' || {
            echo "--gamma-1 must lie in (0, 1)." >&2; exit 2;
        }
fi

if [[ "$kind" == "benchmark" ]]; then
    [[ "$solver" == "CAT" || "$solver" == "CAT-NeurIPS" || "$solver" == "UTR" ]] || {
        echo "--solver must be CAT, CAT-NeurIPS, or UTR." >&2; exit 2;
    }
    [[ -z "$ablation_variant" ]] || {
        echo "--ablation-variant requires --ablation." >&2; exit 2;
    }
    if [[ "$solver" == "CAT-NeurIPS" ]]; then
        [[ "$subproblem_solver_seen" == 0 || "$subproblem_solver" == "OLD" ]] || {
            echo "CAT-NeurIPS requires --subproblem-solver OLD." >&2; exit 2;
        }
        [[ -z "$gamma_1" ]] || { echo "--gamma-1 is unavailable for CAT-NeurIPS." >&2; exit 2; }
        subproblem_solver="OLD"
    else
        [[ "$subproblem_solver" == "DIRECT-NEW" ]] || {
            echo "$solver requires --subproblem-solver DIRECT-NEW." >&2; exit 2;
        }
    fi
else
    [[ -z "$solver" ]] || { echo "--solver is unavailable with --ablation." >&2; exit 2; }
    case "$ablation_variant" in
        original|rho-hat-rule|radius-update-rule|initial-radius|conference-subproblem-solver|xi-zero|b-k-zero) ;;
        *) echo "Unknown or missing ablation variant: $ablation_variant" >&2; exit 2 ;;
    esac
    [[ "$subproblem_solver" == "DIRECT-NEW" ]] || {
        echo "Ablation runs require --subproblem-solver DIRECT-NEW." >&2; exit 2;
    }
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$results_folder" == /* ]] || results_folder="$PWD/$results_folder"
results_folder="${results_folder%/}"
julia_bin="${JULIA_BIN:-julia}"
if [[ -z "$number_of_threads" ]]; then
    number_of_threads="$(
        "$julia_bin" --startup-file=no --threads=1,0 --gcthreads=1,0 \
            --project="$repo_root" "$repo_root/benchmark/benchmark_defaults.jl" num_threads
    )"
fi

export CAT_BENCHMARK_STARTUP_FILE=no
export JULIA_NUM_THREADS="$number_of_threads,0"
export JULIA_NUM_GC_THREADS="$number_of_threads,0"
export OMP_NUM_THREADS="$number_of_threads"
export OMP_THREAD_LIMIT="$number_of_threads"
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS="$number_of_threads"
export OPENBLAS_DEFAULT_NUM_THREADS="$number_of_threads"
export GOTO_NUM_THREADS="$number_of_threads"
export MKL_NUM_THREADS="$number_of_threads"
export MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=$number_of_threads"
export MKL_DYNAMIC=FALSE
export VECLIB_MAXIMUM_THREADS="$number_of_threads"
export BLIS_NUM_THREADS="$number_of_threads"
export BLIS_NT="$number_of_threads"
unset BLIS_JC_NT BLIS_PC_NT BLIS_IC_NT BLIS_JR_NT BLIS_IR_NT BLIS_THREAD_IMPL BLIS_TI

worker="$repo_root/scripts/benchmark_worker.jl"
worker_args=(
    --kind "$kind" --problem-set "$problem_set" --results "$results_folder"
    --seed "$seed" --print-level "$print_level"
    --number-of-threads "$number_of_threads"
    --subproblem-solver "$subproblem_solver"
)
[[ -z "$gamma_1" ]] || worker_args+=(--gamma-1 "$gamma_1")
if [[ "$kind" == "benchmark" ]]; then
    worker_args+=(--solver "$solver")
else
    worker_args+=(--ablation-variant "$ablation_variant")
fi
for problem in "${manual_skips[@]}"; do
    worker_args+=(--manual-skip "$problem")
done

run_worker() {
    "$julia_bin" --startup-file=no --threads="$number_of_threads,0" \
        --gcthreads="$number_of_threads,0" --project="$repo_root" \
        "$worker" "$@"
}

run_worker prepare "${worker_args[@]}"
pending_file="$results_folder/pending_problems.tsv"
while IFS=$'\t' read -r variant batch_index problem; do
    [[ "$variant" == "variant" || -z "$variant" ]] && continue
    echo "Running $variant/$problem in a fresh Julia process."
    invocation_start="$(date +%s)"
    run_worker run "${worker_args[@]}" --variant "$variant" \
        --batch-index "$batch_index" --problem "$problem" \
        --invocation-start "$invocation_start"
done < "$pending_file"
run_worker aggregate "${worker_args[@]}"
echo "Benchmark complete: $results_folder"
