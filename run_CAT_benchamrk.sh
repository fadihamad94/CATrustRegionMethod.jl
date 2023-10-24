echo $1
echo $2
echo $3
../julia-1.7.2/bin/julia --project=./scripts ../CAT-Journal/scripts/solve_cutest.jl --output_dir ./results_benchmark_paper_extension_reproduce_results/CUTEst --max_it 10000 --solver $1 --max_time $2 --default_problems false --min_nvar 1 --max_nvar $3