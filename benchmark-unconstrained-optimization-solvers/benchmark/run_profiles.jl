module BenchmarkRunProfiles

using JSON

const DEFAULT_RUN_PROFILES_FILE = joinpath(@__DIR__, "run-profiles.json")
const RUN_PROFILES_SCHEMA_VERSION = 5

struct ProblemSelection
    name::String
    batches::Vector{Vector{String}}
    problems::Vector{String}
    max_iterations::Int64
    max_time_seconds::Float64
    skipped_problems::Vector{String}
end

function profile_batches(profile::AbstractDict)::Vector{Vector{String}}
    return [String.(batch) for batch in profile["batches"]]
end

function profile_problems(profile::AbstractDict)::Vector{String}
    return sort!(vcat(profile_batches(profile)...))
end

function validate_run_profiles(data::AbstractDict)::Nothing
    get(data, "schema_version", nothing) == RUN_PROFILES_SCHEMA_VERSION ||
        error("Unsupported run-profiles schema.")
    profiles = data["profiles"]
    batching = data["batching"]
    default_max_iterations = Int(data["default_max_iterations"])
    default_max_iterations > 0 || error("The default iteration limit must be positive.")
    default_batch_size = Int(batching["default_batch_size"])
    profile_batch_sizes = batching["profile_batch_sizes"]
    single_batch_profiles = Set(String.(batching["single_batch_profiles"]))
    singleton_problems = Set(String.(batching["singleton_problems"]))

    for (raw_name, profile) in profiles
        name = String(raw_name)
        batches = profile_batches(profile)
        isempty(batches) && error("Profile '$name' has no batches.")
        maximum_batch_size = Int(get(profile_batch_sizes, name, default_batch_size))
        name in single_batch_profiles && length(batches) != 1 &&
            error("Profile '$name' must contain exactly one batch.")
        for (index, batch) in enumerate(batches)
            isempty(batch) && error("Profile '$name' batch $index is empty.")
            length(batch) <= maximum_batch_size ||
                error("Profile '$name' batch $index exceeds $maximum_batch_size problems.")
            issorted(batch) || error("Profile '$name' batch $index must be sorted.")
        end
        problems = vcat(batches...)
        length(problems) == length(unique(problems)) ||
            error("Profile '$name' contains duplicate problems.")
        Int(profile["problem_count"]) == length(problems) ||
            error("Profile '$name' has an incorrect problem_count.")
        max_time_seconds = Float64(profile["max_time_seconds"])
        isfinite(max_time_seconds) && max_time_seconds > 0.0 ||
            error("Profile '$name' must have a positive time limit.")
        if name ∉ single_batch_profiles
            for problem in intersect(Set(problems), singleton_problems)
                [problem] in batches || error("Profile '$name' must isolate '$problem'.")
            end
        end
    end

    subset_chain = String.(data["subset_chain"])
    for index in 1:(length(subset_chain) - 1)
        smaller, larger = subset_chain[index], subset_chain[index + 1]
        issubset(
            Set(profile_problems(profiles[smaller])),
            Set(profile_problems(profiles[larger])),
        ) || error("Profile '$smaller' must be a subset of '$larger'.")
    end

    return nothing
end

function load_run_profiles(path::String = DEFAULT_RUN_PROFILES_FILE)::AbstractDict
    data = JSON.parsefile(path)
    validate_run_profiles(data)
    return data
end

function resolve_problem_selection(
    selector::String,
    profiles_path::String = DEFAULT_RUN_PROFILES_FILE,
    manually_skipped_problems::Vector{String} = String[],
)::ProblemSelection
    length(manually_skipped_problems) == length(unique(manually_skipped_problems)) ||
        error("The manual skip list contains duplicate problems.")
    data = load_run_profiles(profiles_path)
    profiles = data["profiles"]
    haskey(profiles, selector) || error("Unknown problem set '$selector'.")
    profile = profiles[selector]
    available_problems = Set(profile_problems(profile))
    unavailable = setdiff(Set(manually_skipped_problems), available_problems)
    isempty(unavailable) || error(
        "Manually skipped problems are not in problem set '$selector': " *
        "$(join(sort!(collect(unavailable)), ", ")).",
    )
    skipped = sort!(copy(manually_skipped_problems))
    skipped_set = Set(skipped)
    batches = [
        filter(problem -> problem ∉ skipped_set, batch) for
        batch in profile_batches(profile)
    ]
    filter!(batch -> !isempty(batch), batches)
    return ProblemSelection(
        selector,
        batches,
        sort!(vcat(batches...)),
        Int(data["default_max_iterations"]),
        Float64(profile["max_time_seconds"]),
        skipped,
    )
end

export DEFAULT_RUN_PROFILES_FILE,
    ProblemSelection,
    load_run_profiles,
    resolve_problem_selection,
    validate_run_profiles

end
