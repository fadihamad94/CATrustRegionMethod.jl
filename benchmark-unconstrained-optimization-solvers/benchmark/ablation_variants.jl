module AblationVariants

const ABLATION_VARIANT_PAIRS = [
    "original" => "original",
    "rho-hat-rule" => "ρ_hat_rule",
    "radius-update-rule" => "radius_update_rule",
    "initial-radius" => "initial_radius",
    "conference-subproblem-solver" => "conference_subproblem_solver",
    "xi-zero" => "ξ=0.0",
    "b-k-zero" => "b_k=0.0",
]

const ABLATION_VARIANT_SLUGS = first.(ABLATION_VARIANT_PAIRS)
const ABLATION_VARIANTS = last.(ABLATION_VARIANT_PAIRS)
const ABLATION_VARIANT_BY_SLUG = Dict(ABLATION_VARIANT_PAIRS)
const ABLATION_SLUG_BY_VARIANT = Dict(reverse.(ABLATION_VARIANT_PAIRS))

function internal_ablation_variant(slug::String)::String
    haskey(ABLATION_VARIANT_BY_SLUG, slug) || error(
        "Unknown ablation variant '$slug'. Expected one of: " *
        join(ABLATION_VARIANT_SLUGS, ", "),
    )
    return ABLATION_VARIANT_BY_SLUG[slug]
end

function ablation_variant_slug(variant::String)::String
    haskey(ABLATION_SLUG_BY_VARIANT, variant) ||
        error("Unknown internal ablation variant '$variant'.")
    return ABLATION_SLUG_BY_VARIANT[variant]
end

export ABLATION_SLUG_BY_VARIANT,
    ABLATION_VARIANT_BY_SLUG,
    ABLATION_VARIANT_PAIRS,
    ABLATION_VARIANT_SLUGS,
    ABLATION_VARIANTS,
    ablation_variant_slug,
    internal_ablation_variant

end
