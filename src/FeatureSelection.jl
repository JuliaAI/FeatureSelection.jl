module FeatureSelection

using MLJModelInterface, Tables

const MMI = MLJModelInterface



## Includes
include("models/featureselector.jl")
include("models/rfe.jl")

## Pkg Traits
MMI.metadata_pkg.(
    (RecursiveFeatureElimination, FeatureSelector),
    package_name       = "FeatureSelection",
    package_uuid       = "33837fe5-dbff-4c9e-8c2f-c5612fe2b8b6",
    package_url        = "https://github.com/JuliaAI/FeatureSelection.jl",
    is_pure_julia      = true,
    package_license    = "MIT"
)

end # module
