module FeatureSelection

using MLJModelInterface, Tables, ScientificTypesBase

export FeatureSelector, RecursiveFeatureElimination

const MMI = MLJModelInterface

## Includes
include("models/featureselector.jl")
include("models/rfe.jl")
include("shared.jl")

end # module
