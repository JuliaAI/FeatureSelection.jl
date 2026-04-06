module FeatureSelection

using MLJModelInterface, Tables, ScientificTypesBase

export FeatureSelector, RecursiveFeatureElimination

const MMI = MLJModelInterface

## Includes
include("models/featureselector.jl")
include("models/rfe.jl")
include("shared.jl")
include("type_docstrings.jl")

end # module
