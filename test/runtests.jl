using FeatureSelection, MLJBase, MLJTuning, MLJDecisionTreeInterface, 
    MLJScikitLearnInterface, StableRNGs, StatisticalMeasures, Test
import Distributions

const rng = StableRNG(123)

include("Aqua.jl")
include("models/dummy_test_models.jl")
include("models/featureselector.jl")
include("models/rfe.jl")
