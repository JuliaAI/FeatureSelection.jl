# FeatureSelection.jl

| Linux | Coverage |
| :------------ | :------- |
| [![Build Status](https://github.com/JuliaAI/FeatureSelection.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/FeatureSelection.jl/actions) | [![Coverage](https://codecov.io/gh/JuliaAI/FeatureSelection.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/FeatureSelection.jl?branch=dev) |

Repository housing feature selection algorithms for use with the machine learning toolbox
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/).

`FeatureSelector` model builds on contributions originally residing at [MLJModels.jl](https://github.com/JuliaAI/MLJModels.jl/blob/v0.16.15/src/builtins/Transformers.jl#L189-L266)

# Installation
On a running instance of Julia with at least version 1.6 run
```julia
import Pkg;
Pkg.add("FeatureSelection")
```

# Example Usage
Lets build a supervised recursive feature eliminator with `RandomForestRegressor` from `MLJDecisionTreeInterface` as our base model.
But first we need a dataset to train on. We shall create a synthetic dataset popularly known in the R community as the friedman dataset#1. Notice how the target vector for this dataset depends on only the first 
five columns of feature table. So we expect that our recursive feature elimination should return the first
columns as important features.
```julia
using MLJ # or, minimally, `using FeatureSelection, MLJModels, MLJBase`
using StableRNGs
rng = StableRNG(123)
A = rand(rng, 50, 10)
X = MLJ.table(A) # features
y = @views(
    10 .* sin.(pi .* A[:, 1] .* A[:, 2]) + 20 .* (A[:, 3] .- 0.5).^ 2 .+ 10 .* A[:, 4] .+ 5 * A[:, 5]
) # target
```
Now we that we have our data we can create our recursive feature elimination model and train it on our dataset
```julia
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
forest = RandomForestRegressor()
rfe = RecursiveFeatureElimination(
    model = forest, n_features_to_select=5, step=1
) # see doctring for description of defaults
mach = machine(rfe, X, y)
fit!(mach)
```
We can view the important features by inspecting the `fitted_params` object.
```julia
p = fitted_params(mach)
p.features_left == [:x1, :x2, :x3, :x4, :x5]
```
We can even call `predict` and `transform` om the fitted machine. See example 
in `?RecursiveFeatureElimination`.

Okay, let's say that we didn't know that our synthetic dataset depends on only five 
columns from our feature table. We could apply cross fold validation `CV(nfolds=5)` with our 
recursive feature elimination model to select the optimal value of  
`n_features_to_select` for our model. In this case we will use a simple Grid search with 
root mean square as the measure. 
```julia
rfe = RecursiveFeatureElimination(model = forest)
tuning_rfe_model  = TunedModel(
    model = rfe,
    measure = rms,
    tuning = Grid(rng=rng, resolution=10),
    resampling = CV(nfolds = 5),
    range = range(
        rfe, :n_features_to_select, values = collect(2:8)
    )
)
self_tuning_rfe_mach = machine(tuning_rfe_model, X, y)
fit!(self_tuning_rfe_mach)
```
As before we can inspect the important features by inspesting the `fitted_params` object.
```julia
fitted_parms(self_tuning_rfe_mach).features_left == [:x1, :x2, :x3, :x4, :x5]
```
For more information various cross-validation strategies and `TunedModel` see [MLJ Documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/)