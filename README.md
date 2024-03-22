# FeatureSelection.jl

| Linux | Coverage | Code Style
| :------------ | :------- | :------------- |
| [![Build Status](https://github.com/JuliaAI/FeatureSelection.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/FeatureSelection.jl/actions) | [![Coverage](https://codecov.io/gh/JuliaAI/FeatureSelection.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/FeatureSelection.jl?branch=dev) | [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) |

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
Lets build a supervised recursive feature eliminator with `RandomForestRegressor` 
from DecisionTree.jl as our base model.
But first we need a dataset to train on. We shall create a synthetic dataset popularly 
known in the R community as the friedman dataset#1. Notice how the target vector for this 
dataset depends on only the first five columns of feature table. So we expect that our 
recursive feature elimination should return the first columns as important features.
```julia
using MLJ, FeatureSelection
using StableRNGs
rng = StableRNG(10)
A = rand(rng, 50, 10)
X = MLJ.table(A) # features
y = @views(
    10 .* sin.(
        pi .* A[:, 1] .* A[:, 2]
    ) .+ 20 .* (A[:, 3] .- 0.5).^ 2 .+ 10 .* A[:, 4] .+ 5 * A[:, 5]
) # target
```
Now we that we have our data we can create our recursive feature elimination model and 
train it on our dataset
```julia
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
forest = RandomForestRegressor(rng=rng)
rfe = RecursiveFeatureElimination(
    model = forest, n_features=5, step=1
) # see doctring for description of defaults
mach = machine(rfe, X, y)
fit!(mach)
```
We can inspect the feature importances in two ways:
```julia
# A variable with lower rank has more significance than a variable with higher rank.
# A variable with Higher feature importance is better than a variable with lower 
# feature importance
report(mach).ranking # returns [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
feature_importances(mach) # returns dict of feature => importance pairs
```
We can view the important features used by our model by inspecting the `fitted_params` 
object.
```julia
p = fitted_params(mach)
p.features_left == [:x1, :x2, :x3, :x4, :x5]
```
We can also call the `predict` method on the fitted machine, to predict using a 
random forest regressor trained using only the important features, or call the `transform` 
method, to select just those features from some new table including all the original 
features. For more info, type `?RecursiveFeatureElimination` on a Julia REPL.

Okay, let's say that we didn't know that our synthetic dataset depends on only five 
columns from our feature table. We could apply cross fold validation 
`StratifiedCV(nfolds=5)` with our recursive feature elimination model to select the 
optimal value of `n_features` for our model. In this case we will use a simple Grid 
search with root mean square as the measure. 
```julia
rfe = RecursiveFeatureElimination(model = forest)
tuning_rfe_model  = TunedModel(
    model = rfe,
    measure = rms,
    tuning = Grid(rng=rng),
    resampling = StratifiedCV(nfolds = 5),
    range = range(
        rfe, :n_features, values = 1:10
    )
)
self_tuning_rfe_mach = machine(tuning_rfe_model, X, y)
fit!(self_tuning_rfe_mach)
```
As before we can inspect the important features by inspecting the object returned by 
`fitted_params` or `feature_importances` as shown below.
```julia
fitted_params(self_tuning_rfe_mach).best_fitted_params.features_left == [:x1, :x2, :x3, :x4, :x5]
feature_importances(self_tuning_rfe_mach) # returns dict of feature => importance pairs
```
and call `predict` on the tuned model machine as shown below
```julia
Xnew = MLJ.table(rand(rng, 50, 10)) # create test data
predict(self_tuning_rfe_mach, Xnew)
```
In this case, prediction is done using the best recursive feature elimination model gotten 
from the tuning process above.

For resampling methods different from cross-validation, and for other
 `TunedModel` options, such as parallelization, see the 
 [Tuning Models](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/) section of the MLJ manual.
[MLJ Documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/)