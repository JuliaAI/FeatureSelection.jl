# FeatureSelection

FeatureSelction is a julia package containing implementations of feature selection algorithms for use with the machine learning toolbox
[MLJ](https://juliaai.github.io/MLJ.jl/dev/).

# Installation
On a running instance of Julia with at least version 1.6 run
```julia
import Pkg;
Pkg.add("FeatureSelection")
```

# Example Usage
Lets build a supervised recursive feature eliminator with `RandomForestRegressor` 
from [DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl) as our base model.
But first we need a dataset to train on. We shall create a synthetic dataset popularly 
known in the R community as the friedman dataset#1. Notice how the target vector for this 
dataset depends on only the first five columns of feature table. So we expect that our 
recursive feature elimination should return the first columns as important features.
```@meta
DocTestSetup = quote
  using MLJ, FeatureSelection, StableRNGs
  rng = StableRNG(10)
  A = rand(rng, 50, 10)
  X = MLJ.table(A) # features
  y = @views(
      10 .* sin.(
          pi .* A[:, 1] .* A[:, 2]
      ) .+ 20 .* (A[:, 3] .- 0.5).^ 2 .+ 10 .* A[:, 4] .+ 5 * A[:, 5]
  ) # target
  RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
  forest = RandomForestRegressor(rng=rng)
  rfe = RecursiveFeatureElimination(
    model = forest, n_features=5, step=1
  ) # see doctring for description of defaults  
  mach = machine(rfe, X, y)
  fit!(mach)

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
end
```
```@example example1
using MLJ, FeatureSelection, StableRNGs
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
```@example example1
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
forest = RandomForestRegressor(rng=rng)
rfe = RecursiveFeatureElimination(
    model = forest, n_features=5, step=1
) # see doctring for description of defaults
mach = machine(rfe, X, y)
fit!(mach)
```
We can inspect the feature importances in two ways:
```jldoctest
julia> report(mach).ranking
10-element Vector{Int64}:
 1
 1
 1
 1
 1
 2
 3
 4
 5
 6

julia> feature_importances(mach)
10-element Vector{Pair{Symbol, Int64}}:
  :x1 => 6
  :x2 => 5
  :x3 => 4
  :x4 => 3
  :x5 => 2
  :x6 => 1
  :x7 => 1
  :x8 => 1
  :x9 => 1
 :x10 => 1
```
Note that a variable with lower rank has more significance than a variable with higher rank while a variable with higher feature importance is better than a variable with lower feature importance.

We can view the important features used by our model by inspecting the `fitted_params` 
object.
```jldoctest
julia> p = fitted_params(mach)
(features_left = [:x1, :x2, :x3, :x4, :x5],
 model_fitresult = (forest = Ensemble of Decision Trees
Trees:      100
Avg Leaves: 25.26
Avg Depth:  8.36,),)

julia> p.features_left
5-element Vector{Symbol}:
 :x1
 :x2
 :x3
 :x4
 :x5
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
```@example example1
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
```jldoctest
julia> fitted_params(self_tuning_rfe_mach).best_fitted_params.features_left
5-element Vector{Symbol}:
 :x1
 :x2
 :x3
 :x4
 :x5

julia> feature_importances(self_tuning_rfe_mach)
10-element Vector{Pair{Symbol, Int64}}:
  :x1 => 6
  :x2 => 5
  :x3 => 4
  :x4 => 3
  :x5 => 2
  :x6 => 1
  :x7 => 1
  :x8 => 1
  :x9 => 1
 :x10 => 1
```
and call `predict` on the tuned model machine as shown below
```@example example1
Xnew = MLJ.table(rand(rng, 50, 10)) # create test data
predict(self_tuning_rfe_mach, Xnew)
```
In this case, prediction is done using the best recursive feature elimination model gotten 
from the tuning process above.

For resampling methods different from cross-validation, and for other
 `TunedModel` options, such as parallelization, see the 
 [Tuning Models](https://juliaai.github.io/MLJ.jl/dev/tuning_models/) section of the MLJ manual.
[MLJ Documentation](https://juliaai.github.io/MLJ.jl/dev/)
```@meta
DocTestSetup = nothing
```