function warn_double_spec(arg, model)
    return "Using `model=$arg`. Ignoring keyword specification `model=$model`. "
end
 
const ERR_SPECIFY_MODEL = ArgumentError(
    "You need to specify model as positional argument or specify `model=...`."
)

const ERR_MODEL_TYPE = ArgumentError(
    "Only `Deterministic` and `Probabilistic` model types supported."
)

const ERR_FEATURE_IMPORTANCE_SUPPORT = ArgumentError(
        "Model does not report feature importance, hence recursive feature algorithm "*
        "can't be applied."
)

const MODEL_TYPES = [:ProbabilisticRecursiveFeatureElimination, :DeterministicRecursiveFeatureElimination]
const SUPER_TYPES = [:Deterministic, :Probabilistic]
const MODELTYPE_GIVEN_SUPERTYPES = zip(MODEL_TYPES, SUPER_TYPES)

for (ModelType, ModelSuperType) in  MODELTYPE_GIVEN_SUPERTYPES
    ex = quote
        mutable struct $ModelType{M<:Supervised} <: $ModelSuperType
            model::M
            n_features_to_select::Float64
            step::Float64
        end
    end
    eval(ex)
end

eval(:(const RFE{M} = Union{$((Expr(:curly, modeltype, :M) for modeltype in MODEL_TYPES)...)}))

# Common keyword constructor for both model types
"""
    RecursiveFeatureElimination(model, n_features_to_select, step)

This model implements a recursive feature elimination algorithm for feature selection. 
It recursively removes features, training a base model on the remaining features and 
evaluating their importance until the desired number of features is selected.

Construct an instance with default hyper-parameters using the syntax 
`model = RecursiveFeatureElimination(model=...)`. Provide keyword arguments to override 
hyper-parameter defaults.  
        
# Training data
In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

OR, if the base model supports weights, as

    mach = machine(model, X, y, w)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of the scitype
  as that required by the base model; check column scitypes with `schema(X)` and column 
  scitypes required by base model with `input_scitype(basemodel)`.

- `y` is the target, which can be any table of responses whose element scitype is 
    `Continuous` or `Finite` depending on the `target_scitype` required by the base model; 
    check the scitype with `scitype(y)`.

- `w` is the observation weights which can either be `nothing`(default) or an 
  `AbstractVector` whoose element scitype is `Count` or `Continuous`. This is different 
  from `weights` kernel which is an hyperparameter to the model, see below.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters
- model: A base model with a `fit` method that provides information on feature 
  feature importance (i.e `reports_feature_importances(model) == true`)

- n_features_to_select::Real = 0: The number of features to select. If `0`, half of the 
  features are selected. If a positive integer, the parameter is the absolute number 
  of features to select. If a real number between 0 and 1, it is the fraction of features 
  to select.

- step::Real=1: If the value of step is at least 1, it signifies the quantity of features to 
  eliminate in each iteration. Conversely, if step falls strictly within the range of 
  0.0 to 1.0, it denotes the proportion (rounded down) of features to remove during each iteration.

# Operations

- `transform(mach, X)`: transform the input table `X` into a new table containing only 
columns corresponding to features gotten from the RFE algorithm.

- `predict(mach, X)`: transform the input table `X` into a new table same as in 

- `transform(mach, X)` above and predict using the fitted base model on the 
  transformed table.

# Fitted parameters
The fields of `fitted_params(mach)` are:
- `features_left`: names of features remaining after recursive feature elimination.

- `model_fitresult`: fitted parameters of the base model.

# Report
The fields of `report(mach)` are:
- `ranking`: The feature ranking of each features in the training dataset. 

- `model_report`: report for the fitted base model.

- `features`: names of features seen during the training process. 

# Examples
```
using FeatureSelection, MLJ, StableRNGs

RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree

# Creates a dataset where the target only depends on the first 5 columns of the input table.
A = rand(rng, 50, 10);
y = 10 .* sin.(pi .* A[:, 1] .* A[:, 2]) + 20 .* (A[:, 3] .- 0.5).^ 2 .+ 10 .* A[:, 4] .+ 5 * A[:, 5]);
X = MLJ.table(A);

# fit a rfe model
rf = RandomForestRegressor()
selector = RecursiveFeatureElimination(model = rf)
mach = machine(selector, X, y)
fit!(mach)

# view the feature importances 
feature_importances(mach)

# predict using the base model 
Xnew = MLJ.table(rand(rng, 50, 10));
predict(mach, Xnew)

```

"""
function RecursiveFeatureElimination(
    args...;
    model=nothing,
    n_features_to_select::Real=0,
    step::Real = 1
)
    # user can specify model as argument instead of kwarg:
    length(args) < 2 || throw(ERR_TOO_MANY_ARGUMENTS)
    if length(args) == 1
        arg = first(args)
        model === nothing ||
            @warn warn_double_spec(arg, model)
        model = arg
    else
        model === nothing && throw(ERR_SPECIFY_MODEL)
    end

    #TODO: Check that the specifed model implements the predict method.
    # probably add a trait to check this
    MMI.reports_feature_importances(model) || throw(ERR_FEATURE_IMPORTANCE_SUPPORT)
    if model isa Deterministic    
        selector = DeterministicRecursiveFeatureElimination{typeof(model)}(
            model, Float64(n_features_to_select), Float64(step)
        )
    elseif model isa Probabilistic
        selector = ProbabilisticRecursiveFeatureElimination{typeof(model)}(
            model, Float64(n_features_to_select), Float64(step)
        )
    else
        throw(ERR_MODEL_TYPE)
    end 
    message = MMI.clean!(selector)
    isempty(message) || @warn(message)
    return selector
end

function MMI.clean!(selector::RFE)
    msg = ""
    if selector.step <= 0
        msg *= "specified `step` must be greater than zero.\n"
        "Resetting `step = 1`"
    end

    if selector.n_features_to_select < 0
        msg *= "specified `step` must be non-negative.\n"*
        "Resetting `n_features_to_select = 0`"
    end

    return msg
end

function MMI.fit(selector::RFE, verbosity::Int, X, y, args...)
    args = (y, args...)
    Xcols = Tables.Columns(X)
    features = collect(Tables.columnnames(Xcols))
    nfeatures = length(features)
    nfeatures < 2 && throw(ArgumentError("The number of features in the feature matrix must be at least 2."))

    # Compute required number of features to select
    n_features_to_select = selector.n_features_to_select # Remember to modify this estimate later
    ## zero indicates that half of the features be selected.
    if n_features_to_select == 0
        n_features_to_select = div(nfeatures, 2) 
    elseif 0 < n_features_to_select < 1
        n_features_to_select = round(Int, n_features * n_features_to_select)
    else
        n_features_to_select = round(Int, n_features_to_select)
    end

    step = selector.step
    
    if 0 < step < 1
        step = round(Int, max(1, step * n_features))
    else
        step = round(Int, step) 
    end
    
    support = trues(nfeatures)
    ranking = ones(nfeatures) # every feature has equal rank initially
    indexes  = axes(support, 1)

    # Elimination
    features_left = copy(features)
    while sum(support) > n_features_to_select
        # Rank the remaining features
        model = selector.model
        verbosity > 0 && @info("Fitting estimator with $(sum(support)) features.")
    
        data = MMI.reformat(model, MMI.selectcols(X, features_left), args...)

        fitresult, _, report = MMI.fit(model, verbosity - 1, data...)

        # Get absolute values of importance and rank them
        importances = abs.(
            last.(
                MMI.feature_importances(
                    selector.model,
                    fitresult,
                    report
                )
            )
        )

        ranks = sortperm(importances)

        # Eliminate the worse features
        threshold = min(step, sum(support) - n_features_to_select)
        
        support[indexes[ranks][1:threshold]] .= false
        ranking[.!support] .+= 1

        # Remaining features
        features_left = @view(features[support])
    end

    # Set final attributes
    data = MMI.reformat(selector.model, MMI.selectcols(X, features_left), args...)
    verbosity > 0 && @info ("Fitting estimator with $(sum(support)) features.")
    model_fitresult, _, model_report = MMI.fit(selector.model, verbosity - 1, data...)
    
    fitresult = (
        support = support,
        model_fitresult = model_fitresult,
        features_left = copy(features_left)
    )
    report = ( 
        ranking = ranking,
        model_report = model_report,
        features = features
    )

    return fitresult, nothing, report

end

function MMI.fitted_params(model::RFE, fitresult)
    (
        features_left = fitresult.features_left,
        model_fitresult = MMI.fitted_params(model.model, fitresult.model_fitresult)
    )
end

function MMI.predict(model::RFE, fitresult, X)
    X_ = reformat(model.model, MMI.transform(model, fitresult, X))[1]
    yhat = MMI.predict(model.model, fitresult.model_fitresult, X_)
    return yhat
end

function MMI.transform(::RFE, fitresult, X)
    return MMI.selectcols(X, fitresult.features_left)
end

function MMI.feature_importances(::RFE, fitresult, report)
    return Pair.(report.features, report.ranking)
end

## Traits definitions
function MMI.load_path(::Type{<:DeterministicRecursiveFeatureElimination})
    return "FeatureEngineering.DeterministicRecursiveFeatureElimination"
end

function MMI.load_path(::Type{<:ProbabilisticRecursiveFeatureElimination})
    return "FeatureEngineering.ProbabilisticRecursiveFeatureElimination"
end

for trait in [
    :supports_weights,
    :supports_class_weights,
    :is_pure_julia,
    :is_wrapper,
    :input_scitype,
    :target_scitype,
    :output_scitype,
    :supports_training_losses,
    :reports_feature_importances,
    ]

    # try to get trait at level of types ("failure" here just
    # means falling back to `Unknown`):
    quote
    MMI.$trait(::Type{<:RFE{M}}) where M = MMI.$trait(M)
    end |> eval

    # needed because traits are not always deducable from
    # the type (eg, `target_scitype` and `Pipeline` models):
    eval(:(MMI.$trait(model::RFE) = MMI.$trait(model.model)))
end

# ## Iteration parameter
# at level of types:
function MMI.iteration_parameter(::Type{<:RFE{M}}) where {M}
    return MLJModels.prepend(:model, MMI.iteration_parameter(M))
end

# at level of instances:
function MMI.iteration_parameter(model::RFE)
    return MLJModels.prepend(:model, MMI.iteration_parameter(model.model))
end

## TRAINING LOSSES SUPPORT
function MMI.training_losses(model::RFE, rfe_report)
    return MMI.training_losses(model.model, rfe_report.model_report)
end