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

const ERR_FEATURES_SEEN = ArgumentError(
    "Features of new table must be same as those seen during fit process."
)

const MODEL_TYPES = [
    :ProbabilisticRecursiveFeatureElimination, :DeterministicRecursiveFeatureElimination
]
const SUPER_TYPES = [:Deterministic, :Probabilistic]
const MODELTYPE_GIVEN_SUPERTYPES = zip(MODEL_TYPES, SUPER_TYPES)

for (ModelType, ModelSuperType) in  MODELTYPE_GIVEN_SUPERTYPES
    ex = quote
        mutable struct $ModelType{M<:Supervised} <: $ModelSuperType
            model::M
            n_features::Float64
            step::Float64
        end
    end
    eval(ex)
end

eval(:(const RFE{M} = Union{$((Expr(:curly, modeltype, :M) for modeltype in MODEL_TYPES)...)}))

# Common keyword constructor for both model types
"""
    RecursiveFeatureElimination(model, n_features, step)

This model implements a recursive feature elimination algorithm for feature selection. 
It recursively removes features, training a base model on the remaining features and 
evaluating their importance until the desired number of features is selected.

Construct an instance with default hyper-parameters using the syntax 
`rfe_model = RecursiveFeatureElimination(model=...)`. Provide keyword arguments to override 
hyper-parameter defaults.  
        
# Training data
In MLJ or MLJBase, bind an instance `rfe_model` to data with

    mach = machine(rfe_model, X, y)

OR, if the base model supports weights, as

    mach = machine(rfe_model, X, y, w)

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

- n_features::Real = 0: The number of features to select. If `0`, half of the 
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
y = 10 .* sin.(
        pi .* A[:, 1] .* A[:, 2]
    ) + 20 .* (A[:, 3] .- 0.5).^ 2 .+ 10 .* A[:, 4] .+ 5 * A[:, 5]);
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
    n_features::Real=0,
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
            model, Float64(n_features), Float64(step)
        )
    elseif model isa Probabilistic
        selector = ProbabilisticRecursiveFeatureElimination{typeof(model)}(
            model, Float64(n_features), Float64(step)
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

    if selector.n_features < 0
        msg *= "specified `step` must be non-negative.\n"*
        "Resetting `n_features = 0`"
    end

    return msg
end

function MMI.fit(selector::RFE, verbosity::Int, X, y, args...)
    args = (y, args...)
    Xcols = Tables.Columns(X)
    features = collect(Tables.columnnames(Xcols))
    nfeatures = length(features)
    nfeatures < 2 && throw(
        ArgumentError("The number of features in the feature matrix must be at least 2.")
    )

    # Compute required number of features to select
    n_features_select = selector.n_features
    ## zero indicates that half of the features be selected.
    if n_features_select == 0
        n_features_select = div(nfeatures, 2) 
    elseif 0 < n_features_select < 1
        n_features_select = round(Int, n_features_select * nfeatures)
    else
        n_features_select = round(Int, n_features_select)
    end

    step = selector.step
    
    if 0 < step < 1
        step = round(Int, max(1, step * n_features_select))
    else
        step = round(Int, step) 
    end
    
    support = trues(nfeatures)
    ranking = ones(Int, nfeatures) # every feature has equal rank initially
    mask = trues(nfeatures) # for boolean indexing of ranking vector in while loop below

    # Elimination
    features_left = features
    n_features_left = length(features_left)
    while n_features_left > n_features_select
        # Rank the remaining features
        model = selector.model
        verbosity > 0 && @info("Fitting estimator with $(n_features_left) features.")
    
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
        threshold = min(step, n_features_left - n_features_select)
        @views(support[support][ranks[1:threshold]]) .= false
        mask .= support .== false
        @views(ranking[mask]) .+= 1

        # Remaining features
        features_left = features[support]
        n_features_left = length(features_left)
    end

    # Set final attributes
    data = MMI.reformat(selector.model, MMI.selectcols(X, features_left), args...)
    verbosity > 0 && @info ("Fitting estimator with $(n_features_left) features.")
    model_fitresult, _, model_report = MMI.fit(selector.model, verbosity - 1, data...)
    
    fitresult = (
        support = support,
        model_fitresult = model_fitresult,
        features_left = features_left,
        features = features
    )
    report = ( 
        ranking = ranking,
        model_report = model_report
    )

    return fitresult, nothing, report

end

function MMI.fitted_params(model::RFE, fitresult)
    (
        features_left = copy(fitresult.features_left),
        model_fitresult = MMI.fitted_params(model.model, fitresult.model_fitresult)
    )
end

function MMI.predict(model::RFE, fitresult, X)
    X_ = reformat(model.model, MMI.transform(model, fitresult, X))[1]
    yhat = MMI.predict(model.model, fitresult.model_fitresult, X_)
    return yhat
end

function MMI.transform(::RFE, fitresult, X)
    sch = Tables.schema(Tables.columns(X))
    if (length(fitresult.features) == length(sch.names) && 
        !all(e -> e in sch.names, fitresult.features))
            throw(
                ERR_FEATURES_SEEN
            )
    end
    return MMI.selectcols(X, fitresult.features_left)
end

function MMI.feature_importances(::RFE, fitresult, report)
    return Pair.(fitresult.features, Iterators.reverse(report.ranking))
end

function MMI.save(model::RFE, fitresult)
    support = fitresult.support
    atomic_fitresult = fitresult.model_fitresult
    features_left = fitresult.features_left
    features = fitresult.features
    
    atom = model.model
    return (
        support = copy(support),
        model_fitresult = MMI.save(atom, atomic_fitresult),
        features_left = copy(features_left),
        features = copy(features)
    )
end

function MMI.restore(model::RFE, serializable_fitresult)
    support = serializable_fitresult.support
    atomic_serializable_fitresult = serializable_fitresult.model_fitresult
    features_left = serializable_fitresult.features_left
    features = serializable_fitresult.features

    atom = model.model
    return (
        support = support,
        model_fitresult = MMI.restore(atom, atomic_serializable_fitresult),
        features_left = features_left,
        features = features
    )
end

## Traits definitions
function MMI.load_path(::Type{<:DeterministicRecursiveFeatureElimination})
    return "FeatureSelection.DeterministicRecursiveFeatureElimination"
end

function MMI.load_path(::Type{<:ProbabilisticRecursiveFeatureElimination})
    return "FeatureSelection.ProbabilisticRecursiveFeatureElimination"
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
prepend(s::Symbol, ::Nothing) = nothing
prepend(s::Symbol, t::Symbol) = Expr(:(.), s, QuoteNode(t))
prepend(s::Symbol, ex::Expr) = Expr(:(.), prepend(s, ex.args[1]), ex.args[2])

function MMI.iteration_parameter(::Type{<:RFE{M}}) where {M}
    return prepend(:model, MMI.iteration_parameter(M))
end

# at level of instances:
function MMI.iteration_parameter(model::RFE)
    return prepend(:model, MMI.iteration_parameter(model.model))
end

## TRAINING LOSSES SUPPORT
function MMI.training_losses(model::RFE, rfe_report)
    return MMI.training_losses(model.model, rfe_report.model_report)
end

## Pkg Traits
MMI.metadata_pkg.(
    (
        DeterministicRecursiveFeatureElimination,
        ProbabilisticRecursiveFeatureElimination, 
    ),
    package_name       = "FeatureSelection",
    package_uuid       = "33837fe5-dbff-4c9e-8c2f-c5612fe2b8b6",
    package_url        = "https://github.com/JuliaAI/FeatureSelection.jl",
    is_pure_julia      = true,
    package_license    = "MIT"
)