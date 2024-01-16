mutable struct RFE{M<:Model} <: Unsupervised
    model::M
    n_features_to_select::Float64
    step::Float64
end


# keyword constructor
function RFE(
    ;
    model,
    n_features_to_select=0,
    step = 1
)
    selector = RFE(model, Float64(n_features_to_select), Float64(step))
    message = MMI.clean!(selector)
    isempty(message) || throw(ArgumentError(message))
    return selector
end

function MMI.clean!(selector::RFE)
    err = ""
    if !MMI.feature_importances(selector.model)
        err *= "specified model, $(selector.model) does not support feature importances\n"
    end

    if RFE.step <= 0
        err *= "specified `step` must be greater than zero.\n"
    end

    if RFE.n_features_to_select < 0
        err *= "specified `step` must be non-negative.\n"
    end

    return err
end


function MMI.fit(selector::RFE, verbosity::Int, X, y, args...)
    args = (y, args...)
    Xcols = Tables.Columns(X)
    features = Vector(Tables.columnnames(Xcols))
    
    nfeatures = length(features)

    nfeatures < 2 || throw(ArgumentError("The number of features in feature matrix `X` must be at least 2."))

    # Compute required number of features to select
    n_features_to_select = RFE.n_features_to_select # Remember to modify this estimate later
    ## zero indicates that half of the features be selected.
    if n_features_to_select == 0
        n_features_to_select = div(nfeatures, 2) 
    else if 0 < n_features_to_select < 1
        n_features_to_select = round(Int, n_features * n_features_to_select)
    else
        n_features_to_select = round(Int, n_features_to_select)
    end

    step = RFE.step
    
    if 0 < step < 1
        step = round(Int, max(1, step * n_features))
    else
        step = rount(Int, step) 
    end
    
    support = trues(nfeatures)
    ranking = trues(nfeatures) # every feature has equal rank initially
    indexes  = axes(support, 1)

    # Elimination
    features_left = features
    while sum(support) > n_features_to_select:
        # Remaining features
        features_left = features_left[support]

        # Rank the remaining features
        model = RFE.model
        verbosity > 0 && @info ("Fitting estimator with $(sum(support)) features.")
    
        data = reformat(model, MMI.selectcols(X, features_left), args...)

        fitresult, _, report = fit(model, verbosity - 1, data...)

        # Get absolute values of importance and rank them
        importances = abs.last.(
            MMI.feature_importances(
                model,
                fitresult,
                report
            )
        )

        ranks = sortperm(importances)

        # Eliminate the worse features
        threshold = min(step, sum(support) - n_features_to_select)
        
        support[indexes[ranks][1:threshold]] .= false
        ranking[.!support] += 1
    end

    # Set final attributes
    fitresult = features[support]
    report = ( 
        ranking = ranking
    )

    return fitresult, nothing, report

end


metadata_pkg.(
    (RFE),
    package_name       = "FeatureEngineering",
    package_uuid       = "33837fe5-dbff-4c9e-8c2f-c5612fe2b8b6",
    package_url        = "https://github.com/JuliaAI/FeatureEngineering.jl",
    is_pure_julia      = true,
    package_license    = "MIT"
)

## Traits definitions

# Let's start with the load path
MMI.load_path(::Type{<:RFE{M}}) where M = "FeatureEngineering.RFE"

for trait in [
    :supports_weights,
    :supports_class_weights,
    :is_pure_julia,
    :input_scitype,
    :target_scitype,
    :output_scitype,
    :supports_training_losses,
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
function MMI.iteration_parameter(::Type{<:RFE{M}}) where M
    return MLJModels.prepend(:model, MMI.iteration_parameter(M))
end

# at level of instances:
function MMI.iteration_parameter(model::RFE)
    return MLJModels.prepend(:model, MMI.iteration_parameter(model.model))
end

# ## TRAINING LOSSES SUPPORT
function MMI.training_losses(model::RFE, rfe_report)
    return MMI.training_losses(model.model, rfe_report.model_report)
end

"""
$(MMI.doc_header(RFE))
    Recursive Feature Elimination
    
# Training data
    In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

    OR

    mach = machine(model, X, y, w)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any table of responses whose element scitype is 
    `Continuous`; check the scitype with `scitype(y)`.

- `w` is the observation weights which can either be `nothing`(default) or an 
  `AbstractVector` whoose element scitype is `Count` or `Continuous`. This is different 
  from `weights` kernel which is an hyperparameter to the model, see below.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters
    * model : A model with a ``fit`` method that provides information on feature
        feature importance (i.e `reports_feature_importances(model) == true`)

    * n_features_to_select : int or float, default=None
        The number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of features to
        select.

    * step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

# Operations 
- `transform(mach, X)`: transform the input table `X` into a new table 
containing only columns corresponding to features gotten from the RFE algorithm.

# Fitted parameters

# Report


# Examples
```
using MLJ
RecursiveFeatureElimination = @load RFE pkg=FeatureEngineering
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
X, y = @load_boston; # loads the crabs dataset from MLJBase
```

"""