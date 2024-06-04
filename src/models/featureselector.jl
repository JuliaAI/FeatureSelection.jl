# # FOR FEATURE (COLUMN) SELECTION

mutable struct FeatureSelector <: Unsupervised
    # features to be selected; empty means all
    features::Union{Vector{Symbol}, Function}
    ignore::Bool # features to be ignored
end

# keyword constructor
function FeatureSelector(
    ;
    features::Union{AbstractVector{Symbol}, Function}=Symbol[],
    ignore::Bool=false
)
    transformer = FeatureSelector(features, ignore)
    message = MMI.clean!(transformer)
    isempty(message) || throw(ArgumentError(message))
    return transformer
end

function MMI.clean!(transformer::FeatureSelector)
    err = ""
    if (
        typeof(transformer.features) <: AbstractVector{Symbol} &&
        isempty(transformer.features) &&
        transformer.ignore
    )
        err *= "Features to be ignored must be specified in features field."
    end
    return err
end

function MMI.fit(transformer::FeatureSelector, verbosity::Int, X)
    all_features = Tables.schema(X).names

    if transformer.features isa AbstractVector{Symbol}
        if isempty(transformer.features)
           features = collect(all_features)
        else
            features = if transformer.ignore
                !issubset(transformer.features, all_features) && verbosity > -1 &&
                @warn("Excluding non-existent feature(s).")
                filter!(all_features |> collect) do ftr
                   !(ftr in transformer.features)
                end
            else
                issubset(transformer.features, all_features) ||
                throw(ArgumentError("Attempting to select non-existent feature(s)."))
                transformer.features |> collect
            end
        end
    else
        features = if transformer.ignore
            filter!(all_features |> collect) do ftr
                !(transformer.features(ftr))
            end
        else
            filter!(all_features |> collect) do ftr
                transformer.features(ftr)
            end
        end
        isempty(features) && throw(
            ArgumentError("No feature(s) selected.\n The specified Bool-valued"*
              " callable with the `ignore` option set to `$(transformer.ignore)` "*
              "resulted in an empty feature set for selection")
         )
    end

    fitresult = features
    report = NamedTuple()
    return fitresult, nothing, report
end

MMI.fitted_params(::FeatureSelector, fitresult) = (features_to_keep=fitresult,)

function MMI.transform(::FeatureSelector, features, X)
    all(e -> e in Tables.schema(X).names, features) ||
        throw(ArgumentError("Supplied frame does not admit previously selected features."))
    return MMI.selectcols(X, features)
end

## Traits definitions
MMI.metadata_model(
    FeatureSelector,
    input_scitype = Table,
    output_scitype = Table,
    load_path = "FeatureSelection.FeatureSelector"
)

## Pkg Traits
MMI.metadata_pkg(
    FeatureSelector,
    package_name = "FeatureSelection",
    package_uuid = "33837fe5-dbff-4c9e-8c2f-c5612fe2b8b6",
    package_url = "https://github.com/JuliaAI/FeatureSelection.jl",
    is_pure_julia = true,
    package_license = "MIT"
)

## Docstring
"""
$(MMI.doc_header(FeatureSelector))

Use this model to select features (columns) of a table, usually as
part of a model `Pipeline`.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any table of input features, where "table" is in the sense of Tables.jl

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `features`: one of the following, with the behavior indicated:

  - `[]` (empty, the default): filter out all features (columns) which
    were not encountered in training

  - non-empty vector of feature names (symbols): keep only the
    specified features (`ignore=false`) or keep only unspecified
    features (`ignore=true`)

  - function or other callable: keep a feature if the callable returns
    `true` on its name. For example, specifying
    `FeatureSelector(features = name -> name in [:x1, :x3], ignore =
    true)` has the same effect as `FeatureSelector(features = [:x1,
    :x3], ignore = true)`, namely to select all features, with the
    exception of `:x1` and `:x3`.

- `ignore`: whether to ignore or keep specified `features`, as
  explained above


# Operations

- `transform(mach, Xnew)`: select features from the table `Xnew` as
  specified by the model, taking features seen during training into
  account, if relevant


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features_to_keep`: the features that will be selected


# Example

```
using MLJ

X = (ordinal1 = [1, 2, 3],
     ordinal2 = coerce(["x", "y", "x"], OrderedFactor),
     ordinal3 = [10.0, 20.0, 30.0],
     ordinal4 = [-20.0, -30.0, -40.0],
     nominal = coerce(["Your father", "he", "is"], Multiclass));

selector = FeatureSelector(features=[:ordinal3, ], ignore=true);

julia> transform(fit!(machine(selector, X)), X)
(ordinal1 = [1, 2, 3],
 ordinal2 = CategoricalValue{Symbol,UInt32}["x", "y", "x"],
 ordinal4 = [-20.0, -30.0, -40.0],
 nominal = CategoricalValue{String,UInt32}["Your father", "he", "is"],)

```
"""
FeatureSelector
