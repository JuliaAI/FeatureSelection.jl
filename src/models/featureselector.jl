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

# docstring is in "src/type_docstrings.jl"
